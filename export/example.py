# keywords: [example, simple, usage, hdf5]
"""Simple example of the HDF5-based data export system."""

import numpy as np
from export.exporter import DataExporter
from export.loader import ExperimentLoader, quick_load


def simple_snn_experiment():
    """Run a simple SNN experiment with data export."""
    
    # Create exporter - automatically uses HDF5
    with DataExporter(
        experiment_name="simple_snn",
        neural_sampling_rate=50,  # Save neural state every 50 timesteps
        compression='gzip',       # Enable compression
        compression_level=1       # Fast compression
    ) as exporter:
        
        # Save configuration
        config = {
            "n_neurons": 100,
            "dt": 1.0,  # ms
            "tau": 10.0,  # ms
            "learning_rate": 0.01,
            "grid_size": 1.0,
            "reward_radius": 0.5,
            "noise_amplitude": 2.0
        }
        exporter.save_config(config)
        
        # Save metadata
        metadata = {
            "description": "Simple SNN learning to navigate to reward",
            "author": "Your Name",
            "purpose": "Testing data export system",
            "notes": "Uses simplified LIF neurons with STDP-like learning"
        }
        exporter.save_metadata(metadata)
        
        # Save code snapshot (this file)
        import __main__
        if hasattr(__main__, '__file__'):
            exporter.save_code_snapshot([__main__.__file__])
        
        # Save git info if available
        exporter.save_git_info()
        
        # Define network
        n_neurons = config["n_neurons"]
        neurons = {
            "neuron_ids": np.arange(n_neurons),
            "neuron_types": np.array(["excitatory"] * 80 + ["inhibitory"] * 20),
            "positions": np.random.rand(n_neurons, 2)
        }
        
        # Random sparse connectivity
        n_connections = 500
        connections = {
            "source_ids": np.random.randint(0, n_neurons, n_connections),
            "target_ids": np.random.randint(0, n_neurons, n_connections),
        }
        
        # Initial weights
        initial_weights = np.random.randn(n_connections) * 0.1
        
        exporter.save_network_structure(neurons, connections, initial_weights)
        
        # Run 3 episodes
        for episode in range(3):
            print(f"\nRunning episode {episode}")
            
            # Start episode
            ep = exporter.start_episode()
            
            # Initialize state
            v = np.random.randn(n_neurons) * 10 - 65  # Membrane potentials
            weights = initial_weights.copy()
            agent_pos = np.array([0.5, 0.5])
            
            # Run simulation
            for t in range(1000):
                # Simple neural dynamics
                noise = np.random.randn(n_neurons) * 2
                v += noise * config["dt"]
                
                # Spikes
                spikes = v > -50
                v[spikes] = -65  # Reset
                
                # Agent movement
                agent_pos += np.random.randn(2) * 0.01
                agent_pos = np.clip(agent_pos, 0, 1)
                
                # Reward (distance to center)
                distance = np.linalg.norm(agent_pos - 0.5)
                reward = 1.0 - distance if distance < 0.5 else 0
                
                # Log timestep data
                exporter.log(
                    timestep=t,
                    neural_state={"membrane_potentials": v},
                    spikes=spikes,
                    behavior={"x": agent_pos[0], "y": agent_pos[1]},
                    reward=reward
                )
                
                # Occasional weight changes
                if t % 100 == 0 and t > 0:
                    # Pick random synapses to modify
                    n_changes = np.random.poisson(5)
                    for _ in range(n_changes):
                        idx = np.random.randint(n_connections)
                        old_w = weights[idx]
                        new_w = old_w + np.random.randn() * 0.001
                        weights[idx] = new_w
                        
                        exporter.log(
                            timestep=t,
                            synapse_id=idx,
                            old_weight=old_w,
                            new_weight=new_w
                        )
                        
                # Log analysis event
                if t == 500:
                    exporter.log(
                        timestep=t,
                        event_type="midpoint_analysis",
                        data={
                            "mean_v": np.mean(v),
                            "active_neurons": np.sum(np.abs(v + 65) > 5),
                            "mean_weight": np.mean(weights)
                        }
                    )
                    
            # End episode
            total_reward = ep.group.attrs.get('total_reward', 0)
            exporter.end_episode(
                success=total_reward > 50,
                summary={"final_position": agent_pos.tolist()}
            )
            
            print(f"Episode {episode} complete. Total reward: {total_reward:.2f}")
            
        # Save a checkpoint
        exporter.save_checkpoint("final_weights", {"weights": weights})
    
    print(f"\nExperiment saved to: {exporter.output_dir}")
    return exporter.output_dir


def analyze_experiment(experiment_dir):
    """Load and analyze the experiment data."""
    print(f"\n{'='*60}")
    print("LOADING AND ANALYZING DATA")
    print('='*60)
    
    # Method 1: Using ExperimentLoader for detailed access
    with ExperimentLoader(experiment_dir) as loader:
        # Get experiment info
        metadata = loader.get_metadata()
        print(f"\nExperiment: {metadata['experiment_name']}")
        print(f"Started: {metadata['start_time']}")
        print(f"Episodes: {metadata['episode_count']}")
        
        # Get user metadata
        if 'description' in metadata:
            print(f"\nDescription: {metadata['description']}")
            print(f"Author: {metadata.get('author', 'Unknown')}")
            
        # Get configuration
        config = loader.get_config()
        print(f"\nConfiguration:")
        for key, value in sorted(config.items()):
            print(f"  {key}: {value}")
            
        # Get runtime info
        runtime = loader.get_runtime_info()
        if runtime:
            print(f"\nRuntime Environment:")
            print(f"  Python: {runtime.get('python_version', 'Unknown').split()[0]}")
            print(f"  Platform: {runtime.get('platform', 'Unknown')}")
            if 'jax_version' in runtime:
                print(f"  JAX: {runtime['jax_version']}")
                
        # Get git info
        git_info = loader.get_git_info()
        if git_info:
            print(f"\nGit Info:")
            print(f"  Branch: {git_info.get('branch', 'Unknown')}")
            print(f"  Commit: {git_info.get('commit_hash', 'Unknown')[:8]}")
            print(f"  Clean: {'Yes' if not git_info.get('dirty', True) else 'No'}")
        
        # Get episode summaries
        summaries = loader.get_all_summaries()
        print(f"\nEpisode Summary:")
        print(f"{'Episode':<10} {'Timesteps':<12} {'Reward':<12} {'Success':<8}")
        print("-" * 45)
        for summary in summaries:
            print(f"{summary['episode_id']:<10} "
                  f"{summary.get('total_timesteps', 0):<12} "
                  f"{summary.get('total_reward', 0):<12.2f} "
                  f"{summary.get('success', False)!s:<8}")
        
        # Analyze first episode in detail
        episode = loader.get_episode(0)
        
        # Get neural states (just first 10 samples)
        neural_states = episode.get_neural_states(stop=10)
        if neural_states:
            print(f"\nNeural states shape: {neural_states['membrane_potentials'].shape}")
            print(f"Mean potential: {neural_states['membrane_potentials'].mean():.2f} mV")
        
        # Get spikes
        spikes = episode.get_spikes()
        if spikes:
            print(f"\nTotal spikes: {len(spikes['timesteps'])}")
            unique_neurons = np.unique(spikes['neuron_ids'])
            print(f"Active neurons: {len(unique_neurons)}/{metadata['n_neurons']}")
        
        # Get behavior
        behavior = episode.get_behavior()
        if behavior:
            print(f"\nBehavior samples: {len(behavior['timesteps'])}")
            print(f"Final position: ({behavior['x'][-1]:.3f}, {behavior['y'][-1]:.3f})")
        
        # Get custom events
        events = episode.get_events()
        if 'midpoint_analysis' in events:
            midpoint = events['midpoint_analysis']
            print(f"\nMidpoint analysis:")
            print(f"  Mean V: {midpoint['mean_v'][0]:.2f} mV")
            print(f"  Active neurons: {midpoint['active_neurons'][0]}")
    
    # Method 2: Quick load for simple access
    print(f"\n{'='*60}")
    print("QUICK LOAD EXAMPLE")
    print('='*60)
    
    data = quick_load(experiment_dir, episode_id=0)
    print(f"\nQuick load keys: {list(data.keys())}")
    print(f"Episode keys: {list(data['episode'].keys())}")
    print(f"First 5 spike events: {data['episode']['spikes'][:5]}")


if __name__ == "__main__":
    # Run experiment
    output_dir = simple_snn_experiment()
    
    # Analyze results
    analyze_experiment(output_dir)