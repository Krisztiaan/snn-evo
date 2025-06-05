# keywords: [test, optimized, exporter, verification]
"""Test script to verify optimized exporter functionality."""

import numpy as np
import time
from export import DataExporter, ExperimentLoader, quick_load


def test_optimized_exporter():
    """Test the optimized exporter with various data patterns."""
    print("Testing Optimized Data Exporter")
    print("=" * 60)
    
    # Create exporter
    with DataExporter(
        experiment_name="test_optimized",
        neural_sampling_rate=50,
        compression='gzip',
        compression_level=4
    ) as exporter:
        
        # Save configuration
        config = {
            "n_neurons": 100,
            "dt": 1.0,
            "learning_rate": 0.01
        }
        exporter.save_config(config)
        
        # Save metadata
        metadata = {
            "description": "Test of optimized exporter",
            "author": "Test Suite"
        }
        exporter.save_metadata(metadata)
        
        # Save network structure
        n_neurons = 100
        neurons = {
            "neuron_ids": np.arange(n_neurons),
            "neuron_types": np.array(["excitatory"] * 80 + ["inhibitory"] * 20),
            "positions": np.random.rand(n_neurons, 2)
        }
        
        n_connections = 500
        connections = {
            "source_ids": np.random.randint(0, n_neurons, n_connections),
            "target_ids": np.random.randint(0, n_neurons, n_connections),
        }
        initial_weights = np.random.randn(n_connections) * 0.1
        
        exporter.save_network_structure(neurons, connections, initial_weights)
        
        # Test episode with various data patterns
        print("\nRunning test episode...")
        start_time = time.time()
        
        ep = exporter.start_episode()
        
        # Initialize state
        v = np.random.randn(n_neurons) * 10 - 65
        weights = initial_weights.copy()
        
        timesteps_logged = 0
        spikes_logged = 0
        rewards_logged = 0
        weight_changes_logged = 0
        
        # Run simulation
        for t in range(5000):
            # Neural dynamics
            v += np.random.randn(n_neurons) * 2
            
            # Generate spikes (varying rates)
            if t < 1000:
                spike_rate = 0.05
            elif t < 2000:
                spike_rate = 0.2  # High activity burst
            elif t < 3000:
                spike_rate = 0.01  # Low activity
            else:
                spike_rate = 0.1
                
            spikes = np.random.rand(n_neurons) < spike_rate
            if spikes.any():
                spikes_logged += spikes.sum()
            v[spikes] = -65
            
            # Agent movement
            x = 0.5 + 0.3 * np.sin(t * 0.01)
            y = 0.5 + 0.3 * np.cos(t * 0.01)
            
            # Reward (sparse)
            reward = 0
            if t % 100 == 0:
                reward = np.random.rand()
                rewards_logged += 1
            
            # Log data
            exporter.log(
                timestep=t,
                neural_state={"membrane_potentials": v},
                spikes=spikes,
                behavior={"x": x, "y": y},
                reward=reward
            )
            timesteps_logged += 1
            
            # Weight changes (bursty)
            if t % 200 < 50 and np.random.rand() < 0.1:
                n_changes = np.random.poisson(3)
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
                    weight_changes_logged += 1
                    
            # Log analysis event
            if t == 2500:
                exporter.log(
                    timestep=t,
                    event_type="analysis",
                    data={
                        "mean_v": np.mean(v),
                        "active_neurons": np.sum(np.abs(v + 65) > 5),
                        "mean_weight": np.mean(weights)
                    }
                )
        
        # End episode
        exporter.end_episode(success=True, summary={"final_x": x, "final_y": y})
        
        elapsed = time.time() - start_time
        print(f"\nEpisode complete in {elapsed:.2f}s")
        print(f"  Timesteps logged: {timesteps_logged}")
        print(f"  Spikes logged: {spikes_logged}")
        print(f"  Rewards logged: {rewards_logged}")
        print(f"  Weight changes logged: {weight_changes_logged}")
        print(f"  Write speed: {timesteps_logged / elapsed:.0f} timesteps/s")
    
    print(f"\nData saved to: {exporter.output_dir}")
    
    # Test loading
    print("\n" + "="*60)
    print("Testing data loading...")
    
    with ExperimentLoader(exporter.output_dir) as loader:
        # Get metadata
        metadata = loader.get_metadata()
        print(f"\nExperiment: {metadata['experiment_name']}")
        print(f"Optimized: {metadata.get('optimized', False)}")
        print(f"Compression: {metadata.get('compression', 'none')}")
        
        # Get episode data
        episode = loader.get_episode(0)
        
        # Test neural states
        neural_states = episode.get_neural_states(stop=10)
        print(f"\nNeural states loaded: {neural_states['membrane_potentials'].shape}")
        
        # Test spikes (should handle RLE format)
        spikes = episode.get_spikes()
        print(f"Spikes loaded: {len(spikes['timesteps'])} events")
        unique_timesteps = len(np.unique(spikes['timesteps']))
        print(f"Unique spike timesteps: {unique_timesteps}")
        
        # Test rewards (should handle RLE format)
        rewards = episode.get_rewards()
        print(f"Rewards loaded: {len(rewards['timesteps'])} events")
        print(f"Total reward: {rewards['rewards'].sum():.2f}")
        
        # Test weight changes (optimized format)
        weight_changes = episode.get_weight_changes()
        if weight_changes:
            print(f"Weight changes loaded: {len(weight_changes['timesteps'])} events")
            print(f"Keys: {list(weight_changes.keys())}")
        
        # Test events
        events = episode.get_events()
        print(f"Events loaded: {list(events.keys())}")
        
        # Get file size
        file_size = loader.h5_path.stat().st_size / (1024 * 1024)
        print(f"\nFile size: {file_size:.2f} MB")
        print(f"Size per timestep: {file_size / timesteps_logged * 1000:.2f} KB")
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_optimized_exporter()