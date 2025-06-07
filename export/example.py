# keywords: [example, usage, demonstration, export]
"""Example usage of the neural network data export module."""

import numpy as np
from export import DataExporter, ExperimentLoader


def main():
    """Demonstrate basic usage of the export module."""
    
    # Configuration
    n_neurons = 1000
    n_timesteps = 10000
    n_episodes = 3
    
    print("=== Neural Network Data Export Example ===\n")
    
    # 1. Create an experiment and export data
    print("1. Creating experiment and exporting data...")
    
    with DataExporter(
        experiment_name="example_experiment",
        output_base_dir="./experiments",
        neural_sampling_rate=100,  # Sample neural state every 100 timesteps
        compression='gzip',
        compression_level=4
    ) as exporter:
        
        # Save experiment configuration
        config = {
            "n_neurons": n_neurons,
            "learning_rate": 0.001,
            "algorithm": "STDP",
            "network_type": "feedforward"
        }
        exporter.save_config(config)
        print(f"   - Saved configuration")
        
        # Save network structure
        neurons = {
            "neuron_ids": np.arange(n_neurons),
            "neuron_types": np.random.choice([0, 1], n_neurons),  # 0: excitatory, 1: inhibitory
            "positions": np.random.randn(n_neurons, 3)  # 3D positions
        }
        
        connections = {
            "source_ids": np.random.randint(0, n_neurons, 5000),
            "target_ids": np.random.randint(0, n_neurons, 5000),
            "delays": np.random.uniform(1, 10, 5000)
        }
        
        # Initial weight matrix (sparse)
        initial_weights = np.random.randn(n_neurons, n_neurons) * 0.1
        initial_weights[np.random.rand(n_neurons, n_neurons) > 0.1] = 0  # 90% sparse
        
        exporter.save_network_structure(neurons, connections, initial_weights)
        print(f"   - Saved network structure ({n_neurons} neurons)")
        
        # Run multiple episodes
        for episode_id in range(n_episodes):
            print(f"   - Recording episode {episode_id}...")
            
            with exporter.start_episode(episode_id) as episode:
                # Simulate episode data
                for t in range(n_timesteps):
                    # Neural state (sampled based on neural_sampling_rate)
                    membrane_potential = np.random.randn(n_neurons) * 20 - 70  # mV
                    
                    # Spikes (sparse binary data)
                    spike_prob = 0.01  # 1% spike rate
                    spikes = np.random.binomial(1, spike_prob, n_neurons)
                    
                    # Behavior data
                    position = np.array([
                        10 * np.sin(t * 0.01),
                        10 * np.cos(t * 0.01),
                        0.1 * t
                    ])
                    velocity = np.array([0.1, 0.0, 0.1])
                    
                    # Reward (sparse)
                    reward = 1.0 if t % 1000 == 0 else 0.0
                    
                    # Log timestep data
                    episode.log_timestep(
                        timestep=t,
                        neural_state={"membrane_potential": membrane_potential},
                        spikes=spikes,
                        behavior={
                            "position": position,
                            "velocity": velocity,
                            "action": np.random.randint(4)  # discrete action
                        },
                        reward=reward
                    )
                    
                    # Occasionally log weight changes (plasticity)
                    if t % 500 == 0 and t > 0:
                        for _ in range(10):
                            src, tgt = np.random.randint(0, n_neurons, 2)
                            old_w = np.random.randn() * 0.1
                            new_w = old_w + np.random.randn() * 0.01
                            
                            episode.log_weight_change(
                                timestep=t,
                                synapse_id=(src, tgt),
                                old_weight=old_w,
                                new_weight=new_w,
                                learning_rule="STDP"
                            )
                
                # Log custom event
                if episode_id == 0:
                    episode.log_event("phase_transition", n_timesteps // 2, {
                        "from_phase": "exploration",
                        "to_phase": "exploitation",
                        "confidence": 0.85
                    })
    
    print(f"\n   Experiment complete!")
    
    # 2. Load and analyze the data
    print("\n2. Loading and analyzing exported data...\n")
    
    # Find the experiment directory (most recent)
    from pathlib import Path
    exp_dirs = list(Path("./experiments").glob("example_experiment_*"))
    if not exp_dirs:
        print("   No experiment found!")
        return
    
    exp_dir = sorted(exp_dirs)[-1]  # Most recent
    
    with ExperimentLoader(exp_dir) as loader:
        # Get metadata
        metadata = loader.get_metadata()
        print(f"   Experiment: {metadata['experiment_name']}")
        print(f"   Created: {metadata['start_time']}")
        print(f"   Episodes: {metadata['episode_count']}")
        
        # Get configuration
        config = loader.get_config()
        print(f"\n   Configuration:")
        for key, value in config.items():
            print(f"     - {key}: {value}")
        
        # List episodes
        episodes = loader.list_episodes()
        print(f"\n   Found {len(episodes)} episodes: {episodes}")
        
        # Load first episode
        print(f"\n   Loading episode 0...")
        episode_data = loader.get_episode(0)
        
        # Get neural states
        neural_states = episode_data.get_neural_states()
        print(f"     - Neural states: {neural_states['timesteps'].shape[0]} samples")
        print(f"       Mean membrane potential: {neural_states['membrane_potential'].mean():.2f} mV")
        
        # Get spikes
        spikes = episode_data.get_spikes()
        if spikes:
            print(f"     - Spikes: {len(spikes['timesteps'])} total spikes")
            spike_rate = len(spikes['timesteps']) / (n_timesteps * n_neurons)
            print(f"       Average spike rate: {spike_rate*1000:.2f} Hz")
        
        # Get behavior
        behavior = episode_data.get_behavior()
        print(f"     - Behavior: {len(behavior['timesteps'])} timesteps")
        final_pos = behavior['position'][-1]
        print(f"       Final position: ({final_pos[0]:.2f}, {final_pos[1]:.2f}, {final_pos[2]:.2f})")
        
        # Get rewards
        rewards = episode_data.get_rewards()
        if rewards:
            print(f"     - Rewards: {len(rewards['timesteps'])} non-zero rewards")
            print(f"       Total reward: {np.sum(rewards['values']):.2f}")
        
        # Get weight changes
        weight_changes = episode_data.get_weight_changes()
        if weight_changes and 'timesteps' in weight_changes:
            print(f"     - Weight changes: {len(weight_changes['timesteps'])} changes")
            mean_delta = np.mean(weight_changes['deltas'])
            print(f"       Mean weight change: {mean_delta:.6f}")
            
        # Get metadata
        episode_meta = episode_data.get_metadata()
        print(f"\n   Episode summary:")
        print(f"     - Duration: {episode_meta['total_timesteps']} timesteps")
        print(f"     - Total spikes: {episode_meta.get('total_spikes', 0)}")
        print(f"     - Total reward: {episode_meta.get('total_reward', 0)}")
    
    print("\n=== Example completed successfully! ===")


if __name__ == "__main__":
    main()