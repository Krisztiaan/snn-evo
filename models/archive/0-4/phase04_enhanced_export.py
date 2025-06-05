"""
Enhanced data export additions for Phase 0.4 to enable complete traceability
"""

def enhance_episode_data_collection(original_run_episode):
    """
    Wrapper to add complete traceability to episode data collection
    """
    
    def run_episode_enhanced(seed: int, progress_callback=None, full_trace=False) -> Dict:
        # Run original episode
        episode_data = original_run_episode(seed, progress_callback)
        
        if full_trace:
            # Add enhanced tracking
            max_steps = episode_data['metadata']['steps_completed']
            
            # 1. Add reward state evolution
            # Track which rewards are active at each timestep
            reward_states = np.zeros((max_steps, NUM_REWARDS), dtype=bool)
            
            # 2. Add full neural dynamics (warning: memory intensive!)
            # For 256 neurons x 50,000 steps = 12.8M spike values
            if max_steps <= 10000:  # Only for shorter episodes
                full_spike_trains = np.zeros((max_steps, 256), dtype=bool)
                full_voltages = np.zeros((max_steps, 256), dtype=np.float32)
            
            # 3. Add per-timestep weight changes
            weight_changes = []  # List of (step, pre_idx, post_idx, old_w, new_w)
            
            # Enhanced episode loop would go here...
            # This is just the structure
            
            episode_data['enhanced_tracking'] = {
                'reward_states': reward_states,
                'full_neural_dynamics': {
                    'enabled': max_steps <= 10000,
                    'spike_trains': full_spike_trains if max_steps <= 10000 else None,
                    'voltages': full_voltages if max_steps <= 10000 else None
                },
                'weight_changes': weight_changes
            }
        
        return episode_data
    
    return run_episode_enhanced


# Alternative: Streaming approach for very large datasets
class StreamingDataLogger:
    """
    Stream data to disk during episode to avoid memory limitations
    """
    
    def __init__(self, episode_dir: str):
        self.episode_dir = episode_dir
        self.files = {}
        
    def initialize_streams(self, max_steps: int, num_neurons: int, num_rewards: int):
        """Create memory-mapped files for streaming data"""
        import numpy as np
        
        # Agent trajectory (already complete in original)
        
        # Reward states over time
        self.files['reward_states'] = np.memmap(
            f"{self.episode_dir}/reward_states.dat",
            dtype='bool',
            mode='w+',
            shape=(max_steps, num_rewards)
        )
        
        # Full neural activity (if requested)
        self.files['spikes'] = np.memmap(
            f"{self.episode_dir}/spike_trains_full.dat",
            dtype='bool', 
            mode='w+',
            shape=(max_steps, num_neurons)
        )
        
        self.files['voltages'] = np.memmap(
            f"{self.episode_dir}/voltages_full.dat",
            dtype='float32',
            mode='w+', 
            shape=(max_steps, num_neurons)
        )
        
        # Synaptic weight changes log
        self.weight_change_log = open(f"{self.episode_dir}/weight_changes.csv", 'w')
        self.weight_change_log.write("step,pre_idx,post_idx,old_weight,new_weight,delta\n")
        
    def log_timestep(self, step: int, state, env):
        """Log data for current timestep"""
        # Reward states
        self.files['reward_states'][step] = env.active_rewards
        
        # Neural activity
        self.files['spikes'][step] = state.spike
        self.files['voltages'][step] = state.v
        
    def log_weight_change(self, step: int, pre_idx: int, post_idx: int, 
                         old_w: float, new_w: float):
        """Log individual synaptic weight change"""
        delta = new_w - old_w
        self.weight_change_log.write(
            f"{step},{pre_idx},{post_idx},{old_w:.6f},{new_w:.6f},{delta:.6f}\n"
        )
        
    def close(self):
        """Close all file handles"""
        for f in self.files.values():
            if hasattr(f, 'flush'):
                f.flush()
        self.weight_change_log.close()


# Usage example:
"""
# In run_episode function:
if args.full_trace:
    logger = StreamingDataLogger(episode_dir)
    logger.initialize_streams(max_steps, params.NUM_NEURONS, NUM_REWARDS)
    
    for step in range(max_steps):
        # ... existing code ...
        
        # Log every timestep
        logger.log_timestep(step, state, env)
        
        # Log weight changes (need to track in three_factor_update)
        # ... 
        
    logger.close()
"""


# Utility functions for analysis
def reconstruct_reward_field(episode_dir: str, timestep: int):
    """
    Reconstruct exact reward field at any timestep
    """
    reward_states = np.memmap(
        f"{episode_dir}/reward_states.dat",
        dtype='bool',
        mode='r'
    )
    env_data = np.load(f"{episode_dir}/environment.npz")
    
    active_rewards = reward_states[timestep]
    reward_positions = env_data['reward_positions'][active_rewards]
    
    return reward_positions


def trace_spike_causality(episode_dir: str, neuron_idx: int, 
                         start_step: int, window: int = 100):
    """
    Trace what caused a specific neuron to spike
    """
    spikes = np.memmap(f"{episode_dir}/spike_trains_full.dat", dtype='bool', mode='r')
    voltages = np.memmap(f"{episode_dir}/voltages_full.dat", dtype='float32', mode='r')
    
    # Find spike time
    spike_times = np.where(spikes[start_step:start_step+window, neuron_idx])[0]
    
    if len(spike_times) == 0:
        return None
        
    spike_time = start_step + spike_times[0]
    
    # Look at pre-synaptic activity
    network_props = np.load(f"{episode_dir}/network_properties.npz")
    connection_mask = network_props['connection_mask']
    
    # Which neurons connect to this neuron?
    pre_synaptic = np.where(connection_mask[:, neuron_idx])[0]
    
    # Check their recent activity
    recent_window = 20  # ms
    pre_spike_times = {}
    for pre_idx in pre_synaptic:
        pre_spikes = np.where(
            spikes[spike_time-recent_window:spike_time, pre_idx]
        )[0]
        if len(pre_spikes) > 0:
            pre_spike_times[pre_idx] = spike_time - recent_window + pre_spikes
            
    return {
        'neuron': neuron_idx,
        'spike_time': spike_time,
        'voltage_trajectory': voltages[spike_time-50:spike_time+10, neuron_idx],
        'pre_synaptic_spikes': pre_spike_times
    }