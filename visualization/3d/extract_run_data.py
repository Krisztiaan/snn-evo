#!/usr/bin/env python3
"""Extract trajectory data from a specific run."""

import h5py
import json
import numpy as np
from pathlib import Path
import sys


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):  # Handle numpy scalars
            return obj.item()
        return super(NumpyEncoder, self).default(obj)


def extract_and_save_trajectory(h5_path, output_filename=None):
    """Extract trajectory data from H5 file and save as JSON."""
    
    if output_filename is None:
        output_filename = f"trajectory_{Path(h5_path).parent.name}.json"
    
    output_path = Path("visualization/3d") / output_filename
    
    print(f"Extracting from: {h5_path}")
    
    data = {}
    
    with h5py.File(h5_path, 'r') as f:
        # Extract trajectory data
        if 'trajectory' in f:
            traj_group = f['trajectory']
            positions = traj_group['positions'][:]
            actions = traj_group['actions'][:]
            rewards = traj_group['rewards'][:]
            observations = traj_group['observations'][:]
            print(f"Found trajectory data: {len(positions)} steps")
        else:
            print("No trajectory data found!")
            return False
        
        # Extract configuration
        config = {}
        if 'config' in f:
            config_group = f['config']
            for key in config_group.attrs:
                value = config_group.attrs[key]
                if isinstance(value, str) and value.startswith('{'):
                    config[key] = json.loads(value)
                else:
                    config[key] = value
        
        # Extract final state
        final_state = {}
        if 'final_state' in f:
            state_group = f['final_state']
            for key in state_group:
                final_state[key] = state_group[key][:]
        
        # Extract summary
        summary = {}
        if 'summary' in f:
            summary_group = f['summary']
            for key in summary_group.attrs:
                summary[key] = summary_group.attrs[key]
    
    # Build visualization data
    viz_data = {
        'metadata': {
            'gridSize': int(config['world_config']['grid_size']),
            'nRewards': int(config['world_config']['n_rewards']),
            'totalSteps': len(positions) - 1,
            'totalReward': float(summary.get('total_reward', 0)),
            'rewardsCollected': int(summary.get('rewards_collected', 0)),
            'coverage': float(summary.get('coverage', 0)),
            'seed': int(config.get('seed', -1)),
            'sourcePath': str(h5_path)
        },
        'trajectory': []
    }
    
    # Build trajectory with state at each timestep
    reward_positions = final_state['reward_positions']
    
    # Track reward collection over time
    rewards_collected_so_far = np.zeros(len(reward_positions), dtype=bool)
    cumulative_reward = 0.0
    
    for i in range(len(positions)):
        step_data = {
            'step': i,
            'agentPos': positions[i].tolist(),
            'observation': float(observations[i]),
            'cumulativeReward': cumulative_reward
        }
        
        if i > 0:
            step_data['action'] = int(actions[i-1])
            step_data['reward'] = float(rewards[i-1])
            cumulative_reward += rewards[i-1]
            
            # Check if reward was collected
            if rewards[i-1] >= config['world_config']['reward_value']:
                # Find which reward was collected
                agent_pos = positions[i]
                for j, reward_pos in enumerate(reward_positions):
                    if not rewards_collected_so_far[j]:
                        dist = np.linalg.norm(agent_pos - reward_pos)
                        if dist < 0.5:
                            rewards_collected_so_far[j] = True
                            break
        
        step_data['rewardCollected'] = rewards_collected_so_far.tolist()
        viz_data['trajectory'].append(step_data)
    
    # Add static world data
    viz_data['world'] = {
        'rewardPositions': [[int(x), int(y)] for x, y in reward_positions]
    }
    
    # Save as JSON
    with open(output_path, 'w') as f:
        json.dump(viz_data, f, indent=2, cls=NumpyEncoder)
    
    print(f"Visualization data saved to: {output_path}")
    print(f"  Grid size: {viz_data['metadata']['gridSize']}x{viz_data['metadata']['gridSize']}")
    print(f"  Total steps: {viz_data['metadata']['totalSteps']}")
    print(f"  Rewards collected: {viz_data['metadata']['rewardsCollected']}/{viz_data['metadata']['nRewards']}")
    print(f"  Total reward: {viz_data['metadata']['totalReward']:.1f}")
    
    return True


if __name__ == "__main__":
    if len(sys.argv) > 1:
        h5_path = sys.argv[1]
        output_filename = sys.argv[2] if len(sys.argv) > 2 else None
        extract_and_save_trajectory(h5_path, output_filename)
    else:
        print("Usage: python extract_run_data.py <h5_file_path> [output_filename]")