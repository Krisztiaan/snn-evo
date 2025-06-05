#!/usr/bin/env python3
"""Extract trajectory data from minimal exporter H5 files."""

import h5py
import json
import numpy as np
from pathlib import Path


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


def extract_minimal_export_data(h5_path):
    """Extract data from minimal exporter H5 file."""
    print(f"Extracting data from: {h5_path}")
    
    data = {}
    
    with h5py.File(h5_path, 'r') as f:
        # Extract trajectory data
        if 'trajectory' in f:
            traj_group = f['trajectory']
            data['positions'] = traj_group['positions'][:]
            data['actions'] = traj_group['actions'][:]
            data['rewards'] = traj_group['rewards'][:]
            data['observations'] = traj_group['observations'][:]
            print(f"Found trajectory data: {data['positions'].shape[0]} steps")
        
        # Extract configuration
        if 'config' in f:
            config_group = f['config']
            data['config'] = {}
            for key in config_group.attrs:
                value = config_group.attrs[key]
                if isinstance(value, str) and value.startswith('{'):
                    # Parse JSON string
                    data['config'][key] = json.loads(value)
                else:
                    data['config'][key] = value
        
        # Extract final state
        if 'final_state' in f:
            state_group = f['final_state']
            data['final_state'] = {}
            for key in state_group:
                data['final_state'][key] = state_group[key][:]
        
        # Extract summary
        if 'summary' in f:
            summary_group = f['summary']
            data['summary'] = {}
            for key in summary_group.attrs:
                data['summary'][key] = summary_group.attrs[key]
    
    return data


def save_for_visualization(data, output_path):
    """Save data in format suitable for Three.js visualization."""
    
    # Prepare visualization data
    viz_data = {
        'metadata': {
            'gridSize': int(data['config']['world_config']['grid_size']),
            'nRewards': int(data['config']['world_config']['n_rewards']),
            'totalSteps': len(data['positions']) - 1,
            'totalReward': float(data['summary']['total_reward']),
            'rewardsCollected': int(data['summary']['rewards_collected'])
        },
        'trajectory': []
    }
    
    # Build trajectory with state at each timestep
    positions = data['positions']
    actions = data['actions']
    rewards = data['rewards']
    observations = data['observations']
    reward_positions = data['final_state']['reward_positions']
    reward_collected = data['final_state']['reward_collected']
    
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
            if rewards[i-1] >= data['config']['world_config']['reward_value']:
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


def main():
    # Find the latest H5 file
    logs_dir = Path("models/random/logs")
    h5_files = list(logs_dir.glob("*/data.h5"))
    
    if not h5_files:
        print("No H5 files found!")
        return
    
    # Use the most recent file
    latest_h5 = sorted(h5_files, key=lambda p: p.stat().st_mtime)[-1]
    print(f"Using latest file: {latest_h5}")
    
    # Extract data
    data = extract_minimal_export_data(latest_h5)
    
    # Save for visualization
    output_path = Path("visualization/3d/trajectory_data.json")
    save_for_visualization(data, output_path)
    
    # Print summary
    print("\nTrajectory Summary:")
    print(f"  Grid size: {data['config']['world_config']['grid_size']}x{data['config']['world_config']['grid_size']}")
    print(f"  Total steps: {len(data['positions']) - 1}")
    print(f"  Total reward: {data['summary']['total_reward']}")
    print(f"  Rewards collected: {data['summary']['rewards_collected']}/{data['config']['world_config']['n_rewards']}")
    print(f"  Coverage: {data['summary']['coverage']:.1%}")


if __name__ == "__main__":
    main()