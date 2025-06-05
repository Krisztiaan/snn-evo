#!/usr/bin/env python3
"""Scan for all available experiment runs and create an index."""

import json
import h5py
from pathlib import Path
from datetime import datetime
import numpy as np


def scan_experiment_runs(base_dir="models/random/logs"):
    """Scan for all H5 files in the experiment logs."""
    runs = []
    
    # Find all H5 files
    for h5_path in Path(base_dir).rglob("*.h5"):
        try:
            run_info = extract_run_info(h5_path)
            if run_info:
                runs.append(run_info)
        except Exception as e:
            print(f"Error processing {h5_path}: {e}")
    
    # Sort by timestamp (newest first)
    runs.sort(key=lambda x: x['timestamp'], reverse=True)
    
    return runs


def extract_run_info(h5_path):
    """Extract metadata from a single H5 file."""
    info = {
        'path': str(h5_path),
        'filename': h5_path.name,
        'directory': h5_path.parent.name
    }
    
    try:
        with h5py.File(h5_path, 'r') as f:
            # Extract config
            if 'config' in f:
                config_group = f['config']
                if 'world_config' in config_group.attrs:
                    world_config = json.loads(config_group.attrs['world_config'])
                    info['grid_size'] = world_config.get('grid_size', 'N/A')
                    info['n_rewards'] = world_config.get('n_rewards', 'N/A')
                    info['max_timesteps'] = world_config.get('max_timesteps', 'N/A')
                
                info['agent_type'] = config_group.attrs.get('agent_type', 'unknown')
                info['seed'] = int(config_group.attrs.get('seed', -1))
            
            # Extract trajectory info
            if 'trajectory' in f:
                traj_group = f['trajectory']
                if 'positions' in traj_group:
                    positions = traj_group['positions']
                    info['total_steps'] = len(positions) - 1
            
            # Extract summary
            if 'summary' in f:
                summary_group = f['summary']
                info['total_reward'] = float(summary_group.attrs.get('total_reward', 0))
                info['rewards_collected'] = int(summary_group.attrs.get('rewards_collected', 0))
                info['coverage'] = float(summary_group.attrs.get('coverage', 0))
            
            # Extract timestamp from directory name or file
            timestamp_str = h5_path.parent.name.split('_')[-2] + '_' + h5_path.parent.name.split('_')[-1]
            try:
                info['timestamp'] = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                info['timestamp_str'] = info['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            except:
                info['timestamp'] = datetime.now()
                info['timestamp_str'] = 'Unknown'
            
            # Create a descriptive label
            info['label'] = f"{info['timestamp_str']} - Grid {info.get('grid_size', '?')}x{info.get('grid_size', '?')} - {info.get('rewards_collected', 0)}/{info.get('n_rewards', '?')} rewards - {info.get('total_reward', 0):.1f} score"
            
            return info
            
    except Exception as e:
        print(f"Error reading {h5_path}: {e}")
        return None


def create_run_index(output_path="visualization/3d/runs_index.json"):
    """Create an index of all available runs."""
    runs = scan_experiment_runs()
    
    # Save as JSON
    # Convert datetime objects to strings for JSON serialization
    for run in runs:
        if 'timestamp' in run:
            run['timestamp'] = run['timestamp'].isoformat()
    
    with open(output_path, 'w') as f:
        json.dump({
            'runs': runs,
            'last_updated': datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"Found {len(runs)} runs")
    print(f"Index saved to: {output_path}")
    
    # Print summary
    print("\nAvailable runs:")
    for i, run in enumerate(runs[:10]):  # Show first 10
        print(f"  {i+1}. {run['label']}")
    
    if len(runs) > 10:
        print(f"  ... and {len(runs) - 10} more")


if __name__ == "__main__":
    create_run_index()