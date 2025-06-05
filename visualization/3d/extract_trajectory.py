#!/usr/bin/env python3
"""Extract trajectory data from HDF5 experiment files."""

import h5py
import json
import numpy as np
from pathlib import Path


def explore_h5_structure(file_path):
    """Recursively explore HDF5 file structure."""
    print(f"\nExploring HDF5 file: {file_path}")
    
    def print_structure(name, obj):
        indent = "  " * name.count('/')
        if isinstance(obj, h5py.Group):
            print(f"{indent}{name}/ (Group)")
            # Print attributes
            if obj.attrs:
                for attr_name, attr_value in obj.attrs.items():
                    print(f"{indent}  @{attr_name}: {attr_value}")
        elif isinstance(obj, h5py.Dataset):
            print(f"{indent}{name} (Dataset) - shape: {obj.shape}, dtype: {obj.dtype}")
            # Print first few values if small
            if obj.size < 10:
                print(f"{indent}  values: {obj[:]}")
    
    with h5py.File(file_path, 'r') as f:
        f.visititems(print_structure)


def extract_trajectory_data(file_path):
    """Extract trajectory and state data from HDF5 file."""
    data = {}
    
    with h5py.File(file_path, 'r') as f:
        # Check all possible locations for trajectory data
        possible_paths = [
            'episodes/episode_0000/behavior',
            'episodes/episode_0000/trajectory',
            'episodes/episode_0000/positions',
            'episodes/episode_0000/agent_positions',
            'behavior',
            'trajectory',
            'data'
        ]
        
        print("\nSearching for trajectory data...")
        for path in possible_paths:
            if path in f:
                print(f"Found data at: {path}")
                group = f[path]
                if isinstance(group, h5py.Group):
                    for key in group.keys():
                        print(f"  - {key}: {group[key].shape if hasattr(group[key], 'shape') else 'N/A'}")
                        data[key] = group[key][:]
                elif isinstance(group, h5py.Dataset):
                    print(f"  Dataset shape: {group.shape}")
                    data[path] = group[:]
        
        # Also check root level datasets
        print("\nRoot level datasets:")
        for key in f.keys():
            if isinstance(f[key], h5py.Dataset):
                print(f"  - {key}: {f[key].shape}")
                data[key] = f[key][:]
                
        # Check episode attributes
        if 'episodes/episode_0000' in f:
            episode = f['episodes/episode_0000']
            print("\nEpisode attributes:")
            for attr_name, attr_value in episode.attrs.items():
                print(f"  - {attr_name}: {attr_value}")
                data[f'episode_{attr_name}'] = attr_value
    
    return data


def main():
    # Path to the random agent's HDF5 file
    h5_path = Path("/Users/krisztiaan/dev/metalearning/experiments/models/random/logs/episode_000_20250605_191502_20250605_191502/experiment_data.h5")
    
    # First explore the structure
    explore_h5_structure(h5_path)
    
    # Then extract data
    print("\n" + "="*50)
    data = extract_trajectory_data(h5_path)
    
    # Save extracted data
    if data:
        output_path = h5_path.parent / "extracted_trajectory.json"
        # Convert numpy arrays to lists for JSON serialization
        json_data = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                json_data[key] = value.tolist()
            else:
                json_data[key] = value
        
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"\nExtracted data saved to: {output_path}")
    else:
        print("\nNo trajectory data found!")


if __name__ == "__main__":
    main()