# keywords: [minimal exporter, simple hdf5 export, no pandas]
"""Minimal data exporter without pandas dependency."""

import h5py
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


class MinimalExporter:
    """Minimal HDF5 exporter for random agent data."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create HDF5 file
        self.h5_path = self.output_dir / "data.h5"
        self.h5_file = h5py.File(self.h5_path, 'w')
        
        # Add timestamp
        self.h5_file.attrs['created'] = datetime.now().isoformat()
        
    def save_config(self, config: Dict[str, Any]):
        """Save configuration to HDF5."""
        config_group = self.h5_file.create_group('config')
        
        # Save each config item
        for key, value in config.items():
            if isinstance(value, dict):
                # Convert dict to JSON string
                config_group.attrs[key] = json.dumps(value)
            else:
                config_group.attrs[key] = value
                
        # Also save as JSON file
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
    
    def save_metadata(self, metadata: Dict[str, Any]):
        """Save metadata."""
        for key, value in metadata.items():
            self.h5_file.attrs[f'meta_{key}'] = value
            
        # Also save as JSON
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def save_trajectory(self, trajectory: Dict[str, np.ndarray]):
        """Save trajectory data."""
        traj_group = self.h5_file.create_group('trajectory')
        
        for key, data in trajectory.items():
            traj_group.create_dataset(key, data=data, compression='gzip')
    
    def save_summary(self, summary: Dict[str, Any]):
        """Save episode summary."""
        summary_group = self.h5_file.create_group('summary')
        
        for key, value in summary.items():
            if isinstance(value, (tuple, list)):
                summary_group.create_dataset(key, data=np.array(value))
            else:
                summary_group.attrs[key] = value
                
        # Also save as JSON
        with open(self.output_dir / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
    
    def save_final_state(self, state: Dict[str, Any]):
        """Save final world state."""
        state_group = self.h5_file.create_group('final_state')
        
        for key, value in state.items():
            if isinstance(value, np.ndarray):
                state_group.create_dataset(key, data=value, compression='gzip')
            else:
                state_group.attrs[key] = value
    
    def close(self):
        """Close the HDF5 file."""
        self.h5_file.close()
        print(f"Data saved to: {self.h5_path}")