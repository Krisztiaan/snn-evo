# keywords: [utils, numpy, jax, conversion, helpers]
"""Utility functions for data export."""

import json
import numpy as np
from pathlib import Path
from typing import Any, Union
from collections import defaultdict


def ensure_numpy(arr: Any) -> np.ndarray:
    """Convert any array-like object to numpy array.
    
    Handles:
    - JAX DeviceArray
    - TensorFlow tensors
    - PyTorch tensors
    - Lists, tuples
    - Already numpy arrays
    """
    if isinstance(arr, np.ndarray):
        return arr
        
    # Check for JAX arrays
    if hasattr(arr, '__array__'):
        # This works for JAX DeviceArray
        return np.array(arr)
        
    # Check for common tensor types
    if hasattr(arr, 'numpy'):
        # Works for TF and PyTorch tensors
        return arr.numpy()
        
    # Check for JAX-specific conversion
    if hasattr(arr, 'to_py'):
        return np.array(arr.to_py())
        
    # Fallback to numpy conversion
    try:
        return np.array(arr)
    except Exception as e:
        raise TypeError(f"Cannot convert {type(arr)} to numpy array: {e}")


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int_)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float_)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, np.void):
            return None
        return super().default(obj)


def create_output_dir(base_path: Union[str, Path], name: str) -> Path:
    """Create a uniquely named output directory."""
    from datetime import datetime
    
    base_path = Path(base_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = base_path / f"{name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir


def load_episode_data(episode_dir: Union[str, Path]) -> dict:
    """Load all data from an episode directory, handling incremental saves."""
    episode_dir = Path(episode_dir)
    data = {}
    
    # Check if HDF5 file exists
    if (episode_dir / 'episode_data.h5').exists():
        # Load from HDF5
        from .hdf5_backend import load_hdf5_episode
        return load_hdf5_episode(episode_dir)
    
    # Otherwise load JSON format
    # Load metadata
    if (episode_dir / 'metadata.json').exists():
        with open(episode_dir / 'metadata.json', 'r') as f:
            data['metadata'] = json.load(f)
            
    # Load neural states (handle multiple files from flushes)
    neural_files = sorted(episode_dir.glob('neural_states*.npz'))
    if neural_files:
        all_neural = defaultdict(list)
        for file in neural_files:
            neural_data = np.load(file)
            for key in neural_data.files:
                all_neural[key].append(neural_data[key])
        
        # Concatenate all arrays
        data['neural_states'] = {
            key: np.concatenate(arrays) for key, arrays in all_neural.items()
        }
        
    # Load spikes (handle multiple files)
    import gzip
    spike_files = sorted(episode_dir.glob('spikes*.json.gz'))
    if spike_files:
        all_spikes = []
        for file in spike_files:
            with gzip.open(file, 'rt') as f:
                all_spikes.extend(json.load(f))
        data['spikes'] = all_spikes
            
    # Load behavior
    if (episode_dir / 'behavior.csv.gz').exists():
        import csv
        with gzip.open(episode_dir / 'behavior.csv.gz', 'rt') as f:
            reader = csv.DictReader(f)
            data['behavior'] = list(reader)
            
    # Load rewards (handle multiple files)
    reward_files = sorted(episode_dir.glob('rewards*.json.gz'))
    if reward_files:
        all_rewards = []
        for file in reward_files:
            with gzip.open(file, 'rt') as f:
                all_rewards.extend(json.load(f))
        data['rewards'] = all_rewards
            
    # Load weight changes (handle multiple files)
    weight_files = sorted(episode_dir.glob('weight_changes*.json.gz'))
    if weight_files:
        all_weight_changes = []
        for file in weight_files:
            with gzip.open(file, 'rt') as f:
                all_weight_changes.extend(json.load(f))
        data['weight_changes'] = all_weight_changes
        
    # Load custom events
    event_files = episode_dir.glob('events_*.json.gz')
    events = defaultdict(list)
    for file in event_files:
        # Extract event type from filename
        event_type = file.stem.split('_', 1)[1].rsplit('_', 1)[0]
        with gzip.open(file, 'rt') as f:
            events[event_type].extend(json.load(f))
    if events:
        data['events'] = dict(events)
            
    return data


def load_experiment_summary(experiment_dir: Union[str, Path]) -> dict:
    """Load experiment metadata and summaries."""
    experiment_dir = Path(experiment_dir)
    
    # Load experiment metadata
    with open(experiment_dir / 'experiment_metadata.json', 'r') as f:
        metadata = json.load(f)
        
    # Load episode summaries
    import csv
    summaries = []
    if (experiment_dir / 'episode_summaries.csv').exists():
        with open(experiment_dir / 'episode_summaries.csv', 'r') as f:
            reader = csv.DictReader(f)
            summaries = list(reader)
            
    return {
        'metadata': metadata,
        'episode_summaries': summaries
    }