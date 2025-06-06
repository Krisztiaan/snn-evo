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
        # Use abstract base classes for NumPy 2.0 compatibility
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.bool_):
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


# === Data Alignment Utilities ===

def align_timesteps(sparse_data: dict, reference_timesteps: np.ndarray, 
                   default_value: Any = None) -> np.ndarray:
    """Align sparse data to reference timesteps.
    
    Args:
        sparse_data: Dict with 'timesteps' and 'values' arrays
        reference_timesteps: Target timesteps to align to
        default_value: Value to use for missing timesteps
        
    Returns:
        Array of values aligned to reference timesteps
    """
    if not sparse_data or 'timesteps' not in sparse_data:
        return np.full(len(reference_timesteps), default_value)
        
    # Create output array
    aligned = np.full(len(reference_timesteps), default_value)
    
    # Map sparse timesteps to indices
    sparse_ts = sparse_data['timesteps']
    sparse_vals = sparse_data.get('values', sparse_data.get('rewards', []))
    
    # Use searchsorted for efficient matching
    indices = np.searchsorted(reference_timesteps, sparse_ts)
    
    # Only keep valid indices
    valid_mask = (indices < len(reference_timesteps)) & (reference_timesteps[indices] == sparse_ts)
    valid_indices = indices[valid_mask]
    valid_values = sparse_vals[valid_mask]
    
    # Fill in the values
    aligned[valid_indices] = valid_values
    
    return aligned


def interpolate_sampled_data(sampled_data: dict, reference_timesteps: np.ndarray,
                            method: str = 'nearest') -> dict:
    """Interpolate sampled data to match reference timesteps.
    
    Args:
        sampled_data: Dict with 'timesteps' and data arrays
        reference_timesteps: Target timesteps to interpolate to
        method: Interpolation method ('nearest', 'linear', 'zero')
        
    Returns:
        Dict with interpolated data arrays
    """
    if not sampled_data or 'timesteps' not in sampled_data:
        return {}
        
    sampled_ts = np.array(sampled_data['timesteps'])
    interpolated = {}
    
    # Skip timesteps key
    data_keys = [k for k in sampled_data.keys() if k != 'timesteps']
    
    for key in data_keys:
        data = np.array(sampled_data[key])
        
        if len(data.shape) == 1:
            # 1D data - simple interpolation
            if method == 'nearest':
                # Find nearest sampled timestep for each reference timestep
                indices = np.searchsorted(sampled_ts, reference_timesteps)
                indices = np.clip(indices, 0, len(sampled_ts) - 1)
                
                # Check if previous index is closer
                for i in range(len(indices)):
                    if indices[i] > 0:
                        curr_dist = abs(sampled_ts[indices[i]] - reference_timesteps[i])
                        prev_dist = abs(sampled_ts[indices[i]-1] - reference_timesteps[i])
                        if prev_dist < curr_dist:
                            indices[i] -= 1
                            
                interpolated[key] = data[indices]
                
            elif method == 'linear':
                # Linear interpolation
                interpolated[key] = np.interp(reference_timesteps, sampled_ts, data)
                
            elif method == 'zero':
                # Zero-order hold (forward fill)
                indices = np.searchsorted(sampled_ts, reference_timesteps, side='right') - 1
                indices = np.clip(indices, 0, len(sampled_ts) - 1)
                interpolated[key] = data[indices]
                
        else:
            # Multi-dimensional data - interpolate each dimension
            result = np.zeros((len(reference_timesteps),) + data.shape[1:])
            
            if method == 'nearest':
                indices = np.searchsorted(sampled_ts, reference_timesteps)
                indices = np.clip(indices, 0, len(sampled_ts) - 1)
                
                for i in range(len(indices)):
                    if indices[i] > 0:
                        curr_dist = abs(sampled_ts[indices[i]] - reference_timesteps[i])
                        prev_dist = abs(sampled_ts[indices[i]-1] - reference_timesteps[i])
                        if prev_dist < curr_dist:
                            indices[i] -= 1
                            
                result = data[indices]
                
            elif method == 'linear':
                # Interpolate each feature separately
                for j in range(data.shape[1]):
                    result[:, j] = np.interp(reference_timesteps, sampled_ts, data[:, j])
                    
            elif method == 'zero':
                indices = np.searchsorted(sampled_ts, reference_timesteps, side='right') - 1
                indices = np.clip(indices, 0, len(sampled_ts) - 1)
                result = data[indices]
                
            interpolated[key] = result
    
    interpolated['timesteps'] = reference_timesteps
    return interpolated


def create_timestep_mapping(behavior_timesteps: np.ndarray, 
                          neural_timesteps: np.ndarray,
                          spike_timesteps: np.ndarray = None) -> dict:
    """Create mapping between different data stream timesteps.
    
    Returns dict with mappings for efficient lookup.
    """
    mapping = {
        'behavior_to_neural': {},
        'neural_to_behavior': {},
        'spike_bins': {}
    }
    
    # Map behavior timesteps to nearest neural timesteps
    for i, b_ts in enumerate(behavior_timesteps):
        # Find nearest neural timestep
        idx = np.searchsorted(neural_timesteps, b_ts)
        if idx > 0 and (idx == len(neural_timesteps) or 
                        abs(neural_timesteps[idx-1] - b_ts) < abs(neural_timesteps[idx] - b_ts)):
            idx -= 1
        idx = min(idx, len(neural_timesteps) - 1)
        mapping['behavior_to_neural'][i] = idx
        
    # Reverse mapping
    for b_idx, n_idx in mapping['behavior_to_neural'].items():
        if n_idx not in mapping['neural_to_behavior']:
            mapping['neural_to_behavior'][n_idx] = []
        mapping['neural_to_behavior'][n_idx].append(b_idx)
        
    # Create spike bins if spike data provided
    if spike_timesteps is not None:
        # Bin spikes by behavior timestep
        spike_bins = defaultdict(list)
        for spike_ts in spike_timesteps:
            # Find behavior timestep bin
            idx = np.searchsorted(behavior_timesteps, spike_ts)
            if idx > 0:
                spike_bins[behavior_timesteps[idx-1]].append(spike_ts)
        mapping['spike_bins'] = dict(spike_bins)
        
    return mapping