# keywords: [exporter, hdf5, simple, streaming, data, universal]
"""Simple HDF5-based data exporter for SNN experiments."""

import h5py
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Union, Tuple, List
import warnings

from .utils import ensure_numpy
from .schema import validate_timestep_data, validate_weight_change, validate_network_structure, SCHEMA_VERSION


class Episode:
    """HDF5 episode data storage."""
    
    def __init__(self, 
                 episode_id: int, 
                 h5_file: h5py.File,
                 neural_sampling_rate: int = 100,
                 validate_data: bool = True):
        """Initialize episode within an HDF5 file.
        
        Args:
            episode_id: Episode identifier
            h5_file: Open HDF5 file handle
            neural_sampling_rate: Sample neural state every N timesteps
            validate_data: Whether to validate data before saving
        """
        self.episode_id = episode_id
        self.h5_file = h5_file
        self.neural_sampling_rate = neural_sampling_rate
        self.validate_data = validate_data
        
        # Create episode group
        self.group = h5_file.create_group(f'episode_{episode_id:04d}')
        
        # Metadata
        self.group.attrs['episode_id'] = episode_id
        self.group.attrs['start_time'] = datetime.now().isoformat()
        self.group.attrs['neural_sampling_rate'] = neural_sampling_rate
        self.group.attrs['status'] = 'running'
        
        # Create subgroups
        self.neural_group = self.group.create_group('neural_states')
        self.spike_group = self.group.create_group('spikes')
        self.behavior_group = self.group.create_group('behavior')
        self.reward_group = self.group.create_group('rewards')
        self.weight_group = self.group.create_group('weight_changes')
        self.event_group = self.group.create_group('events')
        
        # Tracking
        self.timestep_count = 0
        self.last_neural_sample = -neural_sampling_rate
        self._initialized = {}
        
    def log_timestep(self,
                     timestep: int,
                     neural_state: Optional[Dict[str, Any]] = None,
                     spikes: Optional[Any] = None,
                     behavior: Optional[Dict[str, Any]] = None,
                     reward: Optional[float] = None):
        """Log data for a single timestep."""
        # Validate if enabled
        if self.validate_data:
            warnings_list = validate_timestep_data(timestep, neural_state, spikes, behavior, reward)
            for w in warnings_list:
                warnings.warn(f"Validation: {w}")
                
        self.timestep_count += 1
        
        # Neural state (sampled)
        if neural_state is not None and timestep - self.last_neural_sample >= self.neural_sampling_rate:
            self._append_dict_data(self.neural_group, timestep, neural_state)
            self.last_neural_sample = timestep
            
        # Spikes (sparse)
        if spikes is not None:
            spike_data = ensure_numpy(spikes)
            if spike_data.any():
                indices = np.where(spike_data)[0]
                self._append_sparse_data(self.spike_group, 'spikes', timestep, indices)
                
        # Behavior (all timesteps)
        if behavior is not None:
            self._append_dict_data(self.behavior_group, timestep, behavior)
            
        # Rewards (sparse)
        if reward is not None and reward != 0:
            self._append_sparse_data(self.reward_group, 'rewards', timestep, [reward])
            
    def log_weight_change(self,
                          timestep: int,
                          synapse_id: Union[int, Tuple[int, int]],
                          old_weight: float,
                          new_weight: float,
                          **kwargs):
        """Log weight change event."""
        if self.validate_data:
            warnings_list = validate_weight_change(timestep, synapse_id, old_weight, new_weight)
            for w in warnings_list:
                warnings.warn(f"Validation: {w}")
                
        data = {
            'old_weight': old_weight,
            'new_weight': new_weight,
            'delta': new_weight - old_weight
        }
        
        if isinstance(synapse_id, tuple):
            data['source_id'] = synapse_id[0]
            data['target_id'] = synapse_id[1]
        else:
            data['synapse_id'] = synapse_id
            
        data.update(kwargs)
        self._append_dict_data(self.weight_group, timestep, data)
        
    def log_event(self, event_type: str, timestep: int, data: Dict[str, Any]):
        """Log custom event."""
        if event_type not in self.event_group:
            self.event_group.create_group(event_type)
        event_subgroup = self.event_group[event_type]
        self._append_dict_data(event_subgroup, timestep, data)
        
    def end(self, success: bool = False, final_state: Optional[Dict[str, Any]] = None):
        """End episode and save metadata."""
        # Update attributes
        self.group.attrs['end_time'] = datetime.now().isoformat()
        self.group.attrs['total_timesteps'] = self.timestep_count
        self.group.attrs['success'] = success
        self.group.attrs['status'] = 'completed'
        
        # Calculate summary statistics
        if 'timesteps' in self.spike_group:
            self.group.attrs['total_spikes'] = len(self.spike_group['timesteps'])
        if 'timesteps' in self.reward_group:
            self.group.attrs['total_reward'] = float(np.sum(self.reward_group['values']))
        if 'timesteps' in self.weight_group:
            self.group.attrs['total_weight_changes'] = len(self.weight_group['timesteps'])
            
        # Save final state
        if final_state:
            final_group = self.group.create_group('final_state')
            for key, value in final_state.items():
                final_group.create_dataset(key, data=ensure_numpy(value))
                
    def _append_dict_data(self, group: h5py.Group, timestep: int, data: Dict[str, Any]):
        """Append dictionary data to HDF5 group."""
        # Initialize timesteps dataset if needed
        if 'timesteps' not in group:
            group.create_dataset(
                'timesteps', 
                shape=(0,), 
                maxshape=(None,), 
                dtype='i8',
                chunks=(min(1000, 100)),  # Reasonable chunk size
                # Note: compression could be added here for optimization
            )
            
        # Append timestep
        ts_data = group['timesteps']
        idx = ts_data.shape[0]
        ts_data.resize(idx + 1, axis=0)
        ts_data[idx] = timestep
        
        # Append each data field
        for key, value in data.items():
            value_np = ensure_numpy(value)
            
            if key not in group:
                # Create dataset on first occurrence
                if value_np.ndim == 0:
                    shape, maxshape = (0,), (None,)
                else:
                    shape, maxshape = (0,) + value_np.shape, (None,) + value_np.shape
                    
                group.create_dataset(key, shape=shape, maxshape=maxshape, dtype=value_np.dtype)
                
            # Append value
            dataset = group[key]
            dataset.resize(idx + 1, axis=0)
            dataset[idx] = value_np
            
    def _append_sparse_data(self, group: h5py.Group, name: str, timestep: int, values: Union[List, np.ndarray]):
        """Append sparse data (timestep, values pairs)."""
        values_np = ensure_numpy(values)
        n_values = len(values_np)
        
        if n_values == 0:
            return
            
        # Initialize datasets if needed
        if 'timesteps' not in group:
            group.create_dataset('timesteps', shape=(0,), maxshape=(None,), dtype='i8')
            group.create_dataset('values', shape=(0,), maxshape=(None,), dtype=values_np.dtype)
            
        # Get current size
        start_idx = group['timesteps'].shape[0]
        end_idx = start_idx + n_values
        
        # Resize and append
        group['timesteps'].resize(end_idx, axis=0)
        group['values'].resize(end_idx, axis=0)
        
        group['timesteps'][start_idx:end_idx] = timestep
        group['values'][start_idx:end_idx] = values_np


class DataExporter:
    """Simple HDF5-based data exporter for SNN experiments."""
    
    def __init__(self,
                 experiment_name: str,
                 output_base_dir: str = "experiments",
                 neural_sampling_rate: int = 100,
                 validate_data: bool = True,
                 compression: str = 'gzip',
                 compression_level: int = 4):
        """Initialize exporter.
        
        Args:
            experiment_name: Name of the experiment
            output_base_dir: Base directory for output
            neural_sampling_rate: Sample neural state every N timesteps
            validate_data: Whether to validate data
            compression: HDF5 compression ('gzip', 'lzf', None)
            compression_level: Compression level (1-9 for gzip)
        """
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.neural_sampling_rate = neural_sampling_rate
        self.validate_data = validate_data
        
        # Create output directory
        self.output_dir = Path(output_base_dir) / f"{experiment_name}_{self.timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # HDF5 settings
        self.compression = compression
        self.compression_opts = compression_level if compression == 'gzip' else None
        
        # Store compression settings for dataset creation
        self._compression_kwargs = {}
        if compression:
            self._compression_kwargs = {
                'compression': compression,
                'compression_opts': self.compression_opts
            }
            
        # Create main HDF5 file
        self.h5_path = self.output_dir / 'experiment_data.h5'
        self.h5_file = h5py.File(self.h5_path, 'w')
        
        # Set experiment metadata
        self.h5_file.attrs['experiment_name'] = experiment_name
        self.h5_file.attrs['timestamp'] = self.timestamp
        self.h5_file.attrs['start_time'] = datetime.now().isoformat()
        self.h5_file.attrs['schema_version'] = SCHEMA_VERSION
        self.h5_file.attrs['neural_sampling_rate'] = neural_sampling_rate
        self.h5_file.attrs['compression'] = compression or 'none'
        
        # Create groups
        self.episodes_group = self.h5_file.create_group('episodes')
        self.checkpoints_group = self.h5_file.create_group('checkpoints')
        
        # State
        self.current_episode: Optional[Episode] = None
        self.episode_count = 0
        
        print(f"Initialized DataExporter: {self.output_dir}")
        
        # Automatically capture runtime info
        self.save_runtime_info()
        
        # Capture command line arguments if running as script
        import sys
        if len(sys.argv) > 0:
            self.h5_file.attrs['command_line'] = ' '.join(sys.argv)
        
    def save_config(self, config: Dict[str, Any]):
        """Save experiment configuration."""
        # Save as JSON for readability
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config, f, indent=2)
            
        # Also save in HDF5
        if 'config' in self.h5_file:
            del self.h5_file['config']  # Replace if exists
        config_group = self.h5_file.create_group('config')
        
        # Store both as attributes and datasets for flexibility
        for key, value in config.items():
            if isinstance(value, (dict, list)):
                config_group.attrs[key] = json.dumps(value)
            else:
                config_group.attrs[key] = value
                
    def save_metadata(self, metadata: Dict[str, Any]):
        """Save additional experiment metadata (author, description, etc)."""
        # Update root attributes
        for key, value in metadata.items():
            if isinstance(value, (dict, list)):
                self.h5_file.attrs[f'meta_{key}'] = json.dumps(value)
            else:
                self.h5_file.attrs[f'meta_{key}'] = value
                
        # Also save as JSON for convenience
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
            
    def save_runtime_info(self):
        """Automatically capture runtime environment info."""
        import sys
        import platform
        try:
            import jax
            jax_version = jax.__version__
            jax_devices = str(jax.devices())
        except:
            jax_version = None
            jax_devices = None
            
        runtime_info = {
            'python_version': sys.version,
            'platform': platform.platform(),
            'hostname': platform.node(),
            'numpy_version': np.__version__,
            'h5py_version': h5py.__version__,
            'jax_version': jax_version,
            'jax_devices': jax_devices,
            'working_directory': str(Path.cwd()),
            'output_directory': str(self.output_dir)
        }
        
        # Store in HDF5
        if 'runtime' in self.h5_file:
            del self.h5_file['runtime']
        runtime_group = self.h5_file.create_group('runtime')
        for key, value in runtime_info.items():
            if value is not None:
                runtime_group.attrs[key] = str(value)
                
        # Also save as JSON
        with open(self.output_dir / 'runtime_info.json', 'w') as f:
            json.dump(runtime_info, f, indent=2)
            
        return runtime_info
    
    def save_code_snapshot(self, code_files: Optional[List[Union[str, Path]]] = None):
        """Save snapshot of code files for reproducibility.
        
        Args:
            code_files: List of Python files to snapshot. If None, saves the main script.
        """
        import sys
        import inspect
        
        code_group = self.h5_file.create_group('code_snapshot')
        
        if code_files is None:
            # Try to get the main script
            if hasattr(sys.modules['__main__'], '__file__'):
                main_file = sys.modules['__main__'].__file__
                if main_file:
                    code_files = [main_file]
            else:
                code_files = []
                
        saved_files = []
        for file_path in code_files:
            file_path = Path(file_path)
            if file_path.exists() and file_path.suffix == '.py':
                try:
                    with open(file_path, 'r') as f:
                        code_content = f.read()
                    
                    # Store in HDF5
                    code_group.attrs[file_path.name] = code_content
                    saved_files.append(str(file_path))
                    
                    # Also save as separate file
                    code_dir = self.output_dir / 'code_snapshot'
                    code_dir.mkdir(exist_ok=True)
                    with open(code_dir / file_path.name, 'w') as f:
                        f.write(code_content)
                except Exception as e:
                    warnings.warn(f"Could not save code file {file_path}: {e}")
                    
        code_group.attrs['saved_files'] = json.dumps(saved_files)
        code_group.attrs['snapshot_time'] = datetime.now().isoformat()
        
        return saved_files
    
    def save_git_info(self):
        """Save git repository information if available."""
        try:
            import subprocess
            
            git_info = {}
            
            # Get git commit hash
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True, cwd=Path.cwd())
            if result.returncode == 0:
                git_info['commit_hash'] = result.stdout.strip()
                
            # Get git branch
            result = subprocess.run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                                  capture_output=True, text=True, cwd=Path.cwd())
            if result.returncode == 0:
                git_info['branch'] = result.stdout.strip()
                
            # Check if working directory is clean
            result = subprocess.run(['git', 'status', '--porcelain'],
                                  capture_output=True, text=True, cwd=Path.cwd())
            if result.returncode == 0:
                git_info['dirty'] = len(result.stdout.strip()) > 0
                if git_info['dirty']:
                    git_info['uncommitted_files'] = result.stdout.strip().split('\n')
                    
            # Get remote URL
            result = subprocess.run(['git', 'config', '--get', 'remote.origin.url'],
                                  capture_output=True, text=True, cwd=Path.cwd())
            if result.returncode == 0:
                git_info['remote_url'] = result.stdout.strip()
                
            # Save to HDF5
            if git_info:
                git_group = self.h5_file.create_group('git_info')
                for key, value in git_info.items():
                    if isinstance(value, list):
                        git_group.attrs[key] = json.dumps(value)
                    else:
                        git_group.attrs[key] = str(value)
                        
                # Also save as JSON
                with open(self.output_dir / 'git_info.json', 'w') as f:
                    json.dump(git_info, f, indent=2)
                    
            return git_info
            
        except Exception as e:
            warnings.warn(f"Could not get git info: {e}")
            return None
                
    def save_network_structure(self,
                              neurons: Dict[str, Any],
                              connections: Dict[str, Any],
                              initial_weights: Optional[Union[np.ndarray, Dict[str, Any]]] = None):
        """Save network structure."""
        if self.validate_data:
            warnings_list = validate_network_structure(neurons, connections)
            for w in warnings_list:
                warnings.warn(f"Network validation: {w}")
                
        # Create network group
        net_group = self.h5_file.create_group('network_structure')
        
        # Save neuron data
        neurons_group = net_group.create_group('neurons')
        for key, value in neurons.items():
            neurons_group.create_dataset(key, data=ensure_numpy(value))
            
        # Save connection data
        conn_group = net_group.create_group('connections')
        for key, value in connections.items():
            conn_group.create_dataset(key, data=ensure_numpy(value))
            
        # Save initial weights
        if initial_weights is not None:
            weights_group = net_group.create_group('initial_weights')
            
            if isinstance(initial_weights, dict):
                # Sparse format
                for key, value in initial_weights.items():
                    weights_group.create_dataset(key, data=ensure_numpy(value))
                weights_group.attrs['format'] = 'sparse'
            else:
                # Dense matrix
                weights_np = ensure_numpy(initial_weights)
                if weights_np.ndim == 2:
                    # Convert to sparse
                    nonzero_mask = weights_np != 0
                    indices = np.argwhere(nonzero_mask)
                    values = weights_np[nonzero_mask]
                    
                    weights_group.create_dataset('indices', data=indices)
                    weights_group.create_dataset('values', data=values)
                    weights_group.attrs['format'] = 'sparse'
                    weights_group.attrs['shape'] = weights_np.shape
                    weights_group.attrs['sparsity'] = 1.0 - (len(values) / weights_np.size)
                else:
                    # 1D weights
                    weights_group.create_dataset('weights', data=weights_np)
                    weights_group.attrs['format'] = 'dense_1d'
                    
        # Update metadata
        self.h5_file.attrs['n_neurons'] = len(neurons.get('neuron_ids', []))
        self.h5_file.attrs['n_connections'] = len(connections.get('source_ids', []))
        
    def start_episode(self, episode_id: Optional[int] = None) -> Episode:
        """Start new episode."""
        if self.current_episode is not None:
            warnings.warn(f"Previous episode {self.current_episode.episode_id} not ended")
            self.current_episode.end(success=False)
            
        if episode_id is None:
            episode_id = self.episode_count
            
        self.current_episode = Episode(
            episode_id=episode_id,
            h5_file=self.episodes_group,
            neural_sampling_rate=self.neural_sampling_rate,
            validate_data=self.validate_data
        )
        
        self.episode_count += 1
        self.h5_file.attrs['episode_count'] = self.episode_count
        
        return self.current_episode
        
    def end_episode(self, success: bool = False, summary: Optional[Dict[str, Any]] = None):
        """End current episode."""
        if self.current_episode is None:
            warnings.warn("No active episode to end")
            return
            
        self.current_episode.end(success=success)
        
        # Update experiment summary
        if 'episode_summaries' not in self.h5_file:
            self.h5_file.create_group('episode_summaries')
            
        summary_group = self.h5_file['episode_summaries']
        ep_summary = summary_group.create_group(f'episode_{self.current_episode.episode_id:04d}')
        
        # Copy episode attributes to summary
        for key, value in self.current_episode.group.attrs.items():
            ep_summary.attrs[key] = value
            
        # Add custom summary
        if summary:
            for key, value in summary.items():
                ep_summary.attrs[key] = value
                
        self.current_episode = None
        self.h5_file.flush()
        
    def log(self, **kwargs):
        """Log data to current episode."""
        if self.current_episode is None:
            raise RuntimeError("No active episode. Call start_episode() first.")
            
        timestep = kwargs.get('timestep')
        if timestep is None:
            raise ValueError("timestep is required")
            
        # Route to appropriate method
        if 'synapse_id' in kwargs and 'old_weight' in kwargs and 'new_weight' in kwargs:
            self.current_episode.log_weight_change(**kwargs)
        elif 'event_type' in kwargs and 'data' in kwargs:
            event_type = kwargs.pop('event_type')
            data = kwargs.pop('data')
            self.current_episode.log_event(event_type, timestep, data)
        else:
            self.current_episode.log_timestep(
                timestep=timestep,
                neural_state=kwargs.get('neural_state'),
                spikes=kwargs.get('spikes'),
                behavior=kwargs.get('behavior'),
                reward=kwargs.get('reward')
            )
            
    def save_checkpoint(self, name: str, data: Dict[str, Any]):
        """Save checkpoint."""
        checkpoint = self.checkpoints_group.create_group(f"{name}_{datetime.now().strftime('%H%M%S')}")
        checkpoint.attrs['timestamp'] = datetime.now().isoformat()
        checkpoint.attrs['episode_count'] = self.episode_count
        
        for key, value in data.items():
            checkpoint.create_dataset(key, data=ensure_numpy(value))
            
    def close(self):
        """Close HDF5 file and save final metadata."""
        # End any active episode
        if self.current_episode is not None:
            self.current_episode.end(success=False)
            
        # Update final metadata
        self.h5_file.attrs['end_time'] = datetime.now().isoformat()
        self.h5_file.attrs['total_episodes'] = self.episode_count
        
        # Close file
        self.h5_file.close()
        
        print(f"Experiment complete. Data saved to: {self.output_dir}")
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        
    def __del__(self):
        if hasattr(self, 'h5_file') and self.h5_file:
            try:
                self.h5_file.close()
            except:
                pass