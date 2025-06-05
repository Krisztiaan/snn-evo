# keywords: [exporter, hdf5, optimized, high-performance, streaming, chunked, compressed]
"""High-performance HDF5 data exporter with advanced optimizations."""

import h5py
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Union, Tuple, List, Set
import warnings
from collections import defaultdict
import threading
from queue import Queue
import zlib

from .utils import ensure_numpy, NumpyEncoder
from .schema import validate_timestep_data, validate_weight_change, validate_network_structure, SCHEMA_VERSION


class BufferedDataset:
    """Efficiently buffered dataset wrapper with batch resizing."""
    
    def __init__(self, 
                 group: h5py.Group,
                 name: str,
                 dtype: np.dtype,
                 shape: Tuple[int, ...],
                 chunk_size: int = 10000,
                 growth_factor: float = 1.5,
                 compression: str = 'gzip',
                 compression_opts: int = 4):
        """Initialize buffered dataset.
        
        Args:
            group: HDF5 group to create dataset in
            name: Dataset name
            dtype: Data type
            shape: Initial shape (first dim should be 0 for extensible)
            chunk_size: Chunk size for HDF5
            growth_factor: Growth factor for buffer resizing
            compression: Compression type
            compression_opts: Compression level
        """
        self.name = name
        self.dtype = dtype
        self.chunk_size = chunk_size
        self.growth_factor = growth_factor
        
        # Determine chunk shape
        if len(shape) == 1:
            chunks = (min(chunk_size, 1000),)
        else:
            chunks = (min(chunk_size, 1000),) + shape[1:]
        
        # Create dataset with compression
        maxshape = (None,) + shape[1:] if len(shape) > 1 else (None,)
        
        self.dataset = group.create_dataset(
            name,
            shape=shape,
            maxshape=maxshape,
            dtype=dtype,
            chunks=chunks,
            compression=compression,
            compression_opts=compression_opts,
            shuffle=True,  # Improves compression
            fletcher32=True  # Checksum for data integrity
        )
        
        # Buffer management
        self.buffer = []
        self.current_size = 0
        self.allocated_size = 0
        self.buffer_threshold = min(chunk_size // 10, 1000)
        
    def append(self, data: np.ndarray):
        """Append data to buffer."""
        self.buffer.append(data)
        
        # Flush if buffer is large enough
        if len(self.buffer) >= self.buffer_threshold:
            self.flush()
            
    def flush(self):
        """Flush buffer to dataset."""
        if not self.buffer:
            return
            
        # Stack buffer data
        if self.buffer[0].ndim == 0:
            new_data = np.array(self.buffer)
        else:
            new_data = np.vstack(self.buffer)
            
        n_new = len(new_data)
        
        # Resize if needed
        new_size = self.current_size + n_new
        if new_size > self.allocated_size:
            # Geometric growth
            self.allocated_size = int(max(
                new_size * self.growth_factor,
                self.allocated_size + self.chunk_size
            ))
            self.dataset.resize(self.allocated_size, axis=0)
            
        # Write data
        self.dataset[self.current_size:new_size] = new_data
        self.current_size = new_size
        
        # Clear buffer
        self.buffer = []
        
    def finalize(self):
        """Flush remaining data and trim to actual size."""
        self.flush()
        if self.current_size < self.allocated_size:
            self.dataset.resize(self.current_size, axis=0)


class SparseDataWriter:
    """Efficient sparse data writer with run-length encoding."""
    
    def __init__(self, group: h5py.Group, name: str, dtype: np.dtype = np.float32):
        """Initialize sparse data writer."""
        self.group = group
        self.name = name
        self.dtype = dtype
        
        # Buffers for sparse format
        self.timestep_buffer = []
        self.count_buffer = []
        self.value_buffer = []
        
        # Current run tracking
        self.current_timestep = None
        self.current_values = []
        
    def append(self, timestep: int, values: np.ndarray):
        """Append sparse data."""
        if self.current_timestep == timestep:
            # Same timestep, accumulate values
            self.current_values.extend(values)
        else:
            # New timestep, flush current
            self._flush_current()
            self.current_timestep = timestep
            self.current_values = list(values)
            
    def _flush_current(self):
        """Flush current timestep data."""
        if self.current_timestep is not None and self.current_values:
            self.timestep_buffer.append(self.current_timestep)
            self.count_buffer.append(len(self.current_values))
            self.value_buffer.extend(self.current_values)
            
    def finalize(self):
        """Write all data to HDF5."""
        self._flush_current()
        
        if not self.timestep_buffer:
            return
            
        # Create datasets with compression
        self.group.create_dataset(
            'timesteps',
            data=np.array(self.timestep_buffer, dtype=np.int64),
            compression='gzip',
            compression_opts=4
        )
        
        self.group.create_dataset(
            'counts',
            data=np.array(self.count_buffer, dtype=np.int32),
            compression='gzip',
            compression_opts=4
        )
        
        self.group.create_dataset(
            'values',
            data=np.array(self.value_buffer, dtype=self.dtype),
            compression='gzip',
            compression_opts=4
        )
        
        # Add metadata
        self.group.attrs['format'] = 'sparse_rle'
        self.group.attrs['total_events'] = len(self.value_buffer)
        self.group.attrs['unique_timesteps'] = len(self.timestep_buffer)


class OptimizedEpisode:
    """High-performance episode data storage."""
    
    def __init__(self,
                 episode_id: int,
                 h5_file: h5py.File,
                 neural_sampling_rate: int = 100,
                 validate_data: bool = True,
                 compression: str = 'gzip',
                 compression_opts: int = 4,
                 chunk_size: int = 10000):
        """Initialize optimized episode."""
        self.episode_id = episode_id
        self.h5_file = h5_file
        self.neural_sampling_rate = neural_sampling_rate
        self.validate_data = validate_data
        self.compression = compression
        self.compression_opts = compression_opts
        self.chunk_size = chunk_size
        
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
        
        # Buffered datasets
        self.datasets: Dict[str, BufferedDataset] = {}
        
        # Sparse writers
        self.spike_writer = SparseDataWriter(self.spike_group, 'spikes', dtype=np.int32)
        self.reward_writer = SparseDataWriter(self.reward_group, 'rewards', dtype=np.float32)
        
        # Weight change accumulator
        self.weight_changes = defaultdict(list)
        
        # Event buffers
        self.event_buffers = defaultdict(lambda: defaultdict(list))
        
    def log_timestep(self,
                     timestep: int,
                     neural_state: Optional[Dict[str, Any]] = None,
                     spikes: Optional[Any] = None,
                     behavior: Optional[Dict[str, Any]] = None,
                     reward: Optional[float] = None):
        """Log data for a single timestep with optimized storage."""
        # Validate if enabled
        if self.validate_data:
            warnings_list = validate_timestep_data(timestep, neural_state, spikes, behavior, reward)
            for w in warnings_list:
                warnings.warn(f"Validation: {w}")
                
        self.timestep_count += 1
        
        # Neural state (sampled)
        if neural_state is not None and timestep - self.last_neural_sample >= self.neural_sampling_rate:
            self._append_neural_state(timestep, neural_state)
            self.last_neural_sample = timestep
            
        # Spikes (sparse)
        if spikes is not None:
            spike_data = ensure_numpy(spikes)
            if spike_data.any():
                indices = np.where(spike_data)[0]
                self.spike_writer.append(timestep, indices)
                
        # Behavior (all timesteps)
        if behavior is not None:
            self._append_behavior(timestep, behavior)
            
        # Rewards (sparse)
        if reward is not None and reward != 0:
            self.reward_writer.append(timestep, [reward])
            
    def _append_neural_state(self, timestep: int, data: Dict[str, Any]):
        """Append neural state data with buffering."""
        # Initialize timestep dataset if needed
        if 'timesteps' not in self.datasets:
            self.datasets['timesteps'] = BufferedDataset(
                self.neural_group, 'timesteps', np.int64, (0,),
                self.chunk_size, compression=self.compression,
                compression_opts=self.compression_opts
            )
            
        self.datasets['timesteps'].append(ensure_numpy(timestep))
        
        # Append each field
        for key, value in data.items():
            value_np = ensure_numpy(value)
            
            if key not in self.datasets:
                # Create dataset on first occurrence
                if value_np.ndim == 0:
                    shape = (0,)
                else:
                    shape = (0,) + value_np.shape
                    
                self.datasets[key] = BufferedDataset(
                    self.neural_group, key, value_np.dtype, shape,
                    self.chunk_size, compression=self.compression,
                    compression_opts=self.compression_opts
                )
                
            self.datasets[key].append(value_np)
            
    def _append_behavior(self, timestep: int, data: Dict[str, Any]):
        """Append behavior data with buffering."""
        # Initialize behavior datasets
        beh_key = 'behavior'
        
        if f'{beh_key}_timesteps' not in self.datasets:
            self.datasets[f'{beh_key}_timesteps'] = BufferedDataset(
                self.behavior_group, 'timesteps', np.int64, (0,),
                self.chunk_size, compression=self.compression,
                compression_opts=self.compression_opts
            )
            
        self.datasets[f'{beh_key}_timesteps'].append(ensure_numpy(timestep))
        
        for key, value in data.items():
            value_np = ensure_numpy(value)
            full_key = f'{beh_key}_{key}'
            
            if full_key not in self.datasets:
                shape = (0,) if value_np.ndim == 0 else (0,) + value_np.shape
                self.datasets[full_key] = BufferedDataset(
                    self.behavior_group, key, value_np.dtype, shape,
                    self.chunk_size, compression=self.compression,
                    compression_opts=self.compression_opts
                )
                
            self.datasets[full_key].append(value_np)
            
    def log_weight_change(self,
                          timestep: int,
                          synapse_id: Union[int, Tuple[int, int]],
                          old_weight: float,
                          new_weight: float,
                          **kwargs):
        """Log weight change event with batching."""
        if self.validate_data:
            warnings_list = validate_weight_change(timestep, synapse_id, old_weight, new_weight)
            for w in warnings_list:
                warnings.warn(f"Validation: {w}")
                
        # Accumulate weight changes
        change_data = {
            'timestep': timestep,
            'old_weight': old_weight,
            'new_weight': new_weight,
            'delta': new_weight - old_weight
        }
        
        if isinstance(synapse_id, tuple):
            change_data['source_id'] = synapse_id[0]
            change_data['target_id'] = synapse_id[1]
        else:
            change_data['synapse_id'] = synapse_id
            
        change_data.update(kwargs)
        
        # Group by synapse for better compression
        key = synapse_id if isinstance(synapse_id, int) else f"{synapse_id[0]}_{synapse_id[1]}"
        self.weight_changes[key].append(change_data)
        
    def log_event(self, event_type: str, timestep: int, data: Dict[str, Any]):
        """Log custom event with buffering."""
        for key, value in data.items():
            self.event_buffers[event_type][key].append((timestep, value))
            
    def log_static_data(self, name: str, data: Dict[str, Any]):
        """Save static, one-off data for the episode."""
        group = self.group.create_group(name)
        for key, value in data.items():
            group.create_dataset(
                key, 
                data=ensure_numpy(value),
                compression=self.compression,
                compression_opts=self.compression_opts
            )

    def end(self, success: bool = False, final_state: Optional[Dict[str, Any]] = None):
        """End episode and flush all buffers."""
        # Flush all buffered datasets
        for dataset in self.datasets.values():
            dataset.finalize()
            
        # Finalize sparse writers
        self.spike_writer.finalize()
        self.reward_writer.finalize()
        
        # Write weight changes
        if self.weight_changes:
            self._write_weight_changes()
            
        # Write events
        if self.event_buffers:
            self._write_events()
            
        # Update attributes
        self.group.attrs['end_time'] = datetime.now().isoformat()
        self.group.attrs['total_timesteps'] = self.timestep_count
        self.group.attrs['success'] = success
        self.group.attrs['status'] = 'completed'
        
        # Calculate summary statistics
        if 'timesteps' in self.spike_group:
            total_spikes = len(self.spike_writer.value_buffer)
            self.group.attrs['total_spikes'] = total_spikes
            self.group.attrs['spike_rate'] = total_spikes / max(1, self.timestep_count)
            
        if 'timesteps' in self.reward_group:
            total_reward = float(np.sum(self.reward_writer.value_buffer))
            self.group.attrs['total_reward'] = total_reward
            
        self.group.attrs['total_weight_changes'] = sum(
            len(changes) for changes in self.weight_changes.values()
        )
        
        # Save final state
        if final_state:
            final_group = self.group.create_group('final_state')
            for key, value in final_state.items():
                final_group.create_dataset(
                    key,
                    data=ensure_numpy(value),
                    compression=self.compression,
                    compression_opts=self.compression_opts
                )
                
    def _write_weight_changes(self):
        """Write accumulated weight changes efficiently."""
        # Flatten all changes
        all_changes = []
        for synapse_changes in self.weight_changes.values():
            all_changes.extend(synapse_changes)
            
        if not all_changes:
            return
            
        # Sort by timestep for better compression
        all_changes.sort(key=lambda x: x['timestep'])
        
        # Extract arrays
        timesteps = np.array([c['timestep'] for c in all_changes], dtype=np.int64)
        old_weights = np.array([c['old_weight'] for c in all_changes], dtype=np.float32)
        new_weights = np.array([c['new_weight'] for c in all_changes], dtype=np.float32)
        deltas = np.array([c['delta'] for c in all_changes], dtype=np.float32)
        
        # Write datasets
        self.weight_group.create_dataset(
            'timesteps', data=timesteps,
            compression=self.compression, compression_opts=self.compression_opts
        )
        self.weight_group.create_dataset(
            'old_weights', data=old_weights,
            compression=self.compression, compression_opts=self.compression_opts
        )
        self.weight_group.create_dataset(
            'new_weights', data=new_weights,
            compression=self.compression, compression_opts=self.compression_opts
        )
        self.weight_group.create_dataset(
            'deltas', data=deltas,
            compression=self.compression, compression_opts=self.compression_opts
        )
        
        # Handle synapse IDs
        if 'synapse_id' in all_changes[0]:
            synapse_ids = np.array([c['synapse_id'] for c in all_changes], dtype=np.int32)
            self.weight_group.create_dataset(
                'synapse_ids', data=synapse_ids,
                compression=self.compression, compression_opts=self.compression_opts
            )
        else:
            source_ids = np.array([c['source_id'] for c in all_changes], dtype=np.int32)
            target_ids = np.array([c['target_id'] for c in all_changes], dtype=np.int32)
            self.weight_group.create_dataset(
                'source_ids', data=source_ids,
                compression=self.compression, compression_opts=self.compression_opts
            )
            self.weight_group.create_dataset(
                'target_ids', data=target_ids,
                compression=self.compression, compression_opts=self.compression_opts
            )
            
    def _write_events(self):
        """Write accumulated events efficiently."""
        for event_type, event_data in self.event_buffers.items():
            event_subgroup = self.event_group.create_group(event_type)
            
            for key, values in event_data.items():
                if not values:
                    continue
                    
                # Sort by timestep
                values.sort(key=lambda x: x[0])
                
                # Extract arrays
                timesteps = np.array([v[0] for v in values], dtype=np.int64)
                data_values = np.array([v[1] for v in values])
                
                # Write datasets
                event_subgroup.create_dataset(
                    f'{key}_timesteps', data=timesteps,
                    compression=self.compression, compression_opts=self.compression_opts
                )
                event_subgroup.create_dataset(
                    key, data=data_values,
                    compression=self.compression, compression_opts=self.compression_opts
                )


class OptimizedDataExporter:
    """High-performance HDF5 data exporter with advanced optimizations."""
    
    def __init__(self,
                 experiment_name: str,
                 output_base_dir: str = "experiments",
                 neural_sampling_rate: int = 100,
                 validate_data: bool = True,
                 compression: str = 'gzip',
                 compression_level: int = 4,
                 chunk_size: int = 10000,
                 enable_swmr: bool = False,
                 async_write: bool = False):
        """Initialize optimized exporter.
        
        Args:
            experiment_name: Name of the experiment
            output_base_dir: Base directory for output
            neural_sampling_rate: Sample neural state every N timesteps
            validate_data: Whether to validate data
            compression: HDF5 compression ('gzip', 'lzf', None)
            compression_level: Compression level (1-9 for gzip)
            chunk_size: Chunk size for datasets
            enable_swmr: Enable single-writer multiple-reader mode
            async_write: Enable asynchronous writing (experimental)
        """
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.neural_sampling_rate = neural_sampling_rate
        self.validate_data = validate_data
        self.compression = compression
        self.compression_level = compression_level
        self.chunk_size = chunk_size
        self.enable_swmr = enable_swmr
        self.async_write = async_write
        
        # Create output directory
        self.output_dir = Path(output_base_dir) / f"{experiment_name}_{self.timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create main HDF5 file with optimizations
        self.h5_path = self.output_dir / 'experiment_data.h5'
        
        # File creation flags
        if enable_swmr:
            self.h5_file = h5py.File(self.h5_path, 'w', libver='latest')
            self.h5_file.swmr_mode = True
        else:
            self.h5_file = h5py.File(self.h5_path, 'w')
            
        # Set file-level compression properties
        if compression == 'gzip':
            # Create property list for better compression
            self.h5_file.attrs['compression_info'] = f"{compression}:{compression_level}"
            
        # Set experiment metadata
        self.h5_file.attrs['experiment_name'] = experiment_name
        self.h5_file.attrs['timestamp'] = self.timestamp
        self.h5_file.attrs['start_time'] = datetime.now().isoformat()
        self.h5_file.attrs['schema_version'] = SCHEMA_VERSION
        self.h5_file.attrs['neural_sampling_rate'] = neural_sampling_rate
        self.h5_file.attrs['compression'] = compression or 'none'
        self.h5_file.attrs['chunk_size'] = chunk_size
        self.h5_file.attrs['optimized'] = True
        
        # Create groups
        self.episodes_group = self.h5_file.create_group('episodes')
        self.checkpoints_group = self.h5_file.create_group('checkpoints')
        
        # State
        self.current_episode: Optional[OptimizedEpisode] = None
        self.episode_count = 0
        
        # Async write queue
        if async_write:
            self.write_queue = Queue()
            self.write_thread = threading.Thread(target=self._async_writer)
            self.write_thread.daemon = True
            self.write_thread.start()
            
        print(f"Initialized OptimizedDataExporter: {self.output_dir}")
        print(f"  Compression: {compression}:{compression_level}")
        print(f"  Chunk size: {chunk_size}")
        print(f"  SWMR mode: {enable_swmr}")
        print(f"  Async write: {async_write}")
        
        # Automatically capture runtime info
        self.save_runtime_info()
        
    def save_config(self, config: Dict[str, Any]):
        """Save experiment configuration."""
        def _to_serializable(obj):
            if hasattr(obj, '_asdict'):
                return _to_serializable(obj._asdict())
            if isinstance(obj, dict):
                return {k: _to_serializable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_to_serializable(i) for i in obj]
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            # Use abstract base classes for NumPy 2.0 compatibility
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            return obj

        serializable_config = _to_serializable(config)

        # Save as JSON for readability
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(serializable_config, f, indent=2, cls=NumpyEncoder)
            
        # Also save in HDF5 with compression
        if 'config' in self.h5_file:
            del self.h5_file['config']
        config_group = self.h5_file.create_group('config')
        
        for key, value in serializable_config.items():
            if isinstance(value, (dict, list)):
                # Compress JSON strings
                json_str = json.dumps(value, cls=NumpyEncoder)
                compressed = zlib.compress(json_str.encode('utf-8'))
                config_group.attrs[key] = np.void(compressed)
                config_group.attrs[f'{key}_compressed'] = True
            else:
                try:
                    config_group.attrs[key] = value
                except TypeError:
                    config_group.attrs[key] = str(value)
                
    def save_metadata(self, metadata: Dict[str, Any]):
        """Save additional experiment metadata."""
        for key, value in metadata.items():
            if isinstance(value, (dict, list)):
                self.h5_file.attrs[f'meta_{key}'] = json.dumps(value, cls=NumpyEncoder)
            else:
                self.h5_file.attrs[f'meta_{key}'] = value
                
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2, cls=NumpyEncoder)
            
    def save_runtime_info(self):
        """Capture runtime environment info."""
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
            'output_directory': str(self.output_dir),
            'hdf5_driver': self.h5_file.driver,
            'hdf5_libver': str(self.h5_file.libver)
        }
        
        if 'runtime' in self.h5_file:
            del self.h5_file['runtime']
        runtime_group = self.h5_file.create_group('runtime')
        for key, value in runtime_info.items():
            if value is not None:
                runtime_group.attrs[key] = str(value)
                
        with open(self.output_dir / 'runtime_info.json', 'w') as f:
            json.dump(runtime_info, f, indent=2)
            
        return runtime_info
    
    def save_code_snapshot(self, code_files: Optional[List[Union[str, Path]]] = None):
        """Save snapshot of code files."""
        import sys
        
        code_group = self.h5_file.create_group('code_snapshot')
        
        if code_files is None:
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
                    
                    # Compress code
                    compressed = zlib.compress(code_content.encode('utf-8'))
                    code_group.attrs[file_path.name] = np.void(compressed)
                    code_group.attrs[f'{file_path.name}_compressed'] = True
                    saved_files.append(str(file_path))
                    
                    # Also save uncompressed
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
        """Save git repository information."""
        try:
            import subprocess
            
            git_info = {}
            
            # Get various git info
            commands = {
                'commit_hash': ['git', 'rev-parse', 'HEAD'],
                'branch': ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                'status': ['git', 'status', '--porcelain'],
                'remote_url': ['git', 'config', '--get', 'remote.origin.url']
            }
            
            for key, cmd in commands.items():
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())
                if result.returncode == 0:
                    output = result.stdout.strip()
                    if key == 'status':
                        git_info['dirty'] = len(output) > 0
                        if git_info['dirty']:
                            git_info['uncommitted_files'] = output.split('\n')
                    else:
                        git_info[key] = output
                        
            # Save to HDF5
            if git_info:
                git_group = self.h5_file.create_group('git_info')
                for key, value in git_info.items():
                    if isinstance(value, list):
                        git_group.attrs[key] = json.dumps(value)
                    else:
                        git_group.attrs[key] = str(value)
                        
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
        """Save network structure with optimization."""
        if self.validate_data:
            warnings_list = validate_network_structure(neurons, connections)
            for w in warnings_list:
                warnings.warn(f"Network validation: {w}")
                
        # Create network group
        net_group = self.h5_file.create_group('network_structure')
        
        # Save neuron data with compression
        neurons_group = net_group.create_group('neurons')
        for key, value in neurons.items():
            data = ensure_numpy(value)
            neurons_group.create_dataset(
                key, data=data,
                compression=self.compression,
                compression_opts=self.compression_level,
                chunks=True
            )
            
        # Save connection data
        conn_group = net_group.create_group('connections')
        for key, value in connections.items():
            data = ensure_numpy(value)
            conn_group.create_dataset(
                key, data=data,
                compression=self.compression,
                compression_opts=self.compression_level,
                chunks=True
            )
            
        # Save initial weights
        if initial_weights is not None:
            weights_group = net_group.create_group('initial_weights')
            
            if isinstance(initial_weights, dict):
                # Already sparse
                for key, value in initial_weights.items():
                    weights_group.create_dataset(
                        key, data=ensure_numpy(value),
                        compression=self.compression,
                        compression_opts=self.compression_level
                    )
                weights_group.attrs['format'] = 'sparse'
            else:
                # Convert dense to sparse
                weights_np = ensure_numpy(initial_weights)
                if weights_np.ndim == 2:
                    # Use CSR-like format
                    nonzero_mask = weights_np != 0
                    row_indices, col_indices = np.where(nonzero_mask)
                    values = weights_np[nonzero_mask]
                    
                    # Sort by row for better compression
                    sort_idx = np.lexsort((col_indices, row_indices))
                    row_indices = row_indices[sort_idx]
                    col_indices = col_indices[sort_idx]
                    values = values[sort_idx]
                    
                    weights_group.create_dataset(
                        'row_indices', data=row_indices,
                        compression=self.compression,
                        compression_opts=self.compression_level
                    )
                    weights_group.create_dataset(
                        'col_indices', data=col_indices,
                        compression=self.compression,
                        compression_opts=self.compression_level
                    )
                    weights_group.create_dataset(
                        'values', data=values,
                        compression=self.compression,
                        compression_opts=self.compression_level
                    )
                    weights_group.attrs['format'] = 'csr'
                    weights_group.attrs['shape'] = weights_np.shape
                    weights_group.attrs['nnz'] = len(values)
                    weights_group.attrs['sparsity'] = 1.0 - (len(values) / weights_np.size)
                else:
                    # 1D weights
                    weights_group.create_dataset(
                        'weights', data=weights_np,
                        compression=self.compression,
                        compression_opts=self.compression_level
                    )
                    weights_group.attrs['format'] = 'dense_1d'
                    
        # Update metadata
        self.h5_file.attrs['n_neurons'] = len(neurons.get('neuron_ids', []))
        self.h5_file.attrs['n_connections'] = len(connections.get('source_ids', []))
        
    def start_episode(self, episode_id: Optional[int] = None) -> OptimizedEpisode:
        """Start new episode."""
        if self.current_episode is not None:
            warnings.warn(f"Previous episode {self.current_episode.episode_id} not ended")
            self.current_episode.end(success=False)
            
        if episode_id is None:
            episode_id = self.episode_count
            
        self.current_episode = OptimizedEpisode(
            episode_id=episode_id,
            h5_file=self.episodes_group,
            neural_sampling_rate=self.neural_sampling_rate,
            validate_data=self.validate_data,
            compression=self.compression,
            compression_opts=self.compression_level,
            chunk_size=self.chunk_size
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
        
        # Copy episode attributes
        for key, value in self.current_episode.group.attrs.items():
            ep_summary.attrs[key] = value
            
        # Add custom summary
        if summary:
            for key, value in summary.items():
                if isinstance(value, (dict, list)):
                    ep_summary.attrs[key] = json.dumps(value, cls=NumpyEncoder)
                else:
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
            
    def log_static_episode_data(self, name: str, data: Dict[str, Any]):
        """Save static, one-off data for the current episode."""
        if self.current_episode is None:
            raise RuntimeError("No active episode. Call start_episode() first.")
        self.current_episode.log_static_data(name, data)

    def save_checkpoint(self, name: str, data: Dict[str, Any]):
        """Save checkpoint with compression."""
        checkpoint = self.checkpoints_group.create_group(
            f"{name}_{datetime.now().strftime('%H%M%S')}"
        )
        checkpoint.attrs['timestamp'] = datetime.now().isoformat()
        checkpoint.attrs['episode_count'] = self.episode_count
        
        for key, value in data.items():
            checkpoint.create_dataset(
                key, data=ensure_numpy(value),
                compression=self.compression,
                compression_opts=self.compression_level
            )
            
    def _async_writer(self):
        """Background thread for async writes."""
        while True:
            try:
                task = self.write_queue.get()
                if task is None:
                    break
                task()
            except Exception as e:
                warnings.warn(f"Async write error: {e}")
                
    def close(self):
        """Close HDF5 file and save final metadata."""
        # End any active episode
        if self.current_episode is not None:
            self.current_episode.end(success=False)
            
        # Stop async writer if enabled
        if self.async_write:
            self.write_queue.put(None)
            self.write_thread.join()
            
        # Update final metadata
        self.h5_file.attrs['end_time'] = datetime.now().isoformat()
        self.h5_file.attrs['total_episodes'] = self.episode_count
        
        # Calculate file statistics
        file_size = self.h5_path.stat().st_size
        self.h5_file.attrs['file_size_bytes'] = file_size
        self.h5_file.attrs['file_size_mb'] = file_size / (1024 * 1024)
        
        # Close file
        self.h5_file.close()
        
        print(f"Experiment complete. Data saved to: {self.output_dir}")
        print(f"  File size: {file_size / (1024 * 1024):.2f} MB")
        
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