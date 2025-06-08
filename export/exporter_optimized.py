# keywords: [exporter, hdf5, optimized, high-performance, streaming, chunked, compressed]
"""High-performance HDF5 data exporter with advanced optimizations."""

import h5py
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Union, Tuple, List
import warnings
from collections import defaultdict
import threading
import zlib

from .utils import ensure_numpy, NumpyEncoder, optimize_jax_conversion
from .schema import validate_timestep_data, validate_weight_change, validate_network_structure, SCHEMA_VERSION
from .performance_enhancements import AsyncWriteQueue, AdaptiveCompressor, RealtimeStats, PerformanceProfiler


# Global HDF5 lock for thread safety
_HDF5_LOCK = threading.RLock()

# Constants to replace magic numbers
ASYNC_THRESHOLD = 100  # Minimum items for async write
DEFAULT_BUFFER_THRESHOLD = 1000  # Default buffer size before flush
MAX_CHUNK_SIZE = 100000  # Maximum chunk size for HDF5
MEMORY_MAP_BLOCK_SIZE = 2**20  # 1MB blocks for memory mapping

# Memory management
MAX_MEMORY_MB = 500  # Maximum memory usage in MB
MEMORY_CHECK_INTERVAL = 100  # Check memory every N operations


class MemoryTracker:
    """Track memory usage and trigger flushes when needed."""

    def __init__(self, max_memory_bytes: int = MAX_MEMORY_MB * 1024 * 1024):
        self.max_memory = max_memory_bytes
        self.current_usage = 0
        self._lock = threading.Lock()
        self._check_counter = 0

    def add(self, nbytes: int):
        """Add memory usage."""
        with self._lock:
            self.current_usage += nbytes

    def remove(self, nbytes: int):
        """Remove memory usage."""
        with self._lock:
            self.current_usage = max(0, self.current_usage - nbytes)

    def should_flush(self) -> bool:
        """Check if memory pressure requires flush."""
        with self._lock:
            self._check_counter += 1
            if self._check_counter % MEMORY_CHECK_INTERVAL == 0:
                return self.current_usage > self.max_memory * 0.8  # 80% threshold
            return self.current_usage > self.max_memory

    def get_usage_mb(self) -> float:
        """Get current usage in MB."""
        with self._lock:
            return self.current_usage / (1024 * 1024)


class BufferedDataset:
    """Efficiently buffered dataset wrapper with pre-allocated buffers and thread safety."""

    def __init__(self,
                 group: h5py.Group,
                 name: str,
                 dtype: np.dtype,
                 shape: Tuple[int, ...],
                 chunk_size: int = 10000,
                 growth_factor: float = 1.5,
                 compression: str = 'gzip',
                 compression_opts: int = 4,
                 async_queue: Optional[AsyncWriteQueue] = None,
                 memory_tracker: Optional[MemoryTracker] = None):
        """Initialize buffered dataset with pre-allocated buffer."""
        self.name = name
        self.dtype = dtype
        self.chunk_size = chunk_size
        self.growth_factor = growth_factor
        self._lock = threading.Lock()
        self.async_queue = async_queue
        self.memory_tracker = memory_tracker

        # Determine chunk shape and buffer size
        if len(shape) == 1:
            # Ensure chunk size is at least 1 to avoid empty chunks
            chunk_dim = max(1, min(chunk_size, DEFAULT_BUFFER_THRESHOLD))
            chunks = (chunk_dim,)
            self.item_shape = ()
        else:
            chunk_dim = max(1, min(chunk_size, DEFAULT_BUFFER_THRESHOLD))
            chunks = (chunk_dim,) + shape[1:]
            self.item_shape = shape[1:]

        # Buffer configuration
        self.buffer_threshold = max(
            1, min(chunk_size // 10, DEFAULT_BUFFER_THRESHOLD))

        # Pre-allocate buffer for efficiency
        buffer_shape = (self.buffer_threshold,) + self.item_shape
        self.buffer = np.empty(buffer_shape, dtype=dtype)
        self.buffer_pos = 0

        # Track memory usage
        self.buffer_memory = self.buffer.nbytes
        if self.memory_tracker:
            self.memory_tracker.add(self.buffer_memory)

        # Create dataset with compression (thread-safe)
        maxshape = (None,) + shape[1:] if len(shape) > 1 else (None,)

        with _HDF5_LOCK:
            # Create dataset with appropriate options
            create_kwargs = {
                'name': name,
                'shape': shape,
                'maxshape': maxshape,
                'dtype': dtype
            }

            # Only add chunks if shape is not empty
            # Empty datasets (shape with 0) cannot have chunks
            if all(s > 0 for s in shape):
                create_kwargs['chunks'] = chunks

            # Only add compression-related args if compression is enabled and shape is not empty
            if compression is not None and all(s > 0 for s in shape):
                create_kwargs['compression'] = compression
                create_kwargs['shuffle'] = True
                create_kwargs['fletcher32'] = True

                # Only add compression_opts for algorithms that support it
                if compression != 'lzf':
                    create_kwargs['compression_opts'] = compression_opts

            self.dataset = group.create_dataset(**create_kwargs)

        # Track dataset size
        self.current_size = 0
        self.allocated_size = 0

    def append(self, data: np.ndarray):
        """Append data to pre-allocated buffer (thread-safe)."""
        with self._lock:
            # Store in pre-allocated buffer
            self.buffer[self.buffer_pos] = data
            self.buffer_pos += 1

            # Check flush conditions
            should_flush = self.buffer_pos >= self.buffer_threshold
            if not should_flush and self.memory_tracker:
                should_flush = self.memory_tracker.should_flush()

            if should_flush:
                self._flush_unlocked()

    def flush(self) -> None:
        """Flush buffer to dataset (thread-safe)."""
        with self._lock:
            self._flush_unlocked()

    def _flush_unlocked(self) -> None:
        """Internal flush without locking (must be called with lock held)."""
        if self.buffer_pos == 0:
            return

        # Get data from pre-allocated buffer (no copy needed!)
        new_data = self.buffer[:self.buffer_pos]
        n_new = self.buffer_pos

        # Check if we should use async write
        if self.async_queue and n_new > ASYNC_THRESHOLD:
            # Submit async write - make a copy for async operation
            data_copy = new_data.copy()
            start_idx = self.current_size
            self.current_size += n_new

            # Reset buffer position
            self.buffer_pos = 0

            # Submit async write
            self.async_queue.submit(
                self._do_async_write,
                data_copy,
                start_idx
            )
        else:
            # Small buffer or no async - write synchronously
            self._do_sync_write(new_data)
            self.buffer_pos = 0

    def _do_sync_write(self, new_data: np.ndarray) -> None:
        """Perform synchronous write with HDF5 thread safety."""
        n_new = len(new_data)
        new_size = self.current_size + n_new

        with _HDF5_LOCK:
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

    def _do_async_write(self, data: np.ndarray, start_idx: int) -> None:
        """Perform async write with proper thread safety."""
        n_new = len(data)
        new_size = start_idx + n_new

        # Use HDF5 lock for all HDF5 operations
        with _HDF5_LOCK:
            # Check and resize if needed
            if new_size > self.allocated_size:
                with self._lock:
                    # Double-check under object lock
                    if new_size > self.allocated_size:
                        self.allocated_size = int(max(
                            new_size * self.growth_factor,
                            self.allocated_size + self.chunk_size
                        ))
                        self.dataset.resize(self.allocated_size, axis=0)

            # Write data
            self.dataset[start_idx:new_size] = data

    def finalize(self):
        """Flush remaining data and trim to actual size."""
        self.flush()
        if self.current_size < self.allocated_size:
            with _HDF5_LOCK:
                self.dataset.resize(self.current_size, axis=0)

    def __del__(self):
        """Clean up memory tracking."""
        if hasattr(self, 'memory_tracker') and self.memory_tracker and hasattr(self, 'buffer_memory'):
            self.memory_tracker.remove(self.buffer_memory)


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

        # Check if datasets already exist (in case of multiple finalizations)
        if 'timesteps' in self.group:
            return

        # Create datasets with compression (thread-safe)
        with _HDF5_LOCK:
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


class EpisodeDataManager:
    """Manages data storage for an episode."""

    def __init__(self, group: h5py.Group, config: dict, memory_tracker: MemoryTracker):
        self.group = group
        self.config = config
        self.memory_tracker = memory_tracker
        self.datasets: Dict[str, BufferedDataset] = {}

    def get_or_create_dataset(self, subgroup: h5py.Group, name: str,
                              dtype: np.dtype, shape: Tuple[int, ...]) -> BufferedDataset:
        """Get or create a buffered dataset."""
        key = f"{subgroup.name}/{name}"
        if key not in self.datasets:
            self.datasets[key] = BufferedDataset(
                subgroup, name, dtype, shape,
                chunk_size=self.config.get('chunk_size', 10000),
                compression=self.config.get('compression', 'gzip'),
                compression_opts=self.config.get('compression_opts', 4),
                async_queue=self.config.get('async_queue'),
                memory_tracker=self.memory_tracker
            )
        return self.datasets[key]

    def finalize_all(self):
        """Finalize all datasets."""
        for dataset in self.datasets.values():
            dataset.finalize()


class EpisodeStatsTracker:
    """Tracks statistics for an episode."""

    def __init__(self):
        self.timestep_count = 0
        self.spike_count = 0
        self.reward_sum = 0.0
        self.weight_change_count = 0

    def update_timestep(self):
        self.timestep_count += 1

    def update_spikes(self, count: int):
        self.spike_count += count

    def update_reward(self, reward: float):
        self.reward_sum += reward

    def update_weight_changes(self, count: int):
        self.weight_change_count += count

    def get_summary(self) -> Dict[str, Any]:
        return {
            'total_timesteps': self.timestep_count,
            'total_spikes': self.spike_count,
            'total_reward': self.reward_sum,
            'spike_rate': self.spike_count / max(1, self.timestep_count),
            'total_weight_changes': self.weight_change_count
        }


class OptimizedEpisode:
    """High-performance episode data storage."""

    # Memory limits to prevent unbounded growth
    MAX_WEIGHT_CHANGES = 100000
    MAX_EVENT_BUFFER_SIZE = 50000

    def __init__(self,
                 episode_id: int,
                 h5_file: h5py.File,
                 neural_sampling_rate: int = 100,
                 validate_data: bool = True,
                 compression: str = 'gzip',
                 compression_opts: int = 4,
                 chunk_size: int = 10000,
                 adaptive_compressor: Optional[AdaptiveCompressor] = None,
                 async_queue: Optional[AsyncWriteQueue] = None,
                 realtime_stats: Optional[RealtimeStats] = None,
                 profiler: Optional[PerformanceProfiler] = None,
                 memory_tracker: Optional[MemoryTracker] = None):
        """Initialize optimized episode."""
        self.episode_id = episode_id
        self.neural_sampling_rate = neural_sampling_rate
        self.validate_data = validate_data
        self.adaptive_compressor = adaptive_compressor
        self.realtime_stats = realtime_stats
        self.profiler = profiler

        # Create episode group
        with _HDF5_LOCK:
            self.group = h5_file.create_group(f'episode_{episode_id:04d}')

        # Configuration for datasets
        self.config = {
            'compression': compression,
            'compression_opts': compression_opts,
            'chunk_size': chunk_size,
            'async_queue': async_queue
        }

        # Initialize components
        self.memory_tracker = memory_tracker or MemoryTracker()
        self.data_manager = EpisodeDataManager(
            self.group, self.config, self.memory_tracker)
        self.stats_tracker = EpisodeStatsTracker()

        # Metadata
        self.group.attrs['episode_id'] = episode_id
        self.group.attrs['start_time'] = datetime.now().isoformat()
        self.group.attrs['neural_sampling_rate'] = neural_sampling_rate
        self.group.attrs['status'] = 'running'

        # Create subgroups (thread-safe)
        with _HDF5_LOCK:
            self.neural_group = self.group.create_group('neural_states')
            self.spike_group = self.group.create_group('spikes')
            self.behavior_group = self.group.create_group('behavior')
            self.reward_group = self.group.create_group('rewards')
            self.weight_group = self.group.create_group('weight_changes')
            self.event_group = self.group.create_group('events')

        # Tracking
        self.last_neural_sample = -neural_sampling_rate

        # Sparse writers
        self.spike_writer = SparseDataWriter(
            self.spike_group, 'spikes', dtype=np.int32)
        self.reward_writer = SparseDataWriter(
            self.reward_group, 'rewards', dtype=np.float32)

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
            warnings_list = validate_timestep_data(
                timestep, neural_state, spikes, behavior, reward)
            for w in warnings_list:
                warnings.warn(f"Validation: {w}")

        self.stats_tracker.update_timestep()

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
                self.stats_tracker.update_spikes(len(indices))

        # Behavior (all timesteps)
        if behavior is not None:
            self._append_behavior(timestep, behavior)

        # Rewards (sparse)
        if reward is not None and reward != 0:
            self.reward_writer.append(
                timestep, np.array([reward], dtype=np.float32))
            self.stats_tracker.update_reward(reward)

    def _append_neural_state(self, timestep: int, data: Dict[str, Any]):
        """Append neural state data with buffering."""
        # Profile this operation
        profile_ctx = self.profiler.profile(
            'append_neural_state') if self.profiler else None
        if profile_ctx:
            profile_ctx.__enter__()

        # Optimize JAX array conversion
        data = optimize_jax_conversion(data)

        # Initialize timestep dataset if needed
        timestep_writer = self.data_manager.get_or_create_dataset(
            self.neural_group, 'timesteps', np.int64, (0,)
        )
        timestep_writer.append(np.array(timestep, dtype=np.int64))

        # Append each field
        for key, value in data.items():
            value_np = value if isinstance(
                value, np.ndarray) else ensure_numpy(value)

            # Determine shape
            if value_np.ndim == 0:
                shape = (0,)
            else:
                shape = (0,) + value_np.shape

            # Use adaptive compression if available
            if self.adaptive_compressor:
                comp_algo, comp_level = self.adaptive_compressor.select_compression(
                    value_np)
                # Note: We'd need to pass these to the dataset creation, but HDF5 doesn't support per-dataset compression easily
                # So we'll just track the recommendation for now

            # Get or create dataset
            writer = self.data_manager.get_or_create_dataset(
                self.neural_group, key, value_np.dtype, shape
            )
            writer.append(value_np)

        # Update realtime stats
        if self.realtime_stats:
            data_size = sum(v.nbytes if hasattr(v, 'nbytes') else 0
                            for v in data.values())
            self.realtime_stats.record_operation(
                0.001, data_size)  # Assume 1ms write time

        if profile_ctx:
            profile_ctx.__exit__(None, None, None)

    def _append_behavior(self, timestep: int, data: Dict[str, Any]):
        """Append behavior data with buffering."""
        # Append timestep
        timestep_writer = self.data_manager.get_or_create_dataset(
            self.behavior_group, 'timesteps', np.int64, (0,)
        )
        timestep_writer.append(np.array(timestep, dtype=np.int64))

        # Append behavior data
        for key, value in data.items():
            value_np = ensure_numpy(value)
            shape = (0,) if value_np.ndim == 0 else (0,) + value_np.shape

            writer = self.data_manager.get_or_create_dataset(
                self.behavior_group, key, value_np.dtype, shape
            )
            writer.append(value_np)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - end episode with error handling."""
        try:
            if exc_type is None:
                self.end(success=True)
            else:
                # Log the error before ending
                import traceback
                error_msg = f"{exc_type.__name__}: {exc_val}"
                with _HDF5_LOCK:
                    self.group.attrs['error'] = error_msg
                    self.group.attrs['traceback'] = traceback.format_exc()
                self.end(success=False)
        except Exception as e:
            # Even if end() fails, we should not suppress the original exception
            warnings.warn(f"Failed to properly end episode: {e}")
        return False  # Don't suppress exceptions

    def log_weight_change(self,
                          timestep: int,
                          synapse_id: Union[int, Tuple[int, int]],
                          old_weight: float,
                          new_weight: float,
                          **kwargs):
        """Log weight change event with batching."""
        if self.validate_data:
            warnings_list = validate_weight_change(
                timestep, synapse_id, old_weight, new_weight)
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
        key = synapse_id if isinstance(
            synapse_id, int) else f"{synapse_id[0]}_{synapse_id[1]}"
        self.weight_changes[key].append(change_data)

        # Check memory limit
        total_changes = sum(len(changes)
                            for changes in self.weight_changes.values())
        if total_changes >= self.MAX_WEIGHT_CHANGES:
            self._write_weight_changes()
            self.weight_changes.clear()

        # Update stats
        self.stats_tracker.update_weight_changes(1)

    def log_event(self, event_type: str, timestep: int, data: Dict[str, Any]):
        """Log custom event with buffering."""
        for key, value in data.items():
            self.event_buffers[event_type][key].append((timestep, value))

        # Check memory limit
        total_events = sum(
            len(values) for event_data in self.event_buffers.values()
            for values in event_data.values()
        )
        if total_events >= self.MAX_EVENT_BUFFER_SIZE:
            self._write_events()
            self.event_buffers.clear()

    def log_static_data(self, name: str, data: Dict[str, Any]):
        """Save static, one-off data for the episode."""
        with _HDF5_LOCK:
            group = self.group.create_group(name)

            def save_item(parent_group, key: str, value: Any):
                """Recursively save items, handling nested structures."""
                if isinstance(value, dict):
                    # Create subgroup for nested dict
                    subgroup = parent_group.create_group(key)
                    for k, v in value.items():
                        save_item(subgroup, k, v)
                elif isinstance(value, (list, tuple)):
                    # Convert to numpy array
                    try:
                        numpy_value = ensure_numpy(value)
                        save_array(parent_group, key, numpy_value)
                    except:
                        # If can't convert to array, save as JSON string
                        parent_group.attrs[key] = json.dumps(value)
                else:
                    # Try to save as numpy array
                    try:
                        numpy_value = ensure_numpy(value)
                        save_array(parent_group, key, numpy_value)
                    except:
                        # Fall back to attribute for non-array data
                        parent_group.attrs[key] = value

            def save_array(parent_group, key: str, numpy_value: np.ndarray):
                """Save numpy array with appropriate compression."""
                # Scalar datasets don't support compression
                if numpy_value.ndim == 0 or numpy_value.size == 1:
                    parent_group.create_dataset(key, data=numpy_value)
                else:
                    # Use compression settings from config
                    compression = self.config['compression']
                    compression_opts = self.config['compression_opts']

                    # Handle LZF compression which doesn't support compression_opts
                    if compression == 'lzf':
                        parent_group.create_dataset(
                            key,
                            data=numpy_value,
                            compression=compression
                        )
                    else:
                        parent_group.create_dataset(
                            key,
                            data=numpy_value,
                            compression=compression,
                            compression_opts=compression_opts
                        )

            # Save each item
            for key, value in data.items():
                save_item(group, key, value)

    def end(self, success: bool = False, final_state: Optional[Dict[str, Any]] = None):
        """End episode and flush all buffers."""
        # Check if episode has any data
        if self.stats_tracker.timestep_count == 0:
            self.group.attrs['status'] = 'empty'
            self.group.attrs['warning'] = 'Episode ended with no data logged'
            return

        # Flush all buffered datasets
        self.data_manager.finalize_all()

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
        self.group.attrs['success'] = success
        self.group.attrs['status'] = 'completed'

        # Add statistics from tracker
        stats = self.stats_tracker.get_summary()
        for key, value in stats.items():
            self.group.attrs[key] = value

        # Save final state
        if final_state:
            with _HDF5_LOCK:
                final_group = self.group.create_group('final_state')
                for key, value in final_state.items():
                    final_group.create_dataset(
                        key,
                        data=ensure_numpy(value),
                        compression=self.config['compression'],
                        compression_opts=self.config['compression_opts']
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
        timesteps = np.array([c['timestep']
                             for c in all_changes], dtype=np.int64)
        old_weights = np.array([c['old_weight']
                               for c in all_changes], dtype=np.float32)
        new_weights = np.array([c['new_weight']
                               for c in all_changes], dtype=np.float32)
        deltas = np.array([c['delta'] for c in all_changes], dtype=np.float32)

        # Write datasets (thread-safe)
        with _HDF5_LOCK:
            # Check if datasets already exist
            if 'timesteps' in self.weight_group:
                return  # Already written

            self.weight_group.create_dataset(
                'timesteps', data=timesteps,
                compression=self.config['compression'],
                compression_opts=self.config['compression_opts']
            )
            self.weight_group.create_dataset(
                'old_weights', data=old_weights,
                compression=self.config['compression'],
                compression_opts=self.config['compression_opts']
            )
            self.weight_group.create_dataset(
                'new_weights', data=new_weights,
                compression=self.config['compression'],
                compression_opts=self.config['compression_opts']
            )
            self.weight_group.create_dataset(
                'deltas', data=deltas,
                compression=self.config['compression'],
                compression_opts=self.config['compression_opts']
            )

            # Handle synapse IDs
            if 'synapse_id' in all_changes[0]:
                synapse_ids = np.array([c['synapse_id']
                                       for c in all_changes], dtype=np.int32)
                self.weight_group.create_dataset(
                    'synapse_ids', data=synapse_ids,
                    compression=self.config['compression'],
                    compression_opts=self.config['compression_opts']
                )
            else:
                source_ids = np.array([c['source_id']
                                      for c in all_changes], dtype=np.int32)
                target_ids = np.array([c['target_id']
                                      for c in all_changes], dtype=np.int32)
                self.weight_group.create_dataset(
                    'source_ids', data=source_ids,
                    compression=self.config['compression'],
                    compression_opts=self.config['compression_opts']
                )
                self.weight_group.create_dataset(
                    'target_ids', data=target_ids,
                    compression=self.config['compression'],
                    compression_opts=self.config['compression_opts']
                )

    def _write_events(self):
        """Write accumulated events efficiently."""
        with _HDF5_LOCK:
            for event_type, event_data in self.event_buffers.items():
                # Check if event type already exists
                if event_type in self.event_group:
                    continue

                event_subgroup = self.event_group.create_group(event_type)

                for key, values in event_data.items():
                    if not values:
                        continue

                    # Sort by timestep
                    values.sort(key=lambda x: x[0])

                    # Extract arrays
                    timesteps = np.array([v[0]
                                         for v in values], dtype=np.int64)
                    data_values = np.array([v[1] for v in values])

                    # Write datasets
                    event_subgroup.create_dataset(
                        f'{key}_timesteps', data=timesteps,
                        compression=self.config['compression'],
                        compression_opts=self.config['compression_opts']
                    )
                    event_subgroup.create_dataset(
                        key, data=data_values,
                        compression=self.config['compression'],
                        compression_opts=self.config['compression_opts']
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
                 async_write: bool = False,
                 enable_mmap: bool = False,
                 adaptive_compression: bool = False,
                 enable_profiling: bool = False,
                 n_async_workers: int = 4,
                 no_write: bool = False):
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
            async_write: Enable asynchronous writing
            enable_mmap: Enable memory-mapped mode for large experiments
            adaptive_compression: Enable adaptive compression based on data entropy
            enable_profiling: Enable performance profiling
            n_async_workers: Number of async write workers
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
        self.enable_mmap = enable_mmap
        self.adaptive_compression = adaptive_compression
        self.enable_profiling = enable_profiling

        # Initialize performance components
        self.adaptive_compressor = AdaptiveCompressor() if adaptive_compression else None
        self.realtime_stats = RealtimeStats() if async_write or enable_profiling else None
        self.profiler = PerformanceProfiler() if enable_profiling else None

        # Create output directory
        self.output_dir = Path(output_base_dir) / \
            f"{experiment_name}_{self.timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create main HDF5 file with optimizations
        self.h5_path = self.output_dir / 'experiment_data.h5'

        # File creation with error handling
        try:
            if enable_mmap:
                # Memory-mapped mode for large experiments
                self.h5_file = h5py.File(self.h5_path, 'w', driver='core',
                                         backing_store=True, block_size=MEMORY_MAP_BLOCK_SIZE)
            elif enable_swmr:
                self.h5_file = h5py.File(self.h5_path, 'w', libver='latest')
                self.h5_file.swmr_mode = True
            else:
                self.h5_file = h5py.File(self.h5_path, 'w')
        except Exception as e:
            raise RuntimeError(
                f"Failed to create HDF5 file at {self.h5_path}: {e}")

        # Set file-level compression properties
        if compression == 'gzip':
            # Create property list for better compression
            self.h5_file.attrs['compression_info'] = f"{compression}:{compression_level}"

        # Set experiment metadata (thread-safe)
        with _HDF5_LOCK:
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
            self.async_queue = AsyncWriteQueue(n_workers=n_async_workers)
        else:
            self.async_queue = None

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
                self.h5_file.attrs[f'meta_{key}'] = json.dumps(
                    value, cls=NumpyEncoder)
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
        except ImportError:
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
                result = subprocess.run(
                    cmd, capture_output=True, text=True, cwd=Path.cwd())
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
                    weights_group.attrs['sparsity'] = 1.0 - \
                        (len(values) / weights_np.size)
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
        self.h5_file.attrs['n_connections'] = len(
            connections.get('source_ids', []))

    def start_episode(self, episode_id: Optional[int] = None) -> OptimizedEpisode:
        """Start new episode."""
        if self.current_episode is not None:
            warnings.warn(
                f"Previous episode {self.current_episode.episode_id} not ended")
            self.current_episode.end(success=False)

        if episode_id is None:
            episode_id = self.episode_count

        # Create memory tracker if not exists
        if not hasattr(self, 'memory_tracker'):
            self.memory_tracker = MemoryTracker()

        self.current_episode = OptimizedEpisode(
            episode_id=episode_id,
            h5_file=self.episodes_group,
            neural_sampling_rate=self.neural_sampling_rate,
            validate_data=self.validate_data,
            compression=self.compression,
            compression_opts=self.compression_level,
            chunk_size=self.chunk_size,
            adaptive_compressor=self.adaptive_compressor,
            async_queue=self.async_queue,
            realtime_stats=self.realtime_stats,
            profiler=self.profiler,
            memory_tracker=self.memory_tracker
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
        ep_summary = summary_group.create_group(
            f'episode_{self.current_episode.episode_id:04d}')

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
            raise RuntimeError(
                "No active episode. Call start_episode() first.")

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
            raise RuntimeError(
                "No active episode. Call start_episode() first.")
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

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats = {}

        if self.realtime_stats:
            stats['realtime'] = self.realtime_stats.get_summary()

        if self.async_queue:
            stats['async_io'] = self.async_queue.get_stats()

        if self.profiler:
            stats['profiling'] = self.profiler.get_report()

        if self.adaptive_compressor:
            comp_stats = {}
            for name, data in self.adaptive_compressor.compression_stats.items():
                if data['ratios']:
                    comp_stats[name] = {
                        'avg_ratio': np.mean(data['ratios']),
                        'avg_time': np.mean(data['times'])
                    }
            stats['compression'] = comp_stats

        return stats

    def close(self):
        """Close HDF5 file and save final metadata with error handling."""
        try:
            # End any active episode
            if self.current_episode is not None:
                try:
                    self.current_episode.end(success=False)
                except Exception as e:
                    warnings.warn(f"Failed to end episode during close: {e}")

            # Stop async writer if enabled
            if self.async_queue:
                try:
                    self.async_queue.flush()
                    self.async_queue.shutdown()
                except Exception as e:
                    warnings.warn(f"Failed to shutdown async queue: {e}")

            # Save performance statistics
            if self.enable_profiling or self.async_write:
                perf_stats = self.get_performance_stats()
                if perf_stats:
                    try:
                        with open(self.output_dir / 'performance_stats.json', 'w') as f:
                            json.dump(perf_stats, f, indent=2,
                                      cls=NumpyEncoder)

                        # Also save to HDF5
                        with _HDF5_LOCK:
                            perf_group = self.h5_file.create_group(
                                'performance_stats')
                            perf_group.attrs['stats'] = json.dumps(
                                perf_stats, cls=NumpyEncoder)
                    except Exception as e:
                        warnings.warn(f"Failed to save performance stats: {e}")

            # Update final metadata
            try:
                with _HDF5_LOCK:
                    self.h5_file.attrs['end_time'] = datetime.now().isoformat()
                    self.h5_file.attrs['total_episodes'] = self.episode_count

                    # Calculate file statistics
                    file_size = self.h5_path.stat().st_size
                    self.h5_file.attrs['file_size_bytes'] = file_size
                    self.h5_file.attrs['file_size_mb'] = file_size / \
                        (1024 * 1024)
            except Exception as e:
                warnings.warn(f"Failed to update final metadata: {e}")

            # Print summary
            print(f"Experiment complete. Data saved to: {self.output_dir}")
            if hasattr(self, 'h5_path'):
                try:
                    file_size = self.h5_path.stat().st_size / (1024 * 1024)
                    print(f"  File size: {file_size:.2f} MB")
                except:
                    pass

        finally:
            # Always try to close the file
            if hasattr(self, 'h5_file') and self.h5_file:
                try:
                    with _HDF5_LOCK:
                        self.h5_file.close()
                except Exception as e:
                    warnings.warn(f"Failed to close HDF5 file: {e}")

        # Performance summary if profiling enabled
        if self.enable_profiling and self.profiler:
            self.profiler.print_report()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        if hasattr(self, 'h5_file') and self.h5_file:
            try:
                self.h5_file.close()
            except Exception:
                pass
