# keywords: [episode, hdf5, buffered dataset, performance]
"""Manages data storage for a single episode with high performance."""

import threading
import jax.numpy as jnp
import warnings
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import h5py
import numpy as np

from .schema import validate_timestep_data
from .utils import ensure_numpy

# Global HDF5 lock for thread safety across all exporter instances
_HDF5_LOCK = threading.RLock()

# Constants
DEFAULT_BUFFER_THRESHOLD = 1000


class BufferedDataset:
    """An efficiently buffered, thread-safe HDF5 dataset writer."""

    def __init__(
        self,
        group: h5py.Group,
        name: str,
        dtype: np.dtype,
        shape: Tuple[int, ...],
        chunk_size: int,
        compression: Optional[str],
        compression_opts: int,
        flush_at_episode_end: bool = False,
    ):
        self.name = name
        self.chunk_size = chunk_size
        self.flush_at_episode_end = flush_at_episode_end
        self._lock = threading.Lock()
        self.pending_data_blocks: list[np.ndarray] = []

        item_shape = shape[1:] if len(shape) > 1 else ()
        self.buffer_threshold = max(1, min(chunk_size // 10, DEFAULT_BUFFER_THRESHOLD))
        self.buffer = np.empty((self.buffer_threshold, *item_shape), dtype=dtype)
        self.buffer_pos = 0

        with _HDF5_LOCK:
            create_kwargs: Dict[str, Any] = {
                "name": name,
                "shape": shape,
                "maxshape": (None, *item_shape),
                "dtype": dtype,
            }
            # Always set chunking and compression for resizable datasets
            if len(shape) > 0:
                # For resizable datasets, first dimension can be 0
                chunk_shape = (min(chunk_size, self.buffer_threshold), *item_shape) if item_shape else (min(chunk_size, self.buffer_threshold),)
                create_kwargs["chunks"] = chunk_shape
                
                if compression:
                    create_kwargs.update(
                        {"compression": compression, "shuffle": True, "fletcher32": True}
                    )
                    if compression != "lzf":
                        create_kwargs["compression_opts"] = compression_opts
            self.dataset = group.create_dataset(**create_kwargs)

        self.current_size = 0
        self.allocated_size = 0

    def append(self, data: np.ndarray) -> None:
        with self._lock:
            self.buffer[self.buffer_pos] = data
            self.buffer_pos += 1

            # Buffer is full
            if self.buffer_pos >= self.buffer_threshold:
                if self.flush_at_episode_end:
                    # Accumulate in memory if flush_at_episode_end is true
                    self.pending_data_blocks.append(self.buffer[: self.buffer_pos].copy())
                    self.buffer_pos = 0  # Buffer is now considered "empty" for new appends
                else:
                    # Not flush_at_episode_end, so flush to disk as usual
                    self._flush_current_buffer_to_disk()

    def flush(self) -> None:
        with self._lock:
            # Always handle any remaining data in the primary buffer first.
            if self.buffer_pos > 0:
                if self.flush_at_episode_end:
                    # If configured to flush at episode end, move remaining buffer to pending blocks.
                    self.pending_data_blocks.append(self.buffer[: self.buffer_pos].copy())
                    self.buffer_pos = 0
                else:
                    # Otherwise, flush the current buffer to disk directly.
                    self._flush_current_buffer_to_disk()

            # If there are pending blocks (only if flush_at_episode_end was true), flush them now.
            if self.pending_data_blocks:
                self._flush_all_pending_blocks_to_disk()

    def _flush_current_buffer_to_disk(self) -> None:
        if self.buffer_pos == 0:
            return
        new_data = self.buffer[: self.buffer_pos]
        n_new = self.buffer_pos
        self.buffer_pos = 0  # Reset buffer position
        self._do_sync_write(new_data)

    def _flush_all_pending_blocks_to_disk(self) -> None:
        # Writes all accumulated pending_data_blocks to disk.
        for block_data in self.pending_data_blocks:
            self._do_sync_write(block_data)
        self.pending_data_blocks.clear()

    def _do_sync_write(self, new_data: np.ndarray) -> None:
        n_new = len(new_data)
        new_size = self.current_size + n_new
        with _HDF5_LOCK:
            if new_size > self.allocated_size:
                self.allocated_size = int(
                    max(new_size * 1.5, self.allocated_size + self.chunk_size)
                )
                self.dataset.resize(self.allocated_size, axis=0)
            self.dataset[self.current_size : new_size] = new_data
        self.current_size = new_size

    def finalize(self) -> None:
        self.flush()
        if self.current_size < self.allocated_size:
            with _HDF5_LOCK:
                self.dataset.resize(self.current_size, axis=0)


class Episode:
    """High-performance, buffered episode data manager."""

    def __init__(
        self,
        episode_id: int,
        h5_group: Optional[h5py.Group],
        config: Dict[str, Any],
    ):
        self.episode_id = episode_id
        self.config = config
        self.neural_sampling_rate = config.get("neural_sampling_rate", 100)

        self.timestep_count = 0
        self.last_neural_sample = -self.neural_sampling_rate

        self.group: Optional[h5py.Group] = None
        self.neural_group: Optional[h5py.Group] = None
        self.behavior_group: Optional[h5py.Group] = None
        self.events_group: Optional[h5py.Group] = None
        self.plasticity_group: Optional[h5py.Group] = None
        
        if h5_group is not None:
            self.group = h5_group.create_group(f"episode_{episode_id:04d}")
            self.group.attrs["episode_id"] = episode_id
            self.group.attrs["start_time"] = datetime.now().isoformat()
            self.group.attrs["status"] = "running"
            self.neural_group = self.group.create_group("neural_states")
            self.behavior_group = self.group.create_group("behavior")
            self.events_group = self.group.create_group("events")
            self.plasticity_group = self.group.create_group("plasticity")
            self.data_manager = EpisodeDataManager(self.group, config)
        else:
            # no_write mode, create dummy manager
            self.data_manager = None  # type: ignore

    def log_timestep(self, **kwargs: Any) -> None:
        """Log data for a single timestep with optimized storage."""
        warnings_list = validate_timestep_data(**kwargs)
        for w in warnings_list:
            warnings.warn(f"Validation: {w}", stacklevel=2)

        self.timestep_count += 1
        timestep = kwargs["timestep"]

        if (data := kwargs.get("neural_state")) is not None and (
            timestep - self.last_neural_sample >= self.neural_sampling_rate
        ):
            if self.neural_group is not None:
                # Handle both dict and array inputs
                if isinstance(data, dict):
                    self._append_dict_data(self.neural_group, timestep, data)
                else:
                    # Convert array to dict with single key
                    self._append_dict_data(self.neural_group, timestep, {"state": data})
            self.last_neural_sample = timestep

        if (data := kwargs.get("behavior")) is not None:
            if self.behavior_group is not None:
                self._append_dict_data(self.behavior_group, timestep, data)

        # Handle rewards (sparse data)
        if (reward := kwargs.get("reward")) is not None and self.group is not None:
            if "rewards" not in self.group:
                self.group.create_group("rewards")
            rewards_group = self.group["rewards"]
            self._append_scalar_data(rewards_group, timestep, "values", float(reward))
        
        # Handle spikes (can be sparse or dense)
        spikes = kwargs.get("spikes")
        if spikes is not None and self.group is not None:
            if "spikes" not in self.group:
                self.group.create_group("spikes")
            spikes_group = self.group["spikes"]
            self._append_dict_data(spikes_group, timestep, {"spike_data": spikes})

    def _append_dict_data(self, group: h5py.Group, timestep: int, data: Dict[str, Any]) -> None:
        if self.data_manager is None:
            return

        ts_writer = self.data_manager.get_or_create_dataset(
            group, "timesteps", np.dtype(np.int64), (0,)
        )
        ts_writer.append(np.array(timestep, dtype=np.int64))

        for key, value in data.items():
            value_np = value if isinstance(value, np.ndarray) else ensure_numpy(value)
            shape = (0,) if value_np.ndim == 0 else (0, *value_np.shape)
            writer = self.data_manager.get_or_create_dataset(group, key, value_np.dtype, shape)
            writer.append(value_np)
    
    def _append_neural_state(self, timestep: int, data: Dict[str, Any]) -> None:
        """Append neural state data (used for post-processing sampled data)."""
        if self.neural_group is not None:
            self._append_dict_data(self.neural_group, timestep, data)
    
    def _append_scalar_data(self, group: h5py.Group, timestep: int, name: str, value: float) -> None:
        """Append scalar data efficiently."""
        if self.data_manager is None:
            return
        
        # Timesteps
        ts_writer = self.data_manager.get_or_create_dataset(
            group, "timesteps", np.dtype(np.int64), (0,)
        )
        ts_writer.append(np.array(timestep, dtype=np.int64))
        
        # Values
        value_writer = self.data_manager.get_or_create_dataset(
            group, name, np.dtype(np.float32), (0,)
        )
        value_writer.append(np.array(value, dtype=np.float32))

    def log_static_data(self, name: str, data: Dict[str, Any]) -> None:
        """Save static, one-off data for the episode."""
        if self.group is None:
            return
        with _HDF5_LOCK:
            group = self.group.create_group(name)
            for key, value in data.items():
                # Handle different data types efficiently
                if isinstance(value, str):
                    # HDF5 doesn't like Unicode, use fixed-length ASCII/UTF-8 bytes
                    group.create_dataset(key, data=value.encode('utf-8'))
                elif isinstance(value, (int, float, bool)):
                    # Scalars
                    group.create_dataset(key, data=value)
                elif isinstance(value, (list, tuple)):
                    # Convert to numpy array
                    group.create_dataset(key, data=np.array(value))
                else:
                    # Already numpy array or array-like
                    group.create_dataset(key, data=ensure_numpy(value))

    def log_event(self, name: str, timestep: int, data: Dict[str, Any]) -> None:
        """Log a discrete event with associated metadata."""
        if self.group is None:
            return
        with _HDF5_LOCK:
            # Event group name includes timestep for sorting
            event_group = self.events_group.create_group(f"event_{timestep:08d}_{name}")
            event_group.attrs["timestep"] = timestep
            event_group.attrs["name"] = name
            for key, value in data.items():
                try:
                    event_group.attrs[key] = value
                except TypeError:
                    event_group.attrs[key] = str(value)

    def log_weight_change(
        self,
        timestep: int,
        synapse_id: Tuple[int, int],
        old_weight: float,
        new_weight: float,
        learning_rule: Optional[str] = None,
    ) -> None:
        """Log a single synaptic weight change event."""
        if self.data_manager is None:
            return

        # For efficiency, we buffer these changes.
        writers = {
            "timesteps": (np.dtype(np.int64), timestep),
            "synapse_src": (np.dtype(np.int32), synapse_id[0]),
            "synapse_tgt": (np.dtype(np.int32), synapse_id[1]),
            "old_weights": (np.dtype(np.float32), old_weight),
            "new_weights": (np.dtype(np.float32), new_weight),
        }

        for name, (dtype, value) in writers.items():
            writer = self.data_manager.get_or_create_dataset(
                self.plasticity_group, name, dtype, (0,)
            )
            writer.append(np.array(value, dtype=dtype))
        
        # Store learning rule if provided as fixed-length string
        if learning_rule is not None:
            # Use fixed-length string type for HDF5 compatibility
            rule_bytes = learning_rule.encode('utf-8')[:32]  # Limit to 32 chars
            rule_writer = self.data_manager.get_or_create_dataset(
                self.plasticity_group, "learning_rules", np.dtype('S32'), (0,)
            )
            rule_writer.append(np.array(rule_bytes, dtype='S32'))

    def end(self, success: bool = False) -> None:
        """End episode and flush all buffers."""
        if self.data_manager:
            self.data_manager.finalize_all()
        if self.group:
            with _HDF5_LOCK:
                self.group.attrs["end_time"] = datetime.now().isoformat()
                self.group.attrs["success"] = success
                self.group.attrs["status"] = "completed"
                self.group.attrs["total_timesteps"] = self.timestep_count

    def __enter__(self) -> "Episode":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.end(success=exc_type is None)


class EpisodeDataManager:
    """Manages buffered dataset creation for an episode."""

    def __init__(self, group: h5py.Group, config: Dict[str, Any]):
        self.group = group
        self.config = config
        self.datasets: Dict[str, BufferedDataset] = {}
        self.flush_at_episode_end = self.config.get("flush_at_episode_end", False)

    def get_or_create_dataset(
        self, subgroup: h5py.Group, name: str, dtype: np.dtype, shape: Tuple[int, ...]
    ) -> BufferedDataset:
        key = f"{subgroup.name}/{name}"
        if key not in self.datasets:
            self.datasets[key] = BufferedDataset(
                subgroup,
                name,
                dtype,
                shape,
                chunk_size=self.config.get("chunk_size", 10000),
                compression=self.config.get("compression"),
                compression_opts=self.config.get("compression_opts", 4),
                flush_at_episode_end=self.flush_at_episode_end,
            )
        return self.datasets[key]

    def finalize_all(self) -> None:
        for dataset in self.datasets.values():
            dataset.finalize()