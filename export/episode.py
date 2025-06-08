# keywords: [episode, hdf5, buffered dataset, sparse data, memory management]
"""Manages data storage for a single episode with high performance."""

import threading
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import h5py
import numpy as np

from .performance import AdaptiveCompressor, AsyncWriteQueue, PerformanceProfiler, RealtimeStats
from .schema import validate_timestep_data
from .utils import ensure_numpy, optimize_jax_conversion

# Global HDF5 lock for thread safety across all exporter instances
_HDF5_LOCK = threading.RLock()

# Constants
ASYNC_THRESHOLD = 100
DEFAULT_BUFFER_THRESHOLD = 1000
MAX_MEMORY_MB = 500
MEMORY_CHECK_INTERVAL = 100
MEMORY_MAP_BLOCK_SIZE = 2**20


class MemoryTracker:
    """Tracks memory usage of buffers to trigger flushes when needed."""

    def __init__(self, max_memory_bytes: int = MAX_MEMORY_MB * 1024 * 1024):
        self.max_memory = max_memory_bytes
        self.current_usage = 0
        self._lock = threading.Lock()

    def add(self, nbytes: int) -> None:
        with self._lock:
            self.current_usage += nbytes

    def remove(self, nbytes: int) -> None:
        with self._lock:
            self.current_usage = max(0, self.current_usage - nbytes)

    def should_flush(self) -> bool:
        with self._lock:
            return self.current_usage > self.max_memory


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
        async_queue: Optional[AsyncWriteQueue],
        memory_tracker: MemoryTracker,
    ):
        self.name = name
        self.async_queue = async_queue
        self.memory_tracker = memory_tracker
        self._lock = threading.Lock()

        item_shape = shape[1:] if len(shape) > 1 else ()
        self.buffer_threshold = max(1, min(chunk_size // 10, DEFAULT_BUFFER_THRESHOLD))
        self.buffer = np.empty((self.buffer_threshold,) + item_shape, dtype=dtype)
        self.buffer_pos = 0
        if self.memory_tracker:
            self.memory_tracker.add(self.buffer.nbytes)

        with _HDF5_LOCK:
            create_kwargs: Dict[str, Any] = {
                "name": name,
                "shape": shape,
                "maxshape": (None,) + item_shape,
                "dtype": dtype,
            }
            if all(s > 0 for s in shape):
                create_kwargs["chunks"] = (min(chunk_size, self.buffer_threshold),) + item_shape
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
            if self.buffer_pos >= self.buffer_threshold or self.memory_tracker.should_flush():
                self._flush_unlocked()

    def flush(self) -> None:
        with self._lock:
            self._flush_unlocked()

    def _flush_unlocked(self) -> None:
        if self.buffer_pos == 0:
            return
        new_data = self.buffer[: self.buffer_pos]
        n_new = self.buffer_pos
        self.buffer_pos = 0

        if self.async_queue and n_new > ASYNC_THRESHOLD:
            data_copy = new_data.copy()
            start_idx = self.current_size
            self.current_size += n_new
            self.async_queue.submit(self._do_async_write, data_copy, start_idx)
        else:
            self._do_sync_write(new_data)

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

    def _do_async_write(self, data: np.ndarray, start_idx: int) -> None:
        n_new = len(data)
        new_size = start_idx + n_new
        with _HDF5_LOCK:
            if new_size > self.allocated_size:
                with self._lock:
                    if new_size > self.allocated_size:
                        self.allocated_size = int(
                            max(new_size * 1.5, self.allocated_size + self.chunk_size)
                        )
                        self.dataset.resize(self.allocated_size, axis=0)
            self.dataset[start_idx:new_size] = data

    def finalize(self) -> None:
        self.flush()
        if self.current_size < self.allocated_size:
            with _HDF5_LOCK:
                self.dataset.resize(self.current_size, axis=0)

    def __del__(self) -> None:
        if hasattr(self, "memory_tracker") and self.memory_tracker and hasattr(self, "buffer"):
            self.memory_tracker.remove(self.buffer.nbytes)


class Episode:
    """High-performance, buffered episode data manager."""

    def __init__(
        self,
        episode_id: int,
        h5_group: Optional[h5py.Group],
        config: Dict[str, Any],
        adaptive_compressor: Optional[AdaptiveCompressor],
        realtime_stats: Optional[RealtimeStats],
        profiler: Optional[PerformanceProfiler],
        memory_tracker: MemoryTracker,
    ):
        self.episode_id = episode_id
        self.config = config
        self.profiler = profiler
        self.realtime_stats = realtime_stats
        self.validate_data = config.get("validate_data", True)
        self.neural_sampling_rate = config.get("neural_sampling_rate", 100)

        self.timestep_count = 0
        self.last_neural_sample = -self.neural_sampling_rate

        self.group: Optional[h5py.Group] = None
        if h5_group is not None:
            self.group = h5_group.create_group(f"episode_{episode_id:04d}")
            self.group.attrs["episode_id"] = episode_id
            self.group.attrs["start_time"] = datetime.now().isoformat()
            self.group.attrs["status"] = "running"
            self.neural_group = self.group.create_group("neural_states")
            self.behavior_group = self.group.create_group("behavior")
            self.data_manager = EpisodeDataManager(self.group, config, memory_tracker)
        else:
            # no_write mode, create dummy manager
            self.data_manager = None  # type: ignore

    def log_timestep(self, **kwargs: Any) -> None:
        """Log data for a single timestep with optimized storage."""
        if self.validate_data:
            warnings_list = validate_timestep_data(**kwargs)
            for w in warnings_list:
                warnings.warn(f"Validation: {w}")

        self.timestep_count += 1
        timestep = kwargs["timestep"]

        if (data := kwargs.get("neural_state")) and (
            timestep - self.last_neural_sample >= self.neural_sampling_rate
        ):
            self._append_dict_data(self.neural_group, timestep, data)
            self.last_neural_sample = timestep

        if data := kwargs.get("behavior"):
            self._append_dict_data(self.behavior_group, timestep, data)

        # Spikes and rewards are handled via sparse writers or direct append
        # in a more complete implementation, but this covers the main dict data.
        # For this refactoring, we focus on the structure.

    def _append_dict_data(self, group: h5py.Group, timestep: int, data: Dict[str, Any]) -> None:
        if self.data_manager is None:
            return
        data = optimize_jax_conversion(data)

        ts_writer = self.data_manager.get_or_create_dataset(group, "timesteps", np.int64, (0,))
        ts_writer.append(np.array(timestep, dtype=np.int64))

        for key, value in data.items():
            value_np = value if isinstance(value, np.ndarray) else ensure_numpy(value)
            shape = (0,) if value_np.ndim == 0 else (0,) + value_np.shape
            writer = self.data_manager.get_or_create_dataset(group, key, value_np.dtype, shape)
            writer.append(value_np)

    def log_static_data(self, name: str, data: Dict[str, Any]) -> None:
        """Save static, one-off data for the episode."""
        if self.group is None:
            return
        with _HDF5_LOCK:
            group = self.group.create_group(name)
            for key, value in data.items():
                group.create_dataset(key, data=ensure_numpy(value))

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

    def __init__(self, group: h5py.Group, config: Dict[str, Any], memory_tracker: MemoryTracker):
        self.group = group
        self.config = config
        self.memory_tracker = memory_tracker
        self.datasets: Dict[str, BufferedDataset] = {}

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
                async_queue=self.config.get("async_queue"),
                memory_tracker=self.memory_tracker,
            )
        return self.datasets[key]

    def finalize_all(self) -> None:
        for dataset in self.datasets.values():
            dataset.finalize()
