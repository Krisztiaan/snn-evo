# keywords: [data exporter, hdf5, high-performance, experiment management]
"""The main DataExporter class for managing experiment data files."""

import json
import warnings
import zlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import h5py
import numpy as np

from .episode import _HDF5_LOCK, MEMORY_MAP_BLOCK_SIZE, Episode, MemoryTracker
from .performance import (
    AdaptiveCompressor,
    AsyncWriteQueue,
    PerformanceProfiler,
    RealtimeStats,
)
from .schema import (
    SCHEMA_VERSION,
    validate_network_structure,
)
from .utils import NumpyEncoder, ensure_numpy


class DataExporter:
    """High-performance HDF5 data exporter with advanced optimizations."""

    def __init__(
        self,
        experiment_name: str,
        output_base_dir: str = "experiments",
        neural_sampling_rate: int = 100,
        validate_data: bool = True,
        compression: Optional[str] = "gzip",
        compression_level: int = 4,
        chunk_size: int = 10000,
        enable_swmr: bool = False,
        async_write: bool = False,
        enable_mmap: bool = False,
        adaptive_compression: bool = False,
        enable_profiling: bool = False,
        n_async_workers: int = 4,
        no_write: bool = False,
    ):
        """Initialize the data exporter.

        Args:
            experiment_name: Name of the experiment.
            output_base_dir: Base directory for all experiment outputs.
            neural_sampling_rate: Sample neural state every N timesteps.
            validate_data: Whether to perform runtime data validation. Disable for performance.
            compression: HDF5 compression algorithm ('gzip', 'lzf', None).
            compression_level: Compression level for 'gzip' (1-9).
            chunk_size: Default chunk size for HDF5 datasets.
            enable_swmr: Enable Single-Writer Multiple-Reader (SWMR) mode for live analysis.
            async_write: Enable asynchronous I/O operations for performance.
            enable_mmap: Enable memory-mapped mode for potentially large experiments.
            adaptive_compression: Enable adaptive compression based on data entropy.
            enable_profiling: Enable detailed performance profiling.
            n_async_workers: Number of worker threads for asynchronous writing.
            no_write: If True, all file I/O operations are skipped. For benchmarking.
        """
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_base_dir) / f"{experiment_name}_{self.timestamp}"
        self.no_write = no_write

        # Core configuration
        self.config: Dict[str, Any] = {
            "neural_sampling_rate": neural_sampling_rate,
            "validate_data": validate_data,
            "compression": compression,
            "compression_opts": compression_level,
            "chunk_size": chunk_size,
        }

        # Performance components
        self.async_write = async_write
        if async_write:
            self.async_queue: Optional[AsyncWriteQueue] = AsyncWriteQueue(n_workers=n_async_workers)
            self.config["async_queue"] = self.async_queue
        else:
            self.async_queue = None

        self.adaptive_compressor = AdaptiveCompressor() if adaptive_compression else None
        self.profiler = PerformanceProfiler() if enable_profiling else None
        self.realtime_stats = RealtimeStats() if async_write or enable_profiling else None
        self.memory_tracker = MemoryTracker()

        # State
        self.current_episode: Optional[Episode] = None
        self.episode_count = 0
        self.h5_file: Optional[h5py.File] = None

        if not self.no_write:
            self._setup_files(enable_swmr, enable_mmap)
            self.save_runtime_info()

    def _setup_files(self, enable_swmr: bool, enable_mmap: bool) -> None:
        """Create directories and the main HDF5 file."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        h5_path = self.output_dir / "experiment_data.h5"

        try:
            if enable_mmap:
                self.h5_file = h5py.File(
                    h5_path,
                    "w",
                    driver="core",
                    backing_store=True,
                    block_size=MEMORY_MAP_BLOCK_SIZE,
                )
            elif enable_swmr:
                self.h5_file = h5py.File(h5_path, "w", libver="latest")
                self.h5_file.swmr_mode = True
            else:
                self.h5_file = h5py.File(h5_path, "w")
        except Exception as e:
            raise RuntimeError(f"Failed to create HDF5 file at {h5_path}: {e}")

        with _HDF5_LOCK:
            self.h5_file.attrs["experiment_name"] = self.experiment_name
            self.h5_file.attrs["timestamp"] = self.timestamp
            self.h5_file.attrs["start_time"] = datetime.now().isoformat()
            self.h5_file.attrs["schema_version"] = SCHEMA_VERSION
            self.h5_file.attrs.update(self.config)
            self.episodes_group = self.h5_file.create_group("episodes")
            self.checkpoints_group = self.h5_file.create_group("checkpoints")

    def save_config(self, config: Dict[str, Any]) -> None:
        """Save experiment configuration to JSON and HDF5."""
        if self.no_write or self.h5_file is None:
            return

        with open(self.output_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2, cls=NumpyEncoder)

        with _HDF5_LOCK:
            if "config" in self.h5_file:
                del self.h5_file["config"]
            config_group = self.h5_file.create_group("config")
            for key, value in config.items():
                if isinstance(value, (dict, list)):
                    json_str = json.dumps(value, cls=NumpyEncoder)
                    compressed = zlib.compress(json_str.encode("utf-8"))
                    config_group.attrs[key] = np.void(compressed)
                    config_group.attrs[f"{key}_compressed"] = True
                else:
                    try:
                        config_group.attrs[key] = value
                    except TypeError:
                        config_group.attrs[key] = str(value)

    def save_network_structure(
        self,
        neurons: Dict[str, Any],
        connections: Dict[str, Any],
        initial_weights: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save network structure with optimization."""
        if self.no_write or self.h5_file is None:
            return

        if self.config["validate_data"]:
            warnings_list = validate_network_structure(neurons, connections)
            for w in warnings_list:
                warnings.warn(f"Network validation: {w}")

        with _HDF5_LOCK:
            net_group = self.h5_file.create_group("network_structure")
            neurons_group = net_group.create_group("neurons")
            for key, value in neurons.items():
                neurons_group.create_dataset(key, data=ensure_numpy(value))

            conn_group = net_group.create_group("connections")
            for key, value in connections.items():
                conn_group.create_dataset(key, data=ensure_numpy(value))

            if initial_weights is not None:
                weights_group = net_group.create_group("initial_weights")
                for key, value in initial_weights.items():
                    weights_group.create_dataset(key, data=ensure_numpy(value))

            self.h5_file.attrs["n_neurons"] = len(neurons.get("neuron_ids", []))
            self.h5_file.attrs["n_connections"] = len(connections.get("source_ids", []))

    def start_episode(self, episode_id: Optional[int] = None) -> Episode:
        """Start a new episode."""
        if self.current_episode is not None:
            warnings.warn(f"Previous episode {self.current_episode.episode_id} not ended cleanly.")
            self.current_episode.end(success=False)

        if episode_id is None:
            episode_id = self.episode_count

        if self.no_write or self.h5_file is None:
            h5_group = None
        else:
            with _HDF5_LOCK:
                h5_group = self.episodes_group

        self.current_episode = Episode(
            episode_id=episode_id,
            h5_group=h5_group,
            config=self.config,
            adaptive_compressor=self.adaptive_compressor,
            realtime_stats=self.realtime_stats,
            profiler=self.profiler,
            memory_tracker=self.memory_tracker,
        )

        self.episode_count += 1
        if not self.no_write and self.h5_file:
            with _HDF5_LOCK:
                self.h5_file.attrs["episode_count"] = self.episode_count

        return self.current_episode

    def end_episode(self, success: bool = False, summary: Optional[Dict[str, Any]] = None) -> None:
        """End the current episode and save summary."""
        if self.current_episode is None:
            warnings.warn("No active episode to end.")
            return

        self.current_episode.end(success=success)

        if not self.no_write and self.h5_file:
            with _HDF5_LOCK:
                if "episode_summaries" not in self.h5_file:
                    self.h5_file.create_group("episode_summaries")
                summary_group = self.h5_file["episode_summaries"]
                ep_summary_group = summary_group.create_group(
                    f"episode_{self.current_episode.episode_id:04d}"
                )
                if self.current_episode.group is not None:
                    for key, value in self.current_episode.group.attrs.items():
                        ep_summary_group.attrs[key] = value
                if summary:
                    for key, value in summary.items():
                        ep_summary_group.attrs[key] = value
                self.h5_file.flush()

        self.current_episode = None

    def log(self, **kwargs: Any) -> None:
        """Log data to the current episode."""
        if self.current_episode is None:
            raise RuntimeError("No active episode. Call start_episode() first.")

        timestep = kwargs.get("timestep")
        if timestep is None:
            raise ValueError("'timestep' is a required argument for log()")

        self.current_episode.log_timestep(**kwargs)

    def log_static_episode_data(self, name: str, data: Dict[str, Any]) -> None:
        """Save static, one-off data for the current episode."""
        if self.current_episode is None:
            raise RuntimeError("No active episode. Call start_episode() first.")
        self.current_episode.log_static_data(name, data)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        stats: Dict[str, Any] = {}
        if self.realtime_stats:
            stats["realtime"] = self.realtime_stats.get_summary()
        if self.async_queue:
            stats["async_io"] = self.async_queue.get_stats()
        if self.profiler:
            stats["profiling"] = self.profiler.get_report()
        return stats

    def save_runtime_info(self) -> None:
        """Capture and save runtime environment information."""
        import platform
        import sys

        runtime_info = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "h5py_version": h5py.__version__,
            "numpy_version": np.__version__,
            "output_directory": str(self.output_dir),
            "hdf5_driver": self.h5_file.driver if self.h5_file else "none",
        }
        if self.no_write:
            return

        with _HDF5_LOCK:
            if "runtime" in self.h5_file:
                del self.h5_file["runtime"]
            runtime_group = self.h5_file.create_group("runtime")
            for key, value in runtime_info.items():
                if value is not None:
                    runtime_group.attrs[key] = str(value)

    def close(self) -> None:
        """Finalize the experiment and close all resources."""
        if self.current_episode:
            self.current_episode.end(success=False)

        if self.async_queue:
            self.async_queue.shutdown()

        if not self.no_write and self.h5_file:
            if self.profiler or self.async_queue:
                perf_stats = self.get_performance_stats()
                if perf_stats:
                    with open(self.output_dir / "performance_stats.json", "w") as f:
                        json.dump(perf_stats, f, indent=2, cls=NumpyEncoder)

            with _HDF5_LOCK:
                self.h5_file.attrs["end_time"] = datetime.now().isoformat()
                self.h5_file.attrs["total_episodes"] = self.episode_count
                self.h5_file.close()

        if self.profiler:
            self.profiler.print_report()

    def __enter__(self) -> "DataExporter":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
