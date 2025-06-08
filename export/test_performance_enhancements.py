# keywords: [test, performance, async, compression, profiling]
"""Test the performance enhancement features."""

import numpy as np
import time
import tempfile
import shutil
from pathlib import Path

from export import DataExporter


def test_async_writes():
    """Test async write functionality."""
    print("Testing async writes...")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Test with async writes enabled
        start_time = time.time()

        with DataExporter(
            experiment_name="test_async",
            output_base_dir=temp_dir,
            async_write=True,
            n_async_workers=4,
        ) as exporter:
            with exporter.start_episode(0) as episode:
                for t in range(1000):
                    episode.log_timestep(
                        timestep=t,
                        neural_state={"v": np.random.randn(1000)},
                        spikes=np.random.binomial(1, 0.01, 1000),
                    )

        async_time = time.time() - start_time
        print(f"  Async write time: {async_time:.2f}s")

        # Test without async writes
        start_time = time.time()

        with DataExporter(
            experiment_name="test_sync", output_base_dir=temp_dir, async_write=False
        ) as exporter:
            with exporter.start_episode(0) as episode:
                for t in range(1000):
                    episode.log_timestep(
                        timestep=t,
                        neural_state={"v": np.random.randn(1000)},
                        spikes=np.random.binomial(1, 0.01, 1000),
                    )

        sync_time = time.time() - start_time
        print(f"  Sync write time: {sync_time:.2f}s")
        print(f"  Speedup: {sync_time / async_time:.2f}x")


def test_adaptive_compression():
    """Test adaptive compression."""
    print("\nTesting adaptive compression...")

    with tempfile.TemporaryDirectory() as temp_dir:
        with DataExporter(
            experiment_name="test_adaptive", output_base_dir=temp_dir, adaptive_compression=True
        ) as exporter:
            with exporter.start_episode(0) as episode:
                # Test with different data types

                # Sparse data (should use light compression)
                sparse_data = np.zeros(10000)
                sparse_data[::100] = 1.0
                episode.log_timestep(timestep=0, neural_state={"sparse": sparse_data})

                # Random data (minimal compression benefit)
                random_data = np.random.randn(10000)
                episode.log_timestep(timestep=1, neural_state={"random": random_data})

                # Repetitive data (high compression benefit)
                repetitive_data = np.tile(np.arange(10), 1000)
                episode.log_timestep(timestep=2, neural_state={"repetitive": repetitive_data})

        # Check file size
        exp_dir = list(Path(temp_dir).glob("test_adaptive_*"))[0]
        file_size = (exp_dir / "experiment_data.h5").stat().st_size
        print(f"  File size with adaptive compression: {file_size / 1024:.1f} KB")


def test_performance_profiling():
    """Test performance profiling."""
    print("\nTesting performance profiling...")

    with tempfile.TemporaryDirectory() as temp_dir:
        with DataExporter(
            experiment_name="test_profiling", output_base_dir=temp_dir, enable_profiling=True
        ) as exporter:
            with exporter.start_episode(0) as episode:
                # Generate some data
                for t in range(100):
                    episode.log_timestep(
                        timestep=t,
                        neural_state={"membrane": np.random.randn(1000)},
                        spikes=np.random.binomial(1, 0.01, 1000),
                        behavior={"position": np.random.randn(3)},
                        reward=np.random.rand(),
                    )

                    # Log some weight changes
                    if t % 10 == 0:
                        for i in range(10):
                            episode.log_weight_change(
                                timestep=t, synapse_id=(i, i + 10), old_weight=0.5, new_weight=0.51
                            )

            # Get performance stats
            stats = exporter.get_performance_stats()
            print("\n  Performance Statistics:")
            for key, value in stats.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for k, v in value.items():
                        print(f"    {k}: {v}")
                else:
                    print(f"  {key}: {value}")

            # Print detailed profiling report
            if exporter.profiler:
                exporter.profiler.print_report()


def test_realtime_stats():
    """Test real-time statistics tracking."""
    print("\nTesting real-time stats...")

    with tempfile.TemporaryDirectory() as temp_dir:
        with DataExporter(
            experiment_name="test_realtime",
            output_base_dir=temp_dir,
            async_write=True,  # This enables realtime stats
        ) as exporter:
            with exporter.start_episode(0) as episode:
                # Log data and periodically check stats
                for t in range(1000):
                    episode.log_timestep(timestep=t, neural_state={"data": np.random.randn(500)})

                    # Check stats every 100 timesteps
                    if t > 0 and t % 100 == 0:
                        if episode.realtime_stats:
                            stats = episode.realtime_stats.get_summary()
                            print(
                                f"  Timestep {t}: {stats['recent_throughput_mbps']:.1f} MB/s, "
                                f"Memory: {stats['current_memory_mb']:.1f} MB"
                            )


if __name__ == "__main__":
    print("=== Performance Enhancement Tests ===\n")

    test_async_writes()
    test_adaptive_compression()
    test_performance_profiling()
    test_realtime_stats()

    print("\n=== All tests completed! ===")
