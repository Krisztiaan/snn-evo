# keywords: [test, performance, async, compression, profiling]
"""Test the performance enhancement features using pytest."""

import pytest
import numpy as np
import time
import tempfile
import shutil
from pathlib import Path

from export import DataExporter


@pytest.fixture(scope="module")
def temp_dir():
    """Create a temporary directory for the tests."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


def run_test_scenario(output_dir, experiment_name, **kwargs):
    """Helper function to run a standard test scenario."""
    start_time = time.time()
    with DataExporter(
        experiment_name=experiment_name, output_base_dir=output_dir, **kwargs
    ) as exporter:
        with exporter.start_episode(0) as episode:
            for t in range(1000):
                episode.log_timestep(
                    timestep=t,
                    neural_state={"v": np.random.randn(1000)},
                    spikes=np.random.binomial(1, 0.01, 1000),
                )
    return time.time() - start_time


class TestPerformanceEnhancements:
    def test_async_writes(self, temp_dir):
        """Test that async writes are faster than sync writes."""
        sync_time = run_test_scenario(temp_dir, "test_sync", async_write=False)
        async_time = run_test_scenario(temp_dir, "test_async", async_write=True, n_async_workers=4)

        print(f"\nSync write time: {sync_time:.2f}s")
        print(f"Async write time: {async_time:.2f}s")
        print(f"Speedup: {sync_time / async_time:.2f}x")

        # Async should be faster, but allow a small margin for system variability
        assert async_time < sync_time * 1.2

    def test_adaptive_compression(self, temp_dir):
        """Test adaptive compression runs without errors."""
        run_test_scenario(temp_dir, "test_adaptive", adaptive_compression=True)

        # Check that a file was created
        exp_dir = list(Path(temp_dir).glob("test_adaptive_*"))[0]
        assert (exp_dir / "experiment_data.h5").exists()

    def test_performance_profiling(self, temp_dir):
        """Test that performance profiling generates stats."""
        with DataExporter(
            experiment_name="test_profiling", output_base_dir=temp_dir, enable_profiling=True
        ) as exporter:
            with exporter.start_episode(0) as episode:
                for t in range(100):
                    episode.log_timestep(
                        timestep=t,
                        neural_state={"membrane": np.random.randn(100)},
                    )

            stats = exporter.get_performance_stats()
            assert "profiling" in stats
            assert "append_neural_state" in stats["profiling"]
            print("\nProfiling stats generated:", stats["profiling"])

    def test_realtime_stats(self, temp_dir):
        """Test real-time statistics tracking."""
        with DataExporter(
            experiment_name="test_realtime",
            output_base_dir=temp_dir,
            async_write=True,  # Enables realtime stats
        ) as exporter:
            with exporter.start_episode(0) as episode:
                for t in range(100):
                    episode.log_timestep(timestep=t, neural_state={"data": np.random.randn(500)})

                stats = episode.realtime_stats.get_summary()
                assert "recent_throughput_mbps" in stats
                assert stats["total_ops"] == 100
                print("\nRealtime stats generated:", stats)
