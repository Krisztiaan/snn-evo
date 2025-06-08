# keywords: [test, pytest, performance, benchmark, async, compression]
"""Performance tests for the DataExporter module."""

import pytest
import numpy as np
import time
import tempfile
import shutil
from pathlib import Path

from export import DataExporter


@pytest.fixture(scope="module")
def temp_dir():
    """Create a temporary directory for the tests in this module."""
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)


def run_benchmark_scenario(output_dir: Path, experiment_name: str, **kwargs) -> float:
    """Helper function to run a standard test scenario and return elapsed time."""
    start_time = time.perf_counter()
    with DataExporter(
        experiment_name=experiment_name, output_base_dir=output_dir, **kwargs
    ) as exporter:
        with exporter.start_episode(0) as episode:
            for t in range(1000):
                episode.log_timestep(
                    timestep=t,
                    neural_state={"v": np.random.randn(1000)},
                    behavior={"pos": np.random.randn(3)},
                )
    return time.perf_counter() - start_time


class TestPerformance:
    """A suite of performance tests for the DataExporter."""

    def test_async_writes_are_faster(self, temp_dir: Path):
        """Test that asynchronous writes provide a speedup."""
        # Run sync version
        sync_time = run_benchmark_scenario(temp_dir, "test_sync", async_write=False)
        # Run async version
        async_time = run_benchmark_scenario(
            temp_dir, "test_async", async_write=True, n_async_workers=4
        )

        print(f"\nSync write time: {sync_time:.3f}s")
        print(f"Async write time: {async_time:.3f}s")
        print(f"Speedup: {sync_time / async_time:.2f}x")

        # Async should be faster, but allow a small margin for system variability and overhead
        assert async_time < sync_time * 1.1

    def test_validation_overhead(self, temp_dir: Path):
        """Test the performance overhead of runtime data validation."""
        # With validation (default)
        validation_on_time = run_benchmark_scenario(
            temp_dir, "test_validation_on", validate_data=True, async_write=False
        )
        # Without validation
        validation_off_time = run_benchmark_scenario(
            temp_dir, "test_validation_off", validate_data=False, async_write=False
        )

        print(f"\nValidation ON time: {validation_on_time:.3f}s")
        print(f"Validation OFF time: {validation_off_time:.3f}s")

        # Disabling validation should be faster
        assert validation_off_time < validation_on_time

    def test_profiling_feature(self, temp_dir: Path):
        """Test that performance profiling runs and generates stats without crashing."""
        with DataExporter(
            experiment_name="test_profiling", output_base_dir=temp_dir, enable_profiling=True
        ) as exporter:
            with exporter.start_episode(0) as episode:
                for t in range(100):
                    episode.log_timestep(timestep=t, neural_state={"v": np.random.randn(100)})

            stats = exporter.get_performance_stats()
            assert "profiling" in stats
            assert "append_dict_data" in stats["profiling"]
            print("\nProfiling stats generated:", stats["profiling"])
