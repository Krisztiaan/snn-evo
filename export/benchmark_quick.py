# keywords: [benchmark, performance, quick, comparison]
"""Quick benchmark of performance enhancements."""

import numpy as np
import time
import tempfile
from pathlib import Path

from export import DataExporter


def quick_benchmark(name: str, n_timesteps: int = 1000, n_neurons: int = 500, **config) -> dict:
    """Quick benchmark of a configuration."""

    with tempfile.TemporaryDirectory() as temp_dir:
        start_time = time.perf_counter()

        with DataExporter(experiment_name="bench", output_base_dir=temp_dir, **config) as exporter:
            with exporter.start_episode(0) as episode:
                # Generate data
                for t in range(n_timesteps):
                    episode.log_timestep(
                        timestep=t,
                        neural_state={"v": np.random.randn(n_neurons)},
                        spikes=np.random.binomial(1, 0.01, n_neurons) if t % 10 == 0 else None,
                    )

        elapsed_time = time.perf_counter() - start_time

        # Get file size
        exp_dir = list(Path(temp_dir).glob("bench_*"))[0]
        file_size = (exp_dir / "experiment_data.h5").stat().st_size

        # Calculate throughput
        data_size_mb = (n_timesteps * n_neurons * 8) / (1024 * 1024)
        throughput_mbps = data_size_mb / elapsed_time

        return {
            "name": name,
            "time": elapsed_time,
            "throughput_mbps": throughput_mbps,
            "file_size_mb": file_size / (1024 * 1024),
        }


def main():
    """Run quick benchmarks."""
    print("=== Quick Performance Enhancement Benchmark ===\n")

    configs = [
        ("Baseline", {}),
        ("Async (2 workers)", {"async_write": True, "n_async_workers": 2}),
        ("Async (4 workers)", {"async_write": True, "n_async_workers": 4}),
        ("No Validation", {"validate_data": False}),
        ("Fast Compression", {"compression": "lzf"}),
        (
            "All Enhancements",
            {
                "validate_data": False,
                "async_write": True,
                "n_async_workers": 4,
                "compression": "lzf",
            },
        ),
    ]

    results = []
    for name, config in configs:
        print(f"Testing: {name}...", end=" ")
        result = quick_benchmark(name, **config)
        results.append(result)
        print(f"{result['throughput_mbps']:.1f} MB/s")

    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY (Higher is better)")
    print("=" * 50)

    baseline = results[0]["throughput_mbps"]
    for result in results:
        improvement = (result["throughput_mbps"] / baseline - 1) * 100
        print(f"{result['name']:<20} {result['throughput_mbps']:>6.1f} MB/s ({improvement:+5.1f}%)")


if __name__ == "__main__":
    main()
