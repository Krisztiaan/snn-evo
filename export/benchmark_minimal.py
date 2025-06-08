# keywords: [minimal benchmark, performance test, comparison]
"""Minimal benchmark comparing implementations."""

import time
import numpy as np
import tempfile
import shutil
from pathlib import Path

# Only test the working implementations
from .exporter_optimized import OptimizedDataExporter


def benchmark_implementation(name, exporter_class, params, n_timesteps=1000):
    """Benchmark a single implementation."""
    temp_dir = tempfile.mkdtemp()

    try:
        # Pre-generate data
        neural_data = np.random.randn(n_timesteps, 500).astype(np.float32)
        spike_data = np.random.binomial(1, 0.01, (n_timesteps, 500)).astype(np.int8)

        start_time = time.time()

        with exporter_class(
            experiment_name="bench", output_base_dir=temp_dir, **params
        ) as exporter:
            with exporter.start_episode(0) as episode:
                for t in range(n_timesteps):
                    episode.log_timestep(
                        timestep=t, neural_state={"membrane": neural_data[t]}, spikes=spike_data[t]
                    )

        elapsed = time.time() - start_time

        # Calculate metrics
        data_size = neural_data.nbytes + spike_data.nbytes
        throughput = (data_size / 1024 / 1024) / elapsed

        # Get file size
        h5_file = list(Path(temp_dir).glob("**/experiment_data.h5"))[0]
        file_size = h5_file.stat().st_size / 1024 / 1024

        return {
            "time": elapsed,
            "throughput_mbps": throughput,
            "timesteps_per_sec": n_timesteps / elapsed,
            "file_size_mb": file_size,
            "compression_ratio": (data_size / 1024 / 1024) / file_size,
        }

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Run minimal benchmark."""
    print("Minimal Performance Benchmark")
    print("=" * 50)

    tests = [
        ("Optimized (Validation ON)", OptimizedDataExporter, {"validate_data": True}),
        ("Optimized (Validation OFF)", OptimizedDataExporter, {"validate_data": False}),
        ("Optimized (Async)", OptimizedDataExporter, {"validate_data": False, "async_write": True}),
    ]

    results = {}

    for name, cls, params in tests:
        print(f"\nTesting {name}...")
        try:
            # Warmup
            _ = benchmark_implementation(name, cls, params, n_timesteps=10)

            # Actual benchmark (3 runs)
            runs = []
            for i in range(3):
                result = benchmark_implementation(name, cls, params, n_timesteps=2000)
                runs.append(result)
                print(
                    f"  Run {i + 1}: {result['throughput_mbps']:.2f} MB/s in {result['time']:.2f}s"
                )

            # Average
            avg = {k: np.mean([r[k] for r in runs]) for k in runs[0]}
            results[name] = avg

        except Exception as e:
            print(f"  Error: {e}")
            import traceback

            traceback.print_exc()

    # Display results
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)

    baseline_name = "Optimized (Validation ON)"
    if baseline_name in results:
        baseline = results[baseline_name]["throughput_mbps"]

        print(f"\n{'Implementation':<25} {'Throughput':>12} {'Speedup':>8} {'File Size':>10}")
        print("-" * 60)

        for name, result in results.items():
            speedup = result["throughput_mbps"] / baseline
            print(
                f"{name:<25} {result['throughput_mbps']:>10.2f} MB/s "
                f"{speedup:>6.2f}x {result['file_size_mb']:>8.2f} MB"
            )

    # Key insights
    print("\n" + "=" * 50)
    print("KEY INSIGHTS")
    print("=" * 50)

    if len(results) >= 2:
        # Validation overhead
        if "Optimized (Validation ON)" in results and "Optimized (Validation OFF)" in results:
            val_on = results["Optimized (Validation ON)"]["throughput_mbps"]
            val_off = results["Optimized (Validation OFF)"]["throughput_mbps"]
            overhead = (1 - val_on / val_off) * 100
            print(f"1. Validation overhead: {overhead:.1f}%")

        # Async benefit
        if "Optimized (Validation OFF)" in results and "Optimized (Async)" in results:
            sync = results["Optimized (Validation OFF)"]["throughput_mbps"]
            async_val = results["Optimized (Async)"]["throughput_mbps"]
            benefit = (async_val / sync - 1) * 100
            print(f"2. Async I/O benefit: {benefit:.1f}%")

        # Overall improvement
        best = max(r["throughput_mbps"] for r in results.values())
        worst = min(r["throughput_mbps"] for r in results.values())
        total_improvement = (best / worst - 1) * 100
        print(f"3. Total improvement: {total_improvement:.1f}%")


if __name__ == "__main__":
    main()
