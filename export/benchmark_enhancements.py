# keywords: [benchmark, performance, enhancements, comparison]
"""Benchmark performance enhancements against baseline."""

import numpy as np
import time
import tempfile
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import json

from export import DataExporter


def benchmark_configuration(name: str, n_timesteps: int, n_neurons: int, **config) -> dict:
    """Benchmark a specific configuration."""
    print(f"\nBenchmarking: {name}")

    with tempfile.TemporaryDirectory() as temp_dir:
        start_time = time.perf_counter()

        with DataExporter(
            experiment_name="benchmark", output_base_dir=temp_dir, **config
        ) as exporter:
            # Save some initial data
            exporter.save_config({"n_neurons": n_neurons})
            exporter.save_network_structure(
                {"ids": np.arange(n_neurons)},
                {
                    "sources": np.arange(n_neurons // 2),
                    "targets": np.arange(n_neurons // 2, n_neurons),
                },
            )

            with exporter.start_episode(0) as episode:
                # Generate data
                for t in range(n_timesteps):
                    # Simulate computation time
                    computation_data = np.random.randn(n_neurons) * np.sin(t * 0.1)

                    episode.log_timestep(
                        timestep=t,
                        neural_state={"membrane_potential": computation_data},
                        spikes=np.random.binomial(1, 0.01, n_neurons),
                        behavior={"position": np.array([t * 0.1, t * 0.05, 0])},
                        reward=1.0 if t % 100 == 0 else 0.0,
                    )

                    # Occasional weight changes
                    if t % 50 == 0:
                        for _ in range(20):
                            src, tgt = np.random.randint(0, n_neurons, 2)
                            episode.log_weight_change(
                                timestep=t,
                                synapse_id=(int(src), int(tgt)),
                                old_weight=0.5,
                                new_weight=0.5 + np.random.randn() * 0.01,
                            )

            # Get stats if available
            stats = (
                exporter.get_performance_stats()
                if hasattr(exporter, "get_performance_stats")
                else {}
            )

        elapsed_time = time.perf_counter() - start_time

        # Get file size
        exp_dir = list(Path(temp_dir).glob("benchmark_*"))[0]
        file_size = (exp_dir / "experiment_data.h5").stat().st_size

        # Calculate throughput
        data_size_mb = (n_timesteps * n_neurons * 8) / (1024 * 1024)  # Approximate
        throughput_mbps = data_size_mb / elapsed_time

        return {
            "name": name,
            "elapsed_time": elapsed_time,
            "file_size_mb": file_size / (1024 * 1024),
            "throughput_mbps": throughput_mbps,
            "data_size_mb": data_size_mb,
            "compression_ratio": data_size_mb / (file_size / (1024 * 1024)),
            "stats": stats,
        }


def run_benchmarks():
    """Run comprehensive benchmarks."""
    # Test configurations
    configs = [
        # Baseline
        {
            "name": "Baseline",
            "config": {
                "validate_data": True,
                "async_write": False,
                "compression": "gzip",
                "compression_level": 4,
            },
        },
        # Async writes
        {
            "name": "Async Writes (2 workers)",
            "config": {
                "validate_data": True,
                "async_write": True,
                "n_async_workers": 2,
                "compression": "gzip",
                "compression_level": 4,
            },
        },
        # Async writes with more workers
        {
            "name": "Async Writes (4 workers)",
            "config": {
                "validate_data": True,
                "async_write": True,
                "n_async_workers": 4,
                "compression": "gzip",
                "compression_level": 4,
            },
        },
        # Adaptive compression
        {
            "name": "Adaptive Compression",
            "config": {"validate_data": True, "async_write": False, "adaptive_compression": True},
        },
        # Full optimizations
        {
            "name": "All Enhancements",
            "config": {
                "validate_data": False,
                "async_write": True,
                "n_async_workers": 4,
                "adaptive_compression": True,
                "enable_profiling": True,
            },
        },
        # Ultra performance
        {
            "name": "Ultra Performance",
            "config": {
                "validate_data": False,
                "async_write": True,
                "n_async_workers": 8,
                "compression": "lzf",  # Faster compression
                "chunk_size": 50000,  # Larger chunks
                "enable_profiling": False,
            },
        },
    ]

    # Test with different data sizes
    test_cases = [
        {"n_timesteps": 1000, "n_neurons": 1000, "label": "Small (1K x 1K)"},
        {"n_timesteps": 5000, "n_neurons": 2000, "label": "Medium (5K x 2K)"},
        {"n_timesteps": 10000, "n_neurons": 5000, "label": "Large (10K x 5K)"},
    ]

    results = []

    for test_case in test_cases:
        print(f"\n{'=' * 60}")
        print(f"Test Case: {test_case['label']}")
        print(f"{'=' * 60}")

        case_results = []

        for config in configs:
            result = benchmark_configuration(
                name=config["name"],
                n_timesteps=test_case["n_timesteps"],
                n_neurons=test_case["n_neurons"],
                **config["config"],
            )
            result["test_case"] = test_case["label"]
            case_results.append(result)

            print(f"  Time: {result['elapsed_time']:.2f}s")
            print(f"  Throughput: {result['throughput_mbps']:.1f} MB/s")
            print(f"  File size: {result['file_size_mb']:.1f} MB")
            print(f"  Compression ratio: {result['compression_ratio']:.1f}x")

        results.extend(case_results)

    return results


def plot_results(results):
    """Create visualization of benchmark results."""
    # Group by test case
    test_cases = {}
    for result in results:
        case = result["test_case"]
        if case not in test_cases:
            test_cases[case] = []
        test_cases[case].append(result)

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Performance Enhancement Benchmarks", fontsize=16)

    for idx, (case, case_results) in enumerate(test_cases.items()):
        row = idx // 2
        col = idx % 2

        if idx < 4:  # Only plot first 3 test cases
            ax = axes[row, col] if idx < 3 else axes[1, 1]

            names = [r["name"] for r in case_results]
            throughputs = [r["throughput_mbps"] for r in case_results]

            bars = ax.bar(range(len(names)), throughputs)
            ax.set_title(f"{case}")
            ax.set_ylabel("Throughput (MB/s)")
            ax.set_xticks(range(len(names)))
            ax.set_xticklabels(names, rotation=45, ha="right")

            # Color bars based on performance
            baseline = throughputs[0]
            for i, bar in enumerate(bars):
                if throughputs[i] > baseline * 1.1:
                    bar.set_color("green")
                elif throughputs[i] < baseline * 0.9:
                    bar.set_color("red")
                else:
                    bar.set_color("blue")

            # Add value labels
            for i, v in enumerate(throughputs):
                ax.text(i, v + 1, f"{v:.1f}", ha="center", va="bottom")

    # Summary plot
    ax = axes[1, 0]

    # Average improvement over baseline
    baseline_throughputs = {}
    improvements = {}

    for result in results:
        case = result["test_case"]
        name = result["name"]

        if name == "Baseline":
            baseline_throughputs[case] = result["throughput_mbps"]
        else:
            if name not in improvements:
                improvements[name] = []
            if case in baseline_throughputs:
                improvement = (result["throughput_mbps"] / baseline_throughputs[case] - 1) * 100
                improvements[name].append(improvement)

    # Plot average improvements
    names = list(improvements.keys())
    avg_improvements = [np.mean(improvements[name]) for name in names]

    bars = ax.bar(range(len(names)), avg_improvements)
    ax.set_title("Average Performance Improvement")
    ax.set_ylabel("Improvement over Baseline (%)")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)

    # Color bars
    for i, bar in enumerate(bars):
        if avg_improvements[i] > 0:
            bar.set_color("green")
        else:
            bar.set_color("red")

    plt.tight_layout()
    plt.savefig("enhancement_benchmarks.png", dpi=150, bbox_inches="tight")
    print("\nBenchmark plot saved to: enhancement_benchmarks.png")


def save_results(results):
    """Save benchmark results to JSON."""
    # Clean up results for JSON serialization
    clean_results = []
    for result in results:
        clean_result = result.copy()
        # Remove complex stats object
        if "stats" in clean_result:
            clean_result["stats"] = str(clean_result["stats"])
        clean_results.append(clean_result)

    with open("enhancement_benchmark_results.json", "w") as f:
        json.dump(clean_results, f, indent=2)

    print("Results saved to: enhancement_benchmark_results.json")


if __name__ == "__main__":
    print("=== Performance Enhancement Benchmarks ===\n")
    print("This will test various enhancement configurations...")
    print("Note: Async writes show benefits with larger data or overlapping computation\n")

    results = run_benchmarks()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Calculate average improvements
    baseline_avg = np.mean([r["throughput_mbps"] for r in results if r["name"] == "Baseline"])

    for config_name in [
        "Async Writes (4 workers)",
        "Adaptive Compression",
        "All Enhancements",
        "Ultra Performance",
    ]:
        config_results = [r for r in results if r["name"] == config_name]
        if config_results:
            avg_throughput = np.mean([r["throughput_mbps"] for r in config_results])
            improvement = (avg_throughput / baseline_avg - 1) * 100
            print(f"{config_name}: {avg_throughput:.1f} MB/s ({improvement:+.1f}%)")

    # Save results
    save_results(results)

    # Plot results
    try:
        plot_results(results)
    except ImportError:
        print("\nNote: Install matplotlib to see visualization of results")
