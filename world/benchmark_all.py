#!/usr/bin/env python3
"""Comprehensive benchmark comparing all three grid world implementations."""

import time

import jax
import numpy as np
from jax import random
from simple_grid_0001 import SimpleGridWorld as WorldV1
from simple_grid_0001.types import WorldConfig
from simple_grid_0002 import SimpleGridWorld as WorldV2
from simple_grid_0003 import SimpleGridWorld as WorldV3


def benchmark_world(world_class, config, n_steps=10000, warmup_steps=100):
    """Benchmark a world implementation with proper warmup."""
    world = world_class(config)
    key = random.PRNGKey(42)

    # Warmup for JIT compilation
    print("  Warming up...", end="", flush=True)
    state, _ = world.reset(key)
    for i in range(warmup_steps):
        key, subkey = random.split(key)
        result = world.step(state, i % 4, subkey)
        state = result.state
    print(" done")

    # Reset for actual benchmark
    key = random.PRNGKey(42)

    # Time reset (average of 100 resets)
    reset_times = []
    for i in range(100):
        key, subkey = random.split(key)
        reset_start = time.perf_counter()
        state, obs = world.reset(subkey)
        reset_time = time.perf_counter() - reset_start
        reset_times.append(reset_time)
    avg_reset_time = np.mean(reset_times)

    # Time steps
    step_times = []
    for i in range(n_steps):
        key, subkey = random.split(key)
        step_start = time.perf_counter()
        result = world.step(state, i % 4, subkey)
        step_time = time.perf_counter() - step_start
        step_times.append(step_time)
        state = result.state

    # Calculate statistics
    step_times_arr = np.array(step_times)
    avg_step_time = np.mean(step_times_arr)
    p50_step_time = np.percentile(step_times_arr, 50)
    p95_step_time = np.percentile(step_times_arr, 95)
    p99_step_time = np.percentile(step_times_arr, 99)

    return {
        "reset_time": avg_reset_time,
        "avg_step_time": avg_step_time,
        "p50_step_time": p50_step_time,
        "p95_step_time": p95_step_time,
        "p99_step_time": p99_step_time,
        "steps_per_second": 1.0 / avg_step_time,
        "total_reward": state.total_reward,
    }


def format_time(seconds):
    """Format time in appropriate units."""
    if seconds < 1e-6:
        return f"{seconds * 1e9:.2f}ns"
    elif seconds < 1e-3:
        return f"{seconds * 1e6:.2f}μs"
    else:
        return f"{seconds * 1e3:.2f}ms"


def main():
    """Run comprehensive benchmarks."""
    print("🏃 Grid World Performance Benchmark - All Versions")
    print("=" * 80)
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print()

    # Test configurations
    configs = [
        ("Small", WorldConfig(grid_size=50, n_rewards=10, max_timesteps=100000)),
        ("Medium", WorldConfig(grid_size=100, n_rewards=20, max_timesteps=100000)),
        ("Large", WorldConfig(grid_size=200, n_rewards=40, max_timesteps=100000)),
    ]

    for config_name, config in configs:
        print(
            f"\n📊 Configuration: {config_name} (Grid: {config.grid_size}x{config.grid_size}, Rewards: {config.n_rewards})"
        )
        print("-" * 80)

        # Benchmark all versions
        results = {}

        print("\nV1 - Baseline (simple_grid_0001):")
        results["v1"] = benchmark_world(WorldV1, config, n_steps=10000)
        print(f"  Reset time: {format_time(results['v1']['reset_time'])}")
        print(f"  Step time (avg): {format_time(results['v1']['avg_step_time'])}")
        print(f"  Steps/second: {results['v1']['steps_per_second']:,.0f}")

        print("\nV2 - JAX Optimized (simple_grid_0002):")
        results["v2"] = benchmark_world(WorldV2, config, n_steps=10000)
        print(f"  Reset time: {format_time(results['v2']['reset_time'])}")
        print(f"  Step time (avg): {format_time(results['v2']['avg_step_time'])}")
        print(f"  Steps/second: {results['v2']['steps_per_second']:,.0f}")

        print("\nV3 - Ultra Optimized (simple_grid_0003):")
        results["v3"] = benchmark_world(WorldV3, config, n_steps=10000)
        print(f"  Reset time: {format_time(results['v3']['reset_time'])}")
        print(f"  Step time (avg): {format_time(results['v3']['avg_step_time'])}")
        print(f"  Step time (p50): {format_time(results['v3']['p50_step_time'])}")
        print(f"  Step time (p95): {format_time(results['v3']['p95_step_time'])}")
        print(f"  Step time (p99): {format_time(results['v3']['p99_step_time'])}")
        print(f"  Steps/second: {results['v3']['steps_per_second']:,.0f}")

        # Calculate speedups
        print("\n🚀 Performance Comparison:")
        print("  V2 vs V1:")
        print(f"    Reset: {results['v1']['reset_time'] / results['v2']['reset_time']:.1f}x faster")
        print(
            f"    Steps: {results['v1']['avg_step_time'] / results['v2']['avg_step_time']:.1f}x faster"
        )

        print("  V3 vs V1:")
        print(f"    Reset: {results['v1']['reset_time'] / results['v3']['reset_time']:.1f}x faster")
        print(
            f"    Steps: {results['v1']['avg_step_time'] / results['v3']['avg_step_time']:.1f}x faster"
        )

        print("  V3 vs V2:")
        print(f"    Reset: {results['v2']['reset_time'] / results['v3']['reset_time']:.1f}x faster")
        print(
            f"    Steps: {results['v2']['avg_step_time'] / results['v3']['avg_step_time']:.1f}x faster"
        )

    print("\n" + "=" * 80)
    print("✅ Benchmark complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
