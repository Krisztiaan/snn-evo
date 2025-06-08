#!/usr/bin/env python3
"""Benchmark JAX-optimized grid world performance."""

from world.simple_grid_0001.types import WorldConfig
from world.simple_grid_0002 import SimpleGridWorld as JAXWorld
from world.simple_grid_0001 import SimpleGridWorld as BaselineWorld
import time
import jax
import jax.numpy as jnp
from jax import random

# Set up imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


def benchmark_world(world_class, config, n_steps=1000, warmup=True):
    """Benchmark a world implementation."""
    world = world_class(config)
    key = random.PRNGKey(42)

    # Warmup for JIT compilation
    if warmup:
        state, _ = world.reset(key)
        for i in range(10):
            key, subkey = random.split(key)
            result = world.step(state, i % 4, subkey)
            state = result.state

    # Reset for actual benchmark
    key = random.PRNGKey(42)

    # Time reset
    reset_start = time.time()
    state, obs = world.reset(key)
    reset_time = time.time() - reset_start

    # Time steps
    step_times = []
    for i in range(n_steps):
        key, subkey = random.split(key)
        step_start = time.time()
        result = world.step(state, i % 4, subkey)
        step_time = time.time() - step_start
        step_times.append(step_time)
        state = result.state

    avg_step_time = sum(step_times) / len(step_times)
    total_time = reset_time + sum(step_times)

    return {
        'reset_time': reset_time,
        'avg_step_time': avg_step_time,
        'total_time': total_time,
        'steps_per_second': 1.0 / avg_step_time
    }


def main():
    """Run comprehensive benchmarks."""
    print("JAX Grid World Performance Benchmark")
    print("=" * 60)

    # Test different world sizes
    sizes = [50, 100, 200]
    n_rewards_list = [10, 20, 40]

    for size, n_rewards in zip(sizes, n_rewards_list):
        print(f"\n📊 Grid Size: {size}x{size}, Rewards: {n_rewards}")
        print("-" * 50)

        config = WorldConfig(
            grid_size=size,
            n_rewards=n_rewards,
            max_timesteps=10000
        )

        # Benchmark baseline
        print("Baseline world:")
        baseline_results = benchmark_world(
            BaselineWorld, config, n_steps=1000, warmup=False)
        print(f"  Reset time: {baseline_results['reset_time']*1000:.2f}ms")
        print(
            f"  Avg step time: {baseline_results['avg_step_time']*1000:.2f}ms")
        print(f"  Steps/second: {baseline_results['steps_per_second']:.0f}")

        # Benchmark JAX
        print("\nJAX world (JIT compiled):")
        jax_results = benchmark_world(
            JAXWorld, config, n_steps=1000, warmup=True)
        print(f"  Reset time: {jax_results['reset_time']*1000:.2f}ms")
        print(f"  Avg step time: {jax_results['avg_step_time']*1000:.2f}ms")
        print(f"  Steps/second: {jax_results['steps_per_second']:.0f}")

        # Calculate speedup
        reset_speedup = baseline_results['reset_time'] / \
            jax_results['reset_time']
        step_speedup = baseline_results['avg_step_time'] / \
            jax_results['avg_step_time']
        total_speedup = baseline_results['total_time'] / \
            jax_results['total_time']

        print(f"\n🚀 Speedup:")
        print(f"  Reset: {reset_speedup:.1f}x faster")
        print(f"  Steps: {step_speedup:.1f}x faster")
        print(f"  Total: {total_speedup:.1f}x faster")

    # Test JIT compilation overhead
    print("\n\n📈 JIT Compilation Analysis")
    print("-" * 50)

    config = WorldConfig(grid_size=100, n_rewards=20)
    world = JAXWorld(config)
    key = random.PRNGKey(0)

    # First call (includes compilation)
    start = time.time()
    state, _ = world.reset(key)
    first_reset = time.time() - start

    start = time.time()
    _ = world.step(state, 0, random.PRNGKey(1))
    first_step = time.time() - start

    # Second call (already compiled)
    start = time.time()
    state, _ = world.reset(random.PRNGKey(2))
    second_reset = time.time() - start

    start = time.time()
    _ = world.step(state, 0, random.PRNGKey(3))
    second_step = time.time() - start

    print(f"Reset - First call (with JIT): {first_reset*1000:.2f}ms")
    print(f"Reset - Second call (compiled): {second_reset*1000:.2f}ms")
    print(f"Step - First call (with JIT): {first_step*1000:.2f}ms")
    print(f"Step - Second call (compiled): {second_step*1000:.2f}ms")

    print(f"\nJIT compilation overhead:")
    print(f"  Reset: {(first_reset - second_reset)*1000:.2f}ms")
    print(f"  Step: {(first_step - second_step)*1000:.2f}ms")

    # Memory efficiency test
    print("\n\n💾 Memory Efficiency")
    print("-" * 50)

    import psutil
    import gc

    process = psutil.Process()

    # Baseline memory
    gc.collect()
    baseline_mem = process.memory_info().rss / 1024 / 1024  # MB

    # Create large world
    big_config = WorldConfig(grid_size=500, n_rewards=100)
    baseline_world = BaselineWorld(big_config)
    baseline_world_mem = process.memory_info().rss / 1024 / 1024 - baseline_mem

    # JAX world
    gc.collect()
    jax_world = JAXWorld(big_config)
    jax_world_mem = process.memory_info().rss / 1024 / 1024 - \
        baseline_mem - baseline_world_mem

    print(f"Baseline world memory: {baseline_world_mem:.1f} MB")
    print(f"JAX world memory: {jax_world_mem:.1f} MB")
    print(f"Memory efficiency: {baseline_world_mem/jax_world_mem:.1f}x")

    print("\n" + "=" * 60)
    print("✅ Benchmark complete!")
    print("=" * 60)


if __name__ == "__main__":
    # Check JAX backend
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print()

    main()
