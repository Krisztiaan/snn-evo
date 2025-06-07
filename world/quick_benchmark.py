#!/usr/bin/env python3
"""Quick performance comparison of grid world implementations."""

import time
import jax
import jax.numpy as jnp
from jax import random

from simple_grid_0001 import SimpleGridWorld as WorldV1
from simple_grid_0002 import SimpleGridWorld as WorldV2
from simple_grid_0003 import SimpleGridWorld as WorldV3
from simple_grid_0001.types import WorldConfig


def quick_benchmark(world_class, config, n_steps=1000):
    """Quick benchmark."""
    world = world_class(config)
    key = random.PRNGKey(42)
    
    # Warmup
    state, _ = world.reset(key)
    for i in range(10):
        key, subkey = random.split(key)
        result = world.step(state, i % 4, subkey)
        state = result.state
    
    # Benchmark
    key = random.PRNGKey(42)
    
    # Time reset
    start = time.perf_counter()
    state, _ = world.reset(key)
    reset_time = time.perf_counter() - start
    
    # Time steps
    start = time.perf_counter()
    for i in range(n_steps):
        key, subkey = random.split(key)
        result = world.step(state, i % 4, subkey)
        state = result.state
    total_step_time = time.perf_counter() - start
    
    return {
        'reset_ms': reset_time * 1000,
        'step_us': (total_step_time / n_steps) * 1e6,
        'steps_per_sec': n_steps / total_step_time
    }


def main():
    print("🚀 Grid World Quick Performance Test")
    print("=" * 60)
    
    config = WorldConfig(grid_size=100, n_rewards=20, max_timesteps=10000)
    
    print("Benchmarking...")
    v1 = quick_benchmark(WorldV1, config)
    v2 = quick_benchmark(WorldV2, config)
    v3 = quick_benchmark(WorldV3, config)
    
    print("\n📊 Results (100x100 grid, 20 rewards, 1000 steps):")
    print("-" * 60)
    print(f"{'Version':<10} {'Reset (ms)':<12} {'Step (μs)':<12} {'Steps/sec':<12}")
    print("-" * 60)
    print(f"{'V1':<10} {v1['reset_ms']:<12.2f} {v1['step_us']:<12.1f} {v1['steps_per_sec']:<12,.0f}")
    print(f"{'V2':<10} {v2['reset_ms']:<12.2f} {v2['step_us']:<12.1f} {v2['steps_per_sec']:<12,.0f}")
    print(f"{'V3':<10} {v3['reset_ms']:<12.2f} {v3['step_us']:<12.1f} {v3['steps_per_sec']:<12,.0f}")
    
    print("\n🎯 Speedup:")
    print(f"  V2 vs V1: {v1['step_us']/v2['step_us']:.1f}x")
    print(f"  V3 vs V1: {v1['step_us']/v3['step_us']:.1f}x")
    print(f"  V3 vs V2: {v2['step_us']/v3['step_us']:.1f}x")


if __name__ == "__main__":
    main()