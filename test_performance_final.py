#!/usr/bin/env python3
"""Final performance comparison between baseline and JAX worlds."""

import time
from jax import random

from world.simple_grid_0001 import SimpleGridWorld as BaselineWorld
from world.simple_grid_0002 import SimpleGridWorld as JAXWorld
from world.simple_grid_0001.types import WorldConfig

# Test configuration
config = WorldConfig(grid_size=100, n_rewards=20, max_timesteps=10000)
n_steps = 100  # Reduced for faster testing

print("Performance Comparison: Baseline vs JAX World")
print("=" * 60)
print(f"Configuration: {config.grid_size}x{config.grid_size} grid, {config.n_rewards} rewards")
print(f"Testing with {n_steps} steps")

# 1. Baseline world
print("\n1. Baseline World Performance:")
baseline_world = BaselineWorld(config)
key = random.PRNGKey(42)

# Time reset
reset_start = time.time()
state, obs = baseline_world.reset(key)
baseline_reset_time = time.time() - reset_start

# Time steps
step_times = []
for i in range(n_steps):
    key, subkey = random.split(key)
    start = time.time()
    result = baseline_world.step(state, i % 4, subkey)
    step_times.append(time.time() - start)
    state = result.state

baseline_avg_step = sum(step_times) / len(step_times)
baseline_total = baseline_reset_time + sum(step_times)

print(f"   Reset time: {baseline_reset_time*1000:.1f}ms")
print(f"   Average step time: {baseline_avg_step*1000:.2f}ms")
print(f"   Total time ({n_steps} steps): {baseline_total:.3f}s")
print(f"   Steps per second: {n_steps/sum(step_times):.0f}")

# 2. JAX world
print("\n2. JAX World Performance:")
jax_world = JAXWorld(config)
key = random.PRNGKey(42)

# Warmup JIT
print("   Warming up JIT compilation...")
warmup_start = time.time()
state, obs = jax_world.reset(key)
for i in range(5):
    key, subkey = random.split(key)
    result = jax_world.step(state, i % 4, subkey)
    state = result.state
print(f"   JIT warmup completed in {time.time() - warmup_start:.2f}s")

# Actual timing
key = random.PRNGKey(42)

# Time reset
reset_start = time.time()
state, obs = jax_world.reset(key)
jax_reset_time = time.time() - reset_start

# Time steps
step_times = []
for i in range(n_steps):
    key, subkey = random.split(key)
    start = time.time()
    result = jax_world.step(state, i % 4, subkey)
    step_times.append(time.time() - start)
    state = result.state

jax_avg_step = sum(step_times) / len(step_times)
jax_total = jax_reset_time + sum(step_times)

print(f"   Reset time: {jax_reset_time*1000:.1f}ms")
print(f"   Average step time: {jax_avg_step*1000:.2f}ms")
print(f"   Total time ({n_steps} steps): {jax_total:.3f}s")
print(f"   Steps per second: {n_steps/sum(step_times):.0f}")

# 3. Performance comparison
print("\n3. Performance Improvement:")
print(f"   Reset speedup: {baseline_reset_time/jax_reset_time:.1f}x")
print(f"   Step speedup: {baseline_avg_step/jax_avg_step:.1f}x")
print(f"   Overall speedup: {baseline_total/jax_total:.1f}x")

# 4. Projected performance for full episodes
print("\n4. Projected Episode Performance (10,000 steps):")
baseline_episode = baseline_avg_step * 10000
jax_episode = jax_avg_step * 10000
print(f"   Baseline: {baseline_episode:.1f}s ({baseline_episode/60:.1f} minutes)")
print(f"   JAX: {jax_episode:.1f}s ({jax_episode/60:.1f} minutes)")
print(f"   Time saved per episode: {baseline_episode - jax_episode:.1f}s")
print(f"   Speedup for full episode: {baseline_episode/jax_episode:.1f}x")

print("\n✅ Performance test complete!")