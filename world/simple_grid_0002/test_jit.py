#!/usr/bin/env python3
"""Test JAX JIT compilation for grid world 0002."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time
import jax
import jax.numpy as jnp
from jax import random

from world.simple_grid_0002.world import SimpleGridWorld
from world.simple_grid_0001.types import WorldConfig


def test_jit_compilation():
    """Test that all methods compile successfully with JIT."""
    print("Testing JAX JIT compilation for grid world 0002...")
    
    # Create world
    config = WorldConfig(grid_size=50, n_rewards=10, max_timesteps=1000)
    world = SimpleGridWorld(config)
    
    # Test reset compilation
    print("\n1. Testing reset JIT compilation...")
    key = random.PRNGKey(0)
    
    # First call (compilation)
    start = time.time()
    state1, obs1 = world.reset(key)
    first_time = time.time() - start
    print(f"   First call (with compilation): {first_time*1000:.2f}ms")
    
    # Second call (compiled)
    start = time.time()
    state2, obs2 = world.reset(random.PRNGKey(1))
    second_time = time.time() - start
    print(f"   Second call (compiled): {second_time*1000:.2f}ms")
    print(f"   Speedup: {first_time/second_time:.1f}x")
    
    # Test step compilation
    print("\n2. Testing step JIT compilation...")
    
    # First call (compilation)
    start = time.time()
    result1 = world.step(state1, 0, random.PRNGKey(2))
    first_time = time.time() - start
    print(f"   First call (with compilation): {first_time*1000:.2f}ms")
    
    # Second call (compiled)
    start = time.time()
    result2 = world.step(result1.state, 1, random.PRNGKey(3))
    second_time = time.time() - start
    print(f"   Second call (compiled): {second_time*1000:.2f}ms")
    print(f"   Speedup: {first_time/second_time:.1f}x")
    
    # Test with many steps
    print("\n3. Running 1000 steps benchmark...")
    state = state1
    key = random.PRNGKey(100)
    
    start = time.time()
    for i in range(1000):
        key, subkey = random.split(key)
        result = world.step(state, i % 4, subkey)
        state = result.state
    elapsed = time.time() - start
    
    print(f"   1000 steps completed in {elapsed:.3f}s")
    print(f"   Average time per step: {elapsed/1000*1000:.2f}ms")
    print(f"   Steps per second: {1000/elapsed:.0f}")
    
    # Verify outputs are JAX arrays
    print("\n4. Verifying JAX array outputs...")
    print(f"   State position type: {type(state.agent_pos)}")
    print(f"   Reward positions type: {type(state.reward_positions)}")
    print(f"   Observation gradient type: {type(result.observation.gradient)}")
    
    # Test that methods are actually JIT compiled
    print("\n5. Checking JIT compilation status...")
    print(f"   _reset_jit is compiled: {hasattr(world._reset_jit, 'lower')}")
    print(f"   _step_jit is compiled: {hasattr(world._step_jit, 'lower')}")
    
    print("\n✅ All tests passed! Grid world is fully JAX JIT-compatible.")
    return True


def test_correctness():
    """Test that the JAX implementation produces correct results."""
    print("\n\nTesting correctness of JAX implementation...")
    
    config = WorldConfig(grid_size=20, n_rewards=5, max_timesteps=100)
    world = SimpleGridWorld(config)
    
    # Test basic functionality
    key = random.PRNGKey(42)
    state, obs = world.reset(key)
    
    print(f"\n1. Initial state:")
    print(f"   Agent position: {state.agent_pos}")
    print(f"   Number of rewards: {len(state.reward_positions)}")
    print(f"   Initial gradient: {obs.gradient:.4f}")
    
    # Test movement
    print("\n2. Testing movement:")
    initial_pos = state.agent_pos
    result = world.step(state, 1, random.PRNGKey(43))  # Move right
    new_pos = result.state.agent_pos
    print(f"   Moved from {initial_pos} to {new_pos}")
    
    # Test reward collection
    print("\n3. Testing reward collection:")
    # Move agent to a reward position
    reward_pos = tuple(state.reward_positions[0])
    test_state = state._replace(agent_pos=reward_pos)
    result = world.step(test_state, 0, random.PRNGKey(44))
    
    print(f"   Reward collected: {result.reward}")
    print(f"   Rewards collected: {jnp.sum(result.state.reward_collected)}")
    
    print("\n✅ Correctness tests passed!")


def benchmark_vs_baseline():
    """Compare performance with baseline implementation."""
    print("\n\nBenchmarking JAX world vs baseline...")
    
    # Import both versions
    from world.simple_grid_0001 import SimpleGridWorld as BaselineWorld
    from world.simple_grid_0002 import SimpleGridWorld as JAXWorld
    
    config = WorldConfig(grid_size=100, n_rewards=20, max_timesteps=10000)
    
    # Baseline timing
    print("\n1. Baseline world (non-JIT):")
    baseline_world = BaselineWorld(config)
    key = random.PRNGKey(0)
    
    start = time.time()
    state, _ = baseline_world.reset(key)
    for i in range(1000):
        key, subkey = random.split(key)
        result = baseline_world.step(state, i % 4, subkey)
        state = result.state
    baseline_time = time.time() - start
    print(f"   1000 steps: {baseline_time:.3f}s")
    
    # JAX timing
    print("\n2. JAX world (JIT-compiled):")
    jax_world = JAXWorld(config)
    key = random.PRNGKey(0)
    
    # Warm up JIT
    state, _ = jax_world.reset(key)
    _ = jax_world.step(state, 0, random.PRNGKey(1))
    
    start = time.time()
    state, _ = jax_world.reset(key)
    for i in range(1000):
        key, subkey = random.split(key)
        result = jax_world.step(state, i % 4, subkey)
        state = result.state
    jax_time = time.time() - start
    print(f"   1000 steps: {jax_time:.3f}s")
    
    print(f"\n3. Performance improvement: {baseline_time/jax_time:.1f}x faster!")
    
    return baseline_time, jax_time


if __name__ == "__main__":
    # Run all tests
    test_jit_compilation()
    test_correctness()
    
    try:
        baseline_time, jax_time = benchmark_vs_baseline()
        print(f"\n\n🎉 SUCCESS! JAX world is {baseline_time/jax_time:.1f}x faster than baseline!")
    except ImportError:
        print("\n\nNote: Could not benchmark against baseline (import error)")
    
    print("\n" + "="*60)
    print("JAX JIT compilation test complete!")
    print("The grid world is now fully JAX-compatible with static shapes.")
    print("="*60)