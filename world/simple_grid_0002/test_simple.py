#!/usr/bin/env python3
"""Simple test for JAX JIT compilation debugging."""

from world.simple_grid_0001.types import WorldConfig
from world.simple_grid_0002.world import SimpleGridWorld
from jax import random
import jax.numpy as jnp
import jax
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


def test_basic():
    """Basic test to check if world works at all."""
    print("Testing basic world functionality...")

    # Small world for testing
    config = WorldConfig(grid_size=10, n_rewards=2, max_timesteps=10)
    world = SimpleGridWorld(config)

    print(f"World created with config: {config}")

    # Test reset
    print("\nTesting reset...")
    key = random.PRNGKey(0)
    try:
        state, obs = world.reset(key)
        print(f"✓ Reset successful")
        print(f"  Agent position: {state.agent_pos}")
        print(f"  Reward positions shape: {state.reward_positions.shape}")
        print(f"  Initial gradient: {obs.gradient}")
    except Exception as e:
        print(f"✗ Reset failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test step
    print("\nTesting step...")
    try:
        result = world.step(state, 0, random.PRNGKey(1))
        print(f"✓ Step successful")
        print(f"  New position: {result.state.agent_pos}")
        print(f"  Reward: {result.reward}")
    except Exception as e:
        print(f"✗ Step failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n✅ Basic tests passed!")
    return True


def test_jit_methods():
    """Test individual JIT compiled methods."""
    print("\n\nTesting individual JIT methods...")

    config = WorldConfig(grid_size=10, n_rewards=2, max_timesteps=10)
    world = SimpleGridWorld(config)

    # Test _generate_rewards_static
    print("\n1. Testing _generate_rewards_static...")
    try:
        key = random.PRNGKey(0)
        agent_pos = (5, 5)
        rewards = world._generate_rewards_static(key, agent_pos)
        print(f"✓ Generated rewards shape: {rewards.shape}")
        print(f"  Rewards: {rewards}")
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()

    # Test _calculate_distances
    print("\n2. Testing _calculate_distances...")
    try:
        pos = jnp.array([5, 5])
        positions = jnp.array([[0, 0], [9, 9]])
        distances = world._calculate_distances(pos, positions)
        print(f"✓ Distances: {distances}")
    except Exception as e:
        print(f"✗ Failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if test_basic():
        test_jit_methods()
    else:
        print("\nBasic tests failed, skipping JIT tests.")
