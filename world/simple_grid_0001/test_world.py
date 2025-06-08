# keywords: [grid world test, simple verification]
"""Quick tests to verify the grid world works correctly."""

import jax
import jax.numpy as jnp
from jax import random

from world.simple_grid_0001 import SimpleGridWorld, WorldConfig


def test_basic_functionality():
    """Test basic world operations."""
    print("Testing basic functionality...")

    config = WorldConfig(grid_size=10, n_rewards=5, max_timesteps=100)
    world = SimpleGridWorld(config)

    # Test reset
    key = random.PRNGKey(0)
    state, obs = world.reset(key)

    assert state.agent_pos == (5, 5), "Agent should start at center"
    assert state.reward_positions.shape == (
        5, 2), "Should have 5 reward positions"
    assert jnp.all(
        ~state.reward_collected), "No rewards should be collected initially"
    assert state.total_reward == 0.0, "Initial reward should be 0"
    assert state.timestep == 0, "Initial timestep should be 0"
    assert 0 <= obs.gradient <= 1, "Gradient should be in [0, 1]"

    print("✓ Reset works correctly")

    # Test step
    result = world.step(state, action=0)  # Move up
    assert result.state.timestep == 1, "Timestep should increment"
    assert result.state.agent_pos == (5, 4), "Agent should move up"

    print("✓ Step works correctly")

    # Test wrapping (toroidal)
    state_at_edge = state._replace(agent_pos=(0, 0))
    result = world.step(state_at_edge, action=3)  # Move left
    assert result.state.agent_pos == (9, 0), "Should wrap around horizontally"

    result = world.step(state_at_edge, action=0)  # Move up
    assert result.state.agent_pos == (0, 9), "Should wrap around vertically"

    print("✓ Toroidal wrapping works correctly")

    # Test reward collection with respawn
    # Place agent one step away from a reward position
    reward_pos = state.reward_positions[0]
    state_near_reward = state._replace(agent_pos=(
        int(reward_pos[0]), int(reward_pos[1]) + 1))

    # Step into the reward
    result = world.step(state_near_reward, action=0)  # Move up

    assert result.reward >= config.reward_value, "Should collect reward when stepping into position"
    assert result.state.total_reward > 0, "Total reward should increase"

    # Check that reward was respawned at a different position
    old_pos = state.reward_positions[0]
    new_pos = result.state.reward_positions[0]
    assert not jnp.array_equal(
        old_pos, new_pos), "Reward should respawn at different position"

    # All rewards should still be uncollected (because of respawn)
    assert jnp.all(
        ~result.state.reward_collected), "All rewards should be uncollected after respawn"

    print("✓ Reward collection and respawn works correctly")

    print("\nAll tests passed! ✨")


def test_jax_compilation():
    """Test that JAX compilation works."""
    print("\nTesting JAX compilation...")

    config = WorldConfig(grid_size=20, n_rewards=10)
    world = SimpleGridWorld(config)

    key = random.PRNGKey(42)
    state, _ = world.reset(key)

    # Time uncompiled version
    import time
    start = time.time()
    for _ in range(100):
        result = world.step(state, 0)
        state = result.state
    uncompiled_time = time.time() - start

    # Force compilation and time compiled version
    state, _ = world.reset(key)
    compiled_step = jax.jit(world.step)

    # Warm up
    _ = compiled_step(state, 0)

    start = time.time()
    for _ in range(100):
        result = compiled_step(state, 0)
        state = result.state
    compiled_time = time.time() - start

    print(f"Uncompiled: {uncompiled_time:.4f}s")
    print(f"Compiled: {compiled_time:.4f}s")
    print(f"Speedup: {uncompiled_time/compiled_time:.1f}x")

    print("✓ JAX compilation works")


if __name__ == "__main__":
    test_basic_functionality()
    test_jax_compilation()
