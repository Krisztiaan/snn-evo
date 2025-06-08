#!/usr/bin/env python3
"""Test correctness of ultra-optimized grid world implementation."""

from simple_grid_0001.types import WorldConfig
from simple_grid_0003 import SimpleGridWorld
from jax import random
import jax.numpy as jnp
import jax
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def test_basic_functionality():
    """Test basic world functionality."""
    print("Testing basic functionality...")

    config = WorldConfig(grid_size=20, n_rewards=5, max_timesteps=1000)
    world = SimpleGridWorld(config)

    # Test reset
    key = random.PRNGKey(42)
    state, obs = world.reset(key)

    assert state.agent_pos == (10, 10), f"Agent should start at center, got {state.agent_pos}"
    assert len(state.reward_positions) == 5, (
        f"Should have 5 rewards, got {len(state.reward_positions)}"
    )
    assert jnp.sum(state.reward_collected) == 0, "No rewards should be collected initially"
    assert state.total_reward == 0.0, "Initial reward should be 0"
    assert state.timestep == 0, "Initial timestep should be 0"
    assert 0 <= obs.gradient <= 1, f"Gradient should be in [0,1], got {obs.gradient}"

    print("✓ Reset works correctly")

    # Test movement
    initial_pos = state.agent_pos
    result = world.step(state, 1, random.PRNGKey(43))  # Move right
    new_pos = result.state.agent_pos

    assert new_pos == (initial_pos[0] + 1, initial_pos[1]), (
        f"Agent should move right, got {new_pos}"
    )
    assert result.state.timestep == 1, "Timestep should increment"

    print("✓ Movement works correctly")

    # Test toroidal wrapping
    edge_state = state._replace(agent_pos=(19, 10))
    result = world.step(edge_state, 1, random.PRNGKey(44))  # Move right at edge
    assert result.state.agent_pos == (0, 10), f"Should wrap to x=0, got {result.state.agent_pos}"

    edge_state = state._replace(agent_pos=(10, 0))
    result = world.step(edge_state, 0, random.PRNGKey(45))  # Move up at edge
    assert result.state.agent_pos == (10, 19), f"Should wrap to y=19, got {result.state.agent_pos}"

    print("✓ Toroidal wrapping works correctly")

    return True


def test_reward_collection():
    """Test reward collection and respawning."""
    print("\nTesting reward collection...")

    config = WorldConfig(grid_size=20, n_rewards=3, max_timesteps=1000)
    world = SimpleGridWorld(config)

    key = random.PRNGKey(42)
    state, _ = world.reset(key)

    # Move agent to first reward position
    reward_pos = tuple(state.reward_positions[0])
    test_state = state._replace(agent_pos=reward_pos)

    result = world.step(test_state, 0, random.PRNGKey(43))

    assert result.reward == config.reward_value, (
        f"Should get reward value {config.reward_value}, got {result.reward}"
    )
    assert result.state.total_reward == config.reward_value, "Total reward should update"
    assert ~result.state.reward_collected[0], (
        "Collected reward should respawn (collected flag reset)"
    )

    # Verify reward respawned at different position
    new_reward_pos = tuple(result.state.reward_positions[0])
    assert new_reward_pos != reward_pos, (
        f"Reward should respawn at different position, still at {new_reward_pos}"
    )

    print("✓ Reward collection and respawning work correctly")

    # Test proximity reward
    # Place agent near but not on reward
    near_pos = (reward_pos[0] + 2, reward_pos[1])  # 2 cells away
    test_state = state._replace(agent_pos=near_pos)
    result = world.step(test_state, 0, random.PRNGKey(44))

    assert result.reward == config.proximity_reward, (
        f"Should get proximity reward {config.proximity_reward}, got {result.reward}"
    )

    print("✓ Proximity rewards work correctly")

    return True


def test_observation_gradient():
    """Test observation gradient calculation."""
    print("\nTesting observation gradient...")

    config = WorldConfig(grid_size=20, n_rewards=3, max_timesteps=1000)
    world = SimpleGridWorld(config)

    key = random.PRNGKey(42)
    state, obs = world.reset(key)

    # Test gradient decreases with distance
    gradients = []
    positions = [
        state.agent_pos,  # Starting position
        (state.agent_pos[0] + 2, state.agent_pos[1]),  # Move away
        (state.agent_pos[0] + 4, state.agent_pos[1]),  # Further
    ]

    for pos in positions:
        test_state = state._replace(agent_pos=pos)
        _, test_obs = world.reset(key)  # Get fresh observation
        # Manually calculate observation for test state
        agent_packed = pos[0] * world.grid_size + pos[1]
        reward_packed = (
            state.reward_positions[:, 0] * world.grid_size + state.reward_positions[:, 1]
        )
        test_obs = world._get_observation_fast(agent_packed, reward_packed, state.reward_collected)
        gradients.append(test_obs.gradient)

    # Gradients should decrease as we move away
    assert gradients[0] >= gradients[1] >= gradients[2], (
        f"Gradients should decrease with distance: {gradients}"
    )

    print("✓ Observation gradient calculation works correctly")

    return True


def test_performance_consistency():
    """Test that optimizations don't break consistency."""
    print("\nTesting performance consistency...")

    # Compare with baseline implementation
    from simple_grid_0001 import SimpleGridWorld as BaselineWorld

    config = WorldConfig(grid_size=50, n_rewards=10, max_timesteps=1000)
    world_v3 = SimpleGridWorld(config)
    world_v1 = BaselineWorld(config)

    # Run same sequence on both
    key = random.PRNGKey(123)
    actions = [0, 1, 1, 2, 2, 3, 3, 0, 1, 2]  # Fixed action sequence

    # V3 trajectory
    state_v3, _ = world_v3.reset(key)
    rewards_v3 = []
    positions_v3 = [state_v3.agent_pos]

    for i, action in enumerate(actions):
        result = world_v3.step(state_v3, action, random.PRNGKey(1000 + i))
        state_v3 = result.state
        rewards_v3.append(result.reward)
        positions_v3.append(state_v3.agent_pos)

    # V1 trajectory
    state_v1, _ = world_v1.reset(key)
    rewards_v1 = []
    positions_v1 = [state_v1.agent_pos]

    for i, action in enumerate(actions):
        result = world_v1.step(state_v1, action, random.PRNGKey(1000 + i))
        state_v1 = result.state
        rewards_v1.append(result.reward)
        positions_v1.append(state_v1.agent_pos)

    # Compare trajectories
    for i, (pos_v3, pos_v1) in enumerate(zip(positions_v3, positions_v1)):
        assert pos_v3 == pos_v1, f"Position mismatch at step {i}: V3={pos_v3}, V1={pos_v1}"

    print("✓ Movement consistency verified")

    # Note: Rewards might differ due to different spawn algorithms
    print(f"  V3 total reward: {sum(rewards_v3):.2f}")
    print(f"  V1 total reward: {sum(rewards_v1):.2f}")

    return True


def test_edge_cases():
    """Test edge cases and boundary conditions."""
    print("\nTesting edge cases...")

    # Test with minimal world
    config = WorldConfig(grid_size=5, n_rewards=1, max_timesteps=100)
    world = SimpleGridWorld(config)

    key = random.PRNGKey(42)
    state, obs = world.reset(key)

    # Run many steps to ensure stability
    for i in range(100):
        key, subkey = random.split(key)
        result = world.step(state, i % 4, subkey)
        state = result.state
        assert jnp.isfinite(result.reward), f"Reward became non-finite at step {i}"
        assert jnp.isfinite(result.observation.gradient), f"Gradient became non-finite at step {i}"

    print("✓ Minimal world configuration works")

    # Test with maximum rewards
    config = WorldConfig(grid_size=10, n_rewards=20, max_timesteps=100)
    world = SimpleGridWorld(config)
    state, _ = world.reset(key)

    # Verify all rewards are unique positions
    positions_set = set()
    for pos in state.reward_positions:
        pos_tuple = (int(pos[0]), int(pos[1]))  # Convert to Python ints
        assert pos_tuple not in positions_set, f"Duplicate reward position: {pos_tuple}"
        positions_set.add(pos_tuple)

    print("✓ Many rewards configuration works")

    return True


def main():
    """Run all tests."""
    print("🧪 Testing SimpleGridWorld V3 (Ultra-Optimized)")
    print("=" * 60)

    tests = [
        test_basic_functionality,
        test_reward_collection,
        test_observation_gradient,
        test_performance_consistency,
        test_edge_cases,
    ]

    passed = 0
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")

    print("\n" + "=" * 60)
    print(f"✅ {passed}/{len(tests)} tests passed")

    if passed == len(tests):
        print("All tests passed! The ultra-optimized implementation is correct.")
    else:
        print("Some tests failed. Please fix the implementation.")

    return passed == len(tests)


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
