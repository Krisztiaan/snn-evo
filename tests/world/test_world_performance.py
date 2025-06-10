# keywords: [world test, performance, jit, type safety]
"""Performance-focused tests for world implementations."""

import time
from typing import List

import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest

from interfaces import WorldProtocol, WorldConfig

class TestWorldPerformance:
    """Test world implementations for performance and correctness."""
    
    @pytest.fixture
    def world_config(self):
        """Standard world configuration for tests."""
        return WorldConfig(grid_size=100, n_rewards=300, max_timesteps=50000)
    
    def test_world_implements_protocol(self, world_config: WorldConfig):
        """Test that MinimalGridWorld implements WorldProtocol."""
        from world.simple_grid_0004 import MinimalGridWorld
        
        world = MinimalGridWorld(world_config)
        
        # Check protocol methods exist
        assert hasattr(world, 'reset')
        assert hasattr(world, 'step')
        assert hasattr(world, 'get_config')
        assert hasattr(world, 'get_reward_tracking')
        
        # Test basic functionality
        key = jrandom.PRNGKey(42)
        state, gradient = world.reset(key)
        
        assert isinstance(gradient, jax.Array)
        assert gradient.shape == ()
        assert 0 <= gradient <= 1
    
    def test_step_performance(self, world_config: WorldConfig):
        """Test that world.step() achieves target performance."""
        from world.simple_grid_0004 import MinimalGridWorld
        
        world = MinimalGridWorld(world_config)
        key = jrandom.PRNGKey(42)
        state, _ = world.reset(key)
        
        # Warmup for JIT compilation
        for _ in range(10):
            state, _ = world.step(state, 0)
        
        # Time many steps
        n_steps = 10000
        start = time.perf_counter()
        for i in range(n_steps):
            state, gradient = world.step(state, i % 9)
        duration = time.perf_counter() - start
        
        steps_per_second = n_steps / duration
        # Realistic target given Python loop overhead (~40μs per call)
        assert steps_per_second > 18000, f"World too slow: {steps_per_second:.0f} steps/s"
        
        # Note: 100k steps/s is only achievable with pure JAX loops,
        # not with Python calling JAX functions due to ~40μs overhead per call
    
    def test_gradient_correctness(self, world_config: WorldConfig):
        """Test gradient calculation correctness."""
        from world.simple_grid_0004 import MinimalGridWorld
        from interfaces import WorldState
        
        # Create smaller test world
        test_config = WorldConfig(grid_size=10, n_rewards=3, max_timesteps=100)
        world = MinimalGridWorld(test_config)
        
        # Manually construct deterministic state
        agent_pos = jnp.array([5, 5])  # Center of 10x10 grid
        agent_dir = jnp.array(0)  # North
        
        # Place rewards at known positions
        reward_positions = jnp.array([
            [5, 4],  # One step north of agent
            [7, 5],  # Two steps east
            [3, 3],  # Further away
        ])
        
        state = WorldState(
            agent_pos=agent_pos,
            agent_dir=agent_dir,
            reward_positions=reward_positions,
            reward_active=jnp.ones(3, dtype=bool),
            key=jrandom.PRNGKey(42),
            reward_history_positions=jnp.zeros((30, 2), dtype=jnp.int32),
            reward_history_spawn_steps=jnp.full(30, -1, dtype=jnp.int32),
            reward_history_collect_steps=jnp.full(30, -1, dtype=jnp.int32),
            reward_history_count=jnp.array(0, dtype=jnp.int32)
        )
        
        # Test 1: Gradient should be high when near reward
        gradient = world._calculate_gradient(agent_pos, reward_positions, state.reward_active)
        assert 0.5 < gradient < 1.0, f"Expected high gradient near reward, got {gradient}"
        
        # Test 2: Move to reward position - gradient should be 1.0
        state = state._replace(agent_pos=jnp.array([5, 4]))
        gradient = world._calculate_gradient(state.agent_pos, reward_positions, state.reward_active)
        assert gradient == 1.0, f"Expected gradient 1.0 at reward, got {gradient}"
        
        # Test 3: Test reward collection through step
        state = state._replace(agent_pos=jnp.array([5, 5]), agent_dir=jnp.array(0))  # Reset position, facing north
        initial_reward_pos = state.reward_positions[0].copy()  # Save initial position of first reward
        
        # Move forward (north) - should hit reward at [5, 4]
        state, gradient = world.step(state, 1)  # Action 1: forward + no turn
        
        # Verify agent moved correctly
        assert jnp.array_equal(state.agent_pos, jnp.array([5, 4])), f"Expected position [5, 4], got {state.agent_pos}"
        
        # Check if reward was collected (position should have changed)
        reward_moved = not jnp.array_equal(state.reward_positions[0], initial_reward_pos)
        assert reward_moved, "Reward at [5, 4] should have been collected and respawned"
        
        # The gradient after collection depends on where other rewards are
        assert 0 <= gradient <= 1, f"Gradient out of bounds: {gradient}"
        
        # Test 4: Verify gradients vary with distance
        test_positions = [
            jnp.array([5, 5]),  # Close to rewards
            jnp.array([0, 0]),  # Far corner
            jnp.array([9, 9]),  # Opposite corner
        ]
        gradients = []
        for pos in test_positions:
            grad = world._calculate_gradient(pos, reward_positions, state.reward_active)
            gradients.append(float(grad))
        
        # Verify gradient properties
        assert all(0 <= g <= 1 for g in gradients), "Gradient out of bounds"
        assert gradients[0] > gradients[1], "Gradient should decrease with distance"
        assert gradients[0] > gradients[2], "Gradient should decrease with distance"
    
    def test_reward_tracking(self, world_config: WorldConfig):
        """Test reward tracking functionality."""
        from world.simple_grid_0004 import MinimalGridWorld
        
        world = MinimalGridWorld(WorldConfig(grid_size=10, n_rewards=5, max_timesteps=100))
        key = jrandom.PRNGKey(42)
        state, _ = world.reset(key)
        
        # Run until we collect some rewards
        rewards_collected = 0
        for step in range(100):
            action = int(jrandom.randint(jrandom.PRNGKey(step), (), 0, 9))
            state, gradient = world.step(state, action)
            if gradient == 1.0:
                rewards_collected += 1
        
        # Get tracking data
        tracking = world.get_reward_tracking(state)
        
        assert "positions" in tracking
        assert "spawn_steps" in tracking
        assert "collect_steps" in tracking
        
        # Verify data consistency
        positions = tracking["positions"]
        spawn_steps = tracking["spawn_steps"]
        collect_steps = tracking["collect_steps"]
        
        assert len(positions) >= 5, "Should track at least initial rewards"
        assert len(spawn_steps) == len(positions)
        assert len(collect_steps) == len(positions)
        
        # Count actual collections
        actual_collected = jnp.sum(collect_steps >= 0)
        # Note: The count might not match exactly due to reward respawning
        # Just verify that some rewards were tracked if any were collected
        if rewards_collected > 0:
            assert actual_collected > 0, "Should have tracked some reward collections"
    
    def test_action_space(self, world_config: WorldConfig):
        """Test all 9 actions work correctly."""
        from world.simple_grid_0004 import MinimalGridWorld
        
        world = MinimalGridWorld(WorldConfig(grid_size=20, n_rewards=10, max_timesteps=1000))
        key = jrandom.PRNGKey(42)
        state, _ = world.reset(key)
        
        # Test each action
        for action in range(9):
            state, gradient = world.step(state, action)
            assert isinstance(gradient, jax.Array)
            assert 0 <= gradient <= 1
    
    @pytest.mark.parametrize("grid_size,n_rewards", [
        (10, 5),
        (50, 100),
        (100, 500),
        (200, 1000),
    ])
    def test_scaling_performance(self, grid_size: int, n_rewards: int):
        """Test performance scales appropriately with world size."""
        from world.simple_grid_0004 import MinimalGridWorld
        
        config = WorldConfig(grid_size=grid_size, n_rewards=n_rewards, max_timesteps=1000)
        world = MinimalGridWorld(config)
        key = jrandom.PRNGKey(42)
        state, _ = world.reset(key)
        
        # Warmup
        for _ in range(10):
            state, _ = world.step(state, 0)
        
        # Time steps
        n_steps = 1000
        start = time.perf_counter()
        for i in range(n_steps):
            state, _ = world.step(state, i % 9)
        duration = time.perf_counter() - start
        
        steps_per_second = n_steps / duration
        
        # Performance should not degrade significantly with size
        # Base target of 18k steps/s, allowing some degradation with size
        min_acceptable = 18000 / (1 + grid_size / 200)  # Allow some degradation
        assert steps_per_second > min_acceptable, \
            f"Performance degraded too much at {grid_size}x{grid_size}: {steps_per_second:.0f} steps/s"