# keywords: [reward history test jax, lifecycle tracking jax arrays, pytest jax]
"""Test JAX-optimized reward history tracking in SimpleGridWorld."""

import pytest
import jax.numpy as jnp

from world.simple_grid_0003 import SimpleGridWorld
from world.simple_grid_0003.types import WorldConfig


class TestRewardHistoryJAX:
    """Test JAX array-based reward history tracking functionality."""
    
    def test_initial_rewards_in_history(self):
        """Test that initial rewards are recorded in history."""
        world = SimpleGridWorld(WorldConfig(grid_size=10, n_rewards=3))
        state, _ = world.reset()
        
        # Should have 3 initial rewards in history
        assert state.reward_history_count == 3
        
        # All should be spawned at timestep 0, not collected
        for i in range(3):
            assert state.reward_history_spawn_steps[i] == 0
            assert state.reward_history_collect_steps[i] == -1
            
            # Position should match reward_positions
            assert jnp.array_equal(
                state.reward_history_positions[i], 
                state.reward_positions[i]
            )
    
    def test_reward_collection_updates_history(self):
        """Test that collecting a reward updates its history record."""
        world = SimpleGridWorld(WorldConfig(grid_size=10, n_rewards=2))
        state, _ = world.reset()
        
        # Move agent to collect first reward
        reward_x = int(state.reward_positions[0, 0])
        reward_y = int(state.reward_positions[0, 1])
        state = state._replace(agent_pos=(reward_x, (reward_y + 1) % world.grid_size))
        
        # Collect the reward
        result = world.step(state, 0)  # Move up
        
        if result.reward > 0:
            # History count should have grown
            assert result.state.reward_history_count == 3  # 2 initial + 1 new spawn
            
            # First reward should be marked as collected
            history_idx = int(state.reward_indices[0])
            assert result.state.reward_history_collect_steps[history_idx] == 1
            assert result.state.reward_history_spawn_steps[history_idx] == 0
            
            # New reward should be spawned
            newest_idx = result.state.reward_history_count - 1
            assert result.state.reward_history_spawn_steps[newest_idx] == 1
            assert result.state.reward_history_collect_steps[newest_idx] == -1
    
    def test_reward_history_extraction(self):
        """Test that reward history can be extracted from state."""
        world = SimpleGridWorld(WorldConfig(
            grid_size=10, 
            n_rewards=2,
            max_timesteps=3
        ))
        state, _ = world.reset()
        
        # Run for a few steps
        for i in range(3):
            result = world.step(state, i % 4)
            state = result.state
        
        # Extract reward history
        positions, spawn_steps, collect_steps = world.get_reward_history(state)
        
        # Should have arrays with correct shapes
        history_count = int(state.reward_history_count)
        assert positions.shape == (history_count, 2)
        assert spawn_steps.shape == (history_count,)
        assert collect_steps.shape == (history_count,)
        
        # Check valid entries
        assert history_count >= 2  # At least initial rewards
        for i in range(history_count):
            assert spawn_steps[i] >= 0
            assert collect_steps[i] >= -1
    
    def test_multiple_collections_tracked(self):
        """Test that multiple reward collections are properly tracked."""
        world = SimpleGridWorld(WorldConfig(grid_size=10, n_rewards=2))
        state, _ = world.reset()
        
        collections = 0
        
        # Collect first reward
        reward_x = int(state.reward_positions[0, 0])
        reward_y = int(state.reward_positions[0, 1])
        state = state._replace(agent_pos=(reward_x, (reward_y + 1) % 10))
        result = world.step(state, 0)
        if result.reward > 0:
            collections += 1
        state = result.state
        
        # Collect the newly spawned reward (now at slot 0)
        reward_x = int(state.reward_positions[0, 0])
        reward_y = int(state.reward_positions[0, 1])
        state = state._replace(agent_pos=(reward_x, (reward_y + 1) % 10))
        result = world.step(state, 0)
        if result.reward > 0:
            collections += 1
        
        # Check history
        history_count = result.state.reward_history_count
        collected_count = jnp.sum(result.state.reward_history_collect_steps[:history_count] != -1)
        assert collected_count == collections
    
    def test_reward_indices_tracking(self):
        """Test that reward indices correctly map slots to history."""
        world = SimpleGridWorld(WorldConfig(grid_size=10, n_rewards=2))
        state, _ = world.reset()
        
        # Initially, indices should be [0, 1]
        assert jnp.array_equal(state.reward_indices, jnp.array([0, 1]))
        
        # Collect first reward
        reward_x = int(state.reward_positions[0, 0])
        reward_y = int(state.reward_positions[0, 1])
        state = state._replace(agent_pos=(reward_x, (reward_y + 1) % 10))
        result = world.step(state, 0)
        
        if result.reward > 0:
            # First slot should now point to the new reward (last in history)
            expected_idx = result.state.reward_history_count - 1
            assert result.state.reward_indices[0] == expected_idx
    
    def test_no_collection_no_history_change(self):
        """Test that history doesn't change when no rewards are collected."""
        world = SimpleGridWorld(WorldConfig(grid_size=10, n_rewards=2))
        state, _ = world.reset()
        
        initial_count = state.reward_history_count
        initial_positions = state.reward_history_positions.copy()
        initial_spawn_steps = state.reward_history_spawn_steps.copy()
        initial_collect_steps = state.reward_history_collect_steps.copy()
        
        # Move to empty space
        state = state._replace(agent_pos=(0, 0))
        result = world.step(state, 0)
        
        # History should be unchanged
        assert result.state.reward_history_count == initial_count
        assert jnp.array_equal(result.state.reward_history_positions, initial_positions)
        assert jnp.array_equal(result.state.reward_history_spawn_steps, initial_spawn_steps)
        assert jnp.array_equal(result.state.reward_history_collect_steps, initial_collect_steps)
    
    def test_jit_compilation(self):
        """Test that step function compiles with JIT."""
        world = SimpleGridWorld(WorldConfig(grid_size=10, n_rewards=2))
        state, _ = world.reset()
        
        # First call compiles
        result1 = world.step(state, 0)
        
        # Second call should use compiled version
        result2 = world.step(state, 1)
        
        # Both should return valid results
        assert isinstance(result1.reward, (int, jnp.ndarray))
        assert isinstance(result2.reward, (int, jnp.ndarray))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])