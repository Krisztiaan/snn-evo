# keywords: [reward history test, lifecycle tracking, pytest]
"""Test reward history tracking in SimpleGridWorld."""

import pytest
import jax.numpy as jnp

from world.simple_grid_0003 import SimpleGridWorld
from world.simple_grid_0003.types import WorldConfig, RewardRecord


class TestRewardHistory:
    """Test reward history tracking functionality."""
    
    def test_initial_rewards_in_history(self):
        """Test that initial rewards are recorded in history."""
        world = SimpleGridWorld(WorldConfig(grid_size=10, n_rewards=3))
        state, _ = world.reset()
        
        # Should have 3 initial rewards in history
        assert len(state.reward_history) == 3
        
        # All should be spawned at timestep 0, not collected
        for i, record in enumerate(state.reward_history):
            assert record.spawn_step == 0
            assert record.collect_step == -1
            assert isinstance(record.position, tuple)
            assert len(record.position) == 2
            
            # Position should match reward_positions
            assert record.position == (int(state.reward_positions[i, 0]), 
                                      int(state.reward_positions[i, 1]))
    
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
            # History should have grown (original + new spawn)
            assert len(result.state.reward_history) == 3  # 2 initial + 1 new spawn
            
            # First reward should be marked as collected
            assert result.state.reward_history[0].collect_step == 1
            assert result.state.reward_history[0].spawn_step == 0
            
            # New reward should be spawned
            newest = result.state.reward_history[-1]
            assert newest.spawn_step == 1
            assert newest.collect_step == -1
    
    def test_reward_history_at_episode_end(self):
        """Test that complete history is returned when episode ends."""
        world = SimpleGridWorld(WorldConfig(
            grid_size=10, 
            n_rewards=2,
            max_timesteps=3
        ))
        state, _ = world.reset()
        
        # Run until episode ends
        result = None
        for i in range(3):
            result = world.step(state, i % 4)
            state = result.state
        
        # Should be done
        assert result.done
        
        # Should have reward history
        assert result.reward_history is not None
        assert len(result.reward_history) >= 2  # At least initial rewards
        
        # All records should be valid
        for record in result.reward_history:
            assert isinstance(record, RewardRecord)
            assert record.spawn_step >= 0
            assert record.collect_step >= -1
    
    def test_multiple_collections_tracked(self):
        """Test that multiple reward collections are properly tracked."""
        world = SimpleGridWorld(WorldConfig(grid_size=10, n_rewards=2))
        state, _ = world.reset()
        
        collections = []
        
        # Collect first reward
        reward_x = int(state.reward_positions[0, 0])
        reward_y = int(state.reward_positions[0, 1])
        state = state._replace(agent_pos=(reward_x, (reward_y + 1) % 10))
        result = world.step(state, 0)
        if result.reward > 0:
            collections.append(result.state.timestep)
        state = result.state
        
        # Collect the newly spawned reward (now at slot 0)
        reward_x = int(state.reward_positions[0, 0])
        reward_y = int(state.reward_positions[0, 1])
        state = state._replace(agent_pos=(reward_x, (reward_y + 1) % 10))
        result = world.step(state, 0)
        if result.reward > 0:
            collections.append(result.state.timestep)
        
        # Check history
        history = result.state.reward_history
        collected_count = sum(1 for r in history if r.collect_step != -1)
        assert collected_count == len(collections)
    
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
            expected_indices = jnp.array([2, 1])  # slot 0 -> history[2], slot 1 -> history[1]
            assert jnp.array_equal(result.state.reward_indices, expected_indices)
    
    def test_no_collection_no_history_change(self):
        """Test that history doesn't change when no rewards are collected."""
        world = SimpleGridWorld(WorldConfig(grid_size=10, n_rewards=2))
        state, _ = world.reset()
        
        initial_history = state.reward_history.copy()
        
        # Move to empty space
        state = state._replace(agent_pos=(0, 0))
        result = world.step(state, 0)
        
        # History should be unchanged
        assert len(result.state.reward_history) == len(initial_history)
        for i, record in enumerate(result.state.reward_history):
            assert record == initial_history[i]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])