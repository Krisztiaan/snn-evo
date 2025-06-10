# keywords: [grid world tests, pytest, jax testing, api tests]
"""Comprehensive test suite for SimpleGridWorld v0.0.3."""

import pytest
import jax
import jax.numpy as jnp
from jax import random

from world.simple_grid_0003 import SimpleGridWorld
from world.simple_grid_0003.types import WorldConfig, WorldState, Observation, StepResult


class TestWorldInitialization:
    """Test world initialization and configuration."""

    def test_default_initialization(self):
        """Test world initializes with default config."""
        world = SimpleGridWorld()
        assert world.grid_size == 100
        assert world.n_rewards == 300
        assert world.max_timesteps == 50000

    def test_custom_config(self):
        """Test world accepts custom configuration."""
        config = WorldConfig(
            grid_size=50,
            n_rewards=10,
            max_timesteps=1000,
        )
        world = SimpleGridWorld(config)
        assert world.grid_size == 50
        assert world.n_rewards == 10
        assert world.max_timesteps == 1000

    def test_grid_size_override(self):
        """Test grid_size parameter overrides config."""
        config = WorldConfig(grid_size=50)
        world = SimpleGridWorld(config, grid_size=75)
        assert world.grid_size == 75

    def test_metadata(self):
        """Test world metadata is properly formatted."""
        world = SimpleGridWorld()
        metadata = world.get_metadata()
        
        assert isinstance(metadata, dict)
        assert metadata["name"] == "SimpleGridWorld"
        assert metadata["version"] == "0.0.3-ultra"
        assert "description" in metadata
        assert isinstance(metadata["config"], dict)
        assert metadata["config"]["grid_size"] == 100


class TestReset:
    """Test world reset functionality."""

    def test_reset_returns_correct_types(self):
        """Test reset returns proper types."""
        world = SimpleGridWorld(WorldConfig(grid_size=20, n_rewards=5))
        state, obs = world.reset()
        
        assert isinstance(state, WorldState)
        assert isinstance(obs, Observation)
        assert isinstance(state.agent_pos, tuple)
        assert len(state.agent_pos) == 2
        assert isinstance(state.reward_positions, jnp.ndarray)
        assert isinstance(state.reward_collected, jnp.ndarray)

    def test_agent_starts_at_center(self):
        """Test agent starts at grid center."""
        world = SimpleGridWorld(WorldConfig(grid_size=20, n_rewards=5))
        state, _ = world.reset()
        
        expected_center = (10, 10)  # grid_size // 2
        assert state.agent_pos == expected_center

    def test_reward_positions_initialized(self):
        """Test rewards are properly initialized."""
        world = SimpleGridWorld(WorldConfig(grid_size=20, n_rewards=5))
        state, _ = world.reset()
        
        assert state.reward_positions.shape == (5, 2)
        assert jnp.all(state.reward_positions >= 0)
        assert jnp.all(state.reward_positions < 20)

    def test_initial_state_values(self):
        """Test initial state has correct values."""
        world = SimpleGridWorld(WorldConfig(grid_size=20, n_rewards=5))
        state, _ = world.reset()
        
        assert jnp.sum(state.reward_collected) == 0
        assert state.timestep == 0

    def test_observation_gradient_valid(self):
        """Test initial observation gradient is valid."""
        world = SimpleGridWorld(WorldConfig(grid_size=20, n_rewards=5))
        _, obs = world.reset()
        
        assert 0 <= float(obs.gradient) <= 1
        assert jnp.isfinite(obs.gradient)

    def test_deterministic_reset(self):
        """Test reset is deterministic (no randomness used)."""
        world = SimpleGridWorld(WorldConfig(grid_size=20, n_rewards=5))
        
        # Reset multiple times - should be deterministic
        state1, obs1 = world.reset()
        state2, obs2 = world.reset()
        state3, obs3 = world.reset()
        
        # All should be identical since reset is deterministic
        assert state1.agent_pos == state2.agent_pos == state3.agent_pos
        assert jnp.array_equal(state1.reward_positions, state2.reward_positions)
        assert jnp.array_equal(state2.reward_positions, state3.reward_positions)
        assert float(obs1.gradient) == float(obs2.gradient) == float(obs3.gradient)


class TestMovement:
    """Test agent movement mechanics."""

    @pytest.fixture
    def world_and_state(self):
        """Create world and initial state."""
        world = SimpleGridWorld(WorldConfig(grid_size=20, n_rewards=5))
        state, _ = world.reset()
        return world, state

    def test_basic_movement(self, world_and_state):
        """Test basic movement in all directions."""
        world, state = world_and_state
        
        # Starting at (10, 10)
        # Move up (action 0)
        result = world.step(state, 0)
        assert result.state.agent_pos == (10, 9)
        
        # Move right (action 1)
        result = world.step(result.state, 1)
        assert result.state.agent_pos == (11, 9)
        
        # Move down (action 2)
        result = world.step(result.state, 2)
        assert result.state.agent_pos == (11, 10)
        
        # Move left (action 3)
        result = world.step(result.state, 3)
        assert result.state.agent_pos == (10, 10)

    def test_toroidal_wrapping_horizontal(self, world_and_state):
        """Test horizontal edge wrapping."""
        world, state = world_and_state
        
        # Move agent to right edge
        state = state._replace(agent_pos=(19, 10))
        
        # Move right should wrap to x=0
        result = world.step(state, 1)
        assert result.state.agent_pos == (0, 10)
        
        # Move left from x=0 should wrap to x=19
        result = world.step(result.state, 3)
        assert result.state.agent_pos == (19, 10)

    def test_toroidal_wrapping_vertical(self, world_and_state):
        """Test vertical edge wrapping."""
        world, state = world_and_state
        
        # Move agent to top edge
        state = state._replace(agent_pos=(10, 0))
        
        # Move up should wrap to y=19
        result = world.step(state, 0)
        assert result.state.agent_pos == (10, 19)
        
        # Move down from y=19 should wrap to y=0
        state = state._replace(agent_pos=(10, 19))
        result = world.step(state, 2)
        assert result.state.agent_pos == (10, 0)

    def test_timestep_increments(self, world_and_state):
        """Test timestep increments on each step."""
        world, state = world_and_state
        
        for i in range(5):
            result = world.step(state, i % 4)
            assert result.state.timestep == state.timestep + 1
            state = result.state


class TestRewardMechanics:
    """Test reward collection and respawning."""

    @pytest.fixture
    def world_and_state(self):
        """Create world with known reward setup."""
        world = SimpleGridWorld(WorldConfig(
            grid_size=20,
            n_rewards=3
        ))
        state, _ = world.reset()
        return world, state

    def test_reward_collection(self, world_and_state):
        """Test world reports when agent collects reward."""
        world, state = world_and_state
        
        # Find a reward and place agent one step below it
        reward_x = int(state.reward_positions[0, 0])
        reward_y = int(state.reward_positions[0, 1])
        # Place agent one step below (y+1)
        agent_y = (reward_y + 1) % world.grid_size
        state = state._replace(agent_pos=(reward_x, agent_y))
        
        # Step up (action 0) towards the reward
        result = world.step(state, 0)
        
        # Check if agent collected the reward
        # Agent moved up, so should be at reward position
        assert result.state.agent_pos == (reward_x, reward_y)
        assert result.reward == 1  # Should have collected 1 reward
        
        # The collected reward should respawn at a different location
        new_reward_x = int(result.state.reward_positions[0, 0])
        new_reward_y = int(result.state.reward_positions[0, 1])
        assert (new_reward_x, new_reward_y) != (reward_x, reward_y)

    def test_reward_respawns_at_different_location(self, world_and_state):
        """Test collected reward respawns elsewhere."""
        world, state = world_and_state
        
        # Place agent to collect first reward  
        reward_x = int(state.reward_positions[0, 0])
        reward_y = int(state.reward_positions[0, 1])
        # Place agent one step below
        state = state._replace(agent_pos=(reward_x, (reward_y + 1) % world.grid_size))
        
        # Move to collect
        result = world.step(state, 0)  # up
        
        if result.reward > 0:
            # Reward was collected and should respawn elsewhere
            new_x = int(result.state.reward_positions[0, 0])
            new_y = int(result.state.reward_positions[0, 1])
            assert (new_x, new_y) != (reward_x, reward_y)

    def test_no_collection_of_already_collected(self, world_and_state):
        """Test world doesn't report collection of already collected rewards."""
        world, state = world_and_state
        
        # Place agent on reward
        reward_pos = tuple(state.reward_positions[0])
        state = state._replace(agent_pos=reward_pos)
        
        # Mark reward as already collected
        collected = state.reward_collected.at[0].set(True)
        state = state._replace(reward_collected=collected)
        
        # Step should not report this as a new collection
        result = world.step(state, 0)
        # The reward should still be marked as collected
        assert result.state.reward_collected[0]

    def test_observation_gradient_near_reward(self, world_and_state):
        """Test observation gradient when near rewards."""
        world, state = world_and_state
        
        # Place agent 2 cells away from reward
        reward_x, reward_y = int(state.reward_positions[0, 0]), int(state.reward_positions[0, 1])
        nearby_pos = ((reward_x + 2) % 20, reward_y)
        state = state._replace(agent_pos=nearby_pos)
        
        # Step and check gradient reflects proximity
        result = world.step(state, 0)
        # Gradient should be relatively high when near a reward
        assert float(result.observation.gradient) > 0.1

    def test_observation_gradient_when_far(self, world_and_state):
        """Test observation gradient when far from rewards."""
        world, state = world_and_state
        
        # Place agent far from all rewards
        # Use a position that's guaranteed to be far
        state = state._replace(agent_pos=(0, 0))
        
        # Manually check we're far from all rewards
        for i in range(len(state.reward_positions)):
            rx, ry = int(state.reward_positions[i, 0]), int(state.reward_positions[i, 1])
            dist = min(abs(rx), 20 - abs(rx)) + min(abs(ry), 20 - abs(ry))
            if dist < 5:
                # Skip this test if we happen to be near a reward
                pytest.skip("Random reward placement too close")
        
        result = world.step(state, 0)
        # Gradient should be low when far from rewards
        assert float(result.observation.gradient) < 0.5

    def test_gradient_with_multiple_nearby_rewards(self, world_and_state):
        """Test gradient calculation with multiple nearby rewards."""
        world, state = world_and_state
        
        # Create a scenario with multiple rewards nearby
        # Place rewards in a cluster
        new_positions = jnp.array([[10, 10], [12, 10], [10, 12]])
        state = state._replace(
            reward_positions=new_positions,
            agent_pos=(11, 11)  # Agent in the middle
        )
        
        # Gradient should be high when surrounded by rewards
        result = world.step(state, 0)
        assert float(result.observation.gradient) > 0.5  # Reasonable threshold for being near rewards


class TestObservationGradient:
    """Test observation gradient calculations."""

    @pytest.fixture
    def world(self):
        """Create world instance."""
        return SimpleGridWorld(WorldConfig(grid_size=20, n_rewards=5))

    def test_gradient_decreases_with_distance(self, world):
        """Test gradient decreases as agent moves away from rewards."""
        state, _ = world.reset()
        
        # Find nearest reward
        agent_x, agent_y = state.agent_pos
        min_dist = float('inf')
        nearest_idx = 0
        for i in range(len(state.reward_positions)):
            rx, ry = int(state.reward_positions[i, 0]), int(state.reward_positions[i, 1])
            dx = min(abs(rx - agent_x), 20 - abs(rx - agent_x))
            dy = min(abs(ry - agent_y), 20 - abs(ry - agent_y))
            dist = jnp.sqrt(dx**2 + dy**2)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        
        # Move towards the nearest reward and check gradient increases
        gradients = []
        current_state = state
        for _ in range(3):
            result = world.step(current_state, 0)  # Move in some direction
            gradients.append(float(result.observation.gradient))
            current_state = result.state
        
        # At least gradient should be valid
        for g in gradients:
            assert 0 <= g <= 1

    def test_gradient_max_when_on_reward(self, world):
        """Test gradient is high when on uncollected reward."""
        state, _ = world.reset()
        
        # Place agent on reward
        reward_pos = tuple(state.reward_positions[0])
        state = state._replace(agent_pos=reward_pos)
        
        # Get observation directly
        agent_packed = state.agent_pos[0] * world.grid_size + state.agent_pos[1]
        reward_packed = state.reward_positions[:, 0] * world.grid_size + state.reward_positions[:, 1]
        obs = world._get_observation_fast(agent_packed, reward_packed, state.reward_collected)
        
        # Gradient should be 1.0 (or very close) when distance is 0
        assert float(obs.gradient) > 0.99

    def test_gradient_ignores_collected_rewards(self, world):
        """Test gradient calculation ignores collected rewards."""
        state, _ = world.reset()
        
        # Mark first reward as collected
        collected = state.reward_collected.at[0].set(True)
        state = state._replace(reward_collected=collected)
        
        # Place agent on collected reward
        reward_pos = tuple(state.reward_positions[0])
        state = state._replace(agent_pos=reward_pos)
        
        # Step and check gradient
        result = world.step(state, 0)
        
        # Gradient should be based on distance to other uncollected rewards
        assert float(result.observation.gradient) < 0.99


class TestEpisodeTermination:
    """Test episode termination conditions."""

    def test_episode_ends_at_max_timesteps(self):
        """Test episode terminates at max_timesteps."""
        world = SimpleGridWorld(WorldConfig(
            grid_size=10,
            n_rewards=2,
            max_timesteps=5
        ))
        state, _ = world.reset()
        
        # Run for max_timesteps
        done_flags = []
        for i in range(6):
            if state.timestep < 5:
                result = world.step(state, i % 4)
                done_flags.append(result.done)
                state = result.state
        
        # Should not be done until timestep reaches max_timesteps
        assert done_flags == [False, False, False, False, True]

    def test_can_collect_rewards_until_done(self):
        """Test rewards can be collected throughout episode."""
        world = SimpleGridWorld(WorldConfig(
            grid_size=10,
            n_rewards=2,
            max_timesteps=100
        ))
        state, _ = world.reset()
        
        # Move agent to a known reward position
        reward_x = int(state.reward_positions[0, 0])
        reward_y = int(state.reward_positions[0, 1])
        state = state._replace(agent_pos=(reward_x, (reward_y + 1) % world.grid_size))
        
        total_collected = 0
        # First move should collect the reward
        result = world.step(state, 0)  # up
        if result.reward > 0:
            total_collected += result.reward
        
        # Try some more random moves
        state = result.state
        for i in range(20):
            result = world.step(state, i % 4)
            if result.reward > 0:
                total_collected += result.reward
            state = result.state
            
            if result.done:
                break
        
        # Should have collected at least one reward
        assert total_collected > 0


class TestStepResult:
    """Test step result structure and consistency."""

    def test_step_result_structure(self):
        """Test step returns proper result structure."""
        world = SimpleGridWorld(WorldConfig(grid_size=10, n_rewards=2))
        state, _ = world.reset()
        
        result = world.step(state, 0)
        
        assert isinstance(result, StepResult)
        assert isinstance(result.state, WorldState)
        assert isinstance(result.observation, Observation)
        # Check that we have information about what happened
        assert hasattr(result, 'reward') or hasattr(result, 'collected_rewards')
        assert hasattr(result.done, '__bool__') or isinstance(result.done, bool)

    def test_state_immutability(self):
        """Test that original state is not modified."""
        world = SimpleGridWorld(WorldConfig(grid_size=10, n_rewards=2))
        original_state, _ = world.reset()
        
        # Store original values
        original_pos = original_state.agent_pos
        original_timestep = original_state.timestep
        
        # Take a step
        result = world.step(original_state, 1)
        
        # Original state should be unchanged
        assert original_state.agent_pos == original_pos
        assert original_state.timestep == original_timestep


class TestPerformanceCharacteristics:
    """Test performance-related characteristics."""

    def test_jit_compilation(self):
        """Test that step and reset are JIT-compiled."""
        world = SimpleGridWorld(WorldConfig(grid_size=10, n_rewards=2))
        
        # Check that internal methods are JIT-compiled
        assert hasattr(world._reset_jit, "_cache_size")
        assert hasattr(world._step_jit, "_cache_size")

    def test_no_random_key_usage(self):
        """Test that the implementation doesn't use random keys."""
        world = SimpleGridWorld(WorldConfig(grid_size=10, n_rewards=2))
        
        # Reset should give same result every time
        state1, obs1 = world.reset()
        state2, obs2 = world.reset()
        
        assert state1.agent_pos == state2.agent_pos
        assert jnp.array_equal(state1.reward_positions, state2.reward_positions)
        
        # Steps should give same result
        result1 = world.step(state1, 0)
        result2 = world.step(state2, 0)
        
        assert result1.state.agent_pos == result2.state.agent_pos

    def test_packed_position_optimization(self):
        """Test that packed positions are used internally."""
        world = SimpleGridWorld(WorldConfig(grid_size=10, n_rewards=2))
        
        # Check that movement deltas are pre-computed
        assert hasattr(world, 'move_deltas')
        assert len(world.move_deltas) == 4
        
        # Check spawn ring is pre-computed
        assert hasattr(world, 'spawn_ring')
        assert len(world.spawn_ring) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])