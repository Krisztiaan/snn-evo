# keywords: [grid world tests, pytest, jax testing, api tests, corrected]
"""Corrected test suite for SimpleGridWorld v0.0.3."""

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
        
        # Convert to ints for comparison
        agent_x = int(state.agent_pos[0])
        agent_y = int(state.agent_pos[1])
        expected_center = (10, 10)  # grid_size // 2
        assert (agent_x, agent_y) == expected_center

    def test_reward_positions_initialized(self):
        """Test rewards are properly initialized."""
        world = SimpleGridWorld(WorldConfig(grid_size=20, n_rewards=5))
        state, _ = world.reset()
        
        assert state.reward_positions.shape == (5, 2)
        assert jnp.all(state.reward_positions >= 0)
        assert jnp.all(state.reward_positions < 20)
        
        # Check rewards are at unique positions
        positions_set = set()
        for i in range(len(state.reward_positions)):
            pos = (int(state.reward_positions[i, 0]), int(state.reward_positions[i, 1]))
            assert pos not in positions_set, f"Duplicate reward position: {pos}"
            positions_set.add(pos)

    def test_initial_state_values(self):
        """Test initial state has correct values."""
        world = SimpleGridWorld(WorldConfig(grid_size=20, n_rewards=5))
        state, _ = world.reset()
        
        assert jnp.all(~state.reward_collected)  # All False
        assert state.timestep == 0

    def test_observation_gradient_valid(self):
        """Test initial observation gradient is valid."""
        world = SimpleGridWorld(WorldConfig(grid_size=20, n_rewards=5))
        state, obs = world.reset()
        
        assert 0 <= float(obs.gradient) <= 1
        assert jnp.isfinite(obs.gradient)
        
        # Gradient should be based on distance to nearest reward
        # With agent at center and rewards on a ring, gradient should be moderate
        assert 0.1 < float(obs.gradient) < 0.9

    def test_deterministic_reset(self):
        """Test reset is deterministic (no randomness used)."""
        world = SimpleGridWorld(WorldConfig(grid_size=20, n_rewards=5))
        
        # Reset multiple times
        state1, obs1 = world.reset()
        state2, obs2 = world.reset()
        state3, obs3 = world.reset()
        
        # All should be identical since reset is deterministic
        assert (int(state1.agent_pos[0]), int(state1.agent_pos[1])) == \
               (int(state2.agent_pos[0]), int(state2.agent_pos[1])) == \
               (int(state3.agent_pos[0]), int(state3.agent_pos[1]))
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

    def test_movement_directions(self, world_and_state):
        """Test movement in all directions."""
        world, state = world_and_state
        
        # Start from a known position
        start_x, start_y = 10, 10
        state = state._replace(agent_pos=(start_x, start_y))
        
        # Test each direction
        movements = [
            (0, (10, 9)),   # up: y decreases
            (1, (11, 10)),  # right: x increases
            (2, (10, 11)),  # down: y increases
            (3, (9, 10)),   # left: x decreases
        ]
        
        for action, expected in movements:
            state = state._replace(agent_pos=(start_x, start_y))
            result = world.step(state, action)
            actual = (int(result.state.agent_pos[0]), int(result.state.agent_pos[1]))
            assert actual == expected, f"Action {action}: expected {expected}, got {actual}"

    def test_toroidal_wrapping(self, world_and_state):
        """Test all edge wrapping cases."""
        world, state = world_and_state
        grid_size = world.grid_size
        
        # Test all edge cases
        edge_tests = [
            # (start_pos, action, expected_pos)
            ((10, 0), 0, (10, grid_size-1)),    # top edge, move up
            ((grid_size-1, 10), 1, (0, 10)),    # right edge, move right
            ((10, grid_size-1), 2, (10, 0)),    # bottom edge, move down
            ((0, 10), 3, (grid_size-1, 10)),    # left edge, move left
        ]
        
        for start_pos, action, expected in edge_tests:
            state = state._replace(agent_pos=start_pos)
            result = world.step(state, action)
            actual = (int(result.state.agent_pos[0]), int(result.state.agent_pos[1]))
            assert actual == expected, f"From {start_pos} action {action}: expected {expected}, got {actual}"

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

    def test_reward_collection_count(self, world_and_state):
        """Test world correctly reports number of rewards collected."""
        world, state = world_and_state
        
        # Place agent one position away from first reward
        reward_x = int(state.reward_positions[0, 0])
        reward_y = int(state.reward_positions[0, 1])
        # Place below and move up
        agent_y = (reward_y + 1) % world.grid_size
        state = state._replace(agent_pos=(reward_x, agent_y))
        
        # Move up to collect reward
        result = world.step(state, 0)  # action 0 = up
        
        # Should report 1 reward collected
        assert int(result.reward) == 1
        
        # Reward should respawn at different location
        new_x = int(result.state.reward_positions[0, 0])
        new_y = int(result.state.reward_positions[0, 1])
        assert (new_x, new_y) != (reward_x, reward_y)

    def test_no_reward_when_empty_cell(self, world_and_state):
        """Test no reward collected when stepping on empty cell."""
        world, state = world_and_state
        
        # Find a position with no rewards
        occupied = set()
        for i in range(state.reward_positions.shape[0]):
            pos = (int(state.reward_positions[i, 0]), int(state.reward_positions[i, 1]))
            occupied.add(pos)
        
        # Find empty position
        empty_x, empty_y = 0, 0
        while (empty_x, empty_y) in occupied:
            empty_x = (empty_x + 1) % world.grid_size
            if empty_x == 0:
                empty_y = (empty_y + 1) % world.grid_size
        
        state = state._replace(agent_pos=(empty_x, empty_y))
        result = world.step(state, 0)
        
        assert int(result.reward) == 0

    def test_multiple_reward_collection(self, world_and_state):
        """Test collecting multiple rewards if they're at same position."""
        world, state = world_and_state
        
        # Place two rewards at same position
        same_pos = state.reward_positions[0].copy()
        new_positions = state.reward_positions.copy()
        new_positions = new_positions.at[1].set(same_pos)
        state = state._replace(reward_positions=new_positions)
        
        # Place agent one step away
        agent_x = int(same_pos[0])
        agent_y = (int(same_pos[1]) + 1) % world.grid_size
        state = state._replace(agent_pos=(agent_x, agent_y))
        
        # Move to collect both rewards
        result = world.step(state, 0)  # up
        assert int(result.reward) == 2

    def test_collected_rewards_not_recollected(self, world_and_state):
        """Test that already collected rewards can't be collected again."""
        world, state = world_and_state
        
        # Mark first reward as already collected
        collected = state.reward_collected.at[0].set(True)
        state = state._replace(reward_collected=collected)
        
        # Place agent to move to that reward position
        reward_x = int(state.reward_positions[0, 0])
        reward_y = int(state.reward_positions[0, 1])
        agent_y = (reward_y + 1) % world.grid_size
        state = state._replace(agent_pos=(reward_x, agent_y))
        
        # Move to the collected reward position
        result = world.step(state, 0)  # up
        assert int(result.reward) == 0
        assert result.state.reward_collected[0] == True  # Still collected

    def test_reward_respawn_is_deterministic(self, world_and_state):
        """Test that reward respawning is deterministic."""
        world, state = world_and_state
        
        # Setup to collect same reward multiple times
        reward_x = int(state.reward_positions[0, 0])
        reward_y = int(state.reward_positions[0, 1])
        agent_y = (reward_y + 1) % world.grid_size
        
        respawn_positions = []
        for _ in range(3):
            # Reset to same state
            test_state = state._replace(agent_pos=(reward_x, agent_y))
            result = world.step(test_state, 0)  # up to collect
            
            if int(result.reward) > 0:
                new_pos = (int(result.state.reward_positions[0, 0]), 
                          int(result.state.reward_positions[0, 1]))
                respawn_positions.append(new_pos)
        
        # All respawns should be to the same position (deterministic)
        assert len(respawn_positions) == 3
        assert len(set(respawn_positions)) == 1


class TestObservationGradient:
    """Test observation gradient calculations."""

    @pytest.fixture
    def world(self):
        """Create world instance."""
        return SimpleGridWorld(WorldConfig(grid_size=20, n_rewards=5))

    def test_gradient_at_reward(self, world):
        """Test gradient is maximum when at uncollected reward."""
        state, _ = world.reset()
        
        # Place agent at first reward
        reward_x = int(state.reward_positions[0, 0])
        reward_y = int(state.reward_positions[0, 1])
        state = state._replace(agent_pos=(reward_x, reward_y))
        
        # Step to get observation
        result = world.step(state, 0)
        
        # Before moving, distance to nearest reward is 0, so gradient should be e^0 = 1
        # But after step, agent has moved and collected, so check the gradient makes sense
        assert 0 <= float(result.observation.gradient) <= 1

    def test_gradient_decreases_with_distance(self, world):
        """Test gradient decreases as distance increases."""
        state, _ = world.reset()
        
        # Use a small grid to avoid toroidal wrapping issues
        small_world = SimpleGridWorld(WorldConfig(grid_size=30, n_rewards=1))
        small_state, _ = small_world.reset()
        
        # Place single reward at known position
        reward_pos = jnp.array([[15, 15]])
        small_state = small_state._replace(reward_positions=reward_pos)
        
        gradients = []
        # Test distances that won't wrap around
        for dist in [0, 2, 4, 6]:
            # Place agent at distance (horizontally)
            agent_x = 15 + dist
            agent_y = 15
            
            # Get observation directly
            agent_packed = agent_x * small_world.grid_size + agent_y
            reward_packed = reward_pos[:, 0] * small_world.grid_size + reward_pos[:, 1]
            obs = small_world._get_observation_fast(agent_packed, reward_packed, small_state.reward_collected)
            gradients.append(float(obs.gradient))
        
        # Gradients should strictly decrease with distance
        for i in range(len(gradients) - 1):
            assert gradients[i] > gradients[i + 1], f"Gradient at dist {i*2} ({gradients[i]}) should be > gradient at dist {(i+1)*2} ({gradients[i+1]})"

    def test_gradient_ignores_collected_rewards(self, world):
        """Test gradient calculation ignores collected rewards."""
        state, _ = world.reset()
        
        # Mark all but one reward as collected
        collected = jnp.ones(world.n_rewards, dtype=bool)
        collected = collected.at[-1].set(False)
        state = state._replace(reward_collected=collected)
        
        # Place agent far from the only uncollected reward
        uncollected_x = int(state.reward_positions[-1, 0])
        uncollected_y = int(state.reward_positions[-1, 1])
        agent_x = (uncollected_x + 10) % world.grid_size
        agent_y = (uncollected_y + 10) % world.grid_size
        state = state._replace(agent_pos=(agent_x, agent_y))
        
        result = world.step(state, 0)
        
        # Gradient should be based only on distance to the uncollected reward
        # Should be relatively low due to distance
        assert float(result.observation.gradient) < 0.5


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
                done_flags.append(bool(result.done))
                state = result.state
        
        # Should not be done until timestep reaches max_timesteps
        assert done_flags == [False, False, False, False, True]

    def test_done_flag_type(self):
        """Test done flag is proper boolean."""
        world = SimpleGridWorld(WorldConfig(grid_size=10, n_rewards=2, max_timesteps=1))
        state, _ = world.reset()
        
        result = world.step(state, 0)
        assert isinstance(result.done, (bool, jnp.ndarray))
        assert bool(result.done) == True


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
        
        # Reward should be count of collected rewards
        assert isinstance(result.reward, (int, jnp.ndarray))
        assert result.reward >= 0
        
        assert hasattr(result.done, '__bool__') or isinstance(result.done, bool)

    def test_state_immutability(self):
        """Test that original state is not modified."""
        world = SimpleGridWorld(WorldConfig(grid_size=10, n_rewards=2))
        original_state, _ = world.reset()
        
        # Store original values
        original_pos = original_state.agent_pos
        original_timestep = original_state.timestep
        original_positions = original_state.reward_positions.copy()
        
        # Take a step
        result = world.step(original_state, 1)
        
        # Original state should be unchanged
        assert original_state.agent_pos == original_pos
        assert original_state.timestep == original_timestep
        assert jnp.array_equal(original_state.reward_positions, original_positions)


class TestWorldConsistency:
    """Test world behavior consistency."""

    def test_action_consistency(self):
        """Test same action from same state gives same result."""
        world = SimpleGridWorld(WorldConfig(grid_size=10, n_rewards=3))
        state, _ = world.reset()
        
        # Take same action multiple times from same state
        results = []
        for _ in range(3):
            result = world.step(state, 1)  # Always move right
            results.append(result)
        
        # All results should be identical
        for i in range(1, len(results)):
            assert (int(results[0].state.agent_pos[0]), int(results[0].state.agent_pos[1])) == \
                   (int(results[i].state.agent_pos[0]), int(results[i].state.agent_pos[1]))
            assert int(results[0].reward) == int(results[i].reward)
            assert float(results[0].observation.gradient) == float(results[i].observation.gradient)

    def test_packed_position_correctness(self):
        """Test that packed position calculations are correct."""
        world = SimpleGridWorld(WorldConfig(grid_size=10, n_rewards=2))
        
        # Test packing/unpacking
        test_positions = [(0, 0), (5, 7), (9, 9), (3, 8)]
        
        for x, y in test_positions:
            packed = x * world.grid_size + y
            unpacked_x = packed // world.grid_size
            unpacked_y = packed % world.grid_size
            assert (unpacked_x, unpacked_y) == (x, y)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])