# keywords: [grid world, jax environment, optimized, jit compiled, static shapes]
"""Fully JAX-compatible grid world with static shapes and JIT compilation."""

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from jax import Array, jit, random
from simple_grid_0001.types import Observation, StepResult, WorldConfig, WorldState


class SimpleGridWorld:
    """Fully JAX-compatible grid world with JIT compilation.

    Key design principles:
    1. All arrays have static shapes
    2. No boolean indexing with dynamic results
    3. Use masking instead of filtering
    4. Pre-allocate maximum size arrays
    """

    # World metadata
    NAME = "SimpleGridWorld"
    VERSION = "0.0.2-jax"
    DESCRIPTION = "Fully JIT-compiled toroidal grid navigation environment"

    def __init__(self, config: WorldConfig = None, grid_size: int = None):
        if config is None:
            config = WorldConfig(grid_size=grid_size if grid_size is not None else 100)
        elif grid_size is not None:
            config = config._replace(grid_size=grid_size)
        self.config = config

        # Pre-compute constants
        self.decay_constant = config.grid_size / (2 * 4.605)

        # Maximum candidates for reward generation
        self.max_candidates = config.n_rewards * 20
        self.grid_candidates = 100  # For farthest position search

    def get_metadata(self) -> dict:
        """Get world metadata."""
        return {
            "name": self.NAME,
            "version": self.VERSION,
            "description": self.DESCRIPTION,
            "config": self.config._asdict(),
        }

    def reset(self, key: random.PRNGKey) -> Tuple[WorldState, Observation]:
        """Reset the world to initial state."""
        return self._reset_jit(key)

    @partial(jit, static_argnums=(0,))
    def _reset_jit(self, key: random.PRNGKey) -> Tuple[WorldState, Observation]:
        """Reset the world to initial state (JIT-compiled)."""
        key1, key2 = random.split(key)

        # Agent starts at center
        agent_pos = (self.config.grid_size // 2, self.config.grid_size // 2)

        # Generate reward positions with static shape
        reward_positions = self._generate_rewards_static(key1, agent_pos)
        reward_collected = jnp.zeros(self.config.n_rewards, dtype=bool)

        state = WorldState(
            agent_pos=agent_pos,
            reward_positions=reward_positions,
            reward_collected=reward_collected,
            total_reward=0.0,
            timestep=0,
        )

        # Calculate initial observation
        observation = self._get_observation(state)

        return state, observation

    def step(self, state: WorldState, action: int, key: random.PRNGKey) -> StepResult:
        """Take a step in the world."""
        return self._step_jit(state, action, key)

    @partial(jit, static_argnums=(0,))
    def _step_jit(self, state: WorldState, action: int, key: random.PRNGKey) -> StepResult:
        """Take a step in the world (JIT-compiled)."""
        # Split key for respawning
        move_key, respawn_key = random.split(key)

        # Update position
        new_pos = self._move_agent(state.agent_pos, action)

        # Check reward collection and respawn
        reward, new_collected, new_reward_positions = self._collect_rewards_static(
            new_pos, state.reward_positions, state.reward_collected, respawn_key
        )

        # Update state
        new_state = WorldState(
            agent_pos=new_pos,
            reward_positions=new_reward_positions,
            reward_collected=new_collected,
            total_reward=state.total_reward + reward,
            timestep=state.timestep + 1,
        )

        # Get observation
        observation = self._get_observation(new_state)

        # Check if done
        done = new_state.timestep >= self.config.max_timesteps

        return StepResult(state=new_state, observation=observation, reward=reward, done=done)

    @partial(jit, static_argnums=(0,))
    def _generate_rewards_static(self, key: random.PRNGKey, agent_pos: Tuple[int, int]) -> Array:
        """Generate reward positions with static array shapes."""
        min_agent_distance = 3
        agent_array = jnp.array(agent_pos)

        # Generate all candidates at once (static size)
        positions = random.randint(
            key, shape=(self.max_candidates, 2), minval=0, maxval=self.config.grid_size
        )

        # Calculate distances from agent
        distances_to_agent = self._calculate_distances(agent_array, positions)

        # Create validity mask
        valid_mask = distances_to_agent >= min_agent_distance

        # Remove duplicates using sorting and masking
        # Create unique IDs for positions
        position_ids = positions[:, 0] * self.config.grid_size + positions[:, 1]

        # Sort by ID to group duplicates
        sorted_indices = jnp.argsort(position_ids)
        sorted_ids = position_ids[sorted_indices]
        sorted_positions = positions[sorted_indices]
        sorted_valid = valid_mask[sorted_indices]

        # Mark first occurrence of each position
        is_first = jnp.concatenate([jnp.array([True]), sorted_ids[1:] != sorted_ids[:-1]])

        # Combine validity masks
        final_valid = sorted_valid & is_first

        # Get indices where final_valid is True
        valid_indices = jnp.arange(self.max_candidates)

        # Create a ranking based on validity (valid positions get low ranks)
        # Invalid positions get high ranks
        ranks = jnp.where(final_valid, valid_indices, self.max_candidates + valid_indices)

        # Sort positions by rank (valid ones first)
        rank_order = jnp.argsort(ranks)
        ranked_positions = sorted_positions[rank_order]

        # Take first n_rewards positions
        selected_positions = ranked_positions[: self.config.n_rewards]

        # Check if we got enough valid positions
        n_valid = jnp.sum(final_valid)

        # Generate fallback grid positions with fixed number of points
        # Use linspace to avoid dynamic step size
        n_grid_points = 10  # Fixed number of grid points per dimension
        grid_x = jnp.linspace(0, self.config.grid_size - 1, n_grid_points, dtype=jnp.int32)
        grid_y = jnp.linspace(0, self.config.grid_size - 1, n_grid_points, dtype=jnp.int32)
        xx, yy = jnp.meshgrid(grid_x, grid_y)
        grid_positions = jnp.stack([xx.ravel(), yy.ravel()], axis=1)

        # Filter grid positions by distance
        grid_distances = self._calculate_distances(agent_array, grid_positions)
        grid_valid_mask = grid_distances >= min_agent_distance

        # Rank grid positions by validity
        grid_ranks = jnp.where(
            grid_valid_mask,
            jnp.arange(len(grid_positions)),
            len(grid_positions) + jnp.arange(len(grid_positions)),
        )
        grid_rank_order = jnp.argsort(grid_ranks)
        ranked_grid = grid_positions[grid_rank_order]

        # For each position, decide whether to use selected or grid fallback
        def select_position(i):
            # If we have enough valid positions, use selected
            # Otherwise, use grid fallback
            use_selected = i < n_valid
            grid_idx = jnp.minimum(i, len(ranked_grid) - 1)
            return jax.lax.cond(
                use_selected, lambda: selected_positions[i], lambda: ranked_grid[grid_idx]
            )

        final_positions = jax.vmap(select_position)(jnp.arange(self.config.n_rewards))

        return final_positions

    @partial(jit, static_argnums=(0,))
    def _move_agent(self, pos: Tuple[int, int], action: int) -> Tuple[int, int]:
        """Move agent based on action."""
        x, y = pos

        # Action mapping: 0=up, 1=right, 2=down, 3=left
        dx = jnp.array([0, 1, 0, -1])[action]
        dy = jnp.array([-1, 0, 1, 0])[action]

        new_x = x + dx
        new_y = y + dy

        # Handle boundaries
        new_x = jax.lax.cond(
            self.config.toroidal,
            lambda: new_x % self.config.grid_size,
            lambda: jnp.clip(new_x, 0, self.config.grid_size - 1),
        )
        new_y = jax.lax.cond(
            self.config.toroidal,
            lambda: new_y % self.config.grid_size,
            lambda: jnp.clip(new_y, 0, self.config.grid_size - 1),
        )

        return (new_x, new_y)

    @partial(jit, static_argnums=(0,))
    def _collect_rewards_static(
        self,
        agent_pos: Tuple[int, int],
        reward_positions: Array,
        reward_collected: Array,
        key: random.PRNGKey,
    ) -> Tuple[float, Array, Array]:
        """Check reward collection with static array operations."""
        agent_array = jnp.array(agent_pos)
        distances = self._calculate_distances(agent_array, reward_positions)

        # Check which rewards are collected
        at_reward = (distances < 0.5) & ~reward_collected

        # Calculate rewards
        reward = jnp.sum(at_reward.astype(jnp.float32)) * self.config.reward_value

        # Proximity reward
        near_reward = (distances < 5.0) & ~reward_collected & ~at_reward
        proximity_reward = jnp.sum(near_reward.astype(jnp.float32)) * self.config.proximity_reward

        total_reward = reward + proximity_reward

        # Update collected status
        new_collected = reward_collected | at_reward

        # Respawn collected rewards
        any_collected = jnp.any(at_reward)

        # Find new position for respawn
        new_spawn_pos = self._find_farthest_position_static(
            agent_pos, reward_positions, reward_collected, key
        )

        # Update positions - use scan to handle multiple collections
        def update_single_reward(carry, i):
            positions = carry
            should_update = at_reward[i]
            new_positions = jax.lax.cond(
                should_update, lambda: positions.at[i].set(new_spawn_pos), lambda: positions
            )
            return new_positions, None

        new_reward_positions, _ = jax.lax.scan(
            update_single_reward, reward_positions, jnp.arange(self.config.n_rewards)
        )

        # Reset collected status for respawned rewards
        final_collected = jax.lax.cond(
            any_collected, lambda: new_collected & ~at_reward, lambda: new_collected
        )

        return total_reward, final_collected, new_reward_positions

    @partial(jit, static_argnums=(0,))
    def _get_observation(self, state: WorldState) -> Observation:
        """Generate observation from state."""
        agent_array = jnp.array(state.agent_pos)
        distances = self._calculate_distances(agent_array, state.reward_positions)

        # Mask out collected rewards
        masked_distances = jnp.where(~state.reward_collected, distances, jnp.inf)

        # Find minimum distance
        min_distance = jnp.min(masked_distances)

        # Convert to gradient signal
        gradient = jnp.where(
            jnp.isfinite(min_distance), jnp.exp(-min_distance / self.decay_constant), 0.0
        )

        return Observation(gradient=gradient)

    @partial(jit, static_argnums=(0,))
    def _calculate_distances(self, pos: Array, positions: Array) -> Array:
        """Calculate toroidal distances with static shapes."""
        diff = positions - pos[None, :]  # Ensure broadcasting

        # Handle toroidal wrapping
        wrapped_diff = jax.lax.cond(
            self.config.toroidal,
            lambda: jnp.minimum(jnp.abs(diff), self.config.grid_size - jnp.abs(diff)),
            lambda: jnp.abs(diff),
        )

        return jnp.linalg.norm(wrapped_diff, axis=1)

    @partial(jit, static_argnums=(0,))
    def _find_farthest_position_static(
        self,
        agent_pos: Tuple[int, int],
        reward_positions: Array,
        reward_collected: Array,
        key: random.PRNGKey,
    ) -> Array:
        """Find farthest position using static array operations."""
        agent_array = jnp.array(agent_pos)

        # Generate fixed number of candidate positions
        candidates = random.randint(
            key, shape=(self.grid_candidates, 2), minval=0, maxval=self.config.grid_size
        )

        # Calculate distances from agent
        distances_from_agent = self._calculate_distances(agent_array, candidates)

        # Calculate min distance to each uncollected reward
        # Use masking instead of filtering
        uncollected_mask = ~reward_collected

        # For each candidate, find min distance to uncollected rewards
        def min_distance_to_rewards(candidate):
            reward_dists = self._calculate_distances(candidate, reward_positions)
            # Mask collected rewards with large distance
            masked_dists = jnp.where(uncollected_mask, reward_dists, jnp.inf)
            return jnp.min(masked_dists)

        min_distances_to_rewards = jax.vmap(min_distance_to_rewards)(candidates)

        # Handle case where all rewards collected
        has_uncollected = jnp.any(uncollected_mask)
        min_distances_to_rewards = jax.lax.cond(
            has_uncollected,
            lambda: min_distances_to_rewards,
            lambda: jnp.zeros(self.grid_candidates),
        )

        # Combined score: far from agent and rewards
        scores = distances_from_agent + 0.3 * min_distances_to_rewards

        # Return position with highest score
        best_idx = jnp.argmax(scores)
        return candidates[best_idx]
