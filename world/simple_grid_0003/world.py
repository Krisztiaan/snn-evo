# keywords: [grid world, jax environment, ultra optimized, packed positions, performance]
"""Ultra-optimized JAX grid world with packed positions and simplified algorithms."""

from functools import partial
from typing import Optional, Tuple

import jax.numpy as jnp
from jax import Array, jit, random

from world.simple_grid_0001.types import Observation, StepResult, WorldConfig, WorldState


class SimpleGridWorld:
    """Ultra-optimized grid world with maximum performance.

    Key optimizations:
    1. Packed integer positions (x * grid_size + y)
    2. Simplified deterministic algorithms
    3. Minimal branching and conditionals
    4. Pre-computed spawn positions
    5. Fused operations for XLA optimization
    """

    # World metadata
    NAME = "SimpleGridWorld"
    VERSION = "0.0.3-ultra"
    DESCRIPTION = "Ultra-optimized toroidal grid environment for maximum performance"

    def __init__(self, config: Optional[WorldConfig] = None, grid_size: Optional[int] = None):
        if config is None:
            config = WorldConfig(grid_size=grid_size if grid_size is not None else 100)
        elif grid_size is not None:
            config = config._replace(grid_size=grid_size)
        self.config = config

        # Pre-compute constants
        self.grid_size = config.grid_size
        self.n_rewards = config.n_rewards
        self.reward_value = config.reward_value
        self.proximity_reward = config.proximity_reward
        self.max_timesteps = config.max_timesteps

        # Packed position helpers
        self.total_positions = self.grid_size * self.grid_size

        # Pre-compute spawn ring positions (deterministic, evenly spaced)
        angles = jnp.linspace(0, 2 * jnp.pi, self.n_rewards * 4, endpoint=False)
        radius = jnp.maximum(5, self.grid_size // 4)
        spawn_x = (self.grid_size // 2 + radius * jnp.cos(angles)).astype(jnp.int32)
        spawn_y = (self.grid_size // 2 + radius * jnp.sin(angles)).astype(jnp.int32)
        spawn_x = jnp.clip(spawn_x, 0, self.grid_size - 1)
        spawn_y = jnp.clip(spawn_y, 0, self.grid_size - 1)
        self.spawn_ring = spawn_x * self.grid_size + spawn_y

        # Movement deltas for packed positions
        self.move_deltas = jnp.array(
            [
                -self.grid_size,  # up
                1,  # right
                self.grid_size,  # down
                -1,  # left
            ],
            dtype=jnp.int32,
        )

        # Distance calculation constants
        self.decay_constant = self.grid_size / (2 * 4.605)

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
        """Reset with packed positions."""
        # Agent starts at center (packed)
        agent_pos = (self.grid_size // 2, self.grid_size // 2)
        agent_packed = agent_pos[0] * self.grid_size + agent_pos[1]

        # Select evenly spaced rewards from spawn ring
        indices = jnp.linspace(0, len(self.spawn_ring) - 1, self.n_rewards, dtype=jnp.int32)
        reward_positions_packed = self.spawn_ring[indices]

        # Unpack for compatibility with WorldState
        reward_x = reward_positions_packed // self.grid_size
        reward_y = reward_positions_packed % self.grid_size
        reward_positions = jnp.stack([reward_x, reward_y], axis=1)

        state = WorldState(
            agent_pos=agent_pos,
            reward_positions=reward_positions,
            reward_collected=jnp.zeros(self.n_rewards, dtype=bool),
            total_reward=0.0,
            timestep=0,
        )

        # Calculate initial observation
        observation = self._get_observation_fast(
            agent_packed, reward_positions_packed, state.reward_collected
        )

        return state, observation

    def step(self, state: WorldState, action: int, key: random.PRNGKey) -> StepResult:
        """Take a step in the world."""
        return self._step_jit(state, action, key)

    @partial(jit, static_argnums=(0,))
    def _step_jit(self, state: WorldState, action: int, key: random.PRNGKey) -> StepResult:
        """Optimized step with packed positions."""
        # Pack current position
        agent_packed = state.agent_pos[0] * self.grid_size + state.agent_pos[1]

        # Move agent (packed arithmetic)
        new_agent_packed = self._move_packed(agent_packed, action)

        # Pack reward positions
        reward_packed = state.reward_positions[:, 0] * self.grid_size + state.reward_positions[:, 1]

        # Check collection and calculate rewards (all vectorized)
        at_reward = (reward_packed == new_agent_packed) & ~state.reward_collected
        reward = jnp.sum(at_reward.astype(jnp.float32)) * self.reward_value

        # Proximity rewards using packed distance
        distances_squared = self._packed_distance_squared(new_agent_packed, reward_packed)
        near_mask = (distances_squared < 25) & ~state.reward_collected & ~at_reward  # 5^2 = 25
        proximity_reward = jnp.sum(near_mask.astype(jnp.float32)) * self.proximity_reward

        # Update collected
        new_collected = state.reward_collected | at_reward

        # Respawn: simple deterministic offset
        # Prime for better distribution
        respawn_offset = jnp.sum(at_reward.astype(jnp.int32)) * 37
        new_reward_packed = jnp.where(
            at_reward,
            (reward_packed + self.grid_size * 10 + respawn_offset) % self.total_positions,
            reward_packed,
        )

        # Unpack for state
        new_x = new_agent_packed // self.grid_size
        new_y = new_agent_packed % self.grid_size
        new_agent_pos = (new_x, new_y)

        reward_x = new_reward_packed // self.grid_size
        reward_y = new_reward_packed % self.grid_size
        new_reward_positions = jnp.stack([reward_x, reward_y], axis=1)

        # Reset collected for respawned rewards
        final_collected = new_collected & ~at_reward

        # Create new state
        new_state = WorldState(
            agent_pos=new_agent_pos,
            reward_positions=new_reward_positions,
            reward_collected=final_collected,
            total_reward=state.total_reward + reward + proximity_reward,
            timestep=state.timestep + 1,
        )

        # Get observation
        observation = self._get_observation_fast(
            new_agent_packed, new_reward_packed, final_collected
        )

        return StepResult(
            state=new_state,
            observation=observation,
            reward=reward + proximity_reward,
            done=new_state.timestep >= self.max_timesteps,
        )

    @partial(jit, static_argnums=(0,))
    def _move_packed(self, pos_packed: int, action: int) -> int:
        """Move in packed space with toroidal wrapping."""
        # Extract x, y
        x = pos_packed // self.grid_size
        y = pos_packed % self.grid_size

        # Apply movement
        dx = jnp.array([0, 1, 0, -1])[action]
        dy = jnp.array([-1, 0, 1, 0])[action]

        # Toroidal wrap
        new_x = (x + dx) % self.grid_size
        new_y = (y + dy) % self.grid_size

        return new_x * self.grid_size + new_y

    @partial(jit, static_argnums=(0,))
    def _packed_distance_squared(self, pos1_packed: int, pos2_packed: Array) -> Array:
        """Calculate squared distances in packed space (toroidal)."""
        # Unpack positions
        x1 = pos1_packed // self.grid_size
        y1 = pos1_packed % self.grid_size
        x2 = pos2_packed // self.grid_size
        y2 = pos2_packed % self.grid_size

        # Toroidal differences
        dx = jnp.minimum(jnp.abs(x2 - x1), self.grid_size - jnp.abs(x2 - x1))
        dy = jnp.minimum(jnp.abs(y2 - y1), self.grid_size - jnp.abs(y2 - y1))

        return dx * dx + dy * dy

    @partial(jit, static_argnums=(0,))
    def _get_observation_fast(
        self, agent_packed: int, reward_packed: Array, collected: Array
    ) -> Observation:
        """Fast observation calculation."""
        # Calculate all squared distances
        distances_squared = self._packed_distance_squared(agent_packed, reward_packed)

        # Mask collected rewards
        masked_distances = jnp.where(~collected, distances_squared, jnp.inf)

        # Find minimum and convert to gradient
        min_dist_squared = jnp.min(masked_distances)
        min_dist = jnp.sqrt(min_dist_squared)
        gradient = jnp.exp(-min_dist / self.decay_constant)

        return Observation(gradient=gradient)
