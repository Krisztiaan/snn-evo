# keywords: [grid world, jax environment, ultra optimized, packed positions, performance]
"""Ultra-optimized JAX grid world with packed positions and simplified algorithms."""

from functools import partial

import jax.numpy as jnp
from jax import Array, jit, lax

from world.simple_grid_0003.types import Observation, StepResult, WorldConfig, WorldState


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

    def __init__(self, config: WorldConfig | None = None, grid_size: int | None = None):
        if config is None:
            config = WorldConfig(grid_size=grid_size if grid_size is not None else 100)
        elif grid_size is not None:
            config = config._replace(grid_size=grid_size)
        
        # Validate config parameters
        if config.grid_size < 10:
            raise ValueError(f"grid_size must be at least 10, got {config.grid_size}")
        if config.n_rewards < 1:
            raise ValueError(f"n_rewards must be at least 1, got {config.n_rewards}")
        if config.n_rewards > config.grid_size * config.grid_size:
            raise ValueError(f"n_rewards ({config.n_rewards}) cannot exceed total grid positions ({config.grid_size * config.grid_size})")
        if config.max_timesteps < 1:
            raise ValueError(f"max_timesteps must be at least 1, got {config.max_timesteps}")
        
        self.config = config

        # Pre-compute constants
        self.grid_size = config.grid_size
        self.n_rewards = config.n_rewards
        self.max_timesteps = config.max_timesteps

        # Packed position helpers
        self.total_positions = self.grid_size * self.grid_size
        
        # Pre-allocate max history size
        # Realistic assumption: at most we collect all rewards once per 100 steps on average
        # This provides a good balance between memory usage and capacity
        estimated_collections_per_step = min(self.n_rewards, 5)  # Cap at 5 per step for memory
        self.max_history_size = self.n_rewards + (self.max_timesteps * estimated_collections_per_step)
        # Cap total size to prevent excessive memory usage
        self.max_history_size = min(self.max_history_size, 100000)

        # Pre-compute spawn ring positions (deterministic, evenly spaced)
        # Only compute exactly what we need
        n_spawn_positions = min(self.n_rewards * 4, self.total_positions // 2)
        angles = jnp.linspace(0, 2 * jnp.pi, n_spawn_positions, endpoint=False)
        radius = jnp.maximum(5, self.grid_size // 4)
        center = self.grid_size // 2
        spawn_x = jnp.clip(center + radius * jnp.cos(angles), 0, self.grid_size - 1).astype(jnp.int32)
        spawn_y = jnp.clip(center + radius * jnp.sin(angles), 0, self.grid_size - 1).astype(jnp.int32)
        self.spawn_ring = spawn_x * self.grid_size + spawn_y

        # Movement deltas for packed positions
        # packed = x * grid_size + y
        # up: y-1, right: x+1, down: y+1, left: x-1
        self.move_deltas = jnp.array(
            [
                -1,  # up: decrease y
                self.grid_size,  # right: increase x 
                1,  # down: increase y
                -self.grid_size,  # left: decrease x
            ],
            dtype=jnp.int32,
        )

        # Distance calculation constants
        self.decay_constant = self.grid_size / (2 * 4.605)
        # Pre-compute large value for masking collected rewards
        self.large_distance = self.grid_size * self.grid_size * 4

    def get_metadata(self) -> dict[str, str | int | dict[str, str | int | float | bool]]:
        """Get world metadata."""
        return {
            "name": self.NAME,
            "version": self.VERSION,
            "description": self.DESCRIPTION,
            "config": self.config._asdict(),
        }

    def reset(self) -> tuple[WorldState, Observation]:
        """Reset the world to initial state."""
        return self._reset_jit()

    @partial(jit, static_argnums=(0,))
    def _reset_jit(self) -> tuple[WorldState, Observation]:
        """Reset with packed positions - fully JIT-compiled."""
        # Agent starts at center (packed)
        agent_x = jnp.array(self.grid_size // 2, dtype=jnp.int32)
        agent_y = jnp.array(self.grid_size // 2, dtype=jnp.int32)
        agent_pos = (agent_x, agent_y)
        agent_packed = agent_x * self.grid_size + agent_y

        # Select evenly spaced rewards from spawn ring
        indices = jnp.linspace(0, len(self.spawn_ring) - 1, self.n_rewards, dtype=jnp.int32)
        reward_positions_packed = self.spawn_ring[indices]

        # Unpack for compatibility with WorldState
        reward_x = reward_positions_packed // self.grid_size
        reward_y = reward_positions_packed % self.grid_size
        reward_positions = jnp.stack([reward_x, reward_y], axis=1)
        
        # Initialize reward history arrays
        history_positions = jnp.zeros((self.max_history_size, 2), dtype=jnp.int32)
        history_spawn_steps = jnp.full(self.max_history_size, -1, dtype=jnp.int32)
        history_collect_steps = jnp.full(self.max_history_size, -1, dtype=jnp.int32)
        
        # Set initial rewards in history
        history_positions = history_positions.at[:self.n_rewards].set(reward_positions)
        history_spawn_steps = history_spawn_steps.at[:self.n_rewards].set(0)

        state = WorldState(
            agent_pos=agent_pos,
            reward_positions=reward_positions,
            reward_collected=jnp.zeros(self.n_rewards, dtype=bool),
            timestep=jnp.array(0, dtype=jnp.int32),
            reward_history_positions=history_positions,
            reward_history_spawn_steps=history_spawn_steps,
            reward_history_collect_steps=history_collect_steps,
            reward_history_count=jnp.array(self.n_rewards, dtype=jnp.int32),
            reward_indices=jnp.arange(self.n_rewards, dtype=jnp.int32),
        )

        # Calculate initial observation
        observation = self._get_observation_fast(
            agent_packed, reward_positions_packed, state.reward_collected
        )

        return state, observation

    def step(self, state: WorldState, action: int) -> StepResult:
        """Take a step in the world."""
        return self._step_jit(state, action)

    @partial(jit, static_argnums=(0,))
    def _step_jit(self, state: WorldState, action: int) -> StepResult:
        """Fully JIT-compiled step with optimized reward history tracking."""
        # Pack current position
        agent_packed = state.agent_pos[0] * self.grid_size + state.agent_pos[1]

        # Move agent (packed arithmetic)
        new_agent_packed = self._move_packed(agent_packed, action)

        # Pack reward positions
        reward_packed = state.reward_positions[:, 0] * self.grid_size + state.reward_positions[:, 1]

        # Check collection (all vectorized)
        at_reward = (reward_packed == new_agent_packed) & ~state.reward_collected
        num_collected = jnp.sum(at_reward.astype(jnp.int32))

        # Update collected
        new_collected = state.reward_collected | at_reward

        # Respawn: better distribution using timestep and reward index
        # Each reward gets a unique offset based on its index and current timestep
        # Ensure respawn is never at the same position
        reward_indices = jnp.arange(self.n_rewards)
        base_offsets = (state.timestep * 1009 + reward_indices * 251 + 1) % self.total_positions
        # Ensure offset is at least 1 to guarantee different position
        respawn_offsets = jnp.maximum(base_offsets, 1)
        new_reward_packed = jnp.where(
            at_reward,
            (reward_packed + respawn_offsets) % self.total_positions,
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
        
        # Fully vectorized reward history update using JAX-friendly operations
        n_collected = jnp.sum(at_reward)
        
        # 1. Update collection timestamps for collected rewards
        history_collect_steps = state.reward_history_collect_steps
        # Update collection times where rewards were collected
        history_collect_steps = history_collect_steps.at[state.reward_indices].set(
            jnp.where(at_reward, state.timestep + 1, history_collect_steps[state.reward_indices])
        )
        
        # 2. Add new spawned rewards to history using JAX-friendly approach
        # Calculate cumulative sum to assign sequential indices
        collected_cumsum = jnp.cumsum(at_reward.astype(jnp.int32))
        new_base_idx = state.reward_history_count
        
        # Update reward indices for collected rewards
        new_reward_indices = jnp.where(
            at_reward,
            new_base_idx + collected_cumsum - 1,
            state.reward_indices
        )
        
        # Ultra-efficient vectorized history update
        history_positions = state.reward_history_positions
        history_spawn_steps = state.reward_history_spawn_steps
        
        # Use a segment-based approach for maximum efficiency
        # First, compute where each collected reward goes
        collected_cumsum = jnp.cumsum(at_reward.astype(jnp.int32))
        
        # Create scatter indices for all rewards at once
        # Each collected reward gets position: base + (its order in collected) - 1
        scatter_indices = jnp.where(
            at_reward,
            new_base_idx + collected_cumsum - 1,
            self.max_history_size  # Out of bounds = no-op
        )
        
        # Prepare values to scatter
        # Use the original reward positions for history
        scatter_positions = new_reward_positions
        scatter_spawn_times = jnp.full(self.n_rewards, state.timestep + 1)
        
        # Single scatter operation for all updates
        # This is much more efficient than scan or multiple cond operations
        history_positions = history_positions.at[scatter_indices].set(
            scatter_positions,
            mode='drop'  # Drop out-of-bounds updates
        )
        history_spawn_steps = history_spawn_steps.at[scatter_indices].set(
            scatter_spawn_times,
            mode='drop'
        )
        
        new_history_count = state.reward_history_count + n_collected
        
        # Batch all conditional updates into a single lax.cond for efficiency
        # This reduces overhead from multiple condition checks
        def get_updated_values():
            return (
                history_positions,
                history_spawn_steps,
                history_collect_steps,
                new_reward_indices,
                new_history_count
            )
        
        def get_original_values():
            return (
                state.reward_history_positions,
                state.reward_history_spawn_steps,
                state.reward_history_collect_steps,
                state.reward_indices,
                state.reward_history_count
            )
        
        # Single conditional for all updates
        (final_history_positions, final_history_spawn_steps, 
         final_history_collect_steps, final_reward_indices,
         final_history_count) = lax.cond(
            n_collected > 0,
            get_updated_values,
            get_original_values
        )

        # Create new state
        new_state = WorldState(
            agent_pos=new_agent_pos,
            reward_positions=new_reward_positions,
            reward_collected=final_collected,
            timestep=state.timestep + 1,
            reward_history_positions=final_history_positions,
            reward_history_spawn_steps=final_history_spawn_steps,
            reward_history_collect_steps=final_history_collect_steps,
            reward_history_count=final_history_count,
            reward_indices=final_reward_indices,
        )

        # Get observation
        observation = self._get_observation_fast(
            new_agent_packed, new_reward_packed, final_collected
        )

        # Check if episode is done
        done = new_state.timestep >= self.max_timesteps
        
        return StepResult(
            state=new_state,
            observation=observation,
            reward=num_collected,
            done=done,
        )

    @partial(jit, static_argnums=(0,))
    def _move_packed(self, pos_packed: Array, action: int) -> Array:
        """Move in packed space with toroidal wrapping - ultra-optimized."""
        # Unpack position
        x = pos_packed // self.grid_size
        y = pos_packed % self.grid_size
        
        # Apply movement with modulo for wrapping
        # This avoids all the conditional checks
        dx = jnp.array([0, 1, 0, -1])[action]  # x deltas
        dy = jnp.array([-1, 0, 1, 0])[action]  # y deltas
        
        new_x = (x + dx) % self.grid_size
        new_y = (y + dy) % self.grid_size
        
        # Pack back
        return new_x * self.grid_size + new_y

    @partial(jit, static_argnums=(0,))
    def _packed_distance_squared(self, pos1_packed: Array, pos2_packed: Array) -> Array:
        """Calculate squared distances in packed space (toroidal) - optimized."""
        # Vectorized unpacking
        row1 = pos1_packed // self.grid_size
        col1 = pos1_packed % self.grid_size
        row2 = pos2_packed // self.grid_size
        col2 = pos2_packed % self.grid_size

        # Toroidal differences - vectorized
        dx = jnp.abs(row2 - row1)
        dy = jnp.abs(col2 - col1)
        dx = jnp.minimum(dx, self.grid_size - dx)
        dy = jnp.minimum(dy, self.grid_size - dy)

        return dx * dx + dy * dy

    @partial(jit, static_argnums=(0,))
    def _get_observation_fast(
        self, agent_packed: Array, reward_packed: Array, collected: Array
    ) -> Observation:
        """Fast observation calculation with early masking."""
        # Pre-mask to avoid unnecessary distance calculations
        # Set collected reward positions to agent position (distance = 0)
        # Then we'll mask these out, avoiding the sqrt and exp calculations
        valid_rewards = ~collected
        
        # Only calculate distances for uncollected rewards
        distances_squared = self._packed_distance_squared(agent_packed, reward_packed)
        
        # Use pre-computed large value for collected rewards to exclude from min
        # This is more efficient than using inf which can cause numerical issues
        masked_distances = jnp.where(valid_rewards, distances_squared, self.large_distance)
        
        # Find minimum distance
        min_dist_squared = jnp.min(masked_distances)
        
        # Compute gradient, handling the case where all rewards are collected
        gradient = lax.cond(
            min_dist_squared < self.large_distance,
            lambda: jnp.exp(-jnp.sqrt(min_dist_squared) / self.decay_constant),
            lambda: jnp.array(0.0)  # All collected, no gradient
        )

        return Observation(gradient=gradient)
    
    def get_reward_history(self, state: WorldState) -> tuple[Array, Array, Array]:
        """Extract reward history from state.
        
        Returns:
            positions: Array of shape (history_count, 2) with spawn positions
            spawn_steps: Array of shape (history_count,) with spawn timesteps
            collect_steps: Array of shape (history_count,) with collection timesteps (-1 if not collected)
        """
        count = state.reward_history_count
        return (
            state.reward_history_positions[:count],
            state.reward_history_spawn_steps[:count],
            state.reward_history_collect_steps[:count],
        )
    
    def get_reward_history_full(self, state: WorldState) -> tuple[Array, Array, Array, Array]:
        """Get full reward history arrays without slicing (more efficient).
        
        Returns:
            positions: Full array of shape (max_history_size, 2)
            spawn_steps: Full array of shape (max_history_size,)
            collect_steps: Full array of shape (max_history_size,)
            count: Number of valid entries
        """
        return (
            state.reward_history_positions,
            state.reward_history_spawn_steps,
            state.reward_history_collect_steps,
            state.reward_history_count,
        )
