# keywords: [minimal world, stateful, protocol compliant, pure jax, optimized]
"""Minimal stateful grid world implementation with direct JAX usage."""

from typing import Dict, Tuple
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array, lax
from jax.random import PRNGKey

from interfaces import WorldConfig, WorldState


class MinimalGridWorld:
    """Minimal grid world with internal state management.
    
    Adheres to WorldProtocol:
    - reset(key) -> gradient
    - step(action) -> gradient
    - get_config() -> dict
    - get_reward_tracking() -> dict
    """
    
    VERSION = "0.0.4"
    
    # Static method for maximum performance
    @staticmethod
    @partial(jax.jit, static_argnames=('grid_size', 'n_rewards'))
    def _step_static(state: WorldState, action: Array, grid_size: int, n_rewards: int, gradient_scale: float, dir_vectors: Array, action_to_move_turn: Array) -> Tuple[WorldState, Array]:
        """Static step function with all config as arguments."""
        
        # Decode action directly
        move_turn = action_to_move_turn[action]
        
        # Update direction
        new_dir = (state.agent_dir + move_turn[1] + 4) % 4
        
        # Calculate movement
        dir_vec = dir_vectors[new_dir]
        new_pos = (state.agent_pos + move_turn[0] * dir_vec) % grid_size
        
        # Check reward collection using squared distances
        diffs = state.reward_positions - new_pos[None, :]
        dist_sq = jnp.sum(diffs * diffs, axis=1)
        at_reward = (dist_sq < 0.25) & state.reward_active
        
        # Conditional update with lax.cond
        any_collected = jnp.any(at_reward)
        
        def update_with_collection():
            key, subkey = jax.random.split(state.key)
            new_positions = jax.random.randint(subkey, (n_rewards, 2), 0, grid_size)
            updated_positions = jnp.where(at_reward[:, None], new_positions, state.reward_positions)
            return state._replace(
                agent_pos=new_pos,
                agent_dir=new_dir,
                reward_positions=updated_positions,
                key=key
            )
        
        def update_no_collection():
            return state._replace(agent_pos=new_pos, agent_dir=new_dir)
        
        new_state = lax.cond(any_collected, update_with_collection, update_no_collection)
        
        # Inline gradient calculation
        reward_diffs = new_state.reward_positions - new_pos[None, :]
        reward_dist_sq = jnp.sum(reward_diffs * reward_diffs, axis=1)
        masked_dist_sq = jnp.where(new_state.reward_active, reward_dist_sq, jnp.inf)
        min_dist_sq = jnp.min(masked_dist_sq)
        
        gradient = jnp.where(
            min_dist_sq < 0.25,
            1.0,
            jnp.exp(-jnp.sqrt(min_dist_sq) / gradient_scale)
        )
        
        # Update last_gradient in state
        final_state = new_state._replace(last_gradient=gradient)
        
        return final_state, gradient
    
    @staticmethod
    @jax.jit
    def _calculate_gradient_static(agent_pos: Array, reward_positions: Array, reward_active: Array, gradient_scale: float) -> Array:
        """Static gradient calculation for maximum performance."""
        # Use squared distances (avoid expensive sqrt)
        diffs = reward_positions - agent_pos[None, :]
        dist_sq = jnp.sum(diffs * diffs, axis=1)
        
        # Mask inactive rewards
        masked_dist_sq = jnp.where(reward_active, dist_sq, jnp.inf)
        min_dist_sq = jnp.min(masked_dist_sq)
        
        # Gradient: 1.0 at reward, exponential decay
        return jnp.where(
            min_dist_sq < 0.25,  # 0.5^2 = 0.25
            1.0,
            jnp.exp(-jnp.sqrt(min_dist_sq) / gradient_scale)
        )
    
    def __init__(self, config: WorldConfig):
        """Initialize world with configuration."""
        self.config = config
        self.grid_size = config.grid_size
        self.n_rewards = config.n_rewards
        self.max_timesteps = config.max_timesteps
        
        # Pre-allocate tracking arrays
        self.max_history = self.n_rewards * 10
        
        # Pre-compute constants
        self.gradient_scale = self.grid_size / 4.0
        
        # Direction vectors: N, E, S, W
        self.dir_vectors = jnp.array([
            [0, -1],  # North
            [1, 0],   # East
            [0, 1],   # South
            [-1, 0]   # West
        ])
        
        # Action to (move, turn) mapping
        self.action_to_move_turn = jnp.array([
            [1, -1],   # 0: forward + left
            [1, 0],    # 1: forward + none
            [1, 1],    # 2: forward + right
            [0, -1],   # 3: stay + left
            [0, 0],    # 4: stay + none
            [0, 1],    # 5: stay + right
            [-1, -1],  # 6: backward + left
            [-1, 0],   # 7: backward + none
            [-1, 1],   # 8: backward + right
        ])
        
    
    def reset(self, key: PRNGKey) -> Tuple[WorldState, Array]:
        """Reset world and return initial state and gradient observation."""
        # Agent starts at center
        agent_pos = jnp.array([self.grid_size // 2, self.grid_size // 2])
        agent_dir = jnp.array(0)  # North
        
        # Initialize rewards in ring
        angles = jnp.linspace(0, 2 * jnp.pi, self.n_rewards, endpoint=False)
        radius = self.grid_size // 3
        center = self.grid_size // 2
        
        reward_x = jnp.clip(center + radius * jnp.cos(angles), 0, self.grid_size - 1).astype(jnp.int32)
        reward_y = jnp.clip(center + radius * jnp.sin(angles), 0, self.grid_size - 1).astype(jnp.int32)
        reward_positions = jnp.stack([reward_x, reward_y], axis=1)
        
        # Initialize tracking
        history_positions = jnp.zeros((self.max_history, 2), dtype=jnp.int32)
        history_positions = history_positions.at[:self.n_rewards].set(reward_positions)
        history_spawn_steps = jnp.full(self.max_history, -1, dtype=jnp.int32)
        history_spawn_steps = history_spawn_steps.at[:self.n_rewards].set(0)
        history_collect_steps = jnp.full(self.max_history, -1, dtype=jnp.int32)
        
        # Calculate initial gradient
        initial_gradient = self._calculate_gradient_static(agent_pos, reward_positions, jnp.ones(self.n_rewards, dtype=bool), self.gradient_scale)
        
        state = WorldState(
            agent_pos=agent_pos,
            agent_dir=agent_dir,
            reward_positions=reward_positions,
            reward_active=jnp.ones(self.n_rewards, dtype=bool),
            key=key,
            reward_history_positions=history_positions,
            reward_history_spawn_steps=history_spawn_steps,
            reward_history_collect_steps=history_collect_steps,
            reward_history_count=jnp.array(self.n_rewards, dtype=jnp.int32),
            last_gradient=initial_gradient
        )
        
        return state, initial_gradient
    
    def step(self, state: WorldState, action: int) -> Tuple[WorldState, Array]:
        """Execute action and return new state and gradient observation."""
        return self._step_static(
            state, action, 
            self.grid_size, self.n_rewards, self.gradient_scale,
            self.dir_vectors, self.action_to_move_turn
        )
    
    def get_config(self) -> Dict:
        """Get world configuration."""
        return {
            "version": self.VERSION,
            "grid_size": self.config.grid_size,
            "n_rewards": self.config.n_rewards,
            "max_timesteps": self.config.max_timesteps
        }
    
    def get_reward_tracking(self, state: WorldState) -> Dict[str, Array]:
        """Get reward collection history from state."""
        # Return only valid entries
        count = state.reward_history_count
        return {
            "positions": state.reward_history_positions[:count],
            "spawn_steps": state.reward_history_spawn_steps[:count],
            "collect_steps": state.reward_history_collect_steps[:count]
        }
    
    # Compatibility wrapper
    def _calculate_gradient(self, agent_pos: Array, reward_positions: Array, reward_active: Array) -> Array:
        """Instance method wrapper for compatibility."""
        return self._calculate_gradient_static(agent_pos, reward_positions, reward_active, self.gradient_scale)