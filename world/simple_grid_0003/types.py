# keywords: [grid world types, jax types, world state, observation]
"""Type definitions for simple grid world."""

from typing import NamedTuple

from jax import Array


class WorldState(NamedTuple):
    """Complete state of the grid world.

    All state is immutable and functional for JAX compatibility.
    """

    # Agent state
    agent_pos: tuple[Array, Array]  # (x, y) position as JAX arrays

    # Environment state
    reward_positions: Array  # Shape: (n_rewards, 2) - current active rewards
    reward_collected: Array  # Shape: (n_rewards,) boolean mask
    timestep: Array  # int32 scalar for JAX compatibility
    
    # Reward history tracking (JAX arrays for performance)
    # Pre-allocated arrays with max possible size
    reward_history_positions: Array  # Shape: (max_rewards, 2) - spawn positions
    reward_history_spawn_steps: Array  # Shape: (max_rewards,) - spawn timesteps
    reward_history_collect_steps: Array  # Shape: (max_rewards,) - collection timesteps (-1 if not collected)
    reward_history_count: Array  # Number of rewards in history (int32 scalar)
    reward_indices: Array  # Shape: (n_rewards,) - maps active slot to history index


class Observation(NamedTuple):
    """What the agent sees."""

    gradient: Array  # Distance-based signal to nearest reward (0-1)


class StepResult(NamedTuple):
    """Result of taking a step in the world."""

    state: WorldState
    observation: Observation  
    reward: Array  # Number of rewards collected this step (int32 scalar)
    done: Array  # Episode terminated (bool scalar)


class WorldConfig(NamedTuple):
    """Configuration for the grid world."""

    grid_size: int = 100
    n_rewards: int = 300
    max_timesteps: int = 50000
