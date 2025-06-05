# keywords: [grid world types, jax types, world state, observation]
"""Type definitions for simple grid world."""

from typing import NamedTuple, Tuple
import jax.numpy as jnp
from jax import Array


class WorldState(NamedTuple):
    """Complete state of the grid world.
    
    All state is immutable and functional for JAX compatibility.
    """
    # Agent state
    agent_pos: Tuple[int, int]  # (x, y) position
    
    # Environment state  
    reward_positions: Array  # Shape: (n_rewards, 2)
    reward_collected: Array  # Shape: (n_rewards,) boolean mask
    
    # Metrics
    total_reward: float
    timestep: int
    

class Observation(NamedTuple):
    """What the agent sees."""
    gradient: float  # Distance-based signal to nearest reward (0-1)
    

class StepResult(NamedTuple):
    """Result of taking a step in the world."""
    state: WorldState
    observation: Observation
    reward: float
    done: bool
    

class WorldConfig(NamedTuple):
    """Configuration for the grid world."""
    grid_size: int = 100
    n_rewards: int = 300
    max_timesteps: int = 50000
    reward_value: float = 10.0
    proximity_reward: float = 0.5
    toroidal: bool = True  # Wrap around edges
    seed: int = 0