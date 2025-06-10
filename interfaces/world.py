# keywords: [world protocol, type safety, stateful, minimal observation]
"""World interface protocols for type-safe implementations.

The World maintains internal state and only exposes a single gradient value:
- Gradient: Distance-based signal to nearest reward (0-1)
- When gradient = 1.0, agent is at reward location and automatically collects it

Actions are discrete: move forward/backward/stay (3) × turn left/right/no (3) = 9 total
"""

from typing import Protocol, runtime_checkable, Tuple, NamedTuple
from jax import Array
from jax.random import PRNGKey


class WorldState(NamedTuple):
    """Complete world state used by all implementations.
    
    This contains all necessary fields for a stateful grid world.
    All worlds use this exact state structure.
    """
    # Agent state
    agent_pos: Array  # Shape: (2,) for x,y position
    agent_dir: Array  # Scalar: 0-3 for N,E,S,W
    
    # Reward state
    reward_positions: Array  # Shape: (n_rewards, 2)
    reward_active: Array  # Shape: (n_rewards,) bool mask
    
    # Randomness
    key: Array  # JAX random key (as Array for tracing)
    
    # Reward tracking (for logging/analysis)
    reward_history_positions: Array  # Shape: (max_history, 2)
    reward_history_spawn_steps: Array  # Shape: (max_history,)
    reward_history_collect_steps: Array  # Shape: (max_history,)
    reward_history_count: Array  # Scalar: number of entries in history
    
    # Last gradient computed (for episode loop)
    last_gradient: Array  # Scalar: float32 in [0, 1]


@runtime_checkable
class WorldProtocol(Protocol):
    """Protocol for pure JAX world implementations.
    
    The world maintains all internal state (agent position, orientation, reward locations, etc.)
    and only exposes a gradient signal to the agent.
    
    When gradient = 1.0, the agent is at a reward location and automatically collects it.
    The reward is consumed and won't appear in future gradient calculations.
    """
    
    def reset(self, key: PRNGKey) -> Tuple[WorldState, Array]:
        """Reset world to initial state and return state and initial gradient.
        
        Args:
            key: JAX random key for randomized reward placement
        
        Returns:
            Tuple of:
            - state: World state (concrete subclass of WorldState)
            - gradient: float32 scalar in [0, 1], distance-based signal to nearest reward
        """
        ...
    
    def step(self, state: WorldState, action: Array) -> Tuple[WorldState, Array]:
        """Execute action and return new state and gradient observation.
        
        This method should be JIT-compilable for pure JAX execution.
        
        Args:
            state: Current world state
            action: Integer 0-8 encoding movement and rotation:
                - 0-2: Move forward with turn (left, none, right)
                - 3-5: Stay in place with turn (left, none, right)  
                - 6-8: Move backward with turn (left, none, right)
        
        Returns:
            Tuple of:
            - new_state: Updated world state
            - gradient: float32 scalar in [0, 1], distance-based signal to nearest reward
                       1.0 means agent is at reward (and collects it automatically)
        """
        ...
    
    def get_config(self) -> dict:
        """Get world configuration parameters used for initialization.
        
        Returns:
            Dict containing grid_size, n_rewards, max_timesteps, etc.
        """
        ...
    
    def get_reward_tracking(self) -> dict:
        """Get reward collection history at end of episode.
        
        Returns:
            Dict with arrays tracking:
            - positions: Where rewards were placed
            - spawn_steps: When each reward appeared
            - collect_steps: When each reward was collected (-1 if not collected)
        """
        ...