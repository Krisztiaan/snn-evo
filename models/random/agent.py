# keywords: [random agent, protocol compliant, jax, stateful]
"""Random agent - Compliant with AgentProtocol."""

from typing import Optional, Tuple
import jax
import jax.numpy as jnp
from jax import Array
from jax.random import PRNGKey, split, randint
from functools import partial

from interfaces import AgentProtocol, ExperimentConfig, ExporterProtocol, EpisodeData


class RandomAgent:
    """Random agent that complies with AgentProtocol.
    
    Selects random actions at each timestep.
    """
    
    # Agent metadata (required by protocol)
    VERSION = "2.0.0"
    MODEL_NAME = "Random-Baseline"
    DESCRIPTION = "Random action selection baseline agent"
    
    def __init__(self, config: ExperimentConfig, exporter: ExporterProtocol) -> None:
        """Initialize agent with configuration and exporter."""
        self.config = config
        self.exporter = exporter
        
        # Internal state - we'll collect data in the episode buffer
        self.episode_buffer = None
        self.log_timestep_fn = None
        self.current_timestep = 0
    
    def reset(self, key: PRNGKey) -> None:
        """Reset agent's internal state for new episode."""
        self.current_timestep = 0
    
    @staticmethod
    @jax.jit
    def _act_pure(gradient: Array, key: PRNGKey) -> Array:
        """Pure JAX action selection."""
        # Random action selection
        action = randint(key, (), 0, 9)
        return action
    
    def act(self, gradient: Array, key: PRNGKey) -> Array:
        """Select random action based on gradient observation.
        
        Args:
            gradient: float32 scalar in [0, 1], distance signal to nearest reward
            key: JAX random key for stochastic action selection
            
        Returns:
            action: Array scalar int32 0-8 encoding movement and rotation
        """
        # Use JIT-compiled pure function
        action = self._act_pure(gradient, key)
        
        # Track timestep for buffer management
        self.current_timestep += 1
        
        return action
    
    def get_episode_data(self) -> EpisodeData:
        """Get standardized episode data for logging after episode ends."""
        # The episode buffer contains all the data
        # Extract what we need for EpisodeData
        if self.episode_buffer is not None:
            # Get data up to current timestep
            actions = self.episode_buffer.actions[:self.current_timestep]
            gradients = self.episode_buffer.gradients[:self.current_timestep]
            
            # Count reward events
            reward_count = jnp.sum(gradients == 1.0)
        else:
            # No episode run yet
            actions = jnp.array([], dtype=jnp.int32)
            gradients = jnp.array([], dtype=jnp.float32)
            reward_count = 0
        
        return EpisodeData(
            actions=actions,
            gradients=gradients,
            # No neural data for random agent
            neural_states=None,
            spikes=None,
            # No weight data
            weights_initial=None,
            weights_final=None,
            weight_changes=None,
            # No learning signals
            eligibility_traces=None,
            dopamine_levels=None,
            # Performance metrics
            total_reward_events=int(reward_count),
            exploration_entropy=None  # Could compute action entropy if needed
        )
    
    def start_episode(self, episode_id: int) -> Tuple:
        """Start new episode and get buffer and logging function from exporter."""
        self.episode_buffer, self.log_timestep_fn = self.exporter.start_episode(episode_id)
        self.current_timestep = 0
        return self.episode_buffer, self.log_timestep_fn