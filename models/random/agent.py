# keywords: [random agent, protocol compliant, jax, stateful]
"""Random agent - Compliant with AgentProtocol."""

from typing import Optional, Tuple, Dict, Any
import jax
import jax.numpy as jnp
from jax import Array
from jax.random import PRNGKey, split, randint
from functools import partial
from collections import namedtuple

from interfaces import AgentProtocol, ExperimentConfig, ExporterProtocol, EpisodeData

# Simple state for random agent
RandomAgentState = namedtuple('RandomAgentState', ['timestep', 'reward_count'])


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
        
        # Internal state
        self.state = None
    
    def reset(self, key: PRNGKey) -> RandomAgentState:
        """Reset agent's internal state for new episode."""
        self.state = RandomAgentState(timestep=0, reward_count=0)
        return self.state
    
    @staticmethod
    @jax.jit
    def _act_pure(gradient: Array, key: PRNGKey) -> Array:
        """Pure JAX action selection."""
        # Random action selection
        action = randint(key, (), 0, 9)
        return action
    
    def select_action(self, state: Any, gradient: Array, key: PRNGKey) -> Tuple[Any, Array, Dict[str, Array]]:
        """Select action following the new protocol.
        
        Args:
            state: Current agent state
            gradient: float32 scalar in [0, 1], distance signal to nearest reward
            key: JAX random key for stochastic action selection
            
        Returns:
            Tuple of:
                - new_state: Updated agent state
                - action: Array scalar int32 0-8 encoding movement and rotation
                - neural_data: Empty dict for random agent
        """
        # Call the pure JIT function
        new_state, action, neural_data = self.step(state, gradient, key)
        
        # Update the host-side state
        self.state = new_state
        
        return new_state, action, neural_data
    
    @staticmethod
    @jax.jit
    def step(state: RandomAgentState, gradient: Array, key: PRNGKey) -> Tuple[RandomAgentState, Array, Dict[str, Array]]:
        """Pure, JIT-compilable agent step function."""
        # Random action selection
        action = randint(key, (), 0, 9)
        
        # Update state
        reward_received = gradient >= 0.99
        new_state = RandomAgentState(
            timestep=state.timestep + 1,
            reward_count=state.reward_count + jnp.where(reward_received, 1, 0)
        )
        
        # Minimal neural data for random agent
        neural_data = {
            'v': jnp.zeros(1),  # Dummy membrane potential
            'spikes': jnp.zeros(1, dtype=bool)  # No spikes
        }
        
        return new_state, action, neural_data