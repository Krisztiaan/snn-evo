# keywords: [agent protocol, type safety, stateful, minimal interface]
"""Agent interface protocols for type-safe implementations."""

from typing import Protocol, runtime_checkable, Any, Tuple, Dict

from jax import Array
from jax.random import PRNGKey

from .config import ExperimentConfig
from .exporter import ExporterProtocol


@runtime_checkable
class AgentProtocol(Protocol):
    """Protocol for stateful agent implementations.

    Agents maintain internal state across timesteps within an episode.
    They receive only gradient observations and output actions.
    """

    # Agent metadata (class attributes)
    VERSION: str  # e.g., "1.0.0"
    MODEL_NAME: str  # e.g., "SNN-STDP-Reward"
    DESCRIPTION: str  # e.g., "Spiking neural network with STDP and reward modulation"

    def __init__(self, config: ExperimentConfig, exporter: ExporterProtocol) -> None:
        """Initialize agent with configuration and exporter.

        Args:
            config: Complete experiment configuration
            exporter: Data exporter for logging
        """
        ...

    def reset(self, key: PRNGKey) -> Any:
        """Reset agent's internal state for new episode.

        Args:
            key: JAX random key for stochastic initialization
            
        Returns:
            Initial agent state (implementation-specific type)
        """
        ...

    def step(self, state: Any, gradient: Array, key: PRNGKey) -> Tuple[Any, Array, Dict[str, Array]]:
        """
        Pure, JIT-compilable agent step function.
        
        Args:
            state: Current agent state.
            gradient: Observation from the world (float32 scalar in [0, 1]).
            key: JAX random key.
            
        Returns:
            Tuple of (new_agent_state, action, neural_data).
            - new_agent_state: Updated agent state (implementation-specific type)
            - action: Array scalar int32 0-8 encoding movement and rotation
            - neural_data: Dict of neural data for logging (e.g. {'v': membrane_potentials})
        """
        ...

