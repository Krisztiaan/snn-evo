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

    def select_action(self, state: Any, gradient: Array, key: PRNGKey) -> Tuple[Any, Array, Dict[str, Array]]:
        """
        Pure, JIT-compilable function to select a single action for a world step.
        This can involve multiple internal agent steps (a 'thinking loop').

        Args:
            state: Current agent state.
            gradient: Observation from the world (float32 scalar in [0, 1]).
            key: JAX random key.

        Returns:
            Tuple of (new_agent_state, action, neural_data_for_logging).
            - new_agent_state: Updated agent state after thinking.
            - action: A single integer action (e.g., 0-8) for the world.
            - neural_data: A dict of representative neural data for the last step.
        """
        ...

