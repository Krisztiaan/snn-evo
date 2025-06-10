# keywords: [agent protocol, type safety, stateful, minimal interface]
"""Agent interface protocols for type-safe implementations."""

from typing import Protocol, runtime_checkable

from jax import Array
from jax.random import PRNGKey

from .config import ExperimentConfig
from .episode_data import EpisodeData
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

    def reset(self, key: PRNGKey) -> None:
        """Reset agent's internal state for new episode.

        Args:
            key: JAX random key for stochastic initialization
        """
        ...

    def act(self, gradient: Array, key: PRNGKey) -> Array:
        """Select action based on gradient observation.

        This method should be JIT-compilable for pure JAX execution.

        Args:
            gradient: float32 scalar in [0, 1], distance signal to nearest reward
            key: JAX random key for stochastic action selection

        Returns:
            action: Array scalar int32 0-8 encoding movement and rotation
        """
        ...

    def get_episode_data(self) -> EpisodeData:
        """Get standardized episode data for logging after episode ends.

        Returns:
            EpisodeData structure with trajectory and optional neural/learning data
        """
        ...
