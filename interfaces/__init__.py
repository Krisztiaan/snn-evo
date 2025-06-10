# keywords: [interfaces, protocols, type safety, performance]
"""Type-safe interfaces for all modules.

These protocols define the contracts that implementations must follow,
enabling static type checking and avoiding runtime overhead.
"""

from .world import WorldProtocol, WorldState
from .agent import AgentProtocol
from .exporter import ExporterProtocol, EpisodeBufferProtocol, LogTimestepFunction
from .config import ExperimentConfig, WorldConfig, NeuralConfig, PlasticityConfig, AgentBehaviorConfig
from .episode_data import EpisodeData
from .runner import ProtocolRunner, create_experiment_config

__all__ = [
    # World
    "WorldProtocol",
    "WorldState",
    # Agent
    "AgentProtocol",
    # Exporter
    "ExporterProtocol",
    "EpisodeBufferProtocol",
    "LogTimestepFunction",
    # Config
    "ExperimentConfig",
    "WorldConfig",
    "NeuralConfig", 
    "PlasticityConfig",
    "AgentBehaviorConfig",
    # Data
    "EpisodeData",
    # Runner
    "ProtocolRunner",
    "create_experiment_config",
]