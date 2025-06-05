# keywords: [random agent, baseline model, grid world agent]
"""Random agent - baseline model that selects random actions."""

from .agent import RandomAgent
from .config import RandomAgentConfig

__version__ = "0.1.0"
__all__ = ["RandomAgent", "RandomAgentConfig"]