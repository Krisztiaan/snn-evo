# models/phase_0_12/__init__.py
# keywords: [phase 0.12, init, learning fixes]
"""Phase 0.12: Fixed learning dynamics for actual progress."""

from .agent import SnnAgent
from .config import SnnAgentConfig, NetworkParams, ExperimentConfig

__all__ = ['SnnAgent', 'SnnAgentConfig', 'NetworkParams', 'ExperimentConfig']
