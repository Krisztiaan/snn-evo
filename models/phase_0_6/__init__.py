# models/phase_0_6/__init__.py
# keywords: [snn agent, phase 0.6, goal-directed, functional]
"""
Phase 0.6: A functional, goal-directed, and biologically-plausible SNN agent.

This version fixes the critical flaws of Phase 0.5, ensuring that the SNN's
output drives agent behavior. It features network-driven action selection,
a properly scaled input signal, and functional metaplasticity, resulting
in an active, learning agent.
"""

from .agent import SnnAgent
from .config import ExperimentConfig, NetworkParams, SnnAgentConfig

__version__ = "0.6.0"
__all__ = [
    "ExperimentConfig",
    "NetworkParams",
    "SnnAgent",
    "SnnAgentConfig",
]
