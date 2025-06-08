# models/phase_0.5/__init__.py
# keywords: [snn agent, phase 0.5, biologically plausible]
"""
Phase 0.5: A modular, runnable, and biologically-plausible SNN agent.

This version integrates the best features from the Phase 0.4 prototypes
(E/I balance, homeostasis, metaplasticity, RPE-based learning) into a
clean, configurable, and reproducible experiment package.
"""

from .agent import SnnAgent
from .config import ExperimentConfig, NetworkParams, SnnAgentConfig

__version__ = "0.5.0"
__all__ = [
    "ExperimentConfig",
    "NetworkParams",
    "SnnAgent",
    "SnnAgentConfig",
]
