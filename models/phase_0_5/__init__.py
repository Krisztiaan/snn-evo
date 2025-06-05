# models/phase_0.5/__init__.py
# keywords: [snn agent, phase 0.5, biologically plausible]
"""
Phase 0.5: A modular, runnable, and biologically-plausible SNN agent.

This version integrates the best features from the Phase 0.4 prototypes
(E/I balance, homeostasis, metaplasticity, RPE-based learning) into a
clean, configurable, and reproducible experiment package.
"""

from .config import SnnAgentConfig, NetworkParams, ExperimentConfig
from .agent import SnnAgent

__version__ = "0.5.0"
__all__ = [
    "SnnAgent",
    "SnnAgentConfig",
    "NetworkParams",
    "ExperimentConfig",
]