# models/phase_0_8/__init__.py
# keywords: [phase 0.8, snn agent, best-of-all-worlds]
"""
Phase 0.8: Principled synthesis of best ideas from phases 0.4-0.7

This implementation represents a careful balance between biological realism
and computational efficiency, incorporating lessons learned from all previous
phases.
"""

from .agent import SnnAgent
from .config import ExperimentConfig, NetworkParams, SnnAgentConfig

__all__ = ["ExperimentConfig", "NetworkParams", "SnnAgent", "SnnAgentConfig"]
