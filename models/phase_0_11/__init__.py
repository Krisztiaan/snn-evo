"""Phase 0.11: Optimized SNN Agent with JIT compilation and performance improvements."""

from .agent import OptimizedSnnAgent, SnnAgent
from .config import ExperimentConfig, NetworkParams, SnnAgentConfig

__all__ = ["ExperimentConfig", "NetworkParams", "OptimizedSnnAgent", "SnnAgent", "SnnAgentConfig"]
