"""Phase 0.11: Optimized SNN Agent with JIT compilation and performance improvements."""

from .agent import SnnAgent, OptimizedSnnAgent
from .config import SnnAgentConfig, NetworkParams, ExperimentConfig

__all__ = ['SnnAgent', 'OptimizedSnnAgent', 'SnnAgentConfig', 'NetworkParams', 'ExperimentConfig']