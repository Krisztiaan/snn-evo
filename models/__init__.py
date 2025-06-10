# models/__init__.py
# keywords: [models package, agent registry, unified interface, latest agent]
"""
Models package for MetaLearning agents.

This package provides:
- Various agent implementations (random, phase_0_13, etc.)
- Unified runner interface for all agents
- Configuration management and export utilities
- Helper to get latest agent implementation
"""

from typing import Type, TYPE_CHECKING
from interfaces import AgentProtocol

# Avoid circular import by deferring runner imports
if TYPE_CHECKING:
    from .runner import ModelRunner, AgentRegistry

def get_latest_agent() -> Type[AgentProtocol]:
    """Get the latest agent implementation.
    
    Update this function when creating new agent versions.
    This ensures tests and runners always use the most recent implementation.
    """
    # Import the latest implementation
    # Update this import when creating new versions
    from .random import RandomAgent
    
    return RandomAgent

# Export runner classes when accessed
def __getattr__(name):
    if name == "ModelRunner":
        from .runner import ModelRunner
        return ModelRunner
    elif name == "AgentRegistry":
        from .runner import AgentRegistry
        return AgentRegistry
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["ModelRunner", "AgentRegistry", "get_latest_agent"]