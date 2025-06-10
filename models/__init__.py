# models/__init__.py
# keywords: [models package, agent registry, unified interface, latest agent]
"""
Models package for MetaLearning agents.
"""

from interfaces import AgentProtocol


def get_latest_agent() -> type[AgentProtocol]:
    """Get the latest agent implementation.

    Update this function when creating new agent versions.
    This ensures tests and runners always use the most recent implementation.
    """
    from .phase_0_14_neo.agent import NeoAgent

    return NeoAgent


__all__ = ["get_latest_agent"]
