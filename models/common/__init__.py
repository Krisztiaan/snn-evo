# keywords: [common models, base agent, unified interface]
"""Common model interfaces and utilities."""

from .base_agent import BaseAgent, AgentConfig
from .registry import AgentRegistry, register_agent

__all__ = ["BaseAgent", "AgentConfig", "AgentRegistry", "register_agent"]