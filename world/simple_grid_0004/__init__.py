# keywords: [minimal world, protocol compliant, stateful, performance]
"""Minimal grid world implementation adhering to WorldProtocol.

This implementation:
- Maintains all state internally
- Returns only gradient observations
- Automatically collects rewards when gradient = 1.0
- Supports 9 actions (move × turn)
"""

from .world import MinimalGridWorld
from interfaces import WorldConfig

__version__ = "0.0.4"
__all__ = ["MinimalGridWorld", "WorldConfig"]