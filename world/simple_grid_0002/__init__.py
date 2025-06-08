"""Optimized Simple Grid World 0002 - JIT compiled version.

This is a drop-in replacement for simple_grid_0001.
Just change your import:
    from simple_grid_0001 import SimpleGridWorld, WorldConfig
    # to:
    from simple_grid_0002 import SimpleGridWorld, WorldConfig
"""

from .world import SimpleGridWorld
from simple_grid_0001.types import WorldConfig, WorldState, Observation, StepResult

__version__ = "0.2.0"
__all__ = [
    "SimpleGridWorld",
    "WorldState",
    "Observation",
    "StepResult",
    "WorldConfig",
]
