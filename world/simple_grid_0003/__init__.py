# keywords: [optimized grid world, performance focused, packed positions]
"""Ultra-optimized JAX grid world implementation.

This is a drop-in replacement for simple_grid_0001 or simple_grid_0002.
Just change your import:
    from simple_grid_0001 import SimpleGridWorld, WorldConfig
    # to:
    from simple_grid_0003 import SimpleGridWorld, WorldConfig
"""

from .world import SimpleGridWorld
from ..simple_grid_0001.types import WorldConfig, WorldState, Observation, StepResult

__version__ = "0.3.0"
__all__ = [
    "SimpleGridWorld",
    "WorldState", 
    "Observation",
    "StepResult",
    "WorldConfig",
]