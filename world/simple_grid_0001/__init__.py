# keywords: [simple grid world, jax environment]
"""Simple Grid World - Minimal JAX-compatible navigation environment."""

from .types import WorldState, Observation, StepResult, WorldConfig
from .world import SimpleGridWorld

__version__ = "0.1.0"
__all__ = [
    "SimpleGridWorld",
    "WorldState", 
    "Observation",
    "StepResult",
    "WorldConfig",
]