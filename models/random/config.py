# keywords: [random agent config, experiment configuration]
"""Configuration for random agent experiments."""

from typing import NamedTuple
from world.simple_grid_0001 import WorldConfig


class RandomAgentConfig(NamedTuple):
    """Configuration for random agent experiments."""
    # World configuration
    world_config: WorldConfig = WorldConfig()
    
    # Experiment configuration
    n_episodes: int = 1
    n_steps: int = 1000
    seed: int = 42
    
    # Export configuration
    export_dir: str = "experiments/random"
    export_full_trajectory: bool = True
    export_episode_summary: bool = True