#!/usr/bin/env python3
"""Quick test to verify Phase 0_8 works"""

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

from models.phase_0_8.agent import SnnAgent
from models.phase_0_8.config import ExperimentConfig, NetworkParams, SnnAgentConfig, WorldConfig

# Minimal config for quick testing
config = SnnAgentConfig(
    world_config=WorldConfig(max_timesteps=2000),  # Short episode
    network_params=NetworkParams(
        NUM_SENSORY=8,  # Smaller network for faster init
        NUM_PROCESSING=32,  # Much smaller!
        NUM_READOUT=8,
    ),
    exp_config=ExperimentConfig(n_episodes=1, export_dir="test_output", enable_export=True),
)

print("Creating agent with smaller network...")
agent = SnnAgent(config)

print("\nRunning quick test...")
summaries = agent.run_experiment()

print("\nTest complete!")
print(f"Results: {summaries[0]}")
