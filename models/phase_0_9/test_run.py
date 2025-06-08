#!/usr/bin/env python3
"""Test Phase 0.8 with small network for quick verification"""

from models.phase_0_8.agent import SnnAgent
from models.phase_0_8.config import SnnAgentConfig, WorldConfig, NetworkParams, ExperimentConfig
import sys
sys.path.append('../..')


# Small network for testing
config = SnnAgentConfig(
    world_config=WorldConfig(max_timesteps=1000),  # Short episode
    network_params=NetworkParams(
        NUM_SENSORY=8,
        NUM_PROCESSING=16,  # Very small
        NUM_READOUT=4
    ),
    exp_config=ExperimentConfig(
        n_episodes=1,
        export_dir="experiments/phase_0_8_test"
    )
)

print("Testing Phase 0.8 with small network...")
agent = SnnAgent(config)
summaries = agent.run_experiment()

print("\n✅ Test complete!")
print(f"Results: {summaries[0]}")
