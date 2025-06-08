#!/usr/bin/env python3
"""Quick test script for Phase 0.8"""

from phase_0_8.agent import SnnAgent
from phase_0_8.config import SnnAgentConfig, WorldConfig, ExperimentConfig
import sys
sys.path.append('../..')


def main():
    print("Testing Phase 0.8...")

    # Minimal config for testing
    config = SnnAgentConfig(
        world_config=WorldConfig(max_timesteps=1000),
        exp_config=ExperimentConfig(
            n_episodes=1,
            export_dir="test_output",
            enable_export=True
        )
    )

    # Create agent
    print("\nCreating agent...")
    agent = SnnAgent(config)

    # Run experiment
    print("\nRunning experiment...")
    summaries = agent.run_experiment()

    print("\nTest complete!")
    print(f"Summary: {summaries[0]}")


if __name__ == "__main__":
    main()
