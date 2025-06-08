#!/usr/bin/env python3
# models/phase_0_12/quick_test.py
# keywords: [phase 0.12, quick test, learning validation]
"""Quick test for Phase 0.12 with fewer episodes and steps."""

from world.simple_grid_0001.types import WorldConfig
from models.phase_0_12 import SnnAgent, SnnAgentConfig, NetworkParams, ExperimentConfig
import sys
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))


def main():
    """Run a quick test."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Quick test Phase 0.12 SNN")
    parser.add_argument('--no-write', action='store_true',
                        help='Skip writing data to disk (for testing)')
    parser.add_argument('--performance', action='store_true',
                        help='Run in performance mode (no data export)')
    args = parser.parse_args()

    print("=== Phase 0.12 Quick Test ===")
    if args.no_write:
        print("- Running in no-write mode")
    if args.performance:
        print("- Running in performance mode")

    # Create config with fewer episodes and steps
    world_config = WorldConfig(
        grid_size=20,  # Smaller grid
        n_rewards=3,   # Fewer rewards
        max_timesteps=5000  # Much shorter episodes
    )

    exp_config = ExperimentConfig(
        n_episodes=5,  # Just 5 episodes
        seed=42,
        export_dir="experiments/phase_0_12/quick_test",
        enable_export=True,
        verbose=True
    )

    config = SnnAgentConfig(
        world_config=world_config,
        network_params=NetworkParams(),  # Use default fixed params
        exp_config=exp_config
    )

    # Create and run agent
    agent = SnnAgent(config)
    summaries = agent.run_experiment(
        performance_mode=args.performance, no_write=args.no_write)

    # Analysis
    print("\n=== Quick Test Results ===")
    for i, s in enumerate(summaries):
        print(f"Episode {i+1}: Total reward={s['total_reward']:.1f}, "
              f"Rewards collected={s['rewards_collected']}, "
              f"Temperature={s['current_temperature']:.3f}")

    # Check for improvement
    if len(summaries) >= 3:
        first_reward = summaries[0]['total_reward']
        last_reward = summaries[-1]['total_reward']
        improvement = last_reward - first_reward

        first_rewards = summaries[0]['rewards_collected']
        last_rewards = summaries[-1]['rewards_collected']
        rewards_improvement = last_rewards - first_rewards

        print(f"\nImprovement: {improvement:+.1f} total reward")
        print(
            f"Rewards collected improvement: {rewards_improvement:+d} rewards")

        if improvement > 0:
            print("✓ Learning is working!")
        else:
            print("✗ No improvement yet")

        if last_rewards > 0:
            print("✓ Agent is collecting rewards!")
        else:
            print("✗ Agent not collecting rewards")


if __name__ == "__main__":
    main()
