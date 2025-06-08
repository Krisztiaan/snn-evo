#!/usr/bin/env python3
# models/phase_0_12/run.py
# keywords: [phase 0.12, run experiment, fixed learning]
"""Run Phase 0.12 experiment with fixed learning dynamics."""

import argparse
import sys
from pathlib import Path

from models.phase_0_12 import SnnAgent, SnnAgentConfig

sys.path.append(str(Path(__file__).parent.parent.parent))


def main():
    """Run the experiment."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run Phase 0.12 SNN experiment")
    parser.add_argument(
        "--no-write", action="store_true", help="Skip writing data to disk (for testing)"
    )
    parser.add_argument(
        "--performance", action="store_true", help="Run in performance mode (no data export)"
    )
    args = parser.parse_args()

    print("=== Phase 0.12: Fixed Learning Dynamics ===")
    print("Key fixes:")
    print("- Learning rate: 0.1 (was 0.001)")
    print("- STDP amplitudes: 0.02 (was 0.005)")
    print("- Baseline current: 3.0 (was 0.5)")
    print("- Temperature annealing for exploration")
    print("- Reward amplification and boost on collection")
    print("- More plastic connections (input and output)")

    if args.no_write:
        print("- Running in no-write mode (data not saved)")
    if args.performance:
        print("- Running in performance mode (no data export)")
    print()

    # Create agent with default config
    config = SnnAgentConfig()
    agent = SnnAgent(config)

    # Run experiment
    summaries = agent.run_experiment(performance_mode=args.performance, no_write=args.no_write)

    # Final summary
    print("\n=== Experiment Complete ===")
    print(f"Total episodes: {len(summaries)}")
    print(f"Average total reward: {sum(s['total_reward'] for s in summaries) / len(summaries):.2f}")
    print(
        f"Average rewards collected: {sum(s['rewards_collected'] for s in summaries) / len(summaries):.1f}"
    )

    # Check for learning
    if len(summaries) >= 10:
        first_5_avg = sum(s["total_reward"] for s in summaries[:5]) / 5
        last_5_avg = sum(s["total_reward"] for s in summaries[-5:]) / 5
        improvement = last_5_avg - first_5_avg
        improvement_pct = (improvement / first_5_avg) * 100 if first_5_avg > 0 else 0

        print("\nLearning progress:")
        print(f"First 5 episodes avg: {first_5_avg:.2f}")
        print(f"Last 5 episodes avg: {last_5_avg:.2f}")
        print(f"Improvement: {improvement:+.2f} ({improvement_pct:+.1f}%)")

        # Check actual reward progress
        first_5_rewards = sum(s["rewards_collected"] for s in summaries[:5]) / 5
        last_5_rewards = sum(s["rewards_collected"] for s in summaries[-5:]) / 5
        rewards_improvement = last_5_rewards - first_5_rewards

        print("\nReward collection progress:")
        print(f"First 5 episodes avg: {first_5_rewards:.2f}")
        print(f"Last 5 episodes avg: {last_5_rewards:.2f}")
        print(f"Improvement: {rewards_improvement:+.2f}")


if __name__ == "__main__":
    main()

