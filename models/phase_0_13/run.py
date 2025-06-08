#!/usr/bin/env python3
# models/phase_0_13/run.py
# keywords: [phase 0.13, run experiment, detailed logging, learning analysis]
"""Run Phase 0.13 experiment with detailed learning analysis logging."""

import argparse
import sys
from pathlib import Path

from models.phase_0_13 import SnnAgent, SnnAgentConfig

sys.path.append(str(Path(__file__).parent.parent.parent))


def main():
    """Run the experiment."""
    parser = argparse.ArgumentParser(description="Run Phase 0.13 SNN experiment")
    parser.add_argument(
        "--no-write", action="store_true", help="Skip writing data to disk (for testing)"
    )
    args = parser.parse_args()

    print("=== Phase 0.13: SNN with Detailed Learning Analysis ===")
    print("Key features:")
    print("- Full compatibility with new exporter features")
    print("- Logs discrete events (e.g., reward collection)")
    print("- Logs a sample of individual synaptic weight changes")
    print("- Retains all learning fixes and optimizations from Phase 0.12")

    if args.no_write:
        print("- Running in no-write mode (data not saved)")
    print()

    config = SnnAgentConfig()
    agent = SnnAgent(config)

    summaries = agent.run_experiment(no_write=args.no_write)

    print("\n=== Experiment Complete ===")
    if not summaries:
        print("No episodes were run.")
        return

    print(f"Total episodes: {len(summaries)}")
    print(f"Average total reward: {sum(s['total_reward'] for s in summaries) / len(summaries):.2f}")
    print(
        f"Average rewards collected: {sum(s['rewards_collected'] for s in summaries) / len(summaries):.1f}"
    )

    if len(summaries) >= 10:
        first_5_avg = sum(s["total_reward"] for s in summaries[:5]) / 5
        last_5_avg = sum(s["total_reward"] for s in summaries[-5:]) / 5
        improvement = last_5_avg - first_5_avg
        improvement_pct = (improvement / first_5_avg) * 100 if first_5_avg > 0 else 0

        print("\nLearning progress:")
        print(f"First 5 episodes avg: {first_5_avg:.2f}")
        print(f"Last 5 episodes avg: {last_5_avg:.2f}")
        print(f"Improvement: {improvement:+.2f} ({improvement_pct:+.1f}%)")


if __name__ == "__main__":
    main()