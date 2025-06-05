# keywords: [snn run, experiment runner, phase 0.5]
"""Main script to run Phase 0.5 SNN agent experiments."""

import argparse
from pathlib import Path
import jax.random as random
import numpy as np

from .config import SnnAgentConfig, WorldConfig, NetworkParams, ExperimentConfig
from .agent import SnnAgent


def main():
    parser = argparse.ArgumentParser(
        description="Run Phase 0.5 SNN Agent Experiments")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of episodes to run")
    parser.add_argument("--steps", type=int, default=10000,
                        help="Max steps per episode")
    parser.add_argument("--seed", type=int, default=42,
                        help="Master random seed")
    parser.add_argument("--export-dir", type=str,
                        default="experiments/phase_0_5", help="Directory for data export")
    parser.add_argument("--no-export", action="store_true",
                        help="Disable data exporting")

    args = parser.parse_args()

    # Create configurations
    world_config = WorldConfig(max_timesteps=args.steps)
    network_params = NetworkParams()
    exp_config = ExperimentConfig(
        n_episodes=args.episodes,
        seed=args.seed,
        export_dir=args.export_dir,
        enable_export=not args.no_export
    )

    config = SnnAgentConfig(
        world_config=world_config,
        network_params=network_params,
        exp_config=exp_config
    )

    # Create export directory
    if not args.no_export:
        Path(args.export_dir).mkdir(parents=True, exist_ok=True)

    print("🚀 Starting Phase 0.5 SNN Experiment")
    print("=" * 40)
    print(f"Running {args.episodes} episodes with seed {args.seed}")
    print(f"Max steps per episode: {args.steps}")
    print(f"Data export: {'ENABLED' if not args.no_export else 'DISABLED'}")
    print("=" * 40)

    # Initialize agent and run the full experiment
    agent = SnnAgent(config)
    all_summaries = agent.run_experiment()

    if len(all_summaries) > 1:
        print("\n" + "=" * 40)
        print("📊 Aggregate Experiment Results")
        print("=" * 40)
        avg_reward = np.mean([s['total_reward'] for s in all_summaries])
        std_reward = np.std([s['total_reward'] for s in all_summaries])
        print(f"  Average Total Reward: {avg_reward:.2f} ± {std_reward:.2f}")

        avg_collected = np.mean([s['rewards_collected']
                                for s in all_summaries])
        std_collected = np.std([s['rewards_collected'] for s in all_summaries])
        print(
            f"  Average Rewards Collected: {avg_collected:.1f} ± {std_collected:.1f}")


if __name__ == "__main__":
    main()
