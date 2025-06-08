# keywords: [random agent main, run experiment, baseline experiment]
"""Run random agent experiments."""

import argparse
from pathlib import Path

import jax

from models.random import RandomAgent, RandomAgentConfig
from world.simple_grid_0001 import WorldConfig


def main():
    parser = argparse.ArgumentParser(description="Run random agent in grid world")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes")
    parser.add_argument("--steps", type=int, default=1000, help="Steps per episode")
    parser.add_argument("--grid-size", type=int, default=50, help="Grid world size")
    parser.add_argument("--rewards", type=int, default=50, help="Number of rewards")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--export-dir", type=str, default="experiments/random", help="Export directory"
    )
    parser.add_argument("--no-trajectory", action="store_true", help="Don't export full trajectory")

    args = parser.parse_args()

    # Create configuration
    world_config = WorldConfig(
        grid_size=args.grid_size, n_rewards=args.rewards, max_timesteps=args.steps, seed=args.seed
    )

    config = RandomAgentConfig(
        world_config=world_config,
        n_episodes=args.episodes,
        n_steps=args.steps,
        seed=args.seed,
        export_dir=args.export_dir,
        export_full_trajectory=not args.no_trajectory,
        export_episode_summary=True,
    )

    # Create export directory
    Path(args.export_dir).mkdir(parents=True, exist_ok=True)

    print("Random Agent Experiment")
    print("=" * 50)
    print(f"Grid size: {args.grid_size}x{args.grid_size}")
    print(f"Rewards: {args.rewards}")
    print(f"Steps per episode: {args.steps}")
    print(f"Episodes: {args.episodes}")
    print(f"Export directory: {args.export_dir}")
    print(f"JAX devices: {jax.devices()}")
    print("=" * 50)

    # Run agent
    agent = RandomAgent(config)
    agent.run()

    print("\nExperiment complete!")


if __name__ == "__main__":
    main()
