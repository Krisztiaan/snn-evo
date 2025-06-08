#!/usr/bin/env python3
# models/phase_0_11/run.py
# keywords: [run experiment, phase 0.11, optimized, benchmarking]
"""
Run Phase 0.11 experiments with performance benchmarking.

This phase introduces:
1. Fully optimized grid world with JIT compilation
2. Precomputed exponential constants
3. Efficient spike handling
4. Better memory management
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import jax

from models.phase_0_11 import ExperimentConfig, NetworkParams, SnnAgent, SnnAgentConfig
from world.simple_grid_0003 import WorldConfig

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def create_experiment_configs():
    """Create different experiment configurations."""

    # Default configuration (baseline with optimizations)
    default_config = SnnAgentConfig(
        world_config=WorldConfig(
            grid_size=100,
            n_rewards=20,
            max_timesteps=10000,
            reward_value=10.0,
            proximity_reward=0.1,
            toroidal=True,
        ),
        network_params=NetworkParams(),
        exp_config=ExperimentConfig(n_episodes=10, seed=42, export_dir="experiments/phase_0_11"),
    )

    # Configuration for benchmarking
    benchmark_config = default_config._replace(
        exp_config=ExperimentConfig(
            n_episodes=3,  # Fewer episodes for quick benchmark
            seed=42,
            export_dir="experiments/phase_0_11/benchmark",
        )
    )

    # Configuration for learning curve analysis
    learning_config = default_config._replace(
        network_params=NetworkParams(
            BASE_LEARNING_RATE=0.002,  # Higher initial learning rate
            LEARNING_RATE_DECAY=0.9,  # Slower decay
            GRADIENT_REWARD_SCALE=3.0,  # Stronger gradient following
        ),
        exp_config=ExperimentConfig(
            n_episodes=20,  # More episodes to see learning
            seed=42,
            export_dir="experiments/phase_0_11/learning",
        ),
    )

    return {"default": default_config, "benchmark": benchmark_config, "learning": learning_config}


def benchmark_performance(config):
    """Run a quick benchmark to measure performance improvements."""
    print("\n=== Performance Benchmark ===")
    print("Running 3 episodes to measure average performance...")
    print("Performance mode enabled - no data export")

    # Time the full experiment
    start_time = time.time()

    agent = SnnAgent(config)
    summaries = agent.run_experiment(performance_mode=True)

    total_time = time.time() - start_time

    # Calculate statistics
    total_steps = sum(s["steps_taken"] for s in summaries)
    avg_episode_time = total_time / len(summaries)
    steps_per_second = total_steps / total_time

    print("\n=== Benchmark Results ===")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average episode time: {avg_episode_time:.2f}s")
    print(f"Total steps simulated: {total_steps:,}")
    print(f"Steps per second: {steps_per_second:,.0f}")
    print(f"Milliseconds per step: {1000 / steps_per_second:.2f}ms")

    # Save benchmark results
    benchmark_data = {
        "total_time": total_time,
        "avg_episode_time": avg_episode_time,
        "total_steps": total_steps,
        "steps_per_second": steps_per_second,
        "ms_per_step": 1000 / steps_per_second,
        "jax_devices": str(jax.devices()),
        "n_episodes": len(summaries),
    }

    output_dir = Path(config.exp_config.export_dir).parent / "benchmark_results"
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "phase_0_11_benchmark.json", "w") as f:
        json.dump(benchmark_data, f, indent=2)

    print(f"\nBenchmark results saved to: {output_dir / 'phase_0_11_benchmark.json'}")

    return summaries


def main():
    parser = argparse.ArgumentParser(description="Run Phase 0.11 SNN experiments")
    parser.add_argument(
        "--config",
        choices=["default", "benchmark", "learning"],
        default="default",
        help="Configuration to run",
    )
    parser.add_argument("--episodes", type=int, help="Override number of episodes")
    parser.add_argument("--seed", type=int, help="Override random seed")
    parser.add_argument("--export-dir", type=str, help="Override export directory")
    parser.add_argument("--benchmark", action="store_true", help="Run performance benchmark")
    parser.add_argument(
        "--no-write", action="store_true", help="Skip all file I/O operations (for benchmarking)"
    )

    args = parser.parse_args()

    # Get configuration
    configs = create_experiment_configs()
    config = configs[args.config]

    # Apply overrides
    if args.episodes is not None:
        config = config._replace(exp_config=config.exp_config._replace(n_episodes=args.episodes))

    if args.seed is not None:
        config = config._replace(exp_config=config.exp_config._replace(seed=args.seed))

    if args.export_dir is not None:
        config = config._replace(exp_config=config.exp_config._replace(export_dir=args.export_dir))

    # Print configuration
    print("=== Phase 0.11 Optimized SNN Agent ===")
    print("\nKey optimizations:")
    print("- JIT-compiled grid world simulation")
    print("- Precomputed exponential decay factors")
    print("- Efficient spike handling without type conversions")
    print("- Optimized memory access patterns")
    print("- Proper PRNG key management")

    print(f"\nConfiguration: {args.config}")
    print(f"Episodes: {config.exp_config.n_episodes}")
    print(
        f"Network size: {config.network_params.NUM_SENSORY + config.network_params.NUM_PROCESSING + config.network_params.NUM_READOUT} neurons"
    )
    print(f"World size: {config.world_config.grid_size}x{config.world_config.grid_size}")
    print(f"JAX devices: {jax.devices()}")
    if args.no_write:
        print("No-write mode: File I/O disabled (benchmarking mode)")

    # Run benchmark if requested
    if args.benchmark or args.config == "benchmark":
        summaries = benchmark_performance(configs["benchmark"])
    else:
        # Run regular experiment
        print("\nRunning experiment...")
        agent = SnnAgent(config)
        summaries = agent.run_experiment(no_write=args.no_write)

    # Print final summary
    print("\n=== Experiment Complete ===")
    print(f"Total episodes: {len(summaries)}")

    if len(summaries) > 0:
        avg_reward = sum(s["total_reward"] for s in summaries) / len(summaries)
        avg_collected = sum(s["rewards_collected"] for s in summaries) / len(summaries)

        print(f"Average total reward: {avg_reward:.2f}")
        print(f"Average rewards collected: {avg_collected:.1f}")

        # Check for learning
        if len(summaries) >= 5:
            early_avg = sum(s["total_reward"] for s in summaries[:5]) / 5
            late_avg = sum(s["total_reward"] for s in summaries[-5:]) / 5
            improvement = late_avg - early_avg

            print("\nLearning progress:")
            print(f"First 5 episodes avg: {early_avg:.2f}")
            print(f"Last 5 episodes avg: {late_avg:.2f}")
            print(f"Improvement: {improvement:+.2f} ({improvement / early_avg * 100:+.1f}%)")


if __name__ == "__main__":
    main()
