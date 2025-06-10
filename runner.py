#!/usr/bin/env python3
# keywords: [unified runner, cli, experiment runner, jax performance, protocol compliant]
"""
Unified CLI runner for all metalearning experiments.

This is the main entry point for running experiments. It supports JIT compilation,
vmap for parallel episodes, and configuration via JSON files.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any

import jax
from jax.random import PRNGKey

# Import protocol interfaces and implementations
from interfaces import ExperimentConfig, ProtocolRunner
from interfaces.config import WorldConfig, NeuralConfig, PlasticityConfig, AgentBehaviorConfig
from world.simple_grid_0004 import MinimalGridWorld
from export.jax_data_exporter import JaxDataExporter
from models.random.agent import RandomAgent
from models.phase_0_14_neo.agent import NeoAgent

# Agent registry to select agents from the command line
AGENT_REGISTRY = {
    "random": RandomAgent,
    "neo": NeoAgent,
}


def create_agent(agent_name: str, config: ExperimentConfig, exporter: JaxDataExporter):
    """Create an agent instance based on its name."""
    agent_name = agent_name.lower()
    if agent_name not in AGENT_REGISTRY:
        raise ValueError(f"Unknown agent: {agent_name}. Available: {list(AGENT_REGISTRY.keys())}")

    agent_class = AGENT_REGISTRY[agent_name]
    return agent_class(config, exporter)


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from a JSON file."""
    with open(config_path, "r") as f:
        return json.load(f)


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override config into base config."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result.get(key), dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def create_experiment_config(args) -> ExperimentConfig:
    """Create a final ExperimentConfig from defaults, JSON files, and CLI args."""
    # Start with a sensible default configuration
    config_dict = ExperimentConfig(
        world=WorldConfig(),
        neural=NeuralConfig(),
        plasticity=PlasticityConfig(),
        behavior=AgentBehaviorConfig(),
        experiment_name="default_experiment",
        agent_version="unknown",
        world_version="unknown",
        n_episodes=100,
        seed=42,
        device="gpu" if jax.default_backend() != "cpu" else "cpu",
        export_dir="experiments/",
        log_to_console=True,
    ).to_dict()

    # Load and merge config files if provided
    if args.config:
        file_config = load_config(args.config)
        config_dict = merge_configs(config_dict, file_config)

    # Apply command-line overrides
    config_dict["experiment_name"] = args.name
    config_dict["n_episodes"] = args.episodes
    config_dict["seed"] = args.seed
    config_dict["log_to_console"] = not args.quiet
    if args.output_dir:
        config_dict["export_dir"] = str(args.output_dir)
    if args.max_timesteps:
        config_dict["world"]["max_timesteps"] = args.max_timesteps

    # Create the final, type-safe ExperimentConfig object
    return ExperimentConfig(
        world=WorldConfig(**config_dict["world"]),
        neural=NeuralConfig(**config_dict["neural"]),
        plasticity=PlasticityConfig(**config_dict["plasticity"]),
        behavior=AgentBehaviorConfig(**config_dict["behavior"]),
        **{
            k: v
            for k, v in config_dict.items()
            if k not in ["world", "neural", "plasticity", "behavior"]
        },
    )


def main():
    """Main entry point for the unified runner."""
    parser = argparse.ArgumentParser(
        description="Unified runner for SNN experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core arguments
    parser.add_argument("agent", choices=list(AGENT_REGISTRY.keys()), help="Agent to run.")
    parser.add_argument("--name", default="unified_run", help="Experiment name.")
    parser.add_argument("--episodes", type=int, default=100, help="Number of episodes.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    # Configuration
    parser.add_argument("--config", type=Path, help="Path to a base JSON config file.")

    # Overrides
    parser.add_argument("--max-timesteps", type=int, help="Override max timesteps per episode.")

    # Execution control
    parser.add_argument(
        "--no-jit", action="store_true", help="Disable JIT compilation for debugging."
    )
    parser.add_argument(
        "--parallel-episodes",
        type=int,
        default=1,
        help="Number of episodes to run in parallel with vmap (GPU/TPU only).",
    )

    # Output control
    parser.add_argument("--output-dir", type=Path, help="Base directory for experiment data.")
    parser.add_argument("--quiet", action="store_true", help="Suppress console output.")

    args = parser.parse_args()

    # --- Setup ---
    if not args.quiet:
        print("=" * 60)
        print(f"JAX backend: {jax.default_backend()}, Devices: {jax.devices()}")
        print("=" * 60)

    config = create_experiment_config(args)

    world = MinimalGridWorld(config.world)

    experiment_name = f"{args.agent}_{config.experiment_name}_{config.seed}"
    exporter = JaxDataExporter(
        experiment_name=experiment_name, config=config, output_base_dir=config.export_dir
    )

    agent = create_agent(args.agent, config, exporter)

    runner = ProtocolRunner(world, agent, exporter, config)

    if not args.quiet:
        print(f"Running experiment: {experiment_name}")
        print(f"Agent: {args.agent} (v{agent.VERSION}) | World: v{world.VERSION}")
        print(
            f"Total Episodes: {config.n_episodes}, Timesteps/Episode: {config.world.max_timesteps}"
        )
        print(f"Outputting to: {exporter.output_dir}")
        print("-" * 60)

    # --- Execution ---
    start_time = time.perf_counter()
    if args.parallel_episodes > 1:
        if "cpu" in jax.default_backend() and not args.quiet:
            print(
                "Warning: --parallel-episodes > 1 is not effective on CPU. Use a GPU/TPU for performance."
            )
        runner.run_experiment_vmap(num_parallel_episodes=args.parallel_episodes)
    else:
        runner.run_experiment(use_jit=not args.no_jit)

    duration = time.perf_counter() - start_time

    # --- Summary ---
    if not args.quiet:
        print("-" * 60)
        print("Experiment Complete")
        print(f"Total duration: {duration:.2f} seconds")
        print(f"Data saved to: {exporter.output_dir}")
        print("=" * 60)

    exporter.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
