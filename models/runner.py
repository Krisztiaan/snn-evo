#!/usr/bin/env python3
# keywords: [model runner, unified interface, agent selection, config management, experiment tracking]
"""
Unified Model Runner for MetaLearning Agents

This module provides a common interface for running different agent implementations
with configurable parameters. It supports:
- Dynamic agent selection
- JSON-based configuration for neural/learning parameters
- Command-line arguments for runtime parameters
- Backwards-compatible configuration handling
- Sequential experiment tracking with dedicated output directories
"""

import argparse
import json
import sys
import time
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Tuple
import importlib
from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np


class AgentRegistry:
    """Registry for available agent implementations."""
    
    _agents: Dict[str, Tuple[str, str]] = {
        # name -> (module_path, class_name)
        "random": ("models.random.agent", "RandomAgent"),
        "phase_0_13": ("models.phase_0_13.agent", "SnnAgent"),
        "phase_0_14_neo": ("models.phase_0_14_neo.agent", "NeoAgent"),
        # Add more agents here as they're developed
    }
    
    @classmethod
    def get_agent_class(cls, name: str) -> Type:
        """Get agent class by name."""
        if name not in cls._agents:
            available = ", ".join(cls._agents.keys())
            raise ValueError(f"Unknown agent: {name}. Available: {available}")
        
        module_path, class_name = cls._agents[name]
        try:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Failed to load agent {name}: {e}")
    
    @classmethod
    def list_agents(cls) -> List[str]:
        """List available agent names."""
        return list(cls._agents.keys())


class ExperimentTracker:
    """Tracks and manages experiment output directories."""
    
    @staticmethod
    def get_next_experiment_dir(base_dir: str, agent_name: str) -> Path:
        """Get the next sequential experiment directory."""
        experiments_dir = Path(base_dir)
        experiments_dir.mkdir(parents=True, exist_ok=True)
        
        # Find existing experiment numbers
        existing = list(experiments_dir.glob(f"{agent_name}_*"))
        if not existing:
            next_num = 1
        else:
            # Extract numbers from directory names
            numbers = []
            for path in existing:
                try:
                    num = int(path.name.split('_')[-1])
                    numbers.append(num)
                except ValueError:
                    continue
            next_num = max(numbers) + 1 if numbers else 1
        
        # Create new directory
        exp_dir = experiments_dir / f"{agent_name}_{next_num:04d}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        return exp_dir


class ConfigBuilder:
    """Builds agent configuration from JSON and runtime parameters."""
    
    @staticmethod
    def build_config(
        agent_name: str,
        neural_config: Optional[Dict[str, Any]] = None,
        learning_config: Optional[Dict[str, Any]] = None,
        learning_rules_config: Optional[Dict[str, Any]] = None,
        world_params: Optional[Dict[str, Any]] = None,
        exp_params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Build agent-specific configuration object."""
        
        # Agent-specific configuration builders
        if agent_name == "random":
            # Random agent uses simple config
            from models.random.config import RandomAgentConfig, WorldConfig
            world_config = WorldConfig(**world_params) if world_params else WorldConfig()
            return RandomAgentConfig(world_config=world_config)
        
        elif agent_name == "phase_0_13":
            # Phase 0.13 uses nested configuration
            from models.phase_0_13.config import SnnAgentConfig, NetworkParams, WorldConfig, ExperimentConfig
            
            # Build NetworkParams from neural + learning + learning_rules configs
            network_params_dict = {}
            if neural_config:
                network_params_dict.update(neural_config)
            if learning_config:
                network_params_dict.update(learning_config)
            if learning_rules_config:
                # Apply learning rules overrides
                network_params_dict.update(learning_rules_config)
            
            # Create config objects
            network_params = NetworkParams(**network_params_dict) if network_params_dict else NetworkParams()
            world_config = WorldConfig(**world_params) if world_params else WorldConfig()
            exp_config = ExperimentConfig(**exp_params) if exp_params else ExperimentConfig()
            
            return SnnAgentConfig(
                world_config=world_config,
                network_params=network_params,
                exp_config=exp_config
            )
        
        elif agent_name == "phase_0_14_neo":
            # Neo agent uses modular configuration
            from models.phase_0_14_neo.config import NeoConfig
            
            # Build combined config dict
            config_dict = {}
            if world_params:
                config_dict["world"] = world_params
            if neural_config:
                config_dict["network"] = neural_config
            if learning_config:
                config_dict["dynamics"] = learning_config
            if learning_rules_config:
                config_dict["learning_rules"] = learning_rules_config
            if exp_params:
                # Filter out fields not supported by Neo's ExperimentConfig
                neo_exp_params = {
                    k: v for k, v in exp_params.items()
                    if k in ["n_episodes", "seed", "export_dir", "log_weight_changes", "verbose"]
                }
                config_dict["experiment"] = neo_exp_params
            
            return NeoConfig.from_dict(config_dict)
        
        else:
            raise ValueError(f"No config builder for agent: {agent_name}")


class ModelRunner:
    """Unified runner for MetaLearning agents."""
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.agent_name = args.agent
        
        # Parse JSON configs if provided
        self.neural_config = self._load_json_config(args.neural_config)
        self.learning_config = self._load_json_config(args.learning_config)
        self.learning_rules_config = self._load_json_config(args.learning_rules_config)
        
        # Get experiment directory
        self.experiment_dir = ExperimentTracker.get_next_experiment_dir(
            args.base_dir, 
            self.agent_name
        )
        
        # Build world parameters from args
        self.world_params = {
            "grid_size": args.grid_size,
            "n_rewards": args.n_rewards or int(args.grid_size * args.grid_size * 0.03),
            "max_timesteps": args.max_timesteps,
        }
        
        # Build experiment parameters (use experiment dir)
        self.exp_params = {
            "n_episodes": args.n_episodes,
            "seed": args.seed,
            "export_dir": str(self.experiment_dir),
            "enable_export": not args.no_export,
        }
        
        # Set JAX platform
        if args.device:
            jax.config.update('jax_platform_name', args.device)
    
    def _load_json_config(self, path: Optional[str]) -> Optional[Dict[str, Any]]:
        """Load JSON configuration file."""
        if not path:
            return None
        
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(config_path) as f:
            return json.load(f)
    
    def run(self) -> Dict[str, Any]:
        """Run the agent with configured parameters."""
        print(f"Running {self.agent_name} agent")
        print(f"Experiment directory: {self.experiment_dir}")
        print("=" * 60)
        
        # Save all configurations to experiment directory
        self._save_experiment_metadata()
        
        # Get agent class
        agent_class = AgentRegistry.get_agent_class(self.agent_name)
        
        # Build configuration
        config = ConfigBuilder.build_config(
            self.agent_name,
            self.neural_config,
            self.learning_config,
            self.learning_rules_config,
            self.world_params,
            self.exp_params,
        )
        
        # Print configuration summary
        print(f"Grid: {self.world_params['grid_size']}x{self.world_params['grid_size']}")
        print(f"Rewards: {self.world_params['n_rewards']}")
        print(f"Max steps: {self.world_params['max_timesteps']}")
        print(f"Episodes: {self.exp_params['n_episodes']}")
        print(f"Export: {'Enabled' if self.exp_params['enable_export'] else 'Disabled'}")
        if self.learning_rules_config:
            print(f"Learning rules: Custom configuration applied")
        print()
        
        # Create and run agent
        start_time = time.time()
        
        # Always create exporter for all agents
        from export import DataExporter, ExperimentConfig
        
        # Get neural dimension based on agent type
        if self.agent_name == "random":
            neural_dim = 0  # Random agent has no neural state
        elif self.agent_name == "phase_0_13":
            # Phase 0.13 default network size
            neural_dim = 800
            if self.neural_config:
                neural_dim = self.neural_config.get("n_neurons", 800)
        elif self.agent_name == "phase_0_14_neo":
            # Neo agent network size
            neural_dim = 200  # Default
            if self.neural_config:
                neural_dim = (
                    self.neural_config.get("num_sensory", 80) +
                    self.neural_config.get("num_processing", 96) +
                    self.neural_config.get("num_readout", 24)
                )
        else:
            neural_dim = 100  # Default fallback
        
        # Create experiment config
        exp_config = ExperimentConfig(
            world_version=self.agent_name,
            agent_version="1.0",
            world_params=self.world_params,
            agent_params=self.neural_config or {},
            neural_params=self.neural_config or {},
            learning_params=self.learning_config or {},
            max_timesteps=self.world_params['max_timesteps'],
            neural_dim=neural_dim,
            neural_sampling_rate=100
        )
        
        # Create exporter with new interface
        exporter = DataExporter(
            experiment_name=f"{self.agent_name}_experiment",
            config=exp_config,
            output_base_dir=str(self.experiment_dir.parent),
            compression="gzip" if self.exp_params['enable_export'] else None,
            compression_level=1,
            log_to_console=not self.args.quiet,
        )
        
        # Create agent with exporter
        with exporter:
            agent = agent_class(config, exporter)
            
            # Run experiment using base class method
            summaries = agent.run_experiment()
        
        total_time = time.time() - start_time
        
        # Export results
        results = self._compile_results(summaries, total_time)
        
        # Always save results to experiment directory
        self._export_results(results)
        
        if not self.args.quiet:
            self._print_summary(results)
        
        # Print experiment directory path
        print(f"\nExperiment output: {self.experiment_dir}")
        
        return results
    
    def _compile_results(self, summaries: List[Dict[str, Any]], total_time: float) -> Dict[str, Any]:
        """Compile experiment results."""
        rewards = [s.get('rewards_collected', s.get('total_reward', 0)) for s in summaries]
        
        results = {
            "agent": self.agent_name,
            "experiment_dir": str(self.experiment_dir),
            "timestamp": datetime.now().isoformat(),
            "config": {
                "world": self.world_params,
                "experiment": self.exp_params,
                "neural": self.neural_config,
                "learning": self.learning_config,
                "learning_rules": self.learning_rules_config,
            },
            "performance": {
                "total_time": total_time,
                "episodes_per_hour": len(summaries) / (total_time / 3600),
                "avg_episode_time": total_time / len(summaries) if summaries else 0,
            },
            "rewards": {
                "mean": float(np.mean(rewards)),
                "std": float(np.std(rewards)),
                "min": float(np.min(rewards)),
                "max": float(np.max(rewards)),
                "median": float(np.median(rewards)),
                "total": float(np.sum(rewards)),
            },
            "summaries": summaries,
        }
        
        # Add learning progress if enough episodes
        if len(rewards) >= 10:
            first_10 = np.mean(rewards[:10])
            last_10 = np.mean(rewards[-10:])
            results["learning_progress"] = {
                "first_10_avg": float(first_10),
                "last_10_avg": float(last_10),
                "improvement": float(last_10 - first_10),
                "improvement_pct": float((last_10 - first_10) / first_10 * 100) if first_10 > 0 else 0,
            }
        
        return results
    
    def _save_experiment_metadata(self) -> None:
        """Save experiment metadata and configurations."""
        # Save command line arguments
        with open(self.experiment_dir / "command_args.json", 'w') as f:
            json.dump(vars(self.args), f, indent=2)
        
        # Save individual config files
        if self.neural_config:
            with open(self.experiment_dir / "neural_config.json", 'w') as f:
                json.dump(self.neural_config, f, indent=2)
        
        if self.learning_config:
            with open(self.experiment_dir / "learning_config.json", 'w') as f:
                json.dump(self.learning_config, f, indent=2)
        
        if self.learning_rules_config:
            with open(self.experiment_dir / "learning_rules_config.json", 'w') as f:
                json.dump(self.learning_rules_config, f, indent=2)
        
        # Save metadata
        metadata = {
            "agent": self.agent_name,
            "timestamp": datetime.now().isoformat(),
            "world_params": self.world_params,
            "experiment_params": self.exp_params,
        }
        with open(self.experiment_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _export_results(self, results: Dict[str, Any]) -> None:
        """Export results to experiment directory."""
        output_path = self.experiment_dir / "results.json"
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Also save a summary file
        summary = {
            "agent": results["agent"],
            "timestamp": results["timestamp"],
            "performance": results["performance"],
            "rewards": results["rewards"],
        }
        if "learning_progress" in results:
            summary["learning_progress"] = results["learning_progress"]
        
        with open(self.experiment_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
    
    def _print_summary(self, results: Dict[str, Any]) -> None:
        """Print results summary."""
        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)
        
        perf = results["performance"]
        print(f"Total time: {perf['total_time']:.1f}s")
        print(f"Episodes/hour: {perf['episodes_per_hour']:.0f}")
        
        rewards = results["rewards"]
        print(f"\nReward Statistics:")
        print(f"  Mean: {rewards['mean']:.1f} ± {rewards['std']:.1f}")
        print(f"  Range: [{rewards['min']:.0f}, {rewards['max']:.0f}]")
        print(f"  Median: {rewards['median']:.1f}")
        
        if "learning_progress" in results:
            prog = results["learning_progress"]
            print(f"\nLearning Progress:")
            print(f"  First 10: {prog['first_10_avg']:.1f}")
            print(f"  Last 10: {prog['last_10_avg']:.1f}")
            print(f"  Improvement: {prog['improvement']:+.1f} ({prog['improvement_pct']:+.1f}%)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Unified Model Runner for MetaLearning Agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run random agent with default settings
  python -m models.runner random
  
  # Run phase_0_13 with custom parameters
  python -m models.runner phase_0_13 --grid-size 50 --n-episodes 100
  
  # Run with custom neural config
  python -m models.runner phase_0_13 --neural-config configs/neural_large.json
  
  # Run with learning rules modifications
  python -m models.runner phase_0_13 --learning-rules-config configs/learning_rules/no_dale.json
  
  # Run quietly with no export
  python -m models.runner phase_0_13 --quiet --no-export
        """
    )
    
    # Agent selection
    parser.add_argument(
        "agent",
        choices=AgentRegistry.list_agents(),
        help="Agent to run"
    )
    
    # Configuration files
    parser.add_argument(
        "--neural-config",
        help="JSON file with neural network configuration"
    )
    parser.add_argument(
        "--learning-config", 
        help="JSON file with learning parameters"
    )
    parser.add_argument(
        "--learning-rules-config",
        help="JSON file with learning rules toggles (homeostasis, Dale's principle, etc.)"
    )
    
    # World parameters
    parser.add_argument(
        "--grid-size",
        type=int,
        default=20,
        help="Grid world size (default: 20)"
    )
    parser.add_argument(
        "--n-rewards",
        type=int,
        help="Number of rewards (default: 3%% of grid)"
    )
    parser.add_argument(
        "--max-timesteps",
        type=int,
        default=1000,
        help="Maximum timesteps per episode (default: 1000)"
    )
    
    # Experiment parameters
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=50,
        help="Number of episodes to run (default: 50)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--base-dir",
        default="experiments",
        help="Base directory for experiments (default: experiments)"
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Disable data export for maximum performance"
    )
    
    # Runtime options
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu", "tpu"],
        help="JAX device to use"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress output except errors"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Run the model
    runner = ModelRunner(args)
    runner.run()


if __name__ == "__main__":
    main()