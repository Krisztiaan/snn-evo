# keywords: [neo run script, phase 0.14, experiment runner]
"""Run script for Phase 0.14 Neo agent."""

import json
from pathlib import Path
from typing import Dict, Any, Optional

import jax
from jax import random

from export.jax_data_exporter import JaxDataExporter, ExperimentConfig
from interfaces import EpisodeData
from world.simple_grid_0004 import MinimalGridWorld
from .agent import NeoAgent
from .config import NeoConfig


def run_neo_experiment(
    config_dict: Optional[Dict[str, Any]] = None,
    base_dir: str = "./experiments"
) -> Path:
    """Run Neo agent experiment with proper protocol.
    
    Args:
        config_dict: Configuration overrides
        base_dir: Base directory for experiments
        
    Returns:
        Path to experiment directory
    """
    # Create configuration
    if config_dict:
        config = NeoConfig.from_dict(config_dict)
    else:
        config = NeoConfig()
    
    # Create experiment config for exporter
    exp_config = ExperimentConfig(
        world_version="simple_grid_0004",
        agent_version="phase_0_14_neo",
        world_params={
            "grid_size": config.world_config.grid_size,
            "n_rewards": config.world_config.n_rewards,
            "max_timesteps": config.world_config.max_timesteps
        },
        agent_params={
            "num_sensory": config.network_config.num_sensory,
            "num_processing": config.network_config.num_processing,
            "num_readout": config.network_config.num_readout,
        },
        neural_params={
            "excitatory_ratio": config.network_config.excitatory_ratio,
            "tau_v": config.dynamics_config.tau_v,
            "v_threshold": config.dynamics_config.v_threshold,
        },
        learning_params={
            "enabled_rules": config.learning_rules_config.enabled_rules,
            "base_learning_rate": config.learning_rules_config.base_learning_rate,
            "learning_rate_decay": config.learning_rules_config.learning_rate_decay,
        },
        max_timesteps=config.world_config.max_timesteps,
        neural_dim=config.network_config.num_sensory + config.network_config.num_processing + config.network_config.num_readout,
        neural_sampling_rate=100
    )
    
    # Create exporter with timestamp in name
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"neo_{timestamp}"
    
    # Create exporter
    exporter = JaxDataExporter(
        experiment_name=exp_name,
        config=exp_config,
        output_base_dir=base_dir,
        compression="gzip",
        compression_level=4,
        log_to_console=config.exp_config.verbose
    )
    
    # Get experiment directory
    exp_dir = exporter.output_dir
    
    # Save configuration
    config_path = exp_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump({
            "agent": "phase_0_14_neo",
            "world": {
                "grid_size": config.world_config.grid_size,
                "n_rewards": config.world_config.n_rewards,
                "max_timesteps": config.world_config.max_timesteps
            },
            "network": {
                "num_sensory": config.network_config.num_sensory,
                "num_processing": config.network_config.num_processing,
                "num_readout": config.network_config.num_readout,
                "num_input_channels": config.network_config.num_input_channels,
                "excitatory_ratio": config.network_config.excitatory_ratio
            },
            "dynamics": {
                "tau_v": config.dynamics_config.tau_v,
                "v_threshold": config.dynamics_config.v_threshold,
                "initial_temperature": config.dynamics_config.initial_temperature,
                "final_temperature": config.dynamics_config.final_temperature
            },
            "learning_rules": {
                "enabled_rules": config.learning_rules_config.enabled_rules,
                "base_learning_rate": config.learning_rules_config.base_learning_rate,
                "learning_rate_decay": config.learning_rules_config.learning_rate_decay
            },
            "experiment": {
                "n_episodes": config.exp_config.n_episodes,
                "seed": config.exp_config.seed
            }
        }, f, indent=2)
    
    print(f"Starting Neo experiment: {exp_dir}")
    print(f"Configuration saved to: {config_path}")
    
    # Create agent
    print("Creating agent...")
    agent = NeoAgent(exp_config, exporter)
    print("Agent created.")
    
    # Create world
    world = MinimalGridWorld(config.world_config)
    
    # Run episodes
    key = random.PRNGKey(config.exp_config.seed)
    
    try:
        for episode in range(config.exp_config.n_episodes):
            key, episode_key, agent_key = random.split(key, 3)
            
            # Reset world and agent
            world_state, initial_gradient = world.reset(episode_key)
            agent.reset(agent_key)
            
            # Start episode in exporter
            buffer, log_fn = exporter.start_episode()
            
            print(f"\nEpisode {episode + 1}/{config.exp_config.n_episodes}")
            
            # Episode loop
            gradient = initial_gradient
            max_steps = min(config.world_config.max_timesteps, 100)  # Limit for debugging
            print(f"  Running {max_steps} steps...")
            
            for step in range(max_steps):
                key, step_key = random.split(key)
                
                # Agent acts
                print(f"  Step {step}: gradient={float(gradient):.3f}")
                action = agent.act(gradient, step_key)
                print(f"  Step {step}: action={int(action)}")
                
                # World steps
                world_state, gradient = world.step(world_state, int(action))
                
                # Log to buffer
                buffer = log_fn(
                    buffer,
                    step,
                    agent.state.v,  # Neural state
                    float(gradient >= 0.99),  # Reward when gradient is 1.0
                    int(action)
                )
            
            # Get episode data from agent
            episode_data = agent.get_episode_data()
            
            # End episode
            summary = exporter.end_episode(
                buffer,
                success=True,
                reward_history=None
            )
            
            # Print summary
            print(f"  Completed in {summary['timesteps']} steps")
            print(f"  Rewards collected: {summary['rewards_collected']}")
            print(f"  Final learning rate: {float(agent.state.learning_rate):.4f}")
            mean_rate = float(jax.numpy.mean(agent.state.firing_rate))
            print(f"  Mean firing rate: {mean_rate:.1f} Hz")
        
        # Save final results
        print("\nExperiment completed!")
        print(f"Results saved to: {exp_dir}")
        
    finally:
        # Ensure exporter cleanup
        exporter.close()
    
    return exp_dir


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Neo agent experiment")
    parser.add_argument("--episodes", type=int, help="Number of episodes")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--grid-size", type=int, help="Grid size")
    parser.add_argument("--learning-rate", type=float, help="Base learning rate")
    parser.add_argument("--config", type=str, help="Path to JSON config file")
    parser.add_argument("--disable-rule", action="append", help="Disable specific learning rule")
    parser.add_argument("--enable-rule", action="append", help="Enable specific learning rule")
    
    args = parser.parse_args()
    
    # Build config dict
    config_dict = {}
    
    # Load from file if provided
    if args.config:
        with open(args.config) as f:
            config_dict = json.load(f)
    
    # Override with command line args
    if args.episodes:
        config_dict.setdefault("experiment", {})["n_episodes"] = args.episodes
    if args.seed:
        config_dict.setdefault("experiment", {})["seed"] = args.seed
    if args.grid_size:
        config_dict.setdefault("world", {})["grid_size"] = args.grid_size
    if args.learning_rate:
        config_dict.setdefault("learning_rules", {})["base_learning_rate"] = args.learning_rate
    
    # Handle rule modifications
    if args.disable_rule or args.enable_rule:
        # Get default rules
        default_config = NeoConfig()
        enabled_rules = list(default_config.learning_rules_config.enabled_rules)
        
        # Apply modifications
        if args.disable_rule:
            for rule in args.disable_rule:
                if rule in enabled_rules:
                    enabled_rules.remove(rule)
        
        if args.enable_rule:
            for rule in args.enable_rule:
                if rule not in enabled_rules:
                    enabled_rules.append(rule)
        
        config_dict.setdefault("learning_rules", {})["enabled_rules"] = enabled_rules
    
    # Run experiment
    exp_dir = run_neo_experiment(config_dict)
    print(f"\nExperiment directory: {exp_dir}")