# keywords: [random agent run, minimal export, full episode]
"""Run random agent with minimal data export."""

import argparse
from pathlib import Path
import jax
import jax.numpy as jnp
from jax import random
import numpy as np
from datetime import datetime

from world.simple_grid_0001 import SimpleGridWorld, WorldConfig
from models.random.config import RandomAgentConfig
from models.random.minimal_exporter import MinimalExporter


def run_episode(world: SimpleGridWorld, config: RandomAgentConfig, key: random.PRNGKey):
    """Run a single episode and export data."""
    # Setup export directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_dir = f"{config.export_dir}/episode_{timestamp}"
    
    exporter = MinimalExporter(export_dir)
    
    # Save configuration
    config_dict = {
        "agent_type": "random",
        "world_config": config.world_config._asdict(),
        "n_steps": config.n_steps,
        "seed": config.seed
    }
    exporter.save_config(config_dict)
    
    # Save metadata
    metadata = {
        "description": "Random agent baseline in simple grid world",
        "timestamp": timestamp
    }
    exporter.save_metadata(metadata)
    
    # Reset world
    reset_key, action_key = random.split(key)
    state, obs = world.reset(reset_key)
    
    # Pre-allocate arrays
    positions = np.zeros((config.n_steps + 1, 2), dtype=np.int32)
    actions = np.zeros(config.n_steps, dtype=np.int32)
    rewards = np.zeros(config.n_steps, dtype=np.float32)
    observations = np.zeros(config.n_steps + 1, dtype=np.float32)
    
    # Record initial state
    positions[0] = state.agent_pos
    observations[0] = obs.gradient
    
    # Track metrics
    total_reward = 0.0
    rewards_collected = 0
    unique_positions = {(int(state.agent_pos[0]), int(state.agent_pos[1]))}
    
    print(f"Starting at position: {state.agent_pos}")
    print(f"Initial gradient: {obs.gradient:.3f}")
    
    # Run episode
    actual_steps = 0
    for step in range(config.n_steps):
        # Random action
        action_key, subkey = random.split(action_key)
        action = random.randint(subkey, (), 0, 4)
        
        # Take step
        result = world.step(state, action)
        state = result.state
        obs = result.observation
        
        # Record data
        positions[step + 1] = state.agent_pos
        actions[step] = action
        rewards[step] = result.reward
        observations[step + 1] = obs.gradient
        
        # Update metrics
        total_reward += result.reward
        if result.reward >= config.world_config.reward_value:
            rewards_collected += 1
            print(f"Step {step}: Collected reward! Total: {total_reward:.1f}")
        unique_positions.add((int(state.agent_pos[0]), int(state.agent_pos[1])))
        
        actual_steps = step + 1
        
        if result.done:
            print(f"Episode finished at step {step}")
            break
    
    # Save trajectory data
    trajectory = {
        "positions": positions[:actual_steps + 1],
        "actions": actions[:actual_steps],
        "rewards": rewards[:actual_steps],
        "observations": observations[:actual_steps + 1]
    }
    exporter.save_trajectory(trajectory)
    
    # Save summary
    summary = {
        "total_reward": float(total_reward),
        "rewards_collected": int(rewards_collected),
        "steps_taken": actual_steps,
        "coverage": len(unique_positions) / (config.world_config.grid_size ** 2),
        "final_position": [int(state.agent_pos[0]), int(state.agent_pos[1])],
        "all_rewards_collected": bool(jnp.all(state.reward_collected))
    }
    exporter.save_summary(summary)
    
    # Save final state
    final_state = {
        "agent_position": np.array(state.agent_pos),
        "reward_positions": np.array(state.reward_positions),
        "reward_collected": np.array(state.reward_collected),
        "total_reward": float(state.total_reward),
        "timesteps": int(state.timestep)
    }
    exporter.save_final_state(final_state)
    
    # Close exporter
    exporter.close()
    
    # Print summary
    print(f"\nEpisode Summary:")
    print(f"  Total reward: {summary['total_reward']:.1f}")
    print(f"  Rewards collected: {summary['rewards_collected']}/{config.world_config.n_rewards}")
    print(f"  Coverage: {summary['coverage']:.1%}")
    print(f"  Steps taken: {summary['steps_taken']}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run random agent with minimal export")
    parser.add_argument("--steps", type=int, default=1000, help="Steps per episode")
    parser.add_argument("--grid-size", type=int, default=50, help="Grid world size")
    parser.add_argument("--rewards", type=int, default=50, help="Number of rewards")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--export-dir", type=str, default="experiments/random", help="Export directory")
    
    args = parser.parse_args()
    
    # Create configuration
    world_config = WorldConfig(
        grid_size=args.grid_size,
        n_rewards=args.rewards,
        max_timesteps=args.steps,
        seed=args.seed
    )
    
    config = RandomAgentConfig(
        world_config=world_config,
        n_episodes=1,
        n_steps=args.steps,
        seed=args.seed,
        export_dir=args.export_dir
    )
    
    # Create world
    world = SimpleGridWorld(world_config)
    
    print("Random Agent Experiment (Minimal Export)")
    print("=" * 50)
    print(f"Grid size: {args.grid_size}x{args.grid_size}")
    print(f"Rewards: {args.rewards}")
    print(f"Steps: {args.steps}")
    print(f"Export directory: {args.export_dir}")
    print(f"JAX devices: {jax.devices()}")
    print("=" * 50)
    
    # Run episode
    key = random.PRNGKey(args.seed)
    run_episode(world, config, key)
    
    print("\nExperiment complete!")


if __name__ == "__main__":
    main()