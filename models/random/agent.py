# keywords: [random agent, baseline agent, jax random actions]
"""Random agent implementation."""

from typing import Dict, Any, Tuple
import jax
import jax.numpy as jnp
from jax import random
import numpy as np

from world.simple_grid_0001 import SimpleGridWorld, WorldState, Observation
from export.exporter import DataExporter
from .config import RandomAgentConfig


class RandomAgent:
    """Agent that selects random actions at each step."""
    
    def __init__(self, config: RandomAgentConfig):
        self.config = config
        self.world = SimpleGridWorld(config.world_config)
        self.exporter = None
        
    def setup_exporter(self, episode: int) -> DataExporter:
        """Setup data exporter for this episode."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_path = f"{self.config.export_dir}/episode_{episode:03d}_{timestamp}"
        
        exporter = DataExporter(export_path)
        
        # Save configuration
        config_dict = {
            "agent_type": "random",
            "world_config": self.config.world_config._asdict(),
            "n_steps": self.config.n_steps,
            "seed": self.config.seed,
            "episode": episode
        }
        exporter.save_config(config_dict)
        
        # Save metadata
        metadata = {
            "description": "Random agent baseline in simple grid world",
            "agent_details": "Selects uniformly random actions at each timestep",
            "world_details": "Simple grid world with gradient observations"
        }
        exporter.save_metadata(metadata)
        
        return exporter
    
    def select_action(self, key: random.PRNGKey, state: WorldState, obs: Observation) -> int:
        """Select a random action."""
        return random.randint(key, (), 0, 4)
    
    def run_episode(self, episode: int, key: random.PRNGKey) -> Dict[str, Any]:
        """Run a single episode with random actions."""
        # Setup exporter
        self.exporter = self.setup_exporter(episode)
        
        # Split keys
        reset_key, action_key = random.split(key)
        
        # Reset world
        state, obs = self.world.reset(reset_key)
        
        # Start episode in exporter
        self.exporter.start_episode(episode)
        
        # Pre-allocate arrays for trajectory data
        positions = np.zeros((self.config.n_steps + 1, 2), dtype=np.int32)
        actions = np.zeros(self.config.n_steps, dtype=np.int32)
        rewards = np.zeros(self.config.n_steps, dtype=np.float32)
        observations = np.zeros(self.config.n_steps + 1, dtype=np.float32)
        
        # Record initial state
        positions[0] = state.agent_pos
        observations[0] = obs.gradient
        
        # Track metrics
        total_reward = 0.0
        rewards_collected = 0
        unique_positions = {tuple(state.agent_pos)}
        
        # Run episode
        for step in range(self.config.n_steps):
            # Select random action
            action_key, subkey = random.split(action_key)
            action = self.select_action(subkey, state, obs)
            
            # Take step
            result = self.world.step(state, action)
            state = result.state
            obs = result.observation
            
            # Record trajectory
            positions[step + 1] = state.agent_pos
            actions[step] = action
            rewards[step] = result.reward
            observations[step + 1] = obs.gradient
            
            # Update metrics
            total_reward += result.reward
            if result.reward >= self.config.world_config.reward_value:
                rewards_collected += 1
            unique_positions.add(tuple(state.agent_pos))
            
            # Log timestep data to exporter
            if self.config.export_full_trajectory:
                timestep_data = {
                    "position": np.array(state.agent_pos),
                    "action": action,
                    "reward": result.reward,
                    "observation": obs.gradient,
                    "total_reward": state.total_reward,
                    "rewards_collected": int(jnp.sum(state.reward_collected))
                }
                self.exporter.log_timestep(step, timestep_data)
            
            if result.done:
                break
        
        # Episode summary
        episode_summary = {
            "total_reward": total_reward,
            "rewards_collected": rewards_collected,
            "steps_taken": step + 1,
            "coverage": len(unique_positions) / (self.config.world_config.grid_size ** 2),
            "final_position": state.agent_pos,
            "all_rewards_collected": bool(jnp.all(state.reward_collected))
        }
        
        # Save full trajectory if requested
        if self.config.export_full_trajectory:
            trajectory_data = {
                "positions": positions[:step + 1],
                "actions": actions[:step],
                "rewards": rewards[:step],
                "observations": observations[:step + 1]
            }
            self.exporter.save_episode_data("trajectory", trajectory_data)
        
        # Save episode summary
        if self.config.export_episode_summary:
            self.exporter.save_episode_data("summary", episode_summary)
        
        # Save final world state
        final_state = {
            "agent_position": np.array(state.agent_pos),
            "reward_positions": np.array(state.reward_positions),
            "reward_collected": np.array(state.reward_collected),
            "total_reward": state.total_reward,
            "timesteps": state.timestep
        }
        self.exporter.save_final_state(final_state)
        
        # End episode
        self.exporter.end_episode()
        
        # Close exporter
        self.exporter.close()
        
        return episode_summary
    
    def run(self) -> Dict[str, Any]:
        """Run all episodes."""
        key = random.PRNGKey(self.config.seed)
        
        all_summaries = []
        
        for episode in range(self.config.n_episodes):
            key, episode_key = random.split(key)
            print(f"\nRunning episode {episode + 1}/{self.config.n_episodes}")
            
            summary = self.run_episode(episode, episode_key)
            all_summaries.append(summary)
            
            print(f"Episode {episode + 1} summary:")
            print(f"  Total reward: {summary['total_reward']:.1f}")
            print(f"  Rewards collected: {summary['rewards_collected']}")
            print(f"  Coverage: {summary['coverage']:.1%}")
        
        # Aggregate statistics
        if self.config.n_episodes > 1:
            aggregate_stats = {
                "mean_reward": np.mean([s["total_reward"] for s in all_summaries]),
                "std_reward": np.std([s["total_reward"] for s in all_summaries]),
                "mean_rewards_collected": np.mean([s["rewards_collected"] for s in all_summaries]),
                "mean_coverage": np.mean([s["coverage"] for s in all_summaries])
            }
            
            print(f"\nAggregate statistics over {self.config.n_episodes} episodes:")
            print(f"  Mean reward: {aggregate_stats['mean_reward']:.1f} ± {aggregate_stats['std_reward']:.1f}")
            print(f"  Mean rewards collected: {aggregate_stats['mean_rewards_collected']:.1f}")
            print(f"  Mean coverage: {aggregate_stats['mean_coverage']:.1%}")
            
            return {"episodes": all_summaries, "aggregate": aggregate_stats}
        else:
            return {"episodes": all_summaries}