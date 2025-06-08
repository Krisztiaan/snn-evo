# keywords: [random agent, baseline agent, jax random actions]
"""Random agent implementation."""

from typing import Any, Dict

import jax.numpy as jnp
import numpy as np
from jax import random

from export import DataExporter
from world.simple_grid_0001 import SimpleGridWorld

from .config import RandomAgentConfig


class RandomAgent:
    """Agent that selects random actions at each step."""

    def __init__(self, config: RandomAgentConfig):
        self.config = config
        self.world = SimpleGridWorld(config.world_config)

    def run(self) -> Dict[str, Any]:
        """Run all episodes for the experiment."""
        key = random.PRNGKey(self.config.seed)

        with DataExporter(
            experiment_name="random_agent_baseline",
            output_base_dir=self.config.export_dir,
            neural_sampling_rate=1,
            compression="gzip",
            compression_level=1,
        ) as exporter:
            config_dict = {
                "agent_type": "random",
                "world_config": self.config.world_config,
                "n_episodes": self.config.n_episodes,
                "n_steps": self.config.n_steps,
                "seed": self.config.seed,
            }
            exporter.save_config(config_dict)

            metadata = {
                "description": "Random agent baseline in simple grid world",
                "agent_details": "Selects uniformly random actions at each timestep",
            }
            exporter.save_metadata(metadata)

            all_summaries = []
            for episode_num in range(self.config.n_episodes):
                key, episode_key = random.split(key)
                print(f"\nRunning episode {episode_num + 1}/{self.config.n_episodes}")

                summary = self._run_one_episode(exporter, episode_num, episode_key)
                all_summaries.append(summary)

                print(f"Episode {episode_num + 1} summary:")
                print(f"  Total reward: {summary['total_reward']:.1f}")
                print(f"  Rewards collected: {summary['rewards_collected']}")
                print(f"  Coverage: {summary['coverage']:.1%}")

            return {"episodes": all_summaries}

    def _run_one_episode(
        self, exporter: DataExporter, episode_num: int, key: random.PRNGKey
    ) -> Dict[str, Any]:
        """Run a single episode with random actions and log to exporter."""
        reset_key, action_key = random.split(key)
        state, obs = self.world.reset(reset_key)

        exporter.start_episode(episode_num)
        exporter.log_static_episode_data(
            "world_setup", {"reward_positions": np.array(state.reward_positions)}
        )

        # Convert JAX array to a hashable tuple of Python ints
        unique_positions = {tuple(int(p) for p in state.agent_pos)}

        for step in range(self.config.n_steps):
            action_key, subkey = random.split(action_key)
            action = int(random.randint(subkey, (), 0, 4))

            result = self.world.step(state, action)
            state = result.state
            obs = result.observation

            # Convert JAX array to a hashable tuple of Python ints
            unique_positions.add(tuple(int(p) for p in state.agent_pos))

            if self.config.export_full_trajectory:
                exporter.log(
                    timestep=step,
                    behavior={
                        "pos_x": int(state.agent_pos[0]),
                        "pos_y": int(state.agent_pos[1]),
                        "action": action,
                        "gradient": float(obs.gradient),
                    },
                    reward=float(result.reward),
                )

            if result.done:
                break

        rewards_collected = int(jnp.sum(state.reward_collected))
        coverage = len(unique_positions) / (self.config.world_config.grid_size**2)

        episode_summary = {
            "total_reward": float(state.total_reward),
            "rewards_collected": rewards_collected,
            "steps_taken": state.timestep,
            "coverage": coverage,
            "final_position": [int(p) for p in state.agent_pos],
            "all_rewards_collected": bool(jnp.all(state.reward_collected)),
        }

        exporter.end_episode(success=bool(jnp.all(state.reward_collected)), summary=episode_summary)

        return episode_summary
