# keywords: [protocol runner, jax episode, high performance, strict types]
"""Protocol-compliant runner for high-performance episode execution."""

from typing import Tuple, Dict, List
import time
import jax
import jax.numpy as jnp
from jax import Array, lax
from jax.random import PRNGKey, split
from functools import partial
from pathlib import Path

from interfaces import (
    WorldProtocol, WorldState, WorldConfig,
    AgentProtocol, ExperimentConfig, ExporterProtocol,
    EpisodeBufferProtocol, LogTimestepFunction,
    NeuralConfig, PlasticityConfig, AgentBehaviorConfig,
    EpisodeData
)


@partial(jax.jit, static_argnames=('max_steps',))
def run_episode_jax(
    world_step_fn: callable,
    agent_act_fn: callable,
    log_timestep_fn: LogTimestepFunction,
    initial_state: WorldState,
    initial_buffer: EpisodeBufferProtocol,
    initial_key: PRNGKey,
    max_steps: int
) -> Tuple[WorldState, EpisodeBufferProtocol, Array, Array]:
    """Run entire episode in JAX with proper logging.
    
    Args:
        world_step_fn: JIT-compiled world.step function
        agent_act_fn: JIT-compiled agent.act function
        log_timestep_fn: JIT-compiled logging function from exporter
        initial_state: Initial world state
        initial_buffer: Initial episode buffer from exporter
        initial_key: Initial random key
        max_steps: Maximum episode steps
        
    Returns:
        final_state: Final world state
        final_buffer: Final episode buffer with all logged data
        gradients: Array of gradients for each step
        actions: Array of actions taken
    """
    def step_fn(carry: Tuple, step: int) -> Tuple[Tuple, Tuple[Array, Array]]:
        state, buffer, key = carry
        key, agent_key = split(key)
        
        # Get current gradient from state
        # Note: In first step, we should have initial gradient
        gradient = state.reward_positions  # This is wrong - need gradient calculation
        
        # Agent selects action
        action = agent_act_fn(gradient, agent_key)
        
        # World steps
        new_state, new_gradient = world_step_fn(state, action)
        
        # Log timestep
        new_buffer = log_timestep_fn(buffer, step, new_gradient, action, None)
        
        return (new_state, new_buffer, key), (new_gradient, action)
    
    # Run all steps
    (final_state, final_buffer, _), (gradients, actions) = lax.scan(
        step_fn, (initial_state, initial_buffer, initial_key), jnp.arange(max_steps)
    )
    
    return final_state, final_buffer, gradients, actions


class ProtocolRunner:
    """Runner that uses protocol interfaces for maximum performance."""
    
    def __init__(
        self,
        world: WorldProtocol,
        agent: AgentProtocol,
        exporter: ExporterProtocol,
        config: ExperimentConfig
    ):
        self.world = world
        self.agent = agent
        self.exporter = exporter
        self.config = config
    
    def run_episode(self, episode_id: int, episode_key: PRNGKey) -> Dict[str, float]:
        """Run single episode with protocol-compliant components."""
        # Start episode in exporter
        buffer, log_fn = self.exporter.start_episode(episode_id)
        
        # Reset world and agent
        world_key, agent_key = split(episode_key)
        world_state, initial_gradient = self.world.reset(world_key)
        self.agent.reset(agent_key)
        
        start_time = time.perf_counter()
        
        # Episode execution
        gradients_list: List[float] = []
        actions_list: List[int] = []
        
        gradient = initial_gradient
        for step in range(self.config.world.max_timesteps):
            # Agent acts
            agent_key, step_key = split(agent_key)
            action = self.agent.act(gradient, step_key)
            
            # Convert action to int for logging (JAX returns Array)
            action_int = int(action)
            
            # Log timestep
            buffer = log_fn(buffer, step, gradient, action_int, None)
            
            # Store for tracking
            gradients_list.append(float(gradient))
            actions_list.append(action_int)
            
            # World steps
            world_state, gradient = self.world.step(world_state, action)
        
        # Get episode data from agent
        episode_data = self.agent.get_episode_data()
        
        # Get reward tracking from world
        reward_tracking = self.world.get_reward_tracking(world_state)
        
        # End episode
        stats = self.exporter.end_episode(
            buffer,
            episode_data,
            reward_tracking,
            success=episode_data.total_reward_events == self.config.world.n_rewards
        )
        
        duration = time.perf_counter() - start_time
        stats["duration_seconds"] = duration
        stats["steps_per_second"] = self.config.world.max_timesteps / duration
        
        return stats
    
    def run_experiment(self) -> Dict[str, List[Dict[str, float]]]:
        """Run full experiment."""
        # Save metadata
        world_config = self.world.get_config()
        self.exporter.save_experiment_metadata(
            agent_version=self.agent.VERSION,
            agent_name=self.agent.MODEL_NAME,
            agent_description=self.agent.DESCRIPTION,
            world_version=world_config["version"]
        )
        
        # Run episodes
        episode_stats: List[Dict[str, float]] = []
        key = PRNGKey(self.config.seed)
        
        for episode in range(self.config.n_episodes):
            key, episode_key = split(key)
            stats = self.run_episode(episode, episode_key)
            episode_stats.append(stats)
            
            if self.config.log_to_console and episode % 10 == 0:
                print(f"Episode {episode}: {stats['steps_per_second']:.0f} steps/s, "
                      f"rewards: {stats['total_rewards']}")
        
        # Finalize
        self.exporter.finalize()
        
        # Summary
        speeds = [s["steps_per_second"] for s in episode_stats]
        rewards = [s["total_rewards"] for s in episode_stats]
        avg_speed = sum(speeds) / len(speeds)
        avg_rewards = sum(rewards) / len(rewards)
        
        if self.config.log_to_console:
            print(f"\nExperiment Summary:")
            print(f"  Average speed: {avg_speed:.0f} steps/s")
            print(f"  Average rewards: {avg_rewards:.1f}")
        
        return {
            "episode_stats": episode_stats,
            "average_steps_per_second": avg_speed,
            "average_rewards": avg_rewards
        }


def create_experiment_config(
    world_size: int = 100,
    n_rewards: int = 300,
    n_episodes: int = 100,
    max_timesteps: int = 50000,
    experiment_name: str = "protocol_test",
    export_dir: str = "experiments/protocol"
) -> ExperimentConfig:
    """Create a complete experiment configuration."""
    return ExperimentConfig(
        # World config
        world=WorldConfig(
            grid_size=world_size,
            n_rewards=n_rewards,
            max_timesteps=max_timesteps
        ),
        # Neural config (even for random agent, needed by protocol)
        neural=NeuralConfig(
            n_neurons=100,
            n_excitatory=80,
            n_inhibitory=20,
            n_sensory=10,
            n_motor=9
        ),
        # Plasticity config
        plasticity=PlasticityConfig(
            enable_stdp=False,
            enable_homeostasis=False,
            enable_reward_modulation=False
        ),
        # Behavior config
        behavior=AgentBehaviorConfig(
            action_noise=0.0,
            temperature=1.0
        ),
        # Experiment metadata
        experiment_name=experiment_name,
        agent_version="2.0.0",
        world_version="0.0.4",
        # Runtime
        n_episodes=n_episodes,
        seed=42,
        device="cpu",
        # Export
        export_dir=export_dir,
        log_every_n_steps=10000,
        neural_sampling_rate=100,
        save_checkpoints=True,
        checkpoint_every_n_episodes=10,
        flush_at_episode_end=True,
        log_to_console=True
    )