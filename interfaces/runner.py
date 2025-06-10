# keywords: [protocol runner, jax episode, high performance, strict types]
"""Protocol-compliant runner for high-performance episode execution."""

from typing import Tuple, Dict, List, Any
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
    NeuralConfig, PlasticityConfig, AgentBehaviorConfig
)
from interfaces.episode_data import StepData
from export.jax_data_exporter import create_episode_buffer


@partial(jax.jit, static_argnames=('world_step_fn', 'agent_select_action_fn', 'log_fn', 'max_steps'))
def run_episode_jax(
    world_step_fn: callable,
    agent_select_action_fn: callable,
    log_fn: callable,
    initial_w_state: WorldState,
    initial_a_state: Any,
    initial_buffer: EpisodeBufferProtocol,
    key: PRNGKey,
    max_steps: int,
) -> Tuple[WorldState, Any, EpisodeBufferProtocol]:
    """JIT-compiled function to run an entire episode."""

    # The complex logic is GONE. It's now beautifully simple.
    def episode_step_fn(carry, step_idx):
        w_state, a_state, buffer, key = carry
        
        # Split key for world and agent
        w_key, a_key, loop_key = split(key, 3)
        last_gradient = w_state.last_gradient

        # 1. AGENT THINKS AND ACTS: Agent is a black box that gives us an action.
        new_a_state, action, neural_data = agent_select_action_fn(
            a_state, last_gradient, a_key
        )

        # 2. UPDATE WORLD: Take the action in the environment.
        new_w_state, _ = world_step_fn(w_state, action)

        # 3. LOG: Record the outcome of this world step.
        reward = jnp.where(last_gradient >= 0.99, 1.0, 0.0)
        step_data = StepData(
            timestep=step_idx,
            gradient=last_gradient,
            action=action,
            reward=reward,
            neural_v=neural_data['v'], # Get data from the agent's return
            neural_data={'spikes': neural_data['spikes']}
        )
        new_buffer = log_fn(buffer, step_data)

        return (new_w_state, new_a_state, new_buffer, loop_key), None

    init_carry = (initial_w_state, initial_a_state, initial_buffer, key)
    (final_w_state, final_a_state, final_buffer, _), _ = lax.scan(
        episode_step_fn, init_carry, jnp.arange(max_steps)
    )

    return final_w_state, final_a_state, final_buffer


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
    
    def run_episode(self, episode_id: int, episode_key: PRNGKey, use_jit: bool = True) -> Dict[str, float]:
        """Runs a single episode, using JIT by default.
        
        Args:
            episode_id: Unique episode identifier
            episode_key: JAX random key for this episode
            use_jit: Whether to use JIT compilation (default: True)
            
        Returns:
            Dictionary of episode statistics
        """
        start_time = time.perf_counter()
        
        # Setup
        world_key, agent_key, run_key = split(episode_key, 3)
        world_state, initial_gradient = self.world.reset(world_key)
        agent_state = self.agent.reset(agent_key)
        buffer, log_fn = self.exporter.start_episode(episode_id)

        if use_jit:
            # High-performance path - notice how clean this is now!
            final_w_state, final_a_state, final_buffer = run_episode_jax(
                self.world.step,
                self.agent.select_action,  # <-- Pass the new method
                log_fn,
                world_state, agent_state, buffer, run_key,
                self.config.world.max_timesteps,
            )
            # The agent's host-side state needs to be updated after the JIT run
            self.agent.state = final_a_state
        else:
            # Debug loop
            gradient = world_state.last_gradient
            agent_state_local = agent_state
            for step in range(self.config.world.max_timesteps):
                agent_key, step_key = split(agent_key)
                
                # Call the agent's select_action method
                agent_state_local, action, neural_data = self.agent.select_action(
                    agent_state_local, gradient, step_key
                )
                
                # Log data
                reward = jnp.where(gradient >= 0.99, 1.0, 0.0)
                step_data = StepData(
                    timestep=step,
                    gradient=gradient,
                    action=action,
                    reward=reward,
                    neural_v=neural_data['v'],
                    neural_data={'spikes': neural_data['spikes']}
                )
                buffer = log_fn(buffer, step_data)
                
                # Update world
                world_state, gradient = self.world.step(world_state, action)
            final_w_state, final_buffer = world_state, buffer
            self.agent.state = agent_state_local

        # Finalization
        duration = time.perf_counter() - start_time
        
        # The agent no longer provides episode_data. The exporter
        # gets everything it needs from the buffer.
        reward_tracking = self.world.get_reward_tracking(final_w_state)
        
        # Calculate success based on the final buffer's reward count
        rewards_collected = jnp.sum(final_buffer.rewards[:final_buffer.current_size])
        success = rewards_collected >= self.config.world.n_rewards
        
        stats = self.exporter.end_episode(
            final_buffer, 
            reward_tracking,
            success=success
        )
        stats["duration_seconds"] = duration
        stats["steps_per_second"] = self.config.world.max_timesteps / duration if duration > 0 else 0
        
        return stats
    
    
    def run_experiment(self, use_jit: bool = True) -> Dict[str, List[Dict[str, float]]]:
        """Run full experiment.
        
        Args:
            use_jit: Whether to use JIT compilation (default: True)
        """
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
            stats = self.run_episode(episode, episode_key, use_jit=use_jit)
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
    
    def run_experiment_vmap(self, num_parallel_episodes: int = 10) -> Dict[str, Any]:
        """Run experiment with batched episodes using vmap for parallelism.
        
        Args:
            num_parallel_episodes: Number of episodes to run in parallel
            
        Returns:
            Experiment statistics including throughput metrics
        """
        print(f"Running {self.config.n_episodes} episodes with batch size {num_parallel_episodes}...")
        
        # Save metadata
        world_config = self.world.get_config()
        self.exporter.save_experiment_metadata(
            agent_version=self.agent.VERSION,
            agent_name=self.agent.MODEL_NAME,
            agent_description=self.agent.DESCRIPTION,
            world_version=world_config["version"]
        )
        
        # Track overall statistics
        all_episode_stats = []
        key = PRNGKey(self.config.seed)
        
        # Process episodes in batches
        num_batches = (self.config.n_episodes + num_parallel_episodes - 1) // num_parallel_episodes
        
        for batch_idx in range(num_batches):
            batch_start_time = time.perf_counter()
            
            # Calculate actual batch size (might be smaller for last batch)
            remaining_episodes = self.config.n_episodes - batch_idx * num_parallel_episodes
            current_batch_size = min(num_parallel_episodes, remaining_episodes)
            
            # 1. Create a batch of random keys
            key, batch_key = split(key)
            world_keys = split(batch_key, current_batch_size * 3)
            world_reset_keys = world_keys[:current_batch_size]
            agent_reset_keys = world_keys[current_batch_size:2*current_batch_size]
            run_keys = world_keys[2*current_batch_size:]
            
            # 2. Create batched initial states
            # vmap over reset functions
            batched_world_reset = jax.vmap(self.world.reset)
            batched_agent_reset = jax.vmap(self.agent.reset)
            
            batched_world_states, batched_gradients = batched_world_reset(world_reset_keys)
            batched_agent_states = batched_agent_reset(agent_reset_keys)
            
            # 3. Create batched initial buffers
            # This requires a vmap-compatible buffer creation
            vmap_create_buffer = jax.vmap(
                lambda eid: create_episode_buffer(
                    self.config.world.max_timesteps,
                    self.config.neural.n_neurons,
                    eid
                ),
                in_axes=0
            )
            episode_ids = jnp.arange(batch_idx * num_parallel_episodes, 
                                   batch_idx * num_parallel_episodes + current_batch_size)
            batched_buffers = vmap_create_buffer(episode_ids)
            
            # 4. Get log function (same for all episodes)
            log_fn = self.exporter.log_step
            
            # 5. vmap the main episode runner
            vmapped_runner = jax.vmap(
                run_episode_jax,
                # Map over the first axis of states, buffers, and keys
                # None means don't map (use same value for all)
                in_axes=(None, None, None, 0, 0, 0, 0, None)
            )
            
            # 6. Run all episodes in parallel
            final_w_states, final_a_states, final_buffers = vmapped_runner(
                self.world.step,
                self.agent.select_action,
                log_fn,
                batched_world_states,
                batched_agent_states,
                batched_buffers,
                run_keys,
                self.config.world.max_timesteps
            )
            
            # 7. Process results (device to host transfer)
            batch_duration = time.perf_counter() - batch_start_time
            
            # Transfer to host
            final_states_host = jax.device_get((final_w_states, final_a_states, final_buffers))
            final_w_states_host, final_a_states_host, final_buffers_host = final_states_host
            
            # Process each episode in the batch
            for i in range(current_batch_size):
                # Extract single episode data using tree_map
                single_buffer = jax.tree_util.tree_map(lambda x: x[i], final_buffers_host)
                single_w_state = jax.tree_util.tree_map(lambda x: x[i], final_w_states_host)
                
                # Get reward tracking from world
                reward_tracking = self.world.get_reward_tracking(single_w_state)
                
                # Calculate success based on the final buffer's reward count
                rewards_collected = jnp.sum(single_buffer.rewards[:single_buffer.current_size])
                success = rewards_collected >= self.config.world.n_rewards
                
                # End episode in exporter
                episode_id = batch_idx * num_parallel_episodes + i
                stats = self.exporter.end_episode(
                    single_buffer,
                    reward_tracking,
                    success=success
                )
                
                # Add batch timing info
                stats["batch_idx"] = batch_idx
                stats["batch_size"] = current_batch_size
                stats["duration_seconds"] = batch_duration / current_batch_size  # Per-episode time
                stats["steps_per_second"] = self.config.world.max_timesteps / stats["duration_seconds"]
                
                all_episode_stats.append(stats)
            
            # Log batch progress
            if self.config.log_to_console:
                batch_steps_per_sec = (current_batch_size * self.config.world.max_timesteps) / batch_duration
                print(f"Batch {batch_idx + 1}/{num_batches}: {batch_steps_per_sec:.0f} steps/s total, "
                      f"{current_batch_size} episodes in {batch_duration:.2f}s")
        
        # Finalize experiment
        self.exporter.finalize()
        
        # Calculate summary statistics
        speeds = [s["steps_per_second"] for s in all_episode_stats]
        rewards = [s["total_rewards"] for s in all_episode_stats]
        avg_speed = sum(speeds) / len(speeds)
        avg_rewards = sum(rewards) / len(rewards)
        
        if self.config.log_to_console:
            print(f"\nVMAP Experiment Summary:")
            print(f"  Total episodes: {self.config.n_episodes}")
            print(f"  Batch size: {num_parallel_episodes}")
            print(f"  Average speed: {avg_speed:.0f} steps/s per episode")
            print(f"  Average rewards: {avg_rewards:.1f}")
        
        return {
            "episode_stats": all_episode_stats,
            "average_steps_per_second": avg_speed,
            "average_rewards": avg_rewards,
            "batch_size": num_parallel_episodes,
            "num_batches": num_batches
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
            excitatory_ratio=0.8,  # Use the ratio
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