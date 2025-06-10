# keywords: [jax data exporter, pure jit, minimal io, device arrays]
"""Pure JAX/JIT data exporter with minimal I/O operations."""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Callable

import h5py
import numpy as np
import jax
import jax.numpy as jnp
from jax import Array

from interfaces import ExperimentConfig
from interfaces.episode_data import StepData

from . import __version__ as EXPORTER_VERSION
from .schema import SCHEMA_VERSION
from .utils import NumpyEncoder




class EpisodeBuffer(NamedTuple):
    """Immutable episode buffer for JAX."""
    timesteps: Array
    neural_states: Array
    rewards: Array
    actions: Array
    current_size: int
    max_size: int
    episode_id: int


def create_episode_buffer(max_timesteps: int, neural_dim: int, episode_id: int) -> EpisodeBuffer:
    """Create a new episode buffer with pre-allocated arrays."""
    return EpisodeBuffer(
        timesteps=jnp.zeros(max_timesteps, dtype=jnp.int32),
        neural_states=jnp.zeros((max_timesteps, neural_dim), dtype=jnp.float32),
        rewards=jnp.zeros(max_timesteps, dtype=jnp.float32),
        actions=jnp.zeros(max_timesteps, dtype=jnp.int32),
        current_size=0,
        max_size=max_timesteps,
        episode_id=episode_id
    )


@jax.jit
def log_step(
    buffer: EpisodeBuffer,
    step_data: StepData
) -> EpisodeBuffer:
    """Add a timestep's data to the buffer (pure function)."""
    idx = buffer.current_size
    return buffer._replace(
        timesteps=buffer.timesteps.at[idx].set(step_data.timestep),
        neural_states=buffer.neural_states.at[idx].set(step_data.neural_v),
        rewards=buffer.rewards.at[idx].set(step_data.reward),
        actions=buffer.actions.at[idx].set(step_data.action),
        current_size=buffer.current_size + 1
    )


def sample_neural_states(neural_states: Array, sampling_rate: int) -> Array:
    """Sample neural states with aggregation."""
    n_timesteps = neural_states.shape[0]
    n_samples = n_timesteps // sampling_rate
    
    if n_samples == 0:
        return neural_states[:0]  # Empty array with correct shape
    
    # Use scan for efficient windowed aggregation
    def scan_fn(carry, idx):
        start = idx * sampling_rate
        end = jnp.minimum(start + sampling_rate, n_timesteps)
        window = jax.lax.dynamic_slice(
            neural_states,
            (start, 0),
            (sampling_rate, neural_states.shape[1])
        )
        return carry, jnp.mean(window, axis=0)
    
    _, sampled = jax.lax.scan(scan_fn, None, jnp.arange(n_samples))
    return sampled


def compute_episode_statistics(buffer: EpisodeBuffer) -> Dict[str, Array]:
    """Compute episode statistics from buffer."""
    size = buffer.current_size
    
    # Use dynamic_slice for JAX-compatible indexing
    rewards = jax.lax.dynamic_slice(buffer.rewards, (0,), (size,))
    neural = jax.lax.dynamic_slice(buffer.neural_states, (0, 0), (size, buffer.neural_states.shape[1]))
    actions = jax.lax.dynamic_slice(buffer.actions, (0,), (size,))
    
    # Compute action distribution for entropy
    action_counts = jnp.zeros(4).at[actions].add(1)
    action_probs = action_counts / jnp.sum(action_counts)
    safe_probs = jnp.where(action_probs > 0, action_probs, 1e-10)
    action_entropy = -jnp.sum(safe_probs * jnp.log(safe_probs))
    
    return {
        "total_reward": jnp.sum(rewards),
        "mean_neural_activity": jnp.mean(neural),
        "max_neural_activity": jnp.max(neural),
        "action_entropy": action_entropy,
        "episode_length": size,
        "rewards_collected": jnp.sum(rewards > 0)
    }


class JaxDataExporter:
    """Pure JAX/JIT data exporter with I/O only at episode boundaries."""
    
    def __init__(
        self,
        experiment_name: str,
        config: ExperimentConfig,
        output_base_dir: str = "experiments",
        compression: Optional[str] = "gzip",
        compression_level: int = 4,
        log_to_console: bool = True
    ):
        self.experiment_name = experiment_name
        self.config = config
        self.compression = compression
        self.compression_level = compression_level
        self.log_to_console = log_to_console
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(output_base_dir) / f"{experiment_name}_{self.timestamp}"
        
        # State
        self.episode_count = 0
        self.current_buffer: Optional[EpisodeBuffer] = None
        self.h5_file: Optional[h5py.File] = None
        self.episode_summaries: List[Dict[str, Any]] = []
        self.experiment_start_time = time.time()
        
        # Setup files
        self._setup_files()
        
    def _setup_files(self) -> None:
        """Setup output files (I/O operation)."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create HDF5 file
        h5_path = self.output_dir / "experiment_data.h5"
        self.h5_file = h5py.File(h5_path, "w")
        
        # Save metadata
        self.h5_file.attrs["experiment_name"] = self.experiment_name
        self.h5_file.attrs["timestamp"] = self.timestamp
        self.h5_file.attrs["schema_version"] = SCHEMA_VERSION
        self.h5_file.attrs["exporter_version"] = EXPORTER_VERSION
        self.h5_file.attrs["neural_dim"] = self.config.neural.n_neurons
        self.h5_file.attrs["max_timesteps"] = self.config.world.max_timesteps
        self.h5_file.attrs["neural_sampling_rate"] = self.config.neural_sampling_rate
        
        # Create groups
        self.episodes_group = self.h5_file.create_group("episodes")
        
        # Save experiment config
        config_dict = {
            "experiment_name": self.config.experiment_name,
            "agent_version": self.config.agent_version,
            "world_version": self.config.world_version,
            "world": self.config.world._asdict(),
            "neural": self.config.neural._asdict(),
            "plasticity": self.config.plasticity._asdict(),
            "behavior": self.config.behavior._asdict(),
            "n_episodes": self.config.n_episodes,
            "seed": self.config.seed,
            "device": self.config.device
        }
        with open(self.output_dir / "experiment_config.json", "w") as f:
            json.dump(config_dict, f, indent=2, cls=NumpyEncoder)
            
        if self.log_to_console:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Started experiment: {self.experiment_name}")
    
    def start_episode(self, episode_id: int) -> Tuple[EpisodeBuffer, Callable]:
        """Start a new episode, returns buffer and log function."""
        self.episode_count += 1
        
        # Create buffer on device
        self.current_buffer = create_episode_buffer(
            self.config.world.max_timesteps,
            self.config.neural.n_neurons,
            episode_id
        )
        
        self.episode_start_time = time.time()
        
        # Return buffer and the JIT-compiled log_step function
        return self.current_buffer, log_step
    
    def end_episode(
        self,
        buffer: EpisodeBuffer,
        world_reward_tracking: Dict[str, Array],
        success: bool = False
    ) -> Dict[str, float]:
        """End episode and persist data (I/O operation)."""
        # --- OPTIMIZATION: Compute stats on device BEFORE transfer ---
        stats = compute_episode_statistics(buffer)
        
        # Transfer only the small stats dict and the buffer from device to host
        host_buffer, host_stats = jax.device_get((buffer, stats))
        # --- END OPTIMIZATION ---
        
        # Extract actual data size
        size = host_buffer.current_size
        
        # Sample neural states if needed
        if self.config.neural_sampling_rate > 1:
            # Use dynamic_slice for extracting relevant data
            neural_slice = jax.lax.dynamic_slice(
                buffer.neural_states, 
                (0, 0), 
                (size, buffer.neural_states.shape[1])
            )
            sampled_neural = sample_neural_states(
                neural_slice,
                self.config.neural_sampling_rate
            )
            host_neural = jax.device_get(sampled_neural)
        else:
            host_neural = host_buffer.neural_states[:size]
        
        # Persist to HDF5
        episode_group = self.episodes_group.create_group(f"episode_{host_buffer.episode_id:04d}")
        
        # Save timesteps
        episode_group.create_dataset(
            "timesteps",
            data=host_buffer.timesteps[:size],
            compression=self.compression,
            compression_opts=self.compression_level
        )
        
        # Save neural states
        episode_group.create_dataset(
            "neural_states",
            data=host_neural,
            compression=self.compression,
            compression_opts=self.compression_level
        )
        
        # Save rewards
        episode_group.create_dataset(
            "rewards",
            data=host_buffer.rewards[:size],
            compression=self.compression,
            compression_opts=self.compression_level
        )
        
        # Save actions
        episode_group.create_dataset(
            "actions",
            data=host_buffer.actions[:size],
            compression=self.compression,
            compression_opts=self.compression_level
        )
        
        # Save metadata
        episode_duration = time.time() - self.episode_start_time
        episode_group.attrs["success"] = success
        episode_group.attrs["duration_seconds"] = episode_duration
        episode_group.attrs["timesteps"] = size
        episode_group.attrs["total_reward"] = float(host_stats["total_reward"])
        episode_group.attrs["rewards_collected"] = int(host_stats["rewards_collected"])
        
        # Build summary - ensure all values are float for protocol compliance
        summary = {
            "episode_id": float(host_buffer.episode_id),
            "success": float(success),
            "timesteps": float(size),
            "duration_seconds": float(episode_duration),
            "steps_per_second": float(size / episode_duration if episode_duration > 0 else 0),
            **{k: float(v) if hasattr(v, 'item') else float(v) for k, v in host_stats.items()}
        }
        
        # Add world reward tracking info if provided
        if world_reward_tracking:
            summary["total_rewards"] = float(host_stats["rewards_collected"])
            # Could add more from world_reward_tracking if needed
            
        self.episode_summaries.append(summary)
        
        # Save episode summary JSON
        with open(self.output_dir / f"episode_{host_buffer.episode_id:04d}_summary.json", "w") as f:
            json.dump(summary, f, indent=2, cls=NumpyEncoder)
        
        # Flush HDF5
        self.h5_file.flush()
        
        if self.log_to_console:
            print(f"\r[{datetime.now().strftime('%H:%M:%S')}] Episode {host_buffer.episode_id}: "
                  f"Steps={size}, Reward={host_stats['total_reward']:.2f}, "
                  f"Rate={size/episode_duration:.0f} steps/s")
        
        return summary
    
    def save_network_structure(
        self,
        neurons: Dict[str, Array],
        connections: Dict[str, Array],
        initial_weights: Optional[Array] = None
    ) -> None:
        """Save network structure (I/O operation)."""
        # Transfer from device to host
        host_neurons = jax.device_get(neurons)
        host_connections = jax.device_get(connections)
        
        net_group = self.h5_file.create_group("network_structure")
        
        # Save neurons
        neurons_group = net_group.create_group("neurons")
        for key, value in host_neurons.items():
            neurons_group.create_dataset(key, data=np.asarray(value))
            
        # Save connections
        conn_group = net_group.create_group("connections")
        for key, value in host_connections.items():
            conn_group.create_dataset(key, data=np.asarray(value))
            
        # Save initial weights if provided
        if initial_weights is not None:
            host_weights = jax.device_get(initial_weights)
            net_group.create_dataset("initial_weights", data=np.asarray(host_weights))
            
        self.h5_file.flush()
        
        if self.log_to_console:
            n_neurons = len(host_neurons.get("neuron_ids", []))
            n_connections = len(host_connections.get("source_ids", []))
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Saved network: "
                  f"{n_neurons} neurons, {n_connections} connections")
    
    def save_experiment_metadata(
        self,
        agent_version: str,
        agent_name: str,
        agent_description: str,
        world_version: str
    ) -> None:
        """Save experiment metadata."""
        if self.h5_file:
            self.h5_file.attrs["agent_version"] = agent_version
            self.h5_file.attrs["agent_name"] = agent_name
            self.h5_file.attrs["agent_description"] = agent_description
            self.h5_file.attrs["world_version"] = world_version
    
    def finalize(self) -> None:
        """Finalize experiment and save summary."""
        # Summary is saved in end_episode for each episode
        pass
    
    def close(self) -> None:
        """Close the exporter and save final summary (I/O operation)."""
        # Save experiment summary
        total_duration = time.time() - self.experiment_start_time
        
        summary = {
            "experiment_name": self.experiment_name,
            "timestamp": self.timestamp,
            "total_episodes": self.episode_count,
            "total_duration_seconds": total_duration,
            "config": {
                "experiment_name": self.config.experiment_name,
                "agent_version": self.config.agent_version,
                "world_version": self.config.world_version,
                "n_episodes": self.config.n_episodes,
                "seed": self.config.seed
            }
        }
        
        # Add aggregate statistics
        if self.episode_summaries:
            rewards = [s.get("total_reward", 0) for s in self.episode_summaries]
            summary["reward_stats"] = {
                "mean": np.mean(rewards),
                "std": np.std(rewards),
                "min": np.min(rewards),
                "max": np.max(rewards)
            }
            
            # Learning progress
            if len(rewards) >= 10:
                first_5 = np.mean(rewards[:5])
                last_5 = np.mean(rewards[-5:])
                summary["learning_progress"] = {
                    "first_5_avg": first_5,
                    "last_5_avg": last_5,
                    "improvement": last_5 - first_5
                }
        
        with open(self.output_dir / "experiment_summary.json", "w") as f:
            json.dump(summary, f, indent=2, cls=NumpyEncoder)
            
        # Close HDF5
        self.h5_file.attrs["end_time"] = datetime.now().isoformat()
        self.h5_file.attrs["total_episodes"] = self.episode_count
        self.h5_file.close()
        
        if self.log_to_console:
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Experiment complete")
            print(f"Total episodes: {self.episode_count}")
            print(f"Total time: {timedelta(seconds=int(total_duration))}")
            if "reward_stats" in summary:
                stats = summary["reward_stats"]
                print(f"Reward: {stats['mean']:.2f} ± {stats['std']:.2f}")
    
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()