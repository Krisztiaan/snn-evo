# keywords: [test, integration, jax exporter, end-to-end]
"""Integration tests for the JAX-based data exporter."""

import time
from pathlib import Path
import tempfile
import shutil

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest

from export import DataExporter, ExperimentConfig, ExperimentLoader


@pytest.fixture(scope="module")
def temp_dir():
    """Create a temporary directory for the tests in this module."""
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)


class TestJAXIntegration:
    """End-to-end integration tests for JAX-based exporter."""
    
    def test_full_experiment_workflow(self, temp_dir: Path):
        """Test complete experiment workflow with JAX."""
        # Configuration
        config = ExperimentConfig(
            world_version="1.0",
            agent_version="1.0",
            world_params={"grid_size": 10},
            agent_params={"learning_rate": 0.01},
            neural_params={"layers": [100, 50, 10]},
            learning_params={"algorithm": "STDP"},
            max_timesteps=1000,
            neural_dim=100,
            neural_sampling_rate=10
        )
        
        n_episodes = 5
        
        # Run experiment
        with DataExporter("test_integration", config, temp_dir) as exporter:
            # Save network structure
            neurons = {"neuron_ids": jnp.arange(config.neural_dim)}
            connections = {
                "source_ids": jnp.arange(50),
                "target_ids": jnp.arange(50, 100)
            }
            exporter.save_network_structure(neurons, connections)
            
            # Run episodes
            episode_rewards = []
            for ep in range(n_episodes):
                # JIT-compiled episode execution
                @jax.jit
                def run_episode(key):
                    buffer, log_fn = exporter.start_episode()
                    
                    def step(carry, t):
                        buf, k, total_reward = carry
                        k, sk = jrandom.split(k)
                        
                        # Generate data
                        neural = jrandom.normal(sk, (config.neural_dim,))
                        reward = jnp.where(t % 100 == 0, 1.0, 0.0)
                        action = t % 4
                        
                        # Update buffer
                        new_buf = log_fn(buf, t, neural, reward, action)
                        new_total = total_reward + reward
                        
                        return (new_buf, k, new_total), None
                    
                    (final_buffer, _, total_reward), _ = jax.lax.scan(
                        step, 
                        (buffer, key, 0.0), 
                        jnp.arange(config.max_timesteps)
                    )
                    
                    return final_buffer, total_reward
                
                # Execute episode
                key = jrandom.PRNGKey(ep)
                buffer, _ = exporter.start_episode()
                
                # Run timesteps
                total_reward = 0.0
                for t in range(config.max_timesteps):
                    key, subkey = jrandom.split(key)
                    neural_state = jrandom.normal(subkey, (config.neural_dim,))
                    reward = 1.0 if t % 100 == 0 else 0.0
                    action = t % 4
                    
                    buffer = add_timestep(buffer, t, neural_state, reward, action)
                    total_reward += reward
                
                # End episode
                summary = exporter.end_episode(buffer, success=(ep % 2 == 0))
                episode_rewards.append(summary["total_reward"])
        
        # Verify saved data
        with ExperimentLoader(next(temp_dir.glob("test_integration_*"))) as loader:
            # Check metadata
            metadata = loader.get_metadata()
            assert metadata["experiment_name"] == "test_integration"
            assert metadata["neural_dim"] == config.neural_dim
            assert metadata["neural_sampling_rate"] == config.neural_sampling_rate
            
            # Check episodes
            episodes = loader.list_episodes()
            assert len(episodes) == n_episodes
            
            # Check first episode data
            ep0 = loader.get_episode(0)
            
            # Check that episode has data
            h5 = loader.h5_file
            ep_group = h5["episodes"]["episode_0000"]
            assert "neural_states" in ep_group
            assert "rewards" in ep_group
            assert "actions" in ep_group
            assert "timesteps" in ep_group
            
            # Check network structure
            assert "network_structure" in h5
            assert "neurons" in h5["network_structure"]
            assert "connections" in h5["network_structure"]
            
            # Check experiment summary
            summary_path = loader.h5_path.parent / "experiment_summary.json"
            assert summary_path.exists()
    
    def test_performance_characteristics(self, temp_dir: Path):
        """Test that JAX provides expected performance characteristics."""
        config = ExperimentConfig(
            world_version="1.0",
            agent_version="1.0",
            world_params={},
            agent_params={},
            neural_params={},
            learning_params={},
            max_timesteps=10000,
            neural_dim=1000,
            neural_sampling_rate=100
        )
        
        with DataExporter("test_perf", config, temp_dir) as exporter:
            buffer, log_fn = exporter.start_episode()
            
            # Measure performance (no warmup to avoid buffer overflow)
            start = time.time()
            for t in range(config.max_timesteps):
                neural_state = jnp.ones(config.neural_dim) * t / config.max_timesteps
                reward = 1.0 if t % 1000 == 0 else 0.0
                action = t % 4
                buffer = log_fn(buffer, t, neural_state, reward, action)
            
            compute_time = time.time() - start
            
            # I/O should happen only here
            io_start = time.time()
            summary = exporter.end_episode(buffer, success=True)
            io_time = time.time() - io_start
            
            # Performance assertions
            steps_per_second = config.max_timesteps / compute_time
            print(f"\nPerformance test:")
            print(f"  Compute: {compute_time:.3f}s ({steps_per_second:.0f} steps/s)")
            print(f"  I/O: {io_time:.3f}s")
            print(f"  Ratio: {compute_time/io_time:.1f}:1")
            
            # Should achieve good performance
            assert steps_per_second > 300, "JAX should achieve reasonable performance"
            assert compute_time < io_time * 60, "Compute should be reasonably fast relative to I/O"
    
    def test_data_correctness(self, temp_dir: Path):
        """Test that data is correctly processed and saved."""
        config = ExperimentConfig(
            world_version="1.0",
            agent_version="1.0",
            world_params={},
            agent_params={},
            neural_params={},
            learning_params={},
            max_timesteps=100,
            neural_dim=10,
            neural_sampling_rate=5  # Sample every 5 steps
        )
        
        # Generate predictable data
        neural_pattern = jnp.arange(config.neural_dim, dtype=jnp.float32)
        rewards_pattern = [1.0 if i % 10 == 0 else 0.0 for i in range(config.max_timesteps)]
        
        with DataExporter("test_correct", config, temp_dir) as exporter:
            buffer, log_fn = exporter.start_episode()
            
            # Log predictable data
            for t in range(config.max_timesteps):
                neural_state = neural_pattern * (t + 1)
                buffer = log_fn(buffer, t, neural_state, rewards_pattern[t], t % 4)
            
            summary = exporter.end_episode(buffer, success=True)
            
            # Verify summary
            assert summary["total_reward"] == sum(rewards_pattern)
            assert summary["rewards_collected"] == len([r for r in rewards_pattern if r > 0])
            assert summary["episode_length"] == config.max_timesteps
        
        # Load and verify
        with ExperimentLoader(next(temp_dir.glob("test_correct_*"))) as loader:
            ep = loader.get_episode(0)
            
            # Check data directly from HDF5
            h5 = loader.h5_file
            ep_group = h5["episodes"]["episode_0000"]
            neural_data = ep_group["neural_states"][:]
            
            # Check sampling worked correctly
            expected_samples = config.max_timesteps // config.neural_sampling_rate
            assert neural_data.shape == (expected_samples, config.neural_dim)
            
            # First sample should be average of timesteps 0-4
            # Each timestep t has pattern * (t + 1), so average is pattern * (0+1+2+3+4+1)/5
            expected_first = neural_pattern * 3.0  # Average of 1,2,3,4,5
            np.testing.assert_allclose(neural_data[0], expected_first, rtol=1e-5)


# Import add_timestep for the integration test
from export.jax_data_exporter import add_timestep