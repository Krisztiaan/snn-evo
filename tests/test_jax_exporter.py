# keywords: [test, pytest, jax exporter, jit, minimal io]
"""Tests for JAX-based exporter with minimal I/O."""

import time
from pathlib import Path
import tempfile
import shutil

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest

from export.jax_data_exporter import JaxDataExporter, ExperimentConfig, add_timestep


@pytest.fixture(scope="module")
def temp_dir():
    """Create a temporary directory for the tests in this module."""
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)


class TestJaxExporter:
    """Test suite for JAX-based exporter."""
    
    def test_pure_jit_episode(self, temp_dir: Path):
        """Test that episode can run entirely in JIT without I/O."""
        config = ExperimentConfig(
            world_version="1.0",
            agent_version="1.0",
            world_params={},
            agent_params={},
            neural_params={"layers": [100, 50]},
            learning_params={"lr": 0.01},
            max_timesteps=1000,
            neural_dim=100,
            neural_sampling_rate=10
        )
        
        with JaxDataExporter("test_jit", config, temp_dir) as exporter:
            # Start episode - only I/O is buffer creation
            buffer, log_fn = exporter.start_episode()
            
            # This entire loop can be JIT-compiled
            @jax.jit
            def run_episode(initial_buffer, key):
                def step(carry, t):
                    buffer, key = carry
                    key, subkey = jrandom.split(key)
                    
                    # Generate data on device
                    neural_state = jrandom.normal(subkey, (config.neural_dim,))
                    reward = jrandom.uniform(subkey, minval=0, maxval=1)
                    action = jrandom.randint(subkey, (), 0, 4)
                    
                    # Update buffer (pure function)
                    new_buffer = log_fn(buffer, t, neural_state, reward, action)
                    return (new_buffer, key), None
                
                (final_buffer, _), _ = jax.lax.scan(step, (initial_buffer, key), jnp.arange(config.max_timesteps))
                return final_buffer
            
            # Run entire episode on device
            key = jrandom.PRNGKey(42)
            final_buffer = run_episode(buffer, key)
            
            # End episode - only I/O is persisting data
            summary = exporter.end_episode(final_buffer, success=True)
            
            # Verify results
            assert summary["timesteps"] == config.max_timesteps
            assert summary["total_reward"] > 0
            assert "mean_neural_activity" in summary
    
    def test_io_only_at_boundaries(self, temp_dir: Path):
        """Test that I/O only happens at episode boundaries."""
        config = ExperimentConfig(
            world_version="1.0",
            agent_version="1.0",
            world_params={},
            agent_params={},
            neural_params={},
            learning_params={},
            max_timesteps=10000,
            neural_dim=500,
            neural_sampling_rate=100
        )
        
        # Track file operations
        io_operations = []
        original_create_dataset = h5py.Group.create_dataset
        
        def tracked_create_dataset(self, *args, **kwargs):
            io_operations.append(("write", time.time()))
            return original_create_dataset(self, *args, **kwargs)
        
        # Monkey patch to track I/O
        h5py.Group.create_dataset = tracked_create_dataset
        
        try:
            with JaxDataExporter("test_io", config, temp_dir, log_to_console=False) as exporter:
                io_operations.clear()  # Clear setup I/O
                
                # Run episode
                buffer, log_fn = exporter.start_episode()
                start_time = time.time()
                
                # No I/O should happen during this loop
                for t in range(config.max_timesteps):
                    buffer = log_fn(
                        buffer,
                        t,
                        jnp.ones(config.neural_dim),
                        1.0 if t % 100 == 0 else 0.0,
                        t % 4
                    )
                
                loop_time = time.time() - start_time
                
                # I/O only happens here
                io_start = time.time()
                exporter.end_episode(buffer)
                io_time = time.time() - io_start
                
            # Verify I/O pattern
            assert len(io_operations) > 0, "Should have I/O operations"
            assert all(timestamp >= io_start for op, timestamp in io_operations), "I/O should only happen at end"
            
            print(f"\nI/O timing test:")
            print(f"  Loop time: {loop_time:.3f}s ({config.max_timesteps/loop_time:.0f} steps/s)")
            print(f"  I/O time: {io_time:.3f}s")
            print(f"  I/O operations: {len(io_operations)}")
            
        finally:
            # Restore original method
            h5py.Group.create_dataset = original_create_dataset
    
    def test_neural_sampling_efficiency(self, temp_dir: Path):
        """Test that neural sampling reduces data size efficiently."""
        config = ExperimentConfig(
            world_version="1.0",
            agent_version="1.0",
            world_params={},
            agent_params={},
            neural_params={},
            learning_params={},
            max_timesteps=1000,
            neural_dim=1000,
            neural_sampling_rate=100  # 100x reduction
        )
        
        with JaxDataExporter("test_sampling", config, temp_dir) as exporter:
            buffer, log_fn = exporter.start_episode()
            
            # Fill buffer
            key = jrandom.PRNGKey(0)
            for t in range(config.max_timesteps):
                key, subkey = jrandom.split(key)
                buffer = log_fn(
                    buffer,
                    t,
                    jrandom.normal(subkey, (config.neural_dim,)),
                    0.0,
                    0
                )
            
            exporter.end_episode(buffer)
        
        # Check saved data size
        h5_path = next(temp_dir.glob("test_sampling_*/experiment_data.h5"))
        with h5py.File(h5_path, "r") as f:
            neural_data = f["episodes/episode_0000/neural_states"]
            
            # Should be sampled down
            expected_samples = config.max_timesteps // config.neural_sampling_rate
            assert neural_data.shape[0] == expected_samples
            assert neural_data.shape[1] == config.neural_dim
            
            # Check compression
            assert neural_data.compression is not None
            
            print(f"\nSampling efficiency:")
            print(f"  Original: {config.max_timesteps} x {config.neural_dim}")
            print(f"  Sampled: {neural_data.shape[0]} x {neural_data.shape[1]}")
            print(f"  Reduction: {config.max_timesteps / neural_data.shape[0]:.0f}x")
    
    def test_device_to_host_transfer(self, temp_dir: Path):
        """Test efficient device-to-host transfer at episode end."""
        config = ExperimentConfig(
            world_version="1.0",
            agent_version="1.0",
            world_params={},
            agent_params={},
            neural_params={},
            learning_params={},
            max_timesteps=5000,
            neural_dim=2000,
            neural_sampling_rate=50
        )
        
        with JaxDataExporter("test_transfer", config, temp_dir, log_to_console=False) as exporter:
            buffer, log_fn = exporter.start_episode()
            
            # Generate all data on device at once
            key = jrandom.PRNGKey(123)
            all_neural = jrandom.normal(key, (config.max_timesteps, config.neural_dim))
            
            # Time the pure computation
            compute_start = time.time()
            
            @jax.jit
            def fill_buffer(buffer, neural_states):
                def step(buffer, i):
                    return log_fn(buffer, i, neural_states[i], 0.1, i % 4), None
                
                final_buffer, _ = jax.lax.scan(step, buffer, jnp.arange(config.max_timesteps))
                return final_buffer
            
            final_buffer = fill_buffer(buffer, all_neural)
            compute_time = time.time() - compute_start
            
            # Time the transfer and I/O
            transfer_start = time.time()
            summary = exporter.end_episode(final_buffer)
            transfer_time = time.time() - transfer_start
            
            print(f"\nDevice-to-host transfer:")
            print(f"  Data size: {config.max_timesteps} x {config.neural_dim}")
            print(f"  Compute time: {compute_time:.3f}s")
            print(f"  Transfer + I/O time: {transfer_time:.3f}s")
            print(f"  Compute rate: {config.max_timesteps / compute_time:.0f} steps/s")
            
            # Compute should be much faster than transfer
            assert compute_time < transfer_time * 0.5, "Computation should be faster than I/O"
    
    def test_multiple_episodes_batch_pattern(self, temp_dir: Path):
        """Test running multiple episodes with minimal I/O overhead."""
        config = ExperimentConfig(
            world_version="1.0",
            agent_version="1.0",
            world_params={},
            agent_params={},
            neural_params={},
            learning_params={},
            max_timesteps=1000,
            neural_dim=100,
            neural_sampling_rate=10
        )
        
        n_episodes = 10
        episode_times = []
        
        with JaxDataExporter("test_batch", config, temp_dir, log_to_console=False) as exporter:
            # Save network structure once
            neurons = {"neuron_ids": jnp.arange(config.neural_dim)}
            connections = {"source_ids": jnp.arange(50), "target_ids": jnp.arange(50, 100)}
            exporter.save_network_structure(neurons, connections)
            
            for ep in range(n_episodes):
                ep_start = time.time()
                
                # Start episode
                buffer, log_fn = exporter.start_episode()
                
                # Run on device
                @jax.jit
                def run_episode_jit(buffer, seed):
                    key = jrandom.PRNGKey(seed)
                    
                    def step(carry, t):
                        buf, k = carry
                        k, sk = jrandom.split(k)
                        neural = jrandom.normal(sk, (config.neural_dim,))
                        reward = jnp.where(t % 50 == 0, 1.0, 0.0)
                        action = t % 4
                        return (log_fn(buf, t, neural, reward, action), k), None
                    
                    (final_buf, _), _ = jax.lax.scan(step, (buffer, key), jnp.arange(config.max_timesteps))
                    return final_buf
                
                final_buffer = run_episode_jit(buffer, ep)
                
                # End episode
                exporter.end_episode(final_buffer, success=(ep % 2 == 0))
                
                episode_times.append(time.time() - ep_start)
        
        # Analyze timing
        avg_time = np.mean(episode_times)
        std_time = np.std(episode_times)
        
        print(f"\nMultiple episodes timing:")
        print(f"  Episodes: {n_episodes}")
        print(f"  Avg time per episode: {avg_time:.3f}s ± {std_time:.3f}s")
        print(f"  Total time: {sum(episode_times):.3f}s")
        print(f"  Throughput: {n_episodes * config.max_timesteps / sum(episode_times):.0f} steps/s")
        
        # Verify all episodes saved
        h5_path = next(temp_dir.glob("test_batch_*/experiment_data.h5"))
        with h5py.File(h5_path, "r") as f:
            assert len([k for k in f["episodes"].keys() if k.startswith("episode_")]) == n_episodes
    
    def test_statistics_computation_on_device(self, temp_dir: Path):
        """Test that statistics are computed on device before transfer."""
        config = ExperimentConfig(
            world_version="1.0",
            agent_version="1.0",
            world_params={},
            agent_params={},
            neural_params={},
            learning_params={},
            max_timesteps=1000,
            neural_dim=100,
            neural_sampling_rate=1  # No sampling to test full stats
        )
        
        with JaxDataExporter("test_stats", config, temp_dir) as exporter:
            buffer, log_fn = exporter.start_episode()
            
            # Create specific pattern for testing
            for t in range(config.max_timesteps):
                neural_state = jnp.ones(config.neural_dim) * (t / config.max_timesteps)
                reward = 1.0 if t % 100 == 0 else 0.0
                action = t % 4
                buffer = log_fn(buffer, t, neural_state, reward, action)
            
            # End episode - stats computed on device
            summary = exporter.end_episode(buffer)
            
            # Verify statistics
            assert summary["total_reward"] == 10.0  # 10 rewards of 1.0
            assert summary["rewards_collected"] == 10
            assert summary["episode_length"] == config.max_timesteps
            assert 0.4 < summary["mean_neural_activity"] < 0.6  # Should be ~0.5
            assert summary["action_entropy"] > 1.0  # Should be ~1.386 for uniform distribution
            
            print(f"\nOn-device statistics:")
            for key, value in summary.items():
                if isinstance(value, (int, float)):
                    print(f"  {key}: {value:.3f}")
    
    def test_memory_efficiency_with_jax(self, temp_dir: Path):
        """Test memory efficiency with JAX arrays."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
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
        
        with JaxDataExporter("test_memory", config, temp_dir, log_to_console=False) as exporter:
            # Pre-allocate all data on device
            key = jrandom.PRNGKey(42)
            all_neural = jrandom.normal(key, (config.max_timesteps, config.neural_dim))
            
            mid_memory = process.memory_info().rss / 1024 / 1024
            
            # Run episode
            buffer, log_fn = exporter.start_episode()
            
            @jax.jit
            def process_all(buffer, neural_data):
                def step(buf, i):
                    return log_fn(buf, i, neural_data[i], 0.0, 0), None
                final_buf, _ = jax.lax.scan(step, buffer, jnp.arange(config.max_timesteps))
                return final_buf
            
            final_buffer = process_all(buffer, all_neural)
            exporter.end_episode(final_buffer)
            
            # Clear device memory
            del all_neural
            del final_buffer
            jax.clear_caches()
            
        final_memory = process.memory_info().rss / 1024 / 1024
        
        print(f"\nMemory efficiency:")
        print(f"  Initial: {initial_memory:.1f} MB")
        print(f"  After allocation: {mid_memory:.1f} MB (+{mid_memory - initial_memory:.1f} MB)")
        print(f"  Final: {final_memory:.1f} MB (+{final_memory - initial_memory:.1f} MB)")
        
        # Memory should be released after processing
        memory_increase = final_memory - initial_memory
        assert memory_increase < 200, f"Memory leak detected: {memory_increase:.1f} MB"


import h5py  # Import at module level for monkey patching