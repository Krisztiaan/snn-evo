# keywords: [export test, performance, jit, hdf5]
"""Performance tests for data exporter."""

import time
from pathlib import Path
import tempfile
import shutil

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest

from interfaces import ExperimentConfig, WorldConfig, NeuralConfig, PlasticityConfig, AgentBehaviorConfig
from interfaces.episode_data import EpisodeData

class TestExporterPerformance:
    """Test exporter performance and correctness."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        d = tempfile.mkdtemp()
        yield Path(d)
        shutil.rmtree(d)
    
    @pytest.fixture
    def config(self):
        """Test configuration."""
        return ExperimentConfig(
            world=WorldConfig(grid_size=100, n_rewards=300, max_timesteps=10000),
            neural=NeuralConfig(n_neurons=1000),
            plasticity=PlasticityConfig(),
            behavior=AgentBehaviorConfig(),
            experiment_name="test_export",
            agent_version="1.0.0",
            world_version="0.0.3",
            n_episodes=10,
            seed=42,
            neural_sampling_rate=100,  # Sample every 100 steps
            flush_at_episode_end=True
        )
    
    def test_jit_logging_performance(self, temp_dir: Path, config: ExperimentConfig):
        """Test that timestep logging is JIT-optimized."""
        from export import DataExporter
        
        with DataExporter("perf_test", config, temp_dir) as exporter:
            buffer, log_fn = exporter.start_episode(0)
            
            # Create test data
            neural_state = jnp.ones(config.neural.n_neurons)
            
            # Warmup JIT
            for i in range(10):
                buffer = log_fn(buffer, i, jnp.array(0.5), i % 9, neural_state)
            
            # Time many log calls
            n_calls = 10000
            start = time.perf_counter()
            for i in range(n_calls):
                gradient = jnp.array(0.8 if i % 100 != 0 else 1.0)  # Occasional reward
                buffer = log_fn(buffer, i, gradient, i % 9, neural_state)
            jit_duration = time.perf_counter() - start
            
            logs_per_second = n_calls / jit_duration
            assert logs_per_second > 100000, f"Logging too slow: {logs_per_second:.0f} logs/s"
            
            # Verify buffer was updated correctly
            assert buffer.current_size == n_calls + 10  # Including warmup
    
    def test_episode_end_performance(self, temp_dir: Path, config: ExperimentConfig):
        """Test episode finalization performance."""
        from export import DataExporter
        
        with DataExporter("end_test", config, temp_dir) as exporter:
            # Generate full episode data
            n_steps = 5000
            buffer, log_fn = exporter.start_episode(0)
            
            # Fill buffer
            neural_state = jnp.ones(config.neural.n_neurons)
            for i in range(n_steps):
                buffer = log_fn(buffer, i, jnp.array(0.5), i % 9, neural_state)
            
            # Create episode data
            episode_data = EpisodeData(
                actions=jnp.arange(n_steps) % 9,
                gradients=jnp.ones(n_steps) * 0.5,
                neural_states=jnp.ones((n_steps, config.neural.n_neurons)),
                total_reward_events=50
            )
            
            # Time episode end (includes I/O)
            start = time.perf_counter()
            stats = exporter.end_episode(
                buffer,
                episode_data,
                {"positions": jnp.zeros((300, 2)), 
                 "spawn_steps": jnp.zeros(300),
                 "collect_steps": jnp.ones(300) * -1},
                success=True
            )
            end_duration = time.perf_counter() - start
            
            # Should complete reasonably fast even with I/O
            assert end_duration < 1.0, f"Episode end too slow: {end_duration:.2f}s"
            assert stats["timesteps"] == n_steps
    
    def test_neural_sampling(self, temp_dir: Path, config: ExperimentConfig):
        """Test neural state sampling reduces data size."""
        from export import DataExporter
        from export.loader import ExperimentLoader
        
        # Run with sampling
        with DataExporter("sample_test", config, temp_dir) as exporter:
            n_steps = 1000
            buffer, log_fn = exporter.start_episode(0)
            
            neural_state = jnp.ones(config.neural.n_neurons)
            for i in range(n_steps):
                buffer = log_fn(buffer, i, jnp.array(0.5), 0, neural_state)
            
            episode_data = EpisodeData(
                actions=jnp.zeros(n_steps, dtype=jnp.int32),
                gradients=jnp.ones(n_steps) * 0.5,
                neural_states=jnp.ones((n_steps, config.neural.n_neurons)),
                total_reward_events=0
            )
            
            exporter.end_episode(buffer, episode_data, {}, True)
        
        # Load and verify sampling
        with ExperimentLoader(temp_dir / "sample_test_*") as loader:
            episode = loader.get_episode(0)
            neural_states = episode.get_neural_states()
            
            # Should be sampled
            expected_samples = n_steps // config.neural_sampling_rate
            assert neural_states["neural_states"].shape[0] <= expected_samples + 1
    
    def test_compression_effectiveness(self, temp_dir: Path, config: ExperimentConfig):
        """Test HDF5 compression reduces file size."""
        from export import DataExporter
        import os
        
        # Run without compression
        config_no_comp = config._replace(experiment_name="no_compression")
        with DataExporter("no_comp", config_no_comp, temp_dir, compression=None) as exporter:
            self._run_dummy_episode(exporter, config)
        
        # Run with compression
        config_comp = config._replace(experiment_name="with_compression")
        with DataExporter("comp", config_comp, temp_dir, compression="gzip") as exporter:
            self._run_dummy_episode(exporter, config)
        
        # Compare file sizes
        no_comp_size = os.path.getsize(list(temp_dir.glob("no_comp*/experiment_data.h5"))[0])
        comp_size = os.path.getsize(list(temp_dir.glob("comp*/experiment_data.h5"))[0])
        
        compression_ratio = no_comp_size / comp_size
        assert compression_ratio > 2.0, f"Compression not effective: {compression_ratio:.1f}x"
    
    def test_buffer_overflow_handling(self, temp_dir: Path, config: ExperimentConfig):
        """Test buffer handles more timesteps than allocated."""
        from export import DataExporter
        
        # Use small buffer config
        small_config = config._replace(
            world=WorldConfig(grid_size=10, n_rewards=10, max_timesteps=100)
        )
        
        with DataExporter("overflow_test", small_config, temp_dir) as exporter:
            buffer, log_fn = exporter.start_episode(0)
            
            # Try to log more than max_timesteps
            with pytest.raises((IndexError, RuntimeError)):
                for i in range(200):  # Double the buffer size
                    buffer = log_fn(buffer, i, jnp.array(0.5), 0, None)
    
    def _run_dummy_episode(self, exporter, config):
        """Helper to run a dummy episode."""
        buffer, log_fn = exporter.start_episode(0)
        
        n_steps = 1000
        neural_state = jnp.ones(config.neural.n_neurons)
        for i in range(n_steps):
            buffer = log_fn(buffer, i, jnp.array(0.5), i % 9, neural_state)
        
        episode_data = EpisodeData(
            actions=jnp.arange(n_steps) % 9,
            gradients=jnp.ones(n_steps) * 0.5,
            neural_states=jnp.ones((n_steps, config.neural.n_neurons)),
            total_reward_events=10
        )
        
        exporter.end_episode(buffer, episode_data, {}, True)