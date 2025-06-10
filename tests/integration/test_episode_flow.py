# keywords: [integration test, episode flow, performance, type safety]
"""Integration test for complete episode flow with type-safe interfaces."""

import time
from pathlib import Path
import tempfile
import shutil

import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest

from interfaces import (
    WorldProtocol, 
    AgentProtocol, 
    ExporterProtocol,
    ExperimentConfig,
    WorldConfig,
    NeuralConfig,
    PlasticityConfig,
    AgentBehaviorConfig,
)

class TestEpisodeFlow:
    """Test complete episode execution flow with performance validation."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        d = tempfile.mkdtemp()
        yield Path(d)
        shutil.rmtree(d)
    
    @pytest.fixture
    def config(self):
        """Create test experiment configuration."""
        return ExperimentConfig(
            world=WorldConfig(grid_size=50, n_rewards=50, max_timesteps=1000),
            neural=NeuralConfig(n_neurons=100, n_excitatory=80, n_inhibitory=20),
            plasticity=PlasticityConfig(enable_stdp=True, enable_reward_modulation=True),
            behavior=AgentBehaviorConfig(action_noise=0.1, temperature=1.0),
            experiment_name="integration_test",
            agent_version="1.0.0",
            world_version="0.0.3",
            n_episodes=5,
            seed=42,
            device="cpu",
            export_dir="",
            neural_sampling_rate=10,
            flush_at_episode_end=True
        )
    
    def test_episode_performance(self, temp_dir: Path, config: ExperimentConfig):
        """Test that episode runs at expected performance levels."""
        # Import actual implementations
        from world.simple_grid_0003 import SimpleGridWorld
        from models.phase_0_13 import SnnAgent
        from export import DataExporter
        
        # Verify they implement protocols
        assert isinstance(SimpleGridWorld, type) and hasattr(SimpleGridWorld, 'reset')
        assert isinstance(SnnAgent, type) and hasattr(SnnAgent, 'act')
        assert isinstance(DataExporter, type) and hasattr(DataExporter, 'start_episode')
        
        # Initialize components
        world = SimpleGridWorld(config.world)
        with DataExporter(config.experiment_name, config, temp_dir) as exporter:
            agent = SnnAgent(config, exporter)
            
            # Run episode
            key = jrandom.PRNGKey(config.seed)
            key, reset_key = jrandom.split(key)
            
            # Start timing
            start_time = time.perf_counter()
            
            # Reset world and agent
            gradient = world.reset(reset_key)
            key, agent_key = jrandom.split(key)
            agent.reset(agent_key)
            
            # Get episode buffer and logging function
            buffer, log_fn = exporter.start_episode(0)
            
            # Run episode
            for timestep in range(config.world.max_timesteps):
                # Agent selects action
                key, action_key = jrandom.split(key)
                action = agent.act(gradient, action_key)
                
                # World step
                gradient = world.step(action)
                
                # Log timestep (JIT-compiled, no I/O)
                neural_state = agent.get_neural_state() if hasattr(agent, 'get_neural_state') else None
                buffer = log_fn(buffer, timestep, gradient, action, neural_state)
                
                # Check for episode end (simplified)
                if timestep > 100 and jrandom.uniform(key) < 0.01:
                    break
            
            # End timing before I/O
            runtime = time.perf_counter() - start_time
            steps_per_second = timestep / runtime
            
            # Get episode data and finalize
            episode_data = agent.get_episode_data()
            reward_tracking = world.get_reward_tracking()
            stats = exporter.end_episode(buffer, episode_data, reward_tracking, success=True)
            
            # Performance assertions
            assert steps_per_second > 1000, f"Episode too slow: {steps_per_second:.0f} steps/s"
            assert stats["timesteps"] == timestep + 1
            assert "action_entropy" in stats
            
    def test_type_safety(self, temp_dir: Path, config: ExperimentConfig):
        """Test that type violations are caught."""
        from export import DataExporter
        
        with DataExporter(config.experiment_name, config, temp_dir) as exporter:
            buffer, log_fn = exporter.start_episode(0)
            
            # These should fail type checking (mypy/pyright would catch)
            # but won't fail at runtime without explicit validation
            
            # Test 1: Wrong argument types
            with pytest.raises((TypeError, ValueError)):
                # String instead of int for timestep
                log_fn(buffer, "0", jnp.array(0.5), 1, None)
            
            # Test 2: Wrong array shapes
            with pytest.raises((ValueError, IndexError)):
                # 2D array instead of scalar for gradient
                log_fn(buffer, 0, jnp.ones((2, 2)), 1, None)
    
    def test_jit_compilation(self, temp_dir: Path, config: ExperimentConfig):
        """Test that logging function is properly JIT-compiled."""
        from export import DataExporter
        
        with DataExporter(config.experiment_name, config, temp_dir) as exporter:
            buffer, log_fn = exporter.start_episode(0)
            
            # First call compiles
            start = time.perf_counter()
            buffer = log_fn(buffer, 0, jnp.array(0.5), 1, None)
            first_call = time.perf_counter() - start
            
            # Subsequent calls should be much faster
            start = time.perf_counter()
            for i in range(100):
                buffer = log_fn(buffer, i+1, jnp.array(0.5), i % 9, None)
            subsequent_avg = (time.perf_counter() - start) / 100
            
            # JIT speedup should be significant
            assert subsequent_avg < first_call / 10, "JIT compilation not effective"