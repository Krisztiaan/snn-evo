# keywords: [agent test, interface compliance, performance, generic]
"""Test agent implementations comply with interface and performance requirements."""

import time
from pathlib import Path
import tempfile
from typing import Type

import jax
import jax.numpy as jnp
import jax.random as jrandom
import pytest

from interfaces import AgentProtocol, ExperimentConfig, WorldConfig, NeuralConfig, PlasticityConfig, AgentBehaviorConfig
from interfaces.episode_data import EpisodeData

class TestAgentInterface:
    """Generic tests for any agent implementation."""
    
    @pytest.fixture
    def config(self):
        """Standard test configuration."""
        return ExperimentConfig(
            world=WorldConfig(grid_size=50, n_rewards=50, max_timesteps=1000),
            neural=NeuralConfig(n_neurons=100, n_excitatory=80, n_inhibitory=20),
            plasticity=PlasticityConfig(enable_stdp=True),
            behavior=AgentBehaviorConfig(action_noise=0.1),
            experiment_name="agent_test",
            agent_version="test",
            world_version="0.0.3",
            n_episodes=1,
            seed=42
        )
    
    def _test_agent_protocol(self, agent_class: Type[AgentProtocol], config: ExperimentConfig):
        """Generic test for any agent implementing the protocol."""
        from export import DataExporter
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with DataExporter("test", config, Path(tmpdir)) as exporter:
                # Check class attributes
                assert hasattr(agent_class, 'VERSION'), "Agent must have VERSION"
                assert hasattr(agent_class, 'MODEL_NAME'), "Agent must have MODEL_NAME"
                assert hasattr(agent_class, 'DESCRIPTION'), "Agent must have DESCRIPTION"
                
                # Initialize agent
                agent = agent_class(config, exporter)
                
                # Test reset
                key = jrandom.PRNGKey(42)
                agent.reset(key)
                
                # Test act
                gradient = jnp.array(0.5)
                key, subkey = jrandom.split(key)
                action = agent.act(gradient, subkey)
                assert isinstance(action, (int, jnp.integer)), "Action must be integer"
                assert 0 <= action <= 8, f"Action {action} out of range [0-8]"
                
                # Test episode data
                data = agent.get_episode_data()
                assert isinstance(data, EpisodeData), "Must return EpisodeData instance"
                
                return agent
    
    def test_latest_agent_implementation(self, config: ExperimentConfig):
        """Test the latest agent implementation."""
        # Import the latest agent - update this when creating new versions
        from models import get_latest_agent
        
        agent_class = get_latest_agent()
        self._test_agent_protocol(agent_class, config)
    
    def test_agent_performance_generic(self, config: ExperimentConfig):
        """Test any agent achieves minimum performance."""
        from models import get_latest_agent
        from export import DataExporter
        
        agent_class = get_latest_agent()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with DataExporter("test", config, Path(tmpdir)) as exporter:
                agent = agent_class(config, exporter)
                
                key = jrandom.PRNGKey(42)
                agent.reset(key)
                
                # Warmup for JIT
                for i in range(10):
                    key, subkey = jrandom.split(key)
                    agent.act(jnp.array(0.5), subkey)
                
                # Time many actions
                n_actions = 1000
                start = time.perf_counter()
                for i in range(n_actions):
                    key, subkey = jrandom.split(key)
                    gradient = jnp.array(0.8 if i % 10 != 0 else 1.0)
                    action = agent.act(gradient, subkey)
                duration = time.perf_counter() - start
                
                actions_per_second = n_actions / duration
                # Minimum performance threshold
                assert actions_per_second > 100, f"Agent too slow: {actions_per_second:.0f} actions/s"
    
    def test_episode_data_validity(self, config: ExperimentConfig):
        """Test agent returns valid episode data."""
        from models import get_latest_agent
        from export import DataExporter
        
        agent_class = get_latest_agent()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with DataExporter("test", config, Path(tmpdir)) as exporter:
                agent = agent_class(config, exporter)
                
                key = jrandom.PRNGKey(42)
                agent.reset(key)
                
                # Run some steps
                n_steps = 100
                for i in range(n_steps):
                    key, subkey = jrandom.split(key)
                    agent.act(jnp.array(0.5), subkey)
                
                # Get episode data
                data = agent.get_episode_data()
                
                # Check required fields
                assert data.actions is not None, "Actions must be present"
                assert data.gradients is not None, "Gradients must be present"
                assert len(data.actions) == len(data.gradients), "Actions and gradients must have same length"
                assert len(data.actions) <= n_steps, "Cannot have more data than steps taken"
                
                # Validate data types and ranges
                assert jnp.all((data.actions >= 0) & (data.actions <= 8)), "Actions out of range"
                assert jnp.all((data.gradients >= 0) & (data.gradients <= 1)), "Gradients out of range"
                
                # Check consistency of optional fields
                if data.neural_states is not None:
                    assert data.neural_states.shape[0] == len(data.actions), "Neural states length mismatch"
                
                if data.weights_initial is not None and data.weights_final is not None:
                    assert data.weights_initial.shape == data.weights_final.shape, "Weight shapes mismatch"
    
    def test_deterministic_reset(self, config: ExperimentConfig):
        """Test agent reset is deterministic with same key."""
        from models import get_latest_agent
        from export import DataExporter
        
        agent_class = get_latest_agent()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with DataExporter("test", config, Path(tmpdir)) as exporter:
                agent1 = agent_class(config, exporter)
                agent2 = agent_class(config, exporter)
                
                # Reset with same key
                key = jrandom.PRNGKey(42)
                agent1.reset(key)
                agent2.reset(key)
                
                # Should produce same actions with same inputs
                test_key = jrandom.PRNGKey(123)
                action1 = agent1.act(jnp.array(0.5), test_key)
                action2 = agent2.act(jnp.array(0.5), test_key)
                
                assert action1 == action2, "Reset should be deterministic"
    
    @pytest.mark.parametrize("gradient", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_gradient_handling(self, config: ExperimentConfig, gradient: float):
        """Test agent handles different gradient values."""
        from models import get_latest_agent
        from export import DataExporter
        
        agent_class = get_latest_agent()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with DataExporter("test", config, Path(tmpdir)) as exporter:
                agent = agent_class(config, exporter)
                
                key = jrandom.PRNGKey(42)
                agent.reset(key)
                
                # Test action selection with different gradients
                key, subkey = jrandom.split(key)
                action = agent.act(jnp.array(gradient), subkey)
                
                assert 0 <= action <= 8, f"Invalid action {action} for gradient {gradient}"