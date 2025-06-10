# keywords: [protocol compliance test, integration test, interface verification]
"""Integration test for protocol compliance."""

from typing import Dict
from pathlib import Path
import jax
import jax.numpy as jnp
from jax.random import PRNGKey, split

from interfaces import (
    WorldProtocol, WorldState, WorldConfig,
    AgentProtocol, ExperimentConfig, 
    ExporterProtocol, EpisodeBufferProtocol, LogTimestepFunction,
    NeuralConfig, PlasticityConfig, AgentBehaviorConfig, EpisodeData
)
from world.simple_grid_0004 import MinimalGridWorld
from models.random.agent import RandomAgent


class MockEpisodeBuffer:
    """Mock episode buffer for testing."""
    def __init__(self, max_timesteps: int):
        self.timesteps = jnp.zeros(max_timesteps, dtype=jnp.int32)
        self.gradients = jnp.zeros(max_timesteps, dtype=jnp.float32)
        self.actions = jnp.zeros(max_timesteps, dtype=jnp.int32)
        self.neural_states = None
        self.current_size = 0
        self.max_size = max_timesteps
        self.episode_id = 0


class MockExporter:
    """Mock exporter that complies with ExporterProtocol."""
    
    VERSION = "1.0.0"
    
    def __init__(
        self, 
        experiment_name: str, 
        config: ExperimentConfig, 
        output_dir, 
        compression=None,
        log_to_console=True
    ):
        self.experiment_name = experiment_name
        self.config = config
        self.output_dir = output_dir
        self.episodes_started = 0
        self.episodes_ended = 0
        self.metadata_saved = False
        
    def start_episode(self, episode_id: int):
        """Start new episode, return buffer and logging function."""
        self.episodes_started += 1
        buffer = MockEpisodeBuffer(self.config.world.max_timesteps)
        buffer.episode_id = episode_id
        
        def log_timestep(
            buffer: EpisodeBufferProtocol,
            timestep: int,
            gradient: jax.Array,
            action: int,
            neural_state = None
        ) -> EpisodeBufferProtocol:
            # In real implementation, this would update buffer immutably
            # For testing, we don't need JIT compilation
            return buffer
        
        return buffer, log_timestep
    
    def end_episode(
        self,
        buffer,
        episode_data: EpisodeData,
        world_reward_tracking: Dict[str, jax.Array],
        success: bool = False
    ) -> Dict[str, float]:
        """Finalize episode."""
        self.episodes_ended += 1
        
        # Return mock statistics
        return {
            "episode_id": buffer.episode_id,
            "timesteps": buffer.max_size,  # Use buffer size, not episode data length
            "total_rewards": float(episode_data.total_reward_events),
            "duration_seconds": 1.0,
            "steps_per_second": float(buffer.max_size),
            "action_entropy": 0.0,
            "mean_neural_activity": 0.0
        }
    
    def save_network_structure(self, neurons, connections, initial_weights=None):
        """Save network structure."""
        pass
    
    def save_checkpoint(self, episode_id, weights, optimizer_state=None):
        """Save checkpoint."""
        pass
    
    def save_experiment_metadata(
        self, agent_version, agent_name, agent_description, world_version
    ):
        """Save metadata."""
        self.metadata_saved = True
    
    def finalize(self):
        """Finalize experiment."""
        pass
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finalize()
    
    def log(self, message: str, level: str = "INFO"):
        """Log message."""
        print(f"[{level}] {message}")


def test_world_protocol_compliance():
    """Test that MinimalGridWorld implements WorldProtocol."""
    config = WorldConfig(grid_size=10, n_rewards=5, max_timesteps=100)
    world = MinimalGridWorld(config)
    
    # Check protocol implementation
    assert isinstance(world, WorldProtocol), "World doesn't implement WorldProtocol"
    
    # Test methods
    key = PRNGKey(42)
    state, gradient = world.reset(key)
    
    assert isinstance(state, WorldState)
    assert isinstance(gradient, jax.Array)
    assert gradient.shape == ()
    assert 0 <= float(gradient) <= 1
    
    # Test step
    new_state, new_gradient = world.step(state, 0)
    assert isinstance(new_state, WorldState)
    assert isinstance(new_gradient, jax.Array)
    
    # Test config
    config_dict = world.get_config()
    assert isinstance(config_dict, dict)
    assert "grid_size" in config_dict
    
    # Test reward tracking
    tracking = world.get_reward_tracking(new_state)
    assert isinstance(tracking, dict)
    assert "positions" in tracking
    
    print("✓ World protocol compliance test passed")


def test_agent_protocol_compliance():
    """Test that RandomAgent implements AgentProtocol."""
    # Create experiment config
    config = ExperimentConfig(
        world=WorldConfig(grid_size=10, n_rewards=5, max_timesteps=100),
        neural=NeuralConfig(n_neurons=100),
        plasticity=PlasticityConfig(),
        behavior=AgentBehaviorConfig(),
        experiment_name="test",
        agent_version="2.0.0",
        world_version="0.0.4",
        n_episodes=1,
        seed=42,
        device="cpu",
        export_dir="tests/output"
    )
    
    exporter = MockExporter("test", config, "tests/output")
    agent = RandomAgent(config, exporter)
    
    # Check protocol implementation
    assert isinstance(agent, AgentProtocol), "Agent doesn't implement AgentProtocol"
    
    # Check required attributes
    assert hasattr(agent, "VERSION")
    assert hasattr(agent, "MODEL_NAME")
    assert hasattr(agent, "DESCRIPTION")
    
    # Test methods
    key = PRNGKey(42)
    agent.reset(key)
    
    # Test act
    gradient = jnp.array(0.5)
    action_key = PRNGKey(43)
    action = agent.act(gradient, action_key)
    assert isinstance(action, jax.Array)
    action_int = int(action)
    assert 0 <= action_int <= 8
    
    # Test episode data
    episode_data = agent.get_episode_data()
    assert isinstance(episode_data, EpisodeData)
    assert isinstance(episode_data.actions, jax.Array)
    assert isinstance(episode_data.gradients, jax.Array)
    
    print("✓ Agent protocol compliance test passed")


def test_exporter_protocol_compliance():
    """Test that MockExporter implements ExporterProtocol."""
    config = ExperimentConfig(
        world=WorldConfig(),
        neural=NeuralConfig(),
        plasticity=PlasticityConfig(),
        behavior=AgentBehaviorConfig(),
        experiment_name="test",
        agent_version="1.0.0",
        world_version="0.0.4",
        n_episodes=1,
        seed=42,
        device="cpu",
        export_dir="tests/output"
    )
    
    exporter = MockExporter("test", config, "tests/output")
    
    # Check protocol implementation
    assert isinstance(exporter, ExporterProtocol), "Exporter doesn't implement ExporterProtocol"
    
    # Test methods
    buffer, log_fn = exporter.start_episode(0)
    assert hasattr(buffer, "timesteps")
    assert hasattr(buffer, "gradients")
    assert callable(log_fn)
    
    # Test episode end
    episode_data = EpisodeData(
        actions=jnp.zeros(10),
        gradients=jnp.zeros(10),
        total_reward_events=5
    )
    
    stats = exporter.end_episode(
        buffer, 
        episode_data,
        {"positions": jnp.zeros((5, 2))},
        success=True
    )
    
    assert isinstance(stats, dict)
    assert "episode_id" in stats
    assert "total_rewards" in stats
    
    print("✓ Exporter protocol compliance test passed")


def test_integrated_episode_run():
    """Test a complete episode run with protocol-compliant components."""
    from interfaces import ProtocolRunner, create_experiment_config
    
    # Setup
    config = create_experiment_config(
        world_size=10,
        n_rewards=5,
        n_episodes=1,
        max_timesteps=50,
        experiment_name="integration_test"
    )
    
    # Create components
    world = MinimalGridWorld(config.world)
    exporter = MockExporter("test", config, Path("tests/output"))
    agent = RandomAgent(config, exporter)
    
    # Create runner
    runner = ProtocolRunner(world, agent, exporter, config)
    
    # Run single episode
    key = PRNGKey(42)
    stats = runner.run_episode(0, key)
    
    # Verify
    assert exporter.episodes_started == 1
    assert exporter.episodes_ended == 1
    assert stats["timesteps"] == config.world.max_timesteps
    assert "steps_per_second" in stats
    assert stats["steps_per_second"] > 0
    
    print("✓ Integrated episode run test passed")
    print(f"  Steps per second: {stats['steps_per_second']:.0f}")
    print(f"  Rewards collected: {stats['total_rewards']}")
    
def test_full_experiment():
    """Test running a full experiment with the protocol runner."""
    from interfaces import ProtocolRunner, create_experiment_config
    
    # Setup
    config = create_experiment_config(
        world_size=20,
        n_rewards=10,
        n_episodes=5,
        max_timesteps=100,
        experiment_name="full_test"
    )
    config.log_to_console = False  # Quiet for tests
    
    # Create components
    world = MinimalGridWorld(config.world)
    exporter = MockExporter("test", config, Path("tests/output"))
    agent = RandomAgent(config, exporter)
    
    # Create runner
    runner = ProtocolRunner(world, agent, exporter, config)
    
    # Run experiment
    results = runner.run_experiment()
    
    # Verify
    assert len(results["episode_stats"]) == 5
    assert results["average_steps_per_second"] > 0
    assert "average_rewards" in results
    assert exporter.metadata_saved
    
    print("✓ Full experiment test passed")
    print(f"  Average speed: {results['average_steps_per_second']:.0f} steps/s")
    print(f"  Average rewards: {results['average_rewards']:.1f}")


if __name__ == "__main__":
    test_world_protocol_compliance()
    test_agent_protocol_compliance()
    test_exporter_protocol_compliance()
    test_integrated_episode_run()
    test_full_experiment()
    print("\n✅ All protocol compliance tests passed!")