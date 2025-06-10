# keywords: [experiment runner, type safe, performance, interfaces]
"""type-safe experiment runner using protocol interfaces."""

import time
from pathlib import Path

import jax.random as jrandom

from interfaces import (
    AgentProtocol,
    ExperimentConfig,
    ExporterProtocol,
    WorldProtocol,
)


class ExperimentRunner:
    """Runs experiments with type-safe interfaces and performance monitoring."""

    def __init__(
        self,
        world_class: type[WorldProtocol],
        agent_class: type[AgentProtocol],
        exporter_class: type[ExporterProtocol],
        config: ExperimentConfig,
    ):
        """Initialize runner with component classes and configuration.

        Args:
            world_class: World implementation class
            agent_class: Agent implementation class
            exporter_class: Exporter implementation class
            config: Experiment configuration
        """
        self.world_class = world_class
        self.agent_class = agent_class
        self.exporter_class = exporter_class
        self.config = config

    def run(self) -> None:
        """Run the complete experiment."""
        # Initialize random key
        key = jrandom.PRNGKey(self.config.seed)

        # Create output directory
        output_dir = Path(self.config.export_dir) / self.config.experiment_name

        # Initialize components
        world = self.world_class(self.config.world)

        with self.exporter_class(
            self.config.experiment_name,
            self.config,
            output_dir,
            compression="gzip",
            log_to_console=True,
        ) as exporter:
            # Save experiment metadata
            agent = self.agent_class(self.config, exporter)
            exporter.save_experiment_metadata(
                agent_version=agent.VERSION,
                agent_name=agent.MODEL_NAME,
                agent_description=agent.DESCRIPTION,
                world_version=world.get_config()["version"],
            )

            # Save network structure if applicable
            if hasattr(agent, "get_network_structure"):
                structure = agent.get_network_structure()
                exporter.save_network_structure(**structure)

            # Run episodes
            for episode_id in range(self.config.n_episodes):
                key, episode_key = jrandom.split(key)
                self._run_episode(world, agent, exporter, episode_id, episode_key)

                # Save checkpoint periodically
                if (episode_id + 1) % self.config.checkpoint_every_n_episodes == 0:
                    if hasattr(agent, "get_weights"):
                        weights = agent.get_weights()
                        if weights is not None:
                            exporter.save_checkpoint(episode_id, weights)

    def _run_episode(
        self,
        world: WorldProtocol,
        agent: AgentProtocol,
        exporter: ExporterProtocol,
        episode_id: int,
        key: jrandom.PRNGKey,
    ) -> None:
        """Run a single episode with performance monitoring."""
        # Split keys
        key, world_key, agent_key = jrandom.split(key, 3)

        # Reset world and agent
        initial_gradient = world.reset(world_key)
        agent.reset(agent_key)

        # Start episode
        buffer, log_fn = exporter.start_episode(episode_id)

        # Episode timing
        start_time = time.perf_counter()

        # Run episode
        gradient = initial_gradient
        timestep = 0

        # Main episode loop - optimized for performance
        while timestep < self.config.world.max_timesteps:
            # Agent action selection
            key, action_key = jrandom.split(key)
            action = agent.act(gradient, action_key)

            # World step
            gradient = world.step(action)

            # Log timestep (JIT-compiled, no I/O)
            neural_state = agent.get_neural_state()
            buffer = log_fn(buffer, timestep, gradient, action, neural_state)

            timestep += 1

            # Optional: Early termination based on world state
            # (not implemented in current minimal interface)

        # Get episode data
        episode_data = agent.get_episode_data()
        reward_tracking = world.get_reward_tracking()

        # Finalize episode (I/O happens here)
        stats = exporter.end_episode(buffer, episode_data, reward_tracking, success=True)

        # Log performance
        if self.config.log_every_n_steps > 0:
            print(
                f"Episode {episode_id}: {stats['steps_per_second']:.0f} steps/s, "
                f"{stats['total_rewards']:.0f} rewards"
            )


def validate_implementation(
    world_class: type[WorldProtocol],
    agent_class: type[AgentProtocol],
    exporter_class: type[ExporterProtocol],
) -> None:
    """Validate that implementations satisfy protocol requirements."""
    # Check class attributes
    assert hasattr(agent_class, "VERSION"), "Agent must have VERSION attribute"
    assert hasattr(agent_class, "MODEL_NAME"), "Agent must have MODEL_NAME attribute"
    assert hasattr(agent_class, "DESCRIPTION"), "Agent must have DESCRIPTION attribute"
    assert hasattr(exporter_class, "VERSION"), "Exporter must have VERSION attribute"

    # Check methods exist
    assert hasattr(world_class, "reset"), "World must implement reset()"
    assert hasattr(world_class, "step"), "World must implement step()"
    assert hasattr(world_class, "get_config"), "World must implement get_config()"
    assert hasattr(world_class, "get_reward_tracking"), "World must implement get_reward_tracking()"

    assert hasattr(agent_class, "reset"), "Agent must implement reset()"
    assert hasattr(agent_class, "act"), "Agent must implement act()"
    assert hasattr(agent_class, "get_episode_data"), "Agent must implement get_episode_data()"

    print("✓ All implementations satisfy protocol requirements")
