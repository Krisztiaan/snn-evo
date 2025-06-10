# keywords: [config protocols, type safety, neural config, learning rules]
"""Configuration protocols for type-safe experiment setup."""

from typing import NamedTuple, Dict, Optional, List
from dataclasses import dataclass, field

class WorldConfig(NamedTuple):
    """Configuration for world initialization."""
    grid_size: int = 100
    n_rewards: int = 300
    max_timesteps: int = 50000

@dataclass
class NeuralConfig:
    """Neural network architecture configuration.
    
    Automatically calculates excitatory/inhibitory counts from n_neurons and ratio.
    """
    n_neurons: int = 1000
    excitatory_ratio: float = 0.8  # Ratio of excitatory neurons (e.g., 0.8 = 80%)

    n_sensory: int = 100
    n_motor: int = 9  # Standardized to 9 actions

    # Connectivity
    connection_probability: float = 0.1
    w_initial_scale: float = 0.01

    # Neuron dynamics
    tau_membrane: float = 20.0
    tau_syn_e: float = 5.0
    tau_syn_i: float = 10.0
    v_threshold: float = 1.0
    v_reset: float = 0.0
    refractory_period: int = 5

    # These fields will be calculated automatically.
    # `init=False` means we don't provide them when creating a NeuralConfig.
    n_excitatory: int = field(init=False)
    n_inhibitory: int = field(init=False)

    def __post_init__(self):
        """This function runs automatically after the object is created."""
        if not 0.0 <= self.excitatory_ratio <= 1.0:
            raise ValueError("excitatory_ratio must be between 0.0 and 1.0")

        # Calculate the number of excitatory neurons
        self.n_excitatory = int(self.n_neurons * self.excitatory_ratio)
        # The rest are inhibitory
        self.n_inhibitory = self.n_neurons - self.n_excitatory

class PlasticityConfig(NamedTuple):
    """Synaptic plasticity configuration."""
    # Learning rules toggles
    enable_stdp: bool = True
    enable_homeostasis: bool = True
    enable_reward_modulation: bool = True
    enable_metaplasticity: bool = False
    
    # STDP parameters
    stdp_lr_potentiation: float = 0.001
    stdp_lr_depression: float = 0.0005
    tau_trace_fast: float = 20.0
    tau_trace_slow: float = 100.0
    
    # Reward modulation
    dopamine_baseline: float = 0.1
    dopamine_reward_boost: float = 1.0
    tau_dopamine: float = 200.0
    tau_eligibility: float = 1000.0
    
    # Homeostasis
    target_firing_rate: float = 0.05
    homeostasis_lr: float = 0.0001
    tau_firing_rate: float = 10000.0
    
    # Weight constraints
    w_min: float = 0.0
    w_max: float = 1.0
    
class AgentBehaviorConfig(NamedTuple):
    """Agent behavior configuration."""
    # Action selection
    action_integration_steps: int = 5  # Number of internal steps per world step
    action_noise: float = 0.1
    temperature: float = 1.0
    temperature_decay: float = 0.999
    min_temperature: float = 0.1
    
    # Exploration
    epsilon_initial: float = 0.1
    epsilon_decay: float = 0.999
    epsilon_min: float = 0.01

@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    # Component configs
    world: WorldConfig
    neural: NeuralConfig
    plasticity: PlasticityConfig
    behavior: AgentBehaviorConfig
    
    # Experiment metadata
    experiment_name: str
    agent_version: str
    world_version: str
    
    # Runtime settings
    n_episodes: int = 100
    seed: int = 42
    device: str = "cpu"  # or "gpu"
    
    # Logging settings
    export_dir: str = "experiments/"
    log_every_n_steps: int = 1000
    neural_sampling_rate: int = 100
    save_checkpoints: bool = True
    checkpoint_every_n_episodes: int = 10
    flush_at_episode_end: bool = True  # Performance optimization
    log_to_console: bool = True  # Console output
    
    def to_dict(self) -> Dict:
        """Convert to dict for export."""
        # Convert NeuralConfig dataclass to dict
        # Don't include n_excitatory and n_inhibitory as they are calculated fields
        neural_dict = {
            "n_neurons": self.neural.n_neurons,
            "excitatory_ratio": self.neural.excitatory_ratio,
            "n_sensory": self.neural.n_sensory,
            "n_motor": self.neural.n_motor,
            "connection_probability": self.neural.connection_probability,
            "w_initial_scale": self.neural.w_initial_scale,
            "tau_membrane": self.neural.tau_membrane,
            "tau_syn_e": self.neural.tau_syn_e,
            "tau_syn_i": self.neural.tau_syn_i,
            "v_threshold": self.neural.v_threshold,
            "v_reset": self.neural.v_reset,
            "refractory_period": self.neural.refractory_period,
        }
        
        return {
            "world": self.world._asdict(),
            "neural": neural_dict,
            "plasticity": self.plasticity._asdict(),
            "behavior": self.behavior._asdict(),
            "experiment_name": self.experiment_name,
            "agent_version": self.agent_version,
            "world_version": self.world_version,
            "n_episodes": self.n_episodes,
            "seed": self.seed,
            "device": self.device,
            "export_dir": self.export_dir,
            "log_every_n_steps": self.log_every_n_steps,
            "neural_sampling_rate": self.neural_sampling_rate,
            "save_checkpoints": self.save_checkpoints,
            "checkpoint_every_n_episodes": self.checkpoint_every_n_episodes,
            "flush_at_episode_end": self.flush_at_episode_end,
        }