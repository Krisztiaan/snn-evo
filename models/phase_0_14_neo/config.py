# keywords: [neo config, phase 0.14, modular configuration]
"""Configuration for Phase 0.14 Neo agent."""

from typing import List, Dict, Any, NamedTuple, Optional

from interfaces import WorldConfig


class NetworkConfig(NamedTuple):
    """Neural network architecture configuration."""
    # Architecture
    num_sensory: int = 32
    num_processing: int = 192
    num_readout: int = 16
    num_input_channels: int = 16
    
    # Connectivity
    excitatory_ratio: float = 0.8
    p_input_sensory: float = 0.8
    p_sensory_processing: float = 0.3
    p_processing_readout: float = 0.5
    p_ee: float = 0.15
    p_ei: float = 0.4
    p_ie: float = 0.35
    p_ii: float = 0.1
    
    # Which connections are plastic
    learn_input: bool = True
    learn_recurrent: bool = True
    learn_readout: bool = True


class DynamicsConfig(NamedTuple):
    """Neural dynamics configuration."""
    # Membrane dynamics
    tau_v: float = 20.0
    v_rest: float = -70.0
    v_threshold: float = -50.0
    v_reset: float = -75.0
    refractory_time: float = 2.0
    
    # Synaptic dynamics
    tau_syn_e: float = 5.0
    tau_syn_i: float = 10.0
    
    # Input
    baseline_current: float = 3.0
    noise_scale: float = 1.0
    
    # Motor output
    motor_tau: float = 100.0
    initial_temperature: float = 2.0
    final_temperature: float = 0.3
    temperature_decay: float = 0.95


class InputConfig(NamedTuple):
    """Input encoding configuration."""
    input_gain: float = 15.0
    input_tuning_width: float = 0.15
    input_noise: float = 0.05


class LearningRulesConfig(NamedTuple):
    """Learning rules configuration."""
    # Enabled rules (in order of application)
    enabled_rules: List[str] = [
        "stdp",
        "eligibility_trace",
        "dopamine_modulation", 
        "homeostatic",
        "threshold_adaptation",
        "dale_principle",
        "weight_bounds",
        "weight_decay"
    ]
    
    # Rule-specific parameters
    rule_params: Dict[str, Dict[str, Any]] = {
        "stdp": {
            "a_plus": 0.02,
            "a_minus": 0.021,
            "tau_plus": 20.0,
            "tau_minus": 20.0,
        },
        "eligibility_trace": {
            "tau_eligibility": 2000.0,
            "trace_clip": 1.0,
            "accumulate": True,
        },
        "dopamine_modulation": {
            "baseline_dopamine": 0.2,
            "tau_dopamine": 200.0,
            "reward_scale": 5.0,
            "gradient_scale": 0.3,
        },
        "homeostatic": {
            "tau": 10000.0,
            "target_rate": 5.0,
            "learning_rate": 0.001,
            "multiplicative": True,
        },
        "threshold_adaptation": {
            "tau": 10000.0,
            "target_rate": 5.0,
            "adapt_rate": 0.0001,
            "max_adaptation": 5.0,
        },
        "dale_principle": {
            "strict": True,
        },
        "weight_bounds": {
            "w_min": 0.0,
            "w_max": 5.0,
            "soft_bounds": True,
            "bound_scale": 0.1,
        },
        "weight_decay": {
            "decay_rate": 0.00001,
            "target_weight": 0.0,
        }
    }
    
    # Global learning parameters
    base_learning_rate: float = 0.1
    learning_rate_decay: float = 0.98
    min_learning_rate: float = 0.01


class ExperimentConfig(NamedTuple):
    """Experiment configuration."""
    n_episodes: int = 50
    seed: int = 42
    export_dir: str = "experiments/phase_0_14_neo"
    log_weight_changes: bool = False
    verbose: bool = True


class NeoConfig(NamedTuple):
    """Complete configuration for Neo agent."""
    world_config: WorldConfig = WorldConfig()
    network_config: NetworkConfig = NetworkConfig()
    dynamics_config: DynamicsConfig = DynamicsConfig()
    input_config: InputConfig = InputConfig()
    learning_rules_config: LearningRulesConfig = LearningRulesConfig()
    exp_config: ExperimentConfig = ExperimentConfig()
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "NeoConfig":
        """Create config from dictionary."""
        return cls(
            world_config=WorldConfig(**config_dict.get("world", {})),
            network_config=NetworkConfig(**config_dict.get("network", {})),
            dynamics_config=DynamicsConfig(**config_dict.get("dynamics", {})),
            input_config=InputConfig(**config_dict.get("input", {})),
            learning_rules_config=LearningRulesConfig(**config_dict.get("learning_rules", {})),
            exp_config=ExperimentConfig(**config_dict.get("experiment", {}))
        )