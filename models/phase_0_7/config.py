# models/phase_0_7/config.py
# keywords: [snn config, agent parameters, network configuration, phase 0.7]
"""Configuration for the Phase 0.7 SNN Agent."""

from typing import NamedTuple

from world.simple_grid_0001 import WorldConfig


class NetworkParams(NamedTuple):
    """Network parameters with specialized neuron populations."""

    # Architecture
    NUM_MAIN_NEURONS: int = 256
    NUM_INPUT_NEURONS: int = 16  # Encodes gradient via firing rate
    NUM_CLOCK_NEURONS: int = 4  # Provide intrinsic rhythmic drive
    NUM_MOTOR_NEURONS: int = 6  # move_en, move_fwd, move_bwd, turn_en, turn_l, turn_r
    EXCITATORY_RATIO: float = 0.8

    # Neuron Dynamics
    TAU_V: float = 20.0
    V_REST: float = -65.0
    V_THRESHOLD: float = -50.0
    V_RESET: float = -65.0
    REFRACTORY_TIME: float = 2.0

    # Rate-Coding and Clock
    MAX_INPUT_RATE_HZ: float = 100.0  # Max firing rate for input neurons at gradient=1
    CLOCK_RATE_HZ: float = 10.0  # Firing rate of clock neurons

    # Homeostasis
    TARGET_RATE_HZ: float = 5.0
    HOMEOSTATIC_TAU: float = 10000.0
    THRESHOLD_ADAPT_RATE: float = 0.0001

    # Synaptic & Trace Dynamics
    TAU_SYN_E: float = 5.0
    TAU_SYN_I: float = 10.0
    STDP_TRACE_TAU: float = 20.0

    # Learning & Plasticity
    BASE_LEARNING_RATE: float = 0.01
    STDP_A_PLUS: float = 0.01
    STDP_A_MINUS: float = 0.01
    WEIGHT_DECAY: float = 1e-5

    # Dopamine & RPE
    TAU_ELIGIBILITY: float = 1000.0
    TAU_DOPAMINE: float = 200.0
    BASELINE_DOPAMINE: float = 0.2
    REWARD_PREDICTION_RATE: float = 0.01
    REWARD_DISCOUNT: float = 0.95

    # Action Selection
    DECISION_WINDOW: int = 100  # ms, accumulate motor spikes over this window
    ENABLE_THRESHOLD: int = 5  # Min spikes in window to enable move/turn

    # Connectivity (probabilities)
    P_IN_MAIN: float = 0.5
    P_CLOCK_MAIN: float = 0.2
    P_MAIN_MAIN: float = 0.1
    P_MAIN_MOTOR: float = 0.3

    # Weight Initialization
    W_IN_MAIN_MEAN: float = 0.5
    W_CLOCK_MAIN_MEAN: float = 0.2
    W_MAIN_MAIN_MEAN: float = 0.3
    W_MAIN_MOTOR_MEAN: float = 0.4
    W_STD_SCALE: float = 0.3


class ExperimentConfig(NamedTuple):
    """Configuration for the experiment run."""

    n_episodes: int = 3
    seed: int = 42
    export_dir: str = "experiments/phase_0_7"
    enable_export: bool = True


class SnnAgentConfig(NamedTuple):
    """Master configuration for the SNN agent and experiment."""

    world_config: WorldConfig = WorldConfig()
    network_params: NetworkParams = NetworkParams()
    exp_config: ExperimentConfig = ExperimentConfig()
