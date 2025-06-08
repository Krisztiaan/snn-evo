# models/phase_0_6/config.py
# keywords: [snn config, agent parameters, network configuration, phase 0.6]
"""Configuration for the Phase 0.6 SNN Agent."""

from typing import NamedTuple

from world.simple_grid_0001 import WorldConfig


class NetworkParams(NamedTuple):
    """Biologically-plausible network parameters for a functional agent."""

    # Architecture
    NUM_NEURONS: int = 256
    NUM_INPUTS: int = 16
    NUM_OUTPUTS: int = 4
    EXCITATORY_RATIO: float = 0.8

    # Neuron Dynamics
    TAU_V: float = 20.0
    V_REST: float = -65.0
    V_THRESHOLD: float = -50.0
    V_RESET: float = -65.0
    REFRACTORY_TIME: float = 2.0
    ## FIX: Increased input scale to ensure neurons can overcome the leak potential and fire.
    ## This is the primary fix for the "silent network" problem.
    INPUT_SCALE: float = 25.0

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

    # Metaplasticity
    TAU_METAPLASTICITY: float = 50000.0

    # Dopamine & RPE
    TAU_ELIGIBILITY: float = 1000.0
    TAU_DOPAMINE: float = 200.0
    BASELINE_DOPAMINE: float = 0.2
    REWARD_PREDICTION_RATE: float = 0.01
    REWARD_DISCOUNT: float = 0.95

    # Action Selection
    MOTOR_TAU: float = 10.0
    ACTION_TEMP: float = 0.5

    # Connectivity
    P_EE: float = 0.1
    P_EI: float = 0.4
    P_IE: float = 0.4
    P_II: float = 0.2
    LOCAL_CONNECTIVITY_SCALE: float = 0.1

    # Weight Initialization
    W_EE_MEAN: float = 0.3
    W_EI_MEAN: float = 0.5
    W_IE_MEAN: float = 0.8
    W_II_MEAN: float = 0.2
    W_STD_SCALE: float = 0.3


class ExperimentConfig(NamedTuple):
    """Configuration for the experiment run."""

    n_episodes: int = 3
    seed: int = 42
    export_dir: str = "experiments/phase_0_6"
    enable_export: bool = True


class SnnAgentConfig(NamedTuple):
    """Master configuration for the SNN agent and experiment."""

    world_config: WorldConfig = WorldConfig()
    network_params: NetworkParams = NetworkParams()
    exp_config: ExperimentConfig = ExperimentConfig()
