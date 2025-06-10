# models/phase_0_13/config.py
# keywords: [snn config, phase 0.13, detailed logging, learning analysis]
"""
Phase 0.13 Configuration: SNN Agent with Detailed Learning Analysis

This configuration is based on Phase 0.12, retaining all learning fixes,
and adds parameters for controlling detailed, step-by-step logging.
"""

from typing import NamedTuple

from world.simple_grid_0003.types import WorldConfig  # Using optimized world config


class NetworkParams(NamedTuple):
    """Network parameters based on Phase 0.12 with added logging controls."""

    # === ARCHITECTURE (unchanged from 0.12) ===
    NUM_SENSORY: int = 32
    NUM_PROCESSING: int = 192
    NUM_READOUT: int = 16

    # === CONNECTIVITY (unchanged) ===
    EXCITATORY_RATIO: float = 0.8
    P_INPUT_SENSORY: float = 0.8
    P_SENSORY_PROCESSING: float = 0.3
    P_PROCESSING_READOUT: float = 0.5
    P_EE: float = 0.15
    P_EI: float = 0.4
    P_IE: float = 0.35
    P_II: float = 0.1
    P_PROC_PROC_LOCAL: float = 0.25
    P_PROC_PROC_DIST: float = 0.05
    LOCAL_RADIUS: float = 0.15

    # === DYNAMICS (unchanged) ===
    TAU_V: float = 20.0
    TAU_SYN_E: float = 5.0
    TAU_SYN_I: float = 10.0
    TAU_TRACE_FAST: float = 20.0
    TAU_TRACE_SLOW: float = 100.0
    TAU_ELIGIBILITY: float = 2000.0
    TAU_DOPAMINE: float = 200.0
    TAU_VALUE: float = 5000.0
    V_REST: float = -70.0
    V_THRESHOLD: float = -50.0
    V_RESET: float = -75.0
    REFRACTORY_TIME: float = 2.0
    BASELINE_CURRENT: float = 3.0
    NOISE_SCALE: float = 1.0

    # === HOMEOSTASIS (unchanged) ===
    TARGET_RATE_HZ: float = 5.0
    HOMEOSTATIC_TAU: float = 10000.0
    THRESHOLD_ADAPT_RATE: float = 0.0001
    MAX_THRESHOLD_ADAPT: float = 5.0

    # === LEARNING (Three-factor rule) (unchanged) ===
    STDP_A_PLUS: float = 0.02
    STDP_A_MINUS: float = 0.021
    BASELINE_DOPAMINE: float = 0.2
    GRADIENT_REWARD_SCALE: float = 0.3
    ACTUAL_REWARD_SCALE: float = 5.0
    BASE_LEARNING_RATE: float = 0.1
    LEARNING_RATE_DECAY: float = 0.98
    MIN_LEARNING_RATE: float = 0.01
    REWARD_PREDICTION_RATE: float = 0.05
    REWARD_DISCOUNT: float = 0.9
    WEIGHT_DECAY: float = 0.00001
    MAX_WEIGHT_SCALE: float = 5.0

    # === MOTOR CONTROL & EXPLORATION (unchanged) ===
    MOTOR_TAU: float = 100.0
    INITIAL_ACTION_TEMPERATURE: float = 2.0
    FINAL_ACTION_TEMPERATURE: float = 0.3
    TEMPERATURE_DECAY: float = 0.95

    # === INPUT ENCODING (unchanged) ===
    NUM_INPUT_CHANNELS: int = 16
    INPUT_GAIN: float = 15.0
    INPUT_TUNING_WIDTH: float = 0.15

    # === INITIALIZATION (unchanged) ===
    W_INPUT_SENSORY: float = 2.5
    W_SENSORY_PROC: float = 1.2
    W_PROC_PROC_E: float = 0.4
    W_PROC_READOUT: float = 1.5
    W_EI: float = 0.8
    W_IE: float = -1.5
    W_II: float = -0.4
    W_INIT_SCALE: float = 0.2

    # === MULTI-EPISODE LEARNING (unchanged) ===
    WEIGHT_MOMENTUM_DECAY: float = 0.9
    WEIGHT_CONSOLIDATION: bool = True
    CONSOLIDATION_INTERVAL: int = 5
    WEIGHT_CONSOLIDATION_RATE: float = 0.1
    WEIGHT_CONSOLIDATION_TARGET: float = 1.0

    # === LEARNING CONTROL (unchanged) ===
    LEARN_INPUT_CONNECTIONS: bool = True
    LEARN_SENSORY_PROCESSING: bool = False
    LEARN_PROCESSING_RECURRENT: bool = True
    LEARN_PROCESSING_READOUT: bool = True

    # === SUCCESS DETECTION (unchanged) ===
    REWARD_BOOST_FACTOR: float = 3.0
    REWARD_BOOST_DURATION: float = 500.0
    SUCCESS_THRESHOLD: int = 3

    # === DETAILED LOGGING (NEW for Phase 0.13) ===
    LOG_WEIGHT_CHANGES: bool = True
    # Probability of logging a given plastic synapse's change per step
    WEIGHT_LOG_PROB: float = 1e-4
    # Max weight changes to log per step to avoid performance issues
    MAX_WEIGHT_LOG_PER_STEP: int = 10


class ExperimentConfig(NamedTuple):
    """Experiment configuration."""

    n_episodes: int = 20
    seed: int = 42
    export_dir: str = "experiments/phase_0_13"
    enable_export: bool = True

    # Debugging/monitoring flags
    monitor_rates: bool = True
    monitor_weights: bool = True
    check_stability: bool = True
    verbose: bool = True


class SnnAgentConfig(NamedTuple):
    """Master configuration combining all components."""

    world_config: WorldConfig = WorldConfig(
        grid_size=20,
        n_rewards=12,  # 3% of 400 cells
        max_timesteps=1000
    )
    network_params: NetworkParams = NetworkParams()
    exp_config: ExperimentConfig = ExperimentConfig()
