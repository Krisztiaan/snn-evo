# models/phase_0_12/config.py
# keywords: [snn config, phase 0.12, fixed learning, exploration, credit assignment]
"""
Phase 0.12 Configuration: Fixed learning dynamics for actual progress

Key fixes over 0.11:
1. Proper learning rate (100x increase)
2. Stronger STDP amplitudes (4x increase)
3. Higher baseline current for proper firing rates
4. More plastic connections (input and output)
5. Reduced gradient reward to prevent overshadowing
6. Exploration mechanisms (temperature annealing)
7. Better credit assignment (longer traces, reward amplification)
8. Success detection and learning boost
"""

from typing import NamedTuple

from world.simple_grid_0001.types import WorldConfig  # Using standard config


class NetworkParams(NamedTuple):
    """Network parameters fixing learning issues while maintaining performance."""

    # === ARCHITECTURE (unchanged from 0.11) ===
    NUM_SENSORY: int = 32  # Receive population-coded input
    NUM_PROCESSING: int = 192  # Main processing layer
    NUM_READOUT: int = 16  # Motor command generation

    # === CONNECTIVITY (unchanged) ===
    EXCITATORY_RATIO: float = 0.8  # 80% excitatory neurons

    # Connection probabilities
    P_INPUT_SENSORY: float = 0.8  # Input → Sensory
    P_SENSORY_PROCESSING: float = 0.3  # Sensory → Processing
    P_PROCESSING_READOUT: float = 0.5  # Processing → Readout

    # Processing layer connectivity (E/I specific)
    P_EE: float = 0.15  # E→E base probability
    P_EI: float = 0.4  # E→I (stronger to control activity)
    P_IE: float = 0.35  # I→E (feedback inhibition)
    P_II: float = 0.1  # I→I (sparse)

    # Spatial connectivity
    P_PROC_PROC_LOCAL: float = 0.25  # Higher for nearby neurons
    P_PROC_PROC_DIST: float = 0.05  # Lower for distant neurons
    LOCAL_RADIUS: float = 0.15  # Fraction of layer considered "local"

    # === DYNAMICS ===
    # Time constants (all in ms)
    TAU_V: float = 20.0  # Membrane potential
    TAU_SYN_E: float = 5.0  # Excitatory synapses
    TAU_SYN_I: float = 10.0  # Inhibitory synapses (slower)
    TAU_TRACE_FAST: float = 20.0  # Fast synaptic trace (classic STDP)
    TAU_TRACE_SLOW: float = 100.0  # Slow trace (heterosynaptic)
    TAU_ELIGIBILITY: float = 2000.0  # LONGER eligibility trace (was 1000)
    TAU_DOPAMINE: float = 200.0  # Dopamine dynamics
    TAU_VALUE: float = 5000.0  # Value estimate update

    # Neuron parameters
    V_REST: float = -70.0  # Resting potential
    V_THRESHOLD: float = -50.0  # Spike threshold
    V_RESET: float = -75.0  # Reset after spike
    REFRACTORY_TIME: float = 2.0  # Refractory period (ms)

    # Currents and noise - FIXED
    BASELINE_CURRENT: float = 3.0  # INCREASED from 0.5 for proper activity
    NOISE_SCALE: float = 1.0  # Moderate noise (was 0.5)

    # === HOMEOSTASIS ===
    TARGET_RATE_HZ: float = 5.0  # Target firing rate
    HOMEOSTATIC_TAU: float = 10000.0  # Slow adaptation
    THRESHOLD_ADAPT_RATE: float = 0.0001
    MAX_THRESHOLD_ADAPT: float = 5.0

    # === LEARNING (Three-factor rule) - FIXED ===
    # STDP parameters - MUCH STRONGER
    STDP_A_PLUS: float = 0.02  # LTP amplitude (was 0.005)
    STDP_A_MINUS: float = 0.021  # LTD amplitude (slightly stronger)

    # Neuromodulation - BALANCED
    BASELINE_DOPAMINE: float = 0.2  # Baseline DA level
    GRADIENT_REWARD_SCALE: float = 0.3  # REDUCED from 2.0 to not overwhelm
    ACTUAL_REWARD_SCALE: float = 5.0  # NEW: Amplify actual rewards

    # Learning rates - PROPER SCALE
    BASE_LEARNING_RATE: float = 0.1  # INCREASED from 0.001
    LEARNING_RATE_DECAY: float = 0.98  # Slower decay (was 0.95)
    MIN_LEARNING_RATE: float = 0.01  # Higher floor (was 0.0001)

    # Reward processing
    REWARD_PREDICTION_RATE: float = 0.05  # Faster value learning
    REWARD_DISCOUNT: float = 0.9  # Some temporal discounting

    # Regularization
    WEIGHT_DECAY: float = 0.00001  # L2 regularization
    MAX_WEIGHT_SCALE: float = 5.0  # Soft weight bounds

    # === MOTOR CONTROL & EXPLORATION ===
    MOTOR_TAU: float = 100.0  # Motor command integration

    # Exploration via temperature annealing
    INITIAL_ACTION_TEMPERATURE: float = 2.0  # HIGH for exploration
    FINAL_ACTION_TEMPERATURE: float = 0.3  # LOW for exploitation
    TEMPERATURE_DECAY: float = 0.95  # Per episode decay

    # === INPUT ENCODING ===
    NUM_INPUT_CHANNELS: int = 16  # Population code channels
    INPUT_GAIN: float = 15.0  # Moderate input strength (was 10.0)
    INPUT_TUNING_WIDTH: float = 0.15  # Width of tuning curves

    # === INITIALIZATION ===
    # Initial weight distributions
    W_INPUT_SENSORY: float = 2.5  # Strong but not overwhelming
    W_SENSORY_PROC: float = 1.2  # Feedforward
    W_PROC_PROC_E: float = 0.4  # E→E recurrent
    W_PROC_READOUT: float = 1.5  # Output projection

    # E/I specific weights
    W_EI: float = 0.8  # E→I connections
    W_IE: float = -1.5  # I→E connections (inhibitory)
    W_II: float = -0.4  # I→I connections

    W_INIT_SCALE: float = 0.2  # Weight initialization variance

    # === MULTI-EPISODE LEARNING ===
    # Weight persistence and momentum
    WEIGHT_MOMENTUM_DECAY: float = 0.9  # Momentum decay
    WEIGHT_CONSOLIDATION: bool = True  # Enable consolidation
    CONSOLIDATION_INTERVAL: int = 5  # Episodes between consolidations
    WEIGHT_CONSOLIDATION_RATE: float = 0.1  # Soft scaling rate
    WEIGHT_CONSOLIDATION_TARGET: float = 1.0  # Target mean weight

    # === LEARNING CONTROL - MORE PLASTICITY ===
    # Plasticity flags
    # ENABLED for better input processing
    LEARN_INPUT_CONNECTIONS: bool = True
    LEARN_SENSORY_PROCESSING: bool = False  # Keep feedforward fixed
    LEARN_PROCESSING_RECURRENT: bool = True  # Main learning site
    LEARN_PROCESSING_READOUT: bool = True  # ENABLED for action learning

    # === SUCCESS DETECTION ===
    # Boost learning when rewards are collected
    REWARD_BOOST_FACTOR: float = 3.0  # Multiply learning rate on reward
    REWARD_BOOST_DURATION: float = 500.0  # How long boost lasts (ms)
    SUCCESS_THRESHOLD: int = 3  # Rewards needed for "success"


class ExperimentConfig(NamedTuple):
    """Experiment configuration."""

    n_episodes: int = 20  # More episodes to see learning
    seed: int = 42
    export_dir: str = "experiments/phase_0_12"
    enable_export: bool = True

    # Debugging/monitoring flags
    monitor_rates: bool = True  # Track firing rates
    monitor_weights: bool = True  # Track weight distributions
    check_stability: bool = True  # Warn on instabilities
    verbose: bool = True  # Print detailed progress


class SnnAgentConfig(NamedTuple):
    """Master configuration combining all components."""

    world_config: WorldConfig = WorldConfig()
    network_params: NetworkParams = NetworkParams()
    exp_config: ExperimentConfig = ExperimentConfig()

