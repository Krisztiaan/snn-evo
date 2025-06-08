# models/phase_0_11/config.py
# keywords: [snn config, phase 0.11, optimized, precomputed constants]
"""
Phase 0.11 Configuration: Optimized version with performance improvements

Key improvements over 0.10:
1. Uses optimized grid world (0002)
2. Precomputed exponential constants
3. Optimized spike handling
4. Better memory management
"""

from typing import NamedTuple

from world.simple_grid_0001.types import WorldConfig


class NetworkParams(NamedTuple):
    """Network parameters with optimizations from phase 0.11."""

    # === ARCHITECTURE ===
    # Three logical populations within unified dynamics
    NUM_SENSORY: int = 32  # Receive population-coded input
    NUM_PROCESSING: int = 192  # Main processing layer
    NUM_READOUT: int = 16  # Motor command generation

    # === CONNECTIVITY ===
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
    TAU_ELIGIBILITY: float = 1000.0  # Eligibility trace
    TAU_DOPAMINE: float = 200.0  # Dopamine dynamics
    TAU_VALUE: float = 5000.0  # Value estimate update

    # Neuron parameters
    V_REST: float = -70.0  # Resting potential
    V_THRESHOLD: float = -50.0  # Spike threshold
    V_RESET: float = -75.0  # Reset after spike
    REFRACTORY_TIME: float = 2.0  # Refractory period (ms)

    # Currents and noise
    BASELINE_CURRENT: float = 0.5  # Small positive bias
    NOISE_SCALE: float = 0.5  # Membrane noise amplitude

    # === HOMEOSTASIS ===
    TARGET_RATE_HZ: float = 5.0  # Target firing rate
    HOMEOSTATIC_TAU: float = 10000.0  # Slow adaptation
    THRESHOLD_ADAPT_RATE: float = 0.0001
    MAX_THRESHOLD_ADAPT: float = 5.0

    # === LEARNING (Three-factor rule) ===
    # STDP parameters
    STDP_A_PLUS: float = 0.005  # LTP amplitude
    STDP_A_MINUS: float = 0.00525  # LTD amplitude (slightly stronger)

    # Neuromodulation
    BASELINE_DOPAMINE: float = 0.2  # Baseline DA level
    GRADIENT_REWARD_SCALE: float = 2.0  # Scale gradient input as reward

    # Learning rates
    BASE_LEARNING_RATE: float = 0.001  # Initial learning rate
    LEARNING_RATE_DECAY: float = 0.95  # Decay per episode
    MIN_LEARNING_RATE: float = 0.0001  # Minimum learning rate

    # Reward processing
    REWARD_PREDICTION_RATE: float = 0.01
    REWARD_DISCOUNT: float = 0.0  # No temporal discounting for now

    # Regularization
    WEIGHT_DECAY: float = 0.00001  # L2 regularization
    MAX_WEIGHT_SCALE: float = 5.0  # Soft weight bounds

    # === MOTOR CONTROL ===
    MOTOR_TAU: float = 100.0  # Motor command integration
    ACTION_TEMPERATURE: float = 0.5  # Softmax temperature

    # === INPUT ENCODING ===
    NUM_INPUT_CHANNELS: int = 16  # Population code channels
    INPUT_GAIN: float = 10.0  # Scale input to appropriate range
    INPUT_TUNING_WIDTH: float = 0.15  # Width of tuning curves

    # === INITIALIZATION ===
    # Initial weight distributions
    W_INPUT_SENSORY: float = 3.0  # Strong input drive
    W_SENSORY_PROC: float = 1.5  # Feedforward
    W_PROC_PROC_E: float = 0.5  # E→E recurrent
    W_PROC_READOUT: float = 2.0  # Output projection

    # E/I specific weights
    W_EI: float = 1.0  # E→I connections
    W_IE: float = -2.0  # I→E connections (inhibitory)
    W_II: float = -0.5  # I→I connections

    W_INIT_SCALE: float = 0.1  # Weight initialization variance

    # === MULTI-EPISODE LEARNING ===
    # Weight persistence and momentum
    WEIGHT_MOMENTUM_DECAY: float = 0.9  # Momentum decay
    WEIGHT_CONSOLIDATION: bool = True  # Enable consolidation
    CONSOLIDATION_INTERVAL: int = 5  # Episodes between consolidations
    WEIGHT_CONSOLIDATION_RATE: float = 0.1  # Soft scaling rate
    WEIGHT_CONSOLIDATION_TARGET: float = 1.0  # Target mean weight

    # === LEARNING CONTROL ===
    # Plasticity flags
    LEARN_INPUT_CONNECTIONS: bool = False  # Input weights fixed
    LEARN_SENSORY_PROCESSING: bool = False  # Feedforward fixed
    LEARN_PROCESSING_RECURRENT: bool = True  # Main learning site
    LEARN_PROCESSING_READOUT: bool = False  # Output weights fixed


class ExperimentConfig(NamedTuple):
    """Experiment configuration."""

    n_episodes: int = 10  # More episodes to observe multi-episode learning
    seed: int = 42
    export_dir: str = "experiments/phase_0_11"
    enable_export: bool = True

    # Debugging/monitoring flags
    monitor_rates: bool = True  # Track firing rates
    monitor_weights: bool = True  # Track weight distributions
    check_stability: bool = True  # Warn on instabilities


class SnnAgentConfig(NamedTuple):
    """Master configuration combining all components."""

    world_config: WorldConfig = WorldConfig()
    network_params: NetworkParams = NetworkParams()
    exp_config: ExperimentConfig = ExperimentConfig()
