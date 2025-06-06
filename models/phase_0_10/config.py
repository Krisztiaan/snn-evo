# models/phase_0_10/config.py
# keywords: [snn config, phase 0.10, multi-episode learning, adaptive learning rate]
"""
Phase 0.10 Configuration: Multi-Episode Learning with Weight Persistence

Key additions over phase 0.9:
1. Adaptive learning rate schedule across episodes
2. Weight momentum for smoother learning trajectories
3. Soft state resets between episodes
4. Optional weight consolidation
5. All phase 0.9 improvements retained
"""

from typing import NamedTuple
from world.simple_grid_0001 import WorldConfig


class NetworkParams(NamedTuple):
    """Network parameters synthesizing best practices from all phases."""
    
    # === ARCHITECTURE ===
    # Three logical populations within unified dynamics
    NUM_SENSORY: int = 32      # Receive population-coded input
    NUM_PROCESSING: int = 192   # Main computation layer  
    NUM_READOUT: int = 32       # Project to motor output
    # Total neurons = 256 (matching previous phases for comparison)
    
    EXCITATORY_RATIO: float = 0.8  # Standard 80/20 E/I balance
    
    # === NEURON DYNAMICS ===
    # LIF parameters - well-tested values from phase 0.4
    TAU_V: float = 20.0          # Membrane time constant (ms)
    V_REST: float = -65.0        # Resting potential (mV)
    V_THRESHOLD: float = -50.0   # Spike threshold (mV)
    V_RESET: float = -65.0       # Reset potential (mV)
    REFRACTORY_TIME: float = 2.0 # Absolute refractory period (ms)
    
    # Baseline current to ensure ~5Hz spontaneous firing
    # Calculated: (V_threshold - V_rest) / R ≈ 15mV / 3MΩ = 5pA
    BASELINE_CURRENT: float = 5.0
    NOISE_SCALE: float = 2.0     # Membrane noise (pA)
    
    # === SYNAPTIC DYNAMICS ===
    # Separate E/I dynamics for biological realism
    TAU_SYN_E: float = 5.0       # Fast excitatory (AMPA-like)
    TAU_SYN_I: float = 10.0      # Slower inhibitory (GABA-like)
    
    # Two traces for richer STDP dynamics (missing in 0.5/0.6!)
    TAU_TRACE_FAST: float = 20.0   # Fast trace for basic STDP
    TAU_TRACE_SLOW: float = 100.0  # Slow trace for triplet STDP
    
    # === HOMEOSTASIS ===
    # Simplified from phase 0.4 - just firing rate control
    TARGET_RATE_HZ: float = 5.0     # Target firing rate
    HOMEOSTATIC_TAU: float = 2000.0  # Slow adaptation (10s)
    THRESHOLD_ADAPT_RATE: float = 0.0005  # Conservative adaptation
    MAX_THRESHOLD_ADAPT: float = 10.0    # Prevent runaway adaptation
    
    # === CONNECTIVITY ===
    # Biologically motivated connection probabilities
    # Input pathway
    P_INPUT_SENSORY: float = 0.8   # Dense input (retina → LGN-like)
    
    # Feedforward pathway  
    P_SENSORY_PROCESSING: float = 0.3  # Divergent (LGN → V1-like)
    
    # Recurrent processing
    P_PROC_PROC_LOCAL: float = 0.2    # Local E→E connections
    P_PROC_PROC_DIST: float = 0.05    # Distant E→E connections
    P_EI: float = 0.4                  # E→I connections (dense)
    P_IE: float = 0.4                  # I→E connections (dense)
    P_II: float = 0.2                  # I→I connections (moderate)
    
    # Output pathway
    P_PROCESSING_READOUT: float = 0.2  # Convergent (V1 → motor-like)
    
    LOCAL_RADIUS: float = 0.1  # Fraction of network for "local" connections
    
    # === WEIGHT PARAMETERS ===
    # Mean weights calibrated for stable dynamics
    W_INPUT_SENSORY: float = 2.0      # Strong, reliable input
    W_SENSORY_PROC: float = 0.5       # Divergent, weaker
    W_PROC_PROC_E: float = 0.3        # Moderate recurrent excitation
    W_EI: float = 0.5                 # E→I (feedforward inhibition)
    W_IE: float = 0.8                 # I→E (feedback inhibition)  
    W_II: float = 0.2                 # I→I (disinhibition)
    W_PROC_READOUT: float = 0.4       # Moderate output weights
    
    W_INIT_SCALE: float = 0.3  # Std dev as fraction of mean
    
    # === LEARNING PARAMETERS ===
    # Three-factor learning
    TAU_ELIGIBILITY: float = 150.0    # Eligibility trace decay (1s)
    TAU_DOPAMINE: float = 50.0        # Dopamine decay (200ms)
    BASELINE_DOPAMINE: float = 0.2     # Baseline DA level
    
    # STDP parameters
    STDP_A_PLUS: float = 0.05          # LTP amplitude
    STDP_A_MINUS: float = 0.05         # LTD amplitude  
    
    # Learning rates and bounds
    BASE_LEARNING_RATE: float = 0.2   # Conservative learning rate
    WEIGHT_DECAY: float = 1e-4         # Prevent weight explosion
    MAX_WEIGHT_SCALE: float = 2.0      # Soft bounds via tanh
    
    # Reward prediction  
    REWARD_PREDICTION_RATE: float = 0.1  # TD learning rate
    REWARD_DISCOUNT: float = 0.95          # Future reward discount
    
    # Gradient-based reward
    GRADIENT_REWARD_SCALE: float = 0.3  # Scale factor for gradient-proportional dopamine
    
    # === INPUT/OUTPUT ===
    # Population coding parameters
    NUM_INPUT_CHANNELS: int = 16   # Gradient encoding channels
    INPUT_TUNING_WIDTH: float = 0.15  # Gaussian tuning curve width
    INPUT_GAIN: float = 20.0        # Scales gradient to current
    
    # Motor output
    MOTOR_TAU: float = 50.0         # Motor command integration
    ACTION_TEMPERATURE: float = 1.0  # Softmax temperature
    
    # === LEARNING CONTROL ===
    # Which connections are plastic? 
    LEARN_PROCESSING_RECURRENT: bool = True   # Main learning site
    LEARN_PROCESSING_READOUT: bool = True     # Output learning
    LEARN_INPUT_CONNECTIONS: bool = True 
    
    # === MULTI-EPISODE LEARNING ===
    # Adaptive learning rate
    LEARNING_RATE_DECAY: float = 0.95     # Learning rate decay per episode
    MIN_LEARNING_RATE: float = 0.01       # Minimum learning rate floor
    
    # Weight momentum
    WEIGHT_MOMENTUM_DECAY: float = 0.9    # Momentum coefficient (0-1)
    
    # Weight consolidation
    WEIGHT_CONSOLIDATION: bool = True     # Enable weight consolidation
    CONSOLIDATION_INTERVAL: int = 5       # Consolidate every N episodes
    WEIGHT_CONSOLIDATION_TARGET: float = 0.5  # Target mean weight magnitude
    WEIGHT_CONSOLIDATION_RATE: float = 0.1    # How fast to consolidate (0-1) 
    

class ExperimentConfig(NamedTuple):
    """Experiment configuration."""
    n_episodes: int = 10  # More episodes to observe multi-episode learning
    seed: int = 42
    export_dir: str = "experiments/phase_0_10"
    enable_export: bool = True
    
    # Debugging/monitoring flags
    monitor_rates: bool = True      # Track firing rates
    monitor_weights: bool = True     # Track weight distributions
    check_stability: bool = True     # Warn on instabilities
    

class SnnAgentConfig(NamedTuple):
    """Master configuration combining all components."""
    world_config: WorldConfig = WorldConfig()
    network_params: NetworkParams = NetworkParams()
    exp_config: ExperimentConfig = ExperimentConfig()
