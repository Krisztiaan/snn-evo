# models/phase_0_12/agent.py
# keywords: [snn agent, phase 0.12, fixed learning, exploration, credit assignment, reward boost]
"""
Phase 0.12 SNN Agent: Fixed learning dynamics with exploration and credit assignment

Key improvements over 0.11:
1. Proper learning rates and STDP amplitudes
2. Temperature annealing for exploration
3. Reward amplification for better credit assignment
4. Success detection and learning boost
5. More plastic connections
6. All performance optimizations retained
"""

from functools import partial
from typing import Any, Dict, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, random

from export import DataExporter
from world.simple_grid_0003 import SimpleGridWorld  # Ultra-optimized world

from ..phase_0_10.agent_vectorized import create_connectivity_vectorized
from .config import NetworkParams, SnnAgentConfig


class AgentState(NamedTuple):
    """Complete agent state with reward boost tracking."""

    # Core dynamics
    v: jnp.ndarray  # Membrane potentials
    spike: jnp.ndarray  # Current spikes
    refractory: jnp.ndarray  # Refractory counters

    # Synaptic currents (separate E/I for biological realism)
    syn_current_e: jnp.ndarray  # Excitatory currents
    syn_current_i: jnp.ndarray  # Inhibitory currents

    # Synaptic traces (two timescales for richer STDP)
    trace_fast: jnp.ndarray  # Fast trace (~20ms)
    trace_slow: jnp.ndarray  # Slow trace (~100ms)

    # Homeostasis
    firing_rate: jnp.ndarray  # Running average firing rate
    threshold_adapt: jnp.ndarray  # Adaptive threshold modulation

    # Learning
    eligibility_trace: jnp.ndarray  # Synaptic eligibility
    dopamine: float  # Neuromodulator level
    value_estimate: float  # Predicted future reward

    # Weights and connectivity
    w: jnp.ndarray  # All synaptic weights
    w_mask: jnp.ndarray  # Connection existence mask
    w_plastic_mask: jnp.ndarray  # Which weights can learn

    # Population identity
    is_excitatory: jnp.ndarray  # E/I identity for each neuron
    neuron_types: jnp.ndarray  # 0=sensory, 1=processing, 2=readout

    # Motor output
    motor_trace: jnp.ndarray  # Integrated motor command

    # Input representation
    input_channels: jnp.ndarray  # Current population-coded input

    # Multi-episode learning
    weight_momentum: jnp.ndarray  # Momentum for weight updates
    episodes_completed: int  # Track number of episodes for adaptive learning

    # Success detection and reward boost
    reward_boost_timer: float  # Timer for learning rate boost
    rewards_this_episode: int  # Count rewards in current episode
    current_temperature: float  # Current action temperature for exploration

    # Performance optimization - reusable buffers
    spike_float_buffer: jnp.ndarray  # Reusable spike float conversion


class PrecomputedConstants(NamedTuple):
    """Precomputed constants for efficiency."""

    syn_e_decay: float
    syn_i_decay: float
    trace_fast_decay: float
    trace_slow_decay: float
    eligibility_decay: float
    dopamine_decay: float
    motor_decay: float
    homeostatic_alpha: float
    reward_boost_decay: float  # For reward boost timer
    zero_input_buffer: jnp.ndarray
    action_indices: jnp.ndarray  # For motor decoding
    motor_decode_matrix: jnp.ndarray  # Vectorized motor decoding


# === HELPER FUNCTIONS ===


@jit
def _apply_dale_principle(w: jnp.ndarray, is_excitatory: jnp.ndarray) -> jnp.ndarray:
    """Apply Dale's principle: neurons release consistent neurotransmitter."""
    # Pre-synaptic neuron determines sign (rows in weight matrix)
    E_mask = is_excitatory[:, None]
    return jnp.where(E_mask, jnp.abs(w), -jnp.abs(w))


@partial(jit, static_argnames=["params"])
def _get_adaptive_learning_rate(state: AgentState, params: NetworkParams) -> float:
    """Compute learning rate with episode decay and reward boost."""
    # Base learning rate with episode decay
    decay_rate = params.LEARNING_RATE_DECAY
    min_rate = params.MIN_LEARNING_RATE
    adaptive_rate = params.BASE_LEARNING_RATE * (decay_rate**state.episodes_completed)
    base_rate = jnp.maximum(adaptive_rate, min_rate)

    # Apply reward boost if timer is active
    boost_factor = jnp.where(state.reward_boost_timer > 0, params.REWARD_BOOST_FACTOR, 1.0)

    return base_rate * boost_factor


@partial(jit, static_argnames=["params"])
def _get_action_temperature(state: AgentState, params: NetworkParams) -> float:
    """Get current action temperature for exploration-exploitation balance."""
    return state.current_temperature


@partial(jit, static_argnames=["params"])
def _encode_gradient_population(
    gradient: float, params: NetworkParams, key: random.PRNGKey
) -> jnp.ndarray:
    """Convert scalar gradient to population code with tuned neurons."""
    # Each channel has a preferred gradient value
    preferred_values = jnp.linspace(0, 1, params.NUM_INPUT_CHANNELS)

    # Gaussian tuning curves
    activations = jnp.exp(
        -((gradient - preferred_values) ** 2) / (2 * params.INPUT_TUNING_WIDTH**2)
    )

    # Add small noise for robustness
    noise = random.normal(key, activations.shape) * 0.05
    activations = jnp.maximum(activations + noise, 0.0)

    # Normalize to total input current
    total_activation = jnp.sum(activations)
    normalized = jnp.where(
        total_activation > 0,
        activations / total_activation * params.INPUT_GAIN,
        jnp.ones_like(activations) * params.INPUT_GAIN / params.NUM_INPUT_CHANNELS,
    )

    return normalized


# === NEURAL DYNAMICS ===


@partial(jit, static_argnames=["params"])
def _neuron_step_optimized(
    state: AgentState, params: NetworkParams, constants: PrecomputedConstants, key: random.PRNGKey
) -> Tuple[AgentState, jnp.ndarray]:
    """
    Core neural dynamics with optimizations:
    - Precomputed exponentials
    - Efficient spike handling
    - Reduced allocations
    """
    # Update refractory period
    refractory_new = jnp.maximum(0, state.refractory - 1.0)

    # Decay synaptic currents using precomputed factors
    syn_current_e_new = state.syn_current_e * constants.syn_e_decay
    syn_current_i_new = state.syn_current_i * constants.syn_i_decay

    # Compute synaptic input
    # Use pre-allocated spike float buffer
    spike_float = state.spike_float_buffer
    all_activity = jnp.concatenate([state.input_channels, spike_float])

    # Compute weighted input
    syn_input = state.w @ all_activity

    # Separate E/I components based on weight sign
    syn_current_e_new += jnp.maximum(syn_input, 0)
    syn_current_i_new += jnp.minimum(syn_input, 0)

    # Total input current
    i_total = params.BASELINE_CURRENT + syn_current_e_new + syn_current_i_new

    # Add noise
    noise = random.normal(key, state.v.shape) * params.NOISE_SCALE
    i_total += noise

    # Update membrane potential
    dv = (-state.v + params.V_REST + i_total) / params.TAU_V
    v_new = state.v + dv

    # Apply adaptive threshold
    effective_threshold = params.V_THRESHOLD + state.threshold_adapt

    # Generate spikes
    can_spike = refractory_new == 0
    spike_new = (v_new >= effective_threshold) & can_spike

    # Reset and refractory
    v_new = jnp.where(spike_new, params.V_RESET, v_new)
    refractory_new = jnp.where(spike_new, params.REFRACTORY_TIME, refractory_new)

    # Update synaptic traces using precomputed decays
    trace_fast_new = state.trace_fast * constants.trace_fast_decay + spike_new
    trace_slow_new = state.trace_slow * constants.trace_slow_decay + spike_new * 0.5

    # Update firing rate estimate
    spike_rate = spike_new.astype(float) * 1000.0  # Convert to Hz
    firing_rate_new = (
        state.firing_rate * (1 - constants.homeostatic_alpha)
        + spike_rate * constants.homeostatic_alpha
    )

    # Update threshold adaptation
    rate_error = firing_rate_new - params.TARGET_RATE_HZ
    threshold_adapt_new = state.threshold_adapt + params.THRESHOLD_ADAPT_RATE * rate_error
    threshold_adapt_new = jnp.clip(
        threshold_adapt_new, -params.MAX_THRESHOLD_ADAPT, params.MAX_THRESHOLD_ADAPT
    )

    # Update spike float buffer
    spike_float_new = jnp.where(spike_new, 1.0, 0.0)

    # Decay reward boost timer
    reward_boost_timer_new = state.reward_boost_timer * constants.reward_boost_decay

    return state._replace(
        v=v_new,
        spike=spike_new,
        refractory=refractory_new,
        syn_current_e=syn_current_e_new,
        syn_current_i=syn_current_i_new,
        trace_fast=trace_fast_new,
        trace_slow=trace_slow_new,
        firing_rate=firing_rate_new,
        threshold_adapt=threshold_adapt_new,
        spike_float_buffer=spike_float_new,
        reward_boost_timer=reward_boost_timer_new,
    ), spike_float_new


# === LEARNING ===


@partial(jit, static_argnames=["params"])
def _compute_eligibility_trace_optimized(
    state: AgentState, params: NetworkParams, constants: PrecomputedConstants
) -> jnp.ndarray:
    """
    Compute STDP-based eligibility traces efficiently.
    """
    # Standard STDP using fast trace
    # Use pre-allocated zero buffer and spike float buffer
    pre_trace = constants.zero_input_buffer.at[params.NUM_INPUT_CHANNELS :].set(state.trace_fast)
    post_spike = state.spike_float_buffer

    # LTP: pre trace × post spike
    ltp = pre_trace[None, :] * post_spike[:, None]

    # LTD: pre spike × post trace
    pre_spike = constants.zero_input_buffer.at[params.NUM_INPUT_CHANNELS :].set(
        state.spike_float_buffer
    )
    post_trace = state.trace_fast
    ltd = pre_spike[None, :] * post_trace[:, None]

    # STDP update with stronger amplitudes
    stdp = params.STDP_A_PLUS * ltp - params.STDP_A_MINUS * ltd

    # Only apply to plastic synapses
    stdp = jnp.where(state.w_plastic_mask, stdp, 0.0)

    # Decay existing traces using precomputed factor
    new_eligibility = state.eligibility_trace * constants.eligibility_decay + stdp

    return new_eligibility


@partial(jit, static_argnames=["params"])
def _three_factor_learning_optimized(
    state: AgentState,
    reward: float,
    gradient: float,
    params: NetworkParams,
    constants: PrecomputedConstants,
) -> AgentState:
    """
    Three-factor learning rule with fixed credit assignment.
    """
    # 1. Compute reward prediction error
    td_error = reward + params.REWARD_DISCOUNT * state.value_estimate - state.value_estimate
    new_value = state.value_estimate + params.REWARD_PREDICTION_RATE * td_error

    # 2. Amplify actual rewards vs gradient rewards
    actual_reward_component = reward * params.ACTUAL_REWARD_SCALE
    gradient_reward_component = gradient * params.GRADIENT_REWARD_SCALE

    # Total reward signal with proper weighting
    total_reward_signal = actual_reward_component + gradient_reward_component + td_error * 0.3

    # 3. Update dopamine with stronger response to actual rewards
    dopamine_response = (
        state.dopamine * constants.dopamine_decay
        + params.BASELINE_DOPAMINE * (1 - constants.dopamine_decay)
        + total_reward_signal
    )
    new_dopamine = jnp.clip(dopamine_response, 0.0, 2.0)

    # 4. Compute dopamine modulation factor
    da_factor = (new_dopamine - params.BASELINE_DOPAMINE) / params.BASELINE_DOPAMINE
    modulation = jax.nn.tanh(da_factor * 2.0)

    # 5. Update eligibility traces
    new_eligibility = _compute_eligibility_trace_optimized(state, params, constants)

    # 6. Get adaptive learning rate (includes reward boost)
    adaptive_lr = _get_adaptive_learning_rate(state, params)

    # 7. Compute weight updates with momentum
    dw = adaptive_lr * modulation * new_eligibility

    # Apply momentum
    momentum_decay = params.WEIGHT_MOMENTUM_DECAY
    new_momentum = state.weight_momentum * momentum_decay + dw * (1 - momentum_decay)

    # Add weight decay
    weight_penalty = params.WEIGHT_DECAY * state.w * state.w_plastic_mask

    # 8. Update weights with momentum and soft bounds
    w_new = state.w + new_momentum - weight_penalty

    # Soft saturation using tanh
    n_in = params.NUM_INPUT_CHANNELS
    w_neural = w_new[:, n_in:]

    # Apply soft bounds to neural weights
    scale = params.MAX_WEIGHT_SCALE
    w_neural_sign = jnp.sign(w_neural)
    w_neural_mag = jnp.abs(w_neural)
    w_neural_bounded = w_neural_sign * scale * jnp.tanh(w_neural_mag / scale)

    # Apply Dale's principle
    w_neural_dale = _apply_dale_principle(w_neural_bounded, state.is_excitatory)
    w_new = w_new.at[:, n_in:].set(w_neural_dale)

    # Zero out non-existent connections
    w_new = jnp.where(state.w_mask, w_new, 0.0)

    # 9. Update reward boost timer if reward collected
    new_reward_boost_timer = jnp.where(
        reward > 0,
        params.REWARD_BOOST_DURATION,  # Reset timer on reward
        state.reward_boost_timer,
    )

    # 10. Track rewards
    new_rewards_count = state.rewards_this_episode + (reward > 0).astype(int)

    return state._replace(
        w=w_new,
        weight_momentum=new_momentum,
        eligibility_trace=new_eligibility * 0.9,  # Decay after use
        dopamine=new_dopamine,
        value_estimate=new_value,
        reward_boost_timer=new_reward_boost_timer,
        rewards_this_episode=new_rewards_count,
    )


# === ACTION SELECTION ===


@partial(jit, static_argnames=["params"])
def _decode_action_optimized(
    state: AgentState, params: NetworkParams, constants: PrecomputedConstants, key: random.PRNGKey
) -> Tuple[int, jnp.ndarray]:
    """
    Decode action from readout population activity with exploration.
    """
    # Get readout neuron spikes
    readout_mask = state.neuron_types == 2
    readout_spikes = jnp.where(readout_mask, state.spike, False).astype(float)

    # Update motor trace with precomputed decay
    # Map readout neurons to 4 motor commands using precomputed indices
    readout_start = params.NUM_SENSORY + params.NUM_PROCESSING

    # Vectorized motor input computation using precomputed matrix
    readout_only = readout_spikes[readout_start:]
    motor_input = constants.motor_decode_matrix @ readout_only

    # Update motor trace
    motor_trace_new = state.motor_trace * constants.motor_decay + motor_input

    # Action selection via softmax with current temperature
    action_temperature = _get_action_temperature(state, params)
    action_logits = motor_trace_new / action_temperature
    action_probs = jax.nn.softmax(action_logits)

    # Sample action
    action = random.categorical(key, jnp.log(action_probs + 1e-8))

    return action, motor_trace_new


# === INITIALIZATION FUNCTIONS ===


def _initialize_weights(
    key: random.PRNGKey,
    params: NetworkParams,
    w_mask: jnp.ndarray,
    is_excitatory: jnp.ndarray,
    neuron_types: jnp.ndarray,
) -> jnp.ndarray:
    """Initialize weights with appropriate distributions for each connection type."""
    n = len(neuron_types)
    n_in = params.NUM_INPUT_CHANNELS
    n_total = n_in + n

    w = jnp.zeros((n, n_total))

    # Helper to generate weights with proper statistics
    def sample_weights(key, shape, mean, scale):
        return random.normal(key, shape) * (mean * scale) + mean

    keys = random.split(key, 5)

    # 1. Input → Sensory weights
    input_weights = sample_weights(keys[0], (n, n_in), params.W_INPUT_SENSORY, params.W_INIT_SCALE)
    w = w.at[:, :n_in].set(jnp.where(w_mask[:, :n_in], jnp.abs(input_weights), 0))

    # 2. Sensory → Processing weights
    sens_indices = jnp.where(neuron_types == 0)[0]
    proc_indices = jnp.where(neuron_types == 1)[0]

    for _i, sens_idx in enumerate(sens_indices):
        for _j, proc_idx in enumerate(proc_indices):
            if w_mask[proc_idx, n_in + sens_idx]:
                w_val = jnp.abs(
                    random.normal(keys[1], ()) * params.W_SENSORY_PROC * params.W_INIT_SCALE
                    + params.W_SENSORY_PROC
                )
                w = w.at[proc_idx, n_in + sens_idx].set(w_val)

    # 3. Processing ↔ Processing weights
    for _i, src_idx in enumerate(proc_indices):
        for _j, tgt_idx in enumerate(proc_indices):
            if w_mask[tgt_idx, n_in + src_idx]:
                # Weight based on connection type
                if is_excitatory[src_idx] and is_excitatory[tgt_idx]:  # E→E
                    mean = params.W_PROC_PROC_E
                elif is_excitatory[src_idx] and not is_excitatory[tgt_idx]:  # E→I
                    mean = params.W_EI
                elif not is_excitatory[src_idx] and is_excitatory[tgt_idx]:  # I→E
                    mean = params.W_IE
                else:  # I→I
                    mean = params.W_II

                w_val = random.normal(keys[2], ()) * (mean * params.W_INIT_SCALE) + mean
                w = w.at[tgt_idx, n_in + src_idx].set(jnp.abs(w_val))

    # 4. Processing → Readout weights
    readout_indices = jnp.where(neuron_types == 2)[0]
    for _i, src_idx in enumerate(proc_indices):
        if is_excitatory[src_idx]:  # Only E neurons
            for _j, tgt_idx in enumerate(readout_indices):
                if w_mask[tgt_idx, n_in + src_idx]:
                    w_val = jnp.abs(
                        random.normal(keys[3], ()) * params.W_PROC_READOUT * params.W_INIT_SCALE
                        + params.W_PROC_READOUT
                    )
                    w = w.at[tgt_idx, n_in + src_idx].set(w_val)

    return w


@partial(jit, static_argnames=["params", "soft_reset"])
def _reset_episode_state(
    state: AgentState, params: NetworkParams, soft_reset: bool = True
) -> AgentState:
    """
    Reset state for new episode while preserving weights and some homeostatic information.
    """
    n_total = len(state.neuron_types)

    # Update temperature for next episode
    new_temperature = jnp.maximum(
        state.current_temperature * params.TEMPERATURE_DECAY, params.FINAL_ACTION_TEMPERATURE
    )

    if soft_reset:
        # Soft reset preserves some information
        return state._replace(
            # Neural dynamics - reset completely
            v=jnp.full(n_total, params.V_REST),
            spike=jnp.zeros(n_total, dtype=bool),
            refractory=jnp.zeros(n_total),
            # Synaptic state - reset completely
            syn_current_e=jnp.zeros(n_total),
            syn_current_i=jnp.zeros(n_total),
            trace_fast=jnp.zeros(n_total),
            trace_slow=jnp.zeros(n_total),
            # Homeostasis - partial preservation
            firing_rate=state.firing_rate * 0.9 + params.TARGET_RATE_HZ * 0.1,
            threshold_adapt=state.threshold_adapt * 0.9,
            # Learning - partial preservation
            eligibility_trace=jnp.zeros_like(state.eligibility_trace),
            dopamine=state.dopamine * 0.8 + params.BASELINE_DOPAMINE * 0.2,
            value_estimate=state.value_estimate * 0.5,
            # I/O - reset
            motor_trace=jnp.zeros(4),
            input_channels=jnp.zeros(params.NUM_INPUT_CHANNELS),
            # Multi-episode learning - increment episode counter
            episodes_completed=state.episodes_completed + 1,
            # Success tracking - reset for new episode
            reward_boost_timer=0.0,
            rewards_this_episode=0,
            current_temperature=new_temperature,
            # Performance optimization - reset spike buffer
            spike_float_buffer=jnp.zeros_like(state.spike_float_buffer),
        )
    else:
        # Hard reset - only preserve weights and structure
        return state._replace(
            v=jnp.full(n_total, params.V_REST),
            spike=jnp.zeros(n_total, dtype=bool),
            refractory=jnp.zeros(n_total),
            syn_current_e=jnp.zeros(n_total),
            syn_current_i=jnp.zeros(n_total),
            trace_fast=jnp.zeros(n_total),
            trace_slow=jnp.zeros(n_total),
            firing_rate=jnp.full(n_total, params.TARGET_RATE_HZ),
            threshold_adapt=jnp.zeros(n_total),
            eligibility_trace=jnp.zeros_like(state.eligibility_trace),
            dopamine=params.BASELINE_DOPAMINE,
            value_estimate=0.0,
            motor_trace=jnp.zeros(4),
            input_channels=jnp.zeros(params.NUM_INPUT_CHANNELS),
            episodes_completed=state.episodes_completed + 1,
            reward_boost_timer=0.0,
            rewards_this_episode=0,
            current_temperature=new_temperature,
            spike_float_buffer=jnp.zeros_like(state.spike_float_buffer),
        )


@partial(jit, static_argnames=["params"])
def _consolidate_weights(state: AgentState, params: NetworkParams) -> AgentState:
    """
    Optional weight consolidation between episodes to maintain stability.
    """
    if not params.WEIGHT_CONSOLIDATION:
        return state

    # Compute weight statistics
    n_in = params.NUM_INPUT_CHANNELS
    w_neural = state.w[:, n_in:]

    # Normalize weights to prevent runaway growth
    # Use where instead of boolean indexing for JIT compatibility
    mask = state.w_mask[:, n_in:]
    masked_weights = jnp.where(mask, jnp.abs(w_neural), 0.0)
    num_connections = jnp.sum(mask)
    w_mean = jnp.sum(masked_weights) / jnp.maximum(num_connections, 1.0)
    target_mean = params.WEIGHT_CONSOLIDATION_TARGET

    scale_factor = jnp.where(
        w_mean > 0, 1.0 + (target_mean / w_mean - 1.0) * params.WEIGHT_CONSOLIDATION_RATE, 1.0
    )

    # Scale neural weights
    w_neural_scaled = w_neural * scale_factor

    # Apply Dale's principle
    w_neural_scaled = _apply_dale_principle(w_neural_scaled, state.is_excitatory)

    # Update weights
    w_new = state.w.at[:, n_in:].set(w_neural_scaled)
    w_new = jnp.where(state.w_mask, w_new, 0.0)

    return state._replace(w=w_new)


# === MAIN AGENT CLASS ===


class SnnAgent:
    """Phase 0.12 SNN Agent with fixed learning dynamics."""

    def __init__(self, config: SnnAgentConfig):
        self.config = config
        self.params = config.network_params
        self.world = SimpleGridWorld(config.world_config)

        # Initialize PRNG key
        self.master_key = random.PRNGKey(config.exp_config.seed)

        # Precompute constants
        self.constants = self._precompute_constants()

        print("Initializing network state...")
        import time

        start_time = time.time()
        init_key, self.master_key = random.split(self.master_key)
        self.state: AgentState = self._initialize_state(init_key)
        print(f"Network initialized in {time.time() - start_time:.2f}s")

        # Monitoring
        self.firing_rate_history = []
        self.weight_stats_history = []

    def _precompute_constants(self) -> PrecomputedConstants:
        """Precompute all exponential decays and constants."""
        p = self.params
        n_total = p.NUM_SENSORY + p.NUM_PROCESSING + p.NUM_READOUT

        # Precompute action indices for motor decoding
        n_per_action = p.NUM_READOUT // 4
        action_indices = jnp.repeat(jnp.arange(4), n_per_action)

        # Create motor decode matrix for vectorized computation
        motor_decode_matrix = jnp.zeros((4, p.NUM_READOUT))
        for i in range(4):
            start_idx = i * n_per_action
            end_idx = start_idx + n_per_action
            motor_decode_matrix = motor_decode_matrix.at[i, start_idx:end_idx].set(1.0)

        return PrecomputedConstants(
            syn_e_decay=jnp.exp(-1.0 / p.TAU_SYN_E),
            syn_i_decay=jnp.exp(-1.0 / p.TAU_SYN_I),
            trace_fast_decay=jnp.exp(-1.0 / p.TAU_TRACE_FAST),
            trace_slow_decay=jnp.exp(-1.0 / p.TAU_TRACE_SLOW),
            eligibility_decay=jnp.exp(-1.0 / p.TAU_ELIGIBILITY),
            dopamine_decay=jnp.exp(-1.0 / p.TAU_DOPAMINE),
            motor_decay=jnp.exp(-1.0 / p.MOTOR_TAU),
            homeostatic_alpha=1.0 / p.HOMEOSTATIC_TAU,
            reward_boost_decay=jnp.exp(-1.0 / p.REWARD_BOOST_DURATION),
            zero_input_buffer=jnp.zeros(p.NUM_INPUT_CHANNELS + n_total),
            action_indices=action_indices,
            motor_decode_matrix=motor_decode_matrix,
        )

    def _initialize_state(self, key: random.PRNGKey) -> AgentState:
        """Initialize the complete agent state."""
        p = self.params
        n_total = p.NUM_SENSORY + p.NUM_PROCESSING + p.NUM_READOUT

        # Assign neuron types
        neuron_types = jnp.concatenate(
            [
                jnp.zeros(p.NUM_SENSORY, dtype=int),  # Type 0: sensory
                jnp.ones(p.NUM_PROCESSING, dtype=int),  # Type 1: processing
                jnp.full(p.NUM_READOUT, 2, dtype=int),  # Type 2: readout
            ]
        )

        # Assign E/I identity (applies to all populations)
        is_excitatory = jnp.arange(n_total) < int(n_total * p.EXCITATORY_RATIO)

        # Create connectivity (using vectorized version for speed)
        keys = random.split(key, 3)
        w_mask, w_plastic_mask = create_connectivity_vectorized(
            keys[0], p, n_total, is_excitatory, neuron_types
        )

        # Initialize weights
        w = _initialize_weights(keys[1], p, w_mask, is_excitatory, neuron_types)

        # Initialize state
        return AgentState(
            # Neural dynamics
            v=jnp.full(n_total, p.V_REST),
            spike=jnp.zeros(n_total, dtype=bool),
            refractory=jnp.zeros(n_total),
            # Synaptic state
            syn_current_e=jnp.zeros(n_total),
            syn_current_i=jnp.zeros(n_total),
            trace_fast=jnp.zeros(n_total),
            trace_slow=jnp.zeros(n_total),
            # Homeostasis
            firing_rate=jnp.full(n_total, p.TARGET_RATE_HZ),
            threshold_adapt=jnp.zeros(n_total),
            # Learning
            eligibility_trace=jnp.zeros_like(w),
            dopamine=p.BASELINE_DOPAMINE,
            value_estimate=0.0,
            # Connectivity
            w=w,
            w_mask=w_mask,
            w_plastic_mask=w_plastic_mask,
            # Identity
            is_excitatory=is_excitatory,
            neuron_types=neuron_types,
            # I/O
            motor_trace=jnp.zeros(4),
            input_channels=jnp.zeros(p.NUM_INPUT_CHANNELS),
            # Multi-episode learning
            weight_momentum=jnp.zeros_like(w),
            episodes_completed=0,
            # Success tracking
            reward_boost_timer=0.0,
            rewards_this_episode=0,
            current_temperature=p.INITIAL_ACTION_TEMPERATURE,
            # Performance optimization
            spike_float_buffer=jnp.zeros(n_total),
        )

    def _monitor_stability(self, state: AgentState, step: int):
        """Monitor network stability metrics."""
        if not self.config.exp_config.monitor_rates:
            return

        # Check firing rates
        mean_rate = jnp.mean(state.firing_rate)
        if step % 1000 == 0:
            self.firing_rate_history.append(float(mean_rate))

        # Warn on instabilities
        if self.config.exp_config.check_stability:
            if mean_rate > 50:  # Hz
                print(f"⚠️ Warning: High firing rate at step {step}: {mean_rate:.1f} Hz")
            elif mean_rate < 0.1:
                print(f"⚠️ Warning: Network too quiet at step {step}: {mean_rate:.1f} Hz")

    def run_episode(
        self,
        episode_key: random.PRNGKey,
        episode_num: int,
        exporter: Optional[DataExporter] = None,
        progress_callback=None,
        performance_mode: bool = False,
    ) -> Dict[str, Any]:
        """Run a single episode with optimized performance."""
        import time

        episode_start_time = time.time()

        # Reset world with proper key
        world_key, episode_key = random.split(episode_key)
        world_state, obs = self.world.reset(world_key)

        # Reset agent state (soft reset by default, preserving weights)
        if episode_num > 0:
            self.state = _reset_episode_state(self.state, self.params, soft_reset=True)

        # Optional weight consolidation
        if episode_num > 0 and episode_num % self.params.CONSOLIDATION_INTERVAL == 0:
            self.state = _consolidate_weights(self.state, self.params)

        # Start data export (skip in performance mode)
        if not performance_mode and exporter is not None:
            exporter.start_episode(episode_num)
            exporter.log_static_episode_data(
                "world_setup", {"reward_positions": np.asarray(world_state.reward_positions)}
            )

            # Save network structure
            if episode_num == 0:
                self._export_network_structure(exporter)

        # Episode loop
        rewards_collected = 0
        actual_rewards_collected = 0  # Track non-proximity rewards
        max_steps = self.config.world_config.max_timesteps

        # Pre-split keys for entire episode for efficiency
        all_keys = random.split(episode_key, max_steps + 1)
        episode_key = all_keys[0]
        step_keys = all_keys[1:]

        for step in range(max_steps):
            # Use pre-split keys
            step_key = step_keys[step]
            encode_key, neuron_key, action_key, world_key = random.split(step_key, 4)

            # 1. Encode input
            input_channels = _encode_gradient_population(obs.gradient, self.params, encode_key)
            self.state = self.state._replace(input_channels=input_channels)

            # 2. Neural dynamics step (optimized)
            self.state, spike_float = _neuron_step_optimized(
                self.state, self.params, self.constants, neuron_key
            )

            # 3. Decode action (optimized)
            action, motor_trace = _decode_action_optimized(
                self.state, self.params, self.constants, action_key
            )
            self.state = self.state._replace(motor_trace=motor_trace)

            # 4. Environment step with proper key
            result = self.world.step(world_state, int(action), world_key)
            world_state, obs, reward, done = (
                result.state,
                result.observation,
                result.reward,
                result.done,
            )

            # 5. Learning step (optimized)
            self.state = _three_factor_learning_optimized(
                self.state, reward, obs.gradient, self.params, self.constants
            )

            # 6. Track rewards
            if reward > 0:
                rewards_collected += 1
                # Check if it's an actual reward (not just proximity)
                if reward > self.config.world_config.proximity_reward:
                    actual_rewards_collected += 1

            # 7. Monitoring
            self._monitor_stability(self.state, step)

            # 8. Progress callback (every 500 steps and on final step)
            if progress_callback and (step % 500 == 0 or step == max_steps - 1):
                elapsed = time.time() - episode_start_time
                progress_callback(
                    step, max_steps, rewards_collected, elapsed, actual_rewards_collected
                )

            # 9. Data logging (skip in performance mode)
            if not performance_mode and exporter is not None:
                exporter.log(
                    timestep=step,
                    neural_state={
                        "v": self.state.v  # Keep as JAX array, converted only when saved
                    },
                    spikes=self.state.spike,  # Separate for efficient sparse handling
                    behavior={
                        "action": int(action),
                        "pos_x": int(world_state.agent_pos[0]),
                        "pos_y": int(world_state.agent_pos[1]),
                        "gradient": float(obs.gradient),
                    },
                    reward=float(reward),
                )

            if done:
                break

        # Episode summary
        summary = {
            # Core fields
            "total_reward": float(world_state.total_reward),
            # Fixed: use manually tracked count
            "rewards_collected": actual_rewards_collected,
            "steps_taken": world_state.timestep,
            # Enhanced fields
            "mean_firing_rate": float(jnp.mean(self.state.firing_rate)),
            "final_dopamine": float(self.state.dopamine),
            "final_value_estimate": float(self.state.value_estimate),
            # Multi-episode learning fields
            "episodes_completed": int(self.state.episodes_completed),
            "current_learning_rate": float(_get_adaptive_learning_rate(self.state, self.params)),
            "current_temperature": float(self.state.current_temperature),
            # Success tracking
            "rewards_this_episode": int(self.state.rewards_this_episode),
            "actual_rewards_collected": actual_rewards_collected,  # Non-proximity rewards
        }

        if not performance_mode and exporter is not None:
            exporter.end_episode(success=jnp.all(world_state.reward_collected), summary=summary)

        return summary

    def _export_network_structure(self, exporter: DataExporter):
        """Export network structure for analysis."""
        # Same as original agent
        n_neurons = len(self.state.neuron_types)
        n_in = self.params.NUM_INPUT_CHANNELS

        recurrent_mask = self.state.w_mask[:, n_in:]
        conn_indices = jnp.where(recurrent_mask)

        basic_structure = {
            "neurons": {
                "neuron_ids": np.arange(n_neurons),
                "is_excitatory": np.asarray(self.state.is_excitatory),
            },
            "connections": {
                "source_ids": np.asarray(conn_indices[1]),
                "target_ids": np.asarray(conn_indices[0]),
            },
            "initial_weights": {"weights": np.asarray(self.state.w[:, n_in:][recurrent_mask])},
        }

        exporter.save_network_structure(**basic_structure)

        enhanced_data = {
            "neuron_types": np.asarray(self.state.neuron_types, dtype=np.int32),
            "connectivity_stats": np.array(
                [
                    np.sum(self.state.w_mask[:, :n_in]),
                    np.sum(self.state.w_plastic_mask),
                    np.sum(recurrent_mask),
                ],
                dtype=np.int32,
            ),
        }
        exporter.log_static_episode_data("network_stats", enhanced_data)

    def run_experiment(self, performance_mode: bool = False, no_write: bool = False):
        """Run complete experiment with multi-episode learning."""
        import time

        if not performance_mode:
            if no_write:
                print("Initializing data exporter (no-write mode)...")
            else:
                print("Initializing data exporter...")
            exporter = DataExporter(
                experiment_name="snn_agent_phase12",
                output_base_dir=self.config.exp_config.export_dir,
                compression="gzip",
                compression_level=1,
                no_write=no_write,
            )
            exporter.__enter__()

            if not no_write:
                print("Saving configuration...")
            exporter.save_config(self.config)
        else:
            exporter = None

        # Run episodes
        all_summaries = []
        episode_times = []
        time.time()

        for i in range(self.config.exp_config.n_episodes):
            print(f"\n--- Episode {i + 1}/{self.config.exp_config.n_episodes} ---")
            print(
                f"Current learning rate: {_get_adaptive_learning_rate(self.state, self.params):.6f}"
            )
            print(f"Current temperature: {self.state.current_temperature:.3f}")

            episode_key, self.master_key = random.split(self.master_key)
            episode_start = time.time()

            # Progress callback with ETA
            def progress_callback(step, max_steps, rewards, episode_elapsed, actual_rewards=0):
                # Calculate progress
                progress_pct = ((step + 1) / max_steps) * 100
                bar_width = 30
                filled = int(bar_width * progress_pct / 100)
                bar = "█" * filled + "░" * (bar_width - filled)

                # Episode ETA
                if step > 0:
                    episode_eta = (episode_elapsed / (step + 1)) * (max_steps - step - 1)
                else:
                    episode_eta = 0

                # Total experiment ETA
                remaining_episodes = self.config.exp_config.n_episodes - i - 1

                if episode_times:
                    avg_episode_time = sum(episode_times) / len(episode_times)
                else:
                    avg_episode_time = episode_elapsed / (step + 1) * max_steps if step > 0 else 60

                total_eta = episode_eta + (remaining_episodes * avg_episode_time)

                # Format times
                def format_time(seconds):
                    if seconds < 60:
                        return f"{seconds:.0f}s"
                    elif seconds < 3600:
                        return f"{seconds // 60:.0f}m{seconds % 60:.0f}s"
                    else:
                        return f"{seconds // 3600:.0f}h{(seconds % 3600) // 60:.0f}m"

                print(
                    f"\r  {bar} {progress_pct:5.1f}% | Steps: {step + 1:,}/{max_steps:,} | "
                    f"Rewards: {rewards} (actual: {actual_rewards}) | Episode ETA: {format_time(episode_eta)} | "
                    f"Total ETA: {format_time(total_eta)}",
                    end="",
                    flush=True,
                )

            summary = self.run_episode(
                episode_key,
                episode_num=i,
                exporter=exporter,
                progress_callback=progress_callback,
                performance_mode=performance_mode,
            )
            all_summaries.append(summary)

            # Track episode time
            episode_time = time.time() - episode_start
            episode_times.append(episode_time)

            print()  # New line after progress bar
            print(f"  Episode Time: {episode_time:.1f}s")
            print(f"  Total Reward: {summary['total_reward']:.2f}")
            print(f"  Rewards Collected: {summary['rewards_collected']}")
            print(f"  Mean Firing Rate: {summary['mean_firing_rate']:.1f} Hz")

            # Check for learning
            if i > 0:
                reward_trend = summary["total_reward"] - all_summaries[0]["total_reward"]
                print(f"  Reward Δ from first: {reward_trend:+.2f}")

                # Check improvement over last 5 episodes
                if i >= 5:
                    recent_avg = sum(s["total_reward"] for s in all_summaries[-5:]) / 5
                    early_avg = sum(s["total_reward"] for s in all_summaries[:5]) / 5
                    improvement = recent_avg - early_avg
                    print(f"  Avg reward improvement (last 5 vs first 5): {improvement:+.2f}")

                    # Check actual reward collection improvement
                    recent_actual = (
                        sum(s["actual_rewards_collected"] for s in all_summaries[-5:]) / 5
                    )
                    early_actual = sum(s["actual_rewards_collected"] for s in all_summaries[:5]) / 5
                    actual_improvement = recent_actual - early_actual
                    print(
                        f"  Actual reward improvement (last 5 vs first 5): {actual_improvement:+.2f}"
                    )

        if not performance_mode and exporter is not None:
            exporter.__exit__(None, None, None)

        return all_summaries


# Alias for backward compatibility
OptimizedSnnAgent = SnnAgent
