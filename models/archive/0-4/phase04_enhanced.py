#!/usr/bin/env python3
"""
Phase 0.4 Enhanced - Biologically Plausible Neural Dynamics
keywords: [snn, homeostasis, metaplasticity, population-coding, weight-initialization]

Key enhancements:
1. Connection-type specific weight initialization (E→E, E→I, I→E, I→I)
2. Homeostatic plasticity for stable firing rates
3. Weight-dependent learning with metaplasticity
4. Population coding for inputs and outputs
5. Biologically realistic synaptic dynamics
"""

import argparse
import time
from typing import Dict, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, random

# Configuration
GRID_WORLD_SIZE = 100
NUM_REWARDS = 300
MAX_EPISODE_STEPS = 50000

# Proximity rewards
PROXIMITY_THRESHOLD_HIGH = 0.8
PROXIMITY_THRESHOLD_MED = 0.6
PROXIMITY_REWARD_HIGH = 0.5
PROXIMITY_REWARD_MED = 0.2


class NetworkParams(NamedTuple):
    """Enhanced parameters with biological constraints"""

    # Network size
    NUM_NEURONS: int = 256
    NUM_INPUTS: int = 16  # Population coding for gradient
    NUM_OUTPUTS: int = 4

    # E/I balance (80/20 rule)
    EXCITATORY_RATIO: float = 0.8

    # Neuron parameters
    TAU_V: float = 20.0
    V_REST: float = -65.0
    V_THRESHOLD: float = -50.0
    V_RESET: float = -65.0
    REFRACTORY_TIME: float = 2.0  # ms

    # Homeostatic parameters
    TARGET_RATE: float = 5.0  # Hz target firing rate
    HOMEOSTATIC_TAU: float = 10000.0  # Slow adaptation
    THRESHOLD_ADAPT_RATE: float = 0.0001

    # Synapse parameters
    TAU_FAST: float = 20.0
    TAU_SLOW: float = 100.0
    TAU_SYN_E: float = 5.0  # Fast excitatory synapses
    TAU_SYN_I: float = 10.0  # Slower inhibitory synapses

    # Three-factor learning parameters
    TAU_ELIGIBILITY: float = 1000.0
    TAU_DOPAMINE: float = 200.0
    BASELINE_DOPAMINE: float = 0.2

    # Learning parameters with metaplasticity
    BASE_LEARNING_RATE: float = 0.01
    STDP_WINDOW: float = 20.0
    A_PLUS: float = 0.01
    A_MINUS: float = 0.01

    # Metaplasticity parameters
    TAU_METAPLASTICITY: float = 50000.0  # Very slow
    WEIGHT_DECAY: float = 0.00001  # Prevent unbounded growth

    # Reward prediction
    REWARD_PREDICTION_RATE: float = 0.01
    REWARD_DISCOUNT: float = 0.95

    # Connection-specific weight ranges
    W_EE_MEAN: float = 0.3
    W_EE_STD: float = 0.1
    W_EI_MEAN: float = 0.5  # Strong E→I
    W_EI_STD: float = 0.15
    W_IE_MEAN: float = 0.8  # Strong I→E for balance
    W_IE_STD: float = 0.2
    W_II_MEAN: float = 0.2  # Weak I→I
    W_II_STD: float = 0.05


class EnhancedNetworkState(NamedTuple):
    """Enhanced state with homeostatic and metaplastic variables"""

    # Basic neural state
    v: jnp.ndarray  # Membrane potentials
    spike: jnp.ndarray  # Current spikes
    refractory: jnp.ndarray  # Refractory counters

    # E/I identity
    is_excitatory: jnp.ndarray

    # Homeostatic state
    firing_rate: jnp.ndarray  # Running average firing rate
    threshold_adaptation: jnp.ndarray  # Dynamic threshold adjustment

    # Synaptic state
    syn_current_e: jnp.ndarray  # Excitatory currents
    syn_current_i: jnp.ndarray  # Inhibitory currents
    trace_fast: jnp.ndarray
    trace_slow: jnp.ndarray

    # Three-factor learning state
    eligibility_trace: jnp.ndarray
    dopamine: float

    # Metaplasticity state
    synapse_age: jnp.ndarray  # Track synapse maturity
    learning_rate_factor: jnp.ndarray  # Per-synapse learning modulation

    # Weights
    w: jnp.ndarray  # Recurrent weights
    w_in: jnp.ndarray  # Input weights (population coded)
    w_out: jnp.ndarray  # Output weights

    # Motor output
    motor_spikes: jnp.ndarray
    motor_trace: jnp.ndarray

    # Connection tracking
    connection_mask: jnp.ndarray

    # Reward prediction
    value_estimate: float
    prev_value: float

    # Input population state
    input_population: jnp.ndarray  # Population coded input


def initialize_enhanced_network(
    key: jax.random.PRNGKey, params: NetworkParams
) -> EnhancedNetworkState:
    """Initialize network with biologically plausible parameters"""
    keys = random.split(key, 8)

    # E/I assignment
    num_exc = int(params.NUM_NEURONS * params.EXCITATORY_RATIO)
    is_excitatory = jnp.arange(params.NUM_NEURONS) < num_exc

    # Neural state
    v = jnp.ones(params.NUM_NEURONS) * params.V_REST
    spike = jnp.zeros(params.NUM_NEURONS, dtype=bool)
    refractory = jnp.zeros(params.NUM_NEURONS)

    # Homeostatic state
    firing_rate = jnp.ones(params.NUM_NEURONS) * params.TARGET_RATE
    threshold_adaptation = jnp.zeros(params.NUM_NEURONS)

    # Synaptic state
    syn_current_e = jnp.zeros(params.NUM_NEURONS)
    syn_current_i = jnp.zeros(params.NUM_NEURONS)
    trace_fast = jnp.zeros(params.NUM_NEURONS)
    trace_slow = jnp.zeros(params.NUM_NEURONS)

    # Create biologically plausible connectivity
    connection_mask = create_structured_connectivity(keys[0], params, is_excitatory)

    # Initialize weights with connection-type specific distributions
    w = initialize_synaptic_weights(keys[1], params, is_excitatory, connection_mask)

    # Input weights - population coding with topographic organization
    w_in = initialize_input_weights(keys[2], params)

    # Output weights - only from excitatory neurons
    w_out = random.normal(keys[3], (params.NUM_NEURONS, params.NUM_OUTPUTS)) * 0.1
    w_out = jnp.where(is_excitatory[:, None], jnp.abs(w_out), 0.0)

    # Metaplasticity state
    synapse_age = jnp.zeros((params.NUM_NEURONS, params.NUM_NEURONS))
    learning_rate_factor = jnp.ones((params.NUM_NEURONS, params.NUM_NEURONS))

    # Learning state
    eligibility_trace = jnp.zeros((params.NUM_NEURONS, params.NUM_NEURONS))
    dopamine = params.BASELINE_DOPAMINE

    # Motor state
    motor_spikes = jnp.zeros(params.NUM_OUTPUTS, dtype=bool)
    motor_trace = jnp.zeros(params.NUM_OUTPUTS)

    # Reward prediction
    value_estimate = 0.0
    prev_value = 0.0

    # Input population
    input_population = jnp.zeros(params.NUM_INPUTS)

    return EnhancedNetworkState(
        v=v,
        spike=spike,
        refractory=refractory,
        is_excitatory=is_excitatory,
        firing_rate=firing_rate,
        threshold_adaptation=threshold_adaptation,
        syn_current_e=syn_current_e,
        syn_current_i=syn_current_i,
        trace_fast=trace_fast,
        trace_slow=trace_slow,
        eligibility_trace=eligibility_trace,
        dopamine=dopamine,
        synapse_age=synapse_age,
        learning_rate_factor=learning_rate_factor,
        w=w,
        w_in=w_in,
        w_out=w_out,
        motor_spikes=motor_spikes,
        motor_trace=motor_trace,
        connection_mask=connection_mask,
        value_estimate=value_estimate,
        prev_value=prev_value,
        input_population=input_population,
    )


@jit
def create_structured_connectivity(
    key: jax.random.PRNGKey, params: NetworkParams, is_excitatory: jnp.ndarray
) -> jnp.ndarray:
    """Create biologically realistic connectivity patterns"""
    n = params.NUM_NEURONS

    # Different connection probabilities by type
    P_EE = 0.1  # E→E sparse
    P_EI = 0.4  # E→I dense (feedforward inhibition)
    P_IE = 0.4  # I→E dense (feedback inhibition)
    P_II = 0.2  # I→I moderate

    # Add distance-dependent connectivity (simplified)
    # Neurons are arranged in a ring for simplicity
    positions = jnp.arange(n)
    distance_matrix = jnp.abs(positions[:, None] - positions[None, :])
    distance_matrix = jnp.minimum(distance_matrix, n - distance_matrix)  # Ring topology

    # Local connectivity bias
    local_bias = jnp.exp(-distance_matrix / (n * 0.1))  # 10% characteristic length

    # Create base probability matrix
    prob_matrix = jnp.zeros((n, n))
    EE_mask = jnp.outer(is_excitatory, is_excitatory)
    EI_mask = jnp.outer(is_excitatory, ~is_excitatory)
    IE_mask = jnp.outer(~is_excitatory, is_excitatory)
    II_mask = jnp.outer(~is_excitatory, ~is_excitatory)

    # Apply connection probabilities with local bias
    prob_matrix = jnp.where(EE_mask, P_EE * local_bias, prob_matrix)
    prob_matrix = jnp.where(EI_mask, P_EI, prob_matrix)  # E→I is all-to-all
    prob_matrix = jnp.where(IE_mask, P_IE, prob_matrix)  # I→E is all-to-all
    prob_matrix = jnp.where(II_mask, P_II * local_bias, prob_matrix)

    # Generate connections
    connection_rand = random.uniform(key, (n, n))
    connection_mask = connection_rand < prob_matrix
    connection_mask = connection_mask.at[jnp.diag_indices(n)].set(False)

    return connection_mask


@jit
def initialize_synaptic_weights(
    key: jax.random.PRNGKey,
    params: NetworkParams,
    is_excitatory: jnp.ndarray,
    connection_mask: jnp.ndarray,
) -> jnp.ndarray:
    """Initialize weights with connection-type specific distributions"""
    n = params.NUM_NEURONS
    keys = random.split(key, 4)

    # Create masks for different connection types
    EE_mask = jnp.outer(is_excitatory, is_excitatory) & connection_mask
    EI_mask = jnp.outer(is_excitatory, ~is_excitatory) & connection_mask
    IE_mask = jnp.outer(~is_excitatory, is_excitatory) & connection_mask
    II_mask = jnp.outer(~is_excitatory, ~is_excitatory) & connection_mask

    # Initialize each connection type with appropriate distribution
    w_ee = random.normal(keys[0], (n, n)) * params.W_EE_STD + params.W_EE_MEAN
    w_ei = random.normal(keys[1], (n, n)) * params.W_EI_STD + params.W_EI_MEAN
    w_ie = random.normal(keys[2], (n, n)) * params.W_IE_STD + params.W_IE_MEAN
    w_ii = random.normal(keys[3], (n, n)) * params.W_II_STD + params.W_II_MEAN

    # Ensure positive weights
    w_ee = jnp.abs(w_ee)
    w_ei = jnp.abs(w_ei)
    w_ie = jnp.abs(w_ie)
    w_ii = jnp.abs(w_ii)

    # Combine weights
    w = jnp.zeros((n, n))
    w = jnp.where(EE_mask, w_ee, w)
    w = jnp.where(EI_mask, w_ei, w)
    w = jnp.where(IE_mask, w_ie, w)
    w = jnp.where(II_mask, w_ii, w)

    # Apply Dale's principle
    w = apply_dale_principle(w, is_excitatory)

    return w


@jit
def initialize_input_weights(key: jax.random.PRNGKey, params: NetworkParams) -> jnp.ndarray:
    """Initialize input weights with topographic organization"""
    # Create gaussian receptive fields for population coding
    input_centers = jnp.linspace(0, 1, params.NUM_INPUTS)
    neuron_preferences = jnp.linspace(0, 1, params.NUM_NEURONS)

    # Gaussian tuning curves
    sigma = 0.2  # Receptive field width
    distances = jnp.abs(input_centers[None, :] - neuron_preferences[:, None])
    w_in = jnp.exp(-(distances**2) / (2 * sigma**2))

    # Add noise for diversity
    noise = random.normal(key, w_in.shape) * 0.1
    w_in = w_in + noise
    w_in = jnp.maximum(w_in, 0.0)  # Non-negative

    # Normalize so each neuron receives similar total input
    w_in = w_in / (jnp.sum(w_in, axis=1, keepdims=True) + 1e-8)

    return w_in.T  # Shape: (NUM_INPUTS, NUM_NEURONS)


@jit
def apply_dale_principle(w: jnp.ndarray, is_excitatory: jnp.ndarray) -> jnp.ndarray:
    """Apply Dale's principle based on pre-synaptic neuron type"""
    E_pre = is_excitatory[:, None]
    return jnp.where(E_pre, jnp.abs(w), -jnp.abs(w))


@jit
def encode_gradient_population(
    gradient: float, params: NetworkParams, key: jax.random.PRNGKey
) -> jnp.ndarray:
    """Encode scalar gradient into population code"""
    # Gaussian tuning curves centered at different gradient values
    centers = jnp.linspace(0, 1, params.NUM_INPUTS)
    sigma = 0.15

    # Compute activation for each input neuron
    activations = jnp.exp(-((gradient - centers) ** 2) / (2 * sigma**2))

    # Add noise for robustness
    noise = random.normal(key, activations.shape) * 0.1
    activations = jnp.maximum(activations + noise, 0.0)

    # Normalize to maintain consistent input magnitude
    activations = activations / (jnp.sum(activations) + 1e-8) * params.NUM_INPUTS * 0.5

    return activations


@jit
def neuron_step_enhanced(
    state: EnhancedNetworkState, params: NetworkParams, key: jax.random.PRNGKey
) -> EnhancedNetworkState:
    """Enhanced neuron dynamics with homeostasis"""
    keys = random.split(key, 3)

    # Update refractory counters
    refractory_new = jnp.maximum(0, state.refractory - 1.0)

    # Synaptic current decay (different time constants for E/I)
    syn_current_e_new = state.syn_current_e * jnp.exp(-1.0 / params.TAU_SYN_E)
    syn_current_i_new = state.syn_current_i * jnp.exp(-1.0 / params.TAU_SYN_I)

    # Add incoming spikes to synaptic currents
    spike_input = state.spike.astype(float)
    jnp.dot(state.w, spike_input)

    # Separate E/I currents based on pre-synaptic type
    E_mask = state.is_excitatory[:, None]
    exc_input = jnp.where(E_mask, state.w, 0.0) @ spike_input
    inh_input = jnp.where(~E_mask, state.w, 0.0) @ spike_input

    syn_current_e_new = syn_current_e_new + jnp.abs(exc_input)
    syn_current_i_new = syn_current_i_new + jnp.abs(inh_input)

    # External input from population code
    i_ext = jnp.dot(state.w_in.T, state.input_population)

    # Total current with separate E/I contributions
    i_total = i_ext + syn_current_e_new + syn_current_i_new

    # Add noise (reduced compared to original)
    noise = random.normal(keys[0], state.v.shape) * 1.0
    i_total = i_total + noise

    # Membrane potential dynamics
    dv = (-state.v + params.V_REST + i_total) / params.TAU_V
    v_new = state.v + dv

    # Homeostatic threshold adaptation
    effective_threshold = params.V_THRESHOLD + state.threshold_adaptation

    # Spike generation with adaptive threshold
    can_spike = refractory_new == 0
    spike_new = (v_new >= effective_threshold) & can_spike

    # Reset and refractory
    v_new = jnp.where(spike_new, params.V_RESET, v_new)
    refractory_new = jnp.where(spike_new, params.REFRACTORY_TIME, refractory_new)

    # Update traces
    trace_fast_new = state.trace_fast * jnp.exp(-1.0 / params.TAU_FAST) + spike_new * 1.0
    trace_slow_new = state.trace_slow * jnp.exp(-1.0 / params.TAU_SLOW) + spike_new * 0.5

    # Update firing rate estimate (exponential moving average)
    alpha = 1.0 / params.HOMEOSTATIC_TAU
    firing_rate_new = state.firing_rate * (1 - alpha) + spike_new * 1000.0 * alpha  # Convert to Hz

    # Update threshold adaptation (homeostatic plasticity)
    rate_error = firing_rate_new - params.TARGET_RATE
    threshold_adaptation_new = state.threshold_adaptation + params.THRESHOLD_ADAPT_RATE * rate_error
    threshold_adaptation_new = jnp.clip(threshold_adaptation_new, -10.0, 10.0)

    return state._replace(
        v=v_new,
        spike=spike_new,
        refractory=refractory_new,
        syn_current_e=syn_current_e_new,
        syn_current_i=syn_current_i_new,
        trace_fast=trace_fast_new,
        trace_slow=trace_slow_new,
        firing_rate=firing_rate_new,
        threshold_adaptation=threshold_adaptation_new,
    )


@jit
def compute_eligibility_trace_enhanced(
    state: EnhancedNetworkState, params: NetworkParams
) -> jnp.ndarray:
    """Enhanced eligibility trace with weight-dependent modulation"""
    # Basic STDP as before
    pre_trace_expanded = state.trace_fast[:, None]
    post_spike_expanded = state.spike[None, :]
    pre_spike_expanded = state.spike[:, None]
    post_trace_expanded = state.trace_fast[None, :]

    ltp = pre_trace_expanded * post_spike_expanded.astype(float)
    ltd = pre_spike_expanded.astype(float) * post_trace_expanded

    # Weight-dependent STDP (BCM-like)
    # Stronger weights have reduced LTP, increased LTD
    w_normalized = jnp.abs(state.w) / (jnp.abs(state.w).max() + 1e-8)
    ltp_factor = 1.0 - 0.5 * w_normalized  # Less LTP for strong weights
    ltd_factor = 1.0 + 0.5 * w_normalized  # More LTD for strong weights

    stdp = jnp.where(
        state.connection_mask,
        params.A_PLUS * ltp * ltp_factor - params.A_MINUS * ltd * ltd_factor,
        0.0,
    )

    # Decay eligibility traces
    decay = jnp.exp(-1.0 / params.TAU_ELIGIBILITY)
    new_eligibility = state.eligibility_trace * decay + stdp

    return new_eligibility


@jit
def update_metaplasticity(
    state: EnhancedNetworkState, params: NetworkParams
) -> EnhancedNetworkState:
    """Update metaplastic variables"""
    # Update synapse age (increases with activity)
    spike_product = jnp.outer(state.spike, state.spike).astype(float)
    age_increment = spike_product * state.connection_mask * 0.01
    new_age = state.synapse_age + age_increment

    # Update learning rate factors based on synapse age and weight magnitude
    # Young synapses learn faster, old synapses are more stable
    age_factor = jnp.exp(-new_age / params.TAU_METAPLASTICITY)

    # Weight magnitude factor - very weak or very strong synapses learn less
    w_magnitude = jnp.abs(state.w)
    optimal_weight = 0.5  # Optimal weight magnitude
    weight_factor = jnp.exp(-((w_magnitude - optimal_weight) ** 2) / 0.1)

    new_learning_rate_factor = age_factor * weight_factor

    return state._replace(synapse_age=new_age, learning_rate_factor=new_learning_rate_factor)


@jit
def three_factor_update_enhanced(
    state: EnhancedNetworkState, dopamine_modulation: float, params: NetworkParams
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Enhanced three-factor learning with metaplasticity"""
    # Base weight update
    base_dw = dopamine_modulation * state.eligibility_trace

    # Apply metaplastic learning rate modulation
    effective_lr = params.BASE_LEARNING_RATE * state.learning_rate_factor
    dw = effective_lr * base_dw

    # Weight decay for stability
    decay = params.WEIGHT_DECAY * state.w
    dw = dw - decay

    # Only update existing connections
    dw = jnp.where(state.connection_mask, dw, 0.0)

    # Update weights
    updated_weights = state.w + dw

    # Maintain Dale's principle
    updated_weights = apply_dale_principle(updated_weights, state.is_excitatory)

    # Enforce bounds with soft saturation
    E_mask = state.is_excitatory[:, None]
    max_e_weight = 2.0
    max_i_weight = 2.0

    # Soft bounds using tanh
    updated_weights = jnp.where(
        E_mask,
        max_e_weight * jnp.tanh(updated_weights / max_e_weight),
        -max_i_weight * jnp.tanh(-updated_weights / max_i_weight),
    )

    # Zero out non-existent connections
    updated_weights = jnp.where(state.connection_mask, updated_weights, 0.0)

    # Decay eligibility traces
    new_eligibility = state.eligibility_trace * 0.9

    return updated_weights, new_eligibility


@jit
def make_decision_enhanced(
    state: EnhancedNetworkState, key: jax.random.PRNGKey
) -> Tuple[int, EnhancedNetworkState]:
    """Enhanced decision making with population decoding"""
    # Get motor command from excitatory neurons only
    exc_activity = jnp.where(state.is_excitatory, state.spike.astype(float), 0.0)
    motor_input = jnp.dot(exc_activity, state.w_out)

    # Add exploration noise
    noise = random.normal(key, motor_input.shape) * 0.3
    motor_input = motor_input + noise

    # Softmax with temperature
    temperature = 2.0
    action_probs = jax.nn.softmax(motor_input / temperature)

    # Sample action
    action = random.categorical(key, jnp.log(action_probs + 1e-8))

    # Update motor traces
    motor_spikes = jnp.zeros(4, dtype=bool).at[action].set(True)
    motor_trace = state.motor_trace * 0.95 + motor_spikes * 1.0

    return action, state._replace(motor_spikes=motor_spikes, motor_trace=motor_trace)


# Import environment and data handling from original
from phase04_research import (
    OptimizedGridWorld,
    compute_dopamine_modulation,
    update_dopamine_with_rpe,
)


def run_enhanced_episode(seed: int, progress_callback=None) -> Dict:
    """Run episode with enhanced neural dynamics"""
    key = random.PRNGKey(seed)
    network_key, episode_key = random.split(key)

    params = NetworkParams()
    state = initialize_enhanced_network(network_key, params)
    env = OptimizedGridWorld(seed)

    # Pre-allocate arrays
    max_steps = env.max_steps
    gradients = np.zeros(max_steps, dtype=np.float32)
    positions = np.zeros((max_steps, 2), dtype=np.int32)
    actions = np.zeros(max_steps, dtype=np.int8)
    rewards = np.zeros(max_steps, dtype=np.float32)

    # Sample neural dynamics
    sample_interval = 100
    num_samples = max_steps // sample_interval
    spike_trains = np.zeros((num_samples, params.NUM_NEURONS), dtype=bool)
    membrane_potentials = np.zeros((num_samples, params.NUM_NEURONS), dtype=np.float32)
    firing_rates = np.zeros((num_samples, params.NUM_NEURONS), dtype=np.float32)
    threshold_adaptations = np.zeros((num_samples, params.NUM_NEURONS), dtype=np.float32)

    # Track performance
    reward_steps = []
    proximity_reward_steps = []
    unique_positions = set()

    episode_start_time = time.time()

    for step in range(max_steps):
        # Get observation
        gradient = env.get_observation()
        gradients[step] = gradient
        positions[step] = env.agent_pos

        if step % 100 == 0:
            unique_positions.add(tuple(int(x) for x in env.agent_pos))

        # Encode gradient into population code
        episode_key, encoding_key = random.split(episode_key)
        input_population = encode_gradient_population(gradient, params, encoding_key)
        state = state._replace(input_population=input_population)

        # Neural dynamics step
        episode_key, neuron_key = random.split(episode_key)
        state = neuron_step_enhanced(state, params, neuron_key)

        # Update metaplasticity
        state = update_metaplasticity(state, params)

        # Compute eligibility traces
        new_eligibility = compute_eligibility_trace_enhanced(state, params)
        state = state._replace(eligibility_trace=new_eligibility)

        # Make decision
        episode_key, decision_key = random.split(episode_key)
        action, state = make_decision_enhanced(state, decision_key)
        actions[step] = action

        # Step environment
        reward = env.step(action)
        rewards[step] = reward

        # Update value estimate
        state = state._replace(prev_value=state.value_estimate)

        # Update dopamine based on RPE
        new_dopamine, new_value, td_error = update_dopamine_with_rpe(state, reward, params)
        state = state._replace(dopamine=new_dopamine, value_estimate=new_value)

        # Compute dopamine modulation
        dopamine_modulation = compute_dopamine_modulation(state.dopamine, params.BASELINE_DOPAMINE)

        # Three-factor learning with metaplasticity
        new_w, new_eligibility = three_factor_update_enhanced(
            state._replace(eligibility_trace=new_eligibility), dopamine_modulation, params
        )
        state = state._replace(w=new_w, eligibility_trace=new_eligibility)

        # Track rewards
        if reward >= 1.0:
            reward_steps.append(step)
        elif reward > 0:
            proximity_reward_steps.append(step)

        # Sample neural dynamics
        if step % sample_interval == 0:
            sample_idx = step // sample_interval
            spike_trains[sample_idx] = np.array(state.spike)
            membrane_potentials[sample_idx] = np.array(state.v)
            firing_rates[sample_idx] = np.array(state.firing_rate)
            threshold_adaptations[sample_idx] = np.array(state.threshold_adaptation)

        # Progress callback
        if progress_callback and step % 1000 == 0:
            progress_callback(step, max_steps, len(reward_steps), len(proximity_reward_steps))

    episode_time = time.time() - episode_start_time

    return {
        "metadata": {
            "seed": seed,
            "episode_time_seconds": episode_time,
            "rewards_collected": len(reward_steps),
            "proximity_rewards_collected": len(proximity_reward_steps),
            "unique_positions_visited": len(unique_positions),
            "network_type": "enhanced",
            "enhancements": [
                "connection-type specific weights",
                "homeostatic plasticity",
                "metaplasticity",
                "population coding",
                "separate E/I synaptic dynamics",
            ],
        },
        "trajectory": {
            "positions": positions,
            "actions": actions,
            "gradients": gradients,
            "rewards": rewards,
            "reward_steps": np.array(reward_steps),
            "proximity_reward_steps": np.array(proximity_reward_steps),
        },
        "neural_dynamics": {
            "spike_trains": spike_trains,
            "membrane_potentials": membrane_potentials,
            "firing_rates": firing_rates,
            "threshold_adaptations": threshold_adaptations,
            "sample_interval": sample_interval,
        },
        "network_properties": {
            "is_excitatory": np.array(state.is_excitatory),
            "connection_mask": np.array(state.connection_mask),
            "final_weights": np.array(state.w),
            "final_learning_rates": np.array(state.learning_rate_factor),
            "final_synapse_ages": np.array(state.synapse_age),
        },
    }


def main():
    """Run enhanced version for comparison"""
    parser = argparse.ArgumentParser(
        description="Phase 0.4 Enhanced - Biologically Plausible Dynamics"
    )

    parser.add_argument(
        "-n", "--num-episodes", type=int, default=1, help="Number of episodes to run"
    )
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--progress", action="store_true", default=True, help="Show progress bars")

    args = parser.parse_args()

    print("🧬 Running Enhanced Neural Network")
    print("  Features:")
    print("    ✓ Connection-type specific initialization")
    print("    ✓ Homeostatic plasticity")
    print("    ✓ Metaplasticity and weight-dependent learning")
    print("    ✓ Population coding for inputs")
    print("    ✓ Separate E/I synaptic dynamics\n")

    def progress_callback(step, max_steps, rewards, proximity):
        if not args.progress:
            return

        progress_pct = (step / max_steps) * 100
        bar_width = 30
        filled = int(bar_width * progress_pct / 100)
        bar = "█" * filled + "░" * (bar_width - filled)

        line = f"\r  {bar} {progress_pct:5.1f}% | R: {rewards} | P: {proximity}"

        print(line, end="", flush=True)

    # Run episodes
    results = []
    for i in range(args.num_episodes):
        seed = args.seed + i
        print(f"\nEpisode {i + 1}/{args.num_episodes} (Seed {seed})")

        result = run_enhanced_episode(seed, progress_callback if args.progress else None)

        if args.progress:
            print()

        print(
            f"  ✅ Rewards: {result['metadata']['rewards_collected']}, "
            f"Proximity: {result['metadata']['proximity_rewards_collected']}, "
            f"Coverage: {result['metadata']['unique_positions_visited']}"
        )

        results.append(result)

    # Summary
    if len(results) > 1:
        avg_rewards = np.mean([r["metadata"]["rewards_collected"] for r in results])
        avg_proximity = np.mean([r["metadata"]["proximity_rewards_collected"] for r in results])
        avg_coverage = np.mean([r["metadata"]["unique_positions_visited"] for r in results])

        print("\n📊 Summary:")
        print(f"  Average rewards: {avg_rewards:.1f}")
        print(f"  Average proximity: {avg_proximity:.1f}")
        print(f"  Average coverage: {avg_coverage:.0f}")


if __name__ == "__main__":
    main()
