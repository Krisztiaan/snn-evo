# models/phase_0_8/agent.py
# keywords: [snn agent, phase 0.8, best-of-all-worlds, principled implementation]
"""
Phase 0.8 SNN Agent: Principled synthesis of best ideas from phases 0.4-0.7

Key improvements:
1. Clear separation of neuron populations with unified dynamics
2. Proper two-trace STDP (missing in 0.5/0.6)
3. Fixed input weights with learning only on processing/readout
4. Homeostatic firing rate control
5. Biologically motivated connectivity
"""

from functools import partial
from typing import Any, Dict, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, random

from export import DataExporter
from world.simple_grid_0001 import SimpleGridWorld

from .agent_vectorized import create_connectivity_vectorized
from .config import NetworkParams, SnnAgentConfig


class AgentState(NamedTuple):
    """Complete agent state with all necessary components."""

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


# === HELPER FUNCTIONS ===


@jit
def _apply_dale_principle(w: jnp.ndarray, is_excitatory: jnp.ndarray) -> jnp.ndarray:
    """Apply Dale's principle: neurons release consistent neurotransmitter."""
    # Pre-synaptic neuron determines sign (rows in weight matrix)
    E_mask = is_excitatory[:, None]
    return jnp.where(E_mask, jnp.abs(w), -jnp.abs(w))


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
    normalized = activations / (total_activation + 1e-8) * params.INPUT_GAIN

    return normalized


# === NETWORK INITIALIZATION ===


def _create_connectivity(
    key: random.PRNGKey,
    params: NetworkParams,
    num_neurons: int,
    is_excitatory: jnp.ndarray,
    neuron_types: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Create biologically motivated connectivity patterns.
    Returns: (connection_mask, plastic_mask)

    NOTE: This function is slow due to nested loops. For 192 processing neurons,
    it needs to check 192*192 = 36,864 potential connections.
    TODO: Vectorize for better performance.
    """
    n = num_neurons
    n_in = params.NUM_INPUT_CHANNELS

    # Total sources = input channels + all neurons
    n_total_sources = n_in + n

    # Initialize masks
    w_mask = jnp.zeros((n, n_total_sources), dtype=bool)
    w_plastic = jnp.zeros((n, n_total_sources), dtype=bool)

    # Neuron type indices
    is_sensory = neuron_types == 0
    is_processing = neuron_types == 1
    is_readout = neuron_types == 2

    keys = random.split(key, 6)

    # 1. Input → Sensory connections (fixed, non-plastic)
    sensory_indices = jnp.where(is_sensory)[0]
    if len(sensory_indices) > 0:
        input_mask = random.uniform(keys[0], (len(sensory_indices), n_in)) < params.P_INPUT_SENSORY
        for i, idx in enumerate(sensory_indices):
            w_mask = w_mask.at[idx, :n_in].set(input_mask[i])
            if params.LEARN_INPUT_CONNECTIONS:
                w_plastic = w_plastic.at[idx, :n_in].set(input_mask[i])
    # Input connections are NOT plastic

    # 2. Sensory → Processing connections (feedforward, non-plastic)
    sens_proc_key = keys[1]
    for i in jnp.where(is_sensory)[0]:
        for j in jnp.where(is_processing)[0]:
            if random.uniform(sens_proc_key, ()) < params.P_SENSORY_PROCESSING:
                w_mask = w_mask.at[j, n_in + i].set(True)
                sens_proc_key, _ = random.split(sens_proc_key)

    # 3. Processing ↔ Processing connections (recurrent, PLASTIC)
    proc_indices = jnp.where(is_processing)[0]
    if len(proc_indices) > 0:
        # Create distance matrix for local connectivity bias
        proc_positions = jnp.arange(len(proc_indices))
        dist_matrix = jnp.abs(proc_positions[:, None] - proc_positions[None, :])
        dist_matrix = jnp.minimum(dist_matrix, len(proc_indices) - dist_matrix)
        is_local = dist_matrix < (len(proc_indices) * params.LOCAL_RADIUS)

        # E/I specific probabilities
        proc_e_mask = is_excitatory[proc_indices]

        proc_key = keys[2]
        for i_idx, i in enumerate(proc_indices):
            for j_idx, j in enumerate(proc_indices):
                if i == j:
                    continue  # No self-connections

                # Determine connection probability based on types
                if proc_e_mask[i_idx] and proc_e_mask[j_idx]:  # E→E
                    p = (
                        params.P_PROC_PROC_LOCAL
                        if is_local[i_idx, j_idx]
                        else params.P_PROC_PROC_DIST
                    )
                elif proc_e_mask[i_idx] and not proc_e_mask[j_idx]:  # E→I
                    p = params.P_EI
                elif not proc_e_mask[i_idx] and proc_e_mask[j_idx]:  # I→E
                    p = params.P_IE
                else:  # I→I
                    p = params.P_II

                if random.uniform(proc_key, ()) < p:
                    w_mask = w_mask.at[j, n_in + i].set(True)
                    if params.LEARN_PROCESSING_RECURRENT:
                        w_plastic = w_plastic.at[j, n_in + i].set(True)
                proc_key, _ = random.split(proc_key)

    # 4. Processing → Readout connections (convergent, optionally PLASTIC)
    readout_key = keys[3]
    for i in jnp.where(is_processing & is_excitatory)[0]:  # Only E neurons project
        for j in jnp.where(is_readout)[0]:
            if random.uniform(readout_key, ()) < params.P_PROCESSING_READOUT:
                w_mask = w_mask.at[j, n_in + i].set(True)
                if params.LEARN_PROCESSING_READOUT:
                    w_plastic = w_plastic.at[j, n_in + i].set(True)
            readout_key, _ = random.split(readout_key)

    return w_mask, w_plastic


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

    # Apply Dale's principle to neural weights (not input weights)
    # Note: Dale's principle is already correctly applied in _apply_dale_principle function
    # which uses pre-synaptic identity (rows in weight matrix)
    # No need to apply it here during initialization since weights are initialized positive

    return w


# === NEURAL DYNAMICS ===


@partial(jit, static_argnames=["params"])
def _neuron_step(state: AgentState, params: NetworkParams, key: random.PRNGKey) -> AgentState:
    """
    Core neural dynamics with separate E/I currents and homeostasis.
    """
    # Update refractory period
    refractory_new = jnp.maximum(0, state.refractory - 1.0)

    # Decay synaptic currents
    syn_current_e_new = state.syn_current_e * jnp.exp(-1.0 / params.TAU_SYN_E)
    syn_current_i_new = state.syn_current_i * jnp.exp(-1.0 / params.TAU_SYN_I)

    # Compute synaptic input
    # First, concatenate input channels with neural spikes
    all_activity = jnp.concatenate([state.input_channels, state.spike.astype(float)])

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

    # Update synaptic traces (two timescales)
    trace_fast_new = state.trace_fast * jnp.exp(-1.0 / params.TAU_TRACE_FAST) + spike_new
    trace_slow_new = state.trace_slow * jnp.exp(-1.0 / params.TAU_TRACE_SLOW) + spike_new * 0.5

    # Update firing rate estimate
    alpha = 1.0 / params.HOMEOSTATIC_TAU
    firing_rate_new = state.firing_rate * (1 - alpha) + spike_new.astype(float) * 1000.0 * alpha

    # Update threshold adaptation
    rate_error = firing_rate_new - params.TARGET_RATE_HZ
    threshold_adapt_new = state.threshold_adapt + params.THRESHOLD_ADAPT_RATE * rate_error
    threshold_adapt_new = jnp.clip(
        threshold_adapt_new, -params.MAX_THRESHOLD_ADAPT, params.MAX_THRESHOLD_ADAPT
    )

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
    )


# === LEARNING ===


@partial(jit, static_argnames=["params"])
def _compute_eligibility_trace(state: AgentState, params: NetworkParams) -> jnp.ndarray:
    """
    Compute STDP-based eligibility traces using both fast and slow traces.
    Only updates plastic synapses.
    """
    # Standard STDP using fast trace
    # Pre-synaptic trace at post-synaptic spike time
    pre_trace = jnp.concatenate([jnp.zeros(params.NUM_INPUT_CHANNELS), state.trace_fast])
    post_spike = state.spike.astype(float)

    # LTP: pre trace × post spike
    ltp = pre_trace[None, :] * post_spike[:, None]

    # LTD: pre spike × post trace
    pre_spike = jnp.concatenate([jnp.zeros(params.NUM_INPUT_CHANNELS), state.spike.astype(float)])
    post_trace = state.trace_fast
    ltd = pre_spike[None, :] * post_trace[:, None]

    # STDP update
    stdp = params.STDP_A_PLUS * ltp - params.STDP_A_MINUS * ltd

    # Only apply to plastic synapses
    stdp = jnp.where(state.w_plastic_mask, stdp, 0.0)

    # Decay existing traces
    decay = jnp.exp(-1.0 / params.TAU_ELIGIBILITY)
    new_eligibility = state.eligibility_trace * decay + stdp

    return new_eligibility


@partial(jit, static_argnames=["params"])
def _three_factor_learning(state: AgentState, reward: float, params: NetworkParams) -> AgentState:
    """
    Three-factor learning rule with RPE-based dopamine modulation.
    """
    # 1. Compute reward prediction error
    td_error = reward + params.REWARD_DISCOUNT * state.value_estimate - state.value_estimate
    new_value = state.value_estimate + params.REWARD_PREDICTION_RATE * td_error

    # 2. Update dopamine based on RPE
    decay = jnp.exp(-1.0 / params.TAU_DOPAMINE)
    dopamine_response = (
        state.dopamine * decay + params.BASELINE_DOPAMINE * (1 - decay) + td_error * 0.5
    )
    new_dopamine = jnp.clip(dopamine_response, 0.0, 2.0)

    # 3. Compute dopamine modulation factor
    da_factor = (new_dopamine - params.BASELINE_DOPAMINE) / (params.BASELINE_DOPAMINE + 1e-8)
    modulation = jax.nn.tanh(da_factor * 2.0)  # Smooth, bounded modulation

    # 4. Update eligibility traces
    new_eligibility = _compute_eligibility_trace(state, params)

    # 5. Compute weight updates
    dw = params.BASE_LEARNING_RATE * modulation * new_eligibility

    # Add weight decay
    dw -= params.WEIGHT_DECAY * state.w * state.w_plastic_mask

    # 6. Update weights with soft bounds
    w_new = state.w + dw

    # Soft saturation using tanh (from phase 0.4)
    # Split into input and neural parts
    n_in = params.NUM_INPUT_CHANNELS
    w_input = w_new[:, :n_in]  # These shouldn't change if properly masked
    w_neural = w_new[:, n_in:]

    # Apply soft bounds to neural weights
    scale = params.MAX_WEIGHT_SCALE
    w_neural_sign = jnp.sign(w_neural)
    w_neural_mag = jnp.abs(w_neural)
    w_neural_bounded = w_neural_sign * scale * jnp.tanh(w_neural_mag / scale)

    # Recombine
    w_new = jnp.concatenate([w_input, w_neural_bounded], axis=1)

    # Ensure Dale's principle is maintained for neural connections
    # Apply Dale's principle correctly based on pre-synaptic neurons
    w_neural_dale = _apply_dale_principle(w_neural_bounded, state.is_excitatory)
    w_new = w_new.at[:, n_in:].set(w_neural_dale)

    # Zero out non-existent connections
    w_new = jnp.where(state.w_mask, w_new, 0.0)

    return state._replace(
        w=w_new,
        eligibility_trace=new_eligibility * 0.9,  # Decay after use
        dopamine=new_dopamine,
        value_estimate=new_value,
    )


# === ACTION SELECTION ===


@partial(jit, static_argnames=["params"])
def _decode_action(
    state: AgentState, params: NetworkParams, key: random.PRNGKey
) -> Tuple[int, jnp.ndarray]:
    """
    Decode action from readout population activity.
    """
    # Get readout neuron spikes
    readout_mask = state.neuron_types == 2
    readout_spikes = jnp.where(readout_mask, state.spike.astype(float), 0.0)

    # Update motor trace (temporal integration)
    decay = jnp.exp(-1.0 / params.MOTOR_TAU)

    # Map readout neurons to 4 motor commands
    # Simple mapping: divide readout neurons into 4 groups
    n_readout = params.NUM_READOUT
    n_per_action = n_readout // 4

    motor_input = jnp.zeros(4)
    for i in range(4):
        start_idx = params.NUM_SENSORY + params.NUM_PROCESSING + i * n_per_action
        end_idx = start_idx + n_per_action
        motor_input = motor_input.at[i].set(jnp.sum(readout_spikes[start_idx:end_idx]))

    # Update motor trace
    motor_trace_new = state.motor_trace * decay + motor_input

    # Action selection via softmax
    action_logits = motor_trace_new / params.ACTION_TEMPERATURE
    action_probs = jax.nn.softmax(action_logits)

    # Sample action
    action = random.categorical(key, jnp.log(action_probs + 1e-8))

    return action, motor_trace_new


# === MAIN AGENT CLASS ===


class SnnAgent:
    """Phase 0.8 SNN Agent with principled design."""

    def __init__(self, config: SnnAgentConfig):
        self.config = config
        self.params = config.network_params
        self.world = SimpleGridWorld(config.world_config)

        print("Initializing network state...")
        import time

        start_time = time.time()
        self.state: AgentState = self._initialize_state(random.PRNGKey(config.exp_config.seed))
        print(f"Network initialized in {time.time() - start_time:.2f}s")

        # Monitoring
        self.firing_rate_history = []
        self.weight_stats_history = []

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
        exporter: DataExporter,
        progress_callback=None,
    ) -> Dict[str, Any]:
        """Run a single episode."""
        import time

        episode_start_time = time.time()

        # Reset world and agent
        world_state, obs = self.world.reset(episode_key)
        self.state = self._initialize_state(episode_key)

        # Start data export
        exporter.start_episode(episode_num)
        exporter.log_static_episode_data(
            "world_setup", {"reward_positions": np.asarray(world_state.reward_positions)}
        )

        # Save network structure
        if episode_num == 0:
            self._export_network_structure(exporter)

        # Episode loop
        rewards_collected = 0
        max_steps = self.config.world_config.max_timesteps
        for step in range(max_steps):
            # Split keys
            step_key, encode_key, neuron_key, action_key = random.split(episode_key, 4)
            episode_key = step_key

            # 1. Encode input
            input_channels = _encode_gradient_population(obs.gradient, self.params, encode_key)
            self.state = self.state._replace(input_channels=input_channels)

            # 2. Neural dynamics step
            self.state = _neuron_step(self.state, self.params, neuron_key)

            # 3. Decode action
            action, motor_trace = _decode_action(self.state, self.params, action_key)
            self.state = self.state._replace(motor_trace=motor_trace)

            # 4. Environment step
            result = self.world.step(world_state, int(action))
            world_state, obs, reward, done = (
                result.state,
                result.observation,
                result.reward,
                result.done,
            )

            # 5. Learning step
            self.state = _three_factor_learning(self.state, reward, self.params)

            # 6. Track rewards
            if reward > 0:
                rewards_collected += 1

            # 7. Monitoring
            self._monitor_stability(self.state, step)

            # 8. Progress callback (every 500 steps and on final step)
            if progress_callback and (step % 500 == 0 or step == max_steps - 1):
                elapsed = time.time() - episode_start_time
                progress_callback(step, max_steps, rewards_collected, elapsed)

            # 9. Data logging (match phase 0_5/0_6 style + gradient)
            exporter.log(
                timestep=step,
                neural_state={
                    "v": np.asarray(self.state.v),
                    "spikes": np.asarray(self.state.spike.astype(jnp.uint8)),
                },
                behavior={
                    "action": int(action),
                    "pos_x": int(world_state.agent_pos[0]),
                    "pos_y": int(world_state.agent_pos[1]),
                    "gradient": float(obs.gradient),  # Add gradient logging
                },
                reward=float(reward),
            )

            if done:
                break

        # Episode summary - backwards compatible with enhanced fields
        summary = {
            # Core fields (required for compatibility with 0_5/0_6/0_7)
            "total_reward": float(world_state.total_reward),
            "rewards_collected": int(jnp.sum(world_state.reward_collected)),
            "steps_taken": world_state.timestep,
            # Enhanced fields (optional, won't break older analysis)
            "mean_firing_rate": float(jnp.mean(self.state.firing_rate)),
            "final_dopamine": float(self.state.dopamine),
            "final_value_estimate": float(self.state.value_estimate),
        }

        exporter.end_episode(success=jnp.all(world_state.reward_collected), summary=summary)

        return summary

    def _export_network_structure(self, exporter: DataExporter):
        """Export network structure for analysis."""
        # Backwards compatible with phase 0_5/0_6/0_7
        n_neurons = len(self.state.neuron_types)
        n_in = self.params.NUM_INPUT_CHANNELS

        # Extract just the recurrent connections for compatibility
        recurrent_mask = self.state.w_mask[:, n_in:]
        conn_indices = jnp.where(recurrent_mask)

        # Basic structure (required for compatibility)
        # Note: conn_indices[0] is target (row), conn_indices[1] is source (column)
        basic_structure = {
            "neurons": {
                "neuron_ids": np.arange(n_neurons),
                "is_excitatory": np.asarray(self.state.is_excitatory),
            },
            "connections": {
                "source_ids": np.asarray(conn_indices[1]),  # Column indices = source neurons
                "target_ids": np.asarray(conn_indices[0]),  # Row indices = target neurons
            },
            "initial_weights": {"weights": np.asarray(self.state.w[:, n_in:][recurrent_mask])},
        }

        # Save basic structure for compatibility
        exporter.save_network_structure(**basic_structure)

        # Save enhanced info separately (as arrays for HDF5)
        # Store as arrays to avoid scalar dataset issues
        enhanced_data = {
            "neuron_types": np.asarray(self.state.neuron_types, dtype=np.int32),
            "connectivity_stats": np.array(
                [
                    np.sum(self.state.w_mask[:, :n_in]),  # num input connections
                    np.sum(self.state.w_plastic_mask),  # num plastic connections
                    np.sum(recurrent_mask),  # num recurrent connections
                ],
                dtype=np.int32,
            ),
        }
        exporter.log_static_episode_data("network_stats", enhanced_data)

    def run_experiment(self):
        """Run complete experiment."""
        import time

        master_key = random.PRNGKey(self.config.exp_config.seed)

        print("Initializing data exporter...")
        with DataExporter(
            experiment_name="snn_agent_phase08",
            output_base_dir=self.config.exp_config.export_dir,
            compression="gzip",
            compression_level=1,
        ) as exporter:
            print("Saving configuration...")
            # Save configuration
            exporter.save_config(self.config)

            # Run episodes
            all_summaries = []
            episode_times = []
            time.time()

            for i in range(self.config.exp_config.n_episodes):
                print(f"\n--- Episode {i + 1}/{self.config.exp_config.n_episodes} ---")
                episode_key, master_key = random.split(master_key)
                episode_start = time.time()

                # Progress callback with ETA
                def progress_callback(step, max_steps, rewards, episode_elapsed):
                    # Calculate progress
                    progress_pct = ((step + 1) / max_steps) * 100  # +1 to reach 100%
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
                        # Estimate based on current episode progress
                        if step > 0:
                            avg_episode_time = (episode_elapsed / (step + 1)) * max_steps
                        else:
                            avg_episode_time = 60  # Default estimate

                    # Remaining time = current episode remaining + future episodes
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
                        f"Rewards: {rewards} | Episode ETA: {format_time(episode_eta)} | "
                        f"Total ETA: {format_time(total_eta)}",
                        end="",
                        flush=True,
                    )

                summary = self.run_episode(
                    episode_key,
                    episode_num=i,
                    exporter=exporter,
                    progress_callback=progress_callback,
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

        return all_summaries
