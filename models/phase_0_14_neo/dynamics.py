# keywords: [neo dynamics, neural dynamics, action decoding, input encoding]
"""Neural dynamics for Phase 0.14 Neo agent."""

from typing import Tuple

import jax
import jax.numpy as jnp
from jax import random, lax

from .state import NeoAgentState
from .config import DynamicsConfig, InputConfig, NetworkConfig


def neuron_dynamics_step(
    state: NeoAgentState,
    key: random.PRNGKey,
    config: DynamicsConfig
) -> NeoAgentState:
    """Single step of neural dynamics with LIF neurons (fully vectorized)."""
    n_neurons = state.v.shape[0]
    noise_key, _ = random.split(key)

    # 1. Synaptic dynamics (exponential decay)
    tau_e_decay = jnp.exp(-1.0 / config.tau_syn_e)
    tau_i_decay = jnp.exp(-1.0 / config.tau_syn_i)

    syn_current_e = state.syn_current_e * tau_e_decay
    syn_current_i = state.syn_current_i * tau_i_decay

    # 2. Add spike-driven synaptic input (vectorized)
    # Combine input and recurrent spikes into a single pre-synaptic activity vector
    spike_input = jnp.concatenate([state.input_buffer, state.spike.astype(jnp.float32)])

    # --- Optimization: Use pre-separated weight matrices ---
    # `w_exc` and `w_inh` are now part of NeoAgentState
    syn_current_e += state.w_exc @ spike_input
    syn_current_i += state.w_inh @ spike_input

    # 3. Update membrane potential
    v_decay = jnp.exp(-1.0 / config.tau_v)
    v = state.v * v_decay + config.v_rest * (1 - v_decay)
    v += syn_current_e - syn_current_i

    noise = random.normal(noise_key, (n_neurons,)) * config.noise_scale
    v += config.baseline_current + noise

    # Apply refractory period
    v = jnp.where(state.refractory > 0, config.v_reset, v)

    # 4. Check for spikes with adaptive threshold
    effective_threshold = config.v_threshold + state.threshold_adapt
    spike = (v >= effective_threshold) & (state.refractory == 0)

    # 5. Reset spiking neurons
    v = jnp.where(spike, config.v_reset, v)

    # 6. Update refractory period
    refractory = jnp.where(spike, config.refractory_time, jnp.maximum(0, state.refractory - 1))

    # 7. Update traces
    tau_pre_decay = jnp.exp(-1.0 / 20.0)
    tau_post_decay = jnp.exp(-1.0 / 20.0)
    trace_pre = state.trace_pre * tau_pre_decay + spike.astype(jnp.float32)
    trace_post = state.trace_post * tau_post_decay + spike.astype(jnp.float32)

    # 8. Update firing rate (slow moving average)
    rate_tau = 1000.0
    rate_alpha = 1.0 / rate_tau
    firing_rate = state.firing_rate * (1 - rate_alpha) + spike.astype(jnp.float32) * rate_alpha * 1000.0

    return state._replace(
        v=v,
        spike=spike,
        refractory=refractory,
        syn_current_e=syn_current_e,
        syn_current_i=syn_current_i,
        trace_pre=trace_pre,
        trace_post=trace_post,
        firing_rate=firing_rate
    )


def encode_input(
    gradient: float,
    key: random.PRNGKey,
    config: InputConfig,
    n_channels: int
) -> jnp.ndarray:
    """Encode gradient value into input channels."""
    # Create tuning curves for each channel
    # Channels are tuned to different gradient values
    preferred_values = jnp.linspace(0, 1, n_channels)
    
    # Gaussian tuning curves
    distances = jnp.abs(preferred_values - gradient)
    responses = jnp.exp(-0.5 * (distances / config.input_tuning_width) ** 2)
    
    # Scale by input gain
    responses = responses * config.input_gain
    
    # Add noise
    noise = random.normal(key, (n_channels,)) * config.input_noise
    responses = jnp.maximum(0, responses + noise)
    
    return responses


def decode_action(
    state: NeoAgentState,
    key: random.PRNGKey,
    dynamics_config: DynamicsConfig,
    network_config: NetworkConfig
) -> Tuple[int, jnp.ndarray]:
    """Decode action from readout neurons (vectorized)."""
    # Get readout neuron spikes
    readout_start = network_config.num_sensory + network_config.num_processing
    readout_spikes = state.spike[readout_start:]

    # Update motor trace (leaky integration)
    tau_decay = jnp.exp(-1.0 / dynamics_config.motor_tau)
    motor_trace = state.motor_trace * tau_decay

    # --- Vectorized update of motor trace ---
    n_actions = 4
    neurons_per_action = network_config.num_readout // n_actions
    
    # Reshape readout spikes into (n_actions, neurons_per_action)
    action_spikes_matrix = readout_spikes.reshape((n_actions, neurons_per_action))
    
    # Sum spikes for each action group along axis 1
    total_spikes_per_action = jnp.sum(action_spikes_matrix, axis=1)
    
    # Add the result to the motor trace
    motor_trace = motor_trace + total_spikes_per_action * 10.0
    # --- End of vectorization ---

    # Action selection with softmax and temperature
    logits = motor_trace / state.action_temperature
    action = random.categorical(key, logits)

    return action, motor_trace