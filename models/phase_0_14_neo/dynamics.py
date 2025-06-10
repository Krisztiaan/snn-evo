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
    config: DynamicsConfig,
    network_config: NetworkConfig
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
    
    # 9. Update motor trace from readout spikes
    readout_start = network_config.num_sensory + network_config.num_processing
    readout_end = readout_start + 6  # We only use 6 motor neurons
    readout_spikes = spike[readout_start:readout_end].astype(jnp.float32)
    
    tau_decay = jnp.exp(-1.0 / config.motor_tau)
    motor_trace = state.motor_trace * tau_decay + readout_spikes * 10.0

    return state._replace(
        v=v,
        spike=spike,
        refractory=refractory,
        syn_current_e=syn_current_e,
        syn_current_i=syn_current_i,
        trace_pre=trace_pre,
        trace_post=trace_post,
        firing_rate=firing_rate,
        motor_trace=motor_trace
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
    """Decode an action from a unified pool of 6 motor neurons.
    
    Motor neurons vote for actions:
    - 0: FWD_LEFT
    - 1: FWD
    - 2: FWD_RIGHT
    - 3: STOP_LEFT (pure rotation left)
    - 4: STOP (stay in place)
    - 5: STOP_RIGHT (pure rotation right)
    """
    # 1. Get spikes from all 6 readout neurons
    readout_start = network_config.num_sensory + network_config.num_processing
    readout_end = readout_start + 6  # We only use 6 motor neurons
    readout_spikes = state.spike[readout_start:readout_end].astype(jnp.float32)

    # 2. Update the motor trace (leaky integration for vote accumulation)
    tau_decay = jnp.exp(-1.0 / dynamics_config.motor_tau)
    new_motor_trace = state.motor_trace * tau_decay + readout_spikes * 10.0

    # 3. "Vote" using softmax with temperature for stochastic action selection
    logits = new_motor_trace / state.action_temperature
    action_choice = random.categorical(key, logits)  # Result: an integer from 0 to 5

    # 4. Direct mapping to world actions
    # Our 6 agent actions map to world actions:
    #   Agent 0 -> World 7: FORWARD_LEFT
    #   Agent 1 -> World 0: FORWARD
    #   Agent 2 -> World 1: FORWARD_RIGHT
    #   Agent 3 -> World 6: LEFT (pure rotation)
    #   Agent 4 -> World 8: STAY
    #   Agent 5 -> World 2: RIGHT (pure rotation)
    
    # Create mapping array
    action_map = jnp.array([7, 0, 1, 6, 8, 2])
    world_action = action_map[action_choice]

    return world_action, new_motor_trace


def decode_action_from_trace(
    state: NeoAgentState,
    key: random.PRNGKey,
    dynamics_config: DynamicsConfig
) -> int:
    """Statelessly selects an action based on the current motor trace.
    
    This is used after the integration window to decode the final action.
    """
    # Use softmax with temperature for stochastic action selection
    logits = state.motor_trace / state.action_temperature
    action_choice = random.categorical(key, logits)  # Result: an integer from 0 to 5
    
    # Direct mapping to world actions
    action_map = jnp.array([7, 0, 1, 6, 8, 2])
    world_action = action_map[action_choice]
    
    return world_action