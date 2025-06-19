# keywords: [neo state, agent state, network initialization]
"""State definitions for Phase 0.14 Neo agent."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import random

from .config import NetworkConfig, DynamicsConfig, LearningRulesConfig


class NeoAgentState(NamedTuple):
    """Complete state for Neo agent."""
    # Core neural dynamics
    v: jnp.ndarray  # Membrane potential
    spike: jnp.ndarray  # Current spike state
    refractory: jnp.ndarray  # Refractory period counter
    
    # Synaptic state
    w: jnp.ndarray  # Weight matrix (for compatibility/learning rules)
    w_exc: jnp.ndarray  # Pre-separated excitatory weights
    w_inh: jnp.ndarray  # Pre-separated inhibitory weights
    w_mask: jnp.ndarray  # Connectivity mask
    w_plastic_mask: jnp.ndarray  # Plasticity mask
    syn_current_e: jnp.ndarray  # Excitatory synaptic current
    syn_current_i: jnp.ndarray  # Inhibitory synaptic current
    
    # Plasticity traces
    trace_pre: jnp.ndarray  # Presynaptic trace
    trace_post: jnp.ndarray  # Postsynaptic trace
    eligibility_trace: jnp.ndarray  # Eligibility trace for RL
    
    # Homeostasis
    firing_rate: jnp.ndarray  # Running average firing rate
    threshold_adapt: jnp.ndarray  # Adaptive threshold
    
    # Neuromodulation
    dopamine: jnp.ndarray  # Global dopamine signal
    value_estimate: jnp.ndarray  # Value function estimate
    
    # Learning parameters
    weight_momentum: jnp.ndarray  # Momentum for weight updates
    learning_rate: float  # Current learning rate
    
    # Motor and sensory
    motor_trace: jnp.ndarray  # Motor output trace (6 neurons: 3 movement, 3 turning)
    input_buffer: jnp.ndarray  # Input channel buffer
    
    # Structural properties
    is_excitatory: jnp.ndarray  # Excitatory/inhibitory mask
    neuron_types: jnp.ndarray  # Neuron type labels (0=sensory, 1=processing, 2=readout)
    
    # Episode tracking
    action_temperature: float  # Action selection temperature
    timestep: int  # Current timestep in episode
    episodes_completed: int  # Number of episodes completed
    last_reward_count: int  # Reward count at last update


def create_initial_state(
    key: random.PRNGKey,
    network_config: NetworkConfig,
    dynamics_config: DynamicsConfig,
    learning_config: LearningRulesConfig
) -> NeoAgentState:
    """Create initial agent state with proper network structure."""
    keys = random.split(key, 10)
    
    # Total number of neurons
    n_total = (
        network_config.num_sensory + 
        network_config.num_processing + 
        network_config.num_readout
    )
    n_input = network_config.num_input_channels
    
    # Create neuron type labels
    neuron_types = jnp.concatenate([
        jnp.zeros(network_config.num_sensory, dtype=jnp.int32),
        jnp.ones(network_config.num_processing, dtype=jnp.int32),
        jnp.full(network_config.num_readout, 2, dtype=jnp.int32)
    ])
    
    # Assign excitatory/inhibitory identity
    is_excitatory = random.uniform(keys[0], (n_total,)) < network_config.excitatory_ratio
    # Ensure some diversity in each population
    is_excitatory = is_excitatory.at[:4].set(True)  # First 4 are excitatory
    is_excitatory = is_excitatory.at[4:8].set(False)  # Next 4 are inhibitory
    
    # Initialize weight matrix
    w = jnp.zeros((n_total, n_total + n_input), dtype=jnp.float16)
    w_mask = jnp.zeros((n_total, n_total + n_input), dtype=bool)
    
    # Input connections (to sensory neurons)
    input_mask = random.uniform(keys[1], (network_config.num_sensory, n_input)) < network_config.p_input_sensory
    w_mask = w_mask.at[:network_config.num_sensory, :n_input].set(input_mask)
    
    # Initialize input weights
    input_weights = random.normal(keys[2], (network_config.num_sensory, n_input)) * 0.5 + 0.5
    input_weights = jnp.abs(input_weights).astype(jnp.float16)  # Positive input weights
    w = w.at[:network_config.num_sensory, :n_input].set(input_weights * input_mask)
    
    # Recurrent connections
    recurrent_offset = n_input
    
    # Sensory to processing
    sens_end = network_config.num_sensory
    proc_start = sens_end
    proc_end = proc_start + network_config.num_processing
    
    s2p_mask = random.uniform(keys[3], (network_config.num_processing, network_config.num_sensory)) < network_config.p_sensory_processing
    w_mask = w_mask.at[proc_start:proc_end, recurrent_offset:recurrent_offset+sens_end].set(s2p_mask)
    
    # Processing to readout
    read_start = proc_end
    p2r_mask = random.uniform(keys[4], (network_config.num_readout, network_config.num_processing)) < network_config.p_processing_readout
    w_mask = w_mask.at[read_start:, recurrent_offset+proc_start:recurrent_offset+proc_end].set(p2r_mask)

    # --- Vectorized Processing recurrent connections (E-E, E-I, I-E, I-I) ---
    proc_indices = jnp.arange(proc_start, proc_end)
    is_proc_exc = is_excitatory[proc_indices]

    # Create masks for pre- and post-synaptic types
    pre_is_exc = is_proc_exc[None, :]  # Shape (1, n_processing)
    post_is_exc = is_proc_exc[:, None] # Shape (n_processing, 1)

    # Create probability matrix based on connection types
    p_matrix = jnp.zeros((network_config.num_processing, network_config.num_processing))
    p_matrix = jnp.where(post_is_exc & pre_is_exc,   network_config.p_ee, p_matrix) # E -> E
    p_matrix = jnp.where(post_is_exc & ~pre_is_exc,  network_config.p_ie, p_matrix) # I -> E
    p_matrix = jnp.where(~post_is_exc & pre_is_exc,  network_config.p_ei, p_matrix) # E -> I
    p_matrix = jnp.where(~post_is_exc & ~pre_is_exc, network_config.p_ii, p_matrix) # I -> I

    # Generate random numbers and create the mask
    proc_recurrent_mask = random.uniform(keys[5], p_matrix.shape) < p_matrix
    
    # Prevent self-connections
    diag_indices = jnp.arange(network_config.num_processing)
    proc_recurrent_mask = proc_recurrent_mask.at[diag_indices, diag_indices].set(False)

    # Apply this mask to the global weight mask
    w_mask = w_mask.at[
        proc_start:proc_end, 
        recurrent_offset+proc_start:recurrent_offset+proc_end
    ].set(proc_recurrent_mask)
    # --- End of vectorization ---
    
    # Initialize recurrent weights
    recurrent_weights = random.normal(keys[6], (n_total, n_total)) * 0.1
    recurrent_weights = jnp.abs(recurrent_weights) * 2.0  # Scale up
    
    # Apply Dale's principle to initial weights
    sign_mask = jnp.where(is_excitatory, 1.0, -1.0).astype(jnp.float16)
    recurrent_weights = (jnp.abs(recurrent_weights) * sign_mask[:, None]).astype(jnp.float16)
    
    # Set weights where connections exist
    w = w.at[:, recurrent_offset:].set(recurrent_weights * w_mask[:, recurrent_offset:])
    
    # Create plasticity mask
    w_plastic_mask = w_mask.copy()
    if not network_config.learn_input:
        w_plastic_mask = w_plastic_mask.at[:, :n_input].set(False)
    if not network_config.learn_recurrent:
        w_plastic_mask = w_plastic_mask.at[:, recurrent_offset:].set(False)
    if not network_config.learn_readout:
        # Disable plasticity for connections to/from readout
        w_plastic_mask = w_plastic_mask.at[read_start:, :].set(False)
        w_plastic_mask = w_plastic_mask.at[:, recurrent_offset+read_start:].set(False)
    
    # Initialize neural dynamics
    v_init = jnp.full(n_total, dynamics_config.v_rest, dtype=jnp.float16)
    
    # Pre-compute excitatory and inhibitory weight matrices
    # Create masks for excitatory/inhibitory neurons (including input channels)
    is_exc_full = jnp.concatenate([
        jnp.ones(n_input, dtype=bool),  # Input channels are excitatory
        is_excitatory
    ])
    
    # Create masks for selecting only excitatory or inhibitory weights
    exc_pre_mask = is_exc_full[None, :]  # Shape: (1, n_total + n_input)
    inh_pre_mask = ~is_exc_full[None, :]  # Shape: (1, n_total + n_input)
    
    # Apply masks to create separated weight matrices
    w_exc = jnp.where(exc_pre_mask, w, 0.0).astype(jnp.float16)
    w_inh = jnp.where(inh_pre_mask, w, 0.0).astype(jnp.float16)
    
    return NeoAgentState(
        # Core dynamics
        v=v_init,
        spike=jnp.zeros(n_total, dtype=bool),
        refractory=jnp.zeros(n_total),
        
        # Synaptic
        w=w,
        w_exc=w_exc,
        w_inh=w_inh,
        w_mask=w_mask,
        w_plastic_mask=w_plastic_mask,
        syn_current_e=jnp.zeros(n_total, dtype=jnp.float16),
        syn_current_i=jnp.zeros(n_total, dtype=jnp.float16),
        
        # Plasticity
        trace_pre=jnp.zeros(n_total, dtype=jnp.float16),
        trace_post=jnp.zeros(n_total, dtype=jnp.float16),
        eligibility_trace=jnp.zeros_like(w, dtype=jnp.float16),
        
        # Homeostasis
        firing_rate=jnp.full(n_total, 5.0, dtype=jnp.float16),  # Target rate
        threshold_adapt=jnp.zeros(n_total, dtype=jnp.float16),
        
        # Neuromodulation
        dopamine=jnp.array(0.2, dtype=jnp.float16),  # Baseline dopamine
        value_estimate=jnp.array(0.0, dtype=jnp.float16),
        
        # Learning
        weight_momentum=jnp.zeros_like(w, dtype=jnp.float16),
        learning_rate=learning_config.base_learning_rate,
        
        # Motor/sensory
        motor_trace=jnp.zeros(6, dtype=jnp.float16),  # 6 motor neurons: 3 movement, 3 turning
        input_buffer=jnp.zeros(n_input, dtype=jnp.float16),
        
        # Structure
        is_excitatory=is_excitatory,
        neuron_types=neuron_types,
        
        # Episode
        action_temperature=dynamics_config.initial_temperature,
        timestep=0,
        episodes_completed=0,
        last_reward_count=0
    )


@jax.jit
def update_weight_matrices(state: NeoAgentState) -> NeoAgentState:
    """Update w_exc and w_inh based on the current w matrix.
    
    This function should be called after any learning rule modifies w
    to maintain consistency between the weight representations.
    """
    # Get the number of input channels
    n_input = state.input_buffer.shape[0]
    
    # Create masks for excitatory/inhibitory neurons (including input channels)
    is_exc_full = jnp.concatenate([
        jnp.ones(n_input, dtype=bool),  # Input channels are excitatory
        state.is_excitatory
    ])
    
    # Create masks for selecting only excitatory or inhibitory weights
    exc_pre_mask = is_exc_full[None, :]  # Shape: (1, n_total + n_input)
    inh_pre_mask = ~is_exc_full[None, :]  # Shape: (1, n_total + n_input)
    
    # Apply masks to create separated weight matrices
    w_exc = jnp.where(exc_pre_mask, state.w, 0.0).astype(jnp.float16)
    w_inh = jnp.where(inh_pre_mask, state.w, 0.0).astype(jnp.float16)
    
    return state._replace(w_exc=w_exc, w_inh=w_inh)