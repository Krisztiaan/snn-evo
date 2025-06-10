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
    w: jnp.ndarray  # Weight matrix
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
    motor_trace: jnp.ndarray  # Motor output trace (4 actions)
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
    w = jnp.zeros((n_total, n_total + n_input))
    w_mask = jnp.zeros((n_total, n_total + n_input), dtype=bool)
    
    # Input connections (to sensory neurons)
    input_mask = random.uniform(keys[1], (network_config.num_sensory, n_input)) < network_config.p_input_sensory
    w_mask = w_mask.at[:network_config.num_sensory, :n_input].set(input_mask)
    
    # Initialize input weights
    input_weights = random.normal(keys[2], (network_config.num_sensory, n_input)) * 0.5 + 0.5
    input_weights = jnp.abs(input_weights)  # Positive input weights
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
    
    # Processing recurrent connections (E-E, E-I, I-E, I-I)
    proc_exc = is_excitatory[proc_start:proc_end]
    proc_inh = ~proc_exc
    
    for i in range(network_config.num_processing):
        for j in range(network_config.num_processing):
            if i == j:  # No self-connections
                continue
            
            i_global = proc_start + i
            j_global = proc_start + j
            
            # Determine connection probability based on types
            if proc_exc[i] and proc_exc[j]:
                p = network_config.p_ee
            elif proc_exc[i] and proc_inh[j]:
                p = network_config.p_ei
            elif proc_inh[i] and proc_exc[j]:
                p = network_config.p_ie
            else:  # I-I
                p = network_config.p_ii
            
            if random.uniform(keys[5], ()) < p:
                w_mask = w_mask.at[i_global, recurrent_offset + j_global].set(True)
    
    # Initialize recurrent weights
    recurrent_weights = random.normal(keys[6], (n_total, n_total)) * 0.1
    recurrent_weights = jnp.abs(recurrent_weights) * 2.0  # Scale up
    
    # Apply Dale's principle to initial weights
    sign_mask = jnp.where(is_excitatory, 1.0, -1.0)
    recurrent_weights = jnp.abs(recurrent_weights) * sign_mask[:, None]
    
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
    v_init = jnp.full(n_total, dynamics_config.v_rest)
    
    return NeoAgentState(
        # Core dynamics
        v=v_init,
        spike=jnp.zeros(n_total, dtype=bool),
        refractory=jnp.zeros(n_total),
        
        # Synaptic
        w=w,
        w_mask=w_mask,
        w_plastic_mask=w_plastic_mask,
        syn_current_e=jnp.zeros(n_total),
        syn_current_i=jnp.zeros(n_total),
        
        # Plasticity
        trace_pre=jnp.zeros(n_total),
        trace_post=jnp.zeros(n_total),
        eligibility_trace=jnp.zeros_like(w),
        
        # Homeostasis
        firing_rate=jnp.full(n_total, 5.0),  # Target rate
        threshold_adapt=jnp.zeros(n_total),
        
        # Neuromodulation
        dopamine=jnp.array(0.2),  # Baseline dopamine
        value_estimate=jnp.array(0.0),
        
        # Learning
        weight_momentum=jnp.zeros_like(w),
        learning_rate=learning_config.base_learning_rate,
        
        # Motor/sensory
        motor_trace=jnp.zeros(4),
        input_buffer=jnp.zeros(n_input),
        
        # Structure
        is_excitatory=is_excitatory,
        neuron_types=neuron_types,
        
        # Episode
        action_temperature=dynamics_config.initial_temperature,
        timestep=0,
        episodes_completed=0,
        last_reward_count=0
    )