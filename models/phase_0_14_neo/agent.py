# keywords: [neo agent, phase 0.14, modular rules, pure jax, jit compatible]
"""
Phase 0.14 Neo Agent - Pure JAX implementation with modular learning rules.

This agent demonstrates how to use the modular learning rules system
with a clean JAX architecture that is fully JIT-compilable.
"""

from functools import partial
from typing import Dict, Optional, Tuple, Any

import jax
import jax.numpy as jnp
from jax import Array, jit
from jax.random import PRNGKey, split

from interfaces import AgentProtocol, ExperimentConfig, ExporterProtocol
from rules import RuleContext, create_rule_pipeline

from .config import InputConfig, DynamicsConfig, NetworkConfig, LearningRulesConfig
from .state import NeoAgentState, create_initial_state, update_weight_matrices
from .dynamics import neuron_dynamics_step, decode_action, encode_input


class NeoAgent:
    """Phase 0.14 Neo agent with modular learning rules - Pure JAX."""
    
    # Agent metadata
    VERSION = "0.14.0"
    MODEL_NAME = "Neo-Modular-SNN"
    DESCRIPTION = "Spiking neural network with modular learning rules (Pure JAX)"
    
    def __init__(self, config: ExperimentConfig, exporter: ExporterProtocol):
        """Initialize agent with configuration and exporter."""
        self.config = config
        self.exporter = exporter
        
        # Create component configs directly from ExperimentConfig
        # Calculate total processing neurons
        total_neurons = config.neural.n_excitatory + config.neural.n_inhibitory
        processing_neurons = total_neurons - config.neural.n_sensory - config.neural.n_motor
        
        self.network_config = NetworkConfig(
            num_sensory=config.neural.n_sensory,
            num_processing=processing_neurons,
            num_readout=config.neural.n_motor,
            num_input_channels=4,  # 4 gradient directions
            excitatory_ratio=config.neural.n_excitatory / total_neurons
        )
        
        self.dynamics_config = DynamicsConfig(
            tau_v=config.neural.tau_membrane,
            tau_syn_e=config.neural.tau_syn_e,
            tau_syn_i=config.neural.tau_syn_i,
            v_rest=-70.0,
            v_threshold=-50.0,
            v_reset=-75.0,
            refractory_time=2.0,
            initial_temperature=config.behavior.temperature,
            final_temperature=0.1,
            temperature_decay=0.995
        )
        
        self.input_config = InputConfig(
            input_gain=15.0,
            input_tuning_width=0.15,
            input_noise=config.behavior.action_noise
        )
        
        # Learning config with enabled rules based on plasticity settings
        enabled_rules = []
        if config.plasticity.enable_stdp:
            enabled_rules.extend(["stdp", "eligibility_trace"])
        if config.plasticity.enable_homeostasis:
            enabled_rules.append("homeostatic")
        if config.plasticity.enable_reward_modulation:
            enabled_rules.append("dopamine_modulation")
            
        self.learning_config = LearningRulesConfig(
            enabled_rules=enabled_rules,
            rule_params={},
            base_learning_rate=0.001,
            learning_rate_decay=0.999,
            min_learning_rate=0.0001
        )
        
        # Initialize rule pipeline
        print(f"Creating rule pipeline with rules: {self.learning_config.enabled_rules}")
        self.rule_pipeline = create_rule_pipeline(
            [{"name": rule, "params": self.learning_config.rule_params.get(rule, {})}
             for rule in self.learning_config.enabled_rules]
        )
        print("Rule pipeline created.")
        
        # Agent state (will be initialized in reset)
        self.state: Optional[NeoAgentState] = None
        self.initial_state: Optional[NeoAgentState] = None
        self.timestep = 0
        
        # Create JIT-compiled step function with configs closed over
        self.step = self._create_step_function()
    
    def reset(self, key: PRNGKey) -> NeoAgentState:
        """Reset agent's internal state for new episode."""
        # Initialize state if first time
        if self.initial_state is None:
            self.initial_state = create_initial_state(
                key,
                self.network_config,
                self.dynamics_config,
                self.learning_config
            )
            
            # Initialize rule pipeline with initial state
            self.rule_pipeline.initialize(self.initial_state)
            
            # Save network structure
            self._export_network_structure()
        
        # Reset to initial state or soft reset
        if self.state is None:
            self.state = self.initial_state
        else:
            self.state = self._reset_episode_jit(
                self.state,
                self.learning_config.base_learning_rate,
                self.learning_config.learning_rate_decay,
                self.learning_config.min_learning_rate,
                self.dynamics_config.temperature_decay,
                self.dynamics_config.final_temperature,
                self.dynamics_config.v_rest
            )
        
        # Reset timestep
        self.timestep = 0
        
        return self.state
    
    def _create_step_function(self):
        """Create a JIT-compiled step function with configs closed over."""
        # Capture configs and pipeline in closure
        input_config = self.input_config
        dynamics_config = self.dynamics_config
        network_config = self.network_config
        rule_pipeline = self.rule_pipeline
        
        @jit
        def step_wrapper(state: NeoAgentState, gradient: Array, key: PRNGKey) -> Tuple[Array, NeoAgentState, Dict[str, Array]]:
            """Pure JAX step function for the runner."""
            # Use the agent's current timestep from state
            timestep = state.timestep
            
            # Call the original step function with all parameters
            new_state, action, neural_data = NeoAgent.step_impl(
                state,
                gradient,
                key,
                timestep,
                input_config,
                dynamics_config,
                network_config,
                rule_pipeline
            )
            
            return new_state, action, neural_data
        
        return step_wrapper
    
    def act(self, gradient: Array, key: PRNGKey) -> Tuple[Array, Any, Dict[str, Array]]:
        """Select action and update host-side state for non-JIT execution.
        
        Returns:
            action: The selected action
            state: Updated agent state (for interface compatibility)
            neural_data: Neural data for logging
        """
        # Call the pure JIT function with the current host state
        new_state, action, neural_data = self.step(self.state, gradient, key)
        
        # Update the host-side state
        self.state = new_state
        self.timestep = new_state.timestep
        
        return action, self.state, neural_data
    
    
    @staticmethod
    @partial(jit, static_argnums=(4, 5, 6, 7))
    def step_impl(
        state: NeoAgentState,
        gradient: Array,
        key: PRNGKey,
        timestep: int,
        input_config: InputConfig,
        dynamics_config: DynamicsConfig,
        network_config: NetworkConfig,
        rule_pipeline: Any
    ) -> Tuple[NeoAgentState, Array, Dict[str, Array]]:
        """Single agent step - JIT compiled."""
        keys = split(key, 3)
        
        # 1. Encode input
        input_channels = encode_input(
            gradient, keys[0], input_config, network_config.num_input_channels
        )
        state = state._replace(input_buffer=input_channels)
        
        # 2. Neural dynamics
        state = neuron_dynamics_step(state, keys[1], dynamics_config)
        
        # 3. Decode action
        action, motor_trace = decode_action(
            state, keys[2], dynamics_config, network_config
        )
        state = state._replace(motor_trace=motor_trace)
        
        # 4. Apply learning rules
        # Check if reward received (gradient near 1.0)
        reward_received = gradient >= 0.99
        
        # --- Optimization: Compute expensive values once ---
        current_spikes = state.spike.astype(jnp.float32)
        total_spike_count = jnp.sum(current_spikes)
        n_neurons = state.v.shape[0]
        
        context = RuleContext(
            reward=reward_received.astype(jnp.float32),
            observation=gradient,
            action=action,
            spike_count=total_spike_count,  # Use pre-computed
            population_rate=total_spike_count / n_neurons,  # Use pre-computed
            pre_spike_sum=total_spike_count,  # Use pre-computed
            post_spike_sum=total_spike_count,  # Use pre-computed
            dt=1.0,
            episode_progress=timestep / 10000.0,  # Approximate max timesteps
            params={"learning_rate": state.learning_rate}
        )
        
        # Pass the agent state directly to the pipeline
        state = rule_pipeline.apply(state, context)
        
        # Update separated weight matrices after learning
        state = update_weight_matrices(state)
        
        # 5. Update state metadata
        state = state._replace(
            timestep=timestep,
            last_reward_count=state.last_reward_count + reward_received.astype(jnp.int32)
        )
        
        # Collect neural data
        neural_data = {
            "v": state.v,
            "spikes": state.spike,
            "firing_rate": state.firing_rate,
        }
        
        return state, action, neural_data
    
    @staticmethod
    @jit
    def _reset_episode_jit(
        state: NeoAgentState,
        base_learning_rate: float,
        learning_rate_decay: float,
        min_learning_rate: float,
        temperature_decay: float,
        final_temperature: float,
        v_rest: float
    ) -> NeoAgentState:
        """Soft reset for new episode - JIT compiled."""
        n_neurons = state.v.shape[0]
        
        # Update learning rate
        new_lr = jnp.maximum(
            base_learning_rate * (
                learning_rate_decay ** state.episodes_completed
            ),
            min_learning_rate
        )
        
        # Update temperature
        new_temp = jnp.maximum(
            state.action_temperature * temperature_decay,
            final_temperature
        )
        
        # Get input buffer size from existing state
        input_buffer_size = state.input_buffer.shape[0]
        
        return state._replace(
            # Reset dynamics
            v=jnp.full(n_neurons, v_rest),
            spike=jnp.zeros(n_neurons, dtype=bool),
            refractory=jnp.zeros(n_neurons),
            syn_current_e=jnp.zeros(n_neurons),
            syn_current_i=jnp.zeros(n_neurons),
            
            # Soft reset traces
            trace_pre=state.trace_pre * 0.5,
            trace_post=state.trace_post * 0.5,
            eligibility_trace=state.eligibility_trace * 0.9,
            
            # Soft reset homeostasis
            firing_rate=state.firing_rate * 0.9 + 5.0 * 0.1,
            threshold_adapt=state.threshold_adapt * 0.9,
            
            # Soft reset neuromodulation
            dopamine=state.dopamine * 0.8 + 0.2 * 0.2,
            value_estimate=state.value_estimate * 0.5,
            
            # Reset motor
            motor_trace=jnp.zeros(4),
            input_buffer=jnp.zeros(input_buffer_size),
            
            # Update meta
            learning_rate=new_lr,
            action_temperature=new_temp,
            timestep=0,
            last_reward_count=0,
            episodes_completed=state.episodes_completed + 1
        )
    
    
    
    def _export_network_structure(self) -> None:
        """Export network structure to exporter."""
        n_neurons = self._get_network_size()
        
        # Get connectivity info
        recurrent_mask = self.initial_state.w_mask[:, self.network_config.num_input_channels:]
        conn_indices = jnp.where(recurrent_mask)
        
        neurons = {
            "neuron_ids": jnp.arange(n_neurons),
            "is_excitatory": self.initial_state.is_excitatory,
            "neuron_types": self.initial_state.neuron_types,
        }
        
        connections = {
            "source_ids": conn_indices[1],
            "target_ids": conn_indices[0],
        }
        
        initial_weights = self.initial_state.w[:, self.network_config.num_input_channels:][recurrent_mask]
        
        self.exporter.save_network_structure(neurons, connections, initial_weights)
    
    def _get_network_size(self) -> int:
        """Get total number of neurons."""
        return (
            self.network_config.num_sensory +
            self.network_config.num_processing +
            self.network_config.num_readout
        )
    
