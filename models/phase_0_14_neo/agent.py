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

from interfaces import AgentProtocol, ExperimentConfig, ExporterProtocol, EpisodeData
from rules import NetworkState, RuleContext, create_rule_pipeline

from .config import NeoConfig, InputConfig, DynamicsConfig, NetworkConfig
from .state import NeoAgentState, create_initial_state
from .dynamics import neuron_dynamics_step, decode_action, encode_input


class NeoAgent:
    """Phase 0.14 Neo agent with modular learning rules - Pure JAX."""
    
    # Agent metadata
    VERSION = "0.14.0"
    MODEL_NAME = "Neo-Modular-SNN"
    DESCRIPTION = "Spiking neural network with modular learning rules (Pure JAX)"
    
    def __init__(self, config: ExperimentConfig, exporter: ExporterProtocol):
        """Initialize agent with configuration and exporter."""
        # Convert ExperimentConfig to NeoConfig if needed
        if isinstance(config, NeoConfig):
            self.neo_config = config
        else:
            # Create NeoConfig from ExperimentConfig
            self.neo_config = self._create_neo_config(config)
        
        self.network_config = self.neo_config.network_config
        self.dynamics_config = self.neo_config.dynamics_config
        self.input_config = self.neo_config.input_config
        self.learning_config = self.neo_config.learning_rules_config
        
        self.config = config
        self.exporter = exporter
        
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
        
        # Episode data collection (pure JAX arrays)
        self.max_timesteps = config.world_params.get("max_timesteps", 10000)
        self.episode_buffer = self._create_episode_buffer()
        self.timestep = 0
    
    def reset(self, key: PRNGKey) -> None:
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
            network_state = self._agent_state_to_network_state(self.initial_state)
            self.rule_pipeline.initialize(network_state)
            
            # Save network structure
            self._export_network_structure()
        
        # Reset to initial state or soft reset
        if self.state is None:
            self.state = self.initial_state
        else:
            self.state = self._reset_episode_jit(
                self.state,
                self.learning_config,
                self.dynamics_config,
                self.network_config
            )
        
        # Reset episode tracking
        self.episode_buffer = self._create_episode_buffer()
        self.timestep = 0
    
    def act(self, gradient: Array, key: PRNGKey) -> Array:
        """Select action based on gradient observation - Pure JAX."""
        # Run step computation
        new_state, action, neural_data = self._step_jit(
            self.state,
            gradient,
            key,
            self.timestep,
            self.input_config,
            self.dynamics_config,
            self.network_config,
            self.rule_pipeline
        )
        
        # Update episode buffer
        self.episode_buffer = NeoAgent._update_buffer(
            self.episode_buffer,
            self.timestep,
            gradient,
            action,
            neural_data
        )
        
        # Update internal state
        self.state = new_state
        self.timestep += 1
        
        return action
    
    def get_episode_data(self) -> EpisodeData:
        """Get standardized episode data for logging."""
        # Extract data up to current timestep
        gradients = self.episode_buffer["gradients"][:self.timestep]
        actions = self.episode_buffer["actions"][:self.timestep]
        
        # Compute rewards (gradient == 1.0 means reward)
        rewards = jnp.where(gradients >= 0.99, 1.0, 0.0)
        
        # Neural data
        neural_data = {
            "membrane_potential": self.episode_buffer["membrane_potential"][:self.timestep],
            "spikes": self.episode_buffer["spikes"][:self.timestep],
            "firing_rates": self.episode_buffer["firing_rates"][:self.timestep]
        }
        
        # Learning data
        learning_data = {
            "final_weights": self.state.w,
            "weight_changes": self.state.w - self.initial_state.w,
            "final_learning_rate": self.state.learning_rate,
            "dopamine_trace": self.state.dopamine,
            "eligibility_trace": self.state.eligibility_trace
        }
        
        return EpisodeData(
            gradients=gradients,
            actions=actions,
            rewards=rewards,
            neural_data=neural_data,
            learning_data=learning_data
        )
    
    @staticmethod
    @partial(jit, static_argnums=(4, 5, 6, 7))
    def _step_jit(
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
        
        network_state = NeoAgent._agent_state_to_network_state_static(state)
        context = RuleContext(
            reward=reward_received.astype(jnp.float32),
            observation=gradient,
            action=action,
            spike_count=jnp.sum(state.spike),
            population_rate=jnp.mean(state.spike.astype(jnp.float32)),
            pre_spike_sum=jnp.sum(state.spike),
            post_spike_sum=jnp.sum(state.spike),
            dt=1.0,
            episode_progress=timestep / 10000.0,  # Approximate max timesteps
            params={"learning_rate": state.learning_rate}
        )
        
        updated_network_state = rule_pipeline.apply(network_state, context)
        state = NeoAgent._network_state_to_agent_state_static(updated_network_state, state)
        
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
    @partial(jit, static_argnums=(1, 2, 3))
    def _reset_episode_jit(
        state: NeoAgentState,
        learning_config: Any,
        dynamics_config: DynamicsConfig,
        network_config: NetworkConfig
    ) -> NeoAgentState:
        """Soft reset for new episode - JIT compiled."""
        n_neurons = state.v.shape[0]
        
        # Update learning rate
        new_lr = jnp.maximum(
            learning_config.base_learning_rate * (
                learning_config.learning_rate_decay ** state.episodes_completed
            ),
            learning_config.min_learning_rate
        )
        
        # Update temperature
        new_temp = jnp.maximum(
            state.action_temperature * dynamics_config.temperature_decay,
            dynamics_config.final_temperature
        )
        
        return state._replace(
            # Reset dynamics
            v=jnp.full(n_neurons, dynamics_config.v_rest),
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
            input_buffer=jnp.zeros(network_config.num_input_channels),
            
            # Update meta
            learning_rate=new_lr,
            action_temperature=new_temp,
            timestep=0,
            last_reward_count=0,
            episodes_completed=state.episodes_completed + 1
        )
    
    def _create_episode_buffer(self) -> Dict[str, Array]:
        """Create pre-allocated episode buffer."""
        n_neurons = self._get_network_size()
        return {
            "gradients": jnp.zeros(self.max_timesteps),
            "actions": jnp.zeros(self.max_timesteps, dtype=jnp.int32),
            "membrane_potential": jnp.zeros((self.max_timesteps, n_neurons)),
            "spikes": jnp.zeros((self.max_timesteps, n_neurons), dtype=bool),
            "firing_rates": jnp.zeros((self.max_timesteps, n_neurons))
        }
    
    @staticmethod
    @jit
    def _update_buffer(
        buffer: Dict[str, Array],
        timestep: int,
        gradient: Array,
        action: Array,
        neural_data: Dict[str, Array]
    ) -> Dict[str, Array]:
        """Update episode buffer - JIT compiled."""
        return {
            "gradients": buffer["gradients"].at[timestep].set(gradient),
            "actions": buffer["actions"].at[timestep].set(action),
            "membrane_potential": buffer["membrane_potential"].at[timestep].set(neural_data["v"]),
            "spikes": buffer["spikes"].at[timestep].set(neural_data["spikes"]),
            "firing_rates": buffer["firing_rates"].at[timestep].set(neural_data["firing_rate"])
        }
    
    @staticmethod
    def _agent_state_to_network_state_static(state: NeoAgentState) -> NetworkState:
        """Convert agent state to network state - static for JIT."""
        # Create spike history
        spike_history = jnp.stack([
            state.spike.astype(jnp.float32),
            state.spike.astype(jnp.float32),
            state.spike.astype(jnp.float32),
        ])
        
        # Convert input buffer to spikes (threshold at 0.5)
        input_spikes = state.input_buffer > 0.5
        
        return NetworkState(
            v=state.v,
            spike=state.spike,
            spike_history=spike_history,
            refractory=state.refractory,
            w=state.w,
            w_mask=state.w_mask,
            w_plastic_mask=state.w_plastic_mask,
            syn_current_e=state.syn_current_e,
            syn_current_i=state.syn_current_i,
            trace_pre=state.trace_pre,
            trace_post=state.trace_post,
            eligibility_trace=state.eligibility_trace,
            firing_rate=state.firing_rate,
            threshold_adapt=state.threshold_adapt,
            target_rate=jnp.full_like(state.firing_rate, 5.0),
            dopamine=state.dopamine,
            value_estimate=state.value_estimate,
            reward_prediction_error=0.0,
            is_excitatory=state.is_excitatory,
            neuron_types=state.neuron_types,
            learning_rate=state.learning_rate,
            weight_momentum=state.weight_momentum,
            timestep=state.timestep,
            episode=state.episodes_completed,
            input_spike=input_spikes,
            input_trace=state.input_buffer  # Use input values as trace
        )
    
    @staticmethod
    def _network_state_to_agent_state_static(
        network_state: NetworkState,
        original_state: NeoAgentState
    ) -> NeoAgentState:
        """Convert network state back to agent state - static for JIT."""
        return original_state._replace(
            v=network_state.v,
            spike=network_state.spike,
            refractory=network_state.refractory,
            w=network_state.w,
            syn_current_e=network_state.syn_current_e,
            syn_current_i=network_state.syn_current_i,
            trace_pre=network_state.trace_pre,
            trace_post=network_state.trace_post,
            eligibility_trace=network_state.eligibility_trace,
            firing_rate=network_state.firing_rate,
            threshold_adapt=network_state.threshold_adapt,
            dopamine=network_state.dopamine,
            value_estimate=network_state.value_estimate,
            weight_momentum=network_state.weight_momentum,
        )
    
    def _agent_state_to_network_state(self, state: NeoAgentState) -> NetworkState:
        """Instance method wrapper."""
        return self._agent_state_to_network_state_static(state)
    
    def _network_state_to_agent_state(
        self,
        network_state: NetworkState,
        original_state: NeoAgentState
    ) -> NeoAgentState:
        """Instance method wrapper."""
        return self._network_state_to_agent_state_static(network_state, original_state)
    
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
    
    def _create_neo_config(self, exp_config: ExperimentConfig) -> NeoConfig:
        """Create NeoConfig from ExperimentConfig."""
        # Extract parameters from ExperimentConfig
        
        # Separate network and dynamics params
        network_params = {}
        dynamics_params = {}
        
        for key, value in exp_config.neural_params.items():
            if key in ["num_sensory", "num_processing", "num_readout", "excitatory_ratio"]:
                network_params[key] = value
            elif key in ["tau_v", "v_threshold", "v_rest", "v_reset"]:
                dynamics_params[key] = value
        
        # Add learning params to dynamics
        for key, value in exp_config.learning_params.items():
            if key not in ["enabled_rules", "base_learning_rate", "learning_rate_decay"]:
                dynamics_params[key] = value
        
        config_dict = {
            "world": exp_config.world_params,
            "network": network_params,
            "dynamics": dynamics_params,
            "learning_rules": {
                "enabled_rules": exp_config.learning_params.get("enabled_rules", ["stdp", "homeostasis"]),
                "base_learning_rate": exp_config.learning_params.get("base_learning_rate", 0.1),
                "learning_rate_decay": exp_config.learning_params.get("learning_rate_decay", 0.98)
            },
            "experiment": {
                "n_episodes": exp_config.world_params.get("n_episodes", 50),
                "seed": exp_config.world_params.get("seed", 42)
            }
        }
        return NeoConfig.from_dict(config_dict)