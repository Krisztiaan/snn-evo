# keywords: [rule adapter, agent integration, state conversion, compatibility layer]
"""
Adapter for integrating the modular rule system with existing agents.

This module provides utilities to convert between agent-specific state
representations and the unified NetworkState used by the rule system.
"""

from typing import Any, Dict, List, Optional, Union

import jax
import jax.numpy as jnp
from jax import Array

from .base import NetworkState, RuleContext
from .pipeline import RulePipeline, create_rule_pipeline


class Phase013Adapter:
    """Adapter for integrating rules with phase_0_13 agent.
    
    Converts between the agent's state representation and NetworkState.
    """
    
    @staticmethod
    def agent_state_to_network_state(agent_state, params) -> NetworkState:
        """Convert phase_0_13 AgentState to NetworkState."""
        # Convert spike history for STDP
        # We'll use the current spike and traces
        spike_float = agent_state.spike.astype(jnp.float32)
        
        # Create spike history from traces (approximation)
        spike_history = jnp.stack([
            spike_float,
            agent_state.trace_fast,
            agent_state.trace_slow
        ], axis=0)
        
        return NetworkState(
            # Core neural dynamics
            v=agent_state.v,
            spike=agent_state.spike,
            spike_history=spike_history,
            refractory=agent_state.refractory,
            
            # Synaptic state
            w=agent_state.w,
            w_mask=agent_state.w_mask,
            w_plastic_mask=agent_state.w_plastic_mask,
            syn_current_e=agent_state.syn_current_e,
            syn_current_i=agent_state.syn_current_i,
            
            # Plasticity traces
            trace_pre=agent_state.trace_fast,  # Use fast trace as pre
            trace_post=agent_state.trace_slow,  # Use slow trace as post
            eligibility_trace=agent_state.eligibility_trace,
            
            # Homeostasis
            firing_rate=agent_state.firing_rate,
            threshold_adapt=agent_state.threshold_adapt,
            target_rate=jnp.full_like(agent_state.firing_rate, params.TARGET_RATE_HZ),
            
            # Neuromodulation
            dopamine=agent_state.dopamine,
            value_estimate=agent_state.value_estimate,
            reward_prediction_error=0.0,  # Will be calculated by rules
            
            # Structural properties
            is_excitatory=agent_state.is_excitatory,
            neuron_types=agent_state.neuron_types,
            
            # Learning state
            learning_rate=params.BASE_LEARNING_RATE,
            weight_momentum=agent_state.weight_momentum,
            
            # Metadata
            timestep=0,  # Will be set from context
            episode=agent_state.episodes_completed
        )
    
    @staticmethod
    def network_state_to_agent_state(network_state, original_agent_state):
        """Convert NetworkState back to phase_0_13 AgentState."""
        # Only update fields that were modified
        return original_agent_state._replace(
            # Core dynamics
            v=network_state.v,
            spike=network_state.spike,
            refractory=network_state.refractory,
            syn_current_e=network_state.syn_current_e,
            syn_current_i=network_state.syn_current_i,
            
            # Weights
            w=network_state.w,
            
            # Traces
            trace_fast=network_state.trace_pre,
            trace_slow=network_state.trace_post,
            eligibility_trace=network_state.eligibility_trace,
            
            # Homeostasis
            firing_rate=network_state.firing_rate,
            threshold_adapt=network_state.threshold_adapt,
            
            # Neuromodulation
            dopamine=network_state.dopamine,
            value_estimate=network_state.value_estimate,
            
            # Momentum
            weight_momentum=network_state.weight_momentum,
        )
    
    @staticmethod
    def create_rule_context(
        reward: float,
        observation: Array,
        action: Optional[int] = None,
        dt: float = 1.0,
        timestep: int = 0,
        episode_progress: float = 0.0,
        **kwargs
    ) -> RuleContext:
        """Create a RuleContext for the current timestep."""
        # Pre-compute some common values
        return RuleContext(
            reward=reward,
            observation=observation,
            action=action,
            spike_count=jnp.zeros(1),  # Will be computed if needed
            population_rate=0.0,  # Will be computed if needed
            pre_spike_sum=jnp.zeros(1),  # Will be computed if needed
            post_spike_sum=jnp.zeros(1),  # Will be computed if needed
            dt=dt,
            episode_progress=episode_progress,
            params=kwargs
        )


def create_phase013_rule_pipeline(
    enabled_rules: Optional[List[str]] = None,
    rule_params: Optional[Dict[str, Dict[str, Any]]] = None
) -> RulePipeline:
    """Create a rule pipeline configured for phase_0_13 agent.
    
    Args:
        enabled_rules: List of rule names to enable. If None, uses defaults.
        rule_params: Parameters for each rule.
        
    Returns:
        Configured rule pipeline
    """
    if enabled_rules is None:
        # Default rules that match phase_0_13 behavior
        enabled_rules = [
            "stdp",
            "eligibility_trace", 
            "dopamine_modulation",
            "homeostatic",
            "threshold_adaptation",
            "dale_principle",
            "weight_bounds",
            "weight_decay"
        ]
    
    if rule_params is None:
        rule_params = {}
    
    # Build rule configurations
    rule_configs = []
    for rule_name in enabled_rules:
        config = {
            "name": rule_name,
            "params": rule_params.get(rule_name, {})
        }
        rule_configs.append(config)
    
    return create_rule_pipeline(rule_configs)


class RuleIntegratedAgentMixin:
    """Mixin class to add rule system support to existing agents.
    
    Add this as a base class to enable modular rules:
    
        class MyAgent(BaseAgent, RuleIntegratedAgentMixin):
            ...
    """
    
    def initialize_rules(
        self,
        enabled_rules: Optional[List[str]] = None,
        rule_params: Optional[Dict[str, Dict[str, Any]]] = None
    ):
        """Initialize the rule pipeline."""
        self.rule_pipeline = create_phase013_rule_pipeline(enabled_rules, rule_params)
        
        # Initialize rules with sample state
        if hasattr(self, 'state'):
            network_state = self.state_to_network_state(self.state)
            self.rule_pipeline.initialize(network_state)
    
    def apply_rules(
        self,
        agent_state,
        reward: float,
        observation: Array,
        action: Optional[int] = None,
        **context_kwargs
    ):
        """Apply the rule pipeline to update the agent state."""
        # Convert to NetworkState
        network_state = self.state_to_network_state(agent_state)
        
        # Create context
        context = self.create_rule_context(
            reward=reward,
            observation=observation,
            action=action,
            **context_kwargs
        )
        
        # Apply rules
        updated_network_state = self.rule_pipeline.apply(network_state, context)
        
        # Convert back to agent state
        return self.network_state_to_agent_state(updated_network_state, agent_state)
    
    def state_to_network_state(self, agent_state) -> NetworkState:
        """Convert agent state to NetworkState. Override in subclass."""
        raise NotImplementedError
    
    def network_state_to_agent_state(self, network_state, original_state):
        """Convert NetworkState to agent state. Override in subclass."""
        raise NotImplementedError
    
    def create_rule_context(self, **kwargs) -> RuleContext:
        """Create rule context. Can override in subclass."""
        return Phase013Adapter.create_rule_context(**kwargs)