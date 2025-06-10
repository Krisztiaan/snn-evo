# keywords: [neuromodulation, dopamine, eligibility trace, reward prediction, three-factor]
"""
Neuromodulation learning rules.

This module implements neuromodulatory mechanisms including dopamine
modulation, eligibility traces, and reward prediction for three-factor
learning rules.
"""

import jax
import jax.numpy as jnp
from jax import Array

from .base import AbstractLearningRule, RuleContext
from .registry import register_rule
from models.phase_0_14_neo.state import NeoAgentState


@register_rule("dopamine_modulation", category="neuromodulation",
               description="Dopamine-modulated plasticity")
class DopamineModulationRule(AbstractLearningRule):
    """Dopamine modulation for three-factor learning.
    
    Modulates synaptic plasticity based on dopamine levels, implementing
    reward-based learning where weight changes are gated by dopamine.
    """
    
    def __init__(
        self,
        baseline_dopamine: float = 0.2,
        tau_dopamine: float = 200.0,
        reward_scale: float = 5.0,
        gradient_scale: float = 0.3,
        modulation_threshold: float = 0.0,
        **kwargs
    ):
        """Initialize dopamine modulation rule.
        
        Args:
            baseline_dopamine: Baseline dopamine level
            tau_dopamine: Dopamine decay time constant
            reward_scale: Scaling factor for reward signals
            gradient_scale: Scaling factor for gradient rewards
            modulation_threshold: Minimum dopamine for plasticity
        """
        super().__init__(
            baseline_dopamine=baseline_dopamine,
            tau_dopamine=tau_dopamine,
            reward_scale=reward_scale,
            gradient_scale=gradient_scale,
            modulation_threshold=modulation_threshold,
            **kwargs
        )
        self.baseline_dopamine = baseline_dopamine
        self.tau_dopamine = tau_dopamine
        self.reward_scale = reward_scale
        self.gradient_scale = gradient_scale
        self.modulation_threshold = modulation_threshold
    
    @property
    def name(self) -> str:
        return "dopamine_modulation"
    
    @property
    def requires(self) -> set[str]:
        return {"w", "w_plastic_mask", "eligibility_trace", "dopamine"}
    
    @property
    def modifies(self) -> set[str]:
        return {"w", "dopamine"}
    
    def apply(self, state: NeoAgentState, context: RuleContext) -> NeoAgentState:
        """Apply dopamine-modulated plasticity."""
        dt = context.dt
        
        # Update dopamine level
        dopamine_decay = jnp.exp(-dt / self.tau_dopamine)
        
        # Calculate reward signal
        reward_signal = (
            context.reward * self.reward_scale +
            context.observation * self.gradient_scale
        )
        
        # Update dopamine with decay and reward
        new_dopamine = (
            state.dopamine * dopamine_decay +
            self.baseline_dopamine * (1 - dopamine_decay) +
            reward_signal
        )
        new_dopamine = jnp.clip(new_dopamine, 0.0, 2.0)
        
        # Calculate modulation factor
        da_factor = (new_dopamine - self.baseline_dopamine) / self.baseline_dopamine
        modulation = jax.nn.tanh(da_factor * 2.0)
        
        # Apply modulation only if above threshold
        effective_modulation = jnp.where(
            new_dopamine > self.modulation_threshold,
            modulation,
            0.0
        )
        
        # Update weights based on eligibility trace and modulation
        dw = state.learning_rate * effective_modulation * state.eligibility_trace
        w_new = state.w + dw * state.w_plastic_mask
        
        return state._replace(
            w=w_new,
            dopamine=new_dopamine
        )


@register_rule("eligibility_trace", category="neuromodulation",
               description="Eligibility trace for credit assignment")
class EligibilityTraceRule(AbstractLearningRule):
    """Eligibility trace rule for temporal credit assignment.
    
    Maintains a decaying trace of past synaptic activity that can be
    converted to weight changes when reward signals arrive.
    """
    
    def __init__(
        self,
        tau_eligibility: float = 2000.0,
        trace_clip: float = 1.0,
        accumulate: bool = True,
        **kwargs
    ):
        """Initialize eligibility trace rule.
        
        Args:
            tau_eligibility: Eligibility trace decay time constant
            trace_clip: Maximum eligibility trace value
            accumulate: Whether to accumulate or replace traces
        """
        super().__init__(
            tau_eligibility=tau_eligibility,
            trace_clip=trace_clip,
            accumulate=accumulate,
            **kwargs
        )
        self.tau_eligibility = tau_eligibility
        self.trace_clip = trace_clip
        self.accumulate = accumulate
    
    @property
    def name(self) -> str:
        return "eligibility_trace"
    
    @property
    def requires(self) -> set[str]:
        return {"eligibility_trace", "trace_pre", "trace_post", "spike"}
    
    @property
    def modifies(self) -> set[str]:
        return {"eligibility_trace"}
    
    def apply(self, state: NeoAgentState, context: RuleContext) -> NeoAgentState:
        """Update eligibility traces."""
        dt = context.dt
        
        # Decay existing eligibility trace
        decay_factor = jnp.exp(-dt / self.tau_eligibility)
        decayed_eligibility = state.eligibility_trace * decay_factor
        
        # Calculate new eligibility from current activity
        n_neurons = state.spike.shape[0]
        n_total_pre = state.eligibility_trace.shape[1]
        n_inputs = n_total_pre - n_neurons
        
        # Create full pre-synaptic traces (inputs + neurons)
        full_trace_pre = jnp.concatenate([state.input_buffer, state.trace_pre])
        
        # Create full pre-synaptic spikes
        full_spike_pre = jnp.concatenate([state.input_buffer > 0.5, state.spike.astype(jnp.float32)])
        
        # Calculate eligibility contributions
        pre_post = jnp.outer(state.spike, full_trace_pre)
        post_pre = jnp.outer(state.trace_post, full_spike_pre)
        new_contribution = pre_post - post_pre
        
        # Update eligibility trace
        if self.accumulate:
            new_eligibility = decayed_eligibility + new_contribution
        else:
            new_eligibility = jnp.where(
                jnp.abs(new_contribution) > jnp.abs(decayed_eligibility),
                new_contribution,
                decayed_eligibility
            )
        
        # Clip eligibility trace
        new_eligibility = jnp.clip(new_eligibility, -self.trace_clip, self.trace_clip)
        
        return state._replace(eligibility_trace=new_eligibility)


@register_rule("reward_prediction", category="neuromodulation",
               description="TD learning for reward prediction")
class RewardPredictionRule(AbstractLearningRule):
    """Reward prediction error calculation using TD learning.
    
    Maintains a value estimate and calculates prediction errors that
    can modulate learning in other rules.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.05,
        discount_factor: float = 0.9,
        eligibility_trace_lambda: float = 0.9,
        **kwargs
    ):
        """Initialize reward prediction rule.
        
        Args:
            learning_rate: Value learning rate
            discount_factor: Reward discount factor (gamma)
            eligibility_trace_lambda: TD(lambda) trace decay
        """
        super().__init__(
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            eligibility_trace_lambda=eligibility_trace_lambda,
            **kwargs
        )
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.eligibility_trace_lambda = eligibility_trace_lambda
    
    @property
    def name(self) -> str:
        return "reward_prediction"
    
    @property
    def requires(self) -> set[str]:
        return {"value_estimate", "reward_prediction_error"}
    
    @property
    def modifies(self) -> set[str]:
        return {"value_estimate", "reward_prediction_error"}
    
    def apply(self, state: NeoAgentState, context: RuleContext) -> NeoAgentState:
        """Update value estimate and calculate RPE."""
        # Calculate TD error (reward prediction error)
        next_value = state.value_estimate  # Could be from a critic network
        td_error = (
            context.reward + 
            self.discount_factor * next_value - 
            state.value_estimate
        )
        
        # Update value estimate
        new_value = state.value_estimate + self.learning_rate * td_error
        
        return state._replace(
            value_estimate=new_value,
            reward_prediction_error=td_error
        )


@register_rule("three_factor", category="neuromodulation",
               description="Complete three-factor learning rule")
class ThreeFactorRule(AbstractLearningRule):
    """Complete three-factor learning rule.
    
    Combines Hebbian plasticity, eligibility traces, and neuromodulation
    into a unified learning rule.
    """
    
    def __init__(
        self,
        stdp_a_plus: float = 0.02,
        stdp_a_minus: float = 0.021,
        tau_eligibility: float = 2000.0,
        baseline_dopamine: float = 0.2,
        tau_dopamine: float = 200.0,
        learning_rate: float = 0.1,
        **kwargs
    ):
        """Initialize three-factor rule."""
        super().__init__(
            stdp_a_plus=stdp_a_plus,
            stdp_a_minus=stdp_a_minus,
            tau_eligibility=tau_eligibility,
            baseline_dopamine=baseline_dopamine,
            tau_dopamine=tau_dopamine,
            learning_rate=learning_rate,
            **kwargs
        )
        self.stdp_a_plus = stdp_a_plus
        self.stdp_a_minus = stdp_a_minus
        self.tau_eligibility = tau_eligibility
        self.baseline_dopamine = baseline_dopamine
        self.tau_dopamine = tau_dopamine
        self.learning_rate = learning_rate
    
    @property
    def name(self) -> str:
        return "three_factor"
    
    @property
    def requires(self) -> set[str]:
        return {
            "w", "w_plastic_mask", "spike", "trace_pre", "trace_post",
            "eligibility_trace", "dopamine", "learning_rate"
        }
    
    @property
    def modifies(self) -> set[str]:
        return {"w", "eligibility_trace", "dopamine"}
    
    def apply(self, state: NeoAgentState, context: RuleContext) -> NeoAgentState:
        """Apply complete three-factor learning."""
        dt = context.dt
        
        # Factor 1: Hebbian (STDP)
        ltp = self.stdp_a_plus * jnp.outer(state.spike, state.trace_pre)
        ltd = self.stdp_a_minus * jnp.outer(state.trace_post, state.spike)
        hebbian = ltp - ltd
        
        # Factor 2: Eligibility trace
        eligibility_decay = jnp.exp(-dt / self.tau_eligibility)
        new_eligibility = state.eligibility_trace * eligibility_decay + hebbian
        
        # Factor 3: Neuromodulation (dopamine)
        dopamine_decay = jnp.exp(-dt / self.tau_dopamine)
        reward_signal = context.reward * 5.0  # Reward scaling
        new_dopamine = (
            state.dopamine * dopamine_decay +
            self.baseline_dopamine * (1 - dopamine_decay) +
            reward_signal
        )
        
        # Modulation factor
        da_factor = (new_dopamine - self.baseline_dopamine) / self.baseline_dopamine
        modulation = jax.nn.tanh(da_factor * 2.0)
        
        # Weight update combining all three factors
        dw = self.learning_rate * modulation * new_eligibility * state.w_plastic_mask
        w_new = state.w + dw
        
        return state._replace(
            w=w_new,
            eligibility_trace=new_eligibility,
            dopamine=new_dopamine
        )