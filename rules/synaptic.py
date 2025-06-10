# keywords: [synaptic plasticity, stdp, hebbian learning, jax implementation]
"""
Synaptic plasticity learning rules.

This module implements various forms of synaptic plasticity including
STDP (Spike-Timing-Dependent Plasticity) and Hebbian learning.
"""

from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
from jax import Array

from .base import AbstractLearningRule, RuleContext
from .registry import register_rule
from models.phase_0_14_neo.state import NeoAgentState


@register_rule("stdp", category="synaptic", description="Spike-Timing-Dependent Plasticity")
class STDPRule(AbstractLearningRule):
    """Spike-Timing-Dependent Plasticity (STDP) rule.
    
    Implements classical STDP where the weight change depends on the
    relative timing of pre- and post-synaptic spikes.
    """
    
    def __init__(
        self,
        a_plus: float = 0.02,
        a_minus: float = 0.021,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
        w_max: Optional[float] = None,
        **kwargs
    ):
        """Initialize STDP rule.
        
        Args:
            a_plus: LTP (potentiation) amplitude
            a_minus: LTD (depression) amplitude
            tau_plus: Pre-synaptic trace time constant
            tau_minus: Post-synaptic trace time constant
            w_max: Optional maximum weight for soft bounds
        """
        super().__init__(
            a_plus=a_plus,
            a_minus=a_minus,
            tau_plus=tau_plus,
            tau_minus=tau_minus,
            w_max=w_max,
            **kwargs
        )
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.w_max = w_max
    
    @property
    def name(self) -> str:
        return "stdp"
    
    @property
    def requires(self) -> set[str]:
        return {"w", "w_plastic_mask", "spike", "trace_pre", "trace_post", "timestep"}
    
    @property
    def modifies(self) -> set[str]:
        return {"w", "trace_pre", "trace_post"}
    
    def apply(self, state: NeoAgentState, context: RuleContext) -> NeoAgentState:
        """Apply STDP rule."""
        # Update traces with exponential decay
        dt = context.dt
        trace_pre_decay = jnp.exp(-dt / self.tau_plus)
        trace_post_decay = jnp.exp(-dt / self.tau_minus)
        
        # Update traces
        new_trace_pre = state.trace_pre * trace_pre_decay + state.spike
        new_trace_post = state.trace_post * trace_post_decay + state.spike
        
        # STDP weight updates
        # The weight matrix has shape (n_neurons, n_inputs + n_neurons)
        # We need to handle both input-to-neuron and neuron-to-neuron connections
        
        # Get dimensions
        n_post = state.w.shape[0]  # post-synaptic neurons
        n_pre_total = state.w.shape[1]  # total pre-synaptic (inputs + neurons)
        n_neurons = state.spike.shape[0]
        n_inputs = n_pre_total - n_neurons
        
        # Create pre-synaptic activity vector (inputs + neurons)
        pre_spikes = jnp.concatenate([state.input_buffer > 0.5, state.spike.astype(jnp.float32)])
        
        # Create pre-synaptic trace vector
        pre_traces = jnp.concatenate([state.input_buffer, state.trace_pre])
        
        # LTP: pre-synaptic trace * post-synaptic spike
        ltp = self.a_plus * jnp.outer(state.spike, pre_traces)
        
        # LTD: pre-synaptic spike * post-synaptic trace  
        ltd = self.a_minus * jnp.outer(state.trace_post, pre_spikes)
        
        # Total weight change
        dw = (ltp - ltd) * state.w_plastic_mask
        
        # Update weights
        w_new = state.w + dw
        
        # Apply soft bounds if specified
        if self.w_max is not None:
            w_new = jnp.clip(w_new, 0, self.w_max)
        
        return state._replace(
            w=w_new,
            trace_pre=new_trace_pre,
            trace_post=new_trace_post
        )


@register_rule("hebbian", category="synaptic", description="Basic Hebbian learning")
class HebbianRule(AbstractLearningRule):
    """Basic Hebbian learning rule.
    
    Implements the principle "neurons that fire together, wire together"
    with optional normalization and decay.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        normalize: bool = True,
        decay_rate: float = 0.0,
        **kwargs
    ):
        """Initialize Hebbian rule.
        
        Args:
            learning_rate: Learning rate
            normalize: Whether to normalize by activity levels
            decay_rate: Weight decay rate
        """
        super().__init__(
            learning_rate=learning_rate,
            normalize=normalize,
            decay_rate=decay_rate,
            **kwargs
        )
        self.learning_rate = learning_rate
        self.normalize = normalize
        self.decay_rate = decay_rate
    
    @property
    def name(self) -> str:
        return "hebbian"
    
    @property
    def requires(self) -> set[str]:
        return {"w", "w_plastic_mask", "spike"}
    
    @property
    def modifies(self) -> set[str]:
        return {"w"}
    
    def apply(self, state: NeoAgentState, context: RuleContext) -> NeoAgentState:
        """Apply Hebbian learning rule."""
        # Convert spikes to float
        pre_activity = state.spike.astype(jnp.float32)
        post_activity = state.spike.astype(jnp.float32)
        
        # Hebbian update: correlated activity
        hebbian_term = jnp.outer(post_activity, pre_activity)
        
        # Normalize by activity levels if requested
        if self.normalize:
            pre_sum = jnp.sum(pre_activity) + 1e-8
            post_sum = jnp.sum(post_activity) + 1e-8
            normalization = jnp.sqrt(pre_sum * post_sum)
            hebbian_term = hebbian_term / normalization
        
        # Apply learning rate and plasticity mask
        dw = self.learning_rate * hebbian_term * state.w_plastic_mask
        
        # Add weight decay if specified
        if self.decay_rate > 0:
            dw = dw - self.decay_rate * state.w * state.w_plastic_mask
        
        # Update weights
        w_new = state.w + dw
        
        return state._replace(w=w_new)


@register_rule("covariance", category="synaptic", description="Covariance-based learning")
class CovarianceRule(AbstractLearningRule):
    """Covariance-based learning rule.
    
    Updates weights based on the covariance between pre- and post-synaptic
    activity, implementing a form of BCM (Bienenstock-Cooper-Munro) rule.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        target_rate: float = 5.0,
        threshold_tau: float = 1000.0,
        **kwargs
    ):
        """Initialize covariance rule.
        
        Args:
            learning_rate: Learning rate
            target_rate: Target firing rate (Hz)
            threshold_tau: Time constant for threshold adaptation
        """
        super().__init__(
            learning_rate=learning_rate,
            target_rate=target_rate,
            threshold_tau=threshold_tau,
            **kwargs
        )
        self.learning_rate = learning_rate
        self.target_rate = target_rate
        self.threshold_tau = threshold_tau
    
    @property
    def name(self) -> str:
        return "covariance"
    
    @property
    def requires(self) -> set[str]:
        return {"w", "w_plastic_mask", "spike", "firing_rate"}
    
    @property
    def modifies(self) -> set[str]:
        return {"w"}
    
    def apply(self, state: NeoAgentState, context: RuleContext) -> NeoAgentState:
        """Apply covariance learning rule."""
        # Get firing rates
        post_rate = state.firing_rate
        
        # BCM-like threshold (theta)
        theta = self.target_rate
        
        # Covariance term: (post_rate - theta) * pre_spike
        post_term = (post_rate - theta)[:, None]
        pre_term = state.spike[None, :].astype(jnp.float32)
        
        # Weight update
        dw = self.learning_rate * post_term * pre_term * state.w_plastic_mask
        
        # Update weights
        w_new = state.w + dw
        
        return state._replace(w=w_new)