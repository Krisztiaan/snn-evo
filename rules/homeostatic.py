# keywords: [homeostatic plasticity, firing rate, threshold adaptation, jax]
"""
Homeostatic plasticity rules.

This module implements homeostatic mechanisms that maintain stable
neural activity levels over longer timescales.
"""

import jax
import jax.numpy as jnp
from jax import Array

from .base import AbstractLearningRule, NetworkState, RuleContext
from .registry import register_rule


@register_rule("homeostatic", category="homeostatic", description="Firing rate homeostasis")
class HomeostaticRule(AbstractLearningRule):
    """Homeostatic plasticity rule for maintaining target firing rates.
    
    This rule adjusts synaptic weights to maintain stable firing rates
    across the network, preventing runaway excitation or silencing.
    """
    
    def __init__(
        self,
        tau: float = 10000.0,
        target_rate: float = 5.0,
        learning_rate: float = 0.001,
        multiplicative: bool = True,
        **kwargs
    ):
        """Initialize homeostatic rule.
        
        Args:
            tau: Time constant for firing rate estimation (ms)
            target_rate: Target firing rate (Hz)
            learning_rate: Homeostatic learning rate
            multiplicative: Whether to use multiplicative (True) or additive updates
        """
        super().__init__(
            tau=tau,
            target_rate=target_rate,
            learning_rate=learning_rate,
            multiplicative=multiplicative,
            **kwargs
        )
        self.tau = tau
        self.target_rate = target_rate
        self.learning_rate = learning_rate
        self.multiplicative = multiplicative
    
    @property
    def name(self) -> str:
        return "homeostatic"
    
    @property
    def requires(self) -> set[str]:
        return {"w", "w_plastic_mask", "spike", "firing_rate"}
    
    @property
    def modifies(self) -> set[str]:
        return {"w", "firing_rate"}
    
    def apply(self, state: NetworkState, context: RuleContext) -> NetworkState:
        """Apply homeostatic plasticity."""
        dt = context.dt
        
        # Update firing rate estimate with exponential moving average
        alpha = dt / self.tau
        spike_rate = state.spike.astype(jnp.float32) * 1000.0 / dt  # Convert to Hz
        new_firing_rate = (1 - alpha) * state.firing_rate + alpha * spike_rate
        
        # Calculate homeostatic error
        rate_error = new_firing_rate - self.target_rate
        
        if self.multiplicative:
            # Multiplicative homeostatic scaling
            # Scale all incoming weights by a factor
            scaling_factor = 1.0 - self.learning_rate * rate_error / self.target_rate
            scaling_factor = jnp.clip(scaling_factor, 0.5, 2.0)  # Prevent extreme scaling
            
            # Apply scaling to incoming weights (row-wise for each post-synaptic neuron)
            # scaling_factor has shape (n_neurons,), we need to broadcast correctly
            w_new = state.w * scaling_factor[:, None] * state.w_plastic_mask
        else:
            # Additive homeostatic adjustment
            # Adjust all incoming weights by a fixed amount
            adjustment = -self.learning_rate * rate_error[:, None]
            w_new = state.w + adjustment * state.w_plastic_mask
        
        return state._replace(
            w=w_new,
            firing_rate=new_firing_rate
        )


@register_rule("threshold_adaptation", category="homeostatic", 
               description="Adaptive threshold for firing rate control")
class ThresholdAdaptationRule(AbstractLearningRule):
    """Adaptive threshold rule for firing rate homeostasis.
    
    Instead of adjusting weights, this rule modifies firing thresholds
    to maintain target activity levels.
    """
    
    def __init__(
        self,
        tau: float = 10000.0,
        target_rate: float = 5.0,
        adapt_rate: float = 0.0001,
        max_adaptation: float = 5.0,
        **kwargs
    ):
        """Initialize threshold adaptation rule.
        
        Args:
            tau: Time constant for firing rate estimation
            target_rate: Target firing rate (Hz)
            adapt_rate: Threshold adaptation rate
            max_adaptation: Maximum threshold change (mV)
        """
        super().__init__(
            tau=tau,
            target_rate=target_rate,
            adapt_rate=adapt_rate,
            max_adaptation=max_adaptation,
            **kwargs
        )
        self.tau = tau
        self.target_rate = target_rate
        self.adapt_rate = adapt_rate
        self.max_adaptation = max_adaptation
    
    @property
    def name(self) -> str:
        return "threshold_adaptation"
    
    @property
    def requires(self) -> set[str]:
        return {"spike", "firing_rate", "threshold_adapt"}
    
    @property
    def modifies(self) -> set[str]:
        return {"firing_rate", "threshold_adapt"}
    
    def apply(self, state: NetworkState, context: RuleContext) -> NetworkState:
        """Apply threshold adaptation."""
        dt = context.dt
        
        # Update firing rate estimate
        alpha = dt / self.tau
        spike_rate = state.spike.astype(jnp.float32) * 1000.0 / dt  # Hz
        new_firing_rate = (1 - alpha) * state.firing_rate + alpha * spike_rate
        
        # Calculate rate error
        rate_error = new_firing_rate - self.target_rate
        
        # Update threshold adaptation
        threshold_change = self.adapt_rate * rate_error
        new_threshold_adapt = state.threshold_adapt + threshold_change
        
        # Clip to maximum adaptation
        new_threshold_adapt = jnp.clip(
            new_threshold_adapt,
            -self.max_adaptation,
            self.max_adaptation
        )
        
        return state._replace(
            firing_rate=new_firing_rate,
            threshold_adapt=new_threshold_adapt
        )


@register_rule("synaptic_scaling", category="homeostatic",
               description="Synaptic scaling for network stability")
class SynapticScalingRule(AbstractLearningRule):
    """Synaptic scaling rule for global network stability.
    
    Scales synaptic weights to maintain stable total synaptic input
    while preserving relative weight differences.
    """
    
    def __init__(
        self,
        target_input: float = 10.0,
        tau: float = 50000.0,
        learning_rate: float = 0.0001,
        **kwargs
    ):
        """Initialize synaptic scaling rule.
        
        Args:
            target_input: Target total synaptic input
            tau: Time constant for input estimation
            learning_rate: Scaling adjustment rate
        """
        super().__init__(
            target_input=target_input,
            tau=tau,
            learning_rate=learning_rate,
            **kwargs
        )
        self.target_input = target_input
        self.tau = tau
        self.learning_rate = learning_rate
        
        # We'll need to track average input
        self._avg_input = None
    
    @property
    def name(self) -> str:
        return "synaptic_scaling"
    
    @property
    def requires(self) -> set[str]:
        return {"w", "w_plastic_mask", "syn_current_e", "syn_current_i"}
    
    @property
    def modifies(self) -> set[str]:
        return {"w"}
    
    def initialize(self, sample_state: NetworkState) -> None:
        """Initialize average input tracker."""
        super().initialize(sample_state)
        self._avg_input = jnp.zeros(len(sample_state.v))
    
    def apply(self, state: NetworkState, context: RuleContext) -> NetworkState:
        """Apply synaptic scaling."""
        dt = context.dt
        
        # Calculate total synaptic input
        total_input = jnp.abs(state.syn_current_e) + jnp.abs(state.syn_current_i)
        
        # Update average input estimate
        alpha = dt / self.tau
        if self._avg_input is None:
            self._avg_input = total_input
        else:
            self._avg_input = (1 - alpha) * self._avg_input + alpha * total_input
        
        # Calculate scaling factor
        input_error = self._avg_input - self.target_input
        scaling_factor = 1.0 - self.learning_rate * input_error / self.target_input
        scaling_factor = jnp.clip(scaling_factor, 0.8, 1.2)  # Limit scaling
        
        # Apply scaling to all incoming weights
        w_new = state.w * scaling_factor[None, :] * state.w_plastic_mask
        
        return state._replace(w=w_new)