# keywords: [structural constraints, dale principle, weight bounds, weight decay]
"""
Structural constraint rules.

This module implements rules that enforce structural constraints on
the network, such as Dale's principle, weight bounds, and weight decay.
"""

import jax
import jax.numpy as jnp
from jax import Array

from .base import AbstractLearningRule, RuleContext
from .registry import register_rule
from models.phase_0_14_neo.state import NeoAgentState


@register_rule("dale_principle", category="structural",
               description="Enforce Dale's principle")
class DalePrincipleRule(AbstractLearningRule):
    """Enforce Dale's principle - neurons are either excitatory or inhibitory.
    
    This rule ensures that all outgoing connections from a neuron have
    the same sign (all positive for excitatory, all negative for inhibitory).
    """
    
    def __init__(self, strict: bool = True, **kwargs):
        """Initialize Dale's principle rule.
        
        Args:
            strict: If True, enforce strict sign constraints.
                   If False, allow small violations.
        """
        super().__init__(strict=strict, **kwargs)
        self.strict = strict
    
    @property
    def name(self) -> str:
        return "dale_principle"
    
    @property
    def requires(self) -> set[str]:
        return {"w", "is_excitatory"}
    
    @property
    def modifies(self) -> set[str]:
        return {"w"}
    
    def apply(self, state: NeoAgentState, context: RuleContext) -> NeoAgentState:
        """Enforce Dale's principle on weights."""
        # Create sign mask: +1 for excitatory, -1 for inhibitory
        sign_mask = jnp.where(state.is_excitatory, 1.0, -1.0)
        
        # Get dimensions
        n_post = state.w.shape[0]
        n_pre_total = state.w.shape[1]
        n_neurons = state.spike.shape[0]
        n_inputs = n_pre_total - n_neurons
        
        # Create full pre-synaptic sign mask (inputs are excitatory + neuron signs)
        pre_sign_mask = jnp.concatenate([
            jnp.ones(n_inputs),  # Input connections are excitatory
            sign_mask  # Neuron signs
        ])
        
        if self.strict:
            # Strict enforcement: force all weights to have correct sign
            # Only apply to recurrent connections
            w_new = state.w.copy()
            # Recurrent part: apply Dale's principle
            w_recurrent = state.w[:, n_inputs:]
            w_recurrent_new = jnp.abs(w_recurrent) * sign_mask[:, None]
            w_new = w_new.at[:, n_inputs:].set(w_recurrent_new)
        else:
            # Soft enforcement: only flip weights that have wrong sign
            # For recurrent connections only
            w_new = state.w.copy()
            w_recurrent = state.w[:, n_inputs:]
            correct_sign = w_recurrent * sign_mask[:, None] > 0
            w_recurrent_new = jnp.where(correct_sign, w_recurrent, -w_recurrent)
            w_new = w_new.at[:, n_inputs:].set(w_recurrent_new)
        
        return state._replace(w=w_new)


@register_rule("weight_bounds", category="structural",
               description="Enforce weight bounds")
class WeightBoundsRule(AbstractLearningRule):
    """Enforce bounds on synaptic weights.
    
    Can implement hard bounds (clipping) or soft bounds (sigmoid-like).
    """
    
    def __init__(
        self,
        w_min: float = 0.0,
        w_max: float = 5.0,
        soft_bounds: bool = True,
        bound_scale: float = 0.1,
        separate_ei: bool = True,
        **kwargs
    ):
        """Initialize weight bounds rule.
        
        Args:
            w_min: Minimum weight value
            w_max: Maximum weight value
            soft_bounds: Use soft bounds (tanh) instead of hard clipping
            bound_scale: Scale factor for soft bounds
            separate_ei: Use separate bounds for E and I connections
        """
        super().__init__(
            w_min=w_min,
            w_max=w_max,
            soft_bounds=soft_bounds,
            bound_scale=bound_scale,
            separate_ei=separate_ei,
            **kwargs
        )
        self.w_min = w_min
        self.w_max = w_max
        self.soft_bounds = soft_bounds
        self.bound_scale = bound_scale
        self.separate_ei = separate_ei
    
    @property
    def name(self) -> str:
        return "weight_bounds"
    
    @property
    def requires(self) -> set[str]:
        return {"w"} | ({"is_excitatory"} if self.separate_ei else set())
    
    @property
    def modifies(self) -> set[str]:
        return {"w"}
    
    def apply(self, state: NeoAgentState, context: RuleContext) -> NeoAgentState:
        """Apply weight bounds."""
        if self.soft_bounds:
            # Soft bounds using tanh
            if self.separate_ei:
                # Different bounds for E and I neurons
                # Get dimensions
                n_pre_total = state.w.shape[1]
                n_neurons = state.spike.shape[0]
                n_inputs = n_pre_total - n_neurons
                
                # Create full excitatory mask (inputs + neurons)
                pre_is_excitatory = jnp.concatenate([
                    jnp.ones(n_inputs, dtype=bool),  # Inputs are excitatory
                    state.is_excitatory
                ])
                e_mask = pre_is_excitatory[None, :]  # Broadcast for pre-synaptic
                
                # Excitatory: [0, w_max]
                w_e = self.w_max * jax.nn.sigmoid(state.w / self.bound_scale)
                
                # Inhibitory: [-w_max, 0]
                w_i = -self.w_max * jax.nn.sigmoid(-state.w / self.bound_scale)
                
                w_new = jnp.where(e_mask, w_e, w_i)
            else:
                # Symmetric bounds: [-w_max, w_max]
                scale = (self.w_max - self.w_min) / 2
                center = (self.w_max + self.w_min) / 2
                w_new = center + scale * jnp.tanh(state.w / self.bound_scale)
        else:
            # Hard bounds using clip
            w_new = jnp.clip(state.w, self.w_min, self.w_max)
        
        return state._replace(w=w_new)


@register_rule("weight_decay", category="structural",
               description="L2 weight decay regularization")
class WeightDecayRule(AbstractLearningRule):
    """Weight decay (L2 regularization) rule.
    
    Gradually decays weights towards zero to prevent overfitting
    and maintain sparse connectivity.
    """
    
    def __init__(
        self,
        decay_rate: float = 0.00001,
        target_weight: float = 0.0,
        exclude_diagonal: bool = True,
        **kwargs
    ):
        """Initialize weight decay rule.
        
        Args:
            decay_rate: Decay rate per timestep
            target_weight: Target weight value (usually 0)
            exclude_diagonal: Don't decay self-connections
        """
        super().__init__(
            decay_rate=decay_rate,
            target_weight=target_weight,
            exclude_diagonal=exclude_diagonal,
            **kwargs
        )
        self.decay_rate = decay_rate
        self.target_weight = target_weight
        self.exclude_diagonal = exclude_diagonal
    
    @property
    def name(self) -> str:
        return "weight_decay"
    
    @property
    def requires(self) -> set[str]:
        return {"w", "w_plastic_mask"}
    
    @property
    def modifies(self) -> set[str]:
        return {"w"}
    
    def apply(self, state: NeoAgentState, context: RuleContext) -> NeoAgentState:
        """Apply weight decay."""
        # Calculate decay
        decay = self.decay_rate * (state.w - self.target_weight)
        
        # Apply plasticity mask
        decay = decay * state.w_plastic_mask
        
        # Exclude diagonal if requested (only for recurrent connections)
        if self.exclude_diagonal:
            n_post = state.w.shape[0]
            n_pre_total = state.w.shape[1]
            n_neurons = state.spike.shape[0]
            n_inputs = n_pre_total - n_neurons
            
            # Create diagonal mask for the full weight matrix
            diagonal_mask = jnp.zeros_like(state.w, dtype=bool)
            # Set diagonal in the recurrent part (offset by n_inputs)
            diagonal_indices = jnp.arange(n_neurons)
            diagonal_mask = diagonal_mask.at[diagonal_indices, diagonal_indices + n_inputs].set(True)
            
            decay = decay * ~diagonal_mask
        
        # Update weights
        w_new = state.w - decay
        
        return state._replace(w=w_new)


@register_rule("weight_normalization", category="structural",
               description="Normalize total synaptic weight")
class WeightNormalizationRule(AbstractLearningRule):
    """Weight normalization to maintain constant total synaptic input.
    
    Normalizes incoming or outgoing weights to maintain stable
    total synaptic strength.
    """
    
    def __init__(
        self,
        norm_type: str = "l2",  # "l1" or "l2"
        target_norm: float = 10.0,
        normalize_incoming: bool = True,
        epsilon: float = 1e-8,
        **kwargs
    ):
        """Initialize weight normalization rule.
        
        Args:
            norm_type: Type of normalization ("l1" or "l2")
            target_norm: Target norm value
            normalize_incoming: Normalize incoming (True) or outgoing weights
            epsilon: Small value to prevent division by zero
        """
        super().__init__(
            norm_type=norm_type,
            target_norm=target_norm,
            normalize_incoming=normalize_incoming,
            epsilon=epsilon,
            **kwargs
        )
        self.norm_type = norm_type
        self.target_norm = target_norm
        self.normalize_incoming = normalize_incoming
        self.epsilon = epsilon
    
    @property
    def name(self) -> str:
        return "weight_normalization"
    
    @property
    def requires(self) -> set[str]:
        return {"w", "w_mask"}
    
    @property
    def modifies(self) -> set[str]:
        return {"w"}
    
    def apply(self, state: NeoAgentState, context: RuleContext) -> NeoAgentState:
        """Apply weight normalization."""
        if self.normalize_incoming:
            # Normalize each neuron's incoming weights (column-wise)
            axis = 0
        else:
            # Normalize each neuron's outgoing weights (row-wise)
            axis = 1
        
        # Calculate norms
        if self.norm_type == "l1":
            norms = jnp.sum(jnp.abs(state.w), axis=axis, keepdims=True)
        else:  # l2
            norms = jnp.sqrt(jnp.sum(state.w ** 2, axis=axis, keepdims=True))
        
        # Prevent division by zero
        norms = jnp.maximum(norms, self.epsilon)
        
        # Calculate scaling factors
        scale_factors = self.target_norm / norms
        
        # Apply normalization only to existing connections
        w_normalized = state.w * scale_factors
        w_new = jnp.where(state.w_mask, w_normalized, state.w)
        
        return state._replace(w=w_new)


@register_rule("sparsity", category="structural",
               description="Enforce weight sparsity")
class SparsityRule(AbstractLearningRule):
    """Enforce sparsity in synaptic weights.
    
    Prunes weak connections to maintain a sparse connectivity pattern.
    """
    
    def __init__(
        self,
        sparsity_level: float = 0.1,  # Keep top 10% of weights
        threshold: float = 0.01,
        hard_sparsity: bool = False,
        **kwargs
    ):
        """Initialize sparsity rule.
        
        Args:
            sparsity_level: Fraction of weights to keep
            threshold: Minimum weight magnitude
            hard_sparsity: If True, set small weights to exactly 0
        """
        super().__init__(
            sparsity_level=sparsity_level,
            threshold=threshold,
            hard_sparsity=hard_sparsity,
            **kwargs
        )
        self.sparsity_level = sparsity_level
        self.threshold = threshold
        self.hard_sparsity = hard_sparsity
    
    @property
    def name(self) -> str:
        return "sparsity"
    
    @property
    def requires(self) -> set[str]:
        return {"w", "w_mask"}
    
    @property
    def modifies(self) -> set[str]:
        return {"w"}
    
    def apply(self, state: NeoAgentState, context: RuleContext) -> NeoAgentState:
        """Apply sparsity constraint."""
        # Get weight magnitudes
        w_abs = jnp.abs(state.w)
        
        if self.hard_sparsity:
            # Hard sparsity: set small weights to 0
            w_new = jnp.where(w_abs < self.threshold, 0.0, state.w)
        else:
            # Soft sparsity: gradually reduce small weights
            # Using a soft threshold function
            soft_threshold = jax.nn.relu(w_abs - self.threshold)
            w_new = jnp.sign(state.w) * soft_threshold
        
        # Maintain connectivity mask
        w_new = w_new * state.w_mask
        
        return state._replace(w=w_new)