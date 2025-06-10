# keywords: [learning rule base, network state, rule context, jax types]
"""
Base classes and interfaces for the learning rules system.

This module defines the core types and protocols that all learning rules
must implement to be compatible with the composable pipeline.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, NamedTuple, Optional, Protocol, runtime_checkable

import jax
import jax.numpy as jnp
from jax import Array




class RuleContext(NamedTuple):
    """Shared context passed to all rules.
    
    Contains pre-computed values and external inputs that multiple
    rules might need, avoiding redundant computation.
    """
    # External inputs
    reward: float  # Current reward signal
    observation: Array  # Current sensory input
    action: Optional[int]  # Last action taken
    
    # Pre-computed spike data
    spike_count: Array  # Number of spikes per neuron
    population_rate: float  # Overall population firing rate
    
    # Pre-computed synaptic data  
    pre_spike_sum: Array  # Sum of pre-synaptic spikes
    post_spike_sum: Array  # Sum of post-synaptic spikes
    
    # Timing
    dt: float  # Timestep size
    episode_progress: float  # Progress through episode (0-1)
    
    # Configuration
    params: Dict[str, Any]  # Rule-specific parameters


@runtime_checkable
class LearningRule(Protocol):
    """Protocol that all learning rules must implement.
    
    Rules are pure functions that transform network state based on
    the current context. They should be JAX-compatible (no side effects,
    no Python control flow that depends on traced values).
    """
    
    @property
    def name(self) -> str:
        """Unique name for this rule."""
        ...
    
    @property
    def requires(self) -> set[str]:
        """Set of state fields this rule requires to be present."""
        ...
    
    @property
    def modifies(self) -> set[str]:
        """Set of state fields this rule modifies."""
        ...
    
    def apply(self, state: Any, context: RuleContext) -> Any:
        """Apply the learning rule to transform the network state.
        
        Args:
            state: Current network state
            context: Shared context with pre-computed values
            
        Returns:
            Modified network state
        """
        ...


class AbstractLearningRule(ABC):
    """Base class for learning rules with common functionality."""
    
    def __init__(self, **params):
        """Initialize rule with parameters."""
        self.params = params
        self._initialized = False
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this rule."""
        pass
    
    @property
    @abstractmethod
    def requires(self) -> set[str]:
        """Set of state fields this rule requires."""
        pass
    
    @property
    @abstractmethod
    def modifies(self) -> set[str]:
        """Set of state fields this rule modifies."""
        pass
    
    @abstractmethod
    def apply(self, state: Any, context: RuleContext) -> Any:
        """Apply the learning rule."""
        pass
    
    def initialize(self, sample_state: Any) -> None:
        """Initialize rule based on network architecture.
        
        Called once before the rule is used, allowing it to set up
        any architecture-dependent parameters.
        """
        self._initialized = True
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.params})"


class ComposedRule(AbstractLearningRule):
    """A rule composed of multiple sub-rules applied in sequence."""
    
    def __init__(self, rules: list[LearningRule], name: Optional[str] = None):
        super().__init__()
        self.rules = rules
        self._name = name or f"Composed[{','.join(r.name for r in rules)}]"
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def requires(self) -> set[str]:
        """Union of all sub-rule requirements."""
        return set().union(*(rule.requires for rule in self.rules))
    
    @property
    def modifies(self) -> set[str]:
        """Union of all sub-rule modifications."""
        return set().union(*(rule.modifies for rule in self.rules))
    
    def apply(self, state: Any, context: RuleContext) -> Any:
        """Apply all sub-rules in sequence."""
        for rule in self.rules:
            state = rule.apply(state, context)
        return state
    
    def initialize(self, sample_state: Any) -> None:
        """Initialize all sub-rules."""
        super().initialize(sample_state)
        for rule in self.rules:
            if hasattr(rule, 'initialize'):
                rule.initialize(sample_state)