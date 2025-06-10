# keywords: [learning rules, modular rules, composable, jax compatible]
"""
Modular Learning Rules System

This package provides a composable, JAX-compatible learning rules framework.
Rules can be combined into pipelines without runtime conditionals, enabling
efficient JIT compilation while maintaining modularity.
"""

from .base import LearningRule, RuleContext, NetworkState
from .pipeline import RulePipeline, create_rule_pipeline
from .registry import RuleRegistry, register_rule, get_rule

# Import individual rules
from .synaptic import STDPRule, HebbianRule
from .homeostatic import HomeostaticRule, ThresholdAdaptationRule
from .neuromodulation import DopamineModulationRule, EligibilityTraceRule, RewardPredictionRule
from .structural import DalePrincipleRule, WeightBoundsRule, WeightDecayRule

__all__ = [
    # Base classes
    "LearningRule",
    "RuleContext", 
    "NetworkState",
    "RulePipeline",
    "create_rule_pipeline",
    
    # Registry
    "RuleRegistry",
    "register_rule",
    "get_rule",
    
    # Synaptic rules
    "STDPRule",
    "HebbianRule",
    
    # Homeostatic rules
    "HomeostaticRule",
    "ThresholdAdaptationRule",
    
    # Neuromodulation rules
    "DopamineModulationRule",
    "EligibilityTraceRule",
    "RewardPredictionRule",
    
    # Structural rules
    "DalePrincipleRule",
    "WeightBoundsRule",
    "WeightDecayRule",
]