# keywords: [rule pipeline, composition, jax jit, efficient compilation]
"""
Rule pipeline for composing learning rules efficiently.

The pipeline system allows multiple rules to be combined into a single
JIT-compiled function, eliminating runtime conditionals and maximizing
performance.
"""

from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
from jax import jit

from .base import LearningRule, NetworkState, RuleContext


class RulePipeline:
    """Efficient pipeline for composing multiple learning rules.
    
    The pipeline compiles all rules into a single JIT-compiled function,
    avoiding runtime overhead from rule selection or conditionals.
    """
    
    def __init__(
        self,
        rules: List[LearningRule],
        name: Optional[str] = None,
        jit_compile: bool = True,
    ):
        """Initialize the rule pipeline.
        
        Args:
            rules: List of learning rules to apply in order
            name: Optional name for the pipeline
            jit_compile: Whether to JIT compile the pipeline
        """
        self.rules = rules
        self.name = name or f"Pipeline[{len(rules)} rules]"
        self.jit_compile = jit_compile
        
        # Check rule compatibility
        self._validate_rules()
        
        # Create the pipeline function
        self._pipeline_fn = self._create_pipeline()
        
        # JIT compile if requested
        if jit_compile:
            self.apply = jit(self._pipeline_fn)
        else:
            self.apply = self._pipeline_fn
    
    def _validate_rules(self) -> None:
        """Validate that rules are compatible."""
        # Check that modified fields don't conflict
        all_modifies = set()
        for rule in self.rules:
            modifies = rule.modifies
            # For now, we allow multiple rules to modify the same field
            # In the future, we might want to check for conflicts
            all_modifies.update(modifies)
        
        # Check that required fields are available
        available_fields = set(NetworkState._fields)
        for rule in self.rules:
            missing = rule.requires - available_fields
            if missing:
                raise ValueError(
                    f"Rule {rule.name} requires fields {missing} "
                    f"which are not in NetworkState"
                )
    
    def _create_pipeline(self) -> Callable:
        """Create the pipeline function."""
        def pipeline(state: NetworkState, context: RuleContext) -> NetworkState:
            # Apply each rule in sequence
            for rule in self.rules:
                state = rule.apply(state, context)
            return state
        
        return pipeline
    
    def initialize(self, sample_state: NetworkState) -> None:
        """Initialize all rules in the pipeline."""
        for rule in self.rules:
            if hasattr(rule, 'initialize'):
                rule.initialize(sample_state)
    
    @property
    def requires(self) -> set[str]:
        """Union of all rule requirements."""
        return set().union(*(rule.requires for rule in self.rules))
    
    @property
    def modifies(self) -> set[str]:
        """Union of all rule modifications."""
        return set().union(*(rule.modifies for rule in self.rules))
    
    def __repr__(self) -> str:
        rule_names = [r.name for r in self.rules]
        return f"RulePipeline({self.name}, rules={rule_names})"


def create_rule_pipeline(
    rule_configs: List[Dict[str, Any]],
    available_rules: Optional[Dict[str, type[LearningRule]]] = None,
    jit_compile: bool = True,
) -> RulePipeline:
    """Create a rule pipeline from configuration.
    
    Args:
        rule_configs: List of rule configurations, each with 'name' and 'params'
        available_rules: Dict mapping rule names to rule classes
        jit_compile: Whether to JIT compile the pipeline
        
    Returns:
        Configured rule pipeline
        
    Example:
        configs = [
            {"name": "stdp", "params": {"a_plus": 0.02, "a_minus": 0.021}},
            {"name": "homeostatic", "params": {"tau": 10000}},
            {"name": "dale", "params": {}},
        ]
        pipeline = create_rule_pipeline(configs)
    """
    if available_rules is None:
        from .registry import RuleRegistry
        available_rules = RuleRegistry.get_all_rules()
    
    rules = []
    for config in rule_configs:
        rule_name = config["name"]
        rule_params = config.get("params", {})
        
        if rule_name not in available_rules:
            raise ValueError(f"Unknown rule: {rule_name}")
        
        rule_class = available_rules[rule_name]
        rule = rule_class(**rule_params)
        rules.append(rule)
    
    return RulePipeline(rules, jit_compile=jit_compile)


class ConditionalPipeline(RulePipeline):
    """Pipeline that can conditionally apply rules based on static conditions.
    
    This is useful for A/B testing or gradual rule introduction, but conditions
    must be static (known at compile time) to maintain JAX compatibility.
    """
    
    def __init__(
        self,
        rules: List[Tuple[LearningRule, bool]],
        name: Optional[str] = None,
        jit_compile: bool = True,
    ):
        """Initialize conditional pipeline.
        
        Args:
            rules: List of (rule, enabled) tuples
            name: Optional pipeline name
            jit_compile: Whether to JIT compile
        """
        # Filter to only enabled rules
        enabled_rules = [rule for rule, enabled in rules if enabled]
        super().__init__(enabled_rules, name, jit_compile)
        
        # Store original rules for introspection
        self._all_rules = rules
    
    @classmethod
    def from_config(
        cls,
        rule_configs: List[Dict[str, Any]],
        enabled_rules: set[str],
        available_rules: Optional[Dict[str, type[LearningRule]]] = None,
        jit_compile: bool = True,
    ) -> "ConditionalPipeline":
        """Create conditional pipeline from config with enabled set.
        
        Args:
            rule_configs: All possible rule configurations
            enabled_rules: Set of rule names to enable
            available_rules: Available rule classes
            jit_compile: Whether to JIT compile
            
        Returns:
            Conditional pipeline with only enabled rules active
        """
        if available_rules is None:
            from .registry import RuleRegistry
            available_rules = RuleRegistry.get_all_rules()
        
        rules_with_flags = []
        for config in rule_configs:
            rule_name = config["name"]
            rule_params = config.get("params", {})
            
            if rule_name in available_rules:
                rule_class = available_rules[rule_name]
                rule = rule_class(**rule_params)
                enabled = rule_name in enabled_rules
                rules_with_flags.append((rule, enabled))
        
        return cls(rules_with_flags, jit_compile=jit_compile)