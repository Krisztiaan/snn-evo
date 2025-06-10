# keywords: [rule registry, plugin system, rule discovery]
"""
Registry system for learning rules.

This module provides a central registry for all available learning rules,
making it easy to discover, instantiate, and compose rules dynamically.
"""

from typing import Dict, List, Optional, Type

from .base import LearningRule


class RuleRegistry:
    """Central registry for all available learning rules."""
    
    _rules: Dict[str, Type[LearningRule]] = {}
    _rule_metadata: Dict[str, Dict[str, any]] = {}
    
    @classmethod
    def register(
        cls,
        name: str,
        rule_class: Type[LearningRule],
        category: Optional[str] = None,
        description: Optional[str] = None,
        **metadata
    ) -> None:
        """Register a learning rule.
        
        Args:
            name: Unique name for the rule
            rule_class: The rule class to register
            category: Optional category (e.g., "synaptic", "homeostatic")
            description: Optional description of the rule
            **metadata: Additional metadata
        """
        if name in cls._rules:
            raise ValueError(f"Rule '{name}' is already registered")
        
        cls._rules[name] = rule_class
        cls._rule_metadata[name] = {
            "category": category,
            "description": description,
            **metadata
        }
    
    @classmethod
    def get(cls, name: str) -> Type[LearningRule]:
        """Get a rule class by name."""
        if name not in cls._rules:
            available = ", ".join(cls._rules.keys())
            raise ValueError(f"Unknown rule: {name}. Available: {available}")
        return cls._rules[name]
    
    @classmethod
    def get_all_rules(cls) -> Dict[str, Type[LearningRule]]:
        """Get all registered rules."""
        return cls._rules.copy()
    
    @classmethod
    def get_rules_by_category(cls, category: str) -> Dict[str, Type[LearningRule]]:
        """Get all rules in a specific category."""
        return {
            name: rule_class
            for name, rule_class in cls._rules.items()
            if cls._rule_metadata[name].get("category") == category
        }
    
    @classmethod
    def list_rules(cls) -> List[str]:
        """List all registered rule names."""
        return list(cls._rules.keys())
    
    @classmethod
    def get_metadata(cls, name: str) -> Dict[str, any]:
        """Get metadata for a rule."""
        if name not in cls._rule_metadata:
            raise ValueError(f"No metadata for rule: {name}")
        return cls._rule_metadata[name].copy()
    
    @classmethod
    def describe_rule(cls, name: str) -> str:
        """Get a description of a rule."""
        if name not in cls._rules:
            raise ValueError(f"Unknown rule: {name}")
        
        metadata = cls._rule_metadata[name]
        rule_class = cls._rules[name]
        
        lines = [f"Rule: {name}"]
        if metadata.get("category"):
            lines.append(f"Category: {metadata['category']}")
        if metadata.get("description"):
            lines.append(f"Description: {metadata['description']}")
        
        # Try to get requirements and modifications
        try:
            instance = rule_class()
            lines.append(f"Requires: {', '.join(sorted(instance.requires))}")
            lines.append(f"Modifies: {', '.join(sorted(instance.modifies))}")
        except:
            pass
        
        return "\n".join(lines)
    
    @classmethod
    def clear(cls) -> None:
        """Clear all registered rules (mainly for testing)."""
        cls._rules.clear()
        cls._rule_metadata.clear()


def register_rule(
    name: Optional[str] = None,
    category: Optional[str] = None,
    description: Optional[str] = None,
    **metadata
):
    """Decorator to register a learning rule.
    
    Usage:
        @register_rule("my_rule", category="synaptic")
        class MyRule(AbstractLearningRule):
            ...
    """
    def decorator(rule_class: Type[LearningRule]) -> Type[LearningRule]:
        rule_name = name or rule_class.__name__.lower().replace("rule", "")
        RuleRegistry.register(
            rule_name,
            rule_class,
            category=category,
            description=description,
            **metadata
        )
        return rule_class
    
    return decorator


def get_rule(name: str, **params) -> LearningRule:
    """Get an instantiated rule by name.
    
    Args:
        name: Rule name
        **params: Parameters to pass to rule constructor
        
    Returns:
        Instantiated rule
    """
    rule_class = RuleRegistry.get(name)
    return rule_class(**params)