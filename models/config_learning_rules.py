#!/usr/bin/env python3
# keywords: [learning rules config, dale principle, homeostasis, plasticity toggles]
"""
Learning Rules Configuration Generator

Creates configuration files for toggling various learning rules and mechanisms
in the SNN agents. These configs can be used to study the contribution of
different learning components.
"""

import json
from pathlib import Path


def create_learning_rules_configs():
    """Create learning rule configuration variants."""
    
    # Base configuration with all rules enabled
    base_rules = {
        # Dale's principle
        "APPLY_DALE_PRINCIPLE": True,
        
        # Homeostasis
        "ENABLE_HOMEOSTASIS": True,
        "TARGET_RATE_HZ": 5.0,
        "HOMEOSTATIC_TAU": 10000.0,
        "THRESHOLD_ADAPT_RATE": 0.0001,
        "MAX_THRESHOLD_ADAPT": 5.0,
        
        # STDP
        "ENABLE_STDP": True,
        "STDP_A_PLUS": 0.02,
        "STDP_A_MINUS": 0.021,
        
        # Three-factor learning
        "ENABLE_DOPAMINE_MODULATION": True,
        "BASELINE_DOPAMINE": 0.2,
        "TAU_DOPAMINE": 200.0,
        
        # Eligibility traces
        "ENABLE_ELIGIBILITY_TRACE": True,
        "TAU_ELIGIBILITY": 2000.0,
        
        # Reward prediction
        "ENABLE_REWARD_PREDICTION": True,
        "REWARD_PREDICTION_RATE": 0.05,
        "REWARD_DISCOUNT": 0.9,
        
        # Weight constraints
        "ENABLE_WEIGHT_BOUNDS": True,
        "MAX_WEIGHT_SCALE": 5.0,
        "ENABLE_WEIGHT_DECAY": True,
        "WEIGHT_DECAY": 0.00001,
        
        # Synaptic dynamics
        "ENABLE_SYNAPTIC_DYNAMICS": True,
        "TAU_SYN_E": 5.0,
        "TAU_SYN_I": 10.0,
    }
    
    # Create variants
    variants = {
        # Full model
        "full": base_rules.copy(),
        
        # No Dale's principle
        "no_dale": {
            **base_rules,
            "APPLY_DALE_PRINCIPLE": False,
        },
        
        # No homeostasis
        "no_homeostasis": {
            **base_rules,
            "ENABLE_HOMEOSTASIS": False,
            "THRESHOLD_ADAPT_RATE": 0.0,
        },
        
        # No STDP (Hebbian learning disabled)
        "no_stdp": {
            **base_rules,
            "ENABLE_STDP": False,
            "STDP_A_PLUS": 0.0,
            "STDP_A_MINUS": 0.0,
        },
        
        # No dopamine modulation (only Hebbian)
        "no_dopamine": {
            **base_rules,
            "ENABLE_DOPAMINE_MODULATION": False,
            "BASELINE_DOPAMINE": 0.0,
        },
        
        # No eligibility trace (immediate learning)
        "no_eligibility": {
            **base_rules,
            "ENABLE_ELIGIBILITY_TRACE": False,
            "TAU_ELIGIBILITY": 1.0,  # Very fast decay
        },
        
        # No reward prediction (only immediate rewards)
        "no_reward_prediction": {
            **base_rules,
            "ENABLE_REWARD_PREDICTION": False,
            "REWARD_PREDICTION_RATE": 0.0,
        },
        
        # No weight constraints
        "no_weight_constraints": {
            **base_rules,
            "ENABLE_WEIGHT_BOUNDS": False,
            "ENABLE_WEIGHT_DECAY": False,
            "WEIGHT_DECAY": 0.0,
        },
        
        # Minimal learning (only basic STDP)
        "minimal": {
            **base_rules,
            "ENABLE_HOMEOSTASIS": False,
            "ENABLE_DOPAMINE_MODULATION": False,
            "ENABLE_ELIGIBILITY_TRACE": False,
            "ENABLE_REWARD_PREDICTION": False,
            "ENABLE_WEIGHT_DECAY": False,
            "APPLY_DALE_PRINCIPLE": False,
        },
        
        # Classical three-factor rule only
        "three_factor_only": {
            **base_rules,
            "ENABLE_HOMEOSTASIS": False,
            "ENABLE_REWARD_PREDICTION": False,
            "ENABLE_WEIGHT_DECAY": False,
        },
        
        # Homeostasis study variants
        "strong_homeostasis": {
            **base_rules,
            "HOMEOSTATIC_TAU": 1000.0,  # Faster adaptation
            "THRESHOLD_ADAPT_RATE": 0.001,  # Stronger adaptation
        },
        
        "weak_homeostasis": {
            **base_rules,
            "HOMEOSTATIC_TAU": 50000.0,  # Slower adaptation
            "THRESHOLD_ADAPT_RATE": 0.00001,  # Weaker adaptation
        },
    }
    
    return variants


def create_parameter_sweep_configs():
    """Create configs for parameter sweeps."""
    base = create_learning_rules_configs()["full"]
    
    sweeps = {}
    
    # STDP balance sweep
    for ratio in [0.8, 0.9, 1.0, 1.1, 1.2]:
        sweeps[f"stdp_ratio_{ratio:.1f}"] = {
            **base,
            "STDP_A_PLUS": 0.02,
            "STDP_A_MINUS": 0.02 * ratio,
        }
    
    # Dopamine baseline sweep
    for baseline in [0.0, 0.1, 0.2, 0.3, 0.5]:
        sweeps[f"dopamine_baseline_{baseline:.1f}"] = {
            **base,
            "BASELINE_DOPAMINE": baseline,
        }
    
    # Eligibility trace timescale sweep
    for tau in [500, 1000, 2000, 5000, 10000]:
        sweeps[f"eligibility_tau_{tau}"] = {
            **base,
            "TAU_ELIGIBILITY": float(tau),
        }
    
    return sweeps


def main():
    """Generate all learning rules configuration files."""
    # Create directories
    base_dir = Path("configs/learning_rules")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    sweep_dir = base_dir / "parameter_sweeps"
    sweep_dir.mkdir(exist_ok=True)
    
    # Generate main variants
    variants = create_learning_rules_configs()
    for name, config in variants.items():
        filepath = base_dir / f"{name}.json"
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Created: {filepath}")
    
    # Generate parameter sweeps
    sweeps = create_parameter_sweep_configs()
    for name, config in sweeps.items():
        filepath = sweep_dir / f"{name}.json"
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Created: {filepath}")
    
    # Create README
    readme_content = """# Learning Rules Configurations

This directory contains configurations for toggling and tuning various learning rules
and mechanisms in the SNN agents.

## Main Variants

- `full.json`: All learning rules enabled (baseline)
- `no_dale.json`: Dale's principle disabled
- `no_homeostasis.json`: Homeostatic plasticity disabled
- `no_stdp.json`: STDP (Hebbian learning) disabled
- `no_dopamine.json`: Dopamine modulation disabled
- `no_eligibility.json`: Eligibility traces disabled
- `no_reward_prediction.json`: Reward prediction disabled
- `no_weight_constraints.json`: Weight bounds and decay disabled
- `minimal.json`: Only basic STDP enabled
- `three_factor_only.json`: Classic three-factor rule
- `strong_homeostasis.json`: Stronger homeostatic adaptation
- `weak_homeostasis.json`: Weaker homeostatic adaptation

## Parameter Sweeps

The `parameter_sweeps/` directory contains configurations for systematic parameter studies:

- STDP balance ratios
- Dopamine baseline levels
- Eligibility trace timescales

## Usage

```bash
# Run with specific learning rules
python -m models.runner phase_0_13 --learning-rules-config configs/learning_rules/no_dale.json

# Compare different variants
for variant in full no_dale no_homeostasis minimal; do
    python -m models.runner phase_0_13 \\
        --learning-rules-config configs/learning_rules/$variant.json \\
        --n-episodes 100
done
```
"""
    
    with open(base_dir / "README.md", 'w') as f:
        f.write(readme_content)
    
    print(f"\nCreated {len(variants)} main variants and {len(sweeps)} parameter sweep configs")
    print(f"Configuration files saved to: {base_dir}")


if __name__ == "__main__":
    main()