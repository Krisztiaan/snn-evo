# Learning Rules Configurations

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
    python -m models.runner phase_0_13 \
        --learning-rules-config configs/learning_rules/$variant.json \
        --n-episodes 100
done
```
