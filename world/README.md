# Grid Worlds for Meta-Learning Research

This directory contains standalone grid world implementations for meta-learning experiments.

## Worlds

### simple_grid_0001
A minimal JAX-compatible grid world with:
- Toroidal navigation
- Gradient-based observations
- Clustered reward distribution
- Pure functional interface

Usage:
```python
from world.simple_grid_0001 import SimpleGridWorld, WorldConfig

world = SimpleGridWorld(WorldConfig(grid_size=100, n_rewards=300))
state, obs = world.reset(jax.random.PRNGKey(0))
```

## Design Philosophy

Each world is:
- **Self-contained** - No dependencies on other worlds
- **Minimal** - Only essential mechanics included
- **JAX-optimized** - Functional design for JIT compilation
- **Agent-agnostic** - Clean interface for any neural architecture

Future worlds can extend or build upon these foundations while maintaining independence.