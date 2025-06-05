# Simple Grid World

A minimal JAX-compatible grid world for meta-learning research.

## Features

- **Toroidal grid**: Agents wrap around edges (configurable)
- **Gradient observations**: Distance-based signal to nearest reward
- **JAX optimization**: Fully JIT-compiled step function
- **Functional design**: Immutable state for clean integration

## Interface

```python
from world.simple_grid_0001 import SimpleGridWorld, WorldConfig

# Configure world
config = WorldConfig(
    grid_size=100,      # Grid dimensions (100x100)
    n_rewards=300,      # Number of rewards to place
    max_timesteps=50000,
    reward_value=10.0,  # Points for collecting reward
    proximity_reward=0.5 # Points for being near rewards
)

# Create world
world = SimpleGridWorld(config)

# Reset to get initial state
state, observation = world.reset(jax.random.PRNGKey(0))

# Take actions (0=up, 1=right, 2=down, 3=left)
result = world.step(state, action=1)
new_state = result.state
reward = result.reward
```

## State

The world maintains a simple state:
- Agent position (x, y)
- Reward positions and collection status
- Total accumulated reward
- Current timestep

## Observations

Agents receive a single gradient value (0-1):
- 1.0 = very close to nearest uncollected reward
- 0.0 = far from all rewards
- Exponential decay based on distance

## Design Philosophy

This world is intentionally minimal:
- No awareness of neural networks or learning algorithms
- No built-in curriculum or phases
- Simple, clean interface for any agent to use
- Optimized for JAX performance

Future worlds can extend or build upon this foundation while maintaining their own independent implementations.