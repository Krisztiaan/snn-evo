"""Simple test to check if the world works."""

from world.simple_grid_0003 import SimpleGridWorld
from world.simple_grid_0003.types import WorldConfig

print("Creating world...")
world = SimpleGridWorld(WorldConfig(grid_size=10, n_rewards=2))

print("Reset...")
state, obs = world.reset()
print(f"Initial state: agent at {state.agent_pos}, {len(state.reward_positions)} rewards")

print("Step...")
result = world.step(state, 0)
print(f"After step: agent at {result.state.agent_pos}, reward={result.reward}")

print("Done!")