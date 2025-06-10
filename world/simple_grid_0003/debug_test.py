"""Debug reward collection issue."""
import jax.numpy as jnp
from world.simple_grid_0003 import SimpleGridWorld
from world.simple_grid_0003.types import WorldConfig

# Create world
world = SimpleGridWorld(WorldConfig(
    grid_size=20,
    n_rewards=3,
    reward_value=10.0,
    proximity_reward=0.5
))

# Reset
state, _ = world.reset()
print(f"Initial agent pos: {state.agent_pos}")
print(f"Reward positions: {state.reward_positions}")

# Move agent to first reward
reward_pos = tuple(state.reward_positions[0])
print(f"First reward pos: {reward_pos}")
state = state._replace(agent_pos=reward_pos)
print(f"Agent moved to: {state.agent_pos}")

# Check if positions match
print(f"Agent at reward? {state.agent_pos == reward_pos}")

# Pack positions to check
agent_packed = state.agent_pos[0] * world.grid_size + state.agent_pos[1]
reward_packed = state.reward_positions[:, 0] * world.grid_size + state.reward_positions[:, 1]
print(f"Agent packed: {agent_packed}")
print(f"Rewards packed: {reward_packed}")
print(f"At first reward? {agent_packed == reward_packed[0]}")

# Step
result = world.step(state, 0)
print(f"\nAfter step:")
print(f"New agent pos: {result.state.agent_pos}")
print(f"Reward: {result.reward}")
print(f"Total reward: {result.state.total_reward}")
print(f"Reward collected flags: {result.state.reward_collected}")

# Check distances
new_agent_packed = result.state.agent_pos[0] * world.grid_size + result.state.agent_pos[1]
distances_squared = world._packed_distance_squared(new_agent_packed, reward_packed)
print(f"\nDistances squared to rewards: {distances_squared}")
print(f"Within proximity (< 25)? {distances_squared < 25}")