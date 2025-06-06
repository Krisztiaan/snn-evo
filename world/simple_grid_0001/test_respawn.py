# keywords: [test respawn functionality, reward respawning]
"""Test the reward respawning functionality."""

import jax
import jax.numpy as jnp
from jax import random

from .world import SimpleGridWorld
from .types import WorldConfig


def test_respawn():
    """Test that rewards respawn at farthest position when collected."""
    # Small grid for easier testing
    config = WorldConfig(
        grid_size=10,
        n_rewards=3,
        max_timesteps=100,
        reward_value=10.0
    )
    world = SimpleGridWorld(config)
    
    # Get metadata
    metadata = world.get_metadata()
    print(f"Testing {metadata['name']} v{metadata['version']}")
    print(f"Description: {metadata['description']}\n")
    
    # Reset
    key = random.PRNGKey(42)
    key, reset_key = random.split(key)
    state, obs = world.reset(reset_key)
    
    print(f"Initial agent position: {state.agent_pos}")
    print(f"Initial reward positions:\n{state.reward_positions}")
    
    # Move agent one step below first reward so we can step into it
    target_reward_idx = 0
    target_pos = state.reward_positions[target_reward_idx]
    state = state._replace(agent_pos=(int(target_pos[0]), int(target_pos[1]) + 1))
    
    print(f"\nAgent positioned at {state.agent_pos}, reward at {tuple(target_pos.tolist())}")
    
    # Step up (action 0) to collect the reward
    result = world.step(state, action=0)  # Move up to collect
    
    print(f"\nAfter step:")
    print(f"Agent now at: {result.state.agent_pos}")
    print(f"Reward collected: {result.reward}")
    print(f"New reward positions:\n{result.state.reward_positions}")
    
    # Check that the reward was respawned
    old_pos = state.reward_positions[target_reward_idx]
    new_pos = result.state.reward_positions[target_reward_idx]
    
    if not jnp.array_equal(old_pos, new_pos):
        print(f"\n✓ Reward respawned from {old_pos} to {new_pos}")
        
        # Calculate distance from agent
        agent_array = jnp.array(result.state.agent_pos)
        new_pos_array = jnp.array(new_pos)
        distance = jnp.linalg.norm(new_pos_array - agent_array)
        print(f"Distance from agent to respawned reward: {distance:.2f}")
    else:
        print("\n✗ Reward did not respawn!")
    
    # Check that reward is marked as not collected
    if not result.state.reward_collected[target_reward_idx]:
        print("✓ Respawned reward is marked as not collected")
    else:
        print("✗ Respawned reward is still marked as collected!")


if __name__ == "__main__":
    test_respawn()