# keywords: [grid world example, simple usage, random agent]
"""Example usage of the simple grid world."""

import jax.numpy as jnp
from jax import random

from world.simple_grid_0001 import SimpleGridWorld, WorldConfig


def random_agent(key: random.PRNGKey, n_steps: int = 1000):
    """Run a random agent in the grid world."""
    # Create world with custom config
    config = WorldConfig(
        grid_size=50, n_rewards=50, max_timesteps=n_steps, reward_value=10.0, proximity_reward=0.1
    )
    world = SimpleGridWorld(config)

    # Get and display world metadata
    metadata = world.get_metadata()
    print(f"World: {metadata['name']} v{metadata['version']}")
    print(f"Description: {metadata['description']}")
    print()

    # Reset world
    reset_key, action_key = random.split(key)
    state, obs = world.reset(reset_key)

    print(f"Starting at position: {state.agent_pos}")
    print(f"Initial gradient signal: {obs.gradient:.3f}")
    print(f"Number of rewards: {config.n_rewards}")

    # Run episode
    total_reward = 0.0
    for step in range(n_steps):
        # Random action
        action_key, subkey = random.split(action_key)
        action = random.randint(subkey, (), 0, 4)

        # Take step
        result = world.step(state, action)
        state = result.state

        if result.reward > 0:
            total_reward += result.reward
            if result.reward >= config.reward_value:
                print(
                    f"Step {step}: Collected reward! Total: {total_reward} (respawned at farthest position)"
                )

        if result.done:
            print(f"Episode finished at step {step}")
            break

    print("\nFinal stats:")
    print(f"Total reward: {state.total_reward}")
    print(f"Rewards collected: {jnp.sum(state.reward_collected)}/{config.n_rewards}")
    print(f"Final position: {state.agent_pos}")

    return state


def gradient_following_agent(key: random.PRNGKey, n_steps: int = 1000):
    """Simple agent that follows the gradient signal."""
    config = WorldConfig(grid_size=30, n_rewards=20)
    world = SimpleGridWorld(config)

    # Get and display world metadata
    metadata = world.get_metadata()
    print(f"World: {metadata['name']} v{metadata['version']}")
    print()

    # Reset
    reset_key, action_key = random.split(key)
    state, obs = world.reset(reset_key)

    print("Running gradient-following agent...")

    # Store path for visualization
    path = [state.agent_pos]

    for _step in range(n_steps):
        # Try each action and pick the one with highest gradient
        best_action = 0
        best_gradient = -1.0

        for action in range(4):
            # Simulate taking this action (no key needed for simulation)
            result = world.step(state, action)
            if result.observation.gradient > best_gradient:
                best_gradient = result.observation.gradient
                best_action = action

        # Take best action
        result = world.step(state, best_action)
        state = result.state
        path.append(state.agent_pos)

        if result.done:
            break

    print(f"Collected {jnp.sum(state.reward_collected)}/{config.n_rewards} rewards")
    print(f"Path length: {len(path)}")

    return state, path


if __name__ == "__main__":
    key = random.PRNGKey(42)

    # Run random agent
    print("=== Random Agent ===")
    key, subkey = random.split(key)
    random_agent(subkey)

    print("\n" + "=" * 50 + "\n")

    # Run gradient-following agent
    print("=== Gradient-Following Agent ===")
    key, subkey = random.split(key)
    gradient_following_agent(subkey)
