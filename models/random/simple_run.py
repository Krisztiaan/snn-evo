# keywords: [random agent simple test, minimal export]
"""Simple test script for random agent without pandas dependencies."""

import jax
from models.random import RandomAgent, RandomAgentConfig
from world.simple_grid_0001 import WorldConfig


def run_simple_test():
    """Run a simple test of the random agent."""
    # Configure small world
    world_config = WorldConfig(
        grid_size=10,
        n_rewards=5,
        max_timesteps=100,
        seed=42
    )
    
    config = RandomAgentConfig(
        world_config=world_config,
        n_episodes=1,
        n_steps=100,
        seed=42,
        export_dir="models/random/test_logs",
        export_full_trajectory=False,  # Minimal export
        export_episode_summary=True
    )
    
    print("Random Agent Simple Test")
    print("=" * 30)
    print(f"Grid: {world_config.grid_size}x{world_config.grid_size}")
    print(f"Rewards: {world_config.n_rewards}")
    print(f"Steps: {config.n_steps}")
    
    # Create and run agent
    agent = RandomAgent(config)
    
    # Directly run episode without full data export framework
    import jax.random as random
    key = random.PRNGKey(42)
    
    # Reset world
    state, obs = agent.world.reset(key)
    
    print(f"\nStarting at: {state.agent_pos}")
    print(f"Initial gradient: {obs.gradient:.3f}")
    
    # Run steps
    total_reward = 0
    for step in range(100):
        key, subkey = random.split(key)
        action = agent.select_action(subkey, state, obs)
        
        result = agent.world.step(state, action)
        state = result.state
        obs = result.observation
        
        if result.reward > 0:
            total_reward += result.reward
            if result.reward >= world_config.reward_value:
                print(f"Step {step}: Collected reward! Total: {total_reward}")
        
        if result.done:
            break
    
    print(f"\nFinal position: {state.agent_pos}")
    print(f"Total reward: {total_reward}")
    print(f"Rewards collected: {int(jax.numpy.sum(state.reward_collected))}/{world_config.n_rewards}")
    

if __name__ == "__main__":
    run_simple_test()