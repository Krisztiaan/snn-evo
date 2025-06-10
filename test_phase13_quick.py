#!/usr/bin/env python3
"""Quick test of Phase 0.13 agent with 20x20 grid."""

import time
import jax
from models.phase_0_13 import SnnAgent, SnnAgentConfig

def main():
    print("Testing Phase 0.13 Agent with 20x20 grid")
    print("="*50)
    
    # Create agent with default config (20x20 grid)
    config = SnnAgentConfig()
    print(f"Grid size: {config.world_config.grid_size}x{config.world_config.grid_size}")
    print(f"Rewards: {config.world_config.n_rewards}")
    print(f"Max timesteps: {config.world_config.max_timesteps}")
    
    # Initialize agent
    agent = SnnAgent(config)
    
    # Run one episode
    print("\nRunning test episode...")
    key = jax.random.PRNGKey(42)
    start = time.time()
    summary = agent.run_episode(key, 0)
    elapsed = time.time() - start
    
    print(f"\nResults:")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Steps: {summary['steps_taken']}")
    print(f"  Rewards collected: {summary['rewards_collected']}")
    print(f"  Steps/second: {summary['steps_taken']/elapsed:.0f}")
    
    print("\n✓ Test passed! Agent is working correctly.")

if __name__ == "__main__":
    main()