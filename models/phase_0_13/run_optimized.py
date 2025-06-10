#!/usr/bin/env python3
"""Run the optimized Phase 0.13 agent for maximum performance."""

import sys
import time
import jax
import numpy as np
from agent import SnnAgent
from config import SnnAgentConfig, WorldConfig, NetworkParams, ExperimentConfig

def main():
    # Parse command line arguments
    n_episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    grid_size = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    max_timesteps = int(sys.argv[3]) if len(sys.argv) > 3 else 5000
    
    print(f"Running optimized Phase 0.13 agent")
    print(f"Episodes: {n_episodes}, Grid: {grid_size}x{grid_size}, Max steps: {max_timesteps}")
    print("="*60)
    
    # Create configuration
    config = SnnAgentConfig(
        world_config=WorldConfig(
            grid_size=grid_size,
            n_rewards=int(grid_size * grid_size * 0.03),  # 3% coverage
            max_timesteps=max_timesteps
        ),
        network_params=NetworkParams(),
        exp_config=ExperimentConfig(
            n_episodes=n_episodes,
            export_dir="experiments/phase_0_13_optimized"
        )
    )
    
    # Run experiment
    start_time = time.time()
    agent = SnnAgent(config)
    summaries = agent.run_experiment()
    total_time = time.time() - start_time
    
    # Print summary statistics
    rewards = [s['rewards_collected'] for s in summaries]
    times = [s['episode_time'] for s in summaries]
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE")
    print("="*60)
    print(f"Total time: {total_time:.1f}s")
    print(f"Episodes/hour: {n_episodes / (total_time / 3600):.0f}")
    print(f"\nReward Statistics:")
    print(f"  Mean: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
    print(f"  Range: [{np.min(rewards)}, {np.max(rewards)}]")
    print(f"  Median: {np.median(rewards):.1f}")
    
    # Learning progress
    if len(rewards) >= 10:
        first_10_avg = np.mean(rewards[:10])
        last_10_avg = np.mean(rewards[-10:])
        improvement = last_10_avg - first_10_avg
        print(f"\nLearning Progress:")
        print(f"  First 10 episodes: {first_10_avg:.1f}")
        print(f"  Last 10 episodes: {last_10_avg:.1f}")
        print(f"  Improvement: {improvement:+.1f} ({improvement/first_10_avg*100:+.1f}%)")
    
    # Performance metrics
    total_steps = sum(s['steps_taken'] for s in summaries)
    print(f"\nPerformance:")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Average steps/second: {total_steps/total_time:.0f}")
    print(f"  Average episode time: {np.mean(times):.2f}s")

if __name__ == "__main__":
    main()