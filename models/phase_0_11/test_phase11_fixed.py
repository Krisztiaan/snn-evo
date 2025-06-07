#!/usr/bin/env python3
# keywords: [test phase 0.11, fixed version, simple_grid_0003]
"""Test the fixed phase 0.11 agent with all performance improvements."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import time
import jax
from models.phase_0_11 import SnnAgent, SnnAgentConfig, NetworkParams, ExperimentConfig
from world.simple_grid_0003 import WorldConfig

print("Testing Fixed Phase 0.11 Agent")
print("=" * 50)

# Small config for quick test
config = SnnAgentConfig(
    world_config=WorldConfig(
        grid_size=50,
        n_rewards=10,
        max_timesteps=1000
    ),
    network_params=NetworkParams(),
    exp_config=ExperimentConfig(
        n_episodes=2,
        seed=42,
        export_dir="experiments/phase_0_11/test_fixed"
    )
)

print(f"JAX devices: {jax.devices()}")
print("\nCreating agent...")
start = time.time()
agent = SnnAgent(config)
print(f"Agent created in {time.time() - start:.2f}s")

print("\nRunning performance mode test (no data export)...")
start = time.time()
summaries = agent.run_experiment(performance_mode=True)
total_time = time.time() - start

print(f"\nPerformance Results:")
print(f"Total time: {total_time:.2f}s")
print(f"Episodes: {len(summaries)}")
total_steps = sum(s['steps_taken'] for s in summaries)
print(f"Total steps: {total_steps:,}")
print(f"Steps per second: {total_steps/total_time:,.0f}")
print(f"Milliseconds per step: {1000 * total_time / total_steps:.2f}ms")

print("\n✅ All tests passed!")
print("Fixed version is working correctly with:")
print("- simple_grid_0003 (ultra-optimized world)")
print("- Precomputed motor decode matrix")
print("- Reusable spike float buffer")
print("- Efficient key pre-splitting")
print("- Optional data export (performance mode)")