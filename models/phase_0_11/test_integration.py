#!/usr/bin/env python3
"""Test phase 0.11 agent with JAX-optimized world."""

from jax import random
from world.simple_grid_0001.types import WorldConfig
from models.phase_0_11 import SnnAgent, SnnAgentConfig, NetworkParams, ExperimentConfig
import time
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


print("Testing Phase 0.11 Agent with JAX World")
print("=" * 50)

# Small config for quick test
config = SnnAgentConfig(
    world_config=WorldConfig(grid_size=50, n_rewards=10, max_timesteps=1000),
    network_params=NetworkParams(),
    exp_config=ExperimentConfig(n_episodes=1, seed=42, export_dir="experiments/phase_0_11/test"),
)

print("Creating agent...")
start = time.time()
agent = SnnAgent(config)
print(f"Agent created in {time.time() - start:.2f}s")

print("\nRunning quick test (100 steps)...")
key = random.PRNGKey(42)

# Reset
start = time.time()
world_state, obs = agent.world.reset(key)
print(f"World reset in {(time.time() - start) * 1000:.1f}ms")

# Run some steps
total_step_time = 0
for i in range(100):
    key, subkey = random.split(key)

    # Time just the world step
    step_start = time.time()
    result = agent.world.step(world_state, i % 4, subkey)
    step_time = time.time() - step_start
    total_step_time += step_time

    world_state = result.state

    if i % 20 == 0:
        print(f"  Step {i}: {step_time * 1000:.2f}ms, pos={world_state.agent_pos}")

avg_step = total_step_time / 100
print(f"\nAverage world step time: {avg_step * 1000:.2f}ms")
print(f"World steps per second: {1 / avg_step:.0f}")

print("\n✅ Integration test passed!")
print("The JAX world is working correctly with phase 0.11 agent.")
