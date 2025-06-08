#!/usr/bin/env python3
"""Test that simple_grid_0003 is a drop-in replacement for 0001 and 0002."""

import simple_grid_0003
import simple_grid_0002
import simple_grid_0001
from simple_grid_0003 import StepResult as Result3
from simple_grid_0003 import Observation as Obs3
from simple_grid_0003 import WorldState as State3
from simple_grid_0003 import WorldConfig as Config3
from simple_grid_0003 import SimpleGridWorld as World3
from simple_grid_0001 import WorldConfig, WorldState, Observation, StepResult
from simple_grid_0001 import SimpleGridWorld as World1
import jax
from jax import random

# Test importing from different versions
print("Testing import compatibility...")

# Original imports from 0001

# Same imports but from 0003 (drop-in replacement)

# Verify types are the same
assert WorldConfig == Config3, "WorldConfig should be the same type"
assert WorldState == State3, "WorldState should be the same type"
assert Observation == Obs3, "Observation should be the same type"
assert StepResult == Result3, "StepResult should be the same type"

print("✓ All types match - perfect compatibility!")

# Test that code written for 0001 works with 0003


def run_simulation(world_module):
    """Example code that uses the world - should work with any version."""
    # Import everything from the module
    WorldClass = world_module.SimpleGridWorld
    Config = world_module.WorldConfig

    # Create world with standard config
    config = Config(grid_size=50, n_rewards=10, max_timesteps=1000)
    world = WorldClass(config)

    # Run simulation
    key = random.PRNGKey(42)
    state, obs = world.reset(key)

    total_reward = 0.0
    for i in range(100):
        key, subkey = random.split(key)
        result = world.step(state, i % 4, subkey)
        state = result.state
        total_reward += result.reward

        if result.done:
            break

    return total_reward, state.timestep


# Test with different versions

print("\nTesting functional compatibility...")
reward1, steps1 = run_simulation(simple_grid_0001)
print(f"0001: {steps1} steps, {reward1:.1f} total reward")

reward2, steps2 = run_simulation(simple_grid_0002)
print(f"0002: {steps2} steps, {reward2:.1f} total reward")

reward3, steps3 = run_simulation(simple_grid_0003)
print(f"0003: {steps3} steps, {reward3:.1f} total reward")

print("\n✅ All versions work with the same interface!")
print("simple_grid_0003 can be used as a drop-in replacement for 0001 or 0002")

# Show how to migrate
print("\nMigration is as simple as changing the import:")
print("  # Before:")
print("  from simple_grid_0001 import SimpleGridWorld, WorldConfig")
print("  # After:")
print("  from simple_grid_0003 import SimpleGridWorld, WorldConfig")
print("  # No other code changes needed!")
