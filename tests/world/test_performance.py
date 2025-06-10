# keywords: [performance test, timing verification, jax optimization]
"""Quick performance test for the optimized world."""

import time
from world.simple_grid_0003 import SimpleGridWorld
from world.simple_grid_0003.types import WorldConfig

# Test with default config
config = WorldConfig()
world = SimpleGridWorld(config)

# Warm up JIT
state, obs = world.reset()
for _ in range(10):
    result = world.step(state, 0)
    state = result.state

# Time 1000 steps
start = time.perf_counter()
state, obs = world.reset()
for i in range(1000):
    result = world.step(state, i % 4)
    state = result.state
elapsed = time.perf_counter() - start

print(f"1000 steps took {elapsed:.3f} seconds")
print(f"Steps per second: {1000/elapsed:.0f}")
print(f"Microseconds per step: {elapsed/1000*1e6:.1f}")

# Test history extraction
start = time.perf_counter()
positions, spawn_steps, collect_steps = world.get_reward_history(state)
history_time = time.perf_counter() - start
print(f"\nHistory extraction took {history_time*1e6:.1f} microseconds")
print(f"History size: {int(state.reward_history_count)} entries")