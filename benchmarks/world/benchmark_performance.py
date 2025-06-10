# keywords: [performance benchmark, jax grid world, timing test]
"""Performance benchmark for SimpleGridWorld v0.0.3."""

import time
from world.simple_grid_0003 import SimpleGridWorld
from world.simple_grid_0003.types import WorldConfig


def benchmark_world_performance():
    """Benchmark world performance with various configurations."""
    
    configs = [
        ("Small", WorldConfig(grid_size=20, n_rewards=10, max_timesteps=1000)),
        ("Default", WorldConfig()),  # Default 100x100, 300 rewards
    ]
    
    for name, config in configs:
        print(f"\n{name} World ({config.grid_size}x{config.grid_size}, {config.n_rewards} rewards):")
        
        # Create world
        world = SimpleGridWorld(config)
        
        # Warm up JIT
        state, obs = world.reset()
        for _ in range(10):
            result = world.step(state, 0)
            state = result.state
        
        # Benchmark reset
        start = time.perf_counter()
        for _ in range(10):
            state, obs = world.reset()
        reset_time = (time.perf_counter() - start) / 10
        print(f"  Reset: {reset_time*1000:.3f} ms")
        
        # Benchmark steps
        state, obs = world.reset()
        start = time.perf_counter()
        n_steps = 100
        for i in range(n_steps):
            result = world.step(state, i % 4)
            state = result.state
        step_time = (time.perf_counter() - start) / n_steps
        print(f"  Step: {step_time*1000:.3f} ms")
        print(f"  Steps/second: {1/step_time:.0f}")
        
        # Benchmark with reward collections
        state, obs = world.reset()
        collections = 0
        start = time.perf_counter()
        for i in range(n_steps):
            # Try to collect rewards by moving to known positions
            if i % 10 == 0 and i < len(state.reward_positions):
                # Move towards a reward
                reward_x = int(state.reward_positions[i % config.n_rewards, 0])
                reward_y = int(state.reward_positions[i % config.n_rewards, 1])
                state = state._replace(agent_pos=(reward_x, (reward_y + 1) % config.grid_size))
                result = world.step(state, 0)  # Move up
                collections += int(result.reward)
            else:
                result = world.step(state, i % 4)
            state = result.state
        collection_time = (time.perf_counter() - start) / n_steps
        print(f"  Step with collections: {collection_time*1000:.3f} ms")
        print(f"  Rewards collected: {collections}")
        
        # Benchmark history extraction
        start = time.perf_counter()
        for _ in range(10):
            positions, spawn_steps, collect_steps = world.get_reward_history(state)
        history_time = (time.perf_counter() - start) / 10
        print(f"  History extraction: {history_time*1000:.3f} ms")
        print(f"  History size: {int(state.reward_history_count)} rewards")


if __name__ == "__main__":
    print("SimpleGridWorld v0.0.3 Performance Benchmark")
    print("=" * 50)
    benchmark_world_performance()