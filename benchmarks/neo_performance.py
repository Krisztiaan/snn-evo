#!/usr/bin/env python3
# keywords: [benchmark, neo agent, jit performance, protocol runner]
"""
Benchmark for the JIT-compiled Neo agent and ProtocolRunner.

This script measures the absolute performance of the Neo agent, which is
designed to be fully JIT-compiled for maximum speed.

The 'no-JIT' path is included for debugging and comparison, but it's not a
true Python-vs-JAX comparison. The 'no-JIT' mode still calls JIT-compiled
functions for each step, incurring significant overhead. The true performance
is reflected in the 'with-JIT' numbers, where the entire episode loop is
compiled into a single, highly optimized kernel.
"""

import time
import shutil
from pathlib import Path

import jax
from jax.random import PRNGKey

from interfaces import ExperimentConfig, ProtocolRunner, ExporterProtocol
from interfaces.config import WorldConfig, NeuralConfig, PlasticityConfig, AgentBehaviorConfig
from world.simple_grid_0004 import MinimalGridWorld
from export.jax_data_exporter import create_episode_buffer, log_step
from models.phase_0_14_neo.agent import NeoAgent


# Add this DummyExporter class at the top of the file
class DummyExporter:
    """A mock exporter that does no I/O, for performance testing."""

    def start_episode(self, episode_id: int, *args, **kwargs):
        # We need a real buffer and log_fn for the runner to work
        
        # A minimal config to create a buffer
        dummy_cfg = ExperimentConfig(
            world=WorldConfig(max_timesteps=10000), 
            neural=NeuralConfig(n_neurons=1000), 
            plasticity=None, behavior=None, experiment_name="", 
            agent_version="", world_version="", flush_at_episode_end=False
        )
        buffer = create_episode_buffer(
            dummy_cfg.world.max_timesteps,
            dummy_cfg.neural.n_neurons,
            episode_id
        )
        return buffer, jax.jit(log_step)

    def end_episode(self, *args, **kwargs):
        # Return dummy stats, do no I/O
        return {"total_rewards": 0.0}

    def save_network_structure(self, neurons, connections, initial_weights=None):
        """Save network structure (dummy implementation)."""
        pass

    def close(self):
        pass


def create_benchmark_config() -> ExperimentConfig:
    """Create a standard configuration for benchmarking."""
    return ExperimentConfig(
        world=WorldConfig(
            grid_size=100,
            n_rewards=300,
            max_timesteps=10000  # A decent number of steps for a stable measurement
        ),
        neural=NeuralConfig(
            n_neurons=1000,
            excitatory_ratio=0.8,  # Use the ratio instead
            n_sensory=40,
            n_motor=9,
            tau_membrane=10.0
        ),
        plasticity=PlasticityConfig(
            enable_stdp=True,
            enable_homeostasis=True,
            enable_reward_modulation=True
        ),
        behavior=AgentBehaviorConfig(action_noise=0.01, temperature=1.0),
        experiment_name="neo_performance_benchmark",
        agent_version="0.14.0",
        world_version="0.0.4",
        n_episodes=1,
        seed=42,
        device="cpu",
        export_dir="benchmarks/temp",
        log_to_console=False,  # Keep output clean
        flush_at_episode_end=True
    )


def main():
    """Run the Neo agent performance benchmark."""
    print("=" * 60)
    print("Neo Agent Performance Benchmark")
    print("=" * 60)
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}\n")

    # Setup components
    config = create_benchmark_config()
    world = MinimalGridWorld(config.world)
    
    # Use the new DummyExporter instead of the real one
    exporter = DummyExporter() 
    agent = NeoAgent(config, exporter)
    runner = ProtocolRunner(world, agent, exporter, config)

    # --- JIT Warmup ---
    print("Warming up JIT compiler (this may take a moment)...")
    key = PRNGKey(41)
    # A single, full run is the most robust way to trigger compilation.
    runner.run_episode(0, key, use_jit=True)
    print("Warmup complete.\n")

    # --- Benchmark with JIT (lax.scan) ---
    print("Testing with JIT (entire episode loop compiled)...")
    key = PRNGKey(42)
    start_time = time.perf_counter()
    jit_stats = runner.run_episode(1, key, use_jit=True)
    jit_duration = time.perf_counter() - start_time
    jit_steps_per_sec = config.world.max_timesteps / jit_duration

    print(f"  Duration: {jit_duration:.3f} s")
    print(f"  Performance: {jit_steps_per_sec:,.0f} steps/s\n")

    # --- Benchmark without JIT (Python loop calling JIT functions) ---
    print("Testing without JIT (Python loop for debugging)...")
    key = PRNGKey(43)
    start_time = time.perf_counter()
    no_jit_stats = runner.run_episode(2, key, use_jit=False)
    no_jit_duration = time.perf_counter() - start_time
    no_jit_steps_per_sec = config.world.max_timesteps / no_jit_duration

    print(f"  Duration: {no_jit_duration:.3f} s")
    print(f"  Performance: {no_jit_steps_per_sec:,.0f} steps/s\n")

    # --- Summary ---
    speedup = jit_steps_per_sec / no_jit_steps_per_sec
    print("-" * 60)
    print("Benchmark Summary")
    print("-" * 60)
    print(f"  With JIT (True Performance): {jit_steps_per_sec:>10,.0f} steps/s")
    print(f"  Without JIT (Debug Mode):    {no_jit_steps_per_sec:>10,.0f} steps/s")
    print(f"  Speedup:                     {speedup:>10.1f}x")
    print("=" * 60)

    # Cleanup
    exporter.close()


if __name__ == "__main__":
    main()