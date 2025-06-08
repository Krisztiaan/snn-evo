# models/phase_0_8/run.py
# keywords: [snn run, experiment runner, phase 0.8]
"""Main script to run Phase 0.8 SNN agent experiments."""

import argparse
import time
from pathlib import Path

import numpy as np

from .agent import SnnAgent
from .config import ExperimentConfig, NetworkParams, SnnAgentConfig, WorldConfig


def main():
    parser = argparse.ArgumentParser(
        description="Run Phase 0.8 SNN Agent - Best of All Worlds",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Phase 0.8 synthesizes the best ideas from phases 0.4-0.7:
- Modular architecture with clear neuron populations
- Proper two-trace STDP (missing in 0.5/0.6)
- Fixed input weights with targeted plasticity
- Homeostatic firing rate control
- Biologically motivated connectivity
        """,
    )

    parser.add_argument(
        "--episodes", type=int, default=5, help="Number of episodes to run (default: 5)"
    )
    parser.add_argument(
        "--steps", type=int, default=10000, help="Max steps per episode (default: 10000)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Master random seed (default: 42)")
    parser.add_argument(
        "--export-dir",
        type=str,
        default="experiments/phase_0_8",
        help="Directory for data export (default: experiments/phase_0_8)",
    )
    parser.add_argument(
        "--no-export", action="store_true", help="Disable data exporting for quick tests"
    )
    parser.add_argument(
        "--quick-test", action="store_true", help="Quick test mode: 1 episode, 1000 steps"
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        default=True,
        help="Enable stability monitoring (default: True)",
    )
    parser.add_argument(
        "--no-monitor", action="store_false", dest="monitor", help="Disable stability monitoring"
    )

    args = parser.parse_args()

    # Quick test mode overrides
    if args.quick_test:
        args.episodes = 1
        args.steps = 1000
        print("🚀 Quick test mode: 1 episode, 1000 steps")

    # Create configurations
    world_config = WorldConfig(max_timesteps=args.steps)
    network_params = NetworkParams()
    exp_config = ExperimentConfig(
        n_episodes=args.episodes,
        seed=args.seed,
        export_dir=args.export_dir,
        enable_export=not args.no_export,
        monitor_rates=args.monitor,
        monitor_weights=args.monitor,
        check_stability=args.monitor,
    )

    config = SnnAgentConfig(
        world_config=world_config, network_params=network_params, exp_config=exp_config
    )

    # Create export directory
    if not args.no_export:
        Path(args.export_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("🧠 Phase 0.8 SNN Agent - Best of All Worlds")
    print("=" * 60)
    print(f"Episodes: {args.episodes}")
    print(f"Max steps per episode: {args.steps}")
    print(f"Random seed: {args.seed}")
    print(f"Data export: {'ENABLED' if not args.no_export else 'DISABLED'}")
    print(f"Stability monitoring: {'ON' if args.monitor else 'OFF'}")

    print("\n📊 Network Architecture:")
    print(f"  - Sensory neurons: {network_params.NUM_SENSORY}")
    print(f"  - Processing neurons: {network_params.NUM_PROCESSING} (80% E, 20% I)")
    print(f"  - Readout neurons: {network_params.NUM_READOUT}")
    print(
        f"  - Total neurons: {network_params.NUM_SENSORY + network_params.NUM_PROCESSING + network_params.NUM_READOUT}"
    )

    print("\n🔧 Key Features:")
    print("  ✓ Population-coded sensory input (16 channels)")
    print("  ✓ Two-trace STDP for richer learning dynamics")
    print("  ✓ Homeostatic firing rate control")
    print("  ✓ Three-factor learning with RPE-based dopamine")
    print("  ✓ Targeted plasticity (no learning on input weights)")
    print("  ✓ Soft weight bounds using tanh")
    print("=" * 60 + "\n")

    # Initialize agent and run experiment
    start_time = time.time()
    agent = SnnAgent(config)
    all_summaries = agent.run_experiment()
    total_time = time.time() - start_time

    # Aggregate results
    if len(all_summaries) > 1:
        print("\n" + "=" * 60)
        print("📈 Experiment Summary")
        print("=" * 60)

        # Basic stats
        rewards = [s["total_reward"] for s in all_summaries]
        collected = [s["rewards_collected"] for s in all_summaries]
        rates = [s["mean_firing_rate"] for s in all_summaries]

        print(f"Total Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"Rewards Collected: {np.mean(collected):.1f} ± {np.std(collected):.1f}")
        print(f"Mean Firing Rate: {np.mean(rates):.1f} ± {np.std(rates):.1f} Hz")

        # Learning trend
        first_half = np.mean(rewards[: len(rewards) // 2])
        second_half = np.mean(rewards[len(rewards) // 2 :])
        improvement = second_half - first_half

        print("\n📊 Learning Progress:")
        print(f"First half avg: {first_half:.2f}")
        print(f"Second half avg: {second_half:.2f}")
        print(f"Improvement: {improvement:+.2f} ({improvement / max(first_half, 0.1) * 100:+.1f}%)")

        # Stability check
        if args.monitor and hasattr(agent, "firing_rate_history"):
            if agent.firing_rate_history:
                rate_trend = agent.firing_rate_history[-1] - agent.firing_rate_history[0]
                print("\n🔬 Stability Metrics:")
                print(f"Firing rate drift: {rate_trend:+.2f} Hz")
                print(f"Final mean rate: {agent.firing_rate_history[-1]:.1f} Hz")

    print(f"\n⏱️ Total runtime: {total_time:.1f} seconds")
    print(f"Average per episode: {total_time / args.episodes:.1f} seconds")

    # Success indicators
    print("\n✅ Experiment complete!")
    if not args.no_export:
        print(f"📁 Data saved to: {args.export_dir}")


if __name__ == "__main__":
    main()
