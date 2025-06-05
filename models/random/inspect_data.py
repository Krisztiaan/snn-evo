# keywords: [inspect export data, read hdf5, view trajectory]
"""Inspect exported data from random agent runs."""

import h5py
import numpy as np
import json
from pathlib import Path


def inspect_latest_run():
    """Inspect the most recent run's data."""
    # Find latest export
    logs_dir = Path("experiments/random")
    if not logs_dir.exists():
        print("No experiments found yet. Run an experiment first.")
        return
    latest_dir = max(logs_dir.glob("episode_*"), key=lambda p: p.stat().st_mtime)
    
    print(f"Inspecting: {latest_dir}")
    print("=" * 50)
    
    # Read summary
    with open(latest_dir / "summary.json") as f:
        summary = json.load(f)
    
    print("\nEpisode Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Read HDF5 data
    with h5py.File(latest_dir / "data.h5", 'r') as f:
        print("\nHDF5 Structure:")
        
        def print_structure(name, obj):
            indent = "  " * name.count("/")
            if isinstance(obj, h5py.Dataset):
                print(f"{indent}{name}: shape={obj.shape}, dtype={obj.dtype}")
            else:
                print(f"{indent}{name}/")
        
        f.visititems(print_structure)
        
        # Sample trajectory data
        print("\nTrajectory Sample (first 10 steps):")
        positions = f['trajectory/positions'][:10]
        actions = f['trajectory/actions'][:10]
        rewards = f['trajectory/rewards'][:10]
        
        for i in range(10):
            print(f"  Step {i}: pos={positions[i]}, action={actions[i]}, reward={rewards[i]:.1f}")
        
        # Final state
        print("\nFinal State:")
        print(f"  Agent position: {f['final_state/agent_position'][:]}")
        print(f"  Rewards collected: {np.sum(f['final_state/reward_collected'][:])}")
        print(f"  Total timesteps: {f['final_state'].attrs['timesteps']}")


if __name__ == "__main__":
    inspect_latest_run()