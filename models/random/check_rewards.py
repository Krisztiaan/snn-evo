# keywords: [check reward placement, verify constraints]
"""Check that rewards respect distance constraints."""

import h5py
import numpy as np
from pathlib import Path


def check_reward_constraints():
    """Verify rewards are unique and not too close to agent start."""
    # Find latest export
    logs_dir = Path("experiments/random")
    if not logs_dir.exists():
        print("No experiments found yet. Run an experiment first.")
        return
    latest_dir = max(logs_dir.glob("episode_*"), key=lambda p: p.stat().st_mtime)
    
    print(f"Checking: {latest_dir}")
    
    with h5py.File(latest_dir / "data.h5", 'r') as f:
        # Get initial agent position
        initial_pos = f['trajectory/positions'][0]
        print(f"Agent start position: {initial_pos}")
        
        # Get reward positions
        reward_positions = f['final_state/reward_positions'][:]
        print(f"Number of rewards: {len(reward_positions)}")
        
        # Check minimum distance from agent
        min_dist = float('inf')
        for i, reward_pos in enumerate(reward_positions):
            dist = np.linalg.norm(reward_pos - initial_pos)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        print(f"\nClosest reward to agent:")
        print(f"  Position: {reward_positions[closest_idx]}")
        print(f"  Distance: {min_dist:.2f}")
        print(f"  Constraint met: {min_dist >= 3}")
        
        # Check for duplicates
        unique_positions = set()
        duplicates = []
        for i, pos in enumerate(reward_positions):
            pos_tuple = tuple(pos)
            if pos_tuple in unique_positions:
                duplicates.append((i, pos))
            else:
                unique_positions.add(pos_tuple)
        
        print(f"\nUnique positions: {len(unique_positions)}/{len(reward_positions)}")
        if duplicates:
            print("Duplicates found:")
            for idx, pos in duplicates:
                print(f"  Index {idx}: {pos}")
        else:
            print("No duplicates found! ✓")
        
        # Show distribution of distances
        distances = []
        for pos in reward_positions:
            dist = np.linalg.norm(pos - initial_pos)
            distances.append(dist)
        
        distances = np.array(distances)
        print(f"\nDistance distribution from agent:")
        print(f"  Min: {distances.min():.1f}")
        print(f"  Max: {distances.max():.1f}")
        print(f"  Mean: {distances.mean():.1f}")
        print(f"  Rewards within 10 units: {np.sum(distances < 10)}")


if __name__ == "__main__":
    check_reward_constraints()