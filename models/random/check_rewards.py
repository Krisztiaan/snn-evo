# keywords: [check reward placement, verify constraints]
"""Check that rewards respect distance constraints."""

from export.loader import ExperimentLoader
import h5py
import numpy as np
from pathlib import Path
import sys

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def check_reward_constraints():
    """Verify rewards are unique and not too close to agent start."""
    # Find latest experiment run
    base_dir = Path("experiments/random")
    if not base_dir.exists():
        print("No experiments found for 'random' model. Run an experiment first.")
        return

    all_runs = [d for d in base_dir.iterdir() if d.is_dir()]
    if not all_runs:
        print(f"No experiment runs found in {base_dir}")
        return

    latest_dir = max(all_runs, key=lambda p: p.stat().st_mtime)

    print(f"Checking latest run: {latest_dir.name}")

    try:
        with ExperimentLoader(latest_dir) as loader:
            # Load data from the first episode
            episodes = loader.list_episodes()
            if not episodes:
                print("No episodes found in this run.")
                return
            episode_id = episodes[0]
            episode = loader.get_episode(episode_id)

            # Get initial agent position
            behavior = episode.get_behavior()
            if not behavior or 'pos_x' not in behavior or 'pos_y' not in behavior or len(behavior['pos_x']) == 0:
                print("Could not find behavior data for initial position.")
                return
            initial_pos = np.array(
                [behavior['pos_x'][0], behavior['pos_y'][0]])
            print(f"Agent start position: {initial_pos}")

            # Get reward positions
            world_setup = episode.get_static_data("world_setup")
            if not world_setup or 'reward_positions' not in world_setup:
                print("Could not find reward positions in static data.")
                return
            reward_positions = world_setup['reward_positions']
            print(f"Number of rewards: {len(reward_positions)}")

            # Check minimum distance from agent
            min_dist = float('inf')
            closest_idx = -1
            for i, reward_pos in enumerate(reward_positions):
                dist = np.linalg.norm(reward_pos - initial_pos)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i

            print(f"\nClosest reward to agent:")
            if closest_idx != -1:
                print(f"  Position: {reward_positions[closest_idx]}")
                print(f"  Distance: {min_dist:.2f}")
                print(f"  Constraint met (>= 3): {min_dist >= 3}")
            else:
                print("  No rewards found to check distance.")

            # Check for duplicates
            unique_positions = set()
            duplicates = []
            for i, pos in enumerate(reward_positions):
                pos_tuple = tuple(pos)
                if pos_tuple in unique_positions:
                    duplicates.append((i, pos))
                else:
                    unique_positions.add(pos_tuple)

            print(
                f"\nUnique positions: {len(unique_positions)}/{len(reward_positions)}")
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

    except FileNotFoundError:
        print(f"Could not find 'experiment_data.h5' in {latest_dir}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    check_reward_constraints()
