# keywords: [inspect export data, read hdf5, view trajectory]
"""Inspect exported data from random agent runs."""

from export.loader import ExperimentLoader
import h5py
import numpy as np
import json
from pathlib import Path
import sys

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def inspect_latest_run():
    """Inspect the most recent run's data."""
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

    print(f"Inspecting latest run: {latest_dir.name}")
    print("=" * 50)

    try:
        with ExperimentLoader(latest_dir) as loader:
            # Print metadata
            print("\nExperiment Metadata:")
            metadata = loader.get_metadata()
            for key, value in metadata.items():
                print(f"  {key}: {value}")

            # Print config
            print("\nExperiment Config:")
            config = loader.get_config()
            for key, value in config.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for sub_key, sub_value in value.items():
                        print(f"    {sub_key}: {sub_value}")
                else:
                    print(f"  {key}: {value}")

            # Print HDF5 structure
            print("\nHDF5 Structure:")

            def print_structure(name, obj):
                indent = "  " * name.count("/")
                if isinstance(obj, h5py.Dataset):
                    print(f"{indent}{name}: shape={obj.shape}, dtype={obj.dtype}")
                else:
                    print(f"{indent}{name}/")
            loader.h5_file.visititems(print_structure)

            # Sample data from first episode
            episodes = loader.list_episodes()
            if not episodes:
                print("\nNo episodes found in this run.")
                return

            episode_id = episodes[0]
            episode = loader.get_episode(episode_id)

            print(f"\n--- Episode {episode_id} Sample Data ---")

            # Episode summary
            summary = loader.get_episode_summary(episode_id)
            print("\nEpisode Summary:")
            for key, value in summary.items():
                print(f"  {key}: {value}")

            # Trajectory data
            behavior = episode.get_behavior()
            if behavior and 'pos_x' in behavior and len(behavior['pos_x']) > 0:
                print("\nTrajectory Sample (first 10 steps):")
                positions_x = behavior['pos_x'][:10]
                positions_y = behavior['pos_y'][:10]
                actions = behavior.get('action', [])[:10]
                rewards_log = episode.get_rewards()

                for i in range(len(positions_x)):
                    reward_val = 0
                    if rewards_log and 'timesteps' in rewards_log:
                        mask = rewards_log['timesteps'] == i
                        if np.any(mask):
                            reward_val = np.sum(rewards_log['rewards'][mask])

                    print(f"  Step {i}: pos=({positions_x[i]}, {positions_y[i]}), "
                          f"action={actions[i] if i < len(actions) else 'N/A'}, "
                          f"reward={reward_val:.1f}")
            else:
                print("\nNo behavior data found in episode.")

    except FileNotFoundError:
        print(f"Could not find 'experiment_data.h5' in {latest_dir}")
    except Exception as e:
        import traceback
        print(f"An error occurred: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    inspect_latest_run()
