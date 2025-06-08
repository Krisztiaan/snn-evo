# keywords: [loader, hdf5, data, analysis, simple]
"""Simple data loading utilities for HDF5 exports."""

import json
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import h5py
import numpy as np


class ExperimentLoader:
    """Load and access experiment data from HDF5."""

    def __init__(self, experiment_dir: Union[str, Path]):
        """Initialize loader with experiment directory or H5 file path."""
        path = Path(experiment_dir)

        if path.is_file() and path.suffix in [".h5", ".hdf5"]:
            self.h5_path = path
        else:
            # Assuming a directory, look for a unique .h5 file or a standard name
            h5_files = list(path.glob("*.h5")) + list(path.glob("*.hdf5"))
            if not h5_files:
                raise FileNotFoundError(f"No HDF5 file found in {path}")
            if len(h5_files) > 1:
                # Attempt to find a specific common name if multiple exist, e.g., data.h5
                # This logic might need refinement based on actual naming conventions
                if Path(path / "data.h5") in h5_files:
                    self.h5_path = Path(path / "data.h5")
                elif Path(path / "experiment_data.h5") in h5_files:
                    self.h5_path = Path(path / "experiment_data.h5")
                else:
                    raise FileNotFoundError(
                        f"Multiple HDF5 files found in {path}. Please specify one directly."
                    )
            else:
                self.h5_path = h5_files[0]

        self.h5_file = h5py.File(self.h5_path, "r")

    def get_metadata(self) -> Dict[str, Any]:
        """Get experiment metadata from root attributes."""
        return dict(self.h5_file.attrs)

    def get_config(self) -> Dict[str, Any]:
        """Get experiment configuration."""
        config = {}
        if "config" in self.h5_file and isinstance(self.h5_file["config"], h5py.Group):
            config_group = self.h5_file["config"]
            for key in config_group:
                value_dataset = config_group[key]
                if isinstance(value_dataset, h5py.Dataset):
                    value = value_dataset[()]
                    if isinstance(value, bytes):  # Handle compressed/binary data if necessary
                        try:
                            decompressed = zlib.decompress(value)
                            config[key] = json.loads(decompressed.decode("utf-8"))
                        except (zlib.error, json.JSONDecodeError, UnicodeDecodeError):
                            config[key] = value  # Store as bytes if not decodable JSON
                    else:
                        config[key] = value
        return config

    def get_network_structure(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Get network structure data."""
        if "network_structure" not in self.h5_file or not isinstance(
            self.h5_file["network_structure"], h5py.Group
        ):
            return {}
        net_group = self.h5_file["network_structure"]
        structure: Dict[str, Dict[str, np.ndarray]] = {}
        for key in ["neurons", "connections", "initial_weights"]:
            if key in net_group and isinstance(net_group[key], h5py.Group):
                sub_group = net_group[key]
                structure[key] = {
                    ds_key: ds[:]
                    for ds_key, ds in sub_group.items()
                    if isinstance(ds, h5py.Dataset)
                }
        return structure

    def list_episodes(self) -> List[int]:
        """List all episode IDs."""
        if "episodes" not in self.h5_file or not isinstance(self.h5_file["episodes"], h5py.Group):
            return []
        episode_group = self.h5_file["episodes"]
        episodes = [
            int(name.split("_")[1])
            for name in episode_group
            if name.startswith("episode_") and isinstance(episode_group.get(name), h5py.Group)
        ]
        return sorted(episodes)

    def get_episode(self, episode_id: int) -> "EpisodeData":
        """Get data for a specific episode."""
        episode_name = f"episode_{episode_id:04d}"
        if "episodes" not in self.h5_file or not isinstance(
            self.h5_file.get("episodes"), h5py.Group
        ):
            raise KeyError(f"'episodes' group not found or is not a valid group.")

        episodes_group = self.h5_file["episodes"]
        if episode_name not in episodes_group or not isinstance(
            episodes_group.get(episode_name), h5py.Group
        ):
            raise KeyError(
                f"Episode {episode_id} ({episode_name}) not found or is not a valid group."
            )
        return EpisodeData(episodes_group[episode_name])

    def close(self) -> None:
        """Close HDF5 file."""
        self.h5_file.close()

    def __enter__(self) -> "ExperimentLoader":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


class EpisodeData:
    """Access data from a single episode."""

    def __init__(self, episode_group: h5py.Group):
        self.episode_group = episode_group

    def get_metadata(self) -> Dict[str, Any]:
        return dict(self.episode_group.attrs)

    def get_neural_states(
        self, start: Optional[int] = None, stop: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        neural_states_group = self.episode_group.get("neural_states")
        if not isinstance(neural_states_group, h5py.Group):
            return {}

        states: Dict[str, np.ndarray] = {}
        for dset_name in ["membrane_potential", "spikes", "timesteps"]:
            dataset = neural_states_group.get(dset_name)
            if isinstance(dataset, h5py.Dataset):
                if start is not None and stop is not None:
                    states[dset_name] = dataset[start:stop]
                else:
                    states[dset_name] = dataset[:]
        return states

    def get_behavior(self) -> Dict[str, np.ndarray]:
        behavior_group = self.episode_group.get("behavior")
        if not isinstance(behavior_group, h5py.Group):
            return {}

        behavior_data: Dict[str, np.ndarray] = {}
        for dset_name in ["timesteps", "position", "action", "rewards"]:
            dataset = behavior_group.get(dset_name)
            if isinstance(dataset, h5py.Dataset):
                behavior_data[dset_name] = dataset[:]
        return behavior_data

    def get_static_data(self, name: str) -> Dict[str, np.ndarray]:
        static_group = self.episode_group.get(name)
        if not isinstance(static_group, h5py.Group):
            return {}

        data: Dict[str, np.ndarray] = {}
        for key in static_group:  # Iterate directly over keys if it's a group
            if not isinstance(key, str):  # Ensure key is a string
                continue
            item = static_group.get(key)
            if isinstance(item, h5py.Dataset):
                data[key] = item[:]
        return data

    def get_events(self) -> List[Dict[str, Any]]:
        events_group = self.episode_group.get("events")
        if not isinstance(events_group, h5py.Group):
            return []

        events_list: List[Dict[str, Any]] = []
        for event_name in events_group:  # Iterate directly over member names (keys)
            if not isinstance(event_name, str):  # Ensure key is a string
                continue
            event_item = events_group.get(event_name)
            if isinstance(event_item, h5py.Dataset):
                event_data: Dict[str, Any] = {"name": event_name, "data": event_item[:]}
                # Update with attributes using dict constructor for clarity if preferred, or keep as is
                event_data.update(dict(event_item.attrs.items()))
                events_list.append(event_data)
        return events_list

    def get_weight_changes(self) -> Dict[str, np.ndarray]:
        wc_group = self.episode_group.get("weight_changes")
        if not isinstance(wc_group, h5py.Group):
            return {}

        changes: Dict[str, np.ndarray] = {}
        for key in wc_group:  # Iterate directly over keys
            if not isinstance(key, str):  # Ensure key is a string
                continue
            item = wc_group.get(key)
            if isinstance(item, h5py.Dataset):
                changes[key] = item[:]
        return changes

    def get_rewards(self) -> Dict[str, np.ndarray]:
        """
        Get reward events.
        Returns a dictionary with 'timesteps' and 'values' for non-zero rewards.
        If reward data is not found or no non-zero rewards exist, it returns
        a dictionary with empty numpy arrays for 'timesteps' and 'values'.
        """
        empty_rewards = {"timesteps": np.array([], dtype=int), "values": np.array([], dtype=float)}

        if not isinstance(self.episode_group, h5py.Group):
            return empty_rewards

        behavior_group = self.episode_group.get("behavior")
        if not isinstance(behavior_group, h5py.Group):
            return empty_rewards

        rewards_dataset = behavior_group.get("rewards")
        if not isinstance(rewards_dataset, h5py.Dataset):
            return empty_rewards

        try:
            all_rewards_values = rewards_dataset[:]
        except Exception:
            return empty_rewards

        if not isinstance(all_rewards_values, np.ndarray) or all_rewards_values.ndim == 0:
            return empty_rewards

        try:
            if not np.issubdtype(all_rewards_values.dtype, np.number):
                return empty_rewards
            non_zero_indices = np.where(all_rewards_values > 0)[0]
        except TypeError:
            return empty_rewards

        if non_zero_indices.size > 0:
            return {
                "timesteps": non_zero_indices.astype(int),
                "values": all_rewards_values[non_zero_indices].astype(float),
            }
        else:
            return empty_rewards
