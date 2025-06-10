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
        """Get experiment configuration from HDF5 or JSON file.
        
        First tries HDF5 (faster), then falls back to JSON file.
        """
        # Try HDF5 first
        if "config" in self.h5_file and isinstance(self.h5_file["config"], h5py.Group):
            config_group = self.h5_file["config"]
            config = {}
            # Read from attrs (more efficient than datasets for small data)
            for key, value in config_group.attrs.items():
                if isinstance(value, str) and value.startswith('{'):
                    # JSON-encoded complex type
                    try:
                        config[key] = json.loads(value)
                    except json.JSONDecodeError:
                        config[key] = value
                else:
                    config[key] = value
            return config
        
        # Fallback to JSON file
        config_path = self.h5_path.parent / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                return json.load(f)
        
        return {}

    def get_network_structure(self) -> Dict[str, Any]:
        """Get network structure data."""
        if "network_structure" not in self.h5_file or not isinstance(
            self.h5_file["network_structure"], h5py.Group
        ):
            return {}
        net_group = self.h5_file["network_structure"]
        structure: Dict[str, Any] = {}
        
        # Load neurons and connections groups
        for key in ["neurons", "connections"]:
            if key in net_group and isinstance(net_group[key], h5py.Group):
                sub_group = net_group[key]
                structure[key] = {}
                for ds_key in sub_group:
                    ds = sub_group[ds_key]
                    if isinstance(ds, h5py.Dataset):
                        # Decode string arrays if needed
                        if ds.dtype.kind == 'S' or ds.dtype.kind == 'O':
                            structure[key][ds_key] = np.array([s.decode('utf-8') if isinstance(s, bytes) else s for s in ds[:]])
                        else:
                            structure[key][ds_key] = ds[:]
        
        # Handle initial_weights - can be either a group or a dataset
        if "initial_weights" in net_group:
            iw = net_group["initial_weights"]
            if isinstance(iw, h5py.Group):
                # Dict format
                structure["initial_weights"] = {
                    ds_key: ds[:]
                    for ds_key, ds in iw.items()
                    if isinstance(ds, h5py.Dataset)
                }
            elif isinstance(iw, h5py.Dataset):
                # Direct array format
                structure["initial_weights"] = iw[:]
                
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
    
    def list_checkpoints(self) -> List[str]:
        """List all available checkpoints."""
        if "checkpoints" not in self.h5_file or not isinstance(
            self.h5_file["checkpoints"], h5py.Group
        ):
            return []
        return sorted(self.h5_file["checkpoints"].keys())
    
    def get_checkpoint(self, name: str) -> Dict[str, Any]:
        """Load a specific checkpoint."""
        if "checkpoints" not in self.h5_file:
            raise KeyError("No checkpoints found")
        
        ckpt_group = self.h5_file["checkpoints"].get(name)
        if not isinstance(ckpt_group, h5py.Group):
            raise KeyError(f"Checkpoint '{name}' not found")
        
        checkpoint = {}
        # Load attributes
        for key, value in ckpt_group.attrs.items():
            if isinstance(value, str) and (value.startswith('[') or value.startswith('{')):
                try:
                    checkpoint[key] = json.loads(value)
                except json.JSONDecodeError:
                    checkpoint[key] = value
            else:
                checkpoint[key] = value
        
        # Load datasets and subgroups
        for key in ckpt_group:
            item = ckpt_group[key]
            if isinstance(item, h5py.Dataset):
                checkpoint[key] = item[:]
            elif isinstance(item, h5py.Group):
                # Load nested group
                checkpoint[key] = {}
                for subkey, subvalue in item.attrs.items():
                    checkpoint[key][subkey] = subvalue
                for subkey in item:
                    if isinstance(item[subkey], h5py.Dataset):
                        checkpoint[key][subkey] = item[subkey][:]
        
        return checkpoint

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
        """Get all neural state data from the episode."""
        neural_states_group = self.episode_group.get("neural_states")
        if not isinstance(neural_states_group, h5py.Group):
            return {"timesteps": np.array([])}  # Return empty but valid structure

        states: Dict[str, np.ndarray] = {}
        # Read all datasets in the group, not just hardcoded names
        for key in neural_states_group:
            dataset = neural_states_group[key]
            if isinstance(dataset, h5py.Dataset):
                if start is not None and stop is not None:
                    states[key] = dataset[start:stop]
                else:
                    states[key] = dataset[:]
        return states

    def get_behavior(self) -> Dict[str, np.ndarray]:
        """Get all behavior data from the episode."""
        behavior_group = self.episode_group.get("behavior")
        if not isinstance(behavior_group, h5py.Group):
            return {"timesteps": np.array([])}  # Return empty but valid structure

        behavior_data: Dict[str, np.ndarray] = {}
        # Read all datasets in the group
        for key in behavior_group:
            dataset = behavior_group[key]
            if isinstance(dataset, h5py.Dataset):
                behavior_data[key] = dataset[:]
        return behavior_data

    def get_static_data(self, name: str) -> Dict[str, Any]:
        """Get static data, handling scalars and arrays properly."""
        static_group = self.episode_group.get(name)
        if not isinstance(static_group, h5py.Group):
            return {}

        data: Dict[str, Any] = {}
        for key in static_group:
            item = static_group[key]
            if isinstance(item, h5py.Dataset):
                # Handle scalar datasets
                if item.shape == ():
                    value = item[()]  # Use [()] for scalar access
                    # Decode bytes to string if needed
                    if isinstance(value, bytes):
                        data[key] = value.decode('utf-8')
                    else:
                        data[key] = value
                else:
                    # Array data
                    data[key] = item[:]
        return data

    def get_events(self) -> List[Dict[str, Any]]:
        """Get all events from the episode."""
        events_group = self.episode_group.get("events")
        if not isinstance(events_group, h5py.Group):
            return []

        events_list: List[Dict[str, Any]] = []
        for event_name in sorted(events_group.keys()):  # Sort by timestamp in name
            event_item = events_group[event_name]
            if isinstance(event_item, h5py.Group):
                # Extract event data from group attributes
                event_data: Dict[str, Any] = {
                    "name": event_item.attrs.get("name", event_name),
                    "timestep": event_item.attrs.get("timestep", 0),
                    "data": {}
                }
                # Get all other attributes as data
                for key, value in event_item.attrs.items():
                    if key not in ["name", "timestep"]:
                        event_data["data"][key] = value
                events_list.append(event_data)
        return events_list

    def get_weight_changes(self) -> Dict[str, np.ndarray]:
        """Get weight change data from the plasticity group."""
        # Weight changes are stored in the plasticity group
        plasticity_group = self.episode_group.get("plasticity")
        if not isinstance(plasticity_group, h5py.Group):
            return {}

        changes: Dict[str, np.ndarray] = {}
        for key in plasticity_group:
            item = plasticity_group[key]
            if isinstance(item, h5py.Dataset):
                if key == "learning_rules":
                    # Decode string data
                    changes[key] = np.array([s.decode('utf-8').strip() for s in item[:]])
                else:
                    changes[key] = item[:]
        
        # Calculate deltas if not already present
        if "old_weights" in changes and "new_weights" in changes and "deltas" not in changes:
            changes["deltas"] = changes["new_weights"] - changes["old_weights"]
            
        return changes
    
    def get_rewards(self) -> Dict[str, np.ndarray]:
        """Get reward data from the episode."""
        rewards_group = self.episode_group.get("rewards")
        if not isinstance(rewards_group, h5py.Group):
            return {"timesteps": np.array([]), "values": np.array([])}
        
        rewards_data: Dict[str, np.ndarray] = {}
        for key in rewards_group:
            dataset = rewards_group[key]
            if isinstance(dataset, h5py.Dataset):
                rewards_data[key] = dataset[:]
        return rewards_data
    
    def get_spikes(self) -> Dict[str, np.ndarray]:
        """Get spike data from the episode."""
        spikes_group = self.episode_group.get("spikes")
        if not isinstance(spikes_group, h5py.Group):
            # Try neural_states group as fallback
            neural_states = self.get_neural_states()
            if "spikes" in neural_states:
                return {"timesteps": neural_states.get("timesteps", np.array([])), 
                        "spikes": neural_states["spikes"]}
            return {}
        
        spikes_data: Dict[str, np.ndarray] = {}
        for key in spikes_group:
            dataset = spikes_group[key]
            if isinstance(dataset, h5py.Dataset):
                spikes_data[key] = dataset[:]
        return spikes_data
