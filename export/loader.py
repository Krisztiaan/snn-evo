# keywords: [loader, hdf5, data, analysis, simple]
"""Simple data loading utilities for HDF5 exports."""

import h5py
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import json
import zlib


class ExperimentLoader:
    """Load and access experiment data from HDF5."""

    def __init__(self, experiment_dir: Union[str, Path]):
        """Initialize loader with experiment directory or H5 file path."""
        path = Path(experiment_dir)

        # Check if it's a direct H5 file path
        if path.is_file() and path.suffix in [".h5", ".hdf5"]:
            self.h5_path = path
            self.experiment_dir = path.parent
        else:
            # It's a directory, look for H5 files
            self.experiment_dir = path
            h5_files = list(self.experiment_dir.glob("*.h5")) + list(
                self.experiment_dir.glob("*.hdf5")
            )
            if h5_files:
                # Use the most common name first, otherwise the first found
                if (self.experiment_dir / "experiment_data.h5").exists():
                    self.h5_path = self.experiment_dir / "experiment_data.h5"
                else:
                    self.h5_path = h5_files[0]
            else:
                raise FileNotFoundError(f"No HDF5 data files found in {self.experiment_dir}")

        self.h5_file = h5py.File(self.h5_path, "r")

    def get_metadata(self) -> Dict[str, Any]:
        """Get experiment metadata."""
        metadata = {}

        # Get root attributes
        for key, value in self.h5_file.attrs.items():
            if key.startswith("meta_"):
                # User metadata
                clean_key = key[5:]  # Remove 'meta_' prefix
                if isinstance(value, str) and (value.startswith("{") or value.startswith("[")):
                    try:
                        metadata[clean_key] = json.loads(value)
                    except json.JSONDecodeError:
                        metadata[clean_key] = value
                else:
                    metadata[clean_key] = value
            else:
                # System metadata
                metadata[key] = value

        # Ensure simulation parameters are included with defaults for older files
        if "simulation_dt" not in metadata:
            metadata["simulation_dt"] = 1.0  # Default 1ms
        if "neural_sampling_rate" not in metadata:
            metadata["neural_sampling_rate"] = 100  # Default sampling
        if "behavior_sampling_rate" not in metadata:
            metadata["behavior_sampling_rate"] = 1  # Every timestep
        if "data_format_version" not in metadata:
            metadata["data_format_version"] = "1.0"  # Original version

        return metadata

    def get_runtime_info(self) -> Dict[str, Any]:
        """Get runtime environment information."""
        if "runtime" not in self.h5_file:
            return {}
        return dict(self.h5_file["runtime"].attrs)

    def get_git_info(self) -> Optional[Dict[str, Any]]:
        """Get git repository information."""
        if "git_info" not in self.h5_file:
            return None

        git_info = {}
        for key, value in self.h5_file["git_info"].attrs.items():
            if isinstance(value, str) and value.startswith("["):
                try:
                    git_info[key] = json.loads(value)
                except json.JSONDecodeError:
                    git_info[key] = value
            else:
                git_info[key] = value
        return git_info

    def get_code_snapshot(self) -> Dict[str, str]:
        """Get saved code files."""
        if "code_snapshot" not in self.h5_file:
            return {}

        code_files = {}
        code_group = self.h5_file["code_snapshot"]
        for key, value in code_group.attrs.items():
            if key.endswith("_compressed"):
                continue
            if f"{key}_compressed" in code_group.attrs:
                # Handle np.void for compressed data
                if isinstance(value, np.void):
                    value = value.tobytes()
                decompressed = zlib.decompress(value)
                code_files[key] = decompressed.decode("utf-8")
            elif key not in ["saved_files", "snapshot_time"]:
                code_files[key] = value
        return code_files

    def get_config(self) -> Dict[str, Any]:
        """Get experiment configuration."""
        config = {}
        if "config" in self.h5_file:
            config_group = self.h5_file["config"]
            for key, value in config_group.attrs.items():
                if key.endswith("_compressed"):
                    continue

                if config_group.attrs.get(f"{key}_compressed"):
                    # It's compressed JSON
                    if isinstance(value, np.void):
                        value = value.tobytes()
                    decompressed = zlib.decompress(value)
                    config[key] = json.loads(decompressed.decode("utf-8"))
                elif isinstance(value, str) and (value.startswith("{") or value.startswith("[")):
                    try:
                        config[key] = json.loads(value)
                    except json.JSONDecodeError:
                        config[key] = value
                else:
                    config[key] = value
        return config

    def get_network_structure(self) -> Dict[str, Any]:
        """Get network structure data."""
        if "network_structure" not in self.h5_file:
            return {}

        net_group = self.h5_file["network_structure"]
        structure = {}

        # Load neurons
        if "neurons" in net_group:
            structure["neurons"] = {
                key: dataset[:] for key, dataset in net_group["neurons"].items()
            }

        # Load connections
        if "connections" in net_group:
            structure["connections"] = {
                key: dataset[:] for key, dataset in net_group["connections"].items()
            }

        # Load initial weights
        if "initial_weights" in net_group:
            weights_group = net_group["initial_weights"]
            if weights_group.attrs.get("format") == "sparse":
                structure["initial_weights"] = {
                    "format": "sparse",
                    "indices": weights_group["indices"][:],
                    "values": weights_group["values"][:],
                    "shape": weights_group.attrs.get("shape"),
                    "sparsity": weights_group.attrs.get("sparsity"),
                }
            else:
                structure["initial_weights"] = {
                    "format": "dense_1d",
                    "weights": weights_group["weights"][:],
                }

        return structure

    def list_episodes(self) -> List[int]:
        """List all episode IDs."""
        if "episodes" not in self.h5_file:
            return []

        episodes = []
        for name in self.h5_file["episodes"].keys():
            if name.startswith("episode_"):
                try:
                    episode_id = int(name.split("_")[1])
                    episodes.append(episode_id)
                except (ValueError, IndexError):
                    continue
        return sorted(episodes)

    def get_episode(self, episode_id: int) -> "EpisodeData":
        """Get data for a specific episode."""
        episode_name = f"episode_{episode_id:04d}"
        if "episodes" not in self.h5_file or episode_name not in self.h5_file["episodes"]:
            raise ValueError(f"Episode {episode_id} not found in {self.h5_path}")

        return EpisodeData(self.h5_file["episodes"][episode_name])

    def get_episode_summary(self, episode_id: int) -> Dict[str, Any]:
        """Get summary for a specific episode."""
        if "episode_summaries" in self.h5_file:
            ep_name = f"episode_{episode_id:04d}"
            if ep_name in self.h5_file["episode_summaries"]:
                summary = {}
                for key, value in self.h5_file["episode_summaries"][ep_name].attrs.items():
                    if isinstance(value, str) and (value.startswith("{") or value.startswith("[")):
                        try:
                            summary[key] = json.loads(value)
                        except json.JSONDecodeError:
                            summary[key] = value
                    else:
                        summary[key] = value
                return summary

        # Fallback to episode attributes
        episode = self.get_episode(episode_id)
        return episode.get_metadata()

    def get_all_summaries(self) -> List[Dict[str, Any]]:
        """Get summary list for all episodes."""
        summaries = []
        for episode_id in self.list_episodes():
            summary = self.get_episode_summary(episode_id)
            summary["episode_id"] = episode_id
            summaries.append(summary)

        return summaries

    def get_all_checkpoints(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Load all checkpoints."""
        checkpoints = {}
        if "checkpoints" in self.h5_file:
            for name, group in self.h5_file["checkpoints"].items():
                checkpoints[name] = {key: ds[:] for key, ds in group.items()}
        return checkpoints

    def close(self):
        """Close HDF5 file."""
        self.h5_file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        if hasattr(self, "h5_file"):
            try:
                self.h5_file.close()
            except:
                pass


class EpisodeData:
    """Access data from a single episode."""

    def __init__(self, episode_group: h5py.Group):
        """Initialize with HDF5 group."""
        self.group = episode_group

    def get_metadata(self) -> Dict[str, Any]:
        """Get episode metadata."""
        return dict(self.group.attrs)

    def get_neural_states(
        self, start: Optional[int] = None, stop: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """Get neural state data (with optional slicing)."""
        if "neural_states" not in self.group:
            return {}

        neural_group = self.group["neural_states"]
        if "timesteps" not in neural_group:
            return {}

        # Handle slicing
        slice_obj = slice(start, stop)

        data = {"timesteps": neural_group["timesteps"][slice_obj]}
        for key in neural_group.keys():
            if key != "timesteps":
                data[key] = neural_group[key][slice_obj]

        return data

    def get_spikes(self) -> Dict[str, np.ndarray]:
        """Get spike data in sparse format."""
        if "spikes" not in self.group:
            return {}

        spike_group = self.group["spikes"]

        # Check format
        if spike_group.attrs.get("format") == "sparse_rle":
            # Optimized format with run-length encoding
            timesteps = spike_group["timesteps"][:]
            counts = spike_group["counts"][:]
            values = spike_group["values"][:]

            # Expand RLE format
            expanded_timesteps = np.repeat(timesteps, counts)
            return {"timesteps": expanded_timesteps, "neuron_ids": values}
        else:
            # Standard format
            if "timesteps" not in spike_group:
                return {}

            # For backward compatibility, check for different dataset names
            neuron_id_key = "neuron_ids"
            if neuron_id_key not in spike_group:
                neuron_id_key = "values"  # Fallback to old name

            if neuron_id_key not in spike_group:
                return {}

            return {
                "timesteps": spike_group["timesteps"][:],
                "neuron_ids": spike_group[neuron_id_key][:],
            }

    def get_spikes_as_events(self) -> List[Dict[str, Any]]:
        """Get spikes as list of events (for compatibility)."""
        spike_data = self.get_spikes()
        if not spike_data:
            return []

        # Group by timestep
        events = []
        timesteps = spike_data["timesteps"]
        neuron_ids = spike_data["neuron_ids"]

        unique_timesteps = np.unique(timesteps)
        for t in unique_timesteps:
            mask = timesteps == t
            events.append({"timestep": int(t), "neuron_ids": neuron_ids[mask].tolist()})

        return events

    def get_behavior(self) -> Dict[str, np.ndarray]:
        """Get behavior data."""
        if "behavior" not in self.group:
            return {}

        behavior_group = self.group["behavior"]
        if "timesteps" not in behavior_group:
            return {}

        return {key: dataset[:] for key, dataset in behavior_group.items()}

    def get_rewards(self) -> Dict[str, np.ndarray]:
        """Get reward data."""
        if "rewards" not in self.group:
            return {}

        reward_group = self.group["rewards"]

        # Check format
        if reward_group.attrs.get("format") == "sparse_rle":
            # Optimized format with run-length encoding
            timesteps = reward_group["timesteps"][:]
            counts = reward_group["counts"][:]
            values = reward_group["values"][:]

            # Expand RLE format
            expanded_timesteps = np.repeat(timesteps, counts)
            return {"timesteps": expanded_timesteps, "values": values}
        else:
            # Standard format
            if "timesteps" not in reward_group:
                return {}
            return {"timesteps": reward_group["timesteps"][:], "values": reward_group["values"][:]}

    def get_weight_changes(self) -> Dict[str, np.ndarray]:
        """Get weight change data."""
        if "weight_changes" not in self.group:
            return {}

        wc_group = self.group["weight_changes"]

        # Check if optimized format
        if "timesteps" in wc_group and isinstance(wc_group["timesteps"], h5py.Dataset):
            # Optimized format with separate arrays
            data = {}
            for key in ["timesteps", "old_weights", "new_weights", "deltas"]:
                if key in wc_group:
                    data[key] = wc_group[key][:]

            # Handle synapse IDs
            if "synapse_ids" in wc_group:
                data["synapse_ids"] = wc_group["synapse_ids"][:]
            elif "source_ids" in wc_group and "target_ids" in wc_group:
                data["source_ids"] = wc_group["source_ids"][:]
                data["target_ids"] = wc_group["target_ids"][:]

            return data
        else:
            # Standard format
            if "timesteps" not in wc_group:
                return {}
            return {key: dataset[:] for key, dataset in wc_group.items()}

    def get_events(self, event_type: Optional[str] = None) -> Dict[str, Dict[str, np.ndarray]]:
        """Get custom events."""
        if "events" not in self.group:
            return {}

        events_group = self.group["events"]

        if event_type:
            if event_type in events_group:
                event_group = events_group[event_type]
                return {key: dataset[:] for key, dataset in event_group.items()}
            else:
                return {}
        else:
            # Return all event types
            all_events = {}
            for event_type in events_group.keys():
                event_group = events_group[event_type]
                all_events[event_type] = {key: dataset[:] for key, dataset in event_group.items()}
            return all_events

    def get_static_data(self, name: str) -> Dict[str, np.ndarray]:
        """Get static data saved for the episode."""
        if name in self.group:
            return {key: ds[:] for key, ds in self.group[name].items()}
        return {}

    def get_final_state(self) -> Dict[str, np.ndarray]:
        """Get final state if saved."""
        if "final_state" not in self.group:
            return {}

        final_group = self.group["final_state"]
        return {key: dataset[:] for key, dataset in final_group.items()}


def quick_load(experiment_dir: Union[str, Path], episode_id: int = 0) -> Dict[str, Any]:
    """Quick load function for simple access."""
    with ExperimentLoader(experiment_dir) as loader:
        metadata = loader.get_metadata()
        config = loader.get_config()
        network = loader.get_network_structure()

        episode = loader.get_episode(episode_id)
        episode_data = {
            "metadata": episode.get_metadata(),
            "neural_states": episode.get_neural_states(),
            "spikes": episode.get_spikes_as_events(),
            "behavior": episode.get_behavior(),
            "rewards": episode.get_rewards(),
            "weight_changes": episode.get_weight_changes(),
            "events": episode.get_events(),
            "final_state": episode.get_final_state(),
        }

        return {
            "experiment": {"metadata": metadata, "config": config, "network": network},
            "episode": episode_data,
        }
