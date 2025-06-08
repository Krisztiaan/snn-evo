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

        if path.is_file() and path.suffix in [".h5", ".hdf5"]:
            self.h5_path = path
            self.experiment_dir = path.parent
        else:
            self.experiment_dir = path
            h5_files = list(self.experiment_dir.glob("*.h5")) + list(
                self.experiment_dir.glob("*.hdf5")
            )
            if (self.experiment_dir / "experiment_data.h5").exists():
                self.h5_path = self.experiment_dir / "experiment_data.h5"
            elif h5_files:
                self.h5_path = h5_files[0]
            else:
                raise FileNotFoundError(f"No HDF5 data files found in {self.experiment_dir}")

        self.h5_file = h5py.File(self.h5_path, "r")

    def get_metadata(self) -> Dict[str, Any]:
        """Get experiment metadata from root attributes."""
        return dict(self.h5_file.attrs)

    def get_config(self) -> Dict[str, Any]:
        """Get experiment configuration."""
        config = {}
        if "config" in self.h5_file:
            config_group = self.h5_file["config"]
            for key, value in config_group.attrs.items():
                if key.endswith("_compressed"):
                    continue
                if config_group.attrs.get(f"{key}_compressed"):
                    if isinstance(value, np.void):
                        value = value.tobytes()
                    decompressed = zlib.decompress(value)
                    config[key] = json.loads(decompressed.decode("utf-8"))
                else:
                    config[key] = value
        return config

    def get_network_structure(self) -> Dict[str, Dict[str, np.ndarray]]:
        """Get network structure data."""
        if "network_structure" not in self.h5_file:
            return {}
        net_group = self.h5_file["network_structure"]
        structure: Dict[str, Dict[str, np.ndarray]] = {}
        for key in ["neurons", "connections", "initial_weights"]:
            if key in net_group:
                structure[key] = {ds_key: ds[:] for ds_key, ds in net_group[key].items()}
        return structure

    def list_episodes(self) -> List[int]:
        """List all episode IDs."""
        if "episodes" not in self.h5_file:
            return []
        episodes = [
            int(name.split("_")[1])
            for name in self.h5_file["episodes"]
            if name.startswith("episode_")
        ]
        return sorted(episodes)

    def get_episode(self, episode_id: int) -> "EpisodeData":
        """Get data for a specific episode."""
        episode_name = f"episode_{episode_id:04d}"
        if "episodes" not in self.h5_file or episode_name not in self.h5_file["episodes"]:
            raise ValueError(f"Episode {episode_id} not found in {self.h5_path}")
        return EpisodeData(self.h5_file["episodes"][episode_name])

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
        """Initialize with HDF5 group for a specific episode."""
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
        slc = slice(start, stop)
        return {
            key: ds[slc]
            for key, ds in self.group["neural_states"].items()
            if isinstance(ds, h5py.Dataset)
        }

    def get_behavior(self) -> Dict[str, np.ndarray]:
        """Get behavior data."""
        if "behavior" not in self.group:
            return {}
        return {key: ds[:] for key, ds in self.group["behavior"].items()}

    def get_static_data(self, name: str) -> Dict[str, np.ndarray]:
        """Get static data saved for the episode."""
        if name not in self.group:
            return {}
        return {key: ds[:] for key, ds in self.group[name].items()}
