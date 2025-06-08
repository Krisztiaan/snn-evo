# keywords: [test, pytest, integration, export, hdf5, loader]
"""Comprehensive integration tests for the DataExporter module."""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from export import DataExporter, ExperimentLoader


@pytest.fixture(scope="module")
def temp_dir():
    """Create a temporary directory for the tests in this module."""
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)


class TestDataExporter:
    """Test suite for the DataExporter functionality."""

    def test_basic_export_and_load(self, temp_dir: Path):
        """Test basic experiment export and data loading."""
        exp_name = "test_basic"
        with DataExporter(experiment_name=exp_name, output_base_dir=temp_dir) as exporter:
            # Save config and network structure
            config = {"param": "value", "n_neurons": 100}
            exporter.save_config(config)
            neurons = {"neuron_ids": np.arange(100)}
            connections = {"source_ids": np.arange(50), "target_ids": np.arange(50, 100)}
            exporter.save_network_structure(neurons, connections)

            # Record an episode
            with exporter.start_episode(0) as episode:
                for t in range(50):
                    episode.log_timestep(
                        timestep=t,
                        neural_state={"v": np.random.randn(100)},
                        behavior={"pos": np.array([t, t * 2])},
                        reward=float(t),
                    )

        # Verify and load the data
        exp_dir = next(temp_dir.glob(f"{exp_name}_*"))
        assert (exp_dir / "experiment_data.h5").exists()
        assert (exp_dir / "config.json").exists()

        with ExperimentLoader(exp_dir) as loader:
            loaded_config = loader.get_config()
            assert loaded_config["n_neurons"] == 100
            network = loader.get_network_structure()
            assert len(network["neurons"]["neuron_ids"]) == 100
            ep_data = loader.get_episode(0)
            behavior_data = ep_data.get_behavior()
            assert len(behavior_data["timesteps"]) == 50

    def test_all_log_methods(self, temp_dir: Path):
        """Test all logging methods with various data types."""
        with DataExporter(experiment_name="test_methods", output_base_dir=temp_dir) as exporter:
            with exporter.start_episode(0) as episode:
                # Test timestep logging via exporter.log
                exporter.log(
                    timestep=0,
                    neural_state={"scalar": 1.0, "vector": np.ones(10)},
                    behavior={"position": [1.0, 2.0]},
                    reward=1.0,
                )
                # Test static data
                exporter.log_static_episode_data("metadata", {"type": "training"})

        # Verification
        with ExperimentLoader(next(temp_dir.glob("test_methods_*"))) as loader:
            ep_data = loader.get_episode(0)
            neural = ep_data.get_neural_states()
            assert "scalar" in neural and "vector" in neural
            static = ep_data.get_static_data("metadata")
            assert static["type"].item() == b"training"

    def test_compression_options(self, temp_dir: Path):
        """Test different compression options."""
        base_data = np.random.randn(100, 1000)

        # Uncompressed
        with DataExporter("test_nocomp", temp_dir, compression=None) as e:
            with e.start_episode(0) as ep:
                for t in range(100):
                    ep.log_timestep(timestep=t, neural_state={"data": base_data[t]})

        # Gzip
        with DataExporter("test_gzip", temp_dir, compression="gzip", compression_level=9) as e:
            with e.start_episode(0) as ep:
                for t in range(100):
                    ep.log_timestep(timestep=t, neural_state={"data": base_data[t]})

        # LZF
        with DataExporter("test_lzf", temp_dir, compression="lzf") as e:
            with e.start_episode(0) as ep:
                for t in range(100):
                    ep.log_timestep(timestep=t, neural_state={"data": base_data[t]})

        # Compare file sizes
        nocomp_size = (next(temp_dir.glob("test_nocomp_*")) / "experiment_data.h5").stat().st_size
        gzip_size = (next(temp_dir.glob("test_gzip_*")) / "experiment_data.h5").stat().st_size
        lzf_size = (next(temp_dir.glob("test_lzf_*")) / "experiment_data.h5").stat().st_size

        assert gzip_size < nocomp_size
        assert lzf_size < nocomp_size
        assert gzip_size < lzf_size  # Gzip should be more effective on random data

    def test_edge_cases(self, temp_dir: Path):
        """Test edge cases like empty episodes and missing data."""
        # Empty episode
        with DataExporter("test_empty", temp_dir) as e:
            with e.start_episode(0):
                pass  # No data logged

        with ExperimentLoader(next(temp_dir.glob("test_empty_*"))) as loader:
            summary = loader.get_episode(0).get_metadata()
            assert summary["total_timesteps"] == 0

        # Missing optional fields
        with DataExporter("test_missing", temp_dir) as e:
            with e.start_episode(0) as ep:
                ep.log_timestep(timestep=0, neural_state={"v": np.ones(10)})
                ep.log_timestep(timestep=1, behavior={"pos": [1, 2]})

        with ExperimentLoader(next(temp_dir.glob("test_missing_*"))) as loader:
            ep_data = loader.get_episode(0)
            assert "v" in ep_data.get_neural_states()
            assert "pos" in ep_data.get_behavior()

    def test_sequential_episodes(self, temp_dir: Path):
        """Test that sequential episodes don't corrupt each other's state."""
        with DataExporter("test_sequential", temp_dir) as exporter:
            with exporter.start_episode(0) as ep0:
                ep0.log_timestep(timestep=0, neural_state={"data": np.array([0])})
            with exporter.start_episode(1) as ep1:
                ep1.log_timestep(timestep=0, neural_state={"data": np.array([1])})

        with ExperimentLoader(next(temp_dir.glob("test_sequential_*"))) as loader:
            ep0_data = loader.get_episode(0).get_neural_states()
            ep1_data = loader.get_episode(1).get_neural_states()
            assert ep0_data["data"][0] == 0
            assert ep1_data["data"][0] == 1
