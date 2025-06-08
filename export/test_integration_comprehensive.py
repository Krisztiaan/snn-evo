# keywords: [test, integration, comprehensive, coverage, edge cases]
"""Comprehensive integration tests to catch all edge cases and real-world usage."""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
import json
import h5py
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from export import DataExporter, ExperimentLoader


class TestCompressionSupport:
    """Test all compression options and edge cases."""

    def test_lzf_compression(self):
        """Test LZF compression which doesn't support compression_opts."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with DataExporter(
                experiment_name="test_lzf", output_base_dir=temp_dir, compression="lzf"
            ) as exporter:
                with exporter.start_episode(0) as episode:
                    # Test regular data
                    episode.log_timestep(timestep=0, neural_state={"data": np.random.randn(100)})

                    # Test static data (this was failing)
                    episode.log_static_data(
                        "test_static",
                        {"array_data": np.arange(100), "scalar_data": 42.0, "string_data": "test"},
                    )

    def test_all_compression_types(self):
        """Test all supported compression types."""
        compressions = [("gzip", 1), ("gzip", 9), ("lzf", None), (None, None)]

        with tempfile.TemporaryDirectory() as temp_dir:
            for compression, level in compressions:
                config = {
                    "experiment_name": f"test_{compression}",
                    "output_base_dir": temp_dir,
                    "compression": compression,
                }
                if level is not None:
                    config["compression_level"] = level

                with DataExporter(**config) as exporter:
                    with exporter.start_episode(0) as episode:
                        episode.log_timestep(timestep=0, neural_state={"test": np.ones(1000)})


class TestMethodCompatibility:
    """Test all public methods work correctly."""

    def test_log_static_episode_data(self):
        """Test the log_static_episode_data method used by phase_0_11."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with DataExporter(experiment_name="test_static", output_base_dir=temp_dir) as exporter:
                # Start episode first
                with exporter.start_episode(0) as episode:
                    pass

                # Test the method that was failing
                exporter.log_static_episode_data(
                    "world_setup",
                    {
                        "grid_size": [100, 100],
                        "agent_pos": [50, 50],
                        "obstacles": np.zeros((100, 100)),
                    },
                )

    def test_all_log_methods(self):
        """Test all logging methods with various data types."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with DataExporter(experiment_name="test_methods", output_base_dir=temp_dir) as exporter:
                # Test config saving
                exporter.save_config(
                    {
                        "string": "value",
                        "int": 42,
                        "float": 3.14,
                        "list": [1, 2, 3],
                        "dict": {"nested": "value"},
                    }
                )

                # Test network structure
                exporter.save_network_structure(
                    neurons={"ids": np.arange(100)},
                    connections={"sources": np.arange(50), "targets": np.arange(50, 100)},
                    initial_weights=np.random.randn(100, 100),
                )

                with exporter.start_episode(0) as episode:
                    # Test all timestep data types
                    episode.log_timestep(
                        timestep=0,
                        neural_state={
                            "scalar": 1.0,
                            "vector": np.ones(10),
                            "matrix": np.ones((10, 10)),
                            "empty": np.array([]),
                        },
                        spikes=np.array([1, 5, 10]),  # Sparse format
                        behavior={"position": [1.0, 2.0], "velocity": np.array([0.1, 0.2])},
                        reward=1.0,
                    )

                    # Test weight changes
                    episode.log_weight_change(
                        timestep=1,
                        synapse_id=(0, 1),
                        old_weight=0.5,
                        new_weight=0.6,
                        metadata={"rule": "STDP"},
                    )

                    # Test events
                    episode.log_event(event_type="custom", timestep=2, data={"info": "test"})

                    # Test static data
                    episode.log_static_data(
                        "metadata",
                        {"episode_type": "training", "parameters": {"learning_rate": 0.01}},
                    )


class TestPerformanceFeatures:
    """Test performance enhancement features."""

    def test_async_write_queue(self):
        """Test async write functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with DataExporter(
                experiment_name="test_async",
                output_base_dir=temp_dir,
                async_write=True,
                n_async_workers=2,
                neural_sampling_rate=1,  # Sample every timestep to generate more data
            ) as exporter:
                with exporter.start_episode(0) as episode:
                    # Generate enough data to trigger async writes
                    # Buffer threshold is typically 100-1000 items
                    for t in range(200):
                        episode.log_timestep(
                            timestep=t,
                            neural_state={"data": np.random.randn(100)},
                            # Add more data
                            behavior={"pos": np.random.randn(3)},
                        )

                # Check that async queue processed all writes
                if exporter.async_queue:
                    exporter.async_queue.flush()
                    stats = exporter.async_queue.get_stats()
                    assert stats["writes"] > 0

    def test_adaptive_compression(self):
        """Test adaptive compression feature."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with DataExporter(
                experiment_name="test_adaptive", output_base_dir=temp_dir, adaptive_compression=True
            ) as exporter:
                with exporter.start_episode(0) as episode:
                    # Different data types should trigger different compression

                    # Sparse data
                    sparse = np.zeros(10000)
                    sparse[::100] = 1.0
                    episode.log_timestep(0, neural_state={"sparse": sparse})

                    # Random data
                    episode.log_timestep(1, neural_state={"random": np.random.randn(10000)})

                    # Repetitive data
                    episode.log_timestep(2, neural_state={"repetitive": np.tile([1, 2, 3], 3333)})

    def test_profiling(self):
        """Test performance profiling."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with DataExporter(
                experiment_name="test_profiling", output_base_dir=temp_dir, enable_profiling=True
            ) as exporter:
                with exporter.start_episode(0) as episode:
                    for t in range(10):
                        episode.log_timestep(t, neural_state={"data": np.ones(100)})

                # Get profiling report
                stats = exporter.get_performance_stats()
                assert "profiling" in stats
                assert len(stats["profiling"]) > 0


class TestConcurrency:
    """Test thread safety and concurrent access."""

    def test_concurrent_episodes(self):
        """Test multiple episodes writing concurrently."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with DataExporter(
                experiment_name="test_concurrent", output_base_dir=temp_dir
            ) as exporter:

                def write_episode(episode_id):
                    with exporter.start_episode(episode_id) as episode:
                        for t in range(100):
                            episode.log_timestep(
                                timestep=t,
                                neural_state={
                                    "thread_id": episode_id,
                                    "data": np.random.randn(100),
                                },
                            )

                # Run episodes in parallel
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = [executor.submit(write_episode, i) for i in range(4)]
                    for future in futures:
                        future.result()

                # Verify all episodes were written
                exp_dir = list(Path(temp_dir).glob("test_concurrent_*"))[0]
                with ExperimentLoader(exp_dir) as loader:
                    episodes = loader.list_episodes()
                    assert len(episodes) == 4

    def test_concurrent_timesteps(self):
        """Test concurrent writes within an episode."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with DataExporter(
                experiment_name="test_concurrent_ts",
                output_base_dir=temp_dir,
                async_write=True,
                n_async_workers=4,
            ) as exporter:
                with exporter.start_episode(0) as episode:
                    # Simulate concurrent timestep logging
                    def log_batch(start_t):
                        for t in range(start_t, start_t + 25):
                            episode.log_timestep(
                                timestep=t,
                                neural_state={"batch": start_t, "data": np.random.randn(100)},
                            )

                    # Note: This tests internal thread safety, not external concurrent calls
                    for start in [0, 25, 50, 75]:
                        log_batch(start)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_episode(self):
        """Test episode with no data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with DataExporter(experiment_name="test_empty", output_base_dir=temp_dir) as exporter:
                with exporter.start_episode(0) as episode:
                    pass  # No data logged

    def test_missing_data_fields(self):
        """Test logging with missing optional fields."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with DataExporter(experiment_name="test_missing", output_base_dir=temp_dir) as exporter:
                with exporter.start_episode(0) as episode:
                    # Only neural state
                    episode.log_timestep(0, neural_state={"v": np.ones(10)})

                    # Only spikes
                    episode.log_timestep(1, spikes=np.array([1, 2, 3]))

                    # Only behavior
                    episode.log_timestep(2, behavior={"pos": [1, 2]})

                    # Only reward
                    episode.log_timestep(3, reward=1.0)

    def test_large_data(self):
        """Test with very large data arrays."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with DataExporter(
                experiment_name="test_large", output_base_dir=temp_dir, chunk_size=50000
            ) as exporter:
                with exporter.start_episode(0) as episode:
                    # Large neural state
                    large_state = np.random.randn(10000, 100)  # 1M elements
                    episode.log_timestep(0, neural_state={"large": large_state})

                    # Many weight changes
                    for i in range(1000):
                        episode.log_weight_change(
                            timestep=1, synapse_id=(i, i + 1), old_weight=0.5, new_weight=0.51
                        )

    def test_data_types(self):
        """Test various numpy data types."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with DataExporter(experiment_name="test_dtypes", output_base_dir=temp_dir) as exporter:
                with exporter.start_episode(0) as episode:
                    episode.log_timestep(
                        timestep=0,
                        neural_state={
                            "float32": np.array([1.0], dtype=np.float32),
                            "float64": np.array([1.0], dtype=np.float64),
                            "int32": np.array([1], dtype=np.int32),
                            "int64": np.array([1], dtype=np.int64),
                            "bool": np.array([True, False], dtype=bool),
                            "complex": np.array([1 + 2j], dtype=complex),
                        },
                    )


class TestRealWorldScenarios:
    """Test real-world usage patterns."""

    def test_reinforcement_learning_workflow(self):
        """Test typical RL experiment workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Training phase
            with DataExporter(
                experiment_name="rl_experiment",
                output_base_dir=temp_dir,
                compression="lzf",  # Fast compression for training
                async_write=True,
            ) as exporter:
                # Save hyperparameters
                exporter.save_config(
                    {
                        "algorithm": "A3C",
                        "learning_rate": 0.001,
                        "discount": 0.99,
                        "n_neurons": 1000,
                    }
                )

                # Multiple training episodes
                for ep in range(3):
                    with exporter.start_episode(ep) as episode:
                        episode_return = 0.0

                        for t in range(100):
                            # RL-specific logging
                            state = np.random.randn(84, 84)  # Image observation
                            action = np.random.randint(4)
                            reward = np.random.randn() * 0.1
                            episode_return += reward

                            episode.log_timestep(
                                timestep=t,
                                neural_state={"hidden": np.random.randn(256)},
                                behavior={
                                    "state": state,
                                    "action": action,
                                    "value": np.random.randn(),
                                },
                                reward=reward,
                            )

                        # Episode summary
                        episode.log_static_data(
                            "summary", {"total_return": episode_return, "episode_length": 100}
                        )

    def test_neuroscience_workflow(self):
        """Test typical neuroscience experiment workflow."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with DataExporter(
                experiment_name="neuro_experiment",
                output_base_dir=temp_dir,
                neural_sampling_rate=10,  # Sample every 10ms
                compression="gzip",
                compression_level=6,  # Better compression for archival
            ) as exporter:
                # Experiment metadata
                exporter.save_config(
                    {
                        "subject": "mouse_01",
                        "session": "2024-01-01",
                        "brain_region": "V1",
                        "n_channels": 32,
                    }
                )

                # Continuous recording
                with exporter.start_episode(0) as episode:
                    # Simulate 1 second of recording at 1kHz
                    for t in range(1000):
                        # Multi-channel recordings
                        # Local field potential
                        lfp = np.random.randn(32) * 100
                        spikes = np.random.binomial(1, 0.001, 32)  # Spike trains

                        episode.log_timestep(
                            timestep=t,
                            neural_state={"lfp": lfp},
                            spikes=np.where(spikes)[0],  # Sparse format
                            behavior={"pupil_diameter": 3.0 + np.random.randn() * 0.1},
                        )

                        # Stimulus events
                        if t % 200 == 0:
                            episode.log_event(
                                "stimulus", t, {"type": "grating", "orientation": t // 200 * 45}
                            )


def test_performance_at_scale():
    """Test performance with realistic scale."""
    with tempfile.TemporaryDirectory() as temp_dir:
        start_time = time.time()

        with DataExporter(
            experiment_name="scale_test",
            output_base_dir=temp_dir,
            validate_data=False,  # Maximum performance
            compression="lzf",
            async_write=True,
            n_async_workers=4,
        ) as exporter:
            # Large network
            n_neurons = 10000
            exporter.save_network_structure(
                {"ids": np.arange(n_neurons)},
                {
                    "sources": np.random.randint(0, n_neurons, 50000),
                    "targets": np.random.randint(0, n_neurons, 50000),
                },
            )

            with exporter.start_episode(0) as episode:
                # Simulate 10 seconds at 100Hz
                for t in range(1000):
                    # Realistic neural data
                    membrane_potential = -70 + np.random.randn(n_neurons) * 10
                    spike_indices = np.where(membrane_potential > -50)[0]

                    episode.log_timestep(
                        timestep=t, neural_state={"v": membrane_potential}, spikes=spike_indices
                    )

        elapsed = time.time() - start_time
        data_size_mb = (1000 * n_neurons * 8) / (1024 * 1024)
        throughput = data_size_mb / elapsed

        print(f"Scale test: {throughput:.1f} MB/s")
        assert throughput > 50  # Should achieve at least 50 MB/s


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
