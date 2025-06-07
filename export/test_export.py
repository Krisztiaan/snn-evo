# keywords: [test, pytest, validation, export module]
"""Test suite for the neural network data export module."""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path

from export import DataExporter, ExperimentLoader


class TestDataExporter:
    """Test the DataExporter functionality."""
    
    def setup_method(self):
        """Create temporary directory for tests."""
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_basic_export(self):
        """Test basic experiment export functionality."""
        # Create exporter
        with DataExporter(
            experiment_name="test",
            output_base_dir=self.temp_dir
        ) as exporter:
            # Save config
            config = {"test": True, "n_neurons": 100}
            exporter.save_config(config)
            
            # Save network
            neurons = {"ids": np.arange(100)}
            connections = {"sources": np.arange(50), "targets": np.arange(50, 100)}
            exporter.save_network_structure(neurons, connections)
            
            # Record episode
            with exporter.start_episode(0) as episode:
                for t in range(100):
                    episode.log_timestep(
                        timestep=t,
                        neural_state={"v": np.random.randn(100)},
                        spikes=np.random.binomial(1, 0.01, 100),
                        behavior={"x": t * 0.1},
                        reward=0.1 if t % 10 == 0 else 0.0
                    )
        
        # Verify files created
        exp_dirs = list(Path(self.temp_dir).glob("test_*"))
        assert len(exp_dirs) == 1
        
        h5_file = exp_dirs[0] / "experiment_data.h5"
        assert h5_file.exists()
        
        config_file = exp_dirs[0] / "config.json"
        assert config_file.exists()
        
    def test_data_loading(self):
        """Test loading exported data."""
        # First create some data
        with DataExporter(
            experiment_name="test_load",
            output_base_dir=self.temp_dir,
            neural_sampling_rate=10
        ) as exporter:
            exporter.save_config({"n_neurons": 50})
            
            with exporter.start_episode(0) as episode:
                for t in range(50):
                    episode.log_timestep(
                        timestep=t,
                        neural_state={"membrane": np.ones(50) * t},
                        spikes=np.zeros(50, dtype=bool),
                        behavior={"position": np.array([t, t*2])},
                        reward=float(t)
                    )
        
        # Load the data
        exp_dir = list(Path(self.temp_dir).glob("test_load_*"))[0]
        
        with ExperimentLoader(exp_dir) as loader:
            # Check metadata
            metadata = loader.get_metadata()
            assert metadata['experiment_name'] == 'test_load'
            assert metadata['episode_count'] == 1
            
            # Check config
            config = loader.get_config()
            assert config['n_neurons'] == 50
            
            # Load episode
            episode_data = loader.get_episode(0)
            
            # Check neural states (sampled every 10 timesteps)
            neural_states = episode_data.get_neural_states()
            assert len(neural_states['timesteps']) == 5  # 50/10
            assert neural_states['membrane'].shape == (5, 50)
            
            # Check behavior (all timesteps)
            behavior = episode_data.get_behavior()
            assert len(behavior['timesteps']) == 50
            assert behavior['position'].shape == (50, 2)
            
            # Check rewards
            rewards = episode_data.get_rewards()
            assert len(rewards['timesteps']) == 49  # Non-zero rewards
            assert np.sum(rewards['values']) == sum(range(1, 50))
            
    def test_compression(self):
        """Test that compression reduces file size."""
        n_timesteps = 1000
        n_neurons = 500
        
        # Without compression
        with DataExporter(
            experiment_name="test_nocomp",
            output_base_dir=self.temp_dir,
            compression=None
        ) as exporter:
            with exporter.start_episode(0) as episode:
                for t in range(n_timesteps):
                    episode.log_timestep(
                        timestep=t,
                        neural_state={"data": np.random.randn(n_neurons)}
                    )
        
        # With compression
        with DataExporter(
            experiment_name="test_comp",
            output_base_dir=self.temp_dir,
            compression='gzip',
            compression_level=6
        ) as exporter:
            with exporter.start_episode(0) as episode:
                for t in range(n_timesteps):
                    episode.log_timestep(
                        timestep=t,
                        neural_state={"data": np.random.randn(n_neurons)}
                    )
        
        # Compare file sizes
        nocomp_file = list(Path(self.temp_dir).glob("test_nocomp_*/experiment_data.h5"))[0]
        comp_file = list(Path(self.temp_dir).glob("test_comp_*/experiment_data.h5"))[0]
        
        nocomp_size = nocomp_file.stat().st_size
        comp_size = comp_file.stat().st_size
        
        # Compressed should be smaller
        assert comp_size < nocomp_size
        compression_ratio = nocomp_size / comp_size
        assert compression_ratio > 2  # At least 2x compression
        
    def test_sparse_data(self):
        """Test sparse data handling for spikes and rewards."""
        n_timesteps = 1000
        n_neurons = 100
        
        with DataExporter(
            experiment_name="test_sparse",
            output_base_dir=self.temp_dir
        ) as exporter:
            with exporter.start_episode(0) as episode:
                total_spikes = 0
                total_rewards = 0
                
                for t in range(n_timesteps):
                    # Sparse spikes (1% rate)
                    spikes = np.random.binomial(1, 0.01, n_neurons)
                    total_spikes += np.sum(spikes)
                    
                    # Sparse rewards
                    reward = 1.0 if t % 100 == 0 else 0.0
                    if reward > 0:
                        total_rewards += 1
                    
                    episode.log_timestep(
                        timestep=t,
                        spikes=spikes,
                        reward=reward
                    )
        
        # Load and verify sparse data
        exp_dir = list(Path(self.temp_dir).glob("test_sparse_*"))[0]
        
        with ExperimentLoader(exp_dir) as loader:
            episode_data = loader.get_episode(0)
            
            # Check spikes
            spikes = episode_data.get_spikes()
            assert len(spikes['timesteps']) == total_spikes
            
            # Check rewards  
            rewards = episode_data.get_rewards()
            assert len(rewards['timesteps']) == total_rewards
            
    def test_weight_changes(self):
        """Test weight change logging."""
        with DataExporter(
            experiment_name="test_weights",
            output_base_dir=self.temp_dir
        ) as exporter:
            with exporter.start_episode(0) as episode:
                # Log some weight changes
                for t in [100, 200, 300]:
                    for i in range(5):
                        episode.log_weight_change(
                            timestep=t,
                            synapse_id=(i, i+10),
                            old_weight=0.5,
                            new_weight=0.51 + i*0.01
                        )
        
        # Load and verify
        exp_dir = list(Path(self.temp_dir).glob("test_weights_*"))[0]
        
        with ExperimentLoader(exp_dir) as loader:
            episode_data = loader.get_episode(0)
            weight_changes = episode_data.get_weight_changes()
            
            assert len(weight_changes['timesteps']) == 15  # 3 timesteps * 5 changes
            assert 'source_ids' in weight_changes
            assert 'target_ids' in weight_changes
            assert 'deltas' in weight_changes
            
    def test_performance(self):
        """Test performance is acceptable."""
        import time
        
        n_timesteps = 5000
        n_neurons = 1000
        
        with DataExporter(
            experiment_name="test_perf",
            output_base_dir=self.temp_dir,
            validate_data=False,  # Disable validation for performance
            async_write=True
        ) as exporter:
            start_time = time.time()
            
            with exporter.start_episode(0) as episode:
                for t in range(n_timesteps):
                    episode.log_timestep(
                        timestep=t,
                        neural_state={"v": np.random.randn(n_neurons)},
                        spikes=np.random.binomial(1, 0.01, n_neurons)
                    )
            
            elapsed = time.time() - start_time
            
        # Should complete in reasonable time
        assert elapsed < 5.0  # Less than 5 seconds for 5k timesteps
        
        # Calculate throughput
        data_size_mb = (n_timesteps * n_neurons * 4) / (1024 * 1024)  # Approximate
        throughput = data_size_mb / elapsed
        
        # Should achieve reasonable throughput
        assert throughput > 10  # At least 10 MB/s


if __name__ == "__main__":
    pytest.main([__file__, "-v"])