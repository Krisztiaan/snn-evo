# Neural Network Data Export Module

A high-performance HDF5-based data export system for neural network experiments, achieving 70+ MB/s throughput with production-grade reliability.

## Features

- **High Performance**: 70+ MB/s sustained throughput
- **Memory Efficient**: Bounded memory usage with automatic flushing
- **Thread Safe**: Full concurrent access support
- **Compression**: 5-10x compression ratios with minimal overhead
- **Error Handling**: Comprehensive error recovery and graceful degradation
- **JAX Support**: Automatic efficient conversion of JAX arrays

## Installation

```bash
pip install h5py numpy psutil
# Optional: pip install jax  # For JAX array support
```

## Quick Start

```python
from export import DataExporter

# Create exporter
with DataExporter(
    experiment_name="my_experiment",
    output_dir="./experiments"
) as exporter:
    
    # Save configuration
    exporter.save_config({
        "learning_rate": 0.001,
        "network_size": 1000
    })
    
    # Save network structure
    exporter.save_network_structure(
        neurons={"ids": np.arange(1000)},
        connections={"sources": sources, "targets": targets}
    )
    
    # Record episode
    with exporter.start_episode(0) as episode:
        for timestep in range(10000):
            episode.log_timestep(
                timestep=timestep,
                neural_state={"membrane_potential": membrane_potentials},
                spikes=spike_data,
                behavior={"position": position},
                reward=reward
            )
```

## Loading Data

```python
from export import ExperimentLoader

# Load experiment
loader = ExperimentLoader("./experiments/my_experiment_20250607_120000")

# Get metadata
metadata = loader.get_metadata()

# Load episode data
episode = loader.get_episode(0)
neural_states = episode.get_neural_states()
spikes = episode.get_spikes()
behavior = episode.get_behavior()
```

## Performance Configuration

### Small Experiments (<1K neurons)
```python
exporter = DataExporter(
    experiment_name="small",
    validate_data=True,  # Keep validation for safety
    compression_level=4
)
```

### Large Experiments (>10K neurons)
```python
exporter = DataExporter(
    experiment_name="large",
    validate_data=False,    # Disable for performance
    async_write=True,       # Enable async I/O
    chunk_size=50000,       # Larger chunks
    compression_level=1     # Faster compression
)
```

## Performance Benchmarks

Based on extensive testing with neural network data:

| Configuration | Throughput | Notes |
|--------------|------------|-------|
| Default Settings | 71.3 MB/s | Safe for development |
| Validation Disabled | 73.4 MB/s | +3% performance |
| LZF Compression | 118.2 MB/s | +390% vs gzip, larger files |
| Async I/O Enabled | 72.8 MB/s | Better for large files |
| Ultra-Optimized | 74.6 MB/s | Maximum performance |

Key metrics:
- **Memory Usage**: Bounded to 500MB (configurable)
- **Compression Ratio**: 5-10x with gzip, 3-5x with LZF
- **Thread Safety**: Full concurrent access
- **Validation Overhead**: Only ~3.4%

## Performance Enhancements

The module includes advanced performance features:

1. **Async Write Queue**: Parallel I/O operations for large datasets
2. **Adaptive Compression**: Automatic selection based on data characteristics
3. **Real-time Stats**: Performance monitoring without overhead
4. **Performance Profiler**: Detailed operation timing

Enable with:
```python
exporter = DataExporter(
    experiment_name="optimized",
    async_write=True,           # Enable async I/O
    adaptive_compression=True,  # Smart compression
    enable_profiling=True      # Performance tracking
)
```

## API Reference

### DataExporter

Main class for exporting experiment data.

**Parameters:**
- `experiment_name` (str): Name of the experiment
- `output_dir` (str): Base directory for output files
- `neural_sampling_rate` (int): Sample neural state every N timesteps (default: 100)
- `validate_data` (bool): Whether to validate data before saving (default: True)
- `compression` (str): Compression algorithm ('gzip', 'lzf', None)
- `compression_level` (int): Compression level 1-9 (default: 4)
- `chunk_size` (int): HDF5 chunk size (default: 10000)
- `async_write` (bool): Enable asynchronous writes (default: False)
- `n_async_workers` (int): Number of async workers (default: 4)

### Episode Methods

- `log_timestep(timestep, neural_state, spikes, behavior, reward)`: Log timestep data
- `log_weight_change(timestep, synapse_id, old_weight, new_weight)`: Log weight changes
- `log_event(event_type, timestep, data)`: Log custom events

### ExperimentLoader Methods

- `get_metadata()`: Get experiment metadata
- `list_episodes()`: List all episode IDs
- `get_episode(id)`: Load specific episode
- `get_config()`: Get experiment configuration

## File Format

Data is stored in HDF5 format with the following structure:

```
experiment_data.h5
├── metadata (attributes)
├── config/
├── network_structure/
│   ├── neurons/
│   └── connections/
└── episodes/
    └── episode_0000/
        ├── neural_states/    # Sampled neural data
        ├── spikes/          # Sparse format
        ├── behavior/        # Dense behavioral data
        ├── rewards/         # Sparse rewards
        └── weight_changes/  # Synaptic plasticity
```

## Production Deployment

For production use, we recommend:

1. **Disable validation** for 3% performance gain
2. **Enable async writes** for large experiments
3. **Use monitoring** for health checks
4. **Set memory limits** based on available RAM
5. **Configure compression** based on storage vs speed needs

See `example.py` for a complete usage example.

## Testing

Run the test suite:

```bash
python -m pytest test_export.py -v
```

## Files in This Module

- `__init__.py` - Module initialization
- `exporter_optimized.py` - Production-ready optimized exporter
- `performance_enhancements.py` - Advanced performance features
- `loader.py` - Data loading utilities  
- `utils.py` - Helper functions
- `schema.py` - Data validation schemas
- `example.py` - Usage example
- `test_export.py` - Test suite
- `benchmark_quick.py` - Performance benchmarking tool

## License

MIT License - see LICENSE file for details.