# keywords: [export, hdf5, optimized, high-performance, interface]
"""High-performance HDF5-based data export system for meta-learning SNN experiments.

This module provides an optimized, efficient data export system using HDF5 for storage.
All data is stored in a single HDF5 file per experiment with proper organization.

Features:
- Batch resizing for >100x write speed improvement
- Efficient sparse data formats for spikes and rewards
- Full compression support (5-10x file size reduction)
- Buffered I/O for reduced overhead
- Memory-efficient streaming to disk
"""

# Check for h5py availability
try:
    import h5py
except ImportError:
    raise ImportError(
        "h5py is required for the data export system. "
        "Install it with: pip install h5py"
    )

# Import standard exporter for backwards compatibility
from .exporter import DataExporter as StandardDataExporter
from .exporter import Episode as StandardEpisode

# Import optimized exporter
from .exporter_optimized import OptimizedDataExporter, OptimizedEpisode

# Import loader (works with both exporters)
from .loader import ExperimentLoader, EpisodeData, quick_load

# Import utilities
from .utils import ensure_numpy, create_output_dir
from .schema import SCHEMA_VERSION, validate_timestep_data, validate_weight_change, validate_network_structure

# Use optimized exporter by default
DataExporter = OptimizedDataExporter
Episode = OptimizedEpisode

__all__ = [
    # Default (optimized) exporter
    'DataExporter',
    'Episode',
    
    # Explicit optimized exporter
    'OptimizedDataExporter',
    'OptimizedEpisode',
    
    # Standard exporter (for compatibility)
    'StandardDataExporter',
    'StandardEpisode',
    
    # Loader
    'ExperimentLoader',
    'EpisodeData', 
    'quick_load',
    
    # Utilities
    'ensure_numpy',
    'create_output_dir',
    
    # Validation
    'SCHEMA_VERSION',
    'validate_timestep_data',
    'validate_weight_change',
    'validate_network_structure'
]

__version__ = "2.0.0"