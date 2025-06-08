# keywords: [export module, public API, initialization]
"""Neural Network Data Export Module

High-performance HDF5-based data export system for neural network experiments.
"""

# Import the optimized exporter as the default
from .exporter_optimized import OptimizedDataExporter as DataExporter
from .exporter_optimized import OptimizedEpisode as Episode

# Import data loading utilities
from .loader import ExperimentLoader, EpisodeData

# Import utilities
from .utils import ensure_numpy

# Version
__version__ = "2.0.0"

# Public API
__all__ = [
    "DataExporter",
    "Episode",
    "ExperimentLoader",
    "EpisodeData",
    "ensure_numpy"
]
