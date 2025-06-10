# keywords: [export module, public API, initialization, versioning]
"""Neural Network Data Export Module

A high-performance, HDF5-based data export system designed for
efficiency and comprehensive data capture in neural network experiments.
"""

from typing import List

# Version of the export module. Define this FIRST to avoid circular imports.
__version__ = "3.0.0"

# Import the primary DataExporter class
from .jax_data_exporter import JaxDataExporter as DataExporter, ExperimentConfig

# Import data loading utilities
from .loader import EpisodeData, ExperimentLoader

# Import utility functions
from .utils import ensure_numpy

# Public API definition
__all__: List[str] = [
    "DataExporter",
    "ExperimentConfig",
    "EpisodeData",
    "ExperimentLoader",
    "ensure_numpy",
    "__version__",
]
