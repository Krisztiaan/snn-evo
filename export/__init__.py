# keywords: [export module, public API, initialization]
"""Neural Network Data Export Module

A high-performance, HDF5-based data export system designed for
efficiency and comprehensive data capture in neural network experiments.
"""

from typing import List

# Import the primary DataExporter and Episode classes
from .data_exporter import DataExporter
from .episode import Episode

# Import data loading utilities
from .loader import EpisodeData, ExperimentLoader

# Import utility functions
from .utils import ensure_numpy

# Public API definition
__all__: List[str] = [
    "DataExporter",
    "Episode",
    "EpisodeData",
    "ExperimentLoader",
    "ensure_numpy",
]

# Version of the export module
__version__ = "3.0.0"
