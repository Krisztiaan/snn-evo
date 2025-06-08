# models/phase_0_7/__init__.py
# keywords: [snn agent, phase 0.7, clock neurons, rate-coding, spiking motors]
"""
Phase 0.7: An "Awakened" SNN Agent with Biologically Plausible I/O.

This version introduces critical architectural changes for more realistic dynamics:
- Specialized neuron populations (input, clock, main, motor).
- Rate-coded sensory input and spiking motor output.
- Intrinsic "clock" neurons to provide a rhythmic drive.
- Time-integrated decision-making based on motor neuron "voting".
"""

from .agent import SnnAgent
from .config import ExperimentConfig, NetworkParams, SnnAgentConfig

__version__ = "0.7.0"
__all__ = [
    "ExperimentConfig",
    "NetworkParams",
    "SnnAgent",
    "SnnAgentConfig",
]
