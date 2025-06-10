# keywords: [episode data, standardized format, type safety, step data]
"""Standardized episode data format for agent-exporter interface."""

from typing import NamedTuple, Optional, Dict
from jax import Array

class EpisodeData(NamedTuple):
    """Standardized data format returned by agents after episode completion.
    
    All arrays should have time as first dimension.
    Optional fields can be None for non-neural agents.
    """
    # Core trajectory data (always present)
    actions: Array  # Shape: (timesteps,) int32, actions taken
    gradients: Array  # Shape: (timesteps,) float32, observations received
    
    # Neural data (optional)
    neural_states: Optional[Array] = None  # Shape: (timesteps, n_neurons) float32
    spikes: Optional[Array] = None  # Shape: (timesteps, n_neurons) bool
    
    # Weight data (optional, for plastic networks)
    weights_initial: Optional[Array] = None  # Shape: (n_neurons, n_neurons) float32
    weights_final: Optional[Array] = None  # Shape: (n_neurons, n_neurons) float32
    weight_changes: Optional[Array] = None  # Sparse format: (n_changes, 4) [timestep, src, dst, delta]
    
    # Learning signals (optional)
    eligibility_traces: Optional[Array] = None  # Shape: (timesteps, n_synapses) float32
    dopamine_levels: Optional[Array] = None  # Shape: (timesteps,) float32
    
    # Performance metrics computed by agent
    total_reward_events: int = 0  # Number of times gradient was 1.0
    exploration_entropy: Optional[float] = None  # Action distribution entropy


class StepData(NamedTuple):
    """Data for a single timestep, passed to the logger."""
    timestep: int
    gradient: Array
    action: Array
    reward: Array
    neural_v: Array  # Example: membrane potential
    # Additional neural data can be added via the neural_data dict
    neural_data: Dict[str, Array]  # Other neural data to log per-step