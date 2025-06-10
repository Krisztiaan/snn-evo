# keywords: [exporter protocol, type safety, performance, jit compatible]
"""Exporter interface protocol for high-performance data logging.

Design principles:
- No I/O during episodes (only at boundaries)
- JIT-compatible timestep logging via immutable buffers
- Batched device-to-host transfers
- Type-safe episode data format
"""

from typing import Protocol, runtime_checkable, Tuple, Optional, Dict, Callable
from pathlib import Path
from jax import Array

from .config import ExperimentConfig
from .episode_data import StepData

@runtime_checkable
class EpisodeBufferProtocol(Protocol):
    """Immutable buffer for collecting episode data on device."""
    # Pre-allocated arrays
    timesteps: Array  # Shape: (max_timesteps,) int32
    gradients: Array  # Shape: (max_timesteps,) float32
    actions: Array    # Shape: (max_timesteps,) int32
    neural_states: Optional[Array]  # Shape: (max_timesteps, n_neurons) float32 or None
    
    # Metadata
    current_size: int
    max_size: int
    episode_id: int

# Type alias for the JIT-compiled logging function
LogTimestepFunction = Callable[
    [
        EpisodeBufferProtocol,  # buffer: Current episode buffer
        StepData                # step_data: All data for the current timestep
    ],
    EpisodeBufferProtocol       # Returns: Updated buffer
]

@runtime_checkable
class ExporterProtocol(Protocol):
    """Protocol for high-performance data exporters.
    
    Separates high-frequency calls (during episode) from I/O operations (between episodes).
    """
    
    # Exporter metadata
    VERSION: str  # e.g., "2.0.0"
    
    def __init__(
        self, 
        experiment_name: str, 
        config: ExperimentConfig, 
        output_dir: Path,
        compression: Optional[str] = "gzip",
        log_to_console: bool = True
    ) -> None:
        """Initialize exporter with experiment configuration.
        
        Args:
            experiment_name: Unique identifier for this experiment run
            config: Complete experiment configuration
            output_dir: Base directory for data storage
            compression: HDF5 compression algorithm (None, "gzip", "lzf")
            log_to_console: Whether to print progress to console
        """
        ...
    
    # === High-performance episode operations (JIT-compatible) ===
    
    def start_episode(self, episode_id: int) -> Tuple[EpisodeBufferProtocol, LogTimestepFunction]:
        """Start new episode, return buffer and JIT-compiled logging function.
        
        Args:
            episode_id: Unique episode identifier
            
        Returns:
            Tuple of:
            - Episode buffer (on device)
            - JIT-compiled logging function
        """
        ...
    
    # === I/O operations (between episodes only) ===
    
    def end_episode(
        self,
        buffer: EpisodeBufferProtocol,
        world_reward_tracking: Dict[str, Array],
        success: bool = False
    ) -> Dict[str, float]:
        """Finalize episode, transfer data from device, and persist.
        
        Args:
            buffer: Final episode buffer from device
            world_reward_tracking: Reward placement/collection data from world
            success: Whether episode ended successfully
            
        Returns:
            Episode statistics dict with keys:
            - episode_id: int
            - timesteps: int
            - total_rewards: float (count of gradient=1.0 events)
            - duration_seconds: float
            - steps_per_second: float
            - action_entropy: float
            - mean_neural_activity: float (if applicable)
        """
        ...
    
    def save_network_structure(
        self,
        neurons: Dict[str, Array],
        connections: Dict[str, Array],
        initial_weights: Optional[Array] = None
    ) -> None:
        """Save static network structure (called once at experiment start).
        
        Args:
            neurons: Dict with keys:
                - neuron_ids: Array shape (n_neurons,) int32
                - neuron_types: Array shape (n_neurons,) int32 (0=input, 1=hidden, 2=output)
                - is_excitatory: Array shape (n_neurons,) bool
            connections: Dict with keys:
                - source_ids: Array shape (n_connections,) int32
                - target_ids: Array shape (n_connections,) int32
                - is_plastic: Array shape (n_connections,) bool
            initial_weights: Initial weight matrix shape (n_neurons, n_neurons) float32
        """
        ...
    
    def save_checkpoint(
        self,
        episode_id: int,
        weights: Array,
        optimizer_state: Optional[Dict[str, Array]] = None
    ) -> None:
        """Save model checkpoint for resuming experiments.
        
        Args:
            episode_id: Current episode number
            weights: Current weight matrix shape (n_neurons, n_neurons) float32
            optimizer_state: Optional optimizer state with Arrays
        """
        ...
    
    def save_experiment_metadata(
        self,
        agent_version: str,
        agent_name: str,
        agent_description: str,
        world_version: str
    ) -> None:
        """Save experiment metadata (called at start).
        
        Args:
            agent_version: Agent implementation version
            agent_name: Agent model name
            agent_description: Agent description
            world_version: World implementation version
        """
        ...
    
    def finalize(self) -> None:
        """Finalize experiment, compute aggregate statistics, close files."""
        ...
    
    # === Context manager for safe cleanup ===
    
    def __enter__(self) -> "ExporterProtocol":
        """Enter context manager."""
        ...
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager, ensure cleanup."""
        ...