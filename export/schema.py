# keywords: [schema, validation, versioning, data types]
"""Data schema definitions and validation for export system."""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

SCHEMA_VERSION = "3.0.0"


def validate_timestep_data(
    timestep: int,
    neural_state: Optional[Dict[str, Any]] = None,
    spikes: Optional[Any] = None,
    behavior: Optional[Dict[str, Any]] = None,
    reward: Optional[float] = None,
    **kwargs: Any,
) -> List[str]:
    """Validate timestep data and return list of warnings."""
    warnings: List[str] = []

    if not isinstance(timestep, (int, np.integer)) or timestep < 0:
        warnings.append(f"timestep should be a non-negative int, got {timestep}")

    if neural_state is not None and not isinstance(neural_state, dict):
        warnings.append(f"neural_state should be a dict, got {type(neural_state)}")

    if behavior is not None and not isinstance(behavior, dict):
        warnings.append(f"behavior should be a dict, got {type(behavior)}")

    if reward is not None and not isinstance(reward, (int, float, np.number)):
        warnings.append(f"reward should be numeric, got {type(reward)}")

    return warnings


def validate_weight_change(
    timestep: int, synapse_id: Union[int, Tuple[int, int]], old_weight: float, new_weight: float
) -> List[str]:
    """Validate weight change data."""
    warnings: List[str] = []

    if not isinstance(timestep, (int, np.integer)) or timestep < 0:
        warnings.append(f"Invalid timestep for weight change: {timestep}")

    if not isinstance(synapse_id, (int, np.integer, tuple)):
        warnings.append(f"synapse_id must be int or tuple, got {type(synapse_id)}")

    for name, weight in [("old_weight", old_weight), ("new_weight", new_weight)]:
        if not isinstance(weight, (float, np.floating)):
            warnings.append(f"{name} should be float, got {type(weight)}")

    return warnings


def validate_network_structure(neurons: Dict[str, Any], connections: Dict[str, Any]) -> List[str]:
    """Validate network structure data."""
    warnings: List[str] = []

    if "neuron_ids" not in neurons:
        warnings.append("`neurons` dict missing required field 'neuron_ids'")
    else:
        n_neurons = len(neurons["neuron_ids"])
        for key, value in neurons.items():
            if hasattr(value, "__len__") and len(value) != n_neurons:
                warnings.append(f"neurons['{key}'] length {len(value)} != n_neurons {n_neurons}")

    if "source_ids" not in connections or "target_ids" not in connections:
        warnings.append("`connections` dict missing 'source_ids' or 'target_ids'")
    elif len(connections["source_ids"]) != len(connections["target_ids"]):
        warnings.append("'source_ids' and 'target_ids' have different lengths")

    return warnings
