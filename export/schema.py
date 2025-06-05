# keywords: [schema, validation, versioning, data types]
"""Data schema definitions and validation for export system."""

from typing import Dict, Any, Optional, List, Union
import numpy as np

SCHEMA_VERSION = "1.0.0"

# Expected data shapes and types for validation
EXPECTED_TYPES = {
    'timestep': (int, np.integer),
    'reward': (float, np.floating, int, np.integer),
    'neuron_ids': (list, np.ndarray),
    'membrane_potentials': (np.ndarray, list),
    'spikes': (np.ndarray, list, bool),
    'synapse_id': (int, np.integer, tuple, list),
    'old_weight': (float, np.floating),
    'new_weight': (float, np.floating),
}

def validate_timestep_data(
    timestep: int,
    neural_state: Optional[Dict[str, Any]] = None,
    spikes: Optional[Any] = None,
    behavior: Optional[Dict[str, Any]] = None,
    reward: Optional[float] = None
) -> List[str]:
    """Validate timestep data and return list of warnings."""
    warnings = []
    
    # Check timestep
    if not isinstance(timestep, (int, np.integer)):
        warnings.append(f"timestep should be int, got {type(timestep)}")
    elif timestep < 0:
        warnings.append(f"timestep should be non-negative, got {timestep}")
        
    # Check neural state
    if neural_state is not None:
        if not isinstance(neural_state, dict):
            warnings.append(f"neural_state should be dict, got {type(neural_state)}")
        else:
            for key, value in neural_state.items():
                if hasattr(value, 'shape'):
                    # Array-like object
                    if value.ndim > 2:
                        warnings.append(f"neural_state[{key}] has {value.ndim} dimensions, expected 1 or 2")
                        
    # Check spikes
    if spikes is not None:
        if hasattr(spikes, 'dtype'):
            if spikes.dtype not in [bool, np.bool_]:
                # Convert to boolean array for validation
                try:
                    spikes_bool = np.asarray(spikes, dtype=bool)
                    if spikes_bool.ndim != 1:
                        warnings.append(f"spikes should be 1D, got shape {spikes_bool.shape}")
                except Exception as e:
                    warnings.append(f"Could not validate spikes: {e}")
                    
    # Check behavior
    if behavior is not None:
        if not isinstance(behavior, dict):
            warnings.append(f"behavior should be dict, got {type(behavior)}")
        else:
            # Common fields
            for field in ['x', 'y']:
                if field in behavior:
                    val = behavior[field]
                    if not isinstance(val, (int, float, np.number)):
                        warnings.append(f"behavior[{field}] should be numeric, got {type(val)}")
                        
    # Check reward
    if reward is not None:
        if not isinstance(reward, (int, float, np.number)):
            warnings.append(f"reward should be numeric, got {type(reward)}")
            
    return warnings


def validate_weight_change(
    timestep: int,
    synapse_id: Union[int, tuple],
    old_weight: float,
    new_weight: float,
    **kwargs
) -> List[str]:
    """Validate weight change data."""
    warnings = []
    
    # Validate timestep
    if not isinstance(timestep, (int, np.integer)) or timestep < 0:
        warnings.append(f"Invalid timestep: {timestep}")
        
    # Validate synapse_id
    if isinstance(synapse_id, tuple):
        if len(synapse_id) != 2:
            warnings.append(f"synapse_id tuple should have 2 elements, got {len(synapse_id)}")
        elif not all(isinstance(x, (int, np.integer)) for x in synapse_id):
            warnings.append("synapse_id tuple should contain integers")
    elif not isinstance(synapse_id, (int, np.integer)):
        warnings.append(f"synapse_id should be int or tuple, got {type(synapse_id)}")
        
    # Validate weights
    for name, weight in [('old_weight', old_weight), ('new_weight', new_weight)]:
        if not isinstance(weight, (int, float, np.number)):
            warnings.append(f"{name} should be numeric, got {type(weight)}")
        elif not np.isfinite(weight):
            warnings.append(f"{name} should be finite, got {weight}")
            
    return warnings


def validate_network_structure(
    neurons: Dict[str, Any],
    connections: Dict[str, Any]
) -> List[str]:
    """Validate network structure data."""
    warnings = []
    
    # Check required neuron fields
    if 'neuron_ids' not in neurons:
        warnings.append("neurons missing required field 'neuron_ids'")
    else:
        n_neurons = len(neurons['neuron_ids'])
        
        # Check other neuron arrays have consistent length
        for key, value in neurons.items():
            if hasattr(value, '__len__') and len(value) != n_neurons:
                warnings.append(f"neurons[{key}] length {len(value)} doesn't match n_neurons {n_neurons}")
                
    # Check connection fields
    required_connection_fields = ['source_ids', 'target_ids']
    for field in required_connection_fields:
        if field not in connections:
            warnings.append(f"connections missing required field '{field}'")
            
    # Check connection arrays have consistent length
    if 'source_ids' in connections and 'target_ids' in connections:
        n_connections = len(connections['source_ids'])
        if len(connections['target_ids']) != n_connections:
            warnings.append("source_ids and target_ids have different lengths")
            
        # Check other connection arrays
        for key, value in connections.items():
            if key not in required_connection_fields and hasattr(value, '__len__'):
                if len(value) != n_connections:
                    warnings.append(f"connections[{key}] length {len(value)} doesn't match n_connections {n_connections}")
                    
    return warnings


class SchemaVersion:
    """Handle schema versioning and migration."""
    
    def __init__(self, version: str = SCHEMA_VERSION):
        self.version = version
        self.major, self.minor, self.patch = map(int, version.split('.'))
        
    def is_compatible(self, other_version: str) -> bool:
        """Check if another version is compatible (same major version)."""
        other = SchemaVersion(other_version)
        return self.major == other.major
        
    def __str__(self):
        return self.version