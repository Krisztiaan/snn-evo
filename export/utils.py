# keywords: [utils, numpy, jax, conversion, helpers]
"""Utility functions for data export."""

import json
from typing import Any, Dict

import numpy as np

try:
    import jax

    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False


def ensure_numpy(arr: Any) -> np.ndarray:
    """Convert any array-like object to a numpy array."""
    if isinstance(arr, np.ndarray):
        return arr
    if JAX_AVAILABLE and isinstance(arr, jax.Array):
        return np.array(arr)
    if hasattr(arr, "numpy"):  # For TensorFlow, PyTorch
        return arr.numpy()
    return np.array(arr)


def optimize_jax_conversion(data: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Optimize conversion of a dictionary containing JAX arrays."""
    if not JAX_AVAILABLE:
        return {k: ensure_numpy(v) for k, v in data.items()}

    keys, arrays, non_arrays = [], [], {}
    for key, value in data.items():
        if isinstance(value, jax.Array):
            keys.append(key)
            arrays.append(value)
        else:
            non_arrays[key] = ensure_numpy(value)

    if arrays:
        numpy_arrays = jax.device_get(arrays)
        result = dict(zip(keys, numpy_arrays))
        result.update(non_arrays)
        return result
    return non_arrays


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            # Convert to list. Items will be handled by subsequent calls
            # or by the float handling below if they are float('nan').
            return obj.tolist()
        if JAX_AVAILABLE and isinstance(
            obj, jax.Array
        ):  # Ensure JAX arrays are handled if not pre-converted
            return np.array(obj).tolist()
        # Check for numpy scalar types
        if isinstance(obj, np.integer):  # Catches all numpy integer types
            return int(obj)
        if isinstance(
            obj, np.floating
        ):  # Catches all numpy float types (e.g. np.float32, np.float64)
            if np.isnan(obj):
                return None  # Convert np.nan to null
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        # Handle standard Python float('nan') as well
        if isinstance(obj, float) and np.isnan(obj):
            return None  # Convert float('nan') to null
        return super().default(obj)
