# keywords: [utils, numpy, jax, conversion, helpers]
"""Utility functions for data export."""

import json
from typing import Any, Dict, List
import numpy as np

try:
    import jax
    import jax.numpy as jnp

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
        result = {k: v for k, v in zip(keys, numpy_arrays)}
        result.update(non_arrays)
        return result
    return non_arrays


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int_)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float_)):
            return float(obj)
        if isinstance(obj, (np.bool_)):
            return bool(obj)
        return super().default(obj)
