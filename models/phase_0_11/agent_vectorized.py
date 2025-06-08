# Vectorized connectivity creation for Phase 0_8
import jax.numpy as jnp
from jax import random
from typing import Tuple


def create_connectivity_vectorized(
    key: random.PRNGKey,
    params,
    num_neurons: int,
    is_excitatory: jnp.ndarray,
    neuron_types: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Create biologically motivated connectivity patterns (fully vectorized).
    Returns: (connection_mask, plastic_mask)
    """
    n = num_neurons
    n_in = params.NUM_INPUT_CHANNELS
    n_total_sources = n_in + n

    # Initialize masks
    w_mask = jnp.zeros((n, n_total_sources), dtype=bool)
    w_plastic = jnp.zeros((n, n_total_sources), dtype=bool)

    # Neuron type masks
    is_sensory = neuron_types == 0
    is_processing = neuron_types == 1
    is_readout = neuron_types == 2

    keys = random.split(key, 4)

    # 1. Input → Sensory connections (already vectorized)
    sensory_indices = jnp.where(is_sensory)[0]
    if len(sensory_indices) > 0:
        input_probs = random.uniform(keys[0], (len(sensory_indices), n_in))
        input_mask = input_probs < params.P_INPUT_SENSORY
        w_mask = w_mask.at[sensory_indices[:, None], jnp.arange(n_in)].set(input_mask)

    # 2. Sensory → Processing connections (vectorized)
    sens_indices = jnp.where(is_sensory)[0]
    proc_indices = jnp.where(is_processing)[0]
    if len(sens_indices) > 0 and len(proc_indices) > 0:
        # Create all possible connections at once
        sens_proc_probs = random.uniform(keys[1], (len(proc_indices), len(sens_indices)))
        sens_proc_mask = sens_proc_probs < params.P_SENSORY_PROCESSING

        # Use meshgrid to create index arrays
        proc_idx_grid, sens_idx_grid = jnp.meshgrid(proc_indices, sens_indices, indexing="ij")

        # Set connections where mask is True
        mask_flat = sens_proc_mask.flatten()
        proc_flat = proc_idx_grid.flatten()
        sens_flat = sens_idx_grid.flatten()

        # Update w_mask for connections
        w_mask = w_mask.at[proc_flat[mask_flat], n_in + sens_flat[mask_flat]].set(True)

    # 3. Processing ↔ Processing connections (fully vectorized)
    if len(proc_indices) > 0:
        n_proc = len(proc_indices)

        # Create distance matrix for local connectivity
        proc_positions = jnp.arange(n_proc)
        dist_matrix = jnp.abs(proc_positions[:, None] - proc_positions[None, :])
        dist_matrix = jnp.minimum(dist_matrix, n_proc - dist_matrix)  # Ring topology
        is_local = dist_matrix < (n_proc * params.LOCAL_RADIUS)

        # Get E/I identity for processing neurons
        proc_e_mask = is_excitatory[proc_indices]

        # Create probability matrix based on connection types
        prob_matrix = jnp.zeros((n_proc, n_proc))

        # E→E connections
        ee_mask = jnp.outer(proc_e_mask, proc_e_mask)
        prob_matrix = jnp.where(ee_mask & is_local, params.P_PROC_PROC_LOCAL, prob_matrix)
        prob_matrix = jnp.where(ee_mask & ~is_local, params.P_PROC_PROC_DIST, prob_matrix)

        # Other connection types
        ei_mask = jnp.outer(proc_e_mask, ~proc_e_mask)
        ie_mask = jnp.outer(~proc_e_mask, proc_e_mask)
        ii_mask = jnp.outer(~proc_e_mask, ~proc_e_mask)

        prob_matrix = jnp.where(ei_mask, params.P_EI, prob_matrix)
        prob_matrix = jnp.where(ie_mask, params.P_IE, prob_matrix)
        prob_matrix = jnp.where(ii_mask, params.P_II, prob_matrix)

        # No self-connections
        prob_matrix = prob_matrix.at[jnp.diag_indices(n_proc)].set(0.0)

        # Sample all connections at once
        conn_probs = random.uniform(keys[2], (n_proc, n_proc))
        proc_conn_mask = conn_probs < prob_matrix

        # Create index arrays for where connections exist
        src_proc_idx, tgt_proc_idx = jnp.where(proc_conn_mask)
        src_global = proc_indices[src_proc_idx]
        tgt_global = proc_indices[tgt_proc_idx]

        # Update masks
        w_mask = w_mask.at[tgt_global, n_in + src_global].set(True)
        if params.LEARN_PROCESSING_RECURRENT:
            w_plastic = w_plastic.at[tgt_global, n_in + src_global].set(True)

    # 4. Processing → Readout connections (vectorized)
    readout_indices = jnp.where(is_readout)[0]
    exc_proc_indices = jnp.where(is_processing & is_excitatory)[0]
    if len(readout_indices) > 0 and len(exc_proc_indices) > 0:
        proc_read_probs = random.uniform(keys[3], (len(readout_indices), len(exc_proc_indices)))
        proc_read_mask = proc_read_probs < params.P_PROCESSING_READOUT

        # Create index arrays
        read_idx_grid, proc_idx_grid = jnp.meshgrid(
            readout_indices, exc_proc_indices, indexing="ij"
        )

        # Set connections
        mask_flat = proc_read_mask.flatten()
        read_flat = read_idx_grid.flatten()
        proc_flat = proc_idx_grid.flatten()

        w_mask = w_mask.at[read_flat[mask_flat], n_in + proc_flat[mask_flat]].set(True)
        if params.LEARN_PROCESSING_READOUT:
            w_plastic = w_plastic.at[read_flat[mask_flat], n_in + proc_flat[mask_flat]].set(True)

    return w_mask, w_plastic
