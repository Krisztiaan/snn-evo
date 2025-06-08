#!/usr/bin/env python3
# models/phase_0_9/test_gradient_reward.py
# keywords: [test gradient reward, phase 0.9, dopamine modulation]
"""Test script to verify gradient-based dopamine modulation."""

from phase_0_9.config import NetworkParams, SnnAgentConfig
from phase_0_9.agent import _three_factor_learning, AgentState
import jax
import jax.numpy as jnp
from jax import random
import matplotlib.pyplot as plt

import sys
sys.path.append('..')


def test_gradient_dopamine():
    """Test that dopamine increases with gradient value."""

    # Initialize config
    config = SnnAgentConfig()
    params = config.network_params

    # Create minimal state for testing
    n_neurons = 10
    n_in = params.NUM_INPUT_CHANNELS

    state = AgentState(
        v=jnp.zeros(n_neurons),
        spike=jnp.zeros(n_neurons, dtype=bool),
        refractory=jnp.zeros(n_neurons),
        syn_current_e=jnp.zeros(n_neurons),
        syn_current_i=jnp.zeros(n_neurons),
        trace_fast=jnp.zeros(n_neurons),
        trace_slow=jnp.zeros(n_neurons),
        firing_rate=jnp.zeros(n_neurons),
        threshold_adapt=jnp.zeros(n_neurons),
        eligibility_trace=jnp.zeros((n_neurons, n_in + n_neurons)),
        dopamine=params.BASELINE_DOPAMINE,
        value_estimate=0.0,
        w=jnp.zeros((n_neurons, n_in + n_neurons)),
        w_mask=jnp.zeros((n_neurons, n_in + n_neurons), dtype=bool),
        w_plastic_mask=jnp.zeros((n_neurons, n_in + n_neurons), dtype=bool),
        is_excitatory=jnp.ones(n_neurons, dtype=bool),
        neuron_types=jnp.zeros(n_neurons, dtype=int),
        motor_trace=jnp.zeros(4),
        input_channels=jnp.zeros(n_in)
    )

    # Test different gradient values
    gradients = jnp.linspace(0, 1, 11)
    dopamine_levels = []

    for grad in gradients:
        # No external reward, just gradient
        new_state = _three_factor_learning(
            state, reward=0.0, gradient=float(grad), params=params)
        dopamine_levels.append(float(new_state.dopamine))

    # Plot results
    plt.figure(figsize=(8, 6))
    plt.plot(gradients, dopamine_levels, 'b-o', linewidth=2, markersize=8)
    plt.axhline(y=params.BASELINE_DOPAMINE, color='r', linestyle='--',
                label=f'Baseline DA = {params.BASELINE_DOPAMINE}')
    plt.xlabel('Gradient Value', fontsize=12)
    plt.ylabel('Dopamine Level', fontsize=12)
    plt.title('Dopamine Response to Gradient (No External Reward)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('gradient_dopamine_response.png', dpi=150)
    plt.close()

    # Test with reward interaction
    plt.figure(figsize=(10, 6))

    # Test with different reward levels
    for reward in [0.0, 0.5, 1.0]:
        dopamine_with_reward = []
        for grad in gradients:
            new_state = _three_factor_learning(
                state, reward=reward, gradient=float(grad), params=params)
            dopamine_with_reward.append(float(new_state.dopamine))
        plt.plot(gradients, dopamine_with_reward, '-o', linewidth=2,
                 label=f'Reward = {reward}')

    plt.axhline(y=params.BASELINE_DOPAMINE,
                color='k', linestyle='--', alpha=0.5)
    plt.xlabel('Gradient Value', fontsize=12)
    plt.ylabel('Dopamine Level', fontsize=12)
    plt.title('Dopamine Response: Gradient + Reward Interaction', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('gradient_reward_interaction.png', dpi=150)
    plt.close()

    print("Test complete! Check the generated plots:")
    print("- gradient_dopamine_response.png")
    print("- gradient_reward_interaction.png")
    print(f"\nGradient reward scale: {params.GRADIENT_REWARD_SCALE}")
    print(f"Baseline dopamine: {params.BASELINE_DOPAMINE}")
    print(
        f"Dopamine range tested: {min(dopamine_levels):.3f} - {max(dopamine_levels):.3f}")


if __name__ == "__main__":
    test_gradient_dopamine()
