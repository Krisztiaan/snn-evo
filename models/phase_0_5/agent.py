# models/phase_0_5/agent.py
# keywords: [snn agent, jax implementation, three-factor learning]
"""Phase 0.5 SNN Agent implementation."""

from functools import partial
from typing import Any, Dict, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, random

from export import DataExporter
from world.simple_grid_0001 import SimpleGridWorld

from .config import NetworkParams, SnnAgentConfig

# Agent-specific state


class AgentState(NamedTuple):
    v: jnp.ndarray
    spike: jnp.ndarray
    refractory: jnp.ndarray
    is_excitatory: jnp.ndarray
    firing_rate: jnp.ndarray
    threshold_adaptation: jnp.ndarray
    syn_current_e: jnp.ndarray
    syn_current_i: jnp.ndarray
    trace_fast: jnp.ndarray
    eligibility_trace: jnp.ndarray
    dopamine: float
    learning_rate_factor: jnp.ndarray
    w: jnp.ndarray
    w_in: jnp.ndarray
    w_out: jnp.ndarray
    connection_mask: jnp.ndarray
    value_estimate: float
    prev_value: float
    input_population: jnp.ndarray


# JIT-compiled helper functions (internal to the agent)


@jit
def _apply_dale_principle(w: jnp.ndarray, is_excitatory: jnp.ndarray) -> jnp.ndarray:
    E_pre = is_excitatory[:, None]
    return jnp.where(E_pre, jnp.abs(w), -jnp.abs(w))


@partial(jit, static_argnames=["params"])
def _encode_gradient(gradient: float, params: NetworkParams, key: random.PRNGKey) -> jnp.ndarray:
    centers = jnp.linspace(0, 1, params.NUM_INPUTS)
    sigma = 0.15
    activations = jnp.exp(-((gradient - centers) ** 2) / (2 * sigma**2))
    noise = random.normal(key, activations.shape) * 0.1
    activations = jnp.maximum(activations + noise, 0.0)
    return activations / (jnp.sum(activations) + 1e-8) * params.NUM_INPUTS * 0.5


@partial(jit, static_argnames=["params"])
def _neuron_step(state: AgentState, params: NetworkParams, key: random.PRNGKey) -> AgentState:
    # Refractory update
    refractory_new = jnp.maximum(0, state.refractory - 1.0)
    # Synaptic current dynamics
    syn_current_e_new = state.syn_current_e * jnp.exp(-1.0 / params.TAU_SYN_E)
    syn_current_i_new = state.syn_current_i * jnp.exp(-1.0 / params.TAU_SYN_I)
    spike_input = state.spike.astype(float)
    E_mask = state.is_excitatory[:, None]
    exc_input = jnp.where(E_mask, state.w, 0.0) @ spike_input
    inh_input = jnp.where(~E_mask, state.w, 0.0) @ spike_input
    syn_current_e_new += jnp.abs(exc_input)
    syn_current_i_new += jnp.abs(inh_input)
    # Total current
    i_ext = jnp.dot(state.w_in.T, state.input_population)
    i_total = (
        i_ext + syn_current_e_new + syn_current_i_new + random.normal(key, state.v.shape) * 1.0
    )
    # Membrane potential update
    dv = (-state.v + params.V_REST + i_total) / params.TAU_V
    v_new = state.v + dv

    # Spike generation with homeostasis
    effective_threshold = params.V_THRESHOLD + state.threshold_adaptation
    spike_new = (v_new >= effective_threshold) & (refractory_new == 0)

    v_new = jnp.where(spike_new, params.V_RESET, v_new)
    refractory_new = jnp.where(spike_new, params.REFRACTORY_TIME, refractory_new)

    # Update traces and firing rate
    trace_fast_new = state.trace_fast * jnp.exp(-1.0 / params.TAU_FAST_TRACE) + spike_new
    alpha = 1.0 / params.HOMEOSTATIC_TAU
    firing_rate_new = state.firing_rate * (1 - alpha) + spike_new * 1000.0 * alpha

    # Update homeostatic threshold
    rate_error = firing_rate_new - params.TARGET_RATE_HZ
    threshold_adaptation_new = state.threshold_adaptation + params.THRESHOLD_ADAPT_RATE * rate_error

    return state._replace(
        v=v_new,
        spike=spike_new,
        refractory=refractory_new,
        syn_current_e=syn_current_e_new,
        syn_current_i=syn_current_i_new,
        trace_fast=trace_fast_new,
        firing_rate=firing_rate_new,
        threshold_adaptation=jnp.clip(threshold_adaptation_new, -10.0, 10.0),
    )


@partial(jit, static_argnames=["params"])
def _learning_step(state: AgentState, reward: float, params: NetworkParams) -> AgentState:
    # 1. RPE-based Dopamine
    td_error = reward + params.REWARD_DISCOUNT * state.value_estimate - state.prev_value
    new_value = state.value_estimate + params.REWARD_PREDICTION_RATE * td_error
    decay = jnp.exp(-1.0 / params.TAU_DOPAMINE)
    dopamine_response = (
        state.dopamine * decay + params.BASELINE_DOPAMINE * (1 - decay) + td_error * 0.5
    )
    new_dopamine = jnp.clip(dopamine_response, 0.0, 2.0)

    # 2. Eligibility Trace
    ltp = state.trace_fast[:, None] * state.spike[None, :].astype(float)
    ltd = state.spike[:, None].astype(float) * state.trace_fast[None, :]
    stdp = jnp.where(
        state.connection_mask, params.STDP_A_PLUS * ltp - params.STDP_A_MINUS * ltd, 0.0
    )
    decay_e = jnp.exp(-1.0 / params.TAU_ELIGIBILITY)
    new_eligibility = state.eligibility_trace * decay_e + stdp

    # 3. Weight Update
    dopamine_modulation = jax.nn.tanh(
        (new_dopamine - params.BASELINE_DOPAMINE) / (params.BASELINE_DOPAMINE + 1e-8) * 2.0
    )
    effective_lr = params.BASE_LEARNING_RATE * state.learning_rate_factor
    dw = effective_lr * dopamine_modulation * new_eligibility - params.WEIGHT_DECAY * state.w
    dw = jnp.where(state.connection_mask, dw, 0.0)

    updated_weights = _apply_dale_principle(state.w + dw, state.is_excitatory)

    return state._replace(
        w=updated_weights,
        eligibility_trace=new_eligibility * 0.9,
        dopamine=new_dopamine,
        value_estimate=new_value,
        prev_value=new_value,
    )


class SnnAgent:
    """The main SNN agent class."""

    def __init__(self, config: SnnAgentConfig):
        self.config = config
        self.params = config.network_params
        self.world = SimpleGridWorld(config.world_config)
        self.state: AgentState = self._initialize_state(random.PRNGKey(config.exp_config.seed))

    def _initialize_state(self, key: random.PRNGKey) -> AgentState:
        keys = random.split(key, 8)
        p = self.params
        n = p.NUM_NEURONS

        is_excitatory = jnp.arange(n) < int(n * p.EXCITATORY_RATIO)

        # Connectivity
        positions = jnp.arange(n)
        dist_matrix = jnp.minimum(
            jnp.abs(positions[:, None] - positions[None, :]),
            n - jnp.abs(positions[:, None] - positions[None, :]),
        )
        local_bias = jnp.exp(-dist_matrix / (n * p.LOCAL_CONNECTIVITY_SCALE))

        prob_matrix = jnp.zeros((n, n))
        prob_matrix = jnp.where(
            jnp.outer(is_excitatory, is_excitatory), p.P_EE * local_bias, prob_matrix
        )
        prob_matrix = jnp.where(jnp.outer(is_excitatory, ~is_excitatory), p.P_EI, prob_matrix)
        prob_matrix = jnp.where(jnp.outer(~is_excitatory, is_excitatory), p.P_IE, prob_matrix)
        prob_matrix = jnp.where(
            jnp.outer(~is_excitatory, ~is_excitatory), p.P_II * local_bias, prob_matrix
        )

        connection_mask = random.uniform(keys[0], (n, n)) < prob_matrix
        connection_mask = connection_mask.at[jnp.diag_indices(n)].set(False)

        # Weights
        w = jnp.zeros((n, n))
        w = jnp.where(
            jnp.outer(is_excitatory, is_excitatory) & connection_mask,
            random.normal(keys[1], (n, n)) * p.W_EE_MEAN * p.W_STD_SCALE + p.W_EE_MEAN,
            w,
        )
        w = jnp.where(
            jnp.outer(is_excitatory, ~is_excitatory) & connection_mask,
            random.normal(keys[2], (n, n)) * p.W_EI_MEAN * p.W_STD_SCALE + p.W_EI_MEAN,
            w,
        )
        w = jnp.where(
            jnp.outer(~is_excitatory, is_excitatory) & connection_mask,
            random.normal(keys[3], (n, n)) * p.W_IE_MEAN * p.W_STD_SCALE + p.W_IE_MEAN,
            w,
        )
        w = jnp.where(
            jnp.outer(~is_excitatory, ~is_excitatory) & connection_mask,
            random.normal(keys[4], (n, n)) * p.W_II_MEAN * p.W_STD_SCALE + p.W_II_MEAN,
            w,
        )
        w = _apply_dale_principle(w, is_excitatory)

        # Input weights
        input_centers = jnp.linspace(0, 1, p.NUM_INPUTS)
        neuron_prefs = jnp.linspace(0, 1, n)
        w_in = jnp.exp(-((input_centers[None, :] - neuron_prefs[:, None]) ** 2) / (2 * 0.15**2))
        w_in = (w_in + random.normal(keys[5], w_in.shape) * 0.1) / (
            jnp.sum(w_in, axis=1, keepdims=True) + 1e-8
        )

        # Output weights
        w_out = random.normal(keys[6], (n, p.NUM_OUTPUTS)) * 0.1
        w_out = jnp.where(is_excitatory[:, None], jnp.abs(w_out), 0.0)

        return AgentState(
            v=jnp.full(n, p.V_REST),
            spike=jnp.zeros(n, dtype=bool),
            refractory=jnp.zeros(n),
            is_excitatory=is_excitatory,
            firing_rate=jnp.full(n, p.TARGET_RATE_HZ),
            threshold_adaptation=jnp.zeros(n),
            syn_current_e=jnp.zeros(n),
            syn_current_i=jnp.zeros(n),
            trace_fast=jnp.zeros(n),
            eligibility_trace=jnp.zeros((n, n)),
            dopamine=p.BASELINE_DOPAMINE,
            learning_rate_factor=jnp.ones((n, n)),
            w=w,
            w_in=w_in.T,
            w_out=w_out,
            connection_mask=connection_mask,
            value_estimate=0.0,
            prev_value=0.0,
            input_population=jnp.zeros(p.NUM_INPUTS),
        )

    def run_episode(
        self, episode_key: random.PRNGKey, episode_num: int, exporter: DataExporter
    ) -> Dict[str, Any]:
        """Run a single episode, returning a summary and handling data export."""
        world_state, obs = self.world.reset(episode_key)
        # Re-initialize agent state for each episode
        self.state = self._initialize_state(episode_key)

        ep = exporter.start_episode(episode_num)
        exporter.log_static_episode_data(
            "world_setup", {"reward_positions": np.array(world_state.reward_positions)}
        )

        for step in range(self.config.world_config.max_timesteps):
            episode_key, step_key, encode_key, neuron_key, decision_key = random.split(
                episode_key, 5
            )

            # Agent logic
            input_pop = _encode_gradient(obs.gradient, self.params, encode_key)
            self.state = self.state._replace(input_population=input_pop)
            self.state = _neuron_step(self.state, self.params, neuron_key)

            # World interaction
            action = int(
                random.categorical(decision_key, jnp.ones(self.params.NUM_OUTPUTS))
            )  # Simplified action for now
            result = self.world.step(world_state, action)
            world_state, obs, reward, done = (
                result.state,
                result.observation,
                result.reward,
                result.done,
            )

            # Learning
            self.state = _learning_step(self.state, reward, self.params)

            exporter.log(
                timestep=step,
                neural_state={"v": self.state.v, "spikes": self.state.spike.astype(jnp.uint8)},
                behavior={
                    "action": action,
                    "pos_x": int(world_state.agent_pos[0]),
                    "pos_y": int(world_state.agent_pos[1]),
                },
                reward=float(reward),
            )

            if done:
                break

        summary = {
            "total_reward": float(world_state.total_reward),
            "rewards_collected": int(jnp.sum(world_state.reward_collected)),
            "steps_taken": world_state.timestep,
        }

        exporter.end_episode(success=jnp.all(world_state.reward_collected), summary=summary)

        return summary

    def run_experiment(self):
        """Run a full experiment with multiple episodes."""
        master_key = random.PRNGKey(self.config.exp_config.seed)

        with DataExporter(
            experiment_name="snn_agent_phase05",
            output_base_dir=self.config.exp_config.export_dir,
            compression="gzip",
            compression_level=1,
        ) as exporter:
            exporter.save_config(self.config)
            exporter.save_network_structure(
                neurons={
                    "neuron_ids": np.arange(self.params.NUM_NEURONS),
                    "is_excitatory": np.array(self.state.is_excitatory),
                },
                connections={
                    "source_ids": np.where(self.state.connection_mask)[0],
                    "target_ids": np.where(self.state.connection_mask)[1],
                },
                initial_weights={"weights": np.array(self.state.w[self.state.connection_mask])},
            )

            all_summaries = []
            for i in range(self.config.exp_config.n_episodes):
                print(f"\n--- Episode {i + 1}/{self.config.exp_config.n_episodes} ---")
                episode_key, master_key = random.split(master_key)

                summary = self.run_episode(episode_key, episode_num=i, exporter=exporter)
                all_summaries.append(summary)

                print("  Episode Summary:")
                print(f"    Total Reward: {summary['total_reward']:.2f}")
                print(f"    Rewards Collected: {summary['rewards_collected']}")
                print(f"    Steps Taken: {summary['steps_taken']}")

        return all_summaries
