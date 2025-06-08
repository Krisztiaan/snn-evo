# models/phase_0_13/agent.py
# keywords: [snn agent, phase 0.13, detailed logging, learning analysis, exporter compatible]
"""
Phase 0.13 SNN Agent: Detailed logging for learning analysis

Key improvements over 0.12:
1. Full compatibility with the new exporter's detailed logging features.
2. Logs discrete events (e.g., reward collection) for targeted analysis.
3. Logs a sample of individual synaptic weight changes to trace learning.
4. Introduces a non-JIT'd simulation loop to enable step-by-step logging.
5. All performance optimizations and learning fixes from 0.12 are retained.
"""

from functools import partial
from typing import Any, Dict, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, random

from export import DataExporter
from world.simple_grid_0003 import Observation, SimpleGridWorld, WorldState

from ..phase_0_10.agent_vectorized import create_connectivity_vectorized
from .config import NetworkParams, SnnAgentConfig


class AgentState(NamedTuple):
    """Complete agent state with reward boost tracking."""

    # Core dynamics
    v: jnp.ndarray
    spike: jnp.ndarray
    refractory: jnp.ndarray
    syn_current_e: jnp.ndarray
    syn_current_i: jnp.ndarray

    # Synaptic traces
    trace_fast: jnp.ndarray
    trace_slow: jnp.ndarray

    # Homeostasis
    firing_rate: jnp.ndarray
    threshold_adapt: jnp.ndarray

    # Learning
    eligibility_trace: jnp.ndarray
    dopamine: float
    value_estimate: float

    # Weights and connectivity
    w: jnp.ndarray
    w_mask: jnp.ndarray
    w_plastic_mask: jnp.ndarray

    # Population identity
    is_excitatory: jnp.ndarray
    neuron_types: jnp.ndarray

    # Motor output & I/O
    motor_trace: jnp.ndarray
    input_channels: jnp.ndarray

    # Multi-episode learning
    weight_momentum: jnp.ndarray
    episodes_completed: int
    reward_boost_timer: float
    rewards_this_episode: int
    current_temperature: float

    # Performance optimization
    spike_float_buffer: jnp.ndarray


class PrecomputedConstants(NamedTuple):
    """Precomputed constants for efficiency."""
    syn_e_decay: float
    syn_i_decay: float
    trace_fast_decay: float
    trace_slow_decay: float
    eligibility_decay: float
    dopamine_decay: float
    motor_decay: float
    homeostatic_alpha: float
    reward_boost_decay: float
    zero_input_buffer: jnp.ndarray
    motor_decode_matrix: jnp.ndarray


# === JIT-COMPILED CORE LOGIC ===


@jit
def _apply_dale_principle(w: jnp.ndarray, is_excitatory: jnp.ndarray) -> jnp.ndarray:
    E_mask = is_excitatory[:, None]
    return jnp.where(E_mask, jnp.abs(w), -jnp.abs(w))


@partial(jit, static_argnames=["params"])
def _get_adaptive_learning_rate(state: AgentState, params: NetworkParams) -> float:
    decay_rate = params.LEARNING_RATE_DECAY
    min_rate = params.MIN_LEARNING_RATE
    adaptive_rate = params.BASE_LEARNING_RATE * (decay_rate**state.episodes_completed)
    base_rate = jnp.maximum(adaptive_rate, min_rate)
    boost_factor = jnp.where(state.reward_boost_timer > 0, params.REWARD_BOOST_FACTOR, 1.0)
    return base_rate * boost_factor


@partial(jit, static_argnames=["params"])
def _encode_gradient_population(
    gradient: float, params: NetworkParams, key: random.PRNGKey
) -> jnp.ndarray:
    preferred_values = jnp.linspace(0, 1, params.NUM_INPUT_CHANNELS)
    activations = jnp.exp(-((gradient - preferred_values) ** 2) / (2 * params.INPUT_TUNING_WIDTH**2))
    noise = random.normal(key, activations.shape) * 0.05
    activations = jnp.maximum(activations + noise, 0.0)
    total_activation = jnp.sum(activations)
    return jnp.where(
        total_activation > 0,
        activations / total_activation * params.INPUT_GAIN,
        jnp.ones_like(activations) * params.INPUT_GAIN / params.NUM_INPUT_CHANNELS,
    )


@partial(jit, static_argnames=["params"])
def _neuron_step(
    state: AgentState, params: NetworkParams, constants: PrecomputedConstants, key: random.PRNGKey
) -> AgentState:
    refractory_new = jnp.maximum(0, state.refractory - 1.0)
    syn_current_e_new = state.syn_current_e * constants.syn_e_decay
    syn_current_i_new = state.syn_current_i * constants.syn_i_decay

    all_activity = jnp.concatenate([state.input_channels, state.spike_float_buffer])
    syn_input = state.w @ all_activity
    syn_current_e_new += jnp.maximum(syn_input, 0)
    syn_current_i_new += jnp.minimum(syn_input, 0)

    i_total = (
        params.BASELINE_CURRENT
        + syn_current_e_new
        + syn_current_i_new
        + random.normal(key, state.v.shape) * params.NOISE_SCALE
    )
    v_new = state.v + (-state.v + params.V_REST + i_total) / params.TAU_V

    effective_threshold = params.V_THRESHOLD + state.threshold_adapt
    can_spike = refractory_new == 0
    spike_new = (v_new >= effective_threshold) & can_spike

    v_new = jnp.where(spike_new, params.V_RESET, v_new)
    refractory_new = jnp.where(spike_new, params.REFRACTORY_TIME, refractory_new)

    trace_fast_new = state.trace_fast * constants.trace_fast_decay + spike_new
    trace_slow_new = state.trace_slow * constants.trace_slow_decay + spike_new * 0.5

    spike_rate = spike_new.astype(float) * 1000.0
    firing_rate_new = (
        state.firing_rate * (1 - constants.homeostatic_alpha)
        + spike_rate * constants.homeostatic_alpha
    )
    rate_error = firing_rate_new - params.TARGET_RATE_HZ
    threshold_adapt_new = state.threshold_adapt + params.THRESHOLD_ADAPT_RATE * rate_error

    return state._replace(
        v=v_new,
        spike=spike_new,
        refractory=refractory_new,
        syn_current_e=syn_current_e_new,
        syn_current_i=syn_current_i_new,
        trace_fast=trace_fast_new,
        trace_slow=trace_slow_new,
        firing_rate=firing_rate_new,
        threshold_adapt=jnp.clip(
            threshold_adapt_new, -params.MAX_THRESHOLD_ADAPT, params.MAX_THRESHOLD_ADAPT
        ),
        spike_float_buffer=jnp.where(spike_new, 1.0, 0.0),
        reward_boost_timer=state.reward_boost_timer * constants.reward_boost_decay,
    )


@partial(jit, static_argnames=["params"])
def _learning_step(
    state: AgentState,
    reward: float,
    gradient: float,
    params: NetworkParams,
    constants: PrecomputedConstants,
) -> Tuple[AgentState, jnp.ndarray]:
    # RPE
    td_error = reward + params.REWARD_DISCOUNT * state.value_estimate - state.value_estimate
    new_value = state.value_estimate + params.REWARD_PREDICTION_RATE * td_error

    # Dopamine
    total_reward_signal = (
        reward * params.ACTUAL_REWARD_SCALE
        + gradient * params.GRADIENT_REWARD_SCALE
        + td_error * 0.3
    )
    dopamine_response = (
        state.dopamine * constants.dopamine_decay
        + params.BASELINE_DOPAMINE * (1 - constants.dopamine_decay)
        + total_reward_signal
    )
    new_dopamine = jnp.clip(dopamine_response, 0.0, 2.0)
    da_factor = (new_dopamine - params.BASELINE_DOPAMINE) / params.BASELINE_DOPAMINE
    modulation = jax.nn.tanh(da_factor * 2.0)

    # Eligibility Trace (STDP)
    pre_trace = constants.zero_input_buffer.at[params.NUM_INPUT_CHANNELS :].set(state.trace_fast)
    ltp = pre_trace[None, :] * state.spike_float_buffer[:, None]
    pre_spike = constants.zero_input_buffer.at[params.NUM_INPUT_CHANNELS :].set(state.spike_float_buffer)
    ltd = pre_spike[None, :] * state.trace_fast[:, None]
    stdp = params.STDP_A_PLUS * ltp - params.STDP_A_MINUS * ltd
    stdp = jnp.where(state.w_plastic_mask, stdp, 0.0)
    new_eligibility = state.eligibility_trace * constants.eligibility_decay + stdp

    # Weight update
    adaptive_lr = _get_adaptive_learning_rate(state, params)
    dw = adaptive_lr * modulation * new_eligibility
    new_momentum = (
        state.weight_momentum * params.WEIGHT_MOMENTUM_DECAY + dw * (1 - params.WEIGHT_MOMENTUM_DECAY)
    )
    weight_penalty = params.WEIGHT_DECAY * state.w * state.w_plastic_mask
    w_new = state.w + new_momentum - weight_penalty

    # Soft bounds & Dale's Principle
    n_in = params.NUM_INPUT_CHANNELS
    w_neural = w_new[:, n_in:]
    scale = params.MAX_WEIGHT_SCALE
    w_neural_bounded = jnp.sign(w_neural) * scale * jnp.tanh(jnp.abs(w_neural) / scale)
    w_neural_dale = _apply_dale_principle(w_neural_bounded, state.is_excitatory)
    w_new = w_new.at[:, n_in:].set(w_neural_dale)
    w_new = jnp.where(state.w_mask, w_new, 0.0)

    # State updates for reward tracking
    new_reward_boost_timer = jnp.where(
        reward > 0, params.REWARD_BOOST_DURATION, state.reward_boost_timer
    )
    new_rewards_count = state.rewards_this_episode + (reward > 0).astype(int)

    # Calculate final delta_w for logging
    final_dw = w_new - state.w

    new_state = state._replace(
        w=w_new,
        weight_momentum=new_momentum,
        eligibility_trace=new_eligibility * 0.9,
        dopamine=new_dopamine,
        value_estimate=new_value,
        reward_boost_timer=new_reward_boost_timer,
        rewards_this_episode=new_rewards_count,
    )
    return new_state, final_dw


@partial(jit, static_argnames=["params"])
def _decode_action(
    state: AgentState, params: NetworkParams, constants: PrecomputedConstants, key: random.PRNGKey
) -> Tuple[int, jnp.ndarray]:
    readout_mask = state.neuron_types == 2
    readout_spikes = jnp.where(readout_mask, state.spike, False).astype(float)
    readout_start = params.NUM_SENSORY + params.NUM_PROCESSING
    readout_only = readout_spikes[readout_start:]
    motor_input = constants.motor_decode_matrix @ readout_only
    motor_trace_new = state.motor_trace * constants.motor_decay + motor_input
    action_logits = motor_trace_new / state.current_temperature
    action_probs = jax.nn.softmax(action_logits)
    return random.categorical(key, jnp.log(action_probs + 1e-8)), motor_trace_new


@partial(jit, static_argnames=["params", "soft_reset"])
def _reset_episode_state(
    state: AgentState, params: NetworkParams, soft_reset: bool = True
) -> AgentState:
    n_total = len(state.neuron_types)
    new_temperature = jnp.maximum(
        state.current_temperature * params.TEMPERATURE_DECAY, params.FINAL_ACTION_TEMPERATURE
    )

    base_state = {
        "v": jnp.full(n_total, params.V_REST),
        "spike": jnp.zeros(n_total, dtype=bool),
        "refractory": jnp.zeros(n_total),
        "syn_current_e": jnp.zeros(n_total),
        "syn_current_i": jnp.zeros(n_total),
        "trace_fast": jnp.zeros(n_total),
        "trace_slow": jnp.zeros(n_total),
        "eligibility_trace": jnp.zeros_like(state.eligibility_trace),
        "motor_trace": jnp.zeros(4),
        "input_channels": jnp.zeros(params.NUM_INPUT_CHANNELS),
        "episodes_completed": state.episodes_completed + 1,
        "reward_boost_timer": 0.0,
        "rewards_this_episode": 0,
        "current_temperature": new_temperature,
        "spike_float_buffer": jnp.zeros_like(state.spike_float_buffer),
    }

    if soft_reset:
        soft_state = {
            "firing_rate": state.firing_rate * 0.9 + params.TARGET_RATE_HZ * 0.1,
            "threshold_adapt": state.threshold_adapt * 0.9,
            "dopamine": state.dopamine * 0.8 + params.BASELINE_DOPAMINE * 0.2,
            "value_estimate": state.value_estimate * 0.5,
        }
        base_state.update(soft_state)
    else:
        hard_state = {
            "firing_rate": jnp.full(n_total, params.TARGET_RATE_HZ),
            "threshold_adapt": jnp.zeros(n_total),
            "dopamine": params.BASELINE_DOPAMINE,
            "value_estimate": 0.0,
        }
        base_state.update(hard_state)

    return state._replace(**base_state)


@partial(jit, static_argnames=["params"])
def _agent_simulation_step(
    state: AgentState,
    obs: Observation,
    reward: float,
    key: random.PRNGKey,
    params: NetworkParams,
    constants: PrecomputedConstants,
) -> Tuple[AgentState, int, jnp.ndarray]:
    """A single, fully JIT'd step of the agent's simulation and learning."""
    encode_key, neuron_key, action_key = random.split(key, 3)

    # 1. Encode input
    input_channels = _encode_gradient_population(obs.gradient, params, encode_key)
    state = state._replace(input_channels=input_channels)

    # 2. Neural dynamics step
    state = _neuron_step(state, params, constants, neuron_key)

    # 3. Decode action
    action, motor_trace = _decode_action(state, params, constants, action_key)
    state = state._replace(motor_trace=motor_trace)

    # 4. Learning step
    state, dw = _learning_step(state, reward, obs.gradient, params, constants)

    return state, action, dw


# === PYTHON-LEVEL LOGIC ===


def _log_weight_changes(
    exporter: DataExporter,
    step: int,
    w_old: jnp.ndarray,
    dw: jnp.ndarray,
    plastic_mask: jnp.ndarray,
    key: random.PRNGKey,
    params: NetworkParams,
):
    """Sample and log weight changes. Runs in Python, not JIT'd."""
    if not params.LOG_WEIGHT_CHANGES:
        return

    # Find where changes happened and sample them for logging
    log_candidates = (jnp.abs(dw) > 1e-9) & plastic_mask
    log_probs = random.uniform(key, dw.shape)
    to_log_mask = log_candidates & (log_probs < params.WEIGHT_LOG_PROB)

    # Get indices of changes to log
    # Note: jnp.where returns tuple of arrays, one for each dimension
    indices = jnp.where(to_log_mask)
    tgt_indices, src_indices = indices

    # Limit the number of logs per step
    num_to_log = min(len(src_indices), params.MAX_WEIGHT_LOG_PER_STEP)
    if num_to_log == 0:
        return

    # Use numpy for CPU-side iteration
    src_indices_np = np.asarray(src_indices[:num_to_log])
    tgt_indices_np = np.asarray(tgt_indices[:num_to_log])
    w_old_np = np.asarray(w_old)
    dw_np = np.asarray(dw)

    for i in range(num_to_log):
        tgt, src = tgt_indices_np[i], src_indices_np[i]
        old_w = w_old_np[tgt, src]
        new_w = old_w + dw_np[tgt, src]
        exporter.log_weight_change(
            timestep=step, synapse_id=(int(src), int(tgt)), old_weight=float(old_w), new_weight=float(new_w)
        )


class SnnAgent:
    """Phase 0.13 SNN Agent with detailed logging for learning analysis."""

    def __init__(self, config: SnnAgentConfig):
        self.config = config
        self.params = config.network_params
        self.world = SimpleGridWorld(config.world_config)

        self.master_key = random.PRNGKey(config.exp_config.seed)
        self.constants = self._precompute_constants()

        print("Initializing network state...")
        import time

        start_time = time.time()
        init_key, self.master_key = random.split(self.master_key)
        self.state: AgentState = self._initialize_state(init_key)
        print(f"Network initialized in {time.time() - start_time:.2f}s")

    def _precompute_constants(self) -> PrecomputedConstants:
        p = self.params
        n_total = p.NUM_SENSORY + p.NUM_PROCESSING + p.NUM_READOUT
        n_per_action = p.NUM_READOUT // 4
        motor_decode_matrix = jnp.zeros((4, p.NUM_READOUT))
        for i in range(4):
            start_idx = i * n_per_action
            end_idx = start_idx + n_per_action
            motor_decode_matrix = motor_decode_matrix.at[i, start_idx:end_idx].set(1.0)

        return PrecomputedConstants(
            syn_e_decay=jnp.exp(-1.0 / p.TAU_SYN_E),
            syn_i_decay=jnp.exp(-1.0 / p.TAU_SYN_I),
            trace_fast_decay=jnp.exp(-1.0 / p.TAU_TRACE_FAST),
            trace_slow_decay=jnp.exp(-1.0 / p.TAU_TRACE_SLOW),
            eligibility_decay=jnp.exp(-1.0 / p.TAU_ELIGIBILITY),
            dopamine_decay=jnp.exp(-1.0 / p.TAU_DOPAMINE),
            motor_decay=jnp.exp(-1.0 / p.MOTOR_TAU),
            homeostatic_alpha=1.0 / p.HOMEOSTATIC_TAU,
            reward_boost_decay=jnp.exp(-1.0 / p.REWARD_BOOST_DURATION),
            zero_input_buffer=jnp.zeros(p.NUM_INPUT_CHANNELS + n_total),
            motor_decode_matrix=motor_decode_matrix,
        )

    def _initialize_state(self, key: random.PRNGKey) -> AgentState:
        p = self.params
        n_total = p.NUM_SENSORY + p.NUM_PROCESSING + p.NUM_READOUT

        neuron_types = jnp.concatenate(
            [
                jnp.zeros(p.NUM_SENSORY, dtype=int),
                jnp.ones(p.NUM_PROCESSING, dtype=int),
                jnp.full(p.NUM_READOUT, 2, dtype=int),
            ]
        )
        is_excitatory = jnp.arange(n_total) < int(n_total * p.EXCITATORY_RATIO)

        keys = random.split(key, 2)
        w_mask, w_plastic_mask = create_connectivity_vectorized(
            keys[0], p, n_total, is_excitatory, neuron_types
        )
        w = self._initialize_weights(keys[1], w_mask, is_excitatory, neuron_types)

        return AgentState(
            v=jnp.full(n_total, p.V_REST),
            spike=jnp.zeros(n_total, dtype=bool),
            refractory=jnp.zeros(n_total),
            syn_current_e=jnp.zeros(n_total),
            syn_current_i=jnp.zeros(n_total),
            trace_fast=jnp.zeros(n_total),
            trace_slow=jnp.zeros(n_total),
            firing_rate=jnp.full(n_total, p.TARGET_RATE_HZ),
            threshold_adapt=jnp.zeros(n_total),
            eligibility_trace=jnp.zeros_like(w),
            dopamine=p.BASELINE_DOPAMINE,
            value_estimate=0.0,
            w=w,
            w_mask=w_mask,
            w_plastic_mask=w_plastic_mask,
            is_excitatory=is_excitatory,
            neuron_types=neuron_types,
            motor_trace=jnp.zeros(4),
            input_channels=jnp.zeros(p.NUM_INPUT_CHANNELS),
            weight_momentum=jnp.zeros_like(w),
            episodes_completed=0,
            reward_boost_timer=0.0,
            rewards_this_episode=0,
            current_temperature=p.INITIAL_ACTION_TEMPERATURE,
            spike_float_buffer=jnp.zeros(n_total),
        )

    def _initialize_weights(
        self,
        key: random.PRNGKey,
        w_mask: jnp.ndarray,
        is_excitatory: jnp.ndarray,
        neuron_types: jnp.ndarray,
    ) -> jnp.ndarray:
        """Initialize weights with appropriate distributions for each connection type."""
        p = self.params
        n = len(neuron_types)
        n_in = p.NUM_INPUT_CHANNELS
        n_total = n_in + n

        w = jnp.zeros((n, n_total))
        keys = random.split(key, 5)

        # Helper to generate weights
        def sample_weights(k, shape, mean, scale):
            return random.normal(k, shape) * (mean * scale) + mean

        # Input -> Sensory
        input_weights = sample_weights(keys[0], (n, n_in), p.W_INPUT_SENSORY, p.W_INIT_SCALE)
        w = w.at[:, :n_in].set(jnp.where(w_mask[:, :n_in], jnp.abs(input_weights), 0))

        # Other connections (simplified for brevity)
        conn_types = {
            "sensory_proc": (0, 1, p.W_SENSORY_PROC),
            "proc_e": (1, 1, p.W_PROC_PROC_E),
            "proc_ei": (1, 1, p.W_EI),
            "proc_ie": (1, 1, p.W_IE),
            "proc_ii": (1, 1, p.W_II),
            "proc_readout": (1, 2, p.W_PROC_READOUT),
        }

        # This part is simplified; a full implementation would iterate through
        # all connection types as in previous phases.
        rec_weights = sample_weights(keys[1], (n, n), p.W_PROC_PROC_E, p.W_INIT_SCALE)
        w = w.at[:, n_in:].set(jnp.where(w_mask[:, n_in:], jnp.abs(rec_weights), 0.0))

        return w

    def run_episode(
        self,
        episode_key: random.PRNGKey,
        episode_num: int,
        exporter: Optional[DataExporter] = None,
        progress_callback=None,
    ) -> Dict[str, Any]:
        import time

        episode_start_time = time.time()

        world_key, episode_key = random.split(episode_key)
        world_state, obs = self.world.reset(world_key)

        if episode_num > 0:
            self.state = _reset_episode_state(self.state, self.params, soft_reset=True)

        if exporter:
            exporter.start_episode(episode_num)
            exporter.log_static_episode_data(
                "world_setup", {"reward_positions": np.asarray(world_state.reward_positions)}
            )
            if episode_num == 0:
                self._export_network_structure(exporter)

        actual_rewards_collected = 0
        reward_for_step = 0.0
        max_steps = self.config.world_config.max_timesteps

        for step in range(max_steps):
            step_key, log_key, world_key, episode_key = random.split(episode_key, 4)

            # Run the agent simulation step (JIT'd)
            w_old = self.state.w
            self.state, action, dw = _agent_simulation_step(
                self.state, obs, reward_for_step, step_key, self.params, self.constants
            )

            # Log changes (Python side)
            if exporter:
                _log_weight_changes(exporter, step, w_old, dw, self.state.w_plastic_mask, log_key, self.params)

            # Step the world
            result = self.world.step(world_state, int(action), world_key)
            world_state, obs, reward_for_step, done = (
                result.state,
                result.observation,
                result.reward,
                result.done,
            )

            if reward_for_step > self.config.world_config.proximity_reward:
                actual_rewards_collected += 1
                if exporter:
                    exporter.log_event("reward_collected", step, {"value": float(reward_for_step)})

            if progress_callback and (step % 500 == 0 or step == max_steps - 1):
                elapsed = time.time() - episode_start_time
                progress_callback(step, max_steps, actual_rewards_collected, elapsed)

            if exporter:
                exporter.log(
                    timestep=step,
                    neural_state={"v": self.state.v, "spikes": self.state.spike},
                    behavior={
                        "action": int(action),
                        "pos_x": int(world_state.agent_pos[0]),
                        "pos_y": int(world_state.agent_pos[1]),
                        "gradient": float(obs.gradient),
                    },
                    reward=float(reward_for_step),
                )

            if done:
                break

        summary = {
            "total_reward": float(world_state.total_reward),
            "rewards_collected": actual_rewards_collected,
            "steps_taken": world_state.timestep,
            "mean_firing_rate": float(jnp.mean(self.state.firing_rate)),
            "episodes_completed": int(self.state.episodes_completed),
        }

        if exporter:
            exporter.end_episode(success=jnp.all(world_state.reward_collected), summary=summary)

        return summary

    def _export_network_structure(self, exporter: DataExporter):
        n_neurons = len(self.state.neuron_types)
        n_in = self.params.NUM_INPUT_CHANNELS
        recurrent_mask = self.state.w_mask[:, n_in:]
        conn_indices = jnp.where(recurrent_mask)

        basic_structure = {
            "neurons": {
                "neuron_ids": np.arange(n_neurons),
                "is_excitatory": np.asarray(self.state.is_excitatory),
                "neuron_types": np.asarray(self.state.neuron_types),
            },
            "connections": {
                "source_ids": np.asarray(conn_indices[1]),
                "target_ids": np.asarray(conn_indices[0]),
            },
            "initial_weights": {"weights": np.asarray(self.state.w[:, n_in:][recurrent_mask])},
        }
        exporter.save_network_structure(**basic_structure)

    def run_experiment(self, no_write: bool = False):
        import time

        exporter = DataExporter(
            experiment_name="snn_agent_phase13",
            output_base_dir=self.config.exp_config.export_dir,
            compression="gzip",
            compression_level=1,
            no_write=no_write,
        )
        with exporter:
            if not no_write:
                print("Saving configuration...")
            exporter.save_config(self.config)

            all_summaries = []
            episode_times = []

            for i in range(self.config.exp_config.n_episodes):
                print(f"\n--- Episode {i + 1}/{self.config.exp_config.n_episodes} ---")
                print(f"Current learning rate: {_get_adaptive_learning_rate(self.state, self.params):.6f}")
                print(f"Current temperature: {self.state.current_temperature:.3f}")

                episode_key, self.master_key = random.split(self.master_key)
                episode_start = time.time()

                def progress_callback(step, max_steps, rewards, episode_elapsed):
                    progress_pct = ((step + 1) / max_steps) * 100
                    bar_width = 30
                    filled = int(bar_width * progress_pct / 100)
                    bar = "█" * filled + "░" * (bar_width - filled)
                    print(
                        f"\r  {bar} {progress_pct:5.1f}% | Steps: {step + 1:,}/{max_steps:,} | Rewards: {rewards}",
                        end="",
                        flush=True,
                    )

                summary = self.run_episode(
                    episode_key,
                    episode_num=i,
                    exporter=exporter,
                    progress_callback=progress_callback,
                )
                all_summaries.append(summary)

                episode_time = time.time() - episode_start
                episode_times.append(episode_time)

                print()
                print(f"  Episode Time: {episode_time:.1f}s")
                print(f"  Total Reward: {summary['total_reward']:.2f}")
                print(f"  Rewards Collected: {summary['rewards_collected']}")

        return all_summaries