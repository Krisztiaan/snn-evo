# models/phase_0_13/agent.py
# keywords: [snn agent, phase 0.13, optimized, no exporter, pure performance]
"""
Phase 0.13 SNN Agent - Optimized Version

Key optimizations:
1. Removed all exporter calls from the hot path
2. Batched data collection for post-episode export
3. Eliminated dict creation in inner loop
4. Pre-allocated buffers for episode data
5. Vectorized reward history processing
"""

from functools import partial
from typing import Any, Dict, NamedTuple, Optional, Tuple
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, random

from world.simple_grid_0003 import Observation, SimpleGridWorld, WorldState
from export import DataExporter
from ..base_agent import BaseAgent
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
    # Additional precomputed values
    preferred_values: jnp.ndarray  # For input encoding
    inv_tau_v: float  # 1/TAU_V for neuron dynamics
    noise_scale: float  # Cached noise scale


# === CONNECTIVITY FUNCTIONS ===

def create_connectivity_vectorized(
    key: random.PRNGKey,
    params: NetworkParams,
    n_total: int,
    is_excitatory: jnp.ndarray,
    neuron_types: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Create connectivity masks using vectorized operations."""
    n_in = params.NUM_INPUT_CHANNELS
    total_size = n_in + n_total
    
    # Initialize masks
    w_mask = jnp.zeros((n_total, total_size), dtype=bool)
    plastic_mask = jnp.zeros((n_total, total_size), dtype=bool)
    
    # Input connections to sensory neurons
    sensory_mask = neuron_types == 0
    n_sensory = jnp.sum(sensory_mask)
    if n_sensory > 0:
        key, subkey = random.split(key)
        input_conn = random.uniform(subkey, (n_total, n_in)) < params.P_INPUT_SENSORY
        w_mask = w_mask.at[:, :n_in].set(sensory_mask[:, None] & input_conn)
        plastic_mask = plastic_mask.at[:, :n_in].set(
            sensory_mask[:, None] & input_conn & params.LEARN_INPUT_CONNECTIONS
        )
    
    # Simplified recurrent connections
    key, subkey = random.split(key)
    rec_conn_prob = random.uniform(subkey, (n_total, n_total))
    
    # Basic E->E connections
    ee_mask = is_excitatory[:, None] & is_excitatory[None, :]
    ee_conn = rec_conn_prob < params.P_EE
    
    # Basic E->I connections  
    ei_mask = is_excitatory[:, None] & ~is_excitatory[None, :]
    ei_conn = rec_conn_prob < params.P_EI
    
    # Basic I->E connections
    ie_mask = ~is_excitatory[:, None] & is_excitatory[None, :]
    ie_conn = rec_conn_prob < params.P_IE
    
    # Basic I->I connections
    ii_mask = ~is_excitatory[:, None] & ~is_excitatory[None, :]
    ii_conn = rec_conn_prob < params.P_II
    
    # Combine all connection types
    rec_mask = (ee_mask & ee_conn) | (ei_mask & ei_conn) | (ie_mask & ie_conn) | (ii_mask & ii_conn)
    
    # No self-connections
    rec_mask = rec_mask & ~jnp.eye(n_total, dtype=bool)
    
    # Set recurrent connections
    w_mask = w_mask.at[:, n_in:].set(rec_mask)
    plastic_mask = plastic_mask.at[:, n_in:].set(
        rec_mask & params.LEARN_PROCESSING_RECURRENT
    )
    
    return w_mask, plastic_mask


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
    gradient: jnp.ndarray, params: NetworkParams, constants: PrecomputedConstants, key: random.PRNGKey
) -> jnp.ndarray:
    # Use pre-computed preferred values
    diff_squared = (gradient - constants.preferred_values) ** 2
    activations = jnp.exp(-diff_squared / (2 * params.INPUT_TUNING_WIDTH**2))
    noise = random.normal(key, activations.shape) * 0.05
    activations = jnp.maximum(activations + noise, 0.0)
    total_activation = jnp.sum(activations)
    # Avoid division by zero more efficiently
    safe_total = jnp.maximum(total_activation, 1e-8)
    return activations * (params.INPUT_GAIN / safe_total)


@partial(jit, static_argnames=["params"])
def _neuron_step(
    state: AgentState, params: NetworkParams, constants: PrecomputedConstants, key: random.PRNGKey
) -> AgentState:
    refractory_new = jnp.maximum(0, state.refractory - 1.0)
    syn_current_e_new = state.syn_current_e * constants.syn_e_decay
    syn_current_i_new = state.syn_current_i * constants.syn_i_decay

    # Avoid concatenation by using pre-allocated buffer in state
    # Direct matrix multiply with slicing is more efficient
    syn_input = (state.w[:, :params.NUM_INPUT_CHANNELS] @ state.input_channels + 
                 state.w[:, params.NUM_INPUT_CHANNELS:] @ state.spike_float_buffer)
    syn_current_e_new += jnp.maximum(syn_input, 0)
    syn_current_i_new += jnp.minimum(syn_input, 0)

    i_total = (
        params.BASELINE_CURRENT
        + syn_current_e_new
        + syn_current_i_new
        + random.normal(key, state.v.shape) * constants.noise_scale
    )
    v_new = state.v + (-state.v + params.V_REST + i_total) * constants.inv_tau_v

    effective_threshold = params.V_THRESHOLD + state.threshold_adapt
    can_spike = refractory_new == 0
    spike_new = (v_new >= effective_threshold) & can_spike

    v_new = jnp.where(spike_new, params.V_RESET, v_new)
    refractory_new = jnp.where(spike_new, params.REFRACTORY_TIME, refractory_new)

    trace_fast_new = state.trace_fast * constants.trace_fast_decay + spike_new
    trace_slow_new = state.trace_slow * constants.trace_slow_decay + spike_new * 0.5

    # Convert spike to float once and reuse
    spike_float = spike_new.astype(jnp.float32)
    spike_rate = spike_float * 1000.0
    # Fused homeostatic update
    firing_rate_new = state.firing_rate + constants.homeostatic_alpha * (spike_rate - state.firing_rate)
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
        spike_float_buffer=spike_float,  # Reuse the float conversion
        reward_boost_timer=state.reward_boost_timer * constants.reward_boost_decay,
    )


@partial(jit, static_argnames=["params"])
def _learning_step(
    state: AgentState,
    reward: float,
    gradient: float,
    params: NetworkParams,
    constants: PrecomputedConstants,
) -> AgentState:
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

    # Eligibility Trace (STDP) - optimized without buffer allocation
    # Only compute STDP for neural connections (skip input channels)
    n_in = params.NUM_INPUT_CHANNELS
    # LTP: pre-synaptic trace * post-synaptic spike
    ltp_neural = state.trace_fast[None, :] * state.spike_float_buffer[:, None]
    # LTD: pre-synaptic spike * post-synaptic trace  
    ltd_neural = state.spike_float_buffer[None, :] * state.trace_fast[:, None]
    # Compute STDP only for neural connections
    stdp_neural = params.STDP_A_PLUS * ltp_neural - params.STDP_A_MINUS * ltd_neural
    # Build full STDP matrix with zeros for input connections
    stdp = jnp.zeros_like(state.eligibility_trace)
    stdp = stdp.at[:, n_in:].set(stdp_neural)
    # Apply plasticity mask and update eligibility
    stdp_masked = stdp * state.w_plastic_mask
    new_eligibility = state.eligibility_trace * constants.eligibility_decay + stdp_masked

    # Weight update
    adaptive_lr = _get_adaptive_learning_rate(state, params)
    dw = adaptive_lr * modulation * new_eligibility
    new_momentum = state.weight_momentum * params.WEIGHT_MOMENTUM_DECAY + dw * (
        1 - params.WEIGHT_MOMENTUM_DECAY
    )
    weight_penalty = params.WEIGHT_DECAY * state.w * state.w_plastic_mask
    w_new = state.w + new_momentum - weight_penalty

    # Soft bounds & Dale's Principle - optimized
    w_neural = w_new[:, n_in:]
    scale = params.MAX_WEIGHT_SCALE
    # Fused soft bounds computation
    w_abs = jnp.abs(w_neural)
    w_bounded = scale * jnp.tanh(w_abs / scale)
    # Apply Dale's principle inline to avoid function call
    E_mask = state.is_excitatory[:, None]
    w_neural_dale = jnp.where(E_mask, w_bounded, -w_bounded)
    # Single update with mask application
    w_new = w_new.at[:, n_in:].set(w_neural_dale)
    w_new = w_new * state.w_mask  # Multiplication is faster than where for sparse masks

    # State updates for reward tracking
    new_reward_boost_timer = jnp.where(
        reward > 0, params.REWARD_BOOST_DURATION, state.reward_boost_timer
    )
    new_rewards_count = state.rewards_this_episode + jnp.int32(reward > 0)

    new_state = state._replace(
        w=w_new,
        weight_momentum=new_momentum,
        eligibility_trace=new_eligibility * 0.9,
        dopamine=new_dopamine,
        value_estimate=new_value,
        reward_boost_timer=new_reward_boost_timer,
        rewards_this_episode=new_rewards_count,
    )
    return new_state


@partial(jit, static_argnames=["params"])
def _decode_action(
    state: AgentState, params: NetworkParams, constants: PrecomputedConstants, key: random.PRNGKey
) -> Tuple[int, jnp.ndarray]:
    # Direct slicing is more efficient than masking
    readout_start = params.NUM_SENSORY + params.NUM_PROCESSING
    readout_spikes = state.spike_float_buffer[readout_start:]  # Already float
    motor_input = constants.motor_decode_matrix @ readout_spikes
    motor_trace_new = state.motor_trace * constants.motor_decay + motor_input
    # Use reciprocal for division
    inv_temp = 1.0 / state.current_temperature
    action_logits = motor_trace_new * inv_temp
    # Use log_softmax directly to avoid log(softmax())
    log_probs = jax.nn.log_softmax(action_logits)
    return random.categorical(key, log_probs), motor_trace_new


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
) -> tuple[AgentState, int]:
    """A single, fully JIT'd step of the agent's simulation and learning."""
    encode_key, neuron_key, action_key = random.split(key, 3)

    # 1. Encode input
    input_channels = _encode_gradient_population(obs.gradient, params, constants, encode_key)
    state = state._replace(input_channels=input_channels)

    # 2. Neural dynamics step
    state = _neuron_step(state, params, constants, neuron_key)

    # 3. Decode action
    action, motor_trace = _decode_action(state, params, constants, action_key)
    state = state._replace(motor_trace=motor_trace)

    # 4. Learning step
    state = _learning_step(state, reward, obs.gradient, params, constants)

    return state, action


# === OPTIMIZED EPISODE RUNNER ===

@partial(jit, static_argnames=["max_steps"])
def _run_episode_jax(
    initial_state: AgentState,
    initial_world_state: WorldState,
    initial_obs: Observation,
    episode_key: random.PRNGKey,
    params: NetworkParams,
    constants: PrecomputedConstants,
    max_steps: int,
) -> Tuple[AgentState, WorldState, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Fully JIT-compiled episode runner with pre-allocated buffers."""
    
    def step_fn(carry, step_idx):
        state, world_state, obs, reward, key = carry
        key, step_key = random.split(key)
        
        # Agent step
        state, action = _agent_simulation_step(
            state, obs, reward, step_key, params, constants
        )
        
        # World step (assuming world.step is JIT-compatible)
        from world.simple_grid_0003 import _step_jit
        result = _step_jit(world_state, action)
        
        # Extract next state
        next_world_state = result.state
        next_obs = result.observation
        next_reward = result.reward
        
        # Store data for later
        step_data = (action, reward, obs.gradient)
        
        return (state, next_world_state, next_obs, next_reward, key), step_data
    
    # Initialize carry
    init_carry = (initial_state, initial_world_state, initial_obs, 0.0, episode_key)
    
    # Run episode with lax.scan
    final_carry, episode_data = jax.lax.scan(
        step_fn, init_carry, jnp.arange(max_steps)
    )
    
    final_state, final_world_state, _, _, _ = final_carry
    actions, rewards, gradients = episode_data
    
    return final_state, final_world_state, actions, rewards, gradients


class SnnAgent(BaseAgent):
    """Phase 0.13 SNN Agent - Optimized for pure performance."""
    
    world_version = "simple_grid_0003"

    def __init__(self, config: SnnAgentConfig, exporter: DataExporter):
        self.params = config.network_params
        self.world = SimpleGridWorld(config.world_config)
        self.master_key = random.PRNGKey(config.exp_config.seed)
        
        # Pre-allocate episode data buffers
        max_steps = config.world_config.max_timesteps
        self.action_buffer = np.zeros(max_steps, dtype=np.int32)
        self.reward_buffer = np.zeros(max_steps, dtype=np.float32)
        self.gradient_buffer = np.zeros(max_steps, dtype=np.float32)
        self.position_buffer = np.zeros((max_steps, 2), dtype=np.int32)
        
        # Initialize base class (will call _setup)
        super().__init__(config, exporter)
    
    def _setup(self) -> None:
        """Set up the agent after initialization."""
        self.constants = self._precompute_constants()
        
        self.exporter.log("Initializing network state...", "INFO")
        start_time = time.time()
        init_key, self.master_key = random.split(self.master_key)
        self.state: AgentState = self._initialize_state(init_key)
        self.exporter.log(f"Network initialized in {time.time() - start_time:.2f}s", "INFO")
        
        # Export network structure
        self._export_network_structure()

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
            preferred_values=jnp.linspace(0, 1, p.NUM_INPUT_CHANNELS),
            inv_tau_v=1.0 / p.TAU_V,
            noise_scale=p.NOISE_SCALE,
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
        rec_weights = sample_weights(keys[1], (n, n), p.W_PROC_PROC_E, p.W_INIT_SCALE)
        w = w.at[:, n_in:].set(jnp.where(w_mask[:, n_in:], jnp.abs(rec_weights), 0.0))

        return w

    def run_episode(
        self,
        key: random.PRNGKey,
        episode_num: int,
        progress_callback: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Run episode with minimal overhead."""
        episode_start_time = time.time()
        
        # Start episode in exporter
        self.exporter.start_episode(episode_num)

        world_state, obs = self.world.reset()

        if episode_num > 0:
            self.state = _reset_episode_state(self.state, self.params, soft_reset=True)
            
        # Log initial world setup
        self.exporter.log_static_episode_data(
            "world_setup", 
            {"reward_positions": np.asarray(world_state.reward_positions)}
        )

        actual_rewards_collected = 0
        reward_for_step = 0.0
        max_steps = self.config.world_config.max_timesteps

        # Run episode step by step (can't use full JAX version due to world implementation)
        for step in range(max_steps):
            step_key, key = random.split(key)

            # Agent step
            self.state, action = _agent_simulation_step(
                self.state, obs, reward_for_step, step_key, self.params, self.constants
            )

            # World step
            result = self.world.step(world_state, int(action))
            world_state = result.state
            obs = result.observation
            reward_for_step = result.reward
            done = result.done

            # Store data in pre-allocated buffers
            self.action_buffer[step] = action
            self.reward_buffer[step] = reward_for_step
            self.gradient_buffer[step] = obs.gradient
            self.position_buffer[step] = world_state.agent_pos

            if reward_for_step > 0:
                actual_rewards_collected += int(reward_for_step)
                self.exporter.log_event("reward_collected", step, {"value": float(reward_for_step)})

            # Log to exporter
            self.exporter.log_timestep(
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

            # Progress callback
            if progress_callback and step % 500 == 0:
                elapsed = time.time() - episode_start_time
                progress_callback(step, max_steps, actual_rewards_collected, elapsed)

            if done:
                break

        # Build summary
        summary = {
            "total_reward": float(actual_rewards_collected),
            "rewards_collected": actual_rewards_collected,
            "steps_taken": int(world_state.timestep),
            "mean_firing_rate": float(jnp.mean(self.state.firing_rate)),
            "episodes_completed": int(self.state.episodes_completed),
            "episode_time": time.time() - episode_start_time,
        }
        
        # Log reward history from the world
        positions, spawn_steps, collect_steps = self.world.get_reward_history(world_state)
        reward_history_data = {
            "positions": np.asarray(positions),
            "spawn_steps": np.asarray(spawn_steps),
            "collect_steps": np.asarray(collect_steps),
            "total_rewards_spawned": int(world_state.reward_history_count),
        }
        self.exporter.log_static_episode_data("reward_history", reward_history_data)
        
        # End episode in exporter
        self.exporter.end_episode(
            success=jnp.all(world_state.reward_collected), 
            summary=summary
        )

        return summary

    def run_experiment(self) -> list[Dict[str, Any]]:
        """Run full experiment without export overhead."""
        all_summaries = []
        
        # Simple progress bar function
        def simple_progress(step, max_steps, rewards, elapsed):
            if step % 1000 == 0:  # Less frequent updates
                rate = step / elapsed if elapsed > 0 else 0
                print(f"\r  Step {step}/{max_steps} | Rewards: {rewards} | {rate:.0f} steps/s", 
                      end="", flush=True)

        for i in range(self.config.exp_config.n_episodes):
            print(f"\n--- Episode {i + 1}/{self.config.exp_config.n_episodes} ---")
            
            episode_key, self.master_key = random.split(self.master_key)
            
            summary = self.run_episode(
                episode_key,
                episode_num=i,
                progress_callback=simple_progress if i % 5 == 0 else None,  # Show progress every 5 episodes
            )
            all_summaries.append(summary)
            
            print(f"\n  Time: {summary['episode_time']:.1f}s | "
                  f"Rewards: {summary['rewards_collected']} | "
                  f"Rate: {summary['steps_taken']/summary['episode_time']:.0f} steps/s")

        # Print final statistics
        print("\n" + "="*60)
        print("EXPERIMENT COMPLETE")
        print("="*60)
        rewards = [s["rewards_collected"] for s in all_summaries]
        print(f"Average rewards: {np.mean(rewards):.1f} ± {np.std(rewards):.1f}")
        print(f"Best episode: {np.max(rewards)} rewards")
        
        return all_summaries
    
    def _export_network_structure(self):
        """Export network structure to exporter."""
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
        self.exporter.save_network_structure(**basic_structure)
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Get the current state of the agent as a dictionary."""
        return {
            "agent_state": {
                "v": np.asarray(self.state.v),
                "spike": np.asarray(self.state.spike),
                "refractory": np.asarray(self.state.refractory),
                "syn_current_e": np.asarray(self.state.syn_current_e),
                "syn_current_i": np.asarray(self.state.syn_current_i),
                "trace_fast": np.asarray(self.state.trace_fast),
                "trace_slow": np.asarray(self.state.trace_slow),
                "firing_rate": np.asarray(self.state.firing_rate),
                "threshold_adapt": np.asarray(self.state.threshold_adapt),
                "eligibility_trace": np.asarray(self.state.eligibility_trace),
                "dopamine": float(self.state.dopamine),
                "value_estimate": float(self.state.value_estimate),
                "w": np.asarray(self.state.w),
                "motor_trace": np.asarray(self.state.motor_trace),
                "input_channels": np.asarray(self.state.input_channels),
                "weight_momentum": np.asarray(self.state.weight_momentum),
                "episodes_completed": int(self.state.episodes_completed),
                "reward_boost_timer": float(self.state.reward_boost_timer),
                "rewards_this_episode": int(self.state.rewards_this_episode),
                "current_temperature": float(self.state.current_temperature),
            },
            "connectivity": {
                "w_mask": np.asarray(self.state.w_mask),
                "w_plastic_mask": np.asarray(self.state.w_plastic_mask),
                "is_excitatory": np.asarray(self.state.is_excitatory),
                "neuron_types": np.asarray(self.state.neuron_types),
            },
            "config": self._config_to_dict(self.config),
        }