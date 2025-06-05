# models/phase_0_6/agent.py
# keywords: [snn agent, jax implementation, three-factor learning, goal-directed]
"""Phase 0.6 SNN Agent: Functional, goal-directed implementation."""

from typing import Dict, Any, NamedTuple, Tuple
import jax
import jax.numpy as jnp
from jax import random, jit
import numpy as np
from functools import partial

from .config import SnnAgentConfig, NetworkParams
from world.simple_grid_0001 import SimpleGridWorld, WorldState, Observation
from export import DataExporter


class AgentState(NamedTuple):
    """State of the SNN agent, including metaplasticity and motor variables."""
    # Core dynamics
    v: jnp.ndarray
    spike: jnp.ndarray
    refractory: jnp.ndarray
    is_excitatory: jnp.ndarray
    syn_current_e: jnp.ndarray
    syn_current_i: jnp.ndarray
    
    # Homeostasis
    firing_rate: jnp.ndarray
    threshold_adaptation: jnp.ndarray
    
    ## FIX: Added a proper STDP trace to the state.
    stdp_trace: jnp.ndarray

    # Learning
    eligibility_trace: jnp.ndarray
    dopamine: float
    
    # Metaplasticity
    synapse_age: jnp.ndarray
    learning_rate_factor: jnp.ndarray
    
    # Weights & Connections
    w: jnp.ndarray
    w_in: jnp.ndarray
    w_out: jnp.ndarray
    connection_mask: jnp.ndarray
    
    # Reward Prediction
    value_estimate: float
    prev_value: float
    
    # I/O
    input_population: jnp.ndarray
    motor_trace: jnp.ndarray


@jit
def _apply_dale_principle(w: jnp.ndarray, is_excitatory: jnp.ndarray) -> jnp.ndarray:
    E_pre = is_excitatory[:, None]
    return jnp.where(E_pre, jnp.abs(w), -jnp.abs(w))


@partial(jit, static_argnames=['params'])
def _encode_gradient(gradient: float, params: NetworkParams, key: random.PRNGKey) -> jnp.ndarray:
    centers = jnp.linspace(0, 1, params.NUM_INPUTS)
    sigma = 0.15
    activations = jnp.exp(-(gradient - centers)**2 / (2 * sigma**2))
    noise = random.normal(key, activations.shape) * 0.1
    activations = jnp.maximum(activations + noise, 0.0)
    return activations / (jnp.sum(activations) + 1e-8) * params.NUM_INPUTS * 0.5


@partial(jit, static_argnames=['params'])
def _neuron_step(state: AgentState, params: NetworkParams, key: random.PRNGKey) -> AgentState:
    refractory_new = jnp.maximum(0, state.refractory - 1.0)
    syn_current_e_new = state.syn_current_e * jnp.exp(-1.0 / params.TAU_SYN_E)
    syn_current_i_new = state.syn_current_i * jnp.exp(-1.0 / params.TAU_SYN_I)
    
    spike_input = state.spike.astype(float)
    E_mask = state.is_excitatory[:, None]
    exc_input = jnp.where(E_mask, state.w, 0.0) @ spike_input
    inh_input = jnp.where(~E_mask, state.w, 0.0) @ spike_input
    syn_current_e_new += jnp.abs(exc_input)
    syn_current_i_new += jnp.abs(inh_input)
    
    i_ext = jnp.dot(state.w_in.T, state.input_population) * params.INPUT_SCALE
    i_total = i_ext + syn_current_e_new + syn_current_i_new + random.normal(key, state.v.shape) * 1.0
    
    dv = (-state.v + params.V_REST + i_total) / params.TAU_V
    v_new = state.v + dv

    effective_threshold = params.V_THRESHOLD + state.threshold_adaptation
    spike_new = (v_new >= effective_threshold) & (refractory_new == 0)

    v_new = jnp.where(spike_new, params.V_RESET, v_new)
    refractory_new = jnp.where(spike_new, params.REFRACTORY_TIME, refractory_new)

    alpha = 1.0 / params.HOMEOSTATIC_TAU
    firing_rate_new = state.firing_rate * (1 - alpha) + spike_new * 1000.0 * alpha
    rate_error = firing_rate_new - params.TARGET_RATE_HZ
    threshold_adaptation_new = state.threshold_adaptation + params.THRESHOLD_ADAPT_RATE * rate_error

    ## FIX: Correctly update the STDP trace as a state variable.
    stdp_trace_new = state.stdp_trace * jnp.exp(-1.0 / params.STDP_TRACE_TAU) + spike_new

    return state._replace(
        v=v_new, spike=spike_new, refractory=refractory_new,
        syn_current_e=syn_current_e_new, syn_current_i=syn_current_i_new,
        firing_rate=firing_rate_new,
        threshold_adaptation=jnp.clip(threshold_adaptation_new, -10.0, 10.0),
        stdp_trace=stdp_trace_new
    )

@partial(jit, static_argnames=['params'])
def _update_metaplasticity(state: AgentState, params: NetworkParams) -> AgentState:
    age_increment = jnp.outer(state.spike, state.spike).astype(float) * state.connection_mask * 0.01
    new_age = state.synapse_age + age_increment
    age_factor = jnp.exp(-new_age / params.TAU_METAPLASTICITY)
    
    return state._replace(
        synapse_age=new_age,
        learning_rate_factor=age_factor
    )

@partial(jit, static_argnames=['params'])
def _learning_step(state: AgentState, reward: float, params: NetworkParams) -> AgentState:
    state = _update_metaplasticity(state, params)

    td_error = reward + params.REWARD_DISCOUNT * state.value_estimate - state.prev_value
    new_value = state.value_estimate + params.REWARD_PREDICTION_RATE * td_error
    decay_d = jnp.exp(-1.0 / params.TAU_DOPAMINE)
    dopamine_response = state.dopamine * decay_d + params.BASELINE_DOPAMINE * (1 - decay_d) + td_error * 0.5
    new_dopamine = jnp.clip(dopamine_response, 0.0, 2.0)

    ## FIX: Use the correctly updated state.stdp_trace.
    ltp = state.stdp_trace[:, None] * state.spike[None, :].astype(float)
    ltd = state.spike[:, None].astype(float) * state.stdp_trace[None, :]
    stdp = jnp.where(state.connection_mask, params.STDP_A_PLUS * ltp - params.STDP_A_MINUS * ltd, 0.0)
    decay_e = jnp.exp(-1.0 / params.TAU_ELIGIBILITY)
    new_eligibility = state.eligibility_trace * decay_e + stdp

    dopamine_modulation = jax.nn.tanh(
        (new_dopamine - params.BASELINE_DOPAMINE) / (params.BASELINE_DOPAMINE + 1e-8) * 2.0)
    effective_lr = params.BASE_LEARNING_RATE * state.learning_rate_factor
    dw = effective_lr * dopamine_modulation * new_eligibility - params.WEIGHT_DECAY * state.w
    dw = jnp.where(state.connection_mask, dw, 0.0)
    updated_weights = _apply_dale_principle(state.w + dw, state.is_excitatory)

    return state._replace(
        w=updated_weights, eligibility_trace=new_eligibility * 0.9,
        dopamine=new_dopamine, value_estimate=new_value, prev_value=new_value
    )

@partial(jit, static_argnames=['params'])
def _decode_action(state: AgentState, params: NetworkParams, key: random.PRNGKey) -> Tuple[int, jnp.ndarray]:
    ## FIX: Action decoding should be based on current spikes, not a trace.
    exc_activity = jnp.where(state.is_excitatory, state.spike.astype(float), 0.0)
    motor_input = jnp.dot(exc_activity, state.w_out)
    
    motor_trace_new = state.motor_trace * jnp.exp(-1.0 / params.MOTOR_TAU) + motor_input
    
    action_probs = jax.nn.softmax(motor_trace_new / params.ACTION_TEMP)
    action = random.categorical(key, jnp.log(action_probs + 1e-8))
    
    return action, motor_trace_new


class SnnAgent:
    """The main SNN agent class for Phase 0.6."""

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
        dist_matrix = jnp.minimum(jnp.abs(positions[:, None] - positions[None, :]), n - jnp.abs(positions[:, None] - positions[None, :]))
        local_bias = jnp.exp(-dist_matrix / (n * p.LOCAL_CONNECTIVITY_SCALE))
        prob_matrix = jnp.zeros((n, n))
        prob_matrix = jnp.where(jnp.outer(is_excitatory, is_excitatory), p.P_EE * local_bias, prob_matrix)
        prob_matrix = jnp.where(jnp.outer(is_excitatory, ~is_excitatory), p.P_EI, prob_matrix)
        prob_matrix = jnp.where(jnp.outer(~is_excitatory, is_excitatory), p.P_IE, prob_matrix)
        prob_matrix = jnp.where(jnp.outer(~is_excitatory, ~is_excitatory), p.P_II * local_bias, prob_matrix)
        connection_mask = random.uniform(keys[0], (n, n)) < prob_matrix
        connection_mask = connection_mask.at[jnp.diag_indices(n)].set(False)

        # Weights
        w = jnp.zeros((n, n))
        w = jnp.where(jnp.outer(is_excitatory, is_excitatory) & connection_mask, random.normal(keys[1], (n, n)) * p.W_EE_MEAN * p.W_STD_SCALE + p.W_EE_MEAN, w)
        w = jnp.where(jnp.outer(is_excitatory, ~is_excitatory) & connection_mask, random.normal(keys[2], (n, n)) * p.W_EI_MEAN * p.W_STD_SCALE + p.W_EI_MEAN, w)
        w = jnp.where(jnp.outer(~is_excitatory, is_excitatory) & connection_mask, random.normal(keys[3], (n, n)) * p.W_IE_MEAN * p.W_STD_SCALE + p.W_IE_MEAN, w)
        w = jnp.where(jnp.outer(~is_excitatory, ~is_excitatory) & connection_mask, random.normal(keys[4], (n, n)) * p.W_II_MEAN * p.W_STD_SCALE + p.W_II_MEAN, w)
        w = _apply_dale_principle(w, is_excitatory)

        input_centers = jnp.linspace(0, 1, p.NUM_INPUTS)
        neuron_prefs = jnp.linspace(0, 1, n)
        w_in = jnp.exp(-((input_centers[None, :] - neuron_prefs[:, None])**2) / (2 * 0.15**2))
        w_in = (w_in + random.normal(keys[5], w_in.shape) * 0.1) / (jnp.sum(w_in, axis=1, keepdims=True) + 1e-8)
        w_out = jnp.where(is_excitatory[:, None], jnp.abs(random.normal(keys[6], (n, p.NUM_OUTPUTS)) * 0.1), 0.0)

        return AgentState(
            v=jnp.full(n, p.V_REST), spike=jnp.zeros(n, dtype=bool), refractory=jnp.zeros(n),
            is_excitatory=is_excitatory, firing_rate=jnp.full(n, p.TARGET_RATE_HZ),
            threshold_adaptation=jnp.zeros(n), syn_current_e=jnp.zeros(n), syn_current_i=jnp.zeros(n),
            stdp_trace=jnp.zeros(n),  # Initialize the new trace
            eligibility_trace=jnp.zeros((n, n)), dopamine=p.BASELINE_DOPAMINE,
            synapse_age=jnp.zeros((n, n)), learning_rate_factor=jnp.ones((n, n)),
            w=w, w_in=w_in.T, w_out=w_out, connection_mask=connection_mask,
            value_estimate=0.0, prev_value=0.0, input_population=jnp.zeros(p.NUM_INPUTS),
            motor_trace=jnp.zeros(p.NUM_OUTPUTS)
        )

    def run_episode(self, episode_key: random.PRNGKey, episode_num: int, exporter: DataExporter) -> Dict[str, Any]:
        world_state, obs = self.world.reset(episode_key)
        self.state = self._initialize_state(episode_key)

        ep = exporter.start_episode(episode_num)
        exporter.log_static_episode_data("world_setup", {"reward_positions": np.array(world_state.reward_positions)})

        for step in range(self.config.world_config.max_timesteps):
            episode_key, step_key, encode_key, neuron_key, decision_key = random.split(episode_key, 5)

            input_pop = _encode_gradient(obs.gradient, self.params, encode_key)
            self.state = self.state._replace(input_population=input_pop)
            self.state = _neuron_step(self.state, self.params, neuron_key)
            
            action, motor_trace = _decode_action(self.state, self.params, decision_key)
            self.state = self.state._replace(motor_trace=motor_trace)

            result = self.world.step(world_state, int(action))
            world_state, obs, reward, done = result.state, result.observation, result.reward, result.done

            self.state = _learning_step(self.state, reward, self.params)

            # Log neural state only if there are spikes to avoid empty data
            if jnp.any(self.state.spike):
                exporter.log(
                    timestep=step,
                    neural_state={"v": self.state.v, "spikes": self.state.spike.astype(jnp.uint8)},
                    behavior={"action": int(action), "pos_x": int(world_state.agent_pos[0]), "pos_y": int(world_state.agent_pos[1])},
                    reward=float(reward)
                )
            else: # Log behavior even without spikes
                exporter.log(
                    timestep=step,
                    behavior={"action": int(action), "pos_x": int(world_state.agent_pos[0]), "pos_y": int(world_state.agent_pos[1])},
                    reward=float(reward)
                )


            if done:
                break

        summary = {"total_reward": float(world_state.total_reward), "rewards_collected": int(jnp.sum(world_state.reward_collected)), "steps_taken": world_state.timestep}
        exporter.end_episode(success=jnp.all(world_state.reward_collected), summary=summary)
        return summary

    def run_experiment(self):
        master_key = random.PRNGKey(self.config.exp_config.seed)

        with DataExporter(
            experiment_name="snn_agent_phase06",
            output_base_dir=self.config.exp_config.export_dir,
            compression='gzip', compression_level=1
        ) as exporter:
            exporter.save_config(self.config)
            exporter.save_network_structure(
                neurons={'neuron_ids': np.arange(self.params.NUM_NEURONS), 'is_excitatory': np.array(self.state.is_excitatory)},
                connections={'source_ids': np.where(self.state.connection_mask)[0], 'target_ids': np.where(self.state.connection_mask)[1]},
                initial_weights={'weights': np.array(self.state.w[self.state.connection_mask])}
            )

            all_summaries = []
            for i in range(self.config.exp_config.n_episodes):
                print(f"\n--- Episode {i+1}/{self.config.exp_config.n_episodes} ---")
                episode_key, master_key = random.split(master_key)
                summary = self.run_episode(episode_key, episode_num=i, exporter=exporter)
                all_summaries.append(summary)
                print(f"  Episode Summary: Total Reward: {summary['total_reward']:.2f}, Rewards Collected: {summary['rewards_collected']}, Steps: {summary['steps_taken']}")

        return all_summaries