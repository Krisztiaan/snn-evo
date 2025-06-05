# models/phase_0_7/agent.py
# keywords: [snn agent, jax implementation, clock neurons, rate-coding, spiking motors]
"""Phase 0.7 SNN Agent: "Awakened" agent with biologically plausible I/O."""

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
    """State of the SNN agent with specialized neuron populations."""
    v: jnp.ndarray
    spike: jnp.ndarray
    refractory: jnp.ndarray
    is_excitatory: jnp.ndarray
    syn_current_e: jnp.ndarray
    syn_current_i: jnp.ndarray
    threshold_adaptation: jnp.ndarray
    stdp_trace: jnp.ndarray
    eligibility_trace: jnp.ndarray
    dopamine: float
    w: jnp.ndarray
    value_estimate: float
    motor_spike_buffer: jnp.ndarray
    current_action: int
    steps_since_decision: int


@partial(jit, static_argnames=['params'])
def _get_input_spikes(gradient: float, params: NetworkParams, key: random.PRNGKey) -> jnp.ndarray:
    rate = gradient * params.MAX_INPUT_RATE_HZ
    prob = rate / 1000.0
    return random.uniform(key, (params.NUM_INPUT_NEURONS,)) < prob


@partial(jit, static_argnames=['params'])
def _neuron_step(state: AgentState, input_spikes: jnp.ndarray, clock_spikes: jnp.ndarray, params: NetworkParams, key: random.PRNGKey) -> AgentState:
    num_all_inputs = params.NUM_INPUT_NEURONS + params.NUM_CLOCK_NEURONS
    
    all_input_spikes = jnp.concatenate([input_spikes, clock_spikes])
    all_recurrent_spikes = state.spike
    
    w_in_all = state.w[:, :num_all_inputs]
    w_rec_all = state.w[:, num_all_inputs:]
    
    syn_input = (w_in_all @ all_input_spikes.astype(jnp.float32)) + \
                (w_rec_all @ all_recurrent_spikes.astype(jnp.float32))

    # A more correct way to handle E/I currents based on presynaptic neuron type
    # This assumes the weight sign already reflects the E/I nature.
    syn_current_e_new = state.syn_current_e * jnp.exp(-1.0 / params.TAU_SYN_E) + jnp.maximum(0, syn_input)
    syn_current_i_new = state.syn_current_i * jnp.exp(-1.0 / params.TAU_SYN_I) + jnp.minimum(0, syn_input)

    i_total = syn_current_e_new + syn_current_i_new + random.normal(key, state.v.shape) * 1.0
    dv = (-state.v + params.V_REST + i_total) / params.TAU_V
    v_new = state.v + dv
    
    refractory_new = jnp.maximum(0, state.refractory - 1.0)
    effective_threshold = params.V_THRESHOLD + state.threshold_adaptation
    spike_new = (v_new >= effective_threshold) & (refractory_new == 0)
    
    v_new = jnp.where(spike_new, params.V_RESET, v_new)
    refractory_new = jnp.where(spike_new, params.REFRACTORY_TIME, refractory_new)
    
    stdp_trace_new = state.stdp_trace * jnp.exp(-1.0 / params.STDP_TRACE_TAU) + spike_new
    
    return state._replace(
        v=v_new, spike=spike_new, refractory=refractory_new,
        syn_current_e=syn_current_e_new, syn_current_i=syn_current_i_new,
        stdp_trace=stdp_trace_new
    )


@partial(jit, static_argnames=['params'])
def _learning_step(state: AgentState, reward: float, prev_value: float, params: NetworkParams) -> AgentState:
    td_error = reward + params.REWARD_DISCOUNT * state.value_estimate - prev_value
    new_value = state.value_estimate + params.REWARD_PREDICTION_RATE * td_error
    decay_d = jnp.exp(-1.0 / params.TAU_DOPAMINE)
    dopamine_response = state.dopamine * decay_d + params.BASELINE_DOPAMINE * (1 - decay_d) + td_error * 0.5
    new_dopamine = jnp.clip(dopamine_response, 0.0, 2.0)

    # --- FIX: Apply STDP only to the recurrent part of the weight/eligibility matrix ---
    num_all_inputs = params.NUM_INPUT_NEURONS + params.NUM_CLOCK_NEURONS
    
    # 1. Calculate STDP for all possible recurrent connections
    ltp = state.stdp_trace[:, None] * state.spike[None, :].astype(float)
    ltd = state.spike[:, None].astype(float) * state.stdp_trace[None, :]
    stdp_rec = params.STDP_A_PLUS * ltp - params.STDP_A_MINUS * ltd
    
    # 2. Update eligibility trace only for the recurrent part
    eligibility_rec = state.eligibility_trace[:, num_all_inputs:]
    decay_e = jnp.exp(-1.0 / params.TAU_ELIGIBILITY)
    new_eligibility_rec = eligibility_rec * decay_e + stdp_rec
    
    # 3. Calculate weight change for the recurrent part
    dopamine_modulation = jax.nn.tanh(
        (new_dopamine - params.BASELINE_DOPAMINE) / (params.BASELINE_DOPAMINE + 1e-8) * 2.0)
    
    w_rec = state.w[:, num_all_inputs:]
    dw_rec = params.BASE_LEARNING_RATE * dopamine_modulation * new_eligibility_rec - params.WEIGHT_DECAY * w_rec
    
    # 4. Construct the full dw matrix and update weights
    dw = jnp.zeros_like(state.w).at[:, num_all_inputs:].set(dw_rec)
    # Only apply weight change where connections exist
    dw = jnp.where(state.w != 0, dw, 0.0)
    updated_weights = state.w + dw
    
    # 5. Update the full eligibility trace matrix
    new_eligibility = state.eligibility_trace.at[:, num_all_inputs:].set(new_eligibility_rec * 0.9)

    return state._replace(
        w=updated_weights, eligibility_trace=new_eligibility,
        dopamine=new_dopamine, value_estimate=new_value
    )


@partial(jit, static_argnames=['params'])
def _decode_action_from_race(motor_spike_buffer: jnp.ndarray, params: NetworkParams) -> int:
    spike_counts = jnp.sum(motor_spike_buffer, axis=0)
    
    MOVE_EN, MOVE_FWD, MOVE_BWD, TURN_EN, TURN_L, TURN_R = 0, 1, 2, 3, 4, 5
    
    move_enabled = spike_counts[MOVE_EN] > params.ENABLE_THRESHOLD
    turn_enabled = spike_counts[TURN_EN] > params.ENABLE_THRESHOLD
    
    # World action mapping: 0:up, 1:right, 2:down, 3:left, 4:stay
    action = 4
    
    action = jax.lax.cond(
        move_enabled,
        lambda: jax.lax.cond(spike_counts[MOVE_FWD] > spike_counts[MOVE_BWD], lambda: 0, lambda: 2),
        lambda: action
    )
    
    action = jax.lax.cond(
        ~move_enabled & turn_enabled,
        lambda: jax.lax.cond(spike_counts[TURN_L] > spike_counts[TURN_R], lambda: 3, lambda: 1),
        lambda: action
    )
    
    return action


class SnnAgent:
    """Phase 0.7 SNN Agent with specialized neuron populations."""

    def __init__(self, config: SnnAgentConfig):
        self.config = config
        self.params = config.network_params
        self.world = SimpleGridWorld(config.world_config)
        self.state: AgentState = self._initialize_state(random.PRNGKey(config.exp_config.seed))

    def _initialize_state(self, key: random.PRNGKey) -> AgentState:
        p = self.params
        num_main = p.NUM_MAIN_NEURONS
        num_motor = p.NUM_MOTOR_NEURONS
        num_total_neurons = num_main + num_motor
        
        num_input = p.NUM_INPUT_NEURONS
        num_clock = p.NUM_CLOCK_NEURONS
        num_all_inputs = num_input + num_clock
        
        num_all_sources = num_all_inputs + num_total_neurons
        
        is_excitatory = jnp.arange(num_total_neurons) < int(num_total_neurons * p.EXCITATORY_RATIO)
        
        w = jnp.zeros((num_total_neurons, num_all_sources))
        
        main_row_slice = slice(0, num_main)
        motor_row_slice = slice(num_main, num_total_neurons)
        
        in_col_slice = slice(0, num_input)
        clock_col_slice = slice(num_input, num_all_inputs)
        main_col_slice = slice(num_all_inputs, num_all_inputs + num_main)
        
        key, subkey1, subkey2 = random.split(key, 3)
        in_main_conn = random.uniform(subkey1, (num_main, num_input)) < p.P_IN_MAIN
        w = w.at[main_row_slice, in_col_slice].set(jnp.where(in_main_conn, random.normal(subkey2, (num_main, num_input)) * p.W_STD_SCALE + p.W_IN_MAIN_MEAN, 0))

        key, subkey1, subkey2 = random.split(key, 3)
        clock_main_conn = random.uniform(subkey1, (num_main, num_clock)) < p.P_CLOCK_MAIN
        w = w.at[main_row_slice, clock_col_slice].set(jnp.where(clock_main_conn, random.normal(subkey2, (num_main, num_clock)) * p.W_STD_SCALE + p.W_CLOCK_MAIN_MEAN, 0))

        key, subkey1, subkey2 = random.split(key, 3)
        main_main_conn = random.uniform(subkey1, (num_main, num_main)) < p.P_MAIN_MAIN
        w_main = jnp.where(main_main_conn, random.normal(subkey2, (num_main, num_main)) * p.W_STD_SCALE + p.W_MAIN_MAIN_MEAN, 0)
        w = w.at[main_row_slice, main_col_slice].set(w_main)

        key, subkey1, subkey2 = random.split(key, 3)
        main_motor_conn = random.uniform(subkey1, (num_motor, num_main)) < p.P_MAIN_MOTOR
        w_motor_in = jnp.where(main_motor_conn, random.normal(subkey2, (num_motor, num_main)) * p.W_STD_SCALE + p.W_MAIN_MOTOR_MEAN, 0)
        w = w.at[motor_row_slice, main_col_slice].set(w_motor_in)
        
        return AgentState(
            v=jnp.full(num_total_neurons, p.V_REST),
            spike=jnp.zeros(num_total_neurons, dtype=bool),
            refractory=jnp.zeros(num_total_neurons),
            is_excitatory=is_excitatory,
            syn_current_e=jnp.zeros(num_total_neurons),
            syn_current_i=jnp.zeros(num_total_neurons),
            threshold_adaptation=jnp.zeros(num_total_neurons),
            stdp_trace=jnp.zeros(num_total_neurons),
            eligibility_trace=jnp.zeros_like(w),
            dopamine=p.BASELINE_DOPAMINE,
            w=w,
            value_estimate=0.0,
            motor_spike_buffer=jnp.zeros((p.DECISION_WINDOW, p.NUM_MOTOR_NEURONS), dtype=bool),
            current_action=4,
            steps_since_decision=0
        )

    def run_episode(self, episode_key: random.PRNGKey, episode_num: int, exporter: DataExporter) -> Dict[str, Any]:
        world_state, obs = self.world.reset(episode_key)
        self.state = self._initialize_state(episode_key)
        
        ep = exporter.start_episode(episode_num)
        exporter.log_static_episode_data("world_setup", {"reward_positions": np.array(world_state.reward_positions)})

        motor_slice = slice(self.params.NUM_MAIN_NEURONS, self.params.NUM_MAIN_NEURONS + self.params.NUM_MOTOR_NEURONS)

        prev_value = 0.0
        for step in range(self.config.world_config.max_timesteps):
            episode_key, input_key, clock_key, neuron_key = random.split(episode_key, 4)
            
            input_spikes = _get_input_spikes(obs.gradient, self.params, input_key)
            clock_prob = self.params.CLOCK_RATE_HZ / 1000.0
            clock_spikes = random.uniform(clock_key, (self.params.NUM_CLOCK_NEURONS,)) < clock_prob

            self.state = _neuron_step(self.state, input_spikes, clock_spikes, self.params, neuron_key)
            
            buffer_idx = self.state.steps_since_decision % self.params.DECISION_WINDOW
            motor_spikes = self.state.spike[motor_slice]
            self.state = self.state._replace(
                motor_spike_buffer=self.state.motor_spike_buffer.at[buffer_idx].set(motor_spikes)
            )
            
            if self.state.steps_since_decision > 0 and self.state.steps_since_decision % self.params.DECISION_WINDOW == 0:
                new_action = _decode_action_from_race(self.state.motor_spike_buffer, self.params)
                self.state = self.state._replace(current_action=int(new_action))

            result = self.world.step(world_state, self.state.current_action)
            world_state, obs, reward, done = result.state, result.observation, result.reward, result.done
            
            self.state = _learning_step(self.state, reward, prev_value, self.params)
            prev_value = self.state.value_estimate

            self.state = self.state._replace(steps_since_decision=self.state.steps_since_decision + 1)

            exporter.log(
                timestep=step,
                neural_state={"spikes": self.state.spike.astype(jnp.uint8)},
                behavior={"action": int(self.state.current_action), "pos_x": int(world_state.agent_pos[0]), "pos_y": int(world_state.agent_pos[1])},
                reward=float(reward)
            )

            if done: break

        summary = {"total_reward": float(world_state.total_reward), "rewards_collected": int(jnp.sum(world_state.reward_collected)), "steps_taken": world_state.timestep}
        exporter.end_episode(success=jnp.all(world_state.reward_collected), summary=summary)
        return summary

    def run_experiment(self):
        master_key = random.PRNGKey(self.config.exp_config.seed)
        with DataExporter(
            experiment_name="snn_agent_phase07",
            output_base_dir=self.config.exp_config.export_dir,
            compression='gzip', compression_level=1
        ) as exporter:
            exporter.save_config(self.config)
            
            all_summaries = []
            for i in range(self.config.exp_config.n_episodes):
                print(f"\n--- Episode {i+1}/{self.config.exp_config.n_episodes} ---")
                episode_key, master_key = random.split(master_key)
                summary = self.run_episode(episode_key, episode_num=i, exporter=exporter)
                all_summaries.append(summary)
                print(f"  Episode Summary: Total Reward: {summary['total_reward']:.2f}, Rewards Collected: {summary['rewards_collected']}, Steps: {summary['steps_taken']}")
        return all_summaries