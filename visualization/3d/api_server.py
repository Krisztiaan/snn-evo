#!/usr/bin/env python3
# keywords: [api, server, hdf5, experiment loader, visualization]
"""API server for HDF5 experiment data using ExperimentLoader."""

import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from export.loader import ExperimentLoader
from export.utils import NumpyEncoder

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class ExperimentAPIHandler(BaseHTTPRequestHandler):
    """Handle API requests for experiment data."""

    def do_GET(self):
        """Handle GET requests."""
        parsed_path = urlparse(self.path)
        print(f"GET request: {parsed_path.path}")

        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

            if parsed_path.path == "/api/experiments":
                self.handle_list_experiments()
            elif parsed_path.path == "/api/trajectory":
                params = parse_qs(parsed_path.query)
                path = params.get("path", [""])[0]
                episode = int(params.get("episode", [0])[0])
                self.handle_get_trajectory(path, episode)
            else:
                self.send_error_response(404, "Endpoint not found")

        except Exception as e:
            import traceback

            print(f"Error in GET handler: {e}")
            traceback.print_exc()
            self.send_error_response(500, str(e))

    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def handle_list_experiments(self):
        """List all available experiments organized by model type."""
        experiments_by_model = {}
        base_dir = Path("experiments")

        if not base_dir.exists():
            self.wfile.write(json.dumps({"models": {}, "count": 0}).encode())
            return

        for model_dir in base_dir.iterdir():
            if not model_dir.is_dir():
                continue

            model_name = model_dir.name
            model_experiments = []

            for exp_dir in model_dir.iterdir():
                if not exp_dir.is_dir():
                    continue

                try:
                    with ExperimentLoader(exp_dir) as loader:
                        metadata = loader.get_metadata()
                        config = loader.get_config()
                        episodes = loader.list_episodes()

                        if not episodes:
                            continue

                        exp_info = {
                            "path": str(exp_dir),
                            "directory": exp_dir.name,
                            "metadata": metadata,
                            "config": config,
                            "episodes": episodes,
                            "timestamp": exp_dir.stat().st_mtime,
                        }

                        label = (
                            f"{metadata.get('experiment_name', exp_dir.name)} ({len(episodes)} ep)"
                        )
                        exp_info["label"] = label
                        model_experiments.append(exp_info)

                except Exception as e:
                    print(f"Could not load experiment from {exp_dir}: {e}")

            if model_experiments:
                model_experiments.sort(key=lambda x: x["timestamp"], reverse=True)
                experiments_by_model[model_name] = model_experiments

        total_count = sum(len(runs) for runs in experiments_by_model.values())
        response = {
            "models": experiments_by_model,
            "model_count": len(experiments_by_model),
            "total_count": total_count,
        }
        self.wfile.write(json.dumps(response, cls=NumpyEncoder).encode())

    def handle_get_trajectory(self, path, episode_id):
        """Get trajectory data for visualization."""
        if not path:
            return self.send_error_response(400, "Path parameter required")

        try:
            with ExperimentLoader(Path(path)) as loader:
                config = loader.get_config()
                episode = loader.get_episode(episode_id)

                behavior = episode.get_behavior()
                rewards_log = episode.get_rewards()
                world_setup = episode.get_static_data("world_setup")
                reward_positions = world_setup.get("reward_positions", [])
                neural_states = episode.get_neural_states()
                spike_data = episode.get_spikes()  # Get sparse spike data (may be empty)

                world_config = config.get("world_config", {})

                positions = np.column_stack((behavior.get("pos_x", []), behavior.get("pos_y", [])))
                actions = behavior.get("action", [])
                observations = behavior.get("gradient", [])

                viz_data = {
                    "metadata": {
                        "gridSize": world_config.get("grid_size", 100),
                        "nRewards": len(reward_positions),
                        "totalSteps": len(positions),
                        "seed": config.get("seed", -1),
                        "episodeId": episode_id,
                        "simulationDt": simulation_dt,
                        "neuralSamplingRate": neural_sampling_rate,
                        "hasNeuralData": has_neural_data,
                        "totalNeurons": total_neurons,
                    },
                    "trajectory": [],
                    "world": {"rewardPositions": reward_positions.tolist()},
                }

                cumulative_reward = 0.0
                rewards_collected_mask = np.zeros(len(reward_positions), dtype=bool)

                # Process neural states if available
                has_neural_data = neural_states and "v" in neural_states

                # Get timesteps for behavior data - these are our reference timesteps
                behavior_timesteps = behavior.get("timesteps", np.arange(len(positions)))

                # Create neural state timestep mapping
                neural_timestep_map = {}
                if has_neural_data and "timesteps" in neural_states:
                    neural_timesteps = neural_states["timesteps"]
                    for idx, ts in enumerate(neural_timesteps):
                        neural_timestep_map[ts] = idx

                # Pre-process spike data to count spikes per actual timestep
                spike_counts_per_timestep = {}

                # Check for sparse spike data first (newer format)
                if spike_data and "timesteps" in spike_data:
                    spike_timesteps = spike_data["timesteps"]
                    neuron_ids = spike_data.get("neuron_ids", spike_data.get("values", []))
                    for t, neuron_id in zip(spike_timesteps, neuron_ids):
                        spike_counts_per_timestep[t] = spike_counts_per_timestep.get(t, 0) + 1

                # If no sparse spike data, check for dense spike data in neural_states
                elif neural_states and "spikes" in neural_states:
                    # Dense spike format: rows are timesteps, columns are neurons
                    dense_spikes = neural_states["spikes"]
                    neural_timesteps = neural_states.get("timesteps", np.arange(len(dense_spikes)))

                    for idx, timestep in enumerate(neural_timesteps):
                        if idx < len(dense_spikes):
                            # Count number of spikes (1s) in this timestep
                            spike_count = int(np.sum(dense_spikes[idx]))
                            if spike_count > 0:
                                spike_counts_per_timestep[timestep] = spike_count

                # Get network size for proper normalization
                network_params = config.get("network_params", {})
                total_neurons = network_params.get("n_exc", 0) + network_params.get("n_inh", 0)
                if total_neurons == 0:
                    # Fallback for phase_0_8 config structure
                    total_neurons = (
                        network_params.get("NUM_SENSORY", 32)
                        + network_params.get("NUM_PROCESSING", 192)
                        + network_params.get("NUM_READOUT", 32)
                    )

                # Get simulation parameters from metadata
                metadata = loader.get_metadata()
                simulation_dt = metadata.get("simulation_dt", 1.0)  # Default 1ms timestep
                neural_sampling_rate = metadata.get(
                    "neural_sampling_rate", network_params.get("neural_sampling_rate", 100)
                )

                for i in range(len(positions)):
                    # Get the actual timestep for this behavior sample
                    current_timestep = int(behavior_timesteps[i])

                    step_reward = 0.0
                    if rewards_log and "timesteps" in rewards_log:
                        # Match rewards by actual timestep, not array index
                        reward_mask = rewards_log["timesteps"] == current_timestep
                        if np.any(reward_mask):
                            step_reward = float(np.sum(rewards_log["rewards"][reward_mask]))
                    cumulative_reward += step_reward

                    # Update collected rewards mask
                    if step_reward >= world_config.get("reward_value", 1.0):
                        agent_pos = positions[i]
                        for j, r_pos in enumerate(reward_positions):
                            if (
                                not rewards_collected_mask[j]
                                and np.linalg.norm(agent_pos - r_pos) < 0.5
                            ):
                                rewards_collected_mask[j] = True
                                break

                    step_data = {
                        "step": i,
                        "agentPos": positions[i].tolist(),
                        "action": int(actions[i]) if i < len(actions) else -1,
                        "observation": float(observations[i]) if i < len(observations) else 0.0,
                        "reward": step_reward,
                        "cumulativeReward": cumulative_reward,
                        "rewardCollected": rewards_collected_mask.tolist(),
                    }

                    # Add neural activity metrics if available
                    if has_neural_data:
                        # Find the nearest neural state sample for this timestep
                        neural_idx = None
                        if current_timestep in neural_timestep_map:
                            # Exact match
                            neural_idx = neural_timestep_map[current_timestep]
                        else:
                            # Find nearest sampled timestep
                            if neural_timestep_map:
                                # Get all neural timesteps and find closest
                                neural_ts_list = sorted(neural_timestep_map.keys())
                                # Find the largest timestep <= current_timestep
                                for ts in reversed(neural_ts_list):
                                    if ts <= current_timestep:
                                        neural_idx = neural_timestep_map[ts]
                                        break

                        if neural_idx is not None and neural_idx < len(neural_states["v"]):
                            # Get neural state data
                            v_values = neural_states["v"][neural_idx]

                            # Get spike count from the same neural sample
                            spike_count = 0
                            if "spikes" in neural_states and neural_idx < len(
                                neural_states["spikes"]
                            ):
                                # Count spikes in this neural sample
                                spike_count = int(np.sum(neural_states["spikes"][neural_idx]))
                            else:
                                # Fall back to sparse spike data if available
                                spike_count = spike_counts_per_timestep.get(current_timestep, 0)

                            # Calculate instantaneous firing rate (spikes/neuron/timestep -> Hz)
                            # Convert to Hz: spikes per neuron per millisecond * 1000
                            firing_rate_hz = (
                                (spike_count / total_neurons / simulation_dt) * 1000.0
                                if total_neurons > 0
                                else 0.0
                            )

                            step_data["neuralActivity"] = {
                                "meanPotential": float(np.mean(v_values)),
                                "spikeCount": spike_count,
                                "firingRateHz": firing_rate_hz,
                                "maxPotential": float(np.max(v_values)),
                                "minPotential": float(np.min(v_values)),
                                "timestep": current_timestep,
                                "neuralTimestep": int(neural_ts_list[neural_idx])
                                if "neural_ts_list" in locals()
                                else current_timestep,
                            }

                    viz_data["trajectory"].append(step_data)

                summary = loader.get_episode_summary(episode_id)
                viz_data["metadata"].update(
                    {
                        "totalReward": summary.get("total_reward", 0),
                        "rewardsCollected": summary.get("rewards_collected", 0),
                        "coverage": summary.get("coverage", 0),
                    }
                )

                self.wfile.write(json.dumps(viz_data, cls=NumpyEncoder).encode())
        except Exception as e:
            import traceback

            traceback.print_exc()
            self.send_error_response(500, f"Error loading trajectory: {e}")

    def send_error_response(self, code, message):
        """Send error response."""
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            error_data = {"error": True, "code": code, "message": message}
            self.wfile.write(json.dumps(error_data).encode())
        except:
            pass


def main():
    """Run the API server."""
    port = 8000
    server_address = ("", port)

    class CombinedHandler(ExperimentAPIHandler):
        def do_GET(self):
            if self.path.startswith("/api/"):
                super().do_GET()
            else:
                self.serve_static_file()

        def serve_static_file(self):
            path = self.path.split("?")[0]
            if path == "/":
                path = "/grid_world_trajectory_viewer.html"

            file_path = Path(__file__).parent / path.lstrip("/")

            if file_path.exists() and file_path.is_file():
                content_type = "text/html"
                if path.endswith(".js"):
                    content_type = "application/javascript"
                elif path.endswith(".css"):
                    content_type = "text/css"

                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.end_headers()
                with open(file_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"File not found")

    httpd = HTTPServer(server_address, CombinedHandler)
    print(f"Experiment Visualization Server running at: http://localhost:{port}/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        httpd.shutdown()


if __name__ == "__main__":
    main()
