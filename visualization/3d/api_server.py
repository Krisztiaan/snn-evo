#!/usr/bin/env python3
# keywords: [api, server, hdf5, scientific visualization, caching, performance]
"""
API server for the SNN Scientific Visualization Tool.

This server provides a comprehensive, structured JSON endpoint for a single
experiment episode. It pre-scans and caches the experiment list at startup
for high performance and filters for compatible data versions.
"""

import json
import mimetypes
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import h5py
import numpy as np
import traceback  # For more detailed server-side error logging

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from export import __version__ as COMPATIBLE_EXPORTER_VERSION
from export.loader import ExperimentLoader
from export.utils import NumpyEncoder

# Global cache for the experiment list
EXPERIMENT_CACHE = {}


def scan_and_cache_experiments():
    """
    Scans the experiments directory and caches the list of compatible runs.
    This is a slow operation and should only be run once at startup.
    """
    print(f"Scanning for experiments compatible with exporter v{COMPATIBLE_EXPORTER_VERSION}...")
    experiments_by_model = {}
    base_dir = Path("experiments")

    if not base_dir.exists():
        return {"models": {}, "count": 0}

    for model_dir in sorted(base_dir.iterdir()):
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name
        model_experiments = []

        for exp_dir in sorted(model_dir.iterdir(), reverse=True):
            if not exp_dir.is_dir() or not list(exp_dir.glob("*.h5")):
                continue

            try:
                with ExperimentLoader(exp_dir) as loader:
                    metadata = loader.get_metadata()

                    exp_version = metadata.get("exporter_version")
                    if exp_version != COMPATIBLE_EXPORTER_VERSION:
                        continue

                    episodes = loader.list_episodes()
                    if not episodes:
                        continue

                    label = f"{metadata.get('experiment_name', exp_dir.name)} ({len(episodes)} ep)"
                    model_experiments.append(
                        {
                            "path": str(exp_dir),
                            "label": label,
                            "episodes": episodes,
                            "timestamp": exp_dir.stat().st_mtime,
                        }
                    )
            except Exception as e:
                # This is expected for corrupted or older format files
                print(f"  - Skipping incompatible/corrupt run {exp_dir.name}: {e}")

        if model_experiments:
            experiments_by_model[model_name] = model_experiments

    return {
        "models": experiments_by_model,
        "count": sum(len(runs) for runs in experiments_by_model.values()),
    }


class ScientificAPIHandler(BaseHTTPRequestHandler):
    """Handle API requests for structured experiment data."""

    def do_GET(self):
        """Handle GET requests."""
        parsed_path = urlparse(self.path)
        try:
            if parsed_path.path.startswith("/api/"):
                self.handle_api_request(parsed_path)
            else:
                self.serve_static_file(parsed_path)
        except Exception as e:
            # Log the full traceback to the server console for debugging
            print(f"Exception processing {self.path}:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)

            # Send a generic error response to the client
            self.send_error_response(500, f"Internal Server Error: {type(e).__name__}")
            # Use the standard logging method for http.server, which expects format string and args
            self.log_error("Error processing %s: %s", self.path, str(e))

    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def handle_api_request(self, parsed_path):
        """Route API requests to the correct handler."""
        # Headers will be sent after send_response within each specific path handler

        if parsed_path.path == "/api/experiments":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.handle_list_experiments()
        elif parsed_path.path == "/api/episode-data":
            params = parse_qs(parsed_path.query)
            path = params.get("path", [""])[0]
            episode_id_str = params.get("episode", [None])[0]  # Get as string or None

            if not path or episode_id_str is None:
                self.send_error_response(
                    400, "Experiment 'path' and 'episode' parameters are required."
                )
                return

            try:
                episode_id = int(episode_id_str)
            except ValueError:
                self.send_error_response(400, "Experiment 'episode' parameter must be an integer.")
                return

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.handle_get_episode_data(path, episode_id)
        else:
            self.send_error_response(404, "API endpoint not found.")

    def serve_static_file(self, parsed_path):
        """Serve static files (HTML, JS, CSS) from the current directory."""
        path = parsed_path.path
        if path == "/":
            path = "/grid_world_trajectory_viewer.html"

        file_path = (Path(__file__).parent / path.lstrip("/")).resolve()

        if not file_path.is_relative_to(
            Path(__file__).parent.resolve()
        ):  # Ensure parent path is resolved
            self.send_error_response(403, "Forbidden")
            return

        if file_path.is_file():
            self.send_response(200)
            mime_type, _ = mimetypes.guess_type(file_path)
            self.send_header("Content-Type", mime_type or "application/octet-stream")
            self.end_headers()
            with open(file_path, "rb") as f:
                self.wfile.write(f.read())
        else:
            self.send_error_response(404, f"Static file not found: {path}")

    def handle_list_experiments(self):
        """Serve the cached list of experiments."""
        self.wfile.write(json.dumps(EXPERIMENT_CACHE, cls=NumpyEncoder).encode())

    def handle_get_episode_data(self, path_str: str, episode_id: int):
        """Load, process, and return comprehensive data for one episode."""
        with ExperimentLoader(Path(path_str)) as loader:
            exp_config = loader.get_config()
            exp_meta = loader.get_metadata()
            episode = loader.get_episode(episode_id)
            summary = episode.get_metadata()

            if (
                not episode
                or not hasattr(episode, "episode_group")
                or not isinstance(episode.episode_group, h5py.Group)
            ):
                self.send_error_response(
                    404, f"Episode {episode_id} data not fully loaded or invalid."
                )
                return

            neural_states_log = episode.get_neural_states()
            rewards_log = episode.get_rewards()
            world_setup = episode.get_static_data("world_setup")

            world_cfg = exp_config.get("world_config", {})
            net_cfg = exp_config.get("network_params", {})
            total_neurons = (
                net_cfg.get("NUM_SENSORY", 0)
                + net_cfg.get("NUM_PROCESSING", 0)
                + net_cfg.get("NUM_READOUT", 0)
            )
            if total_neurons == 0:
                total_neurons = exp_meta.get("n_neurons", 256)

            neural_ts_raw = neural_states_log.get("timesteps", np.array([]))
            if not isinstance(neural_ts_raw, np.ndarray):
                neural_ts_raw = np.array([])
            neural_map = {ts: i for i, ts in enumerate(neural_ts_raw)}

            reward_map = dict(
                zip(
                    rewards_log.get("timesteps", np.array([])),
                    rewards_log.get("values", np.array([])),
                )
            )

            spike_counts_at_neural_ts = []
            spikes_dataset = neural_states_log.get("spikes")
            if isinstance(spikes_dataset, np.ndarray) and spikes_dataset.ndim == 2:
                if spikes_dataset.shape[0] == len(neural_ts_raw):
                    spike_counts_at_neural_ts = np.sum(spikes_dataset, axis=1).tolist()
                else:
                    spike_counts_at_neural_ts = [0] * len(neural_ts_raw)
            else:
                spike_counts_at_neural_ts = [0] * len(neural_ts_raw)

            behavior_log = episode.get_behavior()
            timeline = []
            behavior_timesteps_raw = behavior_log.get("timesteps", np.array([]))
            if not isinstance(behavior_timesteps_raw, np.ndarray):
                behavior_timesteps_raw = np.array([])

            num_behavior_steps = len(behavior_timesteps_raw)
            position_data = behavior_log.get("position", np.zeros((num_behavior_steps, 2)))

            if not (
                isinstance(position_data, np.ndarray)
                and position_data.ndim == 2
                and position_data.shape[0] == num_behavior_steps
            ):
                pos_x_data = np.zeros(num_behavior_steps)
                pos_y_data = np.zeros(num_behavior_steps)
            else:
                pos_x_data = (
                    position_data[:, 0]
                    if position_data.shape[1] >= 1
                    else np.zeros(num_behavior_steps)
                )
                pos_y_data = (
                    position_data[:, 1]
                    if position_data.shape[1] >= 2
                    else np.zeros(num_behavior_steps)
                )

            action_data_raw = behavior_log.get("action", np.zeros(num_behavior_steps))
            if not (
                isinstance(action_data_raw, np.ndarray)
                and action_data_raw.shape[0] == num_behavior_steps
            ):
                action_data = np.zeros(num_behavior_steps, dtype=int)
            else:
                action_data = action_data_raw.astype(int)

            gradient_data_raw = behavior_log.get("gradient", np.zeros(num_behavior_steps))
            if not (
                isinstance(gradient_data_raw, np.ndarray)
                and gradient_data_raw.shape[0] == num_behavior_steps
            ):
                gradient_data = np.zeros(num_behavior_steps, dtype=float)
            else:
                gradient_data = gradient_data_raw.astype(float)

            for i in range(num_behavior_steps):
                ts = behavior_timesteps_raw[i]
                step_data = {
                    "step": int(ts),
                    "position": [float(pos_x_data[i]), float(pos_y_data[i])],
                    "action": int(action_data[i]),
                    "gradient": float(gradient_data[i]),
                    "reward": float(reward_map.get(ts, 0.0)),
                    "neural": None,
                }

                if ts in neural_map:
                    idx = neural_map[ts]
                    spike_count = 0
                    if idx < len(spike_counts_at_neural_ts):
                        spike_count = int(spike_counts_at_neural_ts[idx])

                    firing_rate = (
                        (spike_count / total_neurons) * 1000.0 if total_neurons > 0 else 0.0
                    )

                    dopamine_levels = neural_states_log.get("dopamine_levels")
                    value_estimates = neural_states_log.get("value_estimate")
                    membrane_potentials_all = neural_states_log.get("membrane_potential")

                    mean_potential_val = 0.0
                    if (
                        isinstance(membrane_potentials_all, np.ndarray)
                        and membrane_potentials_all.ndim >= 1
                        and idx < membrane_potentials_all.shape[0]
                    ):
                        current_v = membrane_potentials_all[idx]
                        if isinstance(current_v, np.ndarray) and current_v.size > 0:
                            mean_potential_val = float(
                                np.mean(current_v)
                            )  # np.mean returns a float or np.float
                        elif np.isscalar(current_v) and isinstance(
                            current_v, (int, float, np.number)
                        ):
                            mean_potential_val = float(current_v)  # Ensure it's a number type

                    step_data["neural"] = {
                        "mean_potential": mean_potential_val,
                        "spike_count": spike_count,
                        "firing_rate_hz": firing_rate,
                        "dopamine": float(dopamine_levels[idx])
                        if isinstance(dopamine_levels, np.ndarray)
                        and idx < len(dopamine_levels)
                        and np.isscalar(dopamine_levels[idx])
                        and isinstance(dopamine_levels[idx], (int, float, np.number))
                        else None,
                        "value_estimate": float(value_estimates[idx])
                        if isinstance(value_estimates, np.ndarray)
                        and idx < len(value_estimates)
                        and np.isscalar(value_estimates[idx])
                        and isinstance(value_estimates[idx], (int, float, np.number))
                        else None,
                    }
                timeline.append(step_data)

            response = {
                "metadata": {
                    "experimentName": exp_meta.get("experiment_name", "N/A"),
                    "episodeId": episode_id,
                    "totalSteps": len(timeline),
                    "world": {"gridSize": world_cfg.get("grid_size", [100, 100])[0]},
                    "network": {"totalNeurons": total_neurons},
                },
                "worldSetup": {
                    "rewardPositions": world_setup.get("reward_positions", np.array([])).tolist()
                },
                "timeline": timeline,
                "summary": summary,
            }
            self.wfile.write(json.dumps(response, cls=NumpyEncoder).encode())

    def send_error_response(self, code: int, message: str):
        """Send an error response."""
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps({"error": message}).encode())


def run_server(port=8008):
    """Run the API server."""
    global EXPERIMENT_CACHE
    EXPERIMENT_CACHE = scan_and_cache_experiments()
    if EXPERIMENT_CACHE["count"] == 0:
        print("Warning: No compatible experiments found. API will serve empty lists.")
    else:
        print(
            f"Serving {EXPERIMENT_CACHE['count']} experiments from {len(EXPERIMENT_CACHE['models'])} models."
        )

    server_address = ("", port)
    httpd = HTTPServer(server_address, ScientificAPIHandler)
    print(f"Scientific API server running on port {port}...")
    print(f"View: http://localhost:{port}/grid_world_trajectory_viewer.html")
    httpd.serve_forever()


if __name__ == "__main__":
    # Default port, can be overridden by command line argument
    server_port = int(sys.argv[1]) if len(sys.argv) > 1 else 8008
    run_server(port=server_port)
