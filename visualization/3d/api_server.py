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

import numpy as np

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
        except BrokenPipeError:
            self.log_message("Client closed connection.")
        except Exception as e:
            import traceback
            self.log_error(f"Error processing {self.path}: {e}")
            traceback.print_exc()
            self.send_error_response(500, f"Internal Server Error: {e}")

    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def handle_api_request(self, parsed_path):
        """Route API requests to the correct handler."""
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")

        if parsed_path.path == "/api/experiments":
            self.send_response(200)
            self.end_headers()
            self.handle_list_experiments()
        elif parsed_path.path == "/api/episode-data":
            params = parse_qs(parsed_path.query)
            path = params.get("path", [""])[0]
            episode_id = int(params.get("episode", [0])[0])
            if not path:
                self.send_error_response(400, "Experiment 'path' parameter is required.")
                return
            self.send_response(200)
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

        if not file_path.is_relative_to(Path(__file__).parent):
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

    def handle_get_episode_data(self, path: str, episode_id: int):
        """Load, process, and return comprehensive data for one episode."""
        with ExperimentLoader(Path(path)) as loader:
            exp_config = loader.get_config()
            exp_meta = loader.get_metadata()
            episode = loader.get_episode(episode_id)
            summary = episode.get_metadata()

            behavior = episode.get_behavior()
            neural = episode.get_neural_states()
            rewards_log = episode.get_rewards()
            world_setup = episode.get_static_data("world_setup")

            world_cfg = exp_config.get("world_config", {})
            net_cfg = exp_config.get("network_params", {})
            total_neurons = net_cfg.get("NUM_SENSORY", 0) + net_cfg.get(
                "NUM_PROCESSING", 0
            ) + net_cfg.get("NUM_READOUT", 0)
            if total_neurons == 0:
                total_neurons = exp_meta.get("n_neurons", 256)

            neural_ts = neural.get("timesteps", [])
            neural_map = {ts: i for i, ts in enumerate(neural_ts)}
            reward_map = {ts: val for ts, val in zip(rewards_log.get("timesteps", []), rewards_log.get("values", []))}

            spike_counts = {}
            if "spikes" in episode.group and "timesteps" in episode.group["spikes"]:
                spike_ts_arr = episode.group["spikes"]["timesteps"][:]
                unique_ts, counts = np.unique(spike_ts_arr, return_counts=True)
                spike_counts = dict(zip(unique_ts, counts))
            elif "spikes" in neural:
                for i, ts in enumerate(neural.get("timesteps", [])):
                    spike_counts[ts] = int(np.sum(neural["spikes"][i]))

            timeline = []
            behavior_ts = behavior.get("timesteps", np.arange(len(behavior.get("pos_x", []))))

            for i, ts in enumerate(behavior_ts):
                ts = int(ts)
                step_data = {
                    "step": ts,
                    "position": [float(behavior["pos_x"][i]), float(behavior["pos_y"][i])],
                    "action": int(behavior["action"][i]),
                    "gradient": float(behavior.get("gradient", [0])[i]),
                    "reward": float(reward_map.get(ts, 0.0)),
                    "neural": None,
                }

                if ts in neural_map:
                    idx = neural_map[ts]
                    spike_count = int(spike_counts.get(ts, 0))
                    firing_rate = (spike_count / total_neurons) * 1000.0 if total_neurons > 0 else 0
                    
                    dopamine = neural.get("dopamine")
                    value_est = neural.get("value_estimate")

                    step_data["neural"] = {
                        "mean_potential": float(np.mean(neural["v"][idx])),
                        "spike_count": spike_count,
                        "firing_rate_hz": firing_rate,
                        "dopamine": float(dopamine[idx]) if dopamine is not None and idx < len(dopamine) else None,
                        "value_estimate": float(value_est[idx]) if value_est is not None and idx < len(value_est) else None,
                    }
                timeline.append(step_data)

            response = {
                "metadata": {
                    "experimentName": exp_meta.get("experiment_name", "N/A"),
                    "episodeId": episode_id,
                    "totalSteps": len(timeline),
                    "world": { "gridSize": world_cfg.get("grid_size", 100) },
                    "network": {"totalNeurons": total_neurons},
                },
                "worldSetup": {
                    "rewardPositions": world_setup.get("reward_positions", np.array([])).tolist()
                },
                "timeline": timeline,
                "summary": summary,
            }
            self.wfile.write(json.dumps(response, cls=NumpyEncoder).encode())

    def send_error_response(self, code, message):
        """Send a JSON error response."""
        try:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"error": message}).encode())
        except Exception as e:
            self.log_error(f"Failed to send error response for '{message}': {e}")


def main():
    """Run the API server."""
    global EXPERIMENT_CACHE
    
    # Pre-scan and cache experiments at startup
    EXPERIMENT_CACHE = scan_and_cache_experiments()
    print(f"Server ready. Found {EXPERIMENT_CACHE['count']} compatible runs.")

    port = 8000
    server_address = ("", port)
    httpd = HTTPServer(server_address, ScientificAPIHandler)
    print(f"SNN Scientific Visualization Server running at: http://localhost:{port}/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.shutdown()


if __name__ == "__main__":
    main()
