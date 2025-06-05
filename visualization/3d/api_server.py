#!/usr/bin/env python3
# keywords: [api, server, hdf5, experiment loader, visualization]
"""API server for HDF5 experiment data using ExperimentLoader."""

import json
import numpy as np
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import sys
import os

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from export.loader import ExperimentLoader, EpisodeData


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, 'item'):
            return obj.item()
        return super().default(obj)


class ExperimentAPIHandler(BaseHTTPRequestHandler):
    """Handle API requests for experiment data."""
    
    def do_GET(self):
        """Handle GET requests."""
        parsed_path = urlparse(self.path)
        
        # CORS headers
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        try:
            if parsed_path.path == '/api/experiments':
                self.handle_list_experiments()
            elif parsed_path.path == '/api/trajectory':
                params = parse_qs(parsed_path.query)
                path = params.get('path', [''])[0]
                episode = int(params.get('episode', [0])[0])
                self.handle_get_trajectory(path, episode)
            elif parsed_path.path == '/api/metadata':
                params = parse_qs(parsed_path.query)
                path = params.get('path', [''])[0]
                self.handle_get_metadata(path)
            else:
                self.send_error_response(404, "Endpoint not found")
        except Exception as e:
            self.send_error_response(500, str(e))
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def handle_list_experiments(self):
        """List all available experiments."""
        experiments = []
        
        # Search for experiment directories
        # Only look in proper experiment output directories
        base_dirs = [
            Path("models/random/logs"),
            Path("experiments/models/random/logs"),
            # Add other model types as needed:
            # Path("models/snn/logs"),
            # Path("experiments/models/snn/logs"),
        ]
        
        for base_dir in base_dirs:
            if not base_dir.exists():
                continue
                
            for h5_path in base_dir.rglob("*.h5"):
                try:
                    exp_info = None
                    
                    # Check if it's minimal format
                    if self.is_minimal_format(h5_path):
                        # Handle minimal format
                        with h5py.File(h5_path, 'r') as f:
                            config = {}
                            if 'config' in f:
                                for key, value in f['config'].attrs.items():
                                    if isinstance(value, str) and value.startswith('{'):
                                        config[key] = json.loads(value)
                                    else:
                                        config[key] = value
                            
                            summary = dict(f['summary'].attrs) if 'summary' in f else {}
                            
                            exp_info = {
                                'path': str(h5_path),
                                'directory': h5_path.parent.name,
                                'metadata': {},
                                'config': config,
                                'episodes': [0],  # Minimal format has single episode
                                'timestamp': h5_path.stat().st_mtime,
                                'format': 'minimal'
                            }
                            
                            # Create label
                            if 'world_config' in config:
                                wc = config['world_config']
                                rewards_info = f"{summary.get('rewards_collected', 0)}/{wc.get('n_rewards', '?')} rewards - {summary.get('total_reward', 0):.1f} score"
                                exp_info['label'] = f"{h5_path.parent.name} - Grid {wc.get('grid_size', '?')}x{wc.get('grid_size', '?')} - {rewards_info}"
                            else:
                                exp_info['label'] = f"{h5_path.parent.name} - Minimal format"
                    else:
                        # Try standard ExperimentLoader format
                        loader = ExperimentLoader(h5_path.parent)
                        metadata = loader.get_metadata()
                        config = loader.get_config()
                        episodes = loader.list_episodes()
                        
                        exp_info = {
                            'path': str(h5_path),
                            'directory': h5_path.parent.name,
                            'metadata': metadata,
                            'config': config,
                            'episodes': episodes,
                            'timestamp': h5_path.stat().st_mtime,
                            'format': 'standard'
                        }
                        
                        # Create label
                        if 'world_config' in config:
                            wc = config['world_config']
                            exp_info['label'] = f"{h5_path.parent.name} - Grid {wc.get('grid_size', '?')}x{wc.get('grid_size', '?')} - {len(episodes)} episodes"
                        else:
                            exp_info['label'] = f"{h5_path.parent.name} - {len(episodes)} episodes"
                        
                        loader.close()
                    
                    if exp_info:
                        experiments.append(exp_info)
                    
                except Exception as e:
                    print(f"Error loading {h5_path}: {e}")
        
        # Sort by timestamp
        experiments.sort(key=lambda x: x['timestamp'], reverse=True)
        
        response = {
            'experiments': experiments,
            'count': len(experiments)
        }
        
        self.wfile.write(json.dumps(response, cls=NumpyEncoder).encode())
    
    def handle_get_trajectory(self, path, episode_id):
        """Get trajectory data for visualization."""
        if not path:
            self.send_error_response(400, "Path parameter required")
            return
        
        try:
            # Load experiment
            exp_path = Path(path)
            
            # Security check: ensure path is within allowed directories
            allowed_parents = ['logs', 'experiments', 'runs', 'outputs']
            if not any(parent in exp_path.parts for parent in allowed_parents):
                self.send_error_response(403, "Access denied: Path must be within experiment directories")
                return
            
            # Check if it's a minimal exporter format
            if self.is_minimal_format(exp_path):
                return self.handle_minimal_trajectory(exp_path)
            
            # Standard ExperimentLoader format
            if exp_path.name.endswith('.h5'):
                exp_dir = exp_path.parent
            else:
                exp_dir = exp_path
                
            loader = ExperimentLoader(exp_dir)
            
            # Get config and episode data
            config = loader.get_config()
            episode_data = loader.get_episode(episode_id)
            
            # Build visualization data
            viz_data = {
                'metadata': {
                    'gridSize': config.get('world_config', {}).get('grid_size', 10),
                    'nRewards': config.get('world_config', {}).get('n_rewards', 5),
                    'seed': config.get('seed', -1),
                    'episodeId': episode_id
                },
                'trajectory': [],
                'world': {}
            }
            
            # Get behavior data
            positions = episode_data.get_positions()
            actions = episode_data.get_actions()
            
            # Get rewards data
            reward_positions = episode_data.get_reward_positions()
            rewards_collected = episode_data.get_rewards_collected()
            reward_values = episode_data.get_reward_values()
            
            # Get observations
            observations = episode_data.get_custom_data('behavior', 'observations')
            if observations is None:
                observations = episode_data.get_custom_data('behavior', 'gradients')
            
            # Build trajectory
            cumulative_reward = 0.0
            rewards_collected_so_far = np.zeros(len(reward_positions), dtype=bool)
            
            for i in range(len(positions)):
                step_data = {
                    'step': i,
                    'agentPos': positions[i].tolist() if hasattr(positions[i], 'tolist') else list(positions[i]),
                    'observation': float(observations[i]) if observations is not None and i < len(observations) else 0.0,
                    'cumulativeReward': cumulative_reward
                }
                
                if i > 0 and actions is not None and i - 1 < len(actions):
                    step_data['action'] = int(actions[i - 1])
                    
                    if reward_values is not None and i - 1 < len(reward_values):
                        step_reward = float(reward_values[i - 1])
                        step_data['reward'] = step_reward
                        cumulative_reward += step_reward
                
                # Update rewards collected
                if rewards_collected is not None:
                    for j, (collected, when) in enumerate(rewards_collected):
                        if collected and when <= i:
                            rewards_collected_so_far[j] = True
                
                step_data['rewardCollected'] = rewards_collected_so_far.tolist()
                viz_data['trajectory'].append(step_data)
            
            # Set final metadata
            viz_data['metadata']['totalSteps'] = len(positions) - 1
            viz_data['metadata']['totalReward'] = cumulative_reward
            viz_data['metadata']['rewardsCollected'] = int(np.sum(rewards_collected_so_far))
            
            # Calculate coverage
            unique_positions = len(set(tuple(p) for p in positions))
            grid_area = viz_data['metadata']['gridSize'] ** 2
            viz_data['metadata']['coverage'] = unique_positions / grid_area
            
            # Set world data
            viz_data['world']['rewardPositions'] = [[int(x), int(y)] for x, y in reward_positions]
            
            loader.close()
            
            self.wfile.write(json.dumps(viz_data, cls=NumpyEncoder).encode())
            
        except Exception as e:
            self.send_error_response(500, f"Error loading trajectory: {str(e)}")
    
    def handle_get_metadata(self, path):
        """Get experiment metadata."""
        if not path:
            self.send_error_response(400, "Path parameter required")
            return
            
        try:
            exp_path = Path(path)
            if exp_path.name.endswith('.h5'):
                exp_dir = exp_path.parent
            else:
                exp_dir = exp_path
                
            loader = ExperimentLoader(exp_dir)
            
            response = {
                'metadata': loader.get_metadata(),
                'config': loader.get_config(),
                'runtime': loader.get_runtime_info(),
                'episodes': loader.list_episodes()
            }
            
            loader.close()
            
            self.wfile.write(json.dumps(response, cls=NumpyEncoder).encode())
            
        except Exception as e:
            self.send_error_response(500, f"Error loading metadata: {str(e)}")
    
    def is_minimal_format(self, h5_path):
        """Check if H5 file is in minimal exporter format."""
        if isinstance(h5_path, Path) and h5_path.suffix == '.h5' and h5_path.exists():
            with h5py.File(h5_path, 'r') as f:
                # Minimal format has 'trajectory' group at root
                return 'trajectory' in f and 'episodes' not in f
        return False
    
    def handle_minimal_trajectory(self, h5_path):
        """Handle trajectory data from minimal exporter format."""
        with h5py.File(h5_path, 'r') as f:
            # Get config
            config = {}
            if 'config' in f:
                for key, value in f['config'].attrs.items():
                    if isinstance(value, str) and value.startswith('{'):
                        config[key] = json.loads(value)
                    else:
                        config[key] = value
            
            # Get trajectory data
            traj = f['trajectory']
            positions = traj['positions'][:]
            actions = traj['actions'][:]
            rewards = traj['rewards'][:]
            observations = traj['observations'][:]
            
            # Get final state
            final_state = f['final_state']
            reward_positions = final_state['reward_positions'][:]
            reward_collected = final_state['reward_collected'][:]
            
            # Get summary
            summary = dict(f['summary'].attrs)
            
            # Build visualization data
            viz_data = {
                'metadata': {
                    'gridSize': config.get('world_config', {}).get('grid_size', 10),
                    'nRewards': config.get('world_config', {}).get('n_rewards', 5),
                    'totalSteps': len(positions) - 1,
                    'totalReward': float(summary.get('total_reward', 0)),
                    'rewardsCollected': int(summary.get('rewards_collected', 0)),
                    'coverage': float(summary.get('coverage', 0)),
                    'seed': config.get('seed', -1),
                    'episodeId': 0
                },
                'trajectory': [],
                'world': {
                    'rewardPositions': [[int(x), int(y)] for x, y in reward_positions]
                }
            }
            
            # Build trajectory
            cumulative_reward = 0.0
            rewards_collected_so_far = np.zeros(len(reward_positions), dtype=bool)
            
            for i in range(len(positions)):
                step_data = {
                    'step': i,
                    'agentPos': positions[i].tolist(),
                    'observation': float(observations[i]),
                    'cumulativeReward': cumulative_reward
                }
                
                if i > 0:
                    step_data['action'] = int(actions[i - 1])
                    step_data['reward'] = float(rewards[i - 1])
                    cumulative_reward += rewards[i - 1]
                    
                    # Check if reward was collected
                    if rewards[i - 1] >= config.get('world_config', {}).get('reward_value', 10):
                        # Find which reward was collected
                        agent_pos = positions[i]
                        for j, reward_pos in enumerate(reward_positions):
                            if not rewards_collected_so_far[j]:
                                dist = np.linalg.norm(agent_pos - reward_pos)
                                if dist < 0.5:
                                    rewards_collected_so_far[j] = True
                                    break
                
                step_data['rewardCollected'] = rewards_collected_so_far.tolist()
                viz_data['trajectory'].append(step_data)
            
            self.wfile.write(json.dumps(viz_data, cls=NumpyEncoder).encode())
    
    def send_error_response(self, code, message):
        """Send error response."""
        error_data = {
            'error': True,
            'code': code,
            'message': message
        }
        self.wfile.write(json.dumps(error_data).encode())
    
    def log_message(self, format, *args):
        """Override to reduce logging."""
        if '/api/' in args[0]:
            return  # Don't log API requests
        super().log_message(format, *args)


class ExperimentServer(HTTPServer):
    """Custom server with static file serving."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.static_dir = Path(__file__).parent


def main():
    """Run the API server."""
    port = 8000
    server_address = ('', port)
    
    # Create combined handler
    class CombinedHandler(ExperimentAPIHandler):
        def do_GET(self):
            """Handle both API and static file requests."""
            if self.path.startswith('/api/'):
                super().do_GET()
            else:
                # Serve static files
                self.serve_static_file()
        
        def serve_static_file(self):
            """Serve static files from visualization directory."""
            # Default to index
            path = self.path
            if path == '/':
                path = '/grid_world_trajectory_viewer.html'
            
            # Remove query string
            path = path.split('?')[0]
            
            # Security: prevent directory traversal
            path = path.replace('..', '')
            
            # Build file path
            file_path = Path(__file__).parent / path.lstrip('/')
            
            if file_path.exists() and file_path.is_file():
                # Determine content type
                content_type = 'text/html'
                if path.endswith('.js'):
                    content_type = 'application/javascript'
                elif path.endswith('.json'):
                    content_type = 'application/json'
                elif path.endswith('.css'):
                    content_type = 'text/css'
                
                self.send_response(200)
                self.send_header('Content-Type', content_type)
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                
                with open(file_path, 'rb') as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b'File not found')
    
    httpd = ExperimentServer(server_address, CombinedHandler)
    
    print(f"Experiment Visualization Server")
    print(f"===============================")
    print(f"Server running at: http://localhost:{port}/")
    print(f"API endpoints:")
    print(f"  - GET /api/experiments - List all experiments")
    print(f"  - GET /api/trajectory?path=<path>&episode=<id> - Get trajectory data")
    print(f"  - GET /api/metadata?path=<path> - Get experiment metadata")
    print(f"\nPress Ctrl+C to stop")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        httpd.shutdown()


if __name__ == "__main__":
    main()