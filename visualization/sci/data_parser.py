# keywords: [visualization, data parser, experiment loader, hdf5, json, analytics]
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import h5py


def _get_nested(data, keys, default=None):
    """Safely get a nested value from a dict or list."""
    for key in keys:
        try:
            # Handle numeric indices for lists
            if isinstance(data, list) and isinstance(key, int):
                data = data[key]
            else:
                data = data[key]
        except (KeyError, TypeError, IndexError):
            return default
    return data


def parse_episode_data(episode_group: h5py.Group, episode_id: int) -> Dict[str, Any]:
    """Parses a single episode's data into a dict for JSON conversion."""
    # Load metadata
    metadata = dict(episode_group.attrs)
    
    # Load only what's needed for the playback
    data = {
        'episode_id': episode_id,
        'timesteps': episode_group['timesteps'][:].tolist() if 'timesteps' in episode_group else [],
        'rewards': episode_group['rewards'][:].tolist() if 'rewards' in episode_group else [],
        'actions': episode_group['actions'][:].tolist() if 'actions' in episode_group else [],
        'gradients': episode_group['gradients'][:].tolist() if 'gradients' in episode_group else [],
        'total_reward': metadata.get('total_reward', 0),
        'steps': metadata.get('steps', 0),
        'steps_per_second': metadata.get('steps_per_second', 0),
    }
    
    # Get neural states if available (use simple version for performance)
    if 'neural_states_simple' in episode_group:
        neural_states = episode_group['neural_states_simple'][:]
        # Sample every 10th timestep to reduce data size
        data['neural_states'] = neural_states[::10].tolist()
        data['neural_states_sampling'] = 10
    
    return data


def parse_experiment(exp_path: Path) -> Optional[Dict[str, Any]]:
    """Parses a full experiment directory."""
    print(f"Parsing {exp_path.name}...")
    
    try:
        # First check if required files exist
        data_file = exp_path / "experiment_data.h5"
        config_file = exp_path / "experiment_config.json"
        
        if not data_file.exists() or not config_file.exists():
            print(f"  [!] Missing required files in {exp_path.name}")
            return None
            
        # Load config
        with open(config_file) as f:
            config = json.load(f)
        
        # Open HDF5 file
        with h5py.File(data_file, 'r') as hf:
            # FIX: Load root attributes from HDF5
            root_attrs = dict(hf.attrs)
            
            # Load network structure
            network = None
            if 'network_structure' in hf:
                net_group = hf['network_structure']
                network = {
                    'neurons': {},
                    'connections': {}
                }
                
                if 'neurons' in net_group:
                    for key in net_group['neurons'].keys():
                        network['neurons'][key] = net_group['neurons'][key][:].tolist()
                
                if 'connections' in net_group:
                    for key in net_group['connections'].keys():
                        network['connections'][key] = net_group['connections'][key][:].tolist()
            
            # Load summary data
            summary_path = exp_path / "experiment_summary.json"
            if summary_path.exists():
                with open(summary_path) as f:
                    summary = json.load(f)
            else:
                # Create basic summary from episode data
                summary = {
                    'experiment_name': exp_path.name,
                    'agent_name': config.get('agent_name', 'unknown'),
                    'total_episodes': 0,
                    'reward_stats': {'mean': 0, 'std': 0, 'min': 0, 'max': 0},
                    'episode_stats': []
                }
            
            # Load data for a sample of episodes (to avoid memory issues)
            episodes_group = hf.get('episodes', {})
            episode_ids = list(range(len(episodes_group)))
            max_episodes_to_load = min(len(episode_ids), 10)  # Load up to 10 episodes
            
            episodes_data = {}
            for eid in episode_ids[:max_episodes_to_load]:
                episode_key = f"episode_{eid:04d}"
                if episode_key in episodes_group:
                    episodes_data[episode_key] = parse_episode_data(episodes_group[episode_key], eid)
            
            # Extract agent name - try HDF5 attributes first, then config
            agent_name = root_attrs.get('agent_name', config.get('agent_name', 'unknown'))
            if agent_name == 'unknown' and 'agent_version' in config:
                agent_name = config['agent_version']
            if agent_name == 'unknown' and 'model' in config:
                agent_name = config['model']
            if agent_name == 'unknown' and 'agent' in config:
                agent_name = config['agent']
            
            # Calculate metrics from episode data
            metrics = calculate_experiment_metrics({
                'summary': summary,
                'config': config
            })
            
            return {
                "name": exp_path.name,
                "agent": agent_name,
                "summary": summary,
                "config": config,
                "network": network,
                "episodes": episodes_data,
                "total_episodes": len(episode_ids),
                "metrics": metrics
            }
            
    except Exception as e:
        print(f"  [!] Failed to parse {exp_path.name}: {e}")
        return None


def get_all_experiment_data(base_dir: str = "experiments") -> List[Dict[str, Any]]:
    """Scans the base directory and parses all valid experiments."""
    exp_dir = Path(base_dir)
    if not exp_dir.exists():
        print(f"Experiments directory not found: {base_dir}")
        return []
    
    all_data = []
    for p in sorted(exp_dir.iterdir()):
        if p.is_dir() and not p.name.startswith('.'):
            exp_data = parse_experiment(p)
            if exp_data:
                all_data.append(exp_data)
    
    return all_data


def calculate_experiment_metrics(exp_data: Dict[str, Any]) -> Dict[str, float]:
    """Calculate aggregate metrics for an experiment."""
    metrics = {
        'avg_reward': 0,
        'std_reward': 0,
        'avg_steps_per_second': 0,
        'total_duration': 0,
        'learning_progress': 0
    }
    
    if 'episode_stats' in exp_data['summary']:
        stats = exp_data['summary']['episode_stats']
        if stats:
            rewards = [s.get('total_reward', 0) for s in stats]
            steps_per_sec = [s.get('steps_per_second', 0) for s in stats if s.get('steps_per_second', 0) > 0]
            
            if rewards:
                metrics['avg_reward'] = np.mean(rewards)
                metrics['std_reward'] = np.std(rewards)
                
                # Learning progress: difference between last 25% and first 25% of episodes
                quarter = max(1, len(rewards) // 4)
                early_rewards = rewards[:quarter]
                late_rewards = rewards[-quarter:]
                metrics['learning_progress'] = np.mean(late_rewards) - np.mean(early_rewards)
            
            if steps_per_sec:
                metrics['avg_steps_per_second'] = np.mean(steps_per_sec)
    
    return metrics