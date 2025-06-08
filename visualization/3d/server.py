"""
keywords: [visualization, server, api, fastapi, websocket, hdf5, async, streaming, performance]

High-performance async API server for SNN agent visualization.
Provides efficient data streaming, caching, and real-time updates.
"""

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

import h5py
import numpy as np
from fastapi import FastAPI, WebSocket, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ConfigDict
import msgpack
import lz4.frame

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
CACHE_SIZE = 100  # Maximum number of cached experiments
CHUNK_SIZE = 1000  # Data points per chunk for streaming
DECIMATION_THRESHOLD = 10000  # Apply decimation for arrays larger than this
WS_HEARTBEAT_INTERVAL = 30  # WebSocket heartbeat interval in seconds


class ExperimentInfo(BaseModel):
    """Basic experiment metadata"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    experiment_id: str
    timestamp: datetime
    num_episodes: int
    config: Dict[str, Any]
    grid_size: Tuple[int, int]
    num_neurons: int
    duration: float


class DataRequest(BaseModel):
    """Request model for data queries"""
    experiment_id: str
    episode_id: int
    data_type: str = Field(..., pattern="^(trajectory|neural|rewards|metrics)$")
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    decimation_factor: Optional[int] = 1
    
    
@dataclass
class CachedExperiment:
    """Cached experiment data with TTL"""
    data: h5py.File
    last_access: float
    access_count: int = 0
    
    
class DataCache:
    """LRU cache for experiment data with automatic cleanup"""
    
    def __init__(self, max_size: int = CACHE_SIZE):
        self.cache: Dict[str, CachedExperiment] = {}
        self.max_size = max_size
        self._lock = asyncio.Lock()
        
    async def get(self, path: Path) -> h5py.File:
        """Get cached experiment or load from disk"""
        key = str(path)
        
        async with self._lock:
            if key in self.cache:
                entry = self.cache[key]
                entry.last_access = time.time()
                entry.access_count += 1
                return entry.data
                
            # Load new experiment
            if len(self.cache) >= self.max_size:
                await self._evict_lru()
                
            try:
                data = h5py.File(path, 'r')
                self.cache[key] = CachedExperiment(
                    data=data,
                    last_access=time.time()
                )
                return data
            except Exception as e:
                logger.error(f"Failed to load {path}: {e}")
                raise HTTPException(status_code=404, detail=f"Experiment not found: {e}")
                
    async def _evict_lru(self):
        """Evict least recently used entry"""
        if not self.cache:
            return
            
        lru_key = min(self.cache.keys(), 
                     key=lambda k: self.cache[k].last_access)
        entry = self.cache.pop(lru_key)
        entry.data.close()
        logger.info(f"Evicted {lru_key} from cache")
        
    async def clear(self):
        """Clear all cached data"""
        async with self._lock:
            for entry in self.cache.values():
                entry.data.close()
            self.cache.clear()


class DataProcessor:
    """Efficient data processing and decimation"""
    
    @staticmethod
    def decimate_timeseries(data: np.ndarray, factor: int) -> np.ndarray:
        """Decimate time series data using max-min preservation"""
        if factor <= 1 or len(data) < factor * 2:
            return data
            
        # Reshape for efficient processing
        n_chunks = len(data) // factor
        truncated = data[:n_chunks * factor]
        reshaped = truncated.reshape(n_chunks, factor, -1)
        
        # Preserve both max and min values in each chunk
        max_vals = reshaped.max(axis=1)
        min_vals = reshaped.min(axis=1)
        
        # Interleave max and min for better visualization
        result = np.empty((n_chunks * 2, *max_vals.shape[1:]), dtype=data.dtype)
        result[0::2] = min_vals
        result[1::2] = max_vals
        
        return result
        
    @staticmethod
    def compute_spike_rates(spikes: np.ndarray, window_ms: float = 50.0) -> np.ndarray:
        """Compute instantaneous firing rates from spike data"""
        dt = 1.0  # 1ms time step
        window_steps = int(window_ms / dt)
        
        # Convolve with rectangular window
        kernel = np.ones(window_steps) / (window_ms / 1000.0)
        rates = np.apply_along_axis(
            lambda x: np.convolve(x, kernel, mode='same'),
            axis=0, arr=spikes
        )
        
        return rates
        
    @staticmethod
    def extract_trajectory_segments(positions: np.ndarray, 
                                  rewards: np.ndarray,
                                  segment_length: int = 100) -> List[Dict]:
        """Extract trajectory segments around reward collections"""
        segments = []
        reward_indices = np.where(rewards > 0)[0]
        
        for idx in reward_indices:
            start = max(0, idx - segment_length // 2)
            end = min(len(positions), idx + segment_length // 2)
            
            segments.append({
                "reward_time": idx,
                "segment": positions[start:end].tolist(),
                "reward_value": float(rewards[idx])
            })
            
        return segments


# Initialize cache
cache = DataCache()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting visualization server...")
    yield
    logger.info("Shutting down...")
    await cache.clear()


# Create FastAPI app
app = FastAPI(
    title="SNN Agent Visualization API",
    version="2.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
from fastapi.staticfiles import StaticFiles
import os
static_dir = Path(__file__).parent
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/api/experiments")
async def list_experiments() -> List[ExperimentInfo]:
    """List all available experiments with metadata"""
    base_path = Path(__file__).parent.parent.parent / "experiments"
    experiments = []
    
    logger.info(f"Looking for experiments in: {base_path}")
    
    for phase_dir in sorted(base_path.glob("phase_*")):
        for exp_dir in sorted(phase_dir.glob("snn_agent_*")):
            exp_file = exp_dir / "experiment_data.h5"
            if not exp_file.exists():
                continue
                
            try:
                # Quick metadata extraction
                with h5py.File(exp_file, 'r') as f:
                    # Get config from group or attributes
                    config = {}
                    if 'config' in f:
                        # Config is a group, extract its attributes
                        config = dict(f['config'].attrs)
                    elif 'config' in f.attrs:
                        # Config is in attributes
                        try:
                            config = json.loads(f.attrs['config'])
                        except:
                            pass
                    
                    # Get number of episodes
                    num_episodes = 0
                    if 'episodes' in f:
                        episodes_group = f['episodes']
                        num_episodes = len([k for k in episodes_group.keys() if k.startswith('episode_')])
                    
                    # Calculate total duration
                    duration = 0
                    if 'episodes' in f:
                        for ep_key in f['episodes'].keys():
                            if ep_key.startswith('episode_'):
                                if 'behavior' in f['episodes'][ep_key]:
                                    behavior = f['episodes'][ep_key]['behavior']
                                    if 'timesteps' in behavior:
                                        duration += len(behavior['timesteps'][:])
                    
                    # Extract timestamp from directory name
                    timestamp_str = exp_dir.name.split('_')[-1]
                    try:
                        # Convert YYYYMMDD_HHMMSS to datetime object
                        timestamp = datetime.strptime(timestamp_str, '%Y%m%d')
                    except:
                        # Try with time included
                        try:
                            # Get last two parts for date and time
                            parts = exp_dir.name.split('_')
                            if len(parts) >= 2:
                                date_time_str = parts[-2] + parts[-1]
                                timestamp = datetime.strptime(date_time_str, '%Y%m%d%H%M%S')
                            else:
                                timestamp = datetime.now()
                        except:
                            timestamp = datetime.now()
                    
                    info = ExperimentInfo(
                        experiment_id=exp_dir.name,
                        timestamp=timestamp,
                        num_episodes=num_episodes,
                        config=config,
                        grid_size=tuple(config.get('grid_size', [10, 10])),
                        num_neurons=attrs.get('n_neurons', config.get('num_neurons', 1000)),
                        duration=float(duration)
                    )
                    experiments.append(info)
                    
            except Exception as e:
                logger.warning(f"Failed to load {exp_file}: {e}")
                
    return experiments


@app.get("/api/experiment/{experiment_id}/episode/{episode_id}/trajectory")
async def get_trajectory(
    experiment_id: str,
    episode_id: int,
    decimation: int = Query(1, ge=1, le=100)
) -> JSONResponse:
    """Get trajectory data for a specific episode"""
    # Find the experiment directory
    base_path = Path(__file__).parent.parent.parent / "experiments"
    exp_path = None
    
    # Search for the experiment in all phase directories
    for phase_dir in base_path.glob("phase_*"):
        candidate = phase_dir / experiment_id / "experiment_data.h5"
        if candidate.exists():
            exp_path = candidate
            break
    
    if not exp_path:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
    
    data_file = await cache.get(exp_path)
    episode_key = f"episode_{episode_id:04d}"
    
    # Navigate to the correct location in the HDF5 structure
    if 'episodes' not in data_file:
        raise HTTPException(status_code=404, detail="No episodes found in experiment")
        
    episodes_group = data_file['episodes']
    if episode_key not in episodes_group:
        raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found")
        
    episode = episodes_group[episode_key]
    
    # Extract trajectory data from behavior group
    if 'behavior' not in episode:
        raise HTTPException(status_code=404, detail="No behavior data found")
        
    behavior = episode['behavior']
    
    # Get position data
    x = behavior['pos_x'][:] if 'pos_x' in behavior else np.array([])
    y = behavior['pos_y'][:] if 'pos_y' in behavior else np.array([])
    
    # Apply decimation if needed
    if decimation > 1 and len(x) > DECIMATION_THRESHOLD:
        indices = np.arange(0, len(x), decimation)
        x = x[indices]
        y = y[indices]
        
    # Extract rewards from events
    rewards = np.zeros(len(x))
    if 'events' in episode:
        events_group = episode['events']
        for event_key in events_group.keys():
            if 'reward_collected' in event_key:
                event = events_group[event_key]
                timestep = event.attrs.get('timestep', -1)
                value = event.attrs.get('value', 1.0)
                if 0 <= timestep < len(rewards):
                    rewards[timestep] = value
    
    # Values not present in current data format, use zeros
    values = np.zeros(len(x))
    
    return JSONResponse({
        "trajectory": {
            "x": x.tolist(),
            "y": y.tolist(),
            "rewards": rewards.tolist(),
            "values": values.tolist()
        },
        "metadata": {
            "total_reward": float(rewards.sum()),
            "episode_length": len(x),
            "decimation_applied": decimation if len(x) > DECIMATION_THRESHOLD else 1
        }
    })


@app.websocket("/ws/stream/{experiment_id}")
async def websocket_stream(websocket: WebSocket, experiment_id: str):
    """WebSocket endpoint for real-time data streaming"""
    await websocket.accept()
    
    try:
        # Find the experiment path
        base_path = Path(__file__).parent.parent.parent / "experiments"
        exp_path = None
        
        for phase_dir in base_path.glob("phase_*"):
            candidate = phase_dir / experiment_id / "experiment_data.h5"
            if candidate.exists():
                exp_path = candidate
                break
        
        if not exp_path:
            await websocket.send_json({
                "type": "error",
                "message": f"Experiment {experiment_id} not found"
            })
            await websocket.close()
            return
            
        data_file = await cache.get(exp_path)
        
        # Get episode list
        episodes = []
        if 'episodes' in data_file:
            episodes = [k for k in data_file['episodes'].keys() if k.startswith('episode_')]
        
        # Send initial metadata
        await websocket.send_json({
            "type": "metadata",
            "data": {
                "experiment_id": experiment_id,
                "episodes": episodes
            }
        })
        
        # Handle client requests
        while True:
            try:
                message = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=WS_HEARTBEAT_INTERVAL
                )
                
                if message["type"] == "request_data":
                    await handle_data_request(websocket, data_file, message["data"])
                elif message["type"] == "ping":
                    await websocket.send_json({"type": "pong"})
                    
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({"type": "heartbeat"})
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()


async def handle_data_request(websocket: WebSocket, 
                            data_file: h5py.File, 
                            request: Dict[str, Any]):
    """Handle specific data request over WebSocket"""
    episode_key = f"episode_{request['episode_id']:04d}"
    data_type = request['data_type']
    
    # Navigate to episodes group
    if 'episodes' not in data_file:
        await websocket.send_json({
            "type": "error",
            "message": "No episodes found in experiment"
        })
        return
        
    episodes_group = data_file['episodes']
    if episode_key not in episodes_group:
        await websocket.send_json({
            "type": "error",
            "message": f"Episode not found: {episode_key}"
        })
        return
        
    episode = episodes_group[episode_key]
    processor = DataProcessor()
    
    if data_type == "neural":
        # Stream neural data in chunks
        if 'neural_states' in episode and 'spikes' in episode['neural_states']:
            spikes = episode['neural_states']['spikes'][:]
            spike_rates = processor.compute_spike_rates(spikes)
            
            # Send in chunks
            for i in range(0, len(spike_rates), CHUNK_SIZE):
                chunk = spike_rates[i:i + CHUNK_SIZE]
                
                # Send data without LZ4 compression for now
                data = msgpack.packb({
                    "time_start": i,
                    "time_end": i + len(chunk),
                    "data": chunk.tolist()
                })
                
                await websocket.send_bytes(data)
                await asyncio.sleep(0.01)  # Prevent overwhelming client
        else:
            await websocket.send_json({
                "type": "error",
                "message": "No neural data found"
            })
            
    elif data_type == "trajectory":
        # Send trajectory segments
        if 'behavior' in episode:
            behavior = episode['behavior']
            if 'pos_x' in behavior and 'pos_y' in behavior:
                positions = np.column_stack([
                    behavior['pos_x'][:],
                    behavior['pos_y'][:]
                ])
                
                # Extract rewards from events
                rewards = np.zeros(len(positions))
                if 'events' in episode:
                    events_group = episode['events']
                    for event_key in events_group.keys():
                        if 'reward_collected' in event_key:
                            event = events_group[event_key]
                            timestep = event.attrs.get('timestep', -1)
                            value = event.attrs.get('value', 1.0)
                            if 0 <= timestep < len(rewards):
                                rewards[timestep] = value
                
                segments = processor.extract_trajectory_segments(positions, rewards)
                
                await websocket.send_json({
                    "type": "trajectory_segments",
                    "data": segments
                })
            else:
                await websocket.send_json({
                    "type": "error",
                    "message": "No position data found"
                })
        else:
            await websocket.send_json({
                "type": "error",
                "message": "No behavior data found"
            })


@app.get("/api/experiment/{experiment_id}/analysis")
async def get_analysis(experiment_id: str) -> JSONResponse:
    """Get pre-computed analysis for an experiment"""
    # Find the experiment path
    base_path = Path(__file__).parent.parent.parent / "experiments"
    exp_path = None
    
    for phase_dir in base_path.glob("phase_*"):
        candidate = phase_dir / experiment_id / "experiment_data.h5"
        if candidate.exists():
            exp_path = candidate
            break
    
    if not exp_path:
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
        
    data_file = await cache.get(exp_path)
    
    analysis = {
        "episodes": [],
        "aggregate": {
            "total_episodes": 0,
            "avg_reward": 0,
            "avg_episode_length": 0,
            "learning_curve": []
        }
    }
    
    total_rewards = []
    episode_lengths = []
    
    if 'episodes' not in data_file:
        return JSONResponse(analysis)
        
    episodes_group = data_file['episodes']
    
    for episode_key in sorted(episodes_group.keys()):
        if not episode_key.startswith('episode_'):
            continue
            
        episode = episodes_group[episode_key]
        
        # Get episode length from behavior data
        episode_length = 0
        if 'behavior' in episode and 'timesteps' in episode['behavior']:
            episode_length = len(episode['behavior']['timesteps'][:])
        
        # Extract rewards from events
        total_reward = 0
        reward_count = 0
        if 'events' in episode:
            events_group = episode['events']
            for event_key in events_group.keys():
                if 'reward_collected' in event_key:
                    event = events_group[event_key]
                    value = event.attrs.get('value', 1.0)
                    total_reward += value
                    reward_count += 1
        
        ep_analysis = {
            "episode_id": int(episode_key.split('_')[1]),
            "total_reward": float(total_reward),
            "episode_length": episode_length,
            "reward_rate": float(total_reward / episode_length) if episode_length > 0 else 0,
            "reward_count": reward_count
        }
        
        analysis["episodes"].append(ep_analysis)
        total_rewards.append(ep_analysis["total_reward"])
        episode_lengths.append(ep_analysis["episode_length"])
        
    # Compute aggregates
    if len(analysis["episodes"]) > 0:
        analysis["aggregate"]["total_episodes"] = len(analysis["episodes"])
        analysis["aggregate"]["avg_reward"] = float(np.mean(total_rewards))
        analysis["aggregate"]["avg_episode_length"] = float(np.mean(episode_lengths))
        
        # Learning curve (cumulative average)
        cumsum = np.cumsum(total_rewards)
        counts = np.arange(1, len(total_rewards) + 1)
        analysis["aggregate"]["learning_curve"] = (cumsum / counts).tolist()
    
    return JSONResponse(analysis)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)