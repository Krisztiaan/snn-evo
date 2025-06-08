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
from pydantic import BaseModel, Field
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


@app.get("/api/experiments")
async def list_experiments() -> List[ExperimentInfo]:
    """List all available experiments with metadata"""
    base_path = Path("../../experiments")
    experiments = []
    
    for phase_dir in sorted(base_path.glob("phase_*")):
        for exp_dir in sorted(phase_dir.glob("snn_agent_*")):
            exp_file = exp_dir / "experiment_data.h5"
            if not exp_file.exists():
                continue
                
            try:
                # Quick metadata extraction
                with h5py.File(exp_file, 'r') as f:
                    config = json.loads(f.attrs.get('config', '{}'))
                    
                    info = ExperimentInfo(
                        experiment_id=exp_dir.name,
                        timestamp=datetime.fromisoformat(
                            exp_dir.name.split('_')[-1].replace('_', 'T')
                        ),
                        num_episodes=len(f.keys()),
                        config=config,
                        grid_size=tuple(config.get('grid_size', [10, 10])),
                        num_neurons=config.get('num_neurons', 1000),
                        duration=sum(len(f[ep]['trajectory/x'][:]) 
                                   for ep in f.keys() if ep.startswith('episode_'))
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
    exp_path = Path(f"../../experiments/{experiment_id}/experiment_data.h5")
    
    data_file = await cache.get(exp_path)
    episode_key = f"episode_{episode_id}"
    
    if episode_key not in data_file:
        raise HTTPException(status_code=404, detail=f"Episode {episode_id} not found")
        
    episode = data_file[episode_key]
    
    # Extract trajectory data
    x = episode['trajectory/x'][:]
    y = episode['trajectory/y'][:]
    
    # Apply decimation if needed
    if decimation > 1 and len(x) > DECIMATION_THRESHOLD:
        indices = np.arange(0, len(x), decimation)
        x = x[indices]
        y = y[indices]
        
    # Get rewards and values
    rewards = episode.get('rewards', np.zeros(len(x)))[:]
    values = episode.get('values', np.zeros(len(x)))[:]
    
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
        exp_path = Path(f"../../experiments/{experiment_id}/experiment_data.h5")
        data_file = await cache.get(exp_path)
        
        # Send initial metadata
        await websocket.send_json({
            "type": "metadata",
            "data": {
                "experiment_id": experiment_id,
                "episodes": list(data_file.keys())
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
    episode_key = f"episode_{request['episode_id']}"
    data_type = request['data_type']
    
    if episode_key not in data_file:
        await websocket.send_json({
            "type": "error",
            "message": f"Episode not found: {episode_key}"
        })
        return
        
    episode = data_file[episode_key]
    processor = DataProcessor()
    
    if data_type == "neural":
        # Stream neural data in chunks
        spikes = episode['neural/spikes'][:]
        spike_rates = processor.compute_spike_rates(spikes)
        
        # Send in chunks
        for i in range(0, len(spike_rates), CHUNK_SIZE):
            chunk = spike_rates[i:i + CHUNK_SIZE]
            
            # Compress data
            compressed = lz4.frame.compress(
                msgpack.packb({
                    "time_start": i,
                    "time_end": i + len(chunk),
                    "data": chunk.tolist()
                })
            )
            
            await websocket.send_bytes(compressed)
            await asyncio.sleep(0.01)  # Prevent overwhelming client
            
    elif data_type == "trajectory":
        # Send trajectory segments
        positions = np.column_stack([
            episode['trajectory/x'][:],
            episode['trajectory/y'][:]
        ])
        rewards = episode.get('rewards', np.zeros(len(positions)))[:]
        
        segments = processor.extract_trajectory_segments(positions, rewards)
        
        await websocket.send_json({
            "type": "trajectory_segments",
            "data": segments
        })


@app.get("/api/experiment/{experiment_id}/analysis")
async def get_analysis(experiment_id: str) -> JSONResponse:
    """Get pre-computed analysis for an experiment"""
    exp_path = Path(f"../../experiments/{experiment_id}/experiment_data.h5")
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
    
    for episode_key in sorted(data_file.keys()):
        if not episode_key.startswith('episode_'):
            continue
            
        episode = data_file[episode_key]
        rewards = episode.get('rewards', np.array([]))[:]
        
        ep_analysis = {
            "episode_id": int(episode_key.split('_')[1]),
            "total_reward": float(rewards.sum()),
            "episode_length": len(rewards),
            "reward_rate": float(rewards.sum() / len(rewards)) if len(rewards) > 0 else 0
        }
        
        analysis["episodes"].append(ep_analysis)
        total_rewards.append(ep_analysis["total_reward"])
        episode_lengths.append(ep_analysis["episode_length"])
        
    # Compute aggregates
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