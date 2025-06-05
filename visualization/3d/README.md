# Grid World 3D Trajectory Visualization

Interactive 3D visualization for viewing agent trajectories in grid world experiments, now with direct HDF5 support through the ExperimentLoader module.

## Features

- **Direct HDF5 Loading**: Uses the standardized `ExperimentLoader` to read experiment data
- **3D Grid World**: Clean visualization with rounded agent and spherical rewards
- **Experiment Browser**: Browse and select from all available experiments
- **Episode Selection**: Load specific episodes from multi-episode experiments
- **Trajectory Playback**: Step through or play episodes with adjustable speed
- **Trail Visualization**: See the agent's path through the environment
- **Interactive Camera**: Full 3D navigation with mouse controls
- **Real-time Metrics**: Track position, rewards, gradient signal, and coverage

## Quick Start

```bash
# Single command to start everything
python visualization/3d/run_visualization.py
```

This will:
1. Start the API server
2. Open the visualization in your browser
3. Load all available experiments

## Architecture

The new system creates a seamless pipeline:
```
Experiment (HDF5) → ExperimentLoader → API Server → Browser Visualization
```

### Components

1. **`api_server.py`**: Python server that:
   - Uses `ExperimentLoader` to read HDF5 files
   - Provides REST API endpoints for experiments and trajectories
   - Serves the HTML visualization

2. **`grid_world_h5_viewer.html`**: Browser-based 3D visualization
   - Fetches experiment data via API
   - Renders trajectories with Three.js
   - Provides interactive controls

3. **`h5_loader.js`**: JavaScript module for HDF5 handling
   - API client for server communication
   - Alternative browser-based HDF5 reader (using h5wasm)

## API Endpoints

- `GET /api/experiments` - List all available experiments
- `GET /api/trajectory?path=<path>&episode=<id>` - Get trajectory data
- `GET /api/metadata?path=<path>` - Get experiment metadata

## Controls

- **Mouse**: Left-drag to rotate, right-drag to pan, scroll to zoom
- **Timeline**: Scrub to any point in the episode
- **Playback**: Play/pause with adjustable speed (0.1x - 5x)
- **Visual Options**: Toggle trail, grid lines, reward glow, auto-rotate

## Adding New Experiments

New experiments are automatically discovered when:
1. HDF5 files are created in standard locations (`models/*/logs/`)
2. The visualization is refreshed (click "Refresh Experiments")

No manual extraction or conversion needed!

## Performance Optimizations

For large grids (>20x20):
- Reduced grid line density
- Simplified reward animations
- Optimized camera settings
- Disabled shadows and glow effects

## Development

To extend the visualization:
1. Modify `api_server.py` to add new endpoints
2. Update `grid_world_h5_viewer.html` for new visual features
3. Use `ExperimentLoader` methods to access additional data