# SNN Agent Scientific Visualization System

A high-performance, modular visualization system for analyzing Spiking Neural Network (SNN) agent behavior, neural dynamics, and learning progress.

## Features

### Core Capabilities
- **Real-time 3D Trajectory Visualization**: WebGL-accelerated agent movement tracking with dynamic trail rendering
- **Neural Activity Viewer**: Spike raster plots with multiple colormaps and temporal windowing
- **Connectome Visualization**: Real-time weight matrix visualization with excitatory/inhibitory distinction
- **Performance Analytics**: Learning curves, value functions, reward tracking, and statistical analysis
- **Timeline Navigation**: Scrubbing, event markers, activity heatmap, and keyboard shortcuts
- **Data Export**: Multiple formats (JSON, CSV, MATLAB, NumPy) with analysis reports

### Performance Optimizations
- **Async Data Streaming**: WebSocket-based neural data streaming with LZ4 compression
- **Intelligent Caching**: LRU cache for experiment data with automatic eviction
- **Data Decimation**: Automatic LOD for large datasets while preserving features
- **GPU Acceleration**: WebGL rendering with instanced meshes and buffer optimization
- **Virtual Scrolling**: Efficient handling of millions of data points

### Scientific Features
- **Statistical Analysis**: Real-time computation of firing rates, exploration coverage, TD error
- **Event Detection**: Automatic identification of rewards, high-value states, neural bursts
- **Comparison Tools**: Side-by-side episode analysis (planned)
- **Annotation System**: Mark and export interesting events (planned)

## Installation

### Requirements
- Python 3.8+
- Modern web browser with WebGL support

### Setup
```bash
# From the project root directory
uv sync --extra visualization

# Launch the visualization
cd visualization/3d
python launch.py
```

## Usage

### Quick Start
1. Run `python launch.py` to start the server and open the visualization
2. Select an experiment from the dropdown
3. Choose an episode to visualize
4. Use playback controls or timeline scrubbing to explore

### Keyboard Shortcuts
- **Space**: Play/Pause
- **Left/Right Arrow**: Step backward/forward (10ms)
- **Shift + Left/Right**: Large steps (100ms)
- **Home/End**: Jump to start/end
- **PageUp/PageDown**: Jump to previous/next event
- **R**: Reset camera view

### Data Export
1. Click the "Export Data" button
2. Choose format:
   - **JSON**: Complete data with metadata
   - **CSV**: Separate files for trajectory, neural summary, and analysis
   - **MATLAB**: .m file with plotting examples
   - **NumPy**: Python script with analysis code

## Architecture

### Backend (FastAPI + WebSocket)
```
server.py
├── DataCache          # LRU cache for HDF5 files
├── DataProcessor      # Decimation, spike rate computation
├── WebSocket Handler  # Real-time streaming
└── REST Endpoints     # Experiments, episodes, analysis
```

### Frontend (Modular ES6)
```
js/
├── app.js              # Main coordinator
├── data-manager.js     # API communication, caching
├── viewport-3d.js      # Three.js visualization
├── neural-visualizer.js # Spike rasters, connectivity
├── chart-manager.js    # Chart.js analytics
├── timeline-controller.js # Temporal navigation
└── export-manager.js   # Data export functionality
```

## Performance Benchmarks

- **Data Loading**: < 100ms for 1M data points
- **Rendering**: 60 FPS with 10k trajectory points
- **Neural Visualization**: Real-time for 1000 neurons
- **Memory Usage**: ~200MB for typical experiment

## API Reference

### REST Endpoints

#### Get Experiments
```
GET /api/experiments
Returns: List[ExperimentInfo]
```

#### Get Episode Data
```
GET /api/experiment/{experiment_id}/episode/{episode_id}/trajectory
Query: decimation (1-100)
Returns: Trajectory data with metadata
```

#### Get Analysis
```
GET /api/experiment/{experiment_id}/analysis
Returns: Aggregate metrics and learning curves
```

### WebSocket Protocol

#### Connect
```
WS /ws/stream/{experiment_id}
```

#### Request Neural Data
```json
{
  "type": "request_data",
  "data": {
    "data_type": "neural",
    "episode_id": 0,
    "start_time": 0,
    "end_time": 1000
  }
}
```

## Development

### Adding New Visualizations
1. Create new module in `js/` directory
2. Import in `app.js`
3. Add initialization in `init()` method
4. Connect to data updates

### Custom Color Maps
Edit `neural-visualizer.js`:
```javascript
generateCustomColorMap() {
    const colors = [];
    for (let i = 0; i < 256; i++) {
        // Your color logic here
        colors.push(`rgb(${r},${g},${b})`);
    }
    return colors;
}
```

## Troubleshooting

### Server Won't Start
- Check port 8080 is available
- Verify Python dependencies installed
- Check experiment data paths

### Visualization Blank
- Open browser console for errors
- Verify WebGL support
- Check server is running

### Performance Issues
- Reduce max neurons in neural viewer
- Enable decimation for large episodes
- Close other browser tabs

## Future Enhancements
- Multi-experiment comparison
- Live training visualization
- 3D neural connectivity graph
- Advanced statistical tests
- Plugin system for custom analyses