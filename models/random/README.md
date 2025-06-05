# Random Agent

A baseline agent that selects random actions at each timestep.

## Purpose

This agent serves as a baseline for comparison with learning agents. It demonstrates:
- Integration with the simple grid world
- Data export using the standard exporter
- Basic experiment structure

## Usage

```bash
# Run single episode with default settings
python -m models.random.run

# Run multiple episodes
python -m models.random.run --episodes 10

# Custom configuration
python -m models.random.run --grid-size 100 --rewards 300 --steps 5000

# Minimal export (no full trajectory)
python -m models.random.run --no-trajectory
```

## Configuration

- `--episodes`: Number of episodes to run
- `--steps`: Steps per episode (default: 1000)
- `--grid-size`: Size of grid world (default: 50)
- `--rewards`: Number of rewards in world (default: 50)
- `--seed`: Random seed for reproducibility
- `--export-dir`: Directory for exported data
- `--no-trajectory`: Skip exporting full trajectory data

## Exported Data

Each episode creates a timestamped directory with:
- `config.json`: Experiment configuration
- `metadata.json`: Experiment metadata
- `data.h5`: HDF5 file containing:
  - Episode trajectory (positions, actions, rewards, observations)
  - Episode summary (total reward, coverage, etc.)
  - Final world state

## Performance

As a random agent, performance depends entirely on chance:
- Expected to collect few rewards
- Low coverage of the grid
- Serves as lower bound for learning agents