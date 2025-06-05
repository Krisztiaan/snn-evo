# Phase 0.4 - Biologically Enhanced Research Implementation

Optimized research script with key biological features and comprehensive data export.

## Usage

```bash
# Quick test
python phase04_research.py -n 1

# Standard run with progress
python phase04_research.py -n 10 --progress

# Large study without progress (maximum performance)
python phase04_research.py -n 100 --no-progress

# Custom output directory
python phase04_research.py -n 50 -o results
```

## Features

### Biological Enhancements
- **E/I Balance**: 80/20 excitatory/inhibitory neuron ratio
- **Dale's Principle**: Neurons maintain consistent neurotransmitter type
- **Three-Factor Learning**: Eligibility traces modulated by dopamine
- **Refractory Periods**: Biologically plausible spike generation
- **Synaptic Dynamics**: Realistic synaptic current integration

### Technical Features
- 10-100x performance optimization
- Complete neural dynamics recording
- Weight evolution tracking  
- Trajectory and reward data
- Command-line interface
- Progress reporting with ETA
- Compressed data export

## Output Structure

```
logs/YYYYMMDD_HHMMSS/
├── config.json              # Full configuration
├── experiment_summary.json  # Statistical summary
├── results.csv             # Quick-access table
└── episode_XXX/            # Per-episode data
    ├── metadata.json        # Episode metadata
    ├── trajectory.npz       # Positions, actions, rewards
    ├── neural_dynamics.npz  # Spikes, membrane potentials, dopamine
    ├── network_properties.npz # E/I identity, neuron types
    ├── weight_evolution.pkl.gz # Weight snapshots with dopamine
    └── environment.npz      # Reward positions, grid info
```

### New Biological Data Exports
- **neural_dynamics.npz**: Now includes synaptic currents, refractory states, and dopamine levels
- **network_properties.npz**: E/I neuron identities and counts
- **weight_evolution**: Includes dopamine levels at each snapshot

All data is compressed and organized for efficient analysis.