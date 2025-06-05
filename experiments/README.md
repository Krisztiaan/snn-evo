# Experiments Directory

This directory contains all experiment outputs organized by model type.

## Structure

```
experiments/
├── random/           # Random agent baseline experiments
│   ├── episode_*/    # Individual episode runs
│   └── ...
├── [model_name]/     # Future model experiments
└── ...
```

## Data Format

Each experiment creates a timestamped directory containing:
- `data.h5` - HDF5 file with trajectory and network data
- `config.json` - Experiment configuration
- `metadata.json` - Experiment metadata
- `summary.json` - Episode summary statistics

## Notes

- All experiment data is gitignored to keep the repository clean
- Use the loader utilities in `export/` to analyze experiment data
- Each model type has its own subdirectory to avoid conflicts