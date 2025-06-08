# Quick Setup Guide for SNN Visualization

## Using uv (recommended)

```bash
# From project root
uv sync --extra visualization

# Launch
cd visualization/3d
python launch.py
```

## Alternative: Using uv pip

```bash
# From project root
uv pip install -e '.[visualization]'

# Launch
cd visualization/3d
python launch.py
```

## What it installs

The `visualization` extra includes:
- FastAPI & Uvicorn (async web server)
- msgpack & lz4 (data compression)
- pandas (data analysis)
- All base dependencies (numpy, h5py, etc.)

## Troubleshooting

If dependencies are missing:
```bash
# Ensure you're in project root
cd /Users/krisztiaan/dev/metalearning

# Sync all extras
uv sync --all-extras
```