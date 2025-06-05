#!/usr/bin/env python3
# keywords: [unified runner, visualization server, experiment watcher]
"""Unified script to run the HDF5-based visualization server."""

import os
import sys
import time
import subprocess
from pathlib import Path
import webbrowser
from threading import Thread

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def start_api_server():
    """Start the API server."""
    print("Starting API server...")
    cmd = [sys.executable, "visualization/3d/api_server.py"]
    subprocess.run(cmd)


def open_browser(url, delay=2):
    """Open browser after a delay."""
    time.sleep(delay)
    print(f"\nOpening browser at: {url}")
    webbrowser.open(url)


def main():
    """Main entry point."""
    print("=" * 60)
    print("Grid World HDF5 Visualization")
    print("=" * 60)
    print("\nThis visualization directly loads HDF5 experiment files")
    print("using the standardized ExperimentLoader module.")
    print("\nFeatures:")
    print("  - Browse all available experiments")
    print("  - Select and load specific episodes")
    print("  - Real-time trajectory playback")
    print("  - Seamless HDF5 -> Visualization pipeline")
    print("\nStarting server...")

    # Start browser in background
    browser_thread = Thread(
        target=open_browser,
        args=("http://localhost:8000/grid_world_trajectory_viewer.html",)
    )
    browser_thread.daemon = True
    browser_thread.start()

    # Run server (blocks)
    try:
        start_api_server()
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        sys.exit(0)


if __name__ == "__main__":
    main()
