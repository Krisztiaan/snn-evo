#!/usr/bin/env python3
# keywords: [unified runner, visualization server, data analysis tool]
"""
Unified script to run the SNN Scientific Visualization Tool.

This tool provides a web-based interface for in-depth analysis of
SNN experiment data stored in HDF5 files.
"""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from threading import Thread

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def start_api_server():
    """Start the API server."""
    server_script = Path(__file__).parent / "api_server.py"
    print(f"Starting API server from: {server_script}")
    cmd = [sys.executable, str(server_script)]
    process = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
    return process


def open_browser(url, delay=2):
    """Open browser after a delay."""
    time.sleep(delay)
    print(f"\nOpening browser at: {url}")
    try:
        webbrowser.open(url)
    except Exception as e:
        print(f"Could not open browser automatically: {e}")
        print(f"Please navigate to {url} manually.")


def main():
    """Main entry point."""
    print("=" * 60)
    print("SNN Scientific Visualization Tool")
    print("=" * 60)
    print("\nThis tool directly loads and analyzes HDF5 experiment files.")
    print("\nFeatures:")
    print("  - Browse all available experiments by model phase.")
    print("  - 3D trajectory playback with detailed data overlay.")
    print("  - Synchronized charts for behavior, neural dynamics, and learning signals.")
    print("  - Designed for rigorous scientific analysis of agent behavior.")
    print("\nStarting server...")

    url = "http://localhost:8000/"
    browser_thread = Thread(target=open_browser, args=(url,))
    browser_thread.daemon = True
    browser_thread.start()

    server_process = None
    try:
        server_process = start_api_server()
        server_process.wait()
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        if server_process:
            server_process.terminate()
            server_process.wait()
        sys.exit(0)


if __name__ == "__main__":
    main()
