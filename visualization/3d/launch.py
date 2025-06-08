#!/usr/bin/env python3
"""
keywords: [launch, visualization, server, browser]

Launch script for SNN visualization system
"""

import subprocess
import time
import webbrowser
import sys
import os
import signal
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import fastapi
        import uvicorn
        import h5py
        import msgpack
        import lz4
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("\nPlease install visualization dependencies with:")
        print("  uv pip install -e '.[visualization]'")
        print("\nOr from the project root:")
        print("  uv sync --extra visualization")
        return False

def start_server():
    """Start the FastAPI server"""
    server_path = Path(__file__).parent / "server.py"
    
    # Start server process
    process = subprocess.Popen(
        [sys.executable, str(server_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for server to start
    print("Starting visualization server...")
    time.sleep(2)
    
    # Check if server started successfully
    if process.poll() is not None:
        stdout, stderr = process.communicate()
        print("Server failed to start:")
        print(stderr)
        return None
        
    print("Server started successfully on http://localhost:8080")
    return process

def open_browser():
    """Open the visualization in the default browser"""
    html_path = Path(__file__).parent / "index.html"
    url = f"file://{html_path.absolute()}"
    
    print(f"Opening visualization at: {url}")
    webbrowser.open(url)

def main():
    """Main launch function"""
    print("SNN Agent Visualization Launcher")
    print("================================\n")
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Start server
    server_process = start_server()
    if not server_process:
        sys.exit(1)
    
    # Open browser
    time.sleep(1)
    open_browser()
    
    print("\nVisualization is running!")
    print("Press Ctrl+C to stop the server and exit.\n")
    
    # Keep running until interrupted
    try:
        while True:
            # Check if server is still running
            if server_process.poll() is not None:
                print("Server stopped unexpectedly")
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        server_process.terminate()
        server_process.wait(timeout=5)
        print("Server stopped.")

if __name__ == "__main__":
    main()