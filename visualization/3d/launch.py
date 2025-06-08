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

def start_servers():
    """Start both the API server and static file server"""
    processes = []
    
    # Start API server
    api_server_path = Path(__file__).parent / "server.py"
    api_process = subprocess.Popen(
        [sys.executable, str(api_server_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    processes.append(api_process)
    
    # Start static file server
    static_server_path = Path(__file__).parent / "simple_server.py"
    static_process = subprocess.Popen(
        [sys.executable, str(static_server_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    processes.append(static_process)
    
    # Wait for servers to start
    print("Starting visualization servers...")
    time.sleep(3)
    
    # Check if servers started successfully
    for i, process in enumerate(processes):
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            print(f"Server {i} failed to start:")
            print(stderr)
            # Kill other process
            for p in processes:
                if p.poll() is None:
                    p.terminate()
            return None
    
    print("API server started on http://localhost:8080")
    print("Static server started on http://localhost:8081")
    return processes

def open_browser():
    """Open the visualization in the default browser"""
    url = "http://localhost:8081/index.html"
    
    print(f"Opening visualization at: {url}")
    webbrowser.open(url)

def main():
    """Main launch function"""
    print("SNN Agent Visualization Launcher")
    print("================================\n")
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Start servers
    server_processes = start_servers()
    if not server_processes:
        sys.exit(1)
    
    # Open browser
    time.sleep(1)
    open_browser()
    
    print("\nVisualization is running!")
    print("Press Ctrl+C to stop the servers and exit.\n")
    
    # Keep running until interrupted
    try:
        while True:
            # Check if servers are still running
            for process in server_processes:
                if process.poll() is not None:
                    print("A server stopped unexpectedly")
                    # Stop all servers
                    for p in server_processes:
                        if p.poll() is None:
                            p.terminate()
                    return
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nShutting down...")
        for process in server_processes:
            process.terminate()
            process.wait(timeout=5)
        print("Servers stopped.")

if __name__ == "__main__":
    main()