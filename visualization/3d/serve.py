#!/usr/bin/env python3
"""Simple HTTP server for the visualization."""

import http.server
import socketserver
import os

PORT = 8000

class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

os.chdir(os.path.dirname(os.path.abspath(__file__)))

with socketserver.TCPServer(("", PORT), CORSRequestHandler) as httpd:
    print(f"Grid World Visualization Server")
    print(f"==============================")
    print(f"Serving at: http://localhost:{PORT}/")
    print(f"Open: http://localhost:{PORT}/grid_world_trajectory_viewer.html")
    print(f"\nAvailable trajectories:")
    for file in sorted(os.listdir('.')):
        if file.startswith('trajectory_') and file.endswith('.json'):
            print(f"  - {file}")
    print(f"\nPress Ctrl+C to stop")
    httpd.serve_forever()