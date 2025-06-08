#!/usr/bin/env python3
"""
keywords: [simple, server, http, static]

Simple HTTP server for serving static files without CORS issues
"""

import http.server
import socketserver
import os
from pathlib import Path

PORT = 8081

class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

def serve():
    # Change to the visualization directory
    os.chdir(Path(__file__).parent)
    
    with socketserver.TCPServer(("", PORT), CORSHTTPRequestHandler) as httpd:
        print(f"Serving visualization at http://localhost:{PORT}")
        print(f"Open http://localhost:{PORT}/index.html in your browser")
        httpd.serve_forever()

if __name__ == "__main__":
    serve()