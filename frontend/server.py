#!/usr/bin/env python3
"""
Simple HTTP server for serving the Disease Outbreak Early Warning System frontend
"""

import http.server
import socketserver
import os
import sys
from pathlib import Path

class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

def main():
    # Get the directory where this script is located
    script_dir = Path(__file__).parent.absolute()
    
    # Change to the frontend directory
    os.chdir(script_dir)
    
    # Set up server configuration
    PORT = 8080
    Handler = CORSHTTPRequestHandler
    
    try:
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print(f"🚀 Frontend server started at http://localhost:{PORT}")
            print(f"📁 Serving files from: {script_dir}")
            print(f"🌐 Open your browser and navigate to: http://localhost:{PORT}")
            print(f"📱 The website will automatically connect to the API at http://localhost:8000")
            print(f"⏹️  Press Ctrl+C to stop the server")
            print("-" * 60)
            
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"❌ Port {PORT} is already in use. Please stop the existing server or use a different port.")
            print(f"💡 You can change the port by modifying the PORT variable in server.py")
        else:
            print(f"❌ Error starting server: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
