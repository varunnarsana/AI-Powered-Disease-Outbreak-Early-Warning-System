#!/usr/bin/env python3
"""
Script to start all services for the Disease Outbreak Early Warning System.

This script starts:
1. The FastAPI backend server
2. The Streamlit dashboard
3. The frontend server
"""

import subprocess
import sys
import time
import os
from pathlib import Path
import signal
import webbrowser

# Configuration
API_PORT = 8000
DASHBOARD_PORT = 8501
FRONTEND_PORT = 8001

# Paths
BASE_DIR = Path(__file__).parent.absolute()
API_DIR = BASE_DIR / "api"
DASHBOARD_DIR = BASE_DIR / "dashboard"
FRONTEND_DIR = BASE_DIR / "frontend"

# Processes
processes = []

def start_api():
    """Start the FastAPI server."""
    print("Starting API server...")
    cmd = [
        sys.executable, "-m", "uvicorn", 
        "main:app", 
        "--host", "0.0.0.0", 
        "--port", str(API_PORT),
        "--reload"
    ]
    return subprocess.Popen(
        cmd, 
        cwd=str(API_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

def start_dashboard():
    """Start the Streamlit dashboard."""
    print("Starting dashboard...")
    cmd = [
        sys.executable, "-m", "streamlit", "run", 
        "app.py", 
        "--server.port", str(DASHBOARD_PORT),
        "--server.headless", "false"
    ]
    return subprocess.Popen(
        cmd, 
        cwd=str(DASHBOARD_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

def start_frontend():
    """Start the frontend server."""
    print("Starting frontend server...")
    cmd = [sys.executable, "server.py", str(FRONTEND_PORT)]
    return subprocess.Popen(
        cmd, 
        cwd=str(FRONTEND_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

def open_browser():
    """Open the dashboard in the default web browser."""
    print("Opening dashboard in browser...")
    dashboard_url = f"http://localhost:{DASHBOARD_PORT}"
    webbrowser.open(dashboard_url)

def signal_handler(sig, frame):
    """Handle keyboard interrupt to clean up processes."""
    print("\nShutting down services...")
    for process in processes:
        if process.poll() is None:  # Check if process is still running
            print(f"Terminating process {process.pid}...")
            process.terminate()
    sys.exit(0)

def main():
    """Main function to start all services."""
    # Set up signal handler for clean shutdown
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start services
    try:
        # Start API server
        api_process = start_api()
        processes.append(api_process)
        print(f"API server started with PID {api_process.pid}")
        
        # Give API server a moment to start
        time.sleep(2)
        
        # Start dashboard
        dashboard_process = start_dashboard()
        processes.append(dashboard_process)
        print(f"Dashboard started with PID {dashboard_process.pid}")
        
        # Start frontend
        frontend_process = start_frontend()
        processes.append(frontend_process)
        print(f"Frontend server started with PID {frontend_process.pid}")
        
        # Open browser
        open_browser()
        
        # Monitor processes
        while True:
            for i, process in enumerate(processes):
                if process.poll() is not None:  # Process has terminated
                    print(f"Process {process.pid} has terminated with return code {process.returncode}")
                    print(f"STDOUT: {process.stdout.read()}")
                    print(f"STDERR: {process.stderr.read()}")
                    print("Shutting down other processes...")
                    signal_handler(None, None)
                    return
            time.sleep(1)
            
    except Exception as e:
        print(f"Error: {e}")
        signal_handler(None, None)

if __name__ == "__main__":
    print("=== Disease Outbreak Early Warning System ===")
    print(f"API: http://localhost:{API_PORT}")
    print(f"Dashboard: http://localhost:{DASHBOARD_PORT}")
    print(f"Frontend: http://localhost:{FRONTEND_PORT}")
    print("Press Ctrl+C to stop all services\n")
    
    # Create necessary directories
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    main()
