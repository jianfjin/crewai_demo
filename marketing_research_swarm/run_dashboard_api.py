#!/usr/bin/env python
"""
Launch script for Marketing Research Swarm API-Based Dashboard
"""

import subprocess
import sys
import os
import requests
import time

def check_backend_health():
    """Check if FastAPI backend is running"""
    try:
        response = requests.get("http://localhost:8000/", timeout=5)
        return response.status_code == 200
    except:
        return False

def start_backend():
    """Start the FastAPI backend if not running"""
    if check_backend_health():
        print("✅ FastAPI backend is already running")
        return True
    
    print("🚀 Starting FastAPI backend...")
    
    # Check if backend directory exists
    backend_dir = "backend"
    if not os.path.exists(backend_dir):
        print(f"❌ Backend directory '{backend_dir}' not found")
        return False
    
    # Check if main.py exists in backend
    backend_main = os.path.join(backend_dir, "main.py")
    if not os.path.exists(backend_main):
        print(f"❌ Backend main.py not found at '{backend_main}'")
        return False
    
    try:
        # Start backend in background
        backend_process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", "main:app",
            "--reload", "--port", "8000", "--host", "localhost"
        ], cwd=backend_dir)
        
        # Wait for backend to start
        print("⏳ Waiting for backend to start...")
        for i in range(30):  # Wait up to 30 seconds
            if check_backend_health():
                print("✅ FastAPI backend started successfully!")
                return True
            time.sleep(1)
            print(f"   Checking... ({i+1}/30)")
        
        print("❌ Backend failed to start within 30 seconds")
        backend_process.terminate()
        return False
        
    except Exception as e:
        print(f"❌ Error starting backend: {e}")
        return False

def install_requirements():
    """Install dashboard requirements"""
    try:
        # Check if requirements file exists
        if os.path.exists("requirements_dashboard.txt"):
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", "-r", "requirements_dashboard.txt"
            ])
        else:
            # Install basic requirements
            requirements = [
                "streamlit>=1.28.0",
                "requests>=2.31.0",
                "pandas>=2.0.0",
                "plotly>=5.15.0"
            ]
            subprocess.check_call([
                sys.executable, "-m", "pip", "install"
            ] + requirements)
        
        print("✅ Dashboard requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def launch_api_dashboard():
    """Launch the API-based Streamlit dashboard"""
    try:
        print("🚀 Launching API-Based Marketing Research Dashboard...")
        print("📊 Dashboard will open in your default web browser")
        print("🔗 Dashboard URL: http://localhost:8501")
        print("🔗 Backend API URL: http://localhost:8000")
        print("\n" + "="*60)
        print("🎯 API-BASED DASHBOARD FEATURES:")
        print("   ✅ Real-time progress monitoring")
        print("   ✅ Non-blocking user interface")
        print("   ✅ Live performance metrics")
        print("   ✅ Analysis history management")
        print("   ✅ Enhanced error handling")
        print("="*60)
        print("Press Ctrl+C to stop the dashboard")
        print("="*60 + "\n")
        
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "dashboard_api.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n🛑 API Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error launching API dashboard: {e}")

def show_help():
    """Show help information"""
    print("""
🚀 Marketing Research Swarm - API-Based Dashboard Launcher

DESCRIPTION:
    This script launches the new API-based Streamlit dashboard that communicates
    with the FastAPI backend for better architecture separation and performance.

FEATURES:
    ✅ Real-time analysis monitoring with live progress updates
    ✅ Non-blocking user interface that remains responsive
    ✅ Enhanced error handling with structured API responses
    ✅ System performance metrics and monitoring
    ✅ Analysis history with comparison and export features
    ✅ Future-proof architecture ready for scaling

REQUIREMENTS:
    - FastAPI backend running on http://localhost:8000
    - Python 3.8+ with required packages
    - Internet connection for package installation

USAGE:
    python run_dashboard_api.py [options]

OPTIONS:
    --help, -h          Show this help message
    --backend-only      Start only the backend (no dashboard)
    --dashboard-only    Start only the dashboard (assume backend is running)
    --check-health      Check backend health and exit

ARCHITECTURE:
    Streamlit Dashboard → HTTP API → FastAPI Backend → Marketing Research Code
    
    Benefits:
    - Loose coupling with clean separation of concerns
    - Non-blocking UI with real-time updates
    - Better resource management and isolation
    - Independent scaling and deployment

TROUBLESHOOTING:
    1. Backend not starting:
       - Check if port 8000 is available
       - Ensure backend/main.py exists
       - Check backend dependencies
    
    2. Dashboard connection issues:
       - Verify backend is running: curl http://localhost:8000/
       - Check firewall settings
       - Ensure no proxy interference
    
    3. Performance issues:
       - Monitor system resources
       - Check API response times
       - Review backend logs

For more information, see the API_BASED_DASHBOARD_IMPLEMENTATION_SUMMARY.md file.
""")

def main():
    """Main function"""
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        
        if arg in ["--help", "-h"]:
            show_help()
            return
        elif arg == "--backend-only":
            print("🔧 Starting backend only...")
            if start_backend():
                print("✅ Backend is running. You can now start the dashboard separately.")
                print("🔗 Backend API: http://localhost:8000")
                print("📚 API Docs: http://localhost:8000/docs")
                input("Press Enter to stop the backend...")
            return
        elif arg == "--dashboard-only":
            print("🔧 Starting dashboard only (assuming backend is running)...")
            if not check_backend_health():
                print("❌ Backend is not running. Please start it first or run without --dashboard-only")
                return
            launch_api_dashboard()
            return
        elif arg == "--check-health":
            print("🔍 Checking backend health...")
            if check_backend_health():
                print("✅ Backend is healthy and responding")
                try:
                    response = requests.get("http://localhost:8000/", timeout=5)
                    print(f"📊 Response: {response.json()}")
                except:
                    print("📊 Backend responding but couldn't parse response")
            else:
                print("❌ Backend is not responding")
            return
    
    print("🔧 Setting up API-Based Marketing Research Dashboard...")
    
    # Check if we're in the right directory
    if not os.path.exists("dashboard_api.py"):
        print("❌ Error: dashboard_api.py not found. Please run this script from the marketing_research_swarm directory.")
        return
    
    # Install requirements
    print("📦 Installing requirements...")
    if not install_requirements():
        return
    
    # Start backend
    print("🔧 Setting up backend...")
    if not start_backend():
        print("❌ Failed to start backend. Please check the backend configuration.")
        print("💡 Try running: python run_dashboard_api.py --help")
        return
    
    # Launch API dashboard
    print("🎯 Launching API-based dashboard...")
    launch_api_dashboard()

if __name__ == "__main__":
    main()