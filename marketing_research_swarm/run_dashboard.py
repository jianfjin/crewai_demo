#!/usr/bin/env python
"""
Launch script for Marketing Research Swarm Dashboard
"""

import subprocess
import sys
import os

def install_requirements():
    """Install dashboard requirements"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_dashboard.txt"])
        print("✅ Dashboard requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False
    return True

def launch_dashboard():
    """Launch the Streamlit dashboard"""
    try:
        print("🚀 Launching Marketing Research Swarm Dashboard...")
        print("📊 Dashboard will open in your default web browser")
        print("🔗 URL: http://localhost:8501")
        print("\n" + "="*50)
        print("Press Ctrl+C to stop the dashboard")
        print("="*50 + "\n")
        
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "dashboard.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")

def main():
    """Main function"""
    print("🔧 Setting up Marketing Research Swarm Dashboard...")
    
    # Check if we're in the right directory
    if not os.path.exists("dashboard.py"):
        print("❌ Error: dashboard.py not found. Please run this script from the marketing_research_swarm directory.")
        return
    
    # Install requirements
    if not install_requirements():
        return
    
    # Launch dashboard
    launch_dashboard()

if __name__ == "__main__":
    main()