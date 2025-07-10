#!/usr/bin/env python3
"""
Launch script for Marketing Research Swarm Dashboard with Logging
"""

import subprocess
import sys
import os
import datetime

def setup_logging():
    """Setup logging directory"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"dashboard_output_{timestamp}.log")
    
    return log_file

def launch_dashboard_with_logging():
    """Launch the Streamlit dashboard with output logging"""
    try:
        print("🚀 Launching Marketing Research Swarm Dashboard with Logging...")
        print("📊 Dashboard will open in your default web browser")
        print("🔗 URL: http://localhost:8501")
        
        # Setup logging
        log_file = setup_logging()
        print(f"📝 Output will be logged to: {log_file}")
        
        print("\n" + "="*50)
        print("Press Ctrl+C to stop the dashboard")
        print("="*50 + "\n")
        
        # Launch with output redirection
        with open(log_file, 'w', encoding='utf-8') as log:
            process = subprocess.Popen([
                sys.executable, "-m", "streamlit", "run", "dashboard.py",
                "--server.port", "8501",
                "--server.address", "localhost",
                "--browser.gatherUsageStats", "false"
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
               universal_newlines=True, bufsize=1)
            
            # Real-time output to both console and log file
            for line in process.stdout:
                print(line, end='')  # Console output
                log.write(line)      # Log file output
                log.flush()
                
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
        if 'process' in locals():
            process.terminate()
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")

def main():
    """Main function"""
    print("🔧 Setting up Marketing Research Swarm Dashboard with Logging...")
    
    # Check if we're in the right directory
    if not os.path.exists("dashboard.py"):
        print("❌ Error: dashboard.py not found. Please run this script from the marketing_research_swarm directory.")
        return
    
    # Launch dashboard with logging
    launch_dashboard_with_logging()

if __name__ == "__main__":
    main()