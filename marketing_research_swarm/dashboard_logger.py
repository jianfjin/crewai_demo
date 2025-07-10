#!/usr/bin/env python3
"""
Dashboard Logger - Captures all dashboard output to log files
"""

import sys
import os
import datetime
from contextlib import redirect_stdout, redirect_stderr
import streamlit as st

class DashboardLogger:
    """Logger that captures dashboard output to files"""
    
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # Create timestamped log file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"dashboard_{timestamp}.log")
        
        # Open log file
        self.log_handle = open(self.log_file, 'w', encoding='utf-8')
        
        # Store original stdout/stderr
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
    def start_logging(self):
        """Start capturing output to log file"""
        # Create a tee that writes to both console and file
        sys.stdout = TeeOutput(self.original_stdout, self.log_handle)
        sys.stderr = TeeOutput(self.original_stderr, self.log_handle)
        
        self.log(f"Dashboard logging started at {datetime.datetime.now()}")
        self.log(f"Log file: {self.log_file}")
        self.log("=" * 80)
        
    def stop_logging(self):
        """Stop capturing output"""
        self.log("=" * 80)
        self.log(f"Dashboard logging ended at {datetime.datetime.now()}")
        
        # Restore original stdout/stderr
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        
        # Close log file
        self.log_handle.close()
        
    def log(self, message):
        """Write message to log file"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        self.log_handle.write(log_message)
        self.log_handle.flush()
        
    def log_analysis_start(self, analysis_type, optimization_level):
        """Log analysis start"""
        self.log(f"ANALYSIS START: {analysis_type} with optimization: {optimization_level}")
        
    def log_analysis_end(self, analysis_type, duration, success):
        """Log analysis end"""
        status = "SUCCESS" if success else "FAILED"
        self.log(f"ANALYSIS END: {analysis_type} - {status} - Duration: {duration:.2f}s")
        
    def log_error(self, error_message):
        """Log error message"""
        self.log(f"ERROR: {error_message}")
        
    def log_token_usage(self, token_data):
        """Log token usage information"""
        self.log(f"TOKEN USAGE: {token_data}")

class TeeOutput:
    """Output stream that writes to multiple destinations"""
    
    def __init__(self, *streams):
        self.streams = streams
        
    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
            
    def flush(self):
        for stream in self.streams:
            stream.flush()

# Global logger instance
_dashboard_logger = None

def get_dashboard_logger():
    """Get or create dashboard logger"""
    global _dashboard_logger
    if _dashboard_logger is None:
        _dashboard_logger = DashboardLogger()
    return _dashboard_logger

def start_dashboard_logging():
    """Start dashboard logging"""
    logger = get_dashboard_logger()
    logger.start_logging()
    return logger

def stop_dashboard_logging():
    """Stop dashboard logging"""
    global _dashboard_logger
    if _dashboard_logger:
        _dashboard_logger.stop_logging()
        _dashboard_logger = None