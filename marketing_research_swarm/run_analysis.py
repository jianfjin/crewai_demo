#!/usr/bin/env python3
"""
Example script to run Marketing Research Crew analysis.
This script demonstrates how to use the marketing research tools for beverage sales analysis.
"""

import sys
import os
import argparse

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Run the marketing research analysis."""
    print("Marketing Research Swarm - Beverage Sales Analysis")
    print("=" * 60)
    print("This analysis will:")
    print("1. Analyze beverage sales data trends")
    print("2. Compare regional performance (North vs South)")
    print("3. Evaluate product performance (Green Tea vs Latte)")
    print("4. Generate sales forecasts")
    print("5. Create marketing strategy recommendations")
    print("6. Develop budget allocation plans")
    print("=" * 60)
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run Marketing Research Swarm Analysis")
    parser.add_argument(
        "--type",
        type=str,
        default="comprehensive",
        help="Type of analysis to run: comprehensive, sales_forecast, roi_analysis, brand_performance (default: comprehensive)"
    )
    args = parser.parse_args()

    # Import and run the main analysis
    from marketing_research_swarm.main import run_specific_analysis
    
    try:
        run_specific_analysis(args.type)
    except ImportError as e:
        print(f"Import Error: {e}")
        print("Please ensure all dependencies are installed:")
        print("pip install crewai pandas numpy matplotlib seaborn plotly scikit-learn statsmodels")
    except Exception as e:
        print(f"Analysis Error: {e}")
        print("Please check your configuration and data files.")

if __name__ == "__main__":
    main()