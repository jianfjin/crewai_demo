#!/usr/bin/env python3
"""
Test script to run the Marketing Research Crew analysis.
This script tests the main functionality and generates a sample report.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from marketing_research_swarm.main import main, run_specific_analysis

def test_basic_functionality():
    """Test basic functionality of the marketing research tools."""
    print("Testing Marketing Research Swarm...")
    print("=" * 50)
    
    # Test individual tools first
    from marketing_research_swarm.tools.advanced_tools import (
        time_series_analysis,
        cross_sectional_analysis,
        forecast_sales,
        calculate_roi,
        plan_budget
    )
    
    data_path = "data/beverage_sales.csv"
    
    print("1. Testing Time Series Analysis Tool...")
    try:
        result = time_series_analysis._run(data_path)
        print("✓ Time Series Analysis: SUCCESS")
        print(result[:200] + "..." if len(result) > 200 else result)
    except Exception as e:
        print(f"✗ Time Series Analysis: FAILED - {e}")
    
    print("\n2. Testing Cross Sectional Analysis Tool...")
    try:
        result = cross_sectional_analysis._run(data_path)
        print("✓ Cross Sectional Analysis: SUCCESS")
        print(result[:200] + "..." if len(result) > 200 else result)
    except Exception as e:
        print(f"✗ Cross Sectional Analysis: FAILED - {e}")
    
    print("\n3. Testing Sales Forecast Tool...")
    try:
        result = forecast_sales._run(data_path, periods=7)
        print("✓ Sales Forecast: SUCCESS")
        print(result[:200] + "..." if len(result) > 200 else result)
    except Exception as e:
        print(f"✗ Sales Forecast: FAILED - {e}")
    
    print("\n4. Testing ROI Calculator...")
    try:
        result = calculate_roi._run(revenue=250000, cost=100000)
        print("✓ ROI Calculator: SUCCESS")
        print(result)
    except Exception as e:
        print(f"✗ ROI Calculator: FAILED - {e}")
    
    print("\n5. Testing Budget Planner...")
    try:
        result = plan_budget._run(total_budget=100000)
        print("✓ Budget Planner: SUCCESS")
        print(result)
    except Exception as e:
        print(f"✗ Budget Planner: FAILED - {e}")

def test_crew_analysis():
    """Test the full crew analysis."""
    print("\n" + "=" * 50)
    print("Testing Full Crew Analysis...")
    print("=" * 50)
    
    try:
        # Run a simplified analysis
        print("Running comprehensive marketing research analysis...")
        main()
        print("✓ Full Crew Analysis: SUCCESS")
    except Exception as e:
        print(f"✗ Full Crew Analysis: FAILED - {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Marketing Research Swarm - Test Suite")
    print("=" * 50)
    
    # Test individual tools first
    test_basic_functionality()
    
    # Test full crew analysis
    test_crew_analysis()
    
    print("\n" + "=" * 50)
    print("Test Suite Completed!")