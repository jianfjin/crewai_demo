#!/usr/bin/env python3
"""
Test script for updated marketing research tools with enhanced beverage data
"""

import sys
import os
sys.path.append('src')

from marketing_research_swarm.tools.advanced_tools import (
    beverage_market_analysis,
    profitability_analysis,
    time_series_analysis,
    cross_sectional_analysis,
    forecast_sales
)

def test_beverage_tools():
    """Test the updated beverage analysis tools"""
    
    data_path = "data/beverage_sales.csv"
    
    print("="*80)
    print("TESTING UPDATED BEVERAGE ANALYSIS TOOLS")
    print("="*80)
    
    # Test 1: Beverage Market Analysis
    print("\n1. TESTING BEVERAGE MARKET ANALYSIS TOOL")
    print("-" * 50)
    try:
        result = beverage_market_analysis._run(data_path)
        print(result)
        print("✅ Beverage Market Analysis: SUCCESS")
    except Exception as e:
        print(f"❌ Beverage Market Analysis: FAILED - {e}")
    
    # Test 2: Profitability Analysis by Brand
    print("\n2. TESTING PROFITABILITY ANALYSIS TOOL (BY BRAND)")
    print("-" * 50)
    try:
        result = profitability_analysis._run(data_path, analysis_dimension='brand')
        print(result)
        print("✅ Profitability Analysis (Brand): SUCCESS")
    except Exception as e:
        print(f"❌ Profitability Analysis (Brand): FAILED - {e}")
    
    # Test 3: Profitability Analysis by Category
    print("\n3. TESTING PROFITABILITY ANALYSIS TOOL (BY CATEGORY)")
    print("-" * 50)
    try:
        result = profitability_analysis._run(data_path, analysis_dimension='category')
        print(result)
        print("✅ Profitability Analysis (Category): SUCCESS")
    except Exception as e:
        print(f"❌ Profitability Analysis (Category): FAILED - {e}")
    
    # Test 4: Updated Time Series Analysis
    print("\n4. TESTING UPDATED TIME SERIES ANALYSIS TOOL")
    print("-" * 50)
    try:
        result = time_series_analysis._run(data_path)
        print(result)
        print("✅ Time Series Analysis: SUCCESS")
    except Exception as e:
        print(f"❌ Time Series Analysis: FAILED - {e}")
    
    # Test 5: Cross-Sectional Analysis by Region
    print("\n5. TESTING CROSS-SECTIONAL ANALYSIS TOOL (BY REGION)")
    print("-" * 50)
    try:
        result = cross_sectional_analysis._run(data_path, segment_column='region')
        print(result)
        print("✅ Cross-Sectional Analysis (Region): SUCCESS")
    except Exception as e:
        print(f"❌ Cross-Sectional Analysis (Region): FAILED - {e}")
    
    # Test 6: Cross-Sectional Analysis by Brand
    print("\n6. TESTING CROSS-SECTIONAL ANALYSIS TOOL (BY BRAND)")
    print("-" * 50)
    try:
        result = cross_sectional_analysis._run(data_path, segment_column='brand')
        print(result)
        print("✅ Cross-Sectional Analysis (Brand): SUCCESS")
    except Exception as e:
        print(f"❌ Cross-Sectional Analysis (Brand): FAILED - {e}")
    
    # Test 7: Updated Sales Forecasting
    print("\n7. TESTING UPDATED SALES FORECASTING TOOL")
    print("-" * 50)
    try:
        result = forecast_sales._run(data_path, periods=30)
        print(result)
        print("✅ Sales Forecasting: SUCCESS")
    except Exception as e:
        print(f"❌ Sales Forecasting: FAILED - {e}")
    
    print("\n" + "="*80)
    print("TOOL TESTING COMPLETED")
    print("="*80)

if __name__ == "__main__":
    test_beverage_tools()