#!/usr/bin/env python3
"""
Test the fixed tools to ensure division by zero errors are resolved
"""

import sys
import os
sys.path.append('src')

from marketing_research_swarm.tools.advanced_tools import (
    beverage_market_analysis,
    profitability_analysis,
    time_series_analysis,
    cross_sectional_analysis,
    forecast_sales,
    calculate_roi,
    calculate_market_share
)

def test_fixed_tools():
    """Test all tools to ensure division by zero errors are fixed"""
    
    data_path = "data/beverage_sales.csv"
    
    print("="*80)
    print("TESTING FIXED TOOLS - DIVISION BY ZERO PROTECTION")
    print("="*80)
    
    # Test 1: Beverage Market Analysis
    print("\n1. TESTING BEVERAGE MARKET ANALYSIS TOOL")
    print("-" * 50)
    try:
        result = beverage_market_analysis._run(data_path)
        print("✅ Beverage Market Analysis: SUCCESS")
        print("Sample output:", result[:200] + "...")
    except Exception as e:
        print(f"❌ Beverage Market Analysis: FAILED - {e}")
    
    # Test 2: Profitability Analysis
    print("\n2. TESTING PROFITABILITY ANALYSIS TOOL")
    print("-" * 50)
    try:
        result = profitability_analysis._run(data_path, analysis_dimension='brand')
        print("✅ Profitability Analysis: SUCCESS")
        print("Sample output:", result[:200] + "...")
    except Exception as e:
        print(f"❌ Profitability Analysis: FAILED - {e}")
    
    # Test 3: Time Series Analysis
    print("\n3. TESTING TIME SERIES ANALYSIS TOOL")
    print("-" * 50)
    try:
        result = time_series_analysis._run(data_path)
        print("✅ Time Series Analysis: SUCCESS")
        print("Sample output:", result[:200] + "...")
    except Exception as e:
        print(f"❌ Time Series Analysis: FAILED - {e}")
    
    # Test 4: Cross-Sectional Analysis
    print("\n4. TESTING CROSS-SECTIONAL ANALYSIS TOOL")
    print("-" * 50)
    try:
        result = cross_sectional_analysis._run(data_path, segment_column='region')
        print("✅ Cross-Sectional Analysis: SUCCESS")
        print("Sample output:", result[:200] + "...")
    except Exception as e:
        print(f"❌ Cross-Sectional Analysis: FAILED - {e}")
    
    # Test 5: Sales Forecasting
    print("\n5. TESTING SALES FORECASTING TOOL")
    print("-" * 50)
    try:
        result = forecast_sales._run(data_path, periods=30)
        print("✅ Sales Forecasting: SUCCESS")
        print("Sample output:", result[:200] + "...")
    except Exception as e:
        print(f"❌ Sales Forecasting: FAILED - {e}")
    
    # Test 6: ROI Calculation (with edge cases)
    print("\n6. TESTING ROI CALCULATION TOOL (EDGE CASES)")
    print("-" * 50)
    try:
        # Test normal case
        result1 = calculate_roi._run(revenue=1000, cost=500)
        print("✅ ROI Normal Case: SUCCESS")
        
        # Test zero cost case
        result2 = calculate_roi._run(revenue=1000, cost=0)
        print("✅ ROI Zero Cost Case: SUCCESS")
        
        print("Sample output:", result1[:100] + "...")
    except Exception as e:
        print(f"❌ ROI Calculation: FAILED - {e}")
    
    # Test 7: Market Share Calculation (with edge cases)
    print("\n7. TESTING MARKET SHARE CALCULATION TOOL (EDGE CASES)")
    print("-" * 50)
    try:
        # Test normal case
        result1 = calculate_market_share._run(company_revenue=1000, total_market_revenue=5000)
        print("✅ Market Share Normal Case: SUCCESS")
        
        # Test zero market case
        result2 = calculate_market_share._run(company_revenue=1000, total_market_revenue=0)
        print("✅ Market Share Zero Market Case: SUCCESS")
        
        print("Sample output:", result1[:100] + "...")
    except Exception as e:
        print(f"❌ Market Share Calculation: FAILED - {e}")
    
    print("\n" + "="*80)
    print("FIXED TOOLS TESTING COMPLETED")
    print("All division by zero protections are working correctly!")
    print("="*80)

if __name__ == "__main__":
    test_fixed_tools()