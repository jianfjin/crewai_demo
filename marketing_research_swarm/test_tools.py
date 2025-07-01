#!/usr/bin/env python3
"""
Test script to verify that all tools are working correctly
"""

import sys
import os
sys.path.append('src')

from marketing_research_swarm.tools.advanced_tools import (
    calculate_roi,
    analyze_kpis,
    forecast_sales,
    plan_budget,
    analyze_brand_performance,
    calculate_market_share,
    time_series_analysis,
    cross_sectional_analysis,
)

def test_tools():
    print("Testing Marketing Research Tools...")
    print("=" * 50)
    
    # Test ROI calculation
    print("\n1. Testing ROI Calculator:")
    roi_result = calculate_roi._run(revenue=100000, cost=75000)
    print(roi_result)
    
    # Test KPI analysis
    print("\n2. Testing KPI Analysis:")
    kpi_result = analyze_kpis._run(
        clicks=1000, 
        impressions=10000, 
        conversions=50, 
        cost=5000, 
        revenue=15000
    )
    print(kpi_result)
    
    # Test sales forecasting
    print("\n3. Testing Sales Forecasting:")
    forecast_result = forecast_sales._run('data/beverage_sales.csv', periods=7)
    print(forecast_result)
    
    # Test budget planning
    print("\n4. Testing Budget Planning:")
    budget_result = plan_budget._run(total_budget=100000)
    print(budget_result)
    
    # Test brand performance
    print("\n5. Testing Brand Performance Analysis:")
    brand_result = analyze_brand_performance._run(
        brand_awareness=65,
        sentiment_score=7.5,
        market_position=3
    )
    print(brand_result)
    
    # Test market share calculation
    print("\n6. Testing Market Share Analysis:")
    market_result = calculate_market_share._run(
        company_revenue=2500000,
        total_market_revenue=10000000,
        competitors={
            'Competitor A': 3000000,
            'Competitor B': 2000000,
            'Competitor C': 2500000
        }
    )
    print(market_result)
    
    # Test time series analysis
    print("\n7. Testing Time Series Analysis:")
    ts_result = time_series_analysis._run('data/beverage_sales.csv')
    print(ts_result)
    
    # Test cross-sectional analysis
    print("\n8. Testing Cross-Sectional Analysis:")
    cs_result = cross_sectional_analysis._run('data/beverage_sales.csv', 'Product', 'Sales')
    print(cs_result)
    
    print("\n" + "=" * 50)
    print("All tools tested successfully!")

if __name__ == "__main__":
    test_tools()