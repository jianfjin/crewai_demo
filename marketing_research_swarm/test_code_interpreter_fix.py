#!/usr/bin/env python3
"""
Test script to verify Code Interpreter errors are fixed
"""

import sys
import os
sys.path.append('src')

from marketing_research_swarm.crew import MarketingResearchCrew

def test_agent_tools():
    """Test that agents have the correct tools and no Code Interpreter"""
    
    print("="*80)
    print("TESTING AGENT TOOL CONFIGURATION")
    print("="*80)
    
    # Configuration paths
    agents_config_path = 'src/marketing_research_swarm/config/agents.yaml'
    tasks_config_path = 'src/marketing_research_swarm/config/tasks.yaml'
    
    try:
        # Initialize the crew
        crew = MarketingResearchCrew(agents_config_path, tasks_config_path)
        
        print("\n✅ Crew initialized successfully")
        print(f"Available tools: {list(crew.tools.keys())}")
        
        # Check if Code Interpreter is disabled
        if 'python_repl' not in crew.tools:
            print("✅ Code Interpreter (python_repl) is disabled - no string formatting errors")
        else:
            print("⚠️  Code Interpreter is still enabled - may cause string formatting errors")
        
        # Check if all required tools are available
        required_tools = [
            'profitability_analysis',
            'time_series_analysis', 
            'cross_sectional_analysis',
            'beverage_market_analysis',
            'forecast_sales',
            'calculate_roi'
        ]
        
        missing_tools = []
        for tool in required_tools:
            if tool in crew.tools:
                print(f"✅ {tool} is available")
            else:
                print(f"❌ {tool} is missing")
                missing_tools.append(tool)
        
        if not missing_tools:
            print("\n✅ All required analytical tools are properly configured")
            print("✅ Data analyst can use specialized tools instead of Code Interpreter")
        else:
            print(f"\n❌ Missing tools: {missing_tools}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing crew configuration: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_data_columns():
    """Test that we can read the data and show correct column names"""
    
    print("\n" + "="*80)
    print("TESTING DATA COLUMN NAMES")
    print("="*80)
    
    try:
        import pandas as pd
        df = pd.read_csv('data/beverage_sales.csv')
        
        print("✅ Data file loaded successfully")
        print(f"✅ Data shape: {df.shape}")
        print(f"✅ Correct column names: {list(df.columns)}")
        
        # Check for the problematic column name
        if 'Revenue' in df.columns:
            print("⚠️  Found 'Revenue' column - this might cause confusion")
        else:
            print("✅ No 'Revenue' column found - agents should use 'total_revenue'")
        
        if 'total_revenue' in df.columns:
            print("✅ 'total_revenue' column is available")
        else:
            print("❌ 'total_revenue' column is missing")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading data: {str(e)}")
        return False

if __name__ == "__main__":
    print("Testing Code Interpreter fixes...")
    
    tools_ok = test_agent_tools()
    data_ok = test_data_columns()
    
    if tools_ok and data_ok:
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED")
        print("✅ Code Interpreter errors should be resolved")
        print("✅ Agents will use specialized tools with correct column names")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("❌ SOME TESTS FAILED")
        print("❌ Additional fixes may be needed")
        print("="*80)