#!/usr/bin/env python3
"""
Marketing Research Swarm - Demo Script
This script demonstrates the key capabilities of the marketing research system.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def demo_individual_tools():
    """Demonstrate individual analytical tools."""
    print("=" * 60)
    print("MARKETING RESEARCH SWARM - TOOL DEMONSTRATION")
    print("=" * 60)
    
    from marketing_research_swarm.tools.advanced_tools import (
        time_series_analysis,
        cross_sectional_analysis,
        forecast_sales,
        calculate_roi,
        plan_budget
    )
    
    data_path = "data/beverage_sales.csv"
    
    print("\n1. TIME SERIES ANALYSIS")
    print("-" * 30)
    result = time_series_analysis._run(data_path)
    print(result)
    
    print("\n2. REGIONAL PERFORMANCE ANALYSIS")
    print("-" * 30)
    result = cross_sectional_analysis._run(data_path, 'Region')
    print(result)
    
    print("\n3. PRODUCT PERFORMANCE ANALYSIS")
    print("-" * 30)
    result = cross_sectional_analysis._run(data_path, 'Product')
    print(result)
    
    print("\n4. SALES FORECASTING")
    print("-" * 30)
    result = forecast_sales._run(data_path, periods=14)
    print(result)
    
    print("\n5. ROI ANALYSIS")
    print("-" * 30)
    result = calculate_roi._run(revenue=250000, cost=100000)
    print(result)
    
    print("\n6. BUDGET PLANNING")
    print("-" * 30)
    result = plan_budget._run(total_budget=100000)
    print(result)

def demo_crew_setup():
    """Demonstrate crew setup and configuration."""
    print("\n" + "=" * 60)
    print("CREW CONFIGURATION DEMONSTRATION")
    print("=" * 60)
    
    from marketing_research_swarm.crew import MarketingResearchCrew
    
    # Initialize crew
    crew = MarketingResearchCrew(
        'src/marketing_research_swarm/config/agents.yaml',
        'src/marketing_research_swarm/config/tasks.yaml'
    )
    
    print(f"✓ Crew initialized successfully")
    print(f"✓ Number of agents: {len(crew.agents_config)}")
    print(f"✓ Number of tasks: {len(crew.tasks_config)}")
    print(f"✓ Number of tools: {len(crew.tools)}")
    
    print(f"\nAvailable Agents:")
    for agent_name in crew.agents_config.keys():
        print(f"  - {agent_name}")
    
    print(f"\nAvailable Tasks:")
    for task_name in crew.tasks_config.keys():
        print(f"  - {task_name}")
    
    print(f"\nAvailable Tools:")
    for tool_name in crew.tools.keys():
        print(f"  - {tool_name}")

def demo_analysis_inputs():
    """Show the analysis inputs that will be used."""
    print("\n" + "=" * 60)
    print("ANALYSIS CONFIGURATION")
    print("=" * 60)
    
    inputs = {
        "target_audience": "health-conscious millennials interested in premium beverages",
        "campaign_type": "multi-channel digital marketing",
        "budget": 100000,
        "duration": "6 months",
        "data_file_path": "data/beverage_sales.csv",
        "analysis_focus": "beverage market performance and optimization",
        "business_objective": "Launch new organic beverage line and optimize existing product performance",
        "key_metrics": ["sales_trends", "regional_performance", "product_performance", "roi_analysis"],
        "competitive_landscape": "premium beverage market with focus on organic and health-conscious products",
        "market_segments": ["North", "South"],
        "product_lines": ["Green Tea", "Latte"],
        "campaign_goals": [
            "Increase market share in underperforming regions",
            "Optimize product mix based on sales data",
            "Develop targeted marketing strategies for each region",
            "Forecast sales for budget planning",
            "Create compelling content for health-conscious consumers"
        ]
    }
    
    print("Analysis Parameters:")
    for key, value in inputs.items():
        if isinstance(value, list):
            print(f"  {key}:")
            for item in value:
                print(f"    - {item}")
        else:
            print(f"  {key}: {value}")

def main():
    """Run the complete demonstration."""
    print("MARKETING RESEARCH SWARM - COMPLETE DEMONSTRATION")
    print("This demo shows the capabilities of the marketing research system")
    print("without running the full crew analysis (which requires LLM setup).")
    
    # Demo individual tools
    demo_individual_tools()
    
    # Demo crew setup
    demo_crew_setup()
    
    # Demo analysis inputs
    demo_analysis_inputs()
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE!")
    print("=" * 60)
    print("\nTo run the full analysis:")
    print("1. Set up your OpenAI API key in .env file")
    print("2. Run: python run_analysis.py")
    print("3. Or run: python src/marketing_research_swarm/main.py")
    print("\nThe system will generate a comprehensive marketing research report")
    print("with insights, recommendations, and strategic guidance.")

if __name__ == "__main__":
    main()