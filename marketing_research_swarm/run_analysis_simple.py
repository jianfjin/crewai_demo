#!/usr/bin/env python3
"""
Simple analysis runner that uses only the custom tools (no Code Interpreter)
"""

import sys
import os
sys.path.append('src')

from marketing_research_swarm.crew import MarketingResearchCrew

def main():
    """
    Run a simplified marketing research analysis using only custom tools
    """
    
    # Define the data path for beverage sales
    data_path = "data/beverage_sales.csv"
    
    # Complete inputs for the marketing research analysis
    inputs = {
        "target_audience": "health-conscious millennials and premium beverage consumers",
        "campaign_type": "multi-channel global marketing campaign",
        "budget": 250000,
        "duration": "12 months",
        "data_file_path": data_path,
        "analysis_focus": "global beverage market performance and brand optimization",
        "business_objective": "Optimize beverage portfolio performance across global markets",
        "key_metrics": ["brand_performance", "category_trends", "regional_dynamics", "profitability_analysis"],
        "competitive_landscape": "global beverage market with diverse categories",
        "market_segments": ["North America", "Europe", "Asia Pacific", "Latin America", "Middle East", "Africa", "Australia"],
        "product_categories": ["Cola", "Juice", "Energy", "Sports", "Water"],
        "brands": ["Coca-Cola", "Pepsi", "Red Bull", "Monster Energy", "Gatorade"],
        "campaign_goals": [
            "Optimize brand portfolio performance",
            "Identify high-margin opportunities",
            "Develop pricing strategies",
            "Create targeted marketing strategies"
        ]
    }
    
    # Configuration paths
    agents_config_path = 'src/marketing_research_swarm/config/agents.yaml'
    tasks_config_path = 'src/marketing_research_swarm/config/tasks.yaml'
    
    print("Starting Simplified Marketing Research Analysis...")
    print(f"Analyzing data from: {data_path}")
    print(f"Target Audience: {inputs['target_audience']}")
    print(f"Campaign Budget: ${inputs['budget']:,}")
    print("-" * 60)
    
    try:
        # Initialize and run the crew
        crew = MarketingResearchCrew(agents_config_path, tasks_config_path)
        result = crew.kickoff(inputs)
        
        print("\n" + "="*60)
        print("MARKETING RESEARCH ANALYSIS COMPLETED!")
        print("="*60)
        print("\nANALYSIS RESULTS:")
        print("-" * 40)
        print(result)
        
        return result
        
    except Exception as e:
        print(f"Error running marketing research analysis: {str(e)}")
        print("Please check your configuration files and data path.")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()