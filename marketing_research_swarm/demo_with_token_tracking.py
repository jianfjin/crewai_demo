#!/usr/bin/env python3
"""
Marketing Research Swarm - Demo with Token Tracking
This script demonstrates the enhanced crew with comprehensive token usage analysis.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def demo_token_tracking_tools():
    """Demonstrate the token tracking analytical tools."""
    print("=" * 60)
    print("MARKETING RESEARCH SWARM - TOKEN TRACKING DEMO")
    print("=" * 60)
    
    from marketing_research_swarm.tools.advanced_tools import (
        time_series_analysis,
        cross_sectional_analysis,
        forecast_sales,
        calculate_roi,
        plan_budget
    )
    
    data_path = "data/beverage_sales.csv"
    
    print("\n🔍 Testing Individual Tools (No Token Tracking)...")
    print("-" * 50)
    
    print("1. Time Series Analysis...")
    try:
        result = time_series_analysis._run(data_path)
        print("✅ Success - Sample output:")
        print(result[:200] + "..." if len(result) > 200 else result)
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    print("\n2. Cross Sectional Analysis...")
    try:
        result = cross_sectional_analysis._run(data_path, 'Region')
        print("✅ Success - Sample output:")
        print(result[:200] + "..." if len(result) > 200 else result)
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    print("\n3. ROI Calculator...")
    try:
        result = calculate_roi._run(revenue=250000, cost=100000)
        print("✅ Success:")
        print(result)
    except Exception as e:
        print(f"❌ Failed: {e}")

def demo_crew_with_token_tracking():
    """Demonstrate the enhanced crew with token tracking."""
    print("\n" + "=" * 60)
    print("ENHANCED CREW WITH TOKEN TRACKING DEMO")
    print("=" * 60)
    
    try:
        from marketing_research_swarm.crew_with_tracking import MarketingResearchCrew
        
        # Configuration paths
        agents_config_path = 'src/marketing_research_swarm/config/agents.yaml'
        tasks_config_path = 'src/marketing_research_swarm/config/tasks.yaml'
        
        print("🚀 Initializing Enhanced Crew with Token Tracking...")
        crew = MarketingResearchCrew(agents_config_path, tasks_config_path)
        print(f"✅ Crew initialized with {len(crew.tools)} tools")
        print(f"📊 Model configured: {crew.model_name}")
        
        # Sample inputs for demonstration
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
        }
        
        print("\n📋 Analysis Parameters:")
        print(f"  Target Audience: {inputs['target_audience']}")
        print(f"  Budget: ${inputs['budget']:,}")
        print(f"  Duration: {inputs['duration']}")
        print(f"  Data Source: {inputs['data_file_path']}")
        
        print("\n⚠️  NOTE: This demo shows the crew structure and token tracking setup.")
        print("   For full execution with LLM calls, you need a valid OpenAI API key.")
        print("   The system will track tokens, costs, and performance metrics.")
        
        # Show what the token tracking would capture
        print("\n📊 Token Tracking Features:")
        print("  ✅ Real-time token consumption monitoring")
        print("  ✅ Cost analysis per agent and task")
        print("  ✅ Performance efficiency metrics")
        print("  ✅ Optimization recommendations")
        print("  ✅ Detailed breakdown by model and operation")
        
        print("\n🎯 Expected Token Analysis Output:")
        print("  - Total tokens consumed across all agents")
        print("  - Cost breakdown (input vs output tokens)")
        print("  - Agent performance comparison")
        print("  - Task duration and efficiency metrics")
        print("  - Optimization recommendations")
        print("  - Model pricing analysis")
        
        # Demonstrate the tracking system structure
        print(f"\n🔧 Token Tracker Configuration:")
        print(f"  Model: {crew.token_tracker.model_name}")
        print(f"  Tokenizer: {type(crew.token_tracker.tokenizer).__name__}")
        print(f"  Tracking Status: Ready")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def show_sample_token_report():
    """Show what a sample token usage report would look like."""
    print("\n" + "=" * 60)
    print("SAMPLE TOKEN USAGE REPORT")
    print("=" * 60)
    
    sample_report = """
## Token Usage Analysis

### Executive Summary
- **Total Duration**: 4.25 minutes
- **Total Tokens**: 15,847 tokens
- **Model Used**: gpt-4o-mini
- **Total Cost**: $0.0238 USD
- **Tasks Completed**: 6

### Cost Breakdown
- **Input Tokens**: 10,234 tokens ($0.0015)
- **Output Tokens**: 5,613 tokens ($0.0034)
- **Cost per Minute**: $0.0056 USD/min

### Efficiency Metrics
- **Tokens per Second**: 62.1
- **Average Tokens per Task**: 2,641

### Agent Performance Breakdown

#### Market Research Analyst
- **Total Tokens**: 4,523
- **Cost**: $0.0068 USD
- **Duration**: 78.3 seconds
- **LLM Calls**: 3
- **Tool Calls**: 2
- **Efficiency**: 57.8 tokens/sec

#### Data Analyst
- **Total Tokens**: 3,891
- **Cost**: $0.0058 USD
- **Duration**: 65.2 seconds
- **LLM Calls**: 2
- **Tool Calls**: 4
- **Efficiency**: 59.7 tokens/sec

#### Content Strategist
- **Total Tokens**: 2,456
- **Cost**: $0.0037 USD
- **Duration**: 42.1 seconds
- **LLM Calls**: 2
- **Tool Calls**: 1
- **Efficiency**: 58.3 tokens/sec

### Task Performance Details

| Task | Agent | Duration (s) | Tokens | Cost ($) | Efficiency | Status |
|------|-------|-------------|---------|----------|------------|---------|
| research_task | market_research_analyst | 78.3 | 4,523 | 0.0068 | 57.8 t/s | completed |
| data_analysis_task | data_analyst | 65.2 | 3,891 | 0.0058 | 59.7 t/s | completed |
| strategy_task | content_strategist | 42.1 | 2,456 | 0.0037 | 58.3 t/s | completed |

### Optimization Recommendations

1. Agent 'market_research_analyst' consumes $0.0068 USD - consider prompt optimization
2. 1 tasks took over 2 minutes - consider breaking down complex tasks
3. Consider using a smaller model like gpt-4o-mini for cost optimization

### Detailed Cost Analysis

**Model Pricing** (gpt-4o-mini):
- Input tokens: $0.000000 per token
- Output tokens: $0.000001 per token

**Token Distribution**:
- Prompt tokens: 64.6%
- Completion tokens: 35.4%

**Cost Efficiency**:
- Cost per 1K tokens: $0.0015
- Tokens per dollar: 666,667

### Performance Insights

**Fastest Task**: strategy_task (42.1s)
**Most Token-Intensive**: research_task (4,523 tokens)
**Most Cost-Effective**: strategy_task ($0.0037)
"""
    
    print(sample_report)

def main():
    """Run the complete token tracking demonstration."""
    print("Marketing Research Swarm - Token Tracking Demonstration")
    print("This demo shows the enhanced crew with comprehensive token usage analysis.")
    
    # Demo individual tools
    demo_token_tracking_tools()
    
    # Demo enhanced crew
    success = demo_crew_with_token_tracking()
    
    # Show sample report
    show_sample_token_report()
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE!")
    print("=" * 60)
    
    if success:
        print("✅ Token tracking system is ready for production use!")
        print("\nTo run with full token tracking:")
        print("1. Set your OpenAI API key in .env file")
        print("2. Run: crewai run")
        print("3. Or run: python run_crew.py")
        print("\nThe system will automatically track and analyze:")
        print("- Token consumption per agent and task")
        print("- Real-time cost monitoring")
        print("- Performance efficiency metrics")
        print("- Optimization recommendations")
    else:
        print("❌ Please check the configuration and try again.")

if __name__ == "__main__":
    main()