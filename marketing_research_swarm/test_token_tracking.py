#!/usr/bin/env python3
"""
Test script for token tracking functionality.
This script tests the token tracking system without requiring API keys.
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_token_tracker():
    """Test the token tracking utilities."""
    print("Testing Token Tracking System...")
    print("=" * 50)
    
    try:
        from marketing_research_swarm.utils.token_tracker import (
            TokenTracker, TokenAnalyzer, TokenUsage, TaskTokenUsage, CrewTokenUsage
        )
        from datetime import datetime
        import uuid
        
        print("✅ Token tracking imports successful")
        
        # Test TokenUsage
        usage1 = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        usage2 = TokenUsage(prompt_tokens=200, completion_tokens=75, total_tokens=275)
        combined = usage1 + usage2
        
        print(f"✅ TokenUsage math: {usage1.total_tokens} + {usage2.total_tokens} = {combined.total_tokens}")
        
        # Test TokenTracker
        tracker = TokenTracker("gpt-4o-mini")
        print(f"✅ TokenTracker initialized for model: {tracker.model_name}")
        
        # Test crew tracking
        crew_id = str(uuid.uuid4())
        crew_usage = tracker.start_crew_tracking(crew_id)
        print(f"✅ Crew tracking started: {crew_id}")
        
        # Test task tracking
        task_usage = tracker.start_task_tracking("test_task", "test_agent")
        print(f"✅ Task tracking started: {task_usage.task_name}")
        
        # Simulate LLM usage
        test_prompt = "Analyze the marketing data for beverage sales trends."
        test_response = "Based on the analysis, I found increasing sales trends in the South region with Latte products showing 31% higher performance than Green Tea."
        
        token_usage = tracker.record_llm_usage(test_prompt, test_response)
        print(f"✅ LLM usage recorded: {token_usage.total_tokens} tokens")
        
        # Test tool usage
        tracker.record_tool_usage()
        print(f"✅ Tool usage recorded")
        
        # Complete task
        tracker.complete_task("completed")
        print(f"✅ Task completed")
        
        # Complete crew
        final_usage = tracker.complete_crew_tracking()
        print(f"✅ Crew tracking completed")
        
        # Test analysis
        analysis = TokenAnalyzer.analyze_crew_usage(final_usage)
        print(f"✅ Token analysis generated")
        
        # Display results
        print("\n" + "=" * 50)
        print("TOKEN USAGE ANALYSIS RESULTS")
        print("=" * 50)
        
        print(f"Total Duration: {analysis['summary']['total_duration_minutes']:.2f} minutes")
        print(f"Total Tokens: {analysis['summary']['total_tokens']:,}")
        print(f"Estimated Cost: ${analysis['summary']['total_cost']['total_cost']:.4f}")
        print(f"Model Used: {analysis['summary']['model_used']}")
        
        print(f"\nEfficiency Metrics:")
        print(f"- Tokens per second: {analysis['efficiency_metrics']['tokens_per_second']:.2f}")
        print(f"- Cost per minute: ${analysis['efficiency_metrics']['cost_per_minute']:.4f}")
        
        if analysis['recommendations']:
            print(f"\nRecommendations:")
            for i, rec in enumerate(analysis['recommendations'], 1):
                print(f"{i}. {rec}")
        
        print("\n✅ All token tracking tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Token tracking test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_cost_estimation():
    """Test cost estimation for different models."""
    print("\nTesting Cost Estimation...")
    print("-" * 30)
    
    try:
        from marketing_research_swarm.utils.token_tracker import TokenTracker, TokenUsage
        
        # Test different models
        models = ["gpt-4o-mini", "gpt-4o", "gpt-4", "gpt-3.5-turbo"]
        test_usage = TokenUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
        
        for model in models:
            tracker = TokenTracker(model)
            cost = tracker.estimate_cost(test_usage, model)
            print(f"{model:15} | ${cost['total_cost']:.4f} | Input: ${cost['input_cost']:.4f} | Output: ${cost['output_cost']:.4f}")
        
        print("✅ Cost estimation tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Cost estimation test failed: {str(e)}")
        return False

def main():
    """Run all token tracking tests."""
    print("Marketing Research Swarm - Token Tracking Test Suite")
    print("=" * 60)
    
    success = True
    
    # Test basic token tracking
    success &= test_token_tracker()
    
    # Test cost estimation
    success &= test_cost_estimation()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED! Token tracking system is ready.")
    else:
        print("❌ Some tests failed. Please check the implementation.")
    print("=" * 60)

if __name__ == "__main__":
    main()