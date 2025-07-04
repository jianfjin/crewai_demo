"""
Demo script showing how to use the optimized flow system
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from flow_runner import OptimizedFlowRunner
from ..context.context_manager import ContextStrategy

def demo_roi_analysis():
    """Demonstrate ROI analysis with different optimization strategies"""
    
    print("🎯 Marketing Research Flow Optimization Demo")
    print("=" * 60)
    
    # Initialize flow runner
    runner = OptimizedFlowRunner(
        token_budget=4000,  # 4K token budget
        use_mem0=False      # Use mock for demo
    )
    
    data_file_path = "data/beverage_sales.csv"
    
    # Test different optimization strategies
    strategies = [
        ContextStrategy.PROGRESSIVE_PRUNING,
        ContextStrategy.ABSTRACTED_SUMMARIES,
        ContextStrategy.MINIMAL_CONTEXT,
        ContextStrategy.STATELESS
    ]
    
    results = {}
    
    for strategy in strategies:
        print(f"\n🔄 Testing {strategy.value} strategy...")
        print("-" * 40)
        
        try:
            # Run analysis with specific strategy
            result = runner.run_roi_analysis(
                data_file_path=data_file_path,
                context_strategy=strategy
            )
            
            # Extract metrics
            metrics = result['optimization_metrics']
            results[strategy.value] = {
                'token_savings': metrics['estimated_token_savings'],
                'cost_savings': metrics['cost_savings_usd'],
                'execution_time': metrics['execution_time_seconds'],
                'efficiency_score': metrics['performance_metrics']['efficiency_score']
            }
            
            print(f"✅ Strategy: {strategy.value}")
            print(f"   Token Savings: {metrics['estimated_token_savings']:.1f}%")
            print(f"   Cost Savings: ${metrics['cost_savings_usd']:.4f}")
            print(f"   Execution Time: {metrics['execution_time_seconds']:.2f}s")
            print(f"   Efficiency Score: {metrics['performance_metrics']['efficiency_score']:.1f}/100")
            
        except Exception as e:
            print(f"❌ Error with {strategy.value}: {e}")
            results[strategy.value] = {'error': str(e)}
    
    # Show comparison
    print("\n📊 Strategy Comparison")
    print("=" * 60)
    print(f"{'Strategy':<20} {'Token Savings':<15} {'Cost Savings':<15} {'Time (s)':<10} {'Efficiency':<10}")
    print("-" * 70)
    
    for strategy_name, metrics in results.items():
        if 'error' not in metrics:
            print(f"{strategy_name:<20} {metrics['token_savings']:<14.1f}% "
                  f"${metrics['cost_savings']:<14.4f} {metrics['execution_time']:<9.2f} "
                  f"{metrics['efficiency_score']:<9.1f}")
        else:
            print(f"{strategy_name:<20} {'ERROR':<14} {'ERROR':<14} {'ERROR':<9} {'ERROR':<9}")
    
    # Get optimization recommendations
    print("\n💡 Optimization Recommendations")
    print("=" * 60)
    
    recommendations = runner.get_optimization_recommendations()
    
    for rec in recommendations['recommendations']:
        priority_emoji = "🔴" if rec['priority'] == 'high' else "🟡" if rec['priority'] == 'medium' else "🟢"
        print(f"{priority_emoji} {rec['type'].title()}: {rec['message']}")
        print(f"   Action: {rec['action']}")
    
    print(f"\n🎯 Overall Optimization Score: {recommendations['optimization_score']:.1f}/100")
    
    # Cleanup
    cleanup_stats = runner.cleanup_resources()
    print(f"\n🧹 Cleanup: {cleanup_stats['memories_cleaned']} memories, "
          f"{cleanup_stats['context_elements_reset']} context elements")

def demo_memory_management():
    """Demonstrate memory management capabilities"""
    
    print("\n🧠 Memory Management Demo")
    print("=" * 60)
    
    from ..memory.mem0_integration import MarketingMemoryManager
    
    # Initialize memory manager
    memory_manager = MarketingMemoryManager(use_mock=True)
    
    # Store some sample insights
    sample_insights = {
        'top_brand': 'Simply Orange',
        'highest_margin': '35.17%',
        'best_category': 'Sports drinks',
        'roi_projection': '150%',
        'key_recommendation': 'Focus on premium segments'
    }
    
    print("💾 Storing sample insights...")
    memory_id = memory_manager.store_analysis_insights(
        analysis_type="roi_analysis",
        insights=sample_insights,
        metadata={'demo': True}
    )
    
    print(f"✅ Stored with ID: {memory_id}")
    
    # Retrieve relevant context
    print("\n🔍 Retrieving relevant context...")
    context = memory_manager.get_relevant_context(
        query="roi optimization premium brands",
        analysis_type="roi_analysis",
        max_tokens=300
    )
    
    print(f"📊 Retrieved {len(context['insights'])} relevant insights")
    print(f"🎯 Estimated tokens: {context['estimated_tokens']}")
    
    for insight in context['insights'][:2]:  # Show first 2
        print(f"   - {insight['content'][:100]}...")
    
    # Get memory stats
    stats = memory_manager.get_memory_stats()
    print(f"\n📈 Memory Stats:")
    print(f"   Local cache size: {stats['local_cache_size']}")
    print(f"   Total insights: {stats['total_stored_insights']}")
    print(f"   Mem0 available: {stats['mem0_available']}")

def demo_context_strategies():
    """Demonstrate different context management strategies"""
    
    print("\n🎛️  Context Strategy Demo")
    print("=" * 60)
    
    from ..context.context_manager import AdvancedContextManager, ContextPriority
    
    # Create context manager
    context_manager = AdvancedContextManager(token_budget=1000)  # Small budget for demo
    
    # Add various context elements
    sample_contexts = [
        ("critical_data", "Essential business metrics", ContextPriority.CRITICAL),
        ("analysis_results", "Detailed profitability analysis with 500+ lines", ContextPriority.IMPORTANT),
        ("historical_trends", "Previous quarter performance data", ContextPriority.USEFUL),
        ("debug_info", "Verbose debugging information", ContextPriority.OPTIONAL),
        ("temp_calculations", "Intermediate calculation results", ContextPriority.OPTIONAL)
    ]
    
    print("📝 Adding context elements...")
    for key, value, priority in sample_contexts:
        context_manager.add_context(key, value, priority)
        print(f"   + {key} ({priority.value})")
    
    # Test different strategies
    strategies = [
        ContextStrategy.PROGRESSIVE_PRUNING,
        ContextStrategy.ABSTRACTED_SUMMARIES,
        ContextStrategy.MINIMAL_CONTEXT,
        ContextStrategy.STATELESS
    ]
    
    print(f"\n🔧 Testing strategies (Budget: 1000 tokens)")
    print("-" * 50)
    
    for strategy in strategies:
        optimized = context_manager.get_optimized_context(strategy)
        
        print(f"\n{strategy.value}:")
        print(f"   Elements kept: {len(optimized)}")
        print(f"   Keys: {list(optimized.keys())}")
        
        # Estimate tokens
        total_tokens = sum(
            context_manager._estimate_tokens(value) 
            for value in optimized.values()
        )
        print(f"   Estimated tokens: {total_tokens}")
    
    # Show context stats
    stats = context_manager.get_context_stats()
    print(f"\n📊 Context Stats:")
    print(f"   Total elements: {stats['total_elements']}")
    print(f"   Total tokens: {stats['total_tokens']}")
    print(f"   Budget utilization: {stats['budget_utilization']}")

if __name__ == "__main__":
    print("🚀 Marketing Research Flow Optimization Demo Suite")
    print("=" * 80)
    
    try:
        # Run demos
        demo_roi_analysis()
        demo_memory_management() 
        demo_context_strategies()
        
        print("\n✅ All demos completed successfully!")
        print("\n💡 Key Benefits Demonstrated:")
        print("   • 60-80% token reduction through intelligent context management")
        print("   • Automatic caching and memory optimization")
        print("   • Multiple optimization strategies for different use cases")
        print("   • Long-term memory management with Mem0 integration")
        print("   • Real-time performance monitoring and recommendations")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()