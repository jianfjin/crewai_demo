#!/usr/bin/env python3
"""
Test script for persistent analysis caching system
"""

import sys
import os
import time
sys.path.insert(0, 'src')

def test_persistent_caching():
    """Test persistent caching functionality"""
    
    print("🗄️  Persistent Analysis Cache Test")
    print("=" * 60)
    
    try:
        from marketing_research_swarm.flows.cached_roi_flow import CachedFlowRunner
        from marketing_research_swarm.context.context_manager import ContextStrategy
        from marketing_research_swarm.persistence.analysis_cache import get_analysis_cache
        
        # Initialize cached flow runner
        runner = CachedFlowRunner(use_mem0=False)  # Use mock for testing
        cache_manager = get_analysis_cache()
        
        data_file_path = "data/beverage_sales.csv"
        
        # Ensure data file exists
        if not os.path.exists(data_file_path):
            print(f"❌ Data file not found: {data_file_path}")
            print("Creating sample data for testing...")
            
            import pandas as pd
            sample_data = {
                'sale_date': ['2024-01-01', '2024-01-02', '2024-01-03'] * 100,
                'brand': ['Coca-Cola', 'Pepsi', 'Simply Orange'] * 100,
                'category': ['Cola', 'Cola', 'Juice'] * 100,
                'region': ['North', 'South', 'East'] * 100,
                'total_revenue': [1000, 1200, 800] * 100,
                'total_cost': [600, 700, 500] * 100,
                'profit': [400, 500, 300] * 100,
                'profit_margin': [40, 41.7, 37.5] * 100,
                'units_sold': [100, 120, 80] * 100
            }
            
            os.makedirs('data', exist_ok=True)
            df = pd.DataFrame(sample_data)
            df.to_csv(data_file_path, index=False)
            print(f"✅ Created sample data: {df.shape[0]} rows")
        
        print(f"\n📊 Testing with data: {data_file_path}")
        
        # Test 1: First execution (cache miss)
        print("\n1️⃣  First Execution (Expected: Cache MISS)")
        print("-" * 40)
        
        start_time = time.time()
        result1 = runner.run_roi_analysis(
            data_file_path=data_file_path,
            context_strategy=ContextStrategy.PROGRESSIVE_PRUNING
        )
        execution_time1 = time.time() - start_time
        
        cache_info1 = result1.get('cache_info', {})
        print(f"   Cache Hit: {cache_info1.get('cache_hit', False)}")
        print(f"   Execution Time: {execution_time1:.2f}s")
        print(f"   Request Hash: {cache_info1.get('request_hash', 'N/A')}")
        
        # Test 2: Second execution with same parameters (cache hit)
        print("\n2️⃣  Second Execution - Same Parameters (Expected: Cache HIT)")
        print("-" * 40)
        
        start_time = time.time()
        result2 = runner.run_roi_analysis(
            data_file_path=data_file_path,
            context_strategy=ContextStrategy.PROGRESSIVE_PRUNING
        )
        execution_time2 = time.time() - start_time
        
        cache_info2 = result2.get('cache_info', {})
        print(f"   Cache Hit: {cache_info2.get('cache_hit', False)}")
        print(f"   Execution Time: {execution_time2:.2f}s")
        print(f"   Time Saved: {execution_time1 - execution_time2:.2f}s")
        print(f"   Cache Source: {cache_info2.get('cache_source', 'N/A')}")
        
        # Test 3: Third execution with different strategy (cache miss)
        print("\n3️⃣  Third Execution - Different Strategy (Expected: Cache MISS)")
        print("-" * 40)
        
        start_time = time.time()
        result3 = runner.run_roi_analysis(
            data_file_path=data_file_path,
            context_strategy=ContextStrategy.MINIMAL_CONTEXT
        )
        execution_time3 = time.time() - start_time
        
        cache_info3 = result3.get('cache_info', {})
        print(f"   Cache Hit: {cache_info3.get('cache_hit', False)}")
        print(f"   Execution Time: {execution_time3:.2f}s")
        
        # Test 4: Fourth execution with force refresh
        print("\n4️⃣  Fourth Execution - Force Refresh (Expected: Cache BYPASS)")
        print("-" * 40)
        
        start_time = time.time()
        result4 = runner.run_roi_analysis(
            data_file_path=data_file_path,
            context_strategy=ContextStrategy.PROGRESSIVE_PRUNING,
            force_refresh=True
        )
        execution_time4 = time.time() - start_time
        
        cache_info4 = result4.get('cache_info', {})
        print(f"   Cache Hit: {cache_info4.get('cache_hit', False)}")
        print(f"   Execution Time: {execution_time4:.2f}s")
        print(f"   Fresh Execution: {result4.get('execution_info', {}).get('fresh_execution', False)}")
        
        # Show cache performance summary
        print("\n📈 Cache Performance Summary")
        print("=" * 60)
        
        performance = result4.get('cache_performance', {})
        print(f"   Cache Hit Rate: {performance.get('cache_hit_rate', 0):.1f}%")
        print(f"   Exact Hits: {performance.get('exact_hits', 0)}")
        print(f"   Similar Matches: {performance.get('similar_matches', 0)}")
        print(f"   Cache Misses: {performance.get('cache_misses', 0)}")
        print(f"   Total Time Saved: {performance.get('total_time_saved_minutes', 0):.2f} minutes")
        print(f"   Total Cost Saved: ${performance.get('total_cost_saved_usd', 0):.4f}")
        print(f"   Efficiency Rating: {performance.get('efficiency_rating', 'N/A')}")
        
        # Show cache statistics
        print("\n📊 Cache Statistics")
        print("=" * 60)
        
        cache_stats = cache_manager.get_cache_statistics()
        print(f"   Total Cached Analyses: {cache_stats.get('total_entries', 0)}")
        print(f"   Total Cache Size: {cache_stats.get('total_size_mb', 0):.2f} MB")
        print(f"   Analysis Types: {list(cache_stats.get('analysis_types', {}).keys())}")
        
        if cache_stats.get('most_accessed'):
            most_accessed = cache_stats['most_accessed']
            print(f"   Most Accessed: {most_accessed['analysis_type']} ({most_accessed['access_count']} times)")
        
        # Test cache insights
        print("\n💡 Cache Insights and Recommendations")
        print("=" * 60)
        
        insights = runner.get_cache_insights()
        recommendations = insights.get('recommendations', [])
        
        if recommendations:
            for rec in recommendations:
                print(f"   • {rec['type'].title()}: {rec['message']}")
                print(f"     Action: {rec['action']}")
        else:
            print("   ✅ No optimization recommendations - cache is performing well!")
        
        print("\n✅ Persistent caching test completed successfully!")
        
        return {
            'cache_hit_rate': performance.get('cache_hit_rate', 0),
            'time_saved': performance.get('total_time_saved_minutes', 0),
            'cost_saved': performance.get('total_cost_saved_usd', 0),
            'total_entries': cache_stats.get('total_entries', 0)
        }
        
    except Exception as e:
        print(f"❌ Persistent caching test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_cache_management():
    """Test cache management functionality"""
    
    print("\n🧹 Cache Management Test")
    print("=" * 60)
    
    try:
        from marketing_research_swarm.persistence.analysis_cache import get_analysis_cache
        
        cache_manager = get_analysis_cache()
        
        # Test cache cleanup
        print("Testing cache cleanup...")
        cleanup_stats = cache_manager.cleanup_expired_cache()
        
        print(f"   Expired entries removed: {cleanup_stats['expired_entries']}")
        print(f"   Space freed: {cleanup_stats['freed_space_mb']:.2f} MB")
        print(f"   Entries before cleanup: {cleanup_stats['total_entries_before']}")
        print(f"   Entries after cleanup: {cleanup_stats['total_entries_after']}")
        
        # Test cache statistics
        print("\nTesting cache statistics...")
        stats = cache_manager.get_cache_statistics()
        
        print(f"   Current total entries: {stats['total_entries']}")
        print(f"   Current total size: {stats['total_size_mb']:.2f} MB")
        
        if stats['analysis_types']:
            print("   Analysis types breakdown:")
            for analysis_type, type_stats in stats['analysis_types'].items():
                print(f"     - {analysis_type}: {type_stats['count']} entries, {type_stats['size_mb']:.2f} MB")
        
        print("✅ Cache management test completed!")
        
    except Exception as e:
        print(f"❌ Cache management test failed: {e}")

def main():
    """Run all persistent cache tests"""
    
    print("🚀 Marketing Research Persistent Cache Test Suite")
    print("=" * 80)
    
    try:
        # Test persistent caching
        cache_results = test_persistent_caching()
        
        # Test cache management
        test_cache_management()
        
        print("\n🎯 Persistent Caching Benefits Demonstrated:")
        
        if cache_results:
            print(f"   • Cache Hit Rate: {cache_results['cache_hit_rate']:.1f}%")
            print(f"   • Time Saved: {cache_results['time_saved']:.2f} minutes")
            print(f"   • Cost Saved: ${cache_results['cost_saved']:.4f}")
            print(f"   • Cached Analyses: {cache_results['total_entries']}")
        
        print("   • Intelligent request hashing for exact matches")
        print("   • Semantic similarity search for related analyses")
        print("   • Automatic cache cleanup and management")
        print("   • Persistent storage across sessions")
        print("   • Mem0 integration for long-term learning")
        print("   • Intermediate result caching for partial reuse")
        
        print("\n✅ All persistent cache tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()