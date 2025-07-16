"""
Test Script for Performance Optimizations

This script tests all implemented optimizations and provides
performance benchmarks and comparisons.
"""

import asyncio
import time
import json
from typing import Dict, Any
import pandas as pd
import numpy as np

from .optimized_integration import create_optimized_system
from .shared_data_cache import get_shared_cache, clear_global_cache
from .parallel_execution import create_parallel_flow
from .context_optimizer import get_context_optimizer, ContextStrategy
from .performance_profiler import get_profiler

def test_shared_cache_performance():
    """Test shared data cache performance."""
    print("🧪 Testing Shared Data Cache Performance...")
    
    cache = get_shared_cache()
    
    # Test 1: Cache miss (first load)
    start_time = time.time()
    df1, cache_info1 = cache.get_or_load_data()  # Sample data
    first_load_time = time.time() - start_time
    
    print(f"   First load (cache miss): {first_load_time:.3f}s")
    print(f"   Data shape: {df1.shape}")
    print(f"   Cache hit: {cache_info1['cache_hit']}")
    
    # Test 2: Cache hit (second load)
    start_time = time.time()
    df2, cache_info2 = cache.get_or_load_data()  # Same data
    second_load_time = time.time() - start_time
    
    print(f"   Second load (cache hit): {second_load_time:.3f}s")
    print(f"   Cache hit: {cache_info2['cache_hit']}")
    print(f"   Time saved: {cache_info2.get('time_saved_seconds', 0):.3f}s")
    
    # Performance improvement
    improvement = ((first_load_time - second_load_time) / first_load_time) * 100
    print(f"   Performance improvement: {improvement:.1f}%")
    
    # Cache stats
    stats = cache.get_cache_stats()
    print(f"   Cache stats: {stats}")
    
    return {
        'first_load_time': first_load_time,
        'second_load_time': second_load_time,
        'improvement_percent': improvement,
        'cache_stats': stats
    }

async def test_parallel_execution():
    """Test parallel execution performance."""
    print("\n🧪 Testing Parallel Execution Performance...")
    
    parallel_manager, optimized_flow = create_parallel_flow(max_workers=4)
    
    # Create mock tasks for testing
    def mock_agent_task(inputs, duration=1.0):
        time.sleep(duration)  # Simulate work
        return f"Result from agent with inputs: {inputs}"
    
    from .parallel_execution import AgentTask
    
    # Test sequential vs parallel execution
    test_agents = ['market_research_analyst', 'competitive_analyst', 'brand_performance_specialist']
    
    tasks = []
    for i, agent in enumerate(test_agents):
        task = AgentTask(
            agent_id=agent,
            agent_role=agent,
            task_name=f"{agent}_task",
            task_function=lambda inp, dur=1.0 + i*0.5: mock_agent_task(inp, dur),
            inputs={'test': 'data'},
            dependencies=[],
            estimated_duration=1.0 + i*0.5
        )
        tasks.append(task)
    
    # Sequential execution time estimate
    sequential_estimate = sum(task.estimated_duration for task in tasks)
    
    # Parallel execution
    start_time = time.time()
    results = await parallel_manager.execute_tasks_async(tasks)
    parallel_time = time.time() - start_time
    
    # Calculate improvement
    improvement = ((sequential_estimate - parallel_time) / sequential_estimate) * 100
    
    print(f"   Sequential estimate: {sequential_estimate:.2f}s")
    print(f"   Parallel execution: {parallel_time:.2f}s")
    print(f"   Performance improvement: {improvement:.1f}%")
    print(f"   Successful tasks: {len([r for r in results if r.success])}/{len(results)}")
    
    # Parallel stats
    stats = parallel_manager.get_performance_stats()
    print(f"   Parallel stats: {stats}")
    
    return {
        'sequential_estimate': sequential_estimate,
        'parallel_time': parallel_time,
        'improvement_percent': improvement,
        'parallel_stats': stats
    }

def test_context_optimization():
    """Test context optimization performance."""
    print("\n🧪 Testing Context Optimization Performance...")
    
    optimizer = get_context_optimizer()
    
    # Create large mock context
    large_context = {
        'market_data': {f'metric_{i}': np.random.random(1000).tolist() for i in range(10)},
        'sales_data': pd.DataFrame(np.random.random((1000, 10))).to_dict(),
        'competitor_data': {f'competitor_{i}': {'data': list(range(500))} for i in range(20)},
        'historical_trends': {f'trend_{i}': list(range(200)) for i in range(50)},
        'customer_segments': {f'segment_{i}': {'details': f'data_{i}' * 100} for i in range(30)}
    }
    
    # Test different optimization strategies
    strategies = [
        ContextStrategy.FULL_CONTEXT,
        ContextStrategy.ISOLATED_CONTEXT,
        ContextStrategy.REFERENCE_BASED,
        ContextStrategy.SMART_SUMMARY
    ]
    
    results = {}
    
    for strategy in strategies:
        start_time = time.time()
        optimized_context, metrics = optimizer.optimize_context_for_agent(
            agent_role='market_research_analyst',
            full_context=large_context,
            strategy=strategy
        )
        optimization_time = time.time() - start_time
        
        results[strategy.value] = {
            'optimization_time': optimization_time,
            'compression_ratio': metrics.compression_ratio,
            'token_savings': metrics.token_savings_estimated,
            'references_created': metrics.references_created,
            'original_size': metrics.original_size_bytes,
            'optimized_size': metrics.optimized_size_bytes
        }
        
        print(f"   {strategy.value}:")
        print(f"     Compression ratio: {metrics.compression_ratio:.2f}x")
        print(f"     Token savings: {metrics.token_savings_estimated}")
        print(f"     References created: {metrics.references_created}")
        print(f"     Optimization time: {optimization_time:.3f}s")
    
    # Context optimizer stats
    stats = optimizer.get_optimization_stats()
    print(f"   Optimizer stats: {stats}")
    
    return results

def test_integrated_system():
    """Test the complete integrated optimized system."""
    print("\n🧪 Testing Integrated Optimized System...")
    
    # Create optimized system
    system = create_optimized_system(
        max_workers=4,
        enable_mem0=False,  # Disabled for performance
        cache_size=100
    )
    
    # Test workflow
    test_inputs = {
        'data_file_path': None,  # Use sample data
        'target_audience': 'beverage consumers',
        'budget': 100000,
        'duration': '3 months',
        'campaign_goals': 'brand awareness'
    }
    
    selected_agents = ['market_research_analyst', 'competitive_analyst', 'brand_performance_specialist']
    
    async def run_test():
        start_time = time.time()
        
        result = await system.run_optimized_analysis(
            workflow_type='performance_test',
            inputs=test_inputs,
            selected_agents=selected_agents,
            optimization_level='full'
        )
        
        total_time = time.time() - start_time
        
        print(f"   Total execution time: {total_time:.2f}s")
        print(f"   Workflow ID: {result['workflow_id']}")
        print(f"   Agents executed: {result['agents_executed']}")
        
        if result.get('performance_metrics'):
            metrics = result['performance_metrics']
            print(f"   Cache stats: {metrics.get('cache_stats', {})}")
            print(f"   Context stats: {metrics.get('context_stats', {})}")
            print(f"   Parallel stats: {metrics.get('parallel_stats', {})}")
        
        return {
            'total_time': total_time,
            'result': result
        }
    
    return asyncio.run(run_test())

def benchmark_optimizations():
    """Run comprehensive benchmark of all optimizations."""
    print("🚀 Running Comprehensive Performance Benchmark...")
    print("=" * 60)
    
    benchmark_results = {}
    
    # Test 1: Shared Cache
    benchmark_results['shared_cache'] = test_shared_cache_performance()
    
    # Test 2: Parallel Execution
    benchmark_results['parallel_execution'] = asyncio.run(test_parallel_execution())
    
    # Test 3: Context Optimization
    benchmark_results['context_optimization'] = test_context_optimization()
    
    # Test 4: Integrated System
    benchmark_results['integrated_system'] = test_integrated_system()
    
    # Test 5: Performance Profiler
    print("\n🧪 Testing Performance Profiler...")
    profiler = get_profiler()
    
    with profiler.profile_operation("benchmark_test"):
        time.sleep(0.1)  # Simulate work
    
    profiler_stats = profiler.get_performance_summary(time_window_hours=1)
    benchmark_results['profiler_stats'] = profiler_stats
    print(f"   Profiler captured {profiler_stats['total_operations']} operations")
    
    # Generate summary report
    print("\n" + "=" * 60)
    print("📊 PERFORMANCE BENCHMARK SUMMARY")
    print("=" * 60)
    
    # Cache performance
    cache_improvement = benchmark_results['shared_cache']['improvement_percent']
    print(f"✅ Shared Cache: {cache_improvement:.1f}% faster on cache hits")
    
    # Parallel performance
    parallel_improvement = benchmark_results['parallel_execution']['improvement_percent']
    print(f"✅ Parallel Execution: {parallel_improvement:.1f}% faster than sequential")
    
    # Context optimization
    context_results = benchmark_results['context_optimization']
    best_compression = max(
        context_results[strategy]['compression_ratio'] 
        for strategy in context_results 
        if strategy != 'full_context'
    )
    print(f"✅ Context Optimization: Up to {best_compression:.1f}x compression ratio")
    
    # Integrated system
    integrated_time = benchmark_results['integrated_system']['total_time']
    print(f"✅ Integrated System: Complete 3-agent workflow in {integrated_time:.2f}s")
    
    # Overall performance estimate
    estimated_sequential_time = integrated_time * (1 + parallel_improvement/100) * (1 + cache_improvement/100)
    total_improvement = ((estimated_sequential_time - integrated_time) / estimated_sequential_time) * 100
    
    print(f"\n🎯 ESTIMATED OVERALL IMPROVEMENT: {total_improvement:.1f}%")
    print(f"   Optimized time: {integrated_time:.2f}s")
    print(f"   Estimated unoptimized time: {estimated_sequential_time:.2f}s")
    
    return benchmark_results

def export_benchmark_results(results: Dict[str, Any], file_path: str = "performance_benchmark.json"):
    """Export benchmark results to file."""
    try:
        # Convert numpy types to JSON serializable
        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        # Clean results for JSON export
        clean_results = json.loads(json.dumps(results, default=convert_types))
        
        with open(file_path, 'w') as f:
            json.dump(clean_results, f, indent=2)
        
        print(f"\n📁 Benchmark results exported to {file_path}")
        
    except Exception as e:
        print(f"❌ Failed to export benchmark results: {e}")

if __name__ == "__main__":
    # Run the complete benchmark
    results = benchmark_optimizations()
    
    # Export results
    export_benchmark_results(results)
    
    # Clean up
    clear_global_cache()
    print("\n🧹 Cleanup completed")