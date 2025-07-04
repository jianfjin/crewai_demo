#!/usr/bin/env python
"""
Main Entry Point with Persistent Analysis Caching
Provides instant results for repeated requests through intelligent caching
"""

import os
import yaml
import argparse
from datetime import datetime
from typing import Dict, Any

from marketing_research_swarm.flows.cached_roi_flow import CachedFlowRunner
from marketing_research_swarm.context.context_manager import ContextStrategy
from marketing_research_swarm.persistence.analysis_cache import get_analysis_cache

def load_settings():
    """Load configuration settings from settings.yaml"""
    settings_path = 'src/marketing_research_swarm/config/settings.yaml'
    try:
        with open(settings_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        return {
            'data_sources': {'default_data_path': 'data/beverage_sales.csv'},
            'analysis': {
                'default_budget': 100000,
                'default_duration': '6 months',
                'default_target_audience': 'health-conscious millennials and premium beverage consumers'
            },
            'optimization': {
                'token_budget': 4000,
                'default_strategy': 'progressive_pruning',
                'use_mem0': True
            },
            'caching': {
                'enable_persistent_cache': True,
                'cache_ttl_days': 7,
                'max_cache_size_gb': 5,
                'cleanup_interval_hours': 24
            },
            'reports': {'output_directory': 'reports'}
        }

def run_cached_analysis(analysis_type: str = "roi_analysis", 
                       context_strategy: str = "progressive_pruning",
                       force_refresh: bool = False,
                       use_mem0: bool = True,
                       **kwargs) -> Dict[str, Any]:
    """
    Run analysis with intelligent persistent caching
    
    Args:
        analysis_type: Type of analysis (roi_analysis, sales_forecast, brand_performance)
        context_strategy: Context optimization strategy
        force_refresh: Force fresh execution ignoring cache
        use_mem0: Whether to use Mem0 for long-term memory
        **kwargs: Additional parameters
        
    Returns:
        Analysis results with cache information
    """
    
    # Load settings
    settings = load_settings()
    data_path = settings['data_sources']['default_data_path']
    
    # Map string strategy to enum
    strategy_map = {
        'progressive_pruning': ContextStrategy.PROGRESSIVE_PRUNING,
        'abstracted_summaries': ContextStrategy.ABSTRACTED_SUMMARIES,
        'minimal_context': ContextStrategy.MINIMAL_CONTEXT,
        'stateless': ContextStrategy.STATELESS
    }
    
    strategy = strategy_map.get(context_strategy, ContextStrategy.PROGRESSIVE_PRUNING)
    
    # Initialize cached runner
    runner = CachedFlowRunner(use_mem0=use_mem0)
    
    print(f"🚀 Starting Cached {analysis_type.title()} Analysis")
    print(f"📊 Strategy: {strategy.value}")
    print(f"📁 Data Source: {data_path}")
    print(f"🔄 Force Refresh: {force_refresh}")
    print("-" * 60)
    
    try:
        if analysis_type == "roi_analysis":
            result = runner.run_roi_analysis(
                data_file_path=data_path,
                context_strategy=strategy,
                force_refresh=force_refresh,
                **kwargs
            )
        elif analysis_type == "sales_forecast":
            # Placeholder for future implementation
            result = {
                'status': 'not_implemented',
                'message': 'Sales forecast with caching will be implemented in next phase',
                'cache_ready': True
            }
        elif analysis_type == "brand_performance":
            # Placeholder for future implementation
            result = {
                'status': 'not_implemented',
                'message': 'Brand performance with caching will be implemented in next phase',
                'cache_ready': True
            }
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
        
        # Save report if analysis completed successfully
        if 'analysis_results' in result or 'profitability_insights' in result:
            report_path = save_cached_report(result, analysis_type, settings)
            result['report_path'] = report_path
        
        return result
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        raise

def save_cached_report(result: Dict[str, Any], analysis_type: str, settings: Dict[str, Any]) -> str:
    """Save cached analysis report with cache information"""
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create reports directory
    reports_dir = settings.get('reports', {}).get('output_directory', 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    # Generate report filename
    cache_status = "cached" if result.get('cache_info', {}).get('cache_hit') else "fresh"
    report_filename = f"{cache_status}_{analysis_type}_report_{timestamp}.md"
    report_path = os.path.join(reports_dir, report_filename)
    
    # Extract information
    cache_info = result.get('cache_info', {})
    execution_info = result.get('execution_info', {})
    cache_performance = result.get('cache_performance', {})
    execution_stats = result.get('execution_stats', {})
    
    # Format comprehensive report
    report_content = f"""# Cached {analysis_type.title()} Analysis Report

Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Cache Performance Summary

### Cache Status
- **Cache Hit**: {cache_info.get('cache_hit', False)}
- **Cache Source**: {cache_info.get('cache_source', 'fresh_execution')}
- **Request Hash**: {cache_info.get('request_hash', 'N/A')}

### Performance Metrics
- **Cache Hit Rate**: {cache_performance.get('cache_hit_rate', 0):.1f}%
- **Exact Cache Hits**: {cache_performance.get('exact_hits', 0)}
- **Similar Matches**: {cache_performance.get('similar_matches', 0)}
- **Cache Misses**: {cache_performance.get('cache_misses', 0)}
- **Efficiency Rating**: {cache_performance.get('efficiency_rating', 'N/A')}

### Time and Cost Savings
- **Total Time Saved**: {cache_performance.get('total_time_saved_minutes', 0):.2f} minutes
- **Total Cost Saved**: ${cache_performance.get('total_cost_saved_usd', 0):.4f}
- **Execution Time**: {execution_info.get('execution_time_seconds', 0):.2f} seconds

## Analysis Results

{format_cached_analysis_results(result)}

## Cache Information

### Request Details
"""
    
    if cache_info.get('cache_hit'):
        if cache_info.get('similarity_match'):
            report_content += f"""
- **Cache Type**: Similar Analysis Match
- **Similarity Score**: {cache_info.get('similarity_score', 0):.1%}
- **Original Request**: {cache_info.get('original_request_hash', 'N/A')}
- **Time Saved**: Instant retrieval from similar analysis
"""
        else:
            report_content += f"""
- **Cache Type**: Exact Match
- **Cached At**: {cache_info.get('cached_at', 'N/A')}
- **Time Saved**: {cache_info.get('execution_time_saved', 'N/A')}
- **Cost Saved**: {cache_info.get('cost_saved', 'N/A')}
"""
    else:
        report_content += f"""
- **Cache Type**: Fresh Execution
- **Cached for Future**: {cache_info.get('cached_for_future', False)}
- **Cache File**: {cache_info.get('cache_file', 'N/A')}
- **Available for Reuse**: Yes (7 days TTL)
"""
    
    report_content += f"""

### System Statistics
- **Total Requests Processed**: {execution_stats.get('total_requests', 0)}
- **Cache Hits**: {execution_stats.get('cache_hits', 0)}
- **Similar Matches**: {execution_stats.get('similar_matches', 0)}
- **Cache Misses**: {execution_stats.get('cache_misses', 0)}

## Key Insights

{format_cached_key_insights(result)}

## Recommendations

{format_cached_recommendations(result)}

## Technical Details

### Caching Features
- ✅ Intelligent request hashing for exact matches
- ✅ Semantic similarity search for related analyses
- ✅ Persistent storage across sessions
- ✅ Automatic cache cleanup and management
- ✅ Intermediate result caching for partial reuse
- ✅ Mem0 integration for long-term learning

### Cache Benefits
- **Instant Results**: Cached analyses return in <1 second
- **Cost Efficiency**: No token usage for cached results
- **Consistency**: Identical requests return identical results
- **Scalability**: Cache performance improves over time

---

*Report generated by Marketing Research Swarm - Cached Analysis System*  
*Cache Hit: {cache_info.get('cache_hit', False)} | Efficiency: {cache_performance.get('efficiency_rating', 'N/A')}*
"""
    
    # Save report
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"📄 Cached report saved to: {report_path}")
    return report_path

def format_cached_analysis_results(result: Dict[str, Any]) -> str:
    """Format analysis results for cached report"""
    if not result:
        return "No analysis results available."
    
    formatted = []
    
    # Check for different result structures
    if 'profitability_insights' in result:
        insights = result['profitability_insights']
        formatted.append("### Profitability Analysis")
        for key, value in insights.items():
            if isinstance(value, (int, float)):
                formatted.append(f"- **{key.replace('_', ' ').title()}**: {value}")
            else:
                formatted.append(f"- **{key.replace('_', ' ').title()}**: {value}")
    
    if 'budget_optimization' in result:
        budget = result['budget_optimization']
        formatted.append("\n### Budget Optimization")
        formatted.append(f"- **Total Budget**: ${budget.get('total_budget', 0):,.0f}")
        formatted.append(f"- **Optimization Score**: {budget.get('optimization_score', 0):.1f}/100")
    
    if 'roi_projections' in result:
        roi = result['roi_projections']
        formatted.append("\n### ROI Projections")
        for channel, projection in roi.items():
            formatted.append(f"- **{channel}**: {projection.get('roi_percentage', 0):.1f}% ROI")
    
    return "\n".join(formatted) if formatted else "Analysis results processed successfully."

def format_cached_key_insights(result: Dict[str, Any]) -> str:
    """Format key insights for cached report"""
    insights = []
    
    cache_info = result.get('cache_info', {})
    if cache_info.get('cache_hit'):
        insights.append("🚀 **Instant Results**: Analysis retrieved from cache in <1 second")
        insights.append("💰 **Zero Cost**: No tokens consumed for cached result")
    
    if 'profitability_insights' in result:
        prof_insights = result['profitability_insights']
        top_performers = [v for k, v in prof_insights.items() if 'top_performer' in k]
        if top_performers:
            insights.append(f"🏆 **Top Performers**: {', '.join(top_performers[:3])}")
    
    cache_performance = result.get('cache_performance', {})
    if cache_performance.get('cache_hit_rate', 0) > 50:
        insights.append(f"📈 **Cache Efficiency**: {cache_performance['cache_hit_rate']:.1f}% hit rate")
    
    return "\n".join(insights) if insights else "Key insights extracted from cached analysis."

def format_cached_recommendations(result: Dict[str, Any]) -> str:
    """Format recommendations for cached report"""
    recommendations = result.get('recommendations', [])
    
    # Add cache-specific recommendations
    cache_recommendations = []
    
    cache_info = result.get('cache_info', {})
    if not cache_info.get('cache_hit'):
        cache_recommendations.append("💾 This analysis is now cached for future instant retrieval")
    
    cache_performance = result.get('cache_performance', {})
    if cache_performance.get('cache_hit_rate', 0) < 30:
        cache_recommendations.append("🔄 Run similar analyses to improve cache hit rate")
    
    # Combine analysis and cache recommendations
    all_recommendations = cache_recommendations + (recommendations or [])
    
    if not all_recommendations:
        return "Recommendations based on cached analysis results."
    
    formatted = []
    for i, rec in enumerate(all_recommendations, 1):
        formatted.append(f"{i}. {rec}")
    
    return "\n".join(formatted)

def show_cache_status():
    """Show current cache status and statistics"""
    cache_manager = get_analysis_cache()
    stats = cache_manager.get_cache_statistics()
    
    print("📊 Cache Status")
    print("=" * 40)
    print(f"Total Cached Analyses: {stats.get('total_entries', 0)}")
    print(f"Total Cache Size: {stats.get('total_size_mb', 0):.2f} MB")
    print(f"Analysis Types: {list(stats.get('analysis_types', {}).keys())}")
    
    if stats.get('most_accessed'):
        most_accessed = stats['most_accessed']
        print(f"Most Accessed: {most_accessed['analysis_type']} ({most_accessed['access_count']} times)")
    
    if stats.get('oldest_entry'):
        print(f"Oldest Entry: {stats['oldest_entry']}")
    
    if stats.get('newest_entry'):
        print(f"Newest Entry: {stats['newest_entry']}")

def cleanup_cache():
    """Clean up expired cache entries"""
    cache_manager = get_analysis_cache()
    cleanup_stats = cache_manager.cleanup_expired_cache()
    
    print("🧹 Cache Cleanup Results")
    print("=" * 40)
    print(f"Expired entries removed: {cleanup_stats['expired_entries']}")
    print(f"Space freed: {cleanup_stats['freed_space_mb']:.2f} MB")
    print(f"Entries before: {cleanup_stats['total_entries_before']}")
    print(f"Entries after: {cleanup_stats['total_entries_after']}")

def main():
    """Main entry point for cached analysis"""
    parser = argparse.ArgumentParser(description="Run Marketing Research Analysis with Persistent Caching")
    parser.add_argument(
        "--type",
        type=str,
        default="roi_analysis",
        choices=["roi_analysis", "sales_forecast", "brand_performance"],
        help="Type of analysis to run (default: roi_analysis)"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="progressive_pruning",
        choices=["progressive_pruning", "abstracted_summaries", "minimal_context", "stateless"],
        help="Context optimization strategy (default: progressive_pruning)"
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force fresh execution ignoring cache"
    )
    parser.add_argument(
        "--use-mem0",
        action="store_true",
        help="Use Mem0 for long-term memory (requires mem0ai package)"
    )
    parser.add_argument(
        "--cache-status",
        action="store_true",
        help="Show cache status and statistics"
    )
    parser.add_argument(
        "--cleanup-cache",
        action="store_true",
        help="Clean up expired cache entries"
    )
    
    args = parser.parse_args()
    
    if args.cache_status:
        show_cache_status()
        return
    
    if args.cleanup_cache:
        cleanup_cache()
        return
    
    # Run analysis
    result = run_cached_analysis(
        analysis_type=args.type,
        context_strategy=args.strategy,
        force_refresh=args.force_refresh,
        use_mem0=args.use_mem0
    )
    
    # Display summary
    cache_info = result.get('cache_info', {})
    cache_performance = result.get('cache_performance', {})
    
    print(f"\n✅ Analysis Complete!")
    print(f"🎯 Cache Hit: {cache_info.get('cache_hit', False)}")
    
    if cache_info.get('cache_hit'):
        print(f"⚡ Instant retrieval from cache")
        print(f"💰 Zero token cost")
    else:
        execution_info = result.get('execution_info', {})
        print(f"⏱️  Execution Time: {execution_info.get('execution_time_seconds', 0):.2f}s")
        print(f"💾 Cached for future use")
    
    print(f"📈 Cache Hit Rate: {cache_performance.get('cache_hit_rate', 0):.1f}%")
    
    if 'report_path' in result:
        print(f"📄 Report: {result['report_path']}")

if __name__ == "__main__":
    main()