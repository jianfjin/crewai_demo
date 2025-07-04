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
from marketing_research_swarm.flows.optimized_flow_runner import OptimizedFlowRunner
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
    
    # Initialize runners
    cached_runner = CachedFlowRunner(use_mem0=use_mem0)
    optimized_runner = OptimizedFlowRunner(
        token_budget=settings.get('optimization', {}).get('token_budget', 4000),
        use_mem0=use_mem0
    )
    
    print(f"Starting Cached {analysis_type.title()} Analysis")
    print(f"Strategy: {strategy.value}")
    print(f"Data Source: {data_path}")
    print(f"Force Refresh: {force_refresh}")
    print("-" * 60)
    
    try:
        if analysis_type == "roi_analysis":
            result = cached_runner.run_roi_analysis(
                data_file_path=data_path,
                context_strategy=strategy,
                force_refresh=force_refresh,
                **kwargs
            )
        elif analysis_type == "sales_forecast":
            # Use optimized flow runner for sales forecast
            print("Running Sales Forecast Analysis with caching...")
            
            # Check cache first
            cache_manager = get_analysis_cache()
            cache_key = cache_manager.generate_request_hash(
                analysis_type="sales_forecast",
                data_path=data_path,
                parameters={
                    'context_strategy': context_strategy,
                    'use_mem0': use_mem0,
                    **kwargs
                }
            )
            
            if not force_refresh:
                cached_result = cache_manager.get_cached_result(cache_key)
                if cached_result:
                    print("Found cached sales forecast results!")
                    result = {
                        'result': cached_result['result'],
                        'cache_info': {
                            'cache_hit': True,
                            'cached_at': cached_result['cached_at'],
                            'cache_key': cache_key
                        },
                        'execution_time': 0.1  # Minimal time for cache retrieval
                    }
                    return result
            
            # Run fresh analysis
            result = optimized_runner.run_sales_forecast(
                data_file_path=data_path,
                context_strategy=strategy,
                **kwargs
            )
            
            # Cache the results
            if 'result' in result:
                cache_manager.cache_analysis_result(
                    request_hash=cache_key,
                    analysis_type="sales_forecast",
                    result=result['result'],
                    parameters={
                        'context_strategy': context_strategy,
                        'use_mem0': use_mem0,
                        **kwargs
                    }
                )
            
            result['cache_info'] = {
                'cache_hit': False,
                'cache_key': cache_key,
                'cached_at': datetime.now().isoformat()
            }
            
        elif analysis_type == "brand_performance":
            # Use optimized flow runner for brand performance
            print("Running Brand Performance Analysis with caching...")
            
            # Check cache first
            cache_manager = get_analysis_cache()
            cache_key = cache_manager.generate_request_hash(
                analysis_type="brand_performance",
                data_path=data_path,
                parameters={
                    'context_strategy': context_strategy,
                    'use_mem0': use_mem0,
                    **kwargs
                }
            )
            
            if not force_refresh:
                cached_result = cache_manager.get_cached_result(cache_key)
                if cached_result:
                    print("Found cached brand performance results!")
                    result = {
                        'result': cached_result['result'],
                        'cache_info': {
                            'cache_hit': True,
                            'cached_at': cached_result['cached_at'],
                            'cache_key': cache_key
                        },
                        'execution_time': 0.1  # Minimal time for cache retrieval
                    }
                    return result
            
            # Run fresh analysis
            result = optimized_runner.run_brand_performance(
                data_file_path=data_path,
                context_strategy=strategy,
                **kwargs
            )
            
            # Cache the results
            if 'result' in result:
                cache_manager.cache_analysis_result(
                    request_hash=cache_key,
                    analysis_type="brand_performance",
                    result=result['result'],
                    parameters={
                        'context_strategy': context_strategy,
                        'use_mem0': use_mem0,
                        **kwargs
                    }
                )
            
            result['cache_info'] = {
                'cache_hit': False,
                'cache_key': cache_key,
                'cached_at': datetime.now().isoformat()
            }
            
        else:
            raise ValueError(f"Unsupported analysis type: {analysis_type}")
        
        # Save report
        report_path = save_cached_report(result, analysis_type, settings)
        result['report_path'] = report_path
        
        return result
        
    except Exception as e:
        print(f"Error in cached analysis: {str(e)}")
        return {
            'error': str(e),
            'analysis_type': analysis_type,
            'timestamp': datetime.now().isoformat()
        }

def save_cached_report(result: Dict[str, Any], analysis_type: str, settings: Dict[str, Any]) -> str:
    """
    Save analysis results to a report file
    
    Args:
        result: Analysis results
        analysis_type: Type of analysis performed
        settings: Configuration settings
        
    Returns:
        Path to the saved report file
    """
    
    # Create reports directory if it doesn't exist
    reports_dir = settings.get('reports', {}).get('output_directory', 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    # Generate report filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{analysis_type}_cached_report_{timestamp}.md"
    report_path = os.path.join(reports_dir, filename)
    
    # Format results
    formatted_results = format_cached_analysis_results(result)
    
    # Write report
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# {analysis_type.title()} Analysis Report (Cached)\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Analysis Type:** {analysis_type}\n")
        
        # Cache information
        cache_info = result.get('cache_info', {})
        if cache_info:
            f.write(f"**Cache Hit:** {cache_info.get('cache_hit', 'Unknown')}\n")
            if cache_info.get('cache_hit'):
                f.write(f"**Cached At:** {cache_info.get('cached_at', 'Unknown')}\n")
            f.write(f"**Cache Key:** {cache_info.get('cache_key', 'Unknown')}\n")
        
        f.write(f"**Execution Time:** {result.get('execution_time', 'Unknown')} seconds\n\n")
        
        # Optimization metrics if available
        if 'optimization_metrics' in result:
            f.write("## Optimization Metrics\n\n")
            metrics = result['optimization_metrics']
            for key, value in metrics.items():
                f.write(f"- **{key.replace('_', ' ').title()}:** {value}\n")
            f.write("\n")
        
        f.write("## Analysis Results\n\n")
        f.write(formatted_results)
        
        # Error information if present
        if 'error' in result:
            f.write(f"\n## Error Information\n\n")
            f.write(f"**Error:** {result['error']}\n")
    
    print(f"Report saved to: {report_path}")
    return report_path

def format_cached_analysis_results(result: Dict[str, Any]) -> str:
    """
    Format analysis results for display in reports
    
    Args:
        result: Analysis results dictionary
        
    Returns:
        Formatted string representation of results
    """
    
    if 'error' in result:
        return f"**Error occurred during analysis:**\n\n{result['error']}\n"
    
    if 'result' not in result:
        return "**No analysis results available**\n"
    
    analysis_result = result['result']
    
    # Handle different result formats
    if isinstance(analysis_result, dict):
        formatted = ""
        
        # Handle comprehensive analysis results
        if 'comprehensive_analysis' in analysis_result:
            comp_analysis = analysis_result['comprehensive_analysis']
            
            # ROI Analysis specific formatting
            if 'profitability_insights' in comp_analysis:
                formatted += "### Profitability Analysis\n\n"
                insights = comp_analysis['profitability_insights']
                if isinstance(insights, dict) and 'analysis' in insights:
                    formatted += f"{insights['analysis']}\n\n"
            
            # Sales Forecast specific formatting
            if 'market_analysis' in comp_analysis:
                formatted += "### Market Analysis\n\n"
                market = comp_analysis['market_analysis']
                if isinstance(market, dict) and 'analysis' in market:
                    formatted += f"{market['analysis']}\n\n"
            
            if 'forecasting_results' in comp_analysis:
                formatted += "### Sales Forecasting\n\n"
                forecast = comp_analysis['forecasting_results']
                if isinstance(forecast, dict) and 'analysis' in forecast:
                    formatted += f"{forecast['analysis']}\n\n"
            
            # Brand Performance specific formatting
            if 'competitive_intelligence' in comp_analysis:
                formatted += "### Competitive Intelligence\n\n"
                competitive = comp_analysis['competitive_intelligence']
                if isinstance(competitive, dict) and 'analysis' in competitive:
                    formatted += f"{competitive['analysis']}\n\n"
            
            if 'brand_strategy' in comp_analysis:
                formatted += "### Brand Strategy\n\n"
                strategy = comp_analysis['brand_strategy']
                if isinstance(strategy, dict) and 'analysis' in strategy:
                    formatted += f"{strategy['analysis']}\n\n"
            
            # Analysis metadata
            if 'analysis_metadata' in comp_analysis:
                formatted += "### Analysis Metadata\n\n"
                metadata = comp_analysis['analysis_metadata']
                for key, value in metadata.items():
                    if key != 'token_usage':  # Skip detailed token usage
                        formatted += f"- **{key.replace('_', ' ').title()}:** {value}\n"
                formatted += "\n"
        
        # Fallback for other result formats
        if not formatted:
            for key, value in analysis_result.items():
                if isinstance(value, (str, int, float)):
                    formatted += f"**{key.replace('_', ' ').title()}:** {value}\n\n"
                elif isinstance(value, dict):
                    formatted += f"### {key.replace('_', ' ').title()}\n\n"
                    for sub_key, sub_value in value.items():
                        if isinstance(sub_value, (str, int, float)):
                            formatted += f"- **{sub_key.replace('_', ' ').title()}:** {sub_value}\n"
                    formatted += "\n"
        
        return formatted if formatted else str(analysis_result)
    
    else:
        return str(analysis_result)

def display_cache_stats():
    """Display cache statistics and performance metrics"""
    cache_manager = get_analysis_cache()
    stats = cache_manager.get_cache_stats()
    
    print("\n" + "="*60)
    print("CACHE PERFORMANCE STATISTICS")
    print("="*60)
    
    print(f"Total Cached Analyses: {stats.get('total_cached_analyses', 0)}")
    print(f"Cache Hit Rate: {stats.get('cache_hit_rate', 0):.1%}")
    print(f"Average Response Time (Cache Hit): {stats.get('avg_cache_response_time', 0):.2f}s")
    print(f"Average Response Time (Fresh): {stats.get('avg_fresh_response_time', 0):.2f}s")
    print(f"Total Storage Used: {stats.get('storage_used_mb', 0):.1f} MB")
    print(f"Cache Cleanup Last Run: {stats.get('last_cleanup', 'Never')}")
    
    # Analysis type breakdown
    if 'analysis_type_breakdown' in stats:
        print("\nAnalysis Type Breakdown:")
        for analysis_type, count in stats['analysis_type_breakdown'].items():
            print(f"  {analysis_type}: {count} cached results")
    
    print("="*60)

def cleanup_cache(max_age_days: int = 7):
    """Clean up old cache entries"""
    cache_manager = get_analysis_cache()
    cleanup_stats = cache_manager.cleanup_old_entries(max_age_days)
    
    print(f"\nCache Cleanup Complete:")
    print(f"  Removed {cleanup_stats.get('removed_count', 0)} old entries")
    print(f"  Freed {cleanup_stats.get('freed_space_mb', 0):.1f} MB")
    print(f"  Remaining entries: {cleanup_stats.get('remaining_count', 0)}")

def main():
    """Main entry point for cached analysis"""
    parser = argparse.ArgumentParser(description="Marketing Research Analysis with Persistent Caching")
    
    parser.add_argument(
        "--analysis-type",
        choices=["roi_analysis", "sales_forecast", "brand_performance"],
        default="roi_analysis",
        help="Type of analysis to run (default: roi_analysis)"
    )
    
    parser.add_argument(
        "--strategy",
        choices=["progressive_pruning", "abstracted_summaries", "minimal_context", "stateless"],
        default="progressive_pruning",
        help="Context optimization strategy (default: progressive_pruning)"
    )
    
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force fresh execution, ignoring cache"
    )
    
    parser.add_argument(
        "--no-mem0",
        action="store_true",
        help="Disable Mem0 long-term memory integration"
    )
    
    parser.add_argument(
        "--cache-stats",
        action="store_true",
        help="Display cache statistics and exit"
    )
    
    parser.add_argument(
        "--cleanup-cache",
        type=int,
        metavar="DAYS",
        help="Clean up cache entries older than specified days"
    )
    
    args = parser.parse_args()
    
    # Handle cache management commands
    if args.cache_stats:
        display_cache_stats()
        return
    
    if args.cleanup_cache is not None:
        cleanup_cache(args.cleanup_cache)
        return
    
    # Run analysis
    start_time = datetime.now()
    
    result = run_cached_analysis(
        analysis_type=args.analysis_type,
        context_strategy=args.strategy,
        force_refresh=args.force_refresh,
        use_mem0=not args.no_mem0
    )
    
    end_time = datetime.now()
    total_time = (end_time - start_time).total_seconds()
    
    print(f"\nAnalysis completed in {total_time:.2f} seconds")
    
    if 'error' in result:
        print(f"Error: {result['error']}")
        return
    
    # Display cache performance
    cache_info = result.get('cache_info', {})
    if cache_info.get('cache_hit'):
        print("Result retrieved from cache!")
        print(f"Cache key: {cache_info.get('cache_key', 'Unknown')}")
    else:
        print("Fresh analysis executed and cached for future use")
    
    if 'report_path' in result:
        print(f"Report saved to: {result['report_path']}")
    
    # Display optimization metrics if available
    if 'optimization_metrics' in result:
        print("\nOptimization Metrics:")
        for key, value in result['optimization_metrics'].items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    main()