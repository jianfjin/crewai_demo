#!/usr/bin/env python
"""
Optimized Main Entry Point for Marketing Research Swarm
Uses flow-based architecture with token optimization
"""

import os
import yaml
import argparse
from datetime import datetime
from typing import Dict, Any

from marketing_research_swarm.flows.optimized_flow_runner import OptimizedFlowRunner
from marketing_research_swarm.context.context_manager import ContextStrategy

def load_settings():
    """Load configuration settings from settings.yaml"""
    settings_path = 'src/marketing_research_swarm/config/settings.yaml'
    try:
        with open(settings_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Warning: Settings file not found at {settings_path}, using defaults")
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
            'reports': {'output_directory': 'reports'}
        }

def run_optimized_analysis(analysis_type: str = "roi_analysis", 
                          context_strategy: str = "progressive_pruning",
                          token_budget: int = 4000,
                          use_mem0: bool = True,
                          **kwargs) -> Dict[str, Any]:
    """
    Run optimized analysis with token management
    
    Args:
        analysis_type: Type of analysis (roi_analysis, sales_forecast, brand_performance)
        context_strategy: Context optimization strategy
        token_budget: Maximum tokens per analysis step
        use_mem0: Whether to use Mem0 for long-term memory
        **kwargs: Additional parameters
        
    Returns:
        Analysis results with optimization metrics
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
    
    # Initialize optimized runner
    runner = OptimizedFlowRunner(
        token_budget=token_budget,
        use_mem0=use_mem0
    )
    
    print(f"🚀 Starting Optimized {analysis_type.title()} Analysis")
    print(f"📊 Strategy: {strategy.value}")
    print(f"🎯 Token Budget: {token_budget:,}")
    print(f"📁 Data Source: {data_path}")
    print("-" * 60)
    
    try:
        if analysis_type == "roi_analysis":
            result = runner.run_roi_analysis(
                data_file_path=data_path,
                context_strategy=strategy,
                **kwargs
            )
        elif analysis_type == "sales_forecast":
            result = runner.run_sales_forecast(
                data_file_path=data_path,
                **kwargs
            )
        elif analysis_type == "brand_performance":
            result = runner.run_brand_performance(
                data_file_path=data_path,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
        
        # Save report if analysis completed successfully
        if 'analysis_results' in result:
            report_path = save_optimized_report(result, analysis_type, settings)
            result['report_path'] = report_path
        
        return result
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        raise

def save_optimized_report(result: Dict[str, Any], analysis_type: str, settings: Dict[str, Any]) -> str:
    """Save optimized analysis report"""
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create reports directory
    reports_dir = settings.get('reports', {}).get('output_directory', 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    
    # Generate report filename
    report_filename = f"optimized_{analysis_type}_report_{timestamp}.md"
    report_path = os.path.join(reports_dir, report_filename)
    
    # Extract key information
    analysis_results = result.get('analysis_results', {})
    optimization_metrics = result.get('optimization_metrics', {})
    performance_summary = result.get('performance_summary', {})
    execution_metadata = result.get('execution_metadata', {})
    
    # Format comprehensive report
    report_content = f"""# Optimized {analysis_type.title()} Analysis Report

Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Executive Summary

This {analysis_type} analysis was conducted using an optimized AI flow system with advanced token management and caching. The analysis achieved significant efficiency improvements while maintaining high-quality insights.

## Optimization Performance

### Token Efficiency
- **Token Savings**: {optimization_metrics.get('token_optimization', {}).get('token_savings_percent', 0):.1f}% reduction
- **Traditional Approach**: {optimization_metrics.get('token_optimization', {}).get('traditional_tokens', 0):,} tokens
- **Optimized Approach**: {optimization_metrics.get('token_optimization', {}).get('optimized_tokens', 0):,} tokens

### Cost Efficiency  
- **Cost Savings**: ${optimization_metrics.get('cost_optimization', {}).get('cost_savings_usd', 0):.4f} per analysis
- **Traditional Cost**: ${optimization_metrics.get('cost_optimization', {}).get('traditional_cost_usd', 0):.4f}
- **Optimized Cost**: ${optimization_metrics.get('cost_optimization', {}).get('optimized_cost_usd', 0):.4f}

### Performance Metrics
- **Execution Time**: {execution_metadata.get('execution_time_seconds', 0):.2f} seconds
- **Optimization Score**: {optimization_metrics.get('performance_metrics', {}).get('optimization_score', 0):.1f}/100
- **Efficiency Rating**: {optimization_metrics.get('performance_metrics', {}).get('efficiency_rating', 'N/A')}

## Context Management
- **Strategy Used**: {optimization_metrics.get('context_management', {}).get('strategy_used', 'N/A')}
- **Context Elements**: {optimization_metrics.get('context_management', {}).get('total_elements', 0)}
- **Budget Utilization**: {optimization_metrics.get('context_management', {}).get('budget_utilization', 'N/A')}

## Caching Performance
- **Cache Items**: {optimization_metrics.get('caching_performance', {}).get('total_cache_items', 0)}
- **Cache Size**: {optimization_metrics.get('caching_performance', {}).get('cache_size_mb', 0):.2f} MB
- **Memory Items**: {optimization_metrics.get('caching_performance', {}).get('memory_items', 0)}
- **Disk Items**: {optimization_metrics.get('caching_performance', {}).get('disk_items', 0)}

## Analysis Results

{format_analysis_results(analysis_results)}

## Key Insights

{format_key_insights(analysis_results)}

## Recommendations

{format_recommendations(analysis_results)}

## Technical Details

### Optimization Features Used
- ✅ Flow-based execution with state management
- ✅ Smart caching with reference-based data sharing
- ✅ Context optimization with {optimization_metrics.get('context_management', {}).get('strategy_used', 'N/A')} strategy
- ✅ Long-term memory management
- ✅ Structured data models for efficient processing

### System Configuration
- **Analysis Type**: {analysis_type}
- **Context Strategy**: {execution_metadata.get('context_strategy', 'N/A')}
- **Token Budget**: {execution_metadata.get('token_budget', 0):,}
- **Flow Type**: {execution_metadata.get('flow_type', 'N/A')}

---

*Report generated by Marketing Research Swarm - Optimized AI Flow System*  
*Optimization Score: {optimization_metrics.get('performance_metrics', {}).get('optimization_score', 0):.1f}/100*  
*Token Savings: {optimization_metrics.get('token_optimization', {}).get('token_savings_percent', 0):.1f}%*
"""
    
    # Save report
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    print(f"📄 Optimized report saved to: {report_path}")
    return report_path

def format_analysis_results(results: Dict[str, Any]) -> str:
    """Format analysis results for report"""
    if not results:
        return "No analysis results available."
    
    formatted = []
    
    # Format profitability insights
    if 'profitability_insights' in results:
        insights = results['profitability_insights']
        formatted.append("### Profitability Analysis")
        for key, value in insights.items():
            if isinstance(value, (int, float)):
                formatted.append(f"- **{key.replace('_', ' ').title()}**: {value}")
            else:
                formatted.append(f"- **{key.replace('_', ' ').title()}**: {value}")
    
    # Format budget optimization
    if 'budget_optimization' in results:
        budget = results['budget_optimization']
        formatted.append("\n### Budget Optimization")
        formatted.append(f"- **Total Budget**: ${budget.get('total_budget', 0):,.0f}")
        formatted.append(f"- **Optimization Score**: {budget.get('optimization_score', 0):.1f}/100")
        
        if 'channel_allocations' in budget:
            formatted.append("- **Channel Allocations**:")
            for channel, amount in budget['channel_allocations'].items():
                formatted.append(f"  - {channel}: ${amount:,.0f}")
    
    # Format ROI projections
    if 'roi_projections' in results:
        roi = results['roi_projections']
        formatted.append("\n### ROI Projections")
        for channel, projection in roi.items():
            formatted.append(f"- **{channel}**: {projection.get('roi_percentage', 0):.1f}% ROI ({projection.get('category', 'N/A')})")
    
    return "\n".join(formatted) if formatted else "Analysis results are being processed."

def format_key_insights(results: Dict[str, Any]) -> str:
    """Format key insights for report"""
    insights = []
    
    if 'profitability_insights' in results:
        prof_insights = results['profitability_insights']
        
        # Extract top performers
        top_performers = [v for k, v in prof_insights.items() if 'top_performer' in k]
        if top_performers:
            insights.append(f"🏆 **Top Performers**: {', '.join(top_performers[:3])}")
        
        # Extract margins
        margins = [f"{k}: {v}%" for k, v in prof_insights.items() if 'margin' in k and isinstance(v, (int, float))]
        if margins:
            insights.append(f"📊 **Key Margins**: {', '.join(margins[:3])}")
    
    if 'optimization_summary' in results:
        opt_summary = results['optimization_summary']
        insights.append(f"💰 **Projected ROI**: {opt_summary.get('average_roi', 0):.1f}%")
        insights.append(f"🎯 **Optimization Score**: {opt_summary.get('optimization_score', 0):.1f}/100")
    
    return "\n".join(insights) if insights else "Key insights are being extracted from the analysis."

def format_recommendations(results: Dict[str, Any]) -> str:
    """Format recommendations for report"""
    recommendations = results.get('recommendations', [])
    
    if not recommendations:
        return "Recommendations are being generated based on the analysis results."
    
    formatted = []
    for i, rec in enumerate(recommendations, 1):
        formatted.append(f"{i}. {rec}")
    
    return "\n".join(formatted)

def main():
    """Main entry point for optimized analysis"""
    parser = argparse.ArgumentParser(description="Run Optimized Marketing Research Analysis")
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
        "--token-budget",
        type=int,
        default=4000,
        help="Token budget per analysis step (default: 4000)"
    )
    parser.add_argument(
        "--use-mem0",
        action="store_true",
        help="Use Mem0 for long-term memory (requires mem0ai package)"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark comparing all optimization strategies"
    )
    
    args = parser.parse_args()
    
    if args.benchmark:
        # Run benchmark test
        from .flows.optimized_flow_runner import OptimizedFlowRunner
        from .context.context_manager import ContextStrategy
        
        print("🏁 Running Optimization Benchmark")
        print("=" * 60)
        
        runner = OptimizedFlowRunner(token_budget=args.token_budget, use_mem0=args.use_mem0)
        settings = load_settings()
        data_path = settings['data_sources']['default_data_path']
        
        strategies = [
            ContextStrategy.PROGRESSIVE_PRUNING,
            ContextStrategy.ABSTRACTED_SUMMARIES,
            ContextStrategy.MINIMAL_CONTEXT
        ]
        
        for strategy in strategies:
            print(f"\n🔄 Testing {strategy.value}...")
            try:
                result = runner.run_roi_analysis(
                    data_file_path=data_path,
                    context_strategy=strategy
                )
                metrics = result['optimization_metrics']
                print(f"   Token Savings: {metrics['token_optimization']['token_savings_percent']:.1f}%")
                print(f"   Optimization Score: {metrics['performance_metrics']['optimization_score']:.1f}/100")
                print(f"   Execution Time: {metrics['performance_metrics']['execution_time_seconds']:.2f}s")
            except Exception as e:
                print(f"   ❌ Error: {e}")
    else:
        # Run single analysis
        result = run_optimized_analysis(
            analysis_type=args.type,
            context_strategy=args.strategy,
            token_budget=args.token_budget,
            use_mem0=args.use_mem0
        )
        
        # Display summary
        if 'optimization_metrics' in result:
            metrics = result['optimization_metrics']
            print(f"\n✅ Analysis Complete!")
            print(f"🎯 Token Savings: {metrics['token_optimization']['token_savings_percent']:.1f}%")
            print(f"💰 Cost Savings: ${metrics['cost_optimization']['cost_savings_usd']:.4f}")
            print(f"⚡ Optimization Score: {metrics['performance_metrics']['optimization_score']:.1f}/100")
            
            if 'report_path' in result:
                print(f"📄 Report: {result['report_path']}")

if __name__ == "__main__":
    main()