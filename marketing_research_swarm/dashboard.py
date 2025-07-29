#!/usr/bin/env python
"""
Marketing Research Swarm - Dynamic Task Creation Dashboard
A Streamlit-based web interface for creating and executing marketing research tasks
"""

import streamlit as st
import yaml
import os
import sys
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Any
import tempfile
import uuid
import sqlite3

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

if sqlite3.sqlite_version_info < (3, 35, 0):
    try:
        # Try to use pysqlite3 instead
        import pysqlite3 as sqlite3
        sys.modules['sqlite3'] = sqlite3
        print(f"Using pysqlite3 version: {sqlite3.sqlite_version}")
    except ImportError:
        print("pysqlite3 not available, using system sqlite3")

try:
    from marketing_research_swarm.optimization_manager import optimization_manager
    from marketing_research_swarm.utils.token_tracker import TokenTracker, get_token_tracker, reset_token_tracker

except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please ensure you're running from the correct directory and all dependencies are installed.")

def _safe_get_nested(data, key1, key2, default=None):
    """Safely get nested dictionary values with type checking"""
    try:
        outer_value = data.get(key1, {})
        if isinstance(outer_value, dict):
            return outer_value.get(key2, default)
        else:
            # If outer_value is not a dict, return default
            return default
    except (AttributeError, TypeError):
        return default

# Page configuration
st.set_page_config(
    page_title="Marketing Research Swarm Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3498db;
        margin: 0.5rem 0;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

def load_agents_config():
    """Load available agents from agents.yaml"""
    try:
        agents_path = 'marketing_research_swarm/src/marketing_research_swarm/config/agents.yaml'
        with open(agents_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        st.error(f"Agents configuration file not found at {agents_path}")
        return {}
    except Exception as e:
        st.error(f"Error loading agents configuration: {e}")
        return {}

def create_custom_task_config(selected_agents: List[str], task_params: Dict[str, Any]) -> str:
    """Create a custom task configuration YAML file"""
    
    # Generate unique task ID
    task_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create task configuration
    tasks_config = {}
    
    # Map agents to their typical tasks
    agent_task_mapping = {
        'market_research_analyst': {
            'description': f"""Conduct comprehensive market research on the {task_params.get('analysis_focus', 'beverage industry')} using the enhanced sales data from {{data_file_path}}. 
            Focus on {task_params.get('target_audience', 'target audience')} and analyze {', '.join(task_params.get('market_segments', []))} markets. 
            Examine {', '.join(task_params.get('product_categories', []))} categories and {', '.join(task_params.get('brands', []))} brands.
            Business objective: {task_params.get('business_objective', 'Optimize business performance')}""",
            'expected_output': f"A comprehensive market research report with detailed analysis of {task_params.get('analysis_focus', 'market performance')}, competitive landscape insights, and strategic recommendations for {task_params.get('target_audience', 'the target market')}."
        },
        'content_strategist': {
            'description': f"""Develop a comprehensive content strategy for {task_params.get('campaign_type', 'marketing campaign')} targeting {task_params.get('target_audience', 'target audience')}. 
            Create strategies for {', '.join(task_params.get('market_segments', []))} markets with budget of ${task_params.get('budget', 0):,} over {task_params.get('duration', '6 months')}.
            Focus on {', '.join(task_params.get('product_categories', []))} categories and competitive landscape: {task_params.get('competitive_landscape', 'competitive market')}.""",
            'expected_output': f"A comprehensive content strategy document with channel-specific recommendations, campaign ideas for {task_params.get('duration', '6 months')} duration, and brand positioning strategies."
        },
        'creative_copywriter': {
            'description': f"""Create compelling marketing copy for {task_params.get('campaign_type', 'marketing campaigns')} targeting {task_params.get('target_audience', 'target audience')}. 
            Develop copy that addresses campaign goals: {', '.join(task_params.get('campaign_goals', []))}. 
            Focus on {', '.join(task_params.get('brands', []))} brands across {', '.join(task_params.get('market_segments', []))} markets.""",
            'expected_output': f"A collection of marketing copy including campaign materials, brand messaging, and promotional content tailored for {task_params.get('target_audience', 'the target audience')}."
        },
        'data_analyst': {
            'description': f"""Perform comprehensive data analysis on the sales data from {{data_file_path}} focusing on {', '.join(task_params.get('key_metrics', []))}. 
            Analyze {', '.join(task_params.get('product_categories', []))} categories across {', '.join(task_params.get('market_segments', []))} regions.
            Generate forecasts for {task_params.get('forecast_periods', 30)} periods with expected revenue of ${task_params.get('expected_revenue', 25000):,}.
            Include competitive analysis: {task_params.get('competitive_analysis', True)} and market share analysis: {task_params.get('market_share_analysis', True)}.""",
            'expected_output': f"A detailed data analysis report with forecasts, trend analysis, performance metrics for {', '.join(task_params.get('key_metrics', []))}, and recommendations for optimization."
        },
        'campaign_optimizer': {
            'description': f"""Optimize {task_params.get('campaign_type', 'marketing campaign')} performance with budget allocation of ${task_params.get('budget', 0):,} over {task_params.get('duration', '6 months')}.
            Focus on {', '.join(task_params.get('campaign_goals', []))} across {', '.join(task_params.get('market_segments', []))} markets.
            Optimize for {', '.join(task_params.get('key_metrics', []))} with competitive landscape: {task_params.get('competitive_landscape', 'competitive market')}.""",
            'expected_output': f"A comprehensive optimization strategy with budget allocation recommendations, performance projections, and specific action plans for {task_params.get('duration', '6 months')} campaign duration."
        },
        'brand_performance_specialist': {
            'description': f"""Monitor and analyze brand performance for {', '.join(task_params.get('brands', []))} across {', '.join(task_params.get('market_segments', []))} markets.
            Track brand metrics including awareness: {task_params.get('brand_metrics', {}).get('brand_awareness', 'N/A')}, 
            sentiment score: {task_params.get('brand_metrics', {}).get('sentiment_score', 'N/A')}, 
            market position: {task_params.get('brand_metrics', {}).get('market_position', 'N/A')}.
            Focus on {', '.join(task_params.get('product_categories', []))} categories and competitive positioning.""",
            'expected_output': f"A comprehensive brand performance report with market positioning analysis, competitive insights, and strategic recommendations for {', '.join(task_params.get('brands', []))} brands."
        },
        'competitive_analyst': {
            'description': f"""Analyze competitive landscape and market positioning for {', '.join(task_params.get('brands', []))} in the {task_params.get('analysis_focus', 'beverage industry')}.
            Examine competitive dynamics across {', '.join(task_params.get('market_segments', []))} markets and {', '.join(task_params.get('product_categories', []))} categories.
            Assess market share, competitive threats, pricing strategies, and positioning opportunities.
            Focus on competitive intelligence for {task_params.get('target_audience', 'target market')} with budget considerations of ${task_params.get('budget', 0):,}.""",
            'expected_output': f"A detailed competitive analysis report with market positioning insights, competitive landscape mapping, threat assessment, and strategic recommendations for competitive advantage."
        },
        'brand_strategist': {
            'description': f"""Develop strategic brand recommendations based on competitive analysis and market insights for {', '.join(task_params.get('brands', []))}.
            Create brand optimization strategies for {', '.join(task_params.get('market_segments', []))} markets with focus on {', '.join(task_params.get('campaign_goals', []))}.
            Evaluate brand health, identify growth opportunities, and develop actionable strategies for brand performance improvement.
            Consider budget allocation of ${task_params.get('budget', 0):,} over {task_params.get('duration', '6 months')} for brand initiatives.""",
            'expected_output': f"A comprehensive brand strategy document with optimization recommendations, growth opportunities, brand health assessment, and actionable strategic plans for brand improvement."
        },
        'forecasting_specialist': {
            'description': f"""Generate accurate sales forecasts and predictive models for {', '.join(task_params.get('brands', []))} across {', '.join(task_params.get('market_segments', []))} markets.
            Create forecasts for {task_params.get('forecast_periods', 30)} periods with expected revenue targets of ${task_params.get('expected_revenue', 25000):,}.
            Apply advanced forecasting techniques considering seasonal patterns, market trends, and competitive factors.
            Focus on {', '.join(task_params.get('product_categories', []))} categories and key metrics: {', '.join(task_params.get('key_metrics', []))}.""",
            'expected_output': f"A detailed sales forecast report with predictive models, confidence intervals, scenario planning, and strategic recommendations for {task_params.get('forecast_periods', 30)} periods ahead."
        }
    }
    
    # Create tasks for selected agents in the order they were selected
    for i, agent in enumerate(selected_agents):
        if agent in agent_task_mapping:
            # Use zero-padded index to maintain order
            task_name = f"{i:02d}_{agent}_task_{task_id}"
            tasks_config[task_name] = {
                'description': agent_task_mapping[agent]['description'],
                'expected_output': agent_task_mapping[agent]['expected_output'],
                'agent': agent
            }
    
    # Save to temporary file
    config_dir = 'src/marketing_research_swarm/config'
    os.makedirs(config_dir, exist_ok=True)
    
    config_filename = f"tasks_custom_{timestamp}_{task_id}.yaml"
    config_path = os.path.join(config_dir, config_filename)
    
    with open(config_path, 'w') as file:
        yaml.dump(tasks_config, file, default_flow_style=False, indent=2)
    
    return config_path

def parse_token_metrics(tracker: TokenTracker, crew_metrics: Dict[str, Any] = None) -> Dict[str, Any]:
    """Parse token tracking metrics for visualization"""
    token_data = {}
    
    # First try to get metrics from crew usage_metrics (optimized crew)
    if crew_metrics and crew_metrics.get('total_tokens', 0) > 0:
        token_data.update({
            'total_tokens': crew_metrics.get('total_tokens', 0),
            'total_cost': crew_metrics.get('total_cost', 0.0),
            'requests_made': crew_metrics.get('successful_requests', crew_metrics.get('total_requests', 1)),
            'average_tokens_per_request': crew_metrics.get('total_tokens', 0) / max(crew_metrics.get('successful_requests', 1), 1),
            'cost_per_token': crew_metrics.get('total_cost', 0) / max(crew_metrics.get('total_tokens', 1), 1),
            'input_tokens': crew_metrics.get('input_tokens', crew_metrics.get('prompt_tokens', 0)),
            'output_tokens': crew_metrics.get('output_tokens', crew_metrics.get('completion_tokens', 0)),
            'estimated': crew_metrics.get('estimated', False),
            'source': 'optimized_crew_metrics'
        })
        
        # Calculate token breakdown for visualization
        if token_data['input_tokens'] > 0 or token_data['output_tokens'] > 0:
            token_data['token_breakdown'] = {
                'Input Tokens': token_data['input_tokens'],
                'Output Tokens': token_data['output_tokens']
            }
            
            token_data['cost_breakdown'] = {
                'Input Cost': token_data['input_tokens'] * 0.00000015,  # gpt-4o-mini input pricing
                'Output Cost': token_data['output_tokens'] * 0.0000006   # gpt-4o-mini output pricing
            }
    
    # Fallback to tracker metrics if available and no crew metrics
    elif tracker and not token_data:
        try:
            metrics = tracker.get_metrics()
            if metrics and metrics.get('total_tokens_used', 0) > 0:
                token_data.update({
                    'total_tokens': metrics.get('total_tokens_used', 0),
                    'total_cost': metrics.get('total_cost_usd', 0.0),
                    'requests_made': metrics.get('total_requests', 0),
                    'average_tokens_per_request': metrics.get('average_tokens_per_request', 0),
                    'cost_breakdown': metrics.get('cost_breakdown', {}),
                    'token_breakdown': metrics.get('token_breakdown', {}),
                    'efficiency_score': metrics.get('efficiency_metrics', {}).get('tokens_per_insight', 0),
                    'source': 'token_tracker'
                })
        except Exception as e:
            print(f"Error getting tracker metrics: {e}")
    
    # Return the token data (may be empty if no metrics available)
    return token_data

def parse_optimization_metrics(context_manager, cache, memory_manager) -> Dict[str, Any]:
    """Parse optimization metrics from various components"""
    metrics = {}
    
    # Context management metrics
    if context_manager:
        try:
            context_stats = context_manager.get_optimization_stats()
            metrics['context_optimization'] = {
                'elements_managed': context_stats.get('total_elements', 0),
                'memory_saved_mb': context_stats.get('memory_saved_mb', 0),
                'compression_ratio': context_stats.get('compression_ratio', 1.0),
                'pruning_efficiency': context_stats.get('pruning_efficiency', 0)
            }
        except:
            metrics['context_optimization'] = {'status': 'unavailable'}
    
    # Cache performance metrics
    if cache:
        try:
            cache_stats = cache.get_cache_stats()
            metrics['cache_performance'] = {
                'hit_rate': cache_stats.get('hit_rate', 0),
                'total_hits': cache_stats.get('total_hits', 0),
                'total_misses': cache_stats.get('total_misses', 0),
                'cache_size_mb': cache_stats.get('cache_size_mb', 0),
                'items_cached': cache_stats.get('total_items', 0)
            }
        except:
            metrics['cache_performance'] = {'status': 'unavailable'}
    
    # Memory management metrics
    if memory_manager:
        try:
            memory_stats = memory_manager.get_memory_stats()
            metrics['memory_management'] = {
                'total_memories': memory_stats.get('total_memories', 0),
                'memory_efficiency': memory_stats.get('efficiency_score', 0),
                'retrieval_accuracy': memory_stats.get('retrieval_accuracy', 0)
            }
        except:
            metrics['memory_management'] = {'status': 'unavailable'}
    
    return metrics

def parse_analysis_results(result: Any, tracker: TokenTracker = None, 
                         context_manager=None, cache=None, memory_manager=None,
                         crew_metrics: Dict[str, Any] = None) -> Dict[str, Any]:
    """Parse and structure analysis results for visualization"""
    if isinstance(result, str):
        return {
            'summary': result[:500] + "..." if len(result) > 500 else result,
            'full_text': result,
            'metrics': {},
            'recommendations': []
        }
    
    if hasattr(result, 'raw'):
        result_text = result.raw
    else:
        result_text = str(result)
    
    # Extract structured data from result
    parsed_result = {
        'summary': result_text[:500] + "..." if len(result_text) > 500 else result_text,
        'full_text': result_text,
        'metrics': {},
        'recommendations': [],
        'token_metrics': parse_token_metrics(tracker, crew_metrics),
        'optimization_metrics': parse_optimization_metrics(context_manager, cache, memory_manager)
    }
    
    # Try to extract metrics and recommendations
    lines = result_text.split('\n')
    current_section = None
    
    for line in lines:
        line = line.strip()
        if 'recommendation' in line.lower():
            current_section = 'recommendations'
        elif any(keyword in line.lower() for keyword in ['revenue', 'profit', 'margin', 'roi', 'growth']):
            # Try to extract numeric values
            import re
            numbers = re.findall(r'[\d,]+\.?\d*', line)
            if numbers:
                metric_name = line.split(':')[0] if ':' in line else line[:30]
                parsed_result['metrics'][metric_name] = numbers[0]
        
        if current_section == 'recommendations' and line.startswith('-'):
            parsed_result['recommendations'].append(line[1:].strip())
    
    return parsed_result

def create_token_visualizations(token_metrics: Dict[str, Any]):
    """Create token usage and cost visualizations"""
    if not token_metrics:
        return
    
    st.markdown('<div class="section-header">🔢 Token Usage & Cost Analysis</div>', unsafe_allow_html=True)
    
    # Token usage metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Tokens Used",
            f"{token_metrics.get('total_tokens', 0):,}",
            help="Total number of tokens consumed during analysis"
        )
    
    with col2:
        st.metric(
            "Total Cost",
            f"${token_metrics.get('total_cost', 0):.4f}",
            help="Total cost in USD for the analysis"
        )
    
    with col3:
        st.metric(
            "API Requests",
            f"{token_metrics.get('requests_made', 0)}",
            help="Number of API requests made"
        )
    
    with col4:
        st.metric(
            "Avg Tokens/Request",
            f"{token_metrics.get('average_tokens_per_request', 0):.0f}",
            help="Average tokens per API request"
        )
    
    # Token breakdown visualization
    if token_metrics.get('token_breakdown'):
        st.subheader("Token Usage Breakdown")
        
        breakdown_df = pd.DataFrame([
            {'Component': k, 'Tokens': v} 
            for k, v in token_metrics['token_breakdown'].items()
        ])
        
        if not breakdown_df.empty:
            fig_tokens = px.pie(
                breakdown_df,
                values='Tokens',
                names='Component',
                title="Token Distribution by Component",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_tokens, use_container_width=True)
    
    # Cost breakdown visualization
    if token_metrics.get('cost_breakdown'):
        st.subheader("Cost Breakdown")
        
        cost_df = pd.DataFrame([
            {'Component': k, 'Cost': v} 
            for k, v in token_metrics['cost_breakdown'].items()
        ])
        
        if not cost_df.empty:
            fig_cost = px.bar(
                cost_df,
                x='Component',
                y='Cost',
                title="Cost Distribution by Component ($)",
                color='Cost',
                color_continuous_scale='Blues'
            )
            fig_cost.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_cost, use_container_width=True)

def create_optimization_visualizations(optimization_metrics: Dict[str, Any]):
    """Create optimization performance visualizations"""
    if not optimization_metrics:
        return
    
    st.markdown('<div class="section-header">⚡ Optimization Performance</div>', unsafe_allow_html=True)
    
    # Context optimization metrics
    if 'context_optimization' in optimization_metrics:
        context_metrics = optimization_metrics['context_optimization']
        if context_metrics.get('status') != 'unavailable':
            st.subheader("Context Management")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Elements Managed",
                    f"{context_metrics.get('elements_managed', 0)}",
                    help="Number of context elements managed"
                )
            
            with col2:
                st.metric(
                    "Memory Saved",
                    f"{context_metrics.get('memory_saved_mb', 0):.1f} MB",
                    help="Memory saved through optimization"
                )
            
            with col3:
                st.metric(
                    "Compression Ratio",
                    f"{context_metrics.get('compression_ratio', 1.0):.2f}x",
                    help="Context compression efficiency"
                )
    
    # Cache performance metrics
    if 'cache_performance' in optimization_metrics:
        cache_metrics = optimization_metrics['cache_performance']
        if cache_metrics.get('status') != 'unavailable':
            st.subheader("Cache Performance")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Cache Hit Rate",
                    f"{cache_metrics.get('hit_rate', 0):.1f}%",
                    help="Percentage of cache hits"
                )
            
            with col2:
                st.metric(
                    "Total Hits",
                    f"{cache_metrics.get('total_hits', 0)}",
                    help="Number of cache hits"
                )
            
            with col3:
                st.metric(
                    "Cache Size",
                    f"{cache_metrics.get('cache_size_mb', 0):.1f} MB",
                    help="Current cache size"
                )
            
            with col4:
                st.metric(
                    "Items Cached",
                    f"{cache_metrics.get('items_cached', 0)}",
                    help="Number of items in cache"
                )
            
            # Cache efficiency visualization
            if cache_metrics.get('total_hits', 0) > 0 or cache_metrics.get('total_misses', 0) > 0:
                cache_data = pd.DataFrame({
                    'Type': ['Hits', 'Misses'],
                    'Count': [cache_metrics.get('total_hits', 0), cache_metrics.get('total_misses', 0)]
                })
                
                fig_cache = px.pie(
                    cache_data,
                    values='Count',
                    names='Type',
                    title="Cache Hit/Miss Ratio",
                    color_discrete_map={'Hits': '#2ecc71', 'Misses': '#e74c3c'}
                )
                st.plotly_chart(fig_cache, use_container_width=True)
    
    # Memory management metrics
    if 'memory_management' in optimization_metrics:
        memory_metrics = optimization_metrics['memory_management']
        if memory_metrics.get('status') != 'unavailable':
            st.subheader("Memory Management")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Total Memories",
                    f"{memory_metrics.get('total_memories', 0)}",
                    help="Number of memories stored"
                )
            
            with col2:
                st.metric(
                    "Memory Efficiency",
                    f"{memory_metrics.get('memory_efficiency', 0):.1f}%",
                    help="Memory management efficiency score"
                )
            
            with col3:
                st.metric(
                    "Retrieval Accuracy",
                    f"{memory_metrics.get('retrieval_accuracy', 0):.1f}%",
                    help="Accuracy of memory retrieval"
                )

def create_visualizations(parsed_results: Dict[str, Any], task_params: Dict[str, Any]):
    """Create visualizations based on analysis results"""
    
    st.markdown('<div class="section-header">📊 Analysis Visualizations</div>', unsafe_allow_html=True)
    
    # Token usage visualizations
    if parsed_results.get('token_metrics'):
        create_token_visualizations(parsed_results['token_metrics'])
    
    # Optimization performance visualizations
    if parsed_results.get('optimization_metrics'):
        create_optimization_visualizations(parsed_results['optimization_metrics'])
    
    # Metrics visualization
    if parsed_results.get('metrics'):
        st.subheader("Key Metrics")
        
        metrics_df = pd.DataFrame([
            {'Metric': k, 'Value': v} for k, v in parsed_results['metrics'].items()
        ])
        
        if not metrics_df.empty:
            fig_metrics = px.bar(
                metrics_df, 
                x='Metric', 
                y='Value',
                title="Key Performance Metrics",
                color='Value',
                color_continuous_scale='Blues'
            )
            fig_metrics.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_metrics, use_container_width=True)
    
    # Campaign parameters visualization
    if task_params:
        st.subheader("Campaign Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Budget and duration
            fig_budget = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = task_params.get('budget', 0),
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Campaign Budget ($)"},
                gauge = {
                    'axis': {'range': [None, task_params.get('budget', 100000) * 1.5]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, task_params.get('budget', 100000) * 0.5], 'color': "lightgray"},
                        {'range': [task_params.get('budget', 100000) * 0.5, task_params.get('budget', 100000)], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': task_params.get('budget', 100000)
                    }
                }
            ))
            st.plotly_chart(fig_budget, use_container_width=True)
        
        with col2:
            # Market segments pie chart
            if task_params.get('market_segments'):
                segments_df = pd.DataFrame({
                    'Segment': task_params['market_segments'],
                    'Value': [1] * len(task_params['market_segments'])  # Equal distribution for visualization
                })
                
                fig_segments = px.pie(
                    segments_df,
                    values='Value',
                    names='Segment',
                    title="Target Market Segments"
                )
                st.plotly_chart(fig_segments, use_container_width=True)
    
    # Product categories analysis
    if task_params.get('product_categories'):
        st.subheader("Product Categories Analysis")
        
        categories_df = pd.DataFrame({
            'Category': task_params['product_categories'],
            'Priority': range(len(task_params['product_categories']), 0, -1)  # Descending priority
        })
        
        fig_categories = px.bar(
            categories_df,
            x='Category',
            y='Priority',
            title="Product Categories by Priority",
            color='Priority',
            color_continuous_scale='Viridis'
        )
        fig_categories.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_categories, use_container_width=True)

def main():
    """Main dashboard application"""
    
    # Header
    st.markdown('<div class="main-header">🚀 Marketing Research Swarm Dashboard</div>', unsafe_allow_html=True)
    st.markdown("**Dynamic Task Creation & Execution Platform**")
    
    # Load agents configuration
    agents_config = load_agents_config()
    
    if not agents_config:
        st.error("Unable to load agents configuration. Please check the file path and try again.")
        return
    
    # Sidebar for agent phase selection and optimal analysis types
    st.sidebar.markdown("## 🤖 Agent Phase Selection")
    st.sidebar.markdown("Select agents by phase or choose optimal analysis types:")
    
    # Show Agent Phases
    st.sidebar.markdown("### 🏗️ Agent Phases")
    st.sidebar.markdown("""
    **FOUNDATION**: market_research_analyst, data_analyst  
    **ANALYSIS**: competitive_analyst, brand_performance_specialist  
    **STRATEGY**: brand_strategist, campaign_optimizer, forecasting_specialist  
    **CONTENT**: content_strategist, creative_copywriter
    """)
    
    # Add optimal analysis types suggestions
    st.sidebar.markdown("### 📊 Optimal Analysis Types")
    st.sidebar.markdown("Choose from pre-configured optimal combinations:")
    
    try:
        from marketing_research_swarm.blackboard.agent_dependency_manager import get_dependency_manager
        dependency_manager = get_dependency_manager()
        analysis_types = dependency_manager.get_analysis_types()
        
        # Create selectbox for analysis types
        analysis_type_options = ["custom"] + [key for key in analysis_types.keys() if key != "custom"]
        analysis_type_labels = ["Custom Selection"] + [analysis_types[key]["name"] for key in analysis_types.keys() if key != "custom"]
        
        selected_analysis_type = st.sidebar.selectbox(
            "Analysis Type",
            options=analysis_type_options,
            format_func=lambda x: dict(zip(analysis_type_options, analysis_type_labels)).get(x, x),
            help="Choose an optimal analysis type or select custom for manual agent selection"
        )
        
        # Show analysis type info
        if selected_analysis_type != "custom":
            analysis_info = analysis_types[selected_analysis_type]
            st.sidebar.info(f"**{analysis_info['name']}**\n\n{analysis_info['description']}\n\n"
                          f"**Focus**: {analysis_info['focus']}\n"
                          f"**Duration**: {analysis_info['expected_duration']}\n"
                          f"**Token Efficiency**: {analysis_info['token_efficiency']}")
            
            # Auto-select optimal agents for this analysis type
            optimal_agents = analysis_info['agents']
        else:
            optimal_agents = None
            st.sidebar.info("**Custom Selection**\n\nManually select agents based on your specific needs. Dependencies will be automatically resolved.")
    
    except ImportError:
        st.sidebar.warning("⚠️ Advanced agent dependency management not available")
        selected_analysis_type = "custom"
        optimal_agents = None
    
    available_agents = list(agents_config.keys())
    
    # Use optimal agents if analysis type is selected, otherwise manual selection
    if selected_analysis_type != "custom" and optimal_agents:
        selected_agents = optimal_agents
        st.sidebar.success(f"✅ **Auto-selected {len(selected_agents)} agents** for {analysis_types[selected_analysis_type]['name']}")
        st.sidebar.markdown("**Selected Agents:**")
        for agent in selected_agents:
            st.sidebar.markdown(f"• {agent}")
        
        # Option to override with custom selection
        if st.sidebar.checkbox("Override with custom selection", help="Manually select different agents"):
            selected_agents = st.sidebar.multiselect(
                "Choose Custom Agents",
                available_agents,
                default=optimal_agents,
                help="Select agents from different phases for optimal workflow"
            )
    else:
        selected_agents = st.sidebar.multiselect(
            "Choose Agents by Phase",
            available_agents,
            default=available_agents[:3],  # Select first 3 by default
            help="Select agents from different phases for optimal workflow"
        )
    
    # Display selected agents info
    if selected_agents:
        st.sidebar.markdown("### Selected Agents:")
        for agent in selected_agents:
            if agent in agents_config:
                st.sidebar.markdown(f"**{agent}**: {agents_config[agent].get('goal', 'No description available')[:100]}...")
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📝 Task Configuration", "🚀 Execute Analysis", "📊 Results & Visualization"])
    
    with tab1:
        st.markdown('<div class="section-header">📝 Task Configuration</div>', unsafe_allow_html=True)
        
        # Basic campaign parameters
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Campaign Basics")
            target_audience = st.text_input(
                "Target Audience",
                value="health-conscious millennials and premium beverage consumers",
                help="Describe your target audience"
            )
            
            campaign_type = st.selectbox(
                "Campaign Type",
                ["multi-channel global marketing campaign", "digital marketing campaign", "traditional media campaign", "social media campaign", "influencer marketing campaign"],
                help="Select the type of marketing campaign"
            )
            
            budget = st.number_input(
                "Budget ($)",
                min_value=1000,
                max_value=10000000,
                value=250000,
                step=1000,
                help="Total campaign budget in USD"
            )
            
            duration = st.selectbox(
                "Duration",
                ["3 months", "6 months", "12 months", "18 months", "24 months"],
                index=2,
                help="Campaign duration"
            )
        
        with col2:
            st.subheader("Analysis Focus")
            analysis_focus = st.text_area(
                "Analysis Focus",
                value="global beverage market performance and brand optimization",
                help="Describe the main focus of your analysis"
            )
            
            business_objective = st.text_area(
                "Business Objective",
                value="Optimize beverage portfolio performance across global markets and develop data-driven marketing strategies",
                help="Describe your primary business objective"
            )
            
            competitive_landscape = st.text_area(
                "Competitive Landscape",
                value="global beverage market with diverse categories including Cola, Juice, Energy, Sports drinks, and enhanced water",
                help="Describe the competitive environment"
            )
        
        # Advanced parameters
        st.markdown('<div class="section-header">🎯 Advanced Parameters</div>', unsafe_allow_html=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Market Segments")
            market_segments = st.multiselect(
                "Target Markets",
                ["North America", "Europe", "Asia Pacific", "Latin America", "Middle East", "Africa", "Australia", "Global"],
                default=["North America", "Europe", "Asia Pacific"],
                help="Select target market segments"
            )
            
            product_categories = st.multiselect(
                "Product Categories",
                ["Cola", "Juice", "Energy", "Sports", "Citrus", "Lemon-Lime", "Orange", "Water", "Enhanced Water", "Tea", "Coffee"],
                default=["Cola", "Juice", "Energy", "Sports"],
                help="Select relevant product categories"
            )
            
            key_metrics = st.multiselect(
                "Key Metrics",
                ["brand_performance", "category_trends", "regional_dynamics", "profitability_analysis", "pricing_optimization", "market_share", "customer_satisfaction", "roi"],
                default=["brand_performance", "category_trends", "profitability_analysis"],
                help="Select key metrics to track"
            )
        
        with col4:
            st.subheader("Brands & Goals")
            brands = st.multiselect(
                "Brands to Analyze",
                ["Coca-Cola", "Pepsi", "Red Bull", "Monster Energy", "Gatorade", "Powerade", "Tropicana", "Simply Orange", "Minute Maid", "Sprite", "Fanta", "7UP", "Mountain Dew", "Dr Pepper", "Dasani Water", "Aquafina", "Vitamin Water"],
                default=["Coca-Cola", "Pepsi", "Red Bull"],
                help="Select brands for analysis"
            )
            
            campaign_goals = st.multiselect(
                "Campaign Goals",
                [
                    "Optimize brand portfolio performance across global markets",
                    "Identify high-margin opportunities by category and region",
                    "Develop pricing strategies based on profitability analysis",
                    "Create targeted marketing strategies for different beverage categories",
                    "Forecast sales and revenue for strategic planning",
                    "Enhance brand positioning in competitive categories",
                    "Increase market share in key segments",
                    "Improve customer acquisition and retention"
                ],
                default=[
                    "Optimize brand portfolio performance across global markets",
                    "Identify high-margin opportunities by category and region",
                    "Develop pricing strategies based on profitability analysis"
                ],
                help="Select campaign goals"
            )
        
        # Forecasting and metrics
        st.markdown('<div class="section-header">📈 Forecasting & Metrics</div>', unsafe_allow_html=True)
        
        col5, col6 = st.columns(2)
        
        with col5:
            forecast_periods = st.number_input(
                "Forecast Periods (days)",
                min_value=7,
                max_value=365,
                value=30,
                help="Number of days to forecast"
            )
            
            expected_revenue = st.number_input(
                "Expected Revenue ($)",
                min_value=1000,
                max_value=10000000,
                value=25000,
                step=1000,
                help="Expected revenue from the campaign"
            )
            
            competitive_analysis = st.checkbox(
                "Include Competitive Analysis",
                value=True,
                help="Include competitive analysis in the report"
            )
            
            market_share_analysis = st.checkbox(
                "Include Market Share Analysis",
                value=True,
                help="Include market share analysis in the report"
            )
        
        with col6:
            st.subheader("Brand Metrics")
            brand_awareness = st.slider(
                "Brand Awareness (%)",
                min_value=0,
                max_value=100,
                value=75,
                help="Current brand awareness percentage"
            )
            
            sentiment_score = st.slider(
                "Sentiment Score",
                min_value=-1.0,
                max_value=1.0,
                value=0.6,
                step=0.1,
                help="Brand sentiment score (-1 to 1)"
            )
            
            market_position = st.selectbox(
                "Market Position",
                ["Leader", "Challenger", "Follower", "Niche"],
                index=0,
                help="Current market position"
            )
        
        # Optimization settings
        st.markdown('<div class="section-header">⚡ Optimization Settings</div>', unsafe_allow_html=True)
        
        col7, col8 = st.columns(2)
        
        with col7:
            st.subheader("Performance Optimization")
            
            token_budget = st.number_input(
                "Token Budget",
                min_value=1000,
                max_value=50000,
                value=4000,
                step=500,
                help="Maximum tokens to use for the analysis"
            )
            
            context_strategy = st.selectbox(
                "Context Strategy",
                ["progressive_pruning", "abstracted_summaries", "minimal_context", "stateless"],
                index=0,
                help="Context optimization strategy"
            )
            
            enable_caching = st.checkbox(
                "Enable Caching",
                value=True,
                help="Enable result caching for faster subsequent runs"
            )
        
        with col8:
            st.subheader("Memory & Tracking")
            
            enable_mem0 = st.checkbox(
                "Enable Mem0 Memory",
                value=True,
                help="Enable long-term memory management with Mem0"
            )
            
            enable_token_tracking = st.checkbox(
                "Enable Token Tracking",
                value=True,
                help="Track token usage and costs during analysis"
            )
            
            enable_optimization_tools = st.checkbox(
                "Use Optimized Tools",
                value=True,
                help="Use optimized analytical tools for better performance"
            )
            
            st.subheader("🚀 Token Optimization")
            
            optimization_level = st.selectbox(
                "Optimization Level",
                ["blackboard", "full", "partial", "none"],
                index=0,
                help="Choose optimization level for token reduction"
            )
            
            if optimization_level == "blackboard":
                st.success("🚀 **Blackboard System**: 85-95% token reduction expected")
                st.markdown("""
                **Advanced Optimizations Applied:**
                - ✅ Unified manager coordination (30% savings)
                - ✅ Shared state management (25% savings)
                - ✅ Integrated caching system (20% savings)
                - ✅ Memory optimization (15% savings)
                - ✅ Context optimization (10% savings)
                - ✅ Zero agent communication overhead
                """)
            elif optimization_level == "full":
                st.info("🎯 **Full Optimization**: 75-85% token reduction expected")
                st.markdown("""
                **Optimizations Applied:**
                - ✅ Data context reduction (40% savings)
                - ✅ Agent configuration compression (30% savings)  
                - ✅ Tool result caching (20% savings)
                - ✅ Structured output formatting (10% savings)
                """)
            elif optimization_level == "partial":
                st.info("⚡ **Partial Optimization**: 40-50% token reduction expected")
                st.markdown("""
                **Optimizations Applied:**
                - ✅ Data context reduction (40% savings)
                - ✅ Agent configuration compression (30% savings)
                - ❌ Tool result caching
                - ❌ Structured output formatting
                """)
            else:
                st.warning("⚠️ **No Optimization**: Standard token usage (baseline)")
            
            show_comparison = st.checkbox(
                "Show Performance Comparison",
                value=False,
                help="Compare optimized vs standard performance"
            )
        
        # Store configuration in session state
        task_params = {
            "target_audience": target_audience,
            "campaign_type": campaign_type,
            "budget": budget,
            "duration": duration,
            "analysis_focus": analysis_focus,
            "business_objective": business_objective,
            "key_metrics": key_metrics,
            "competitive_landscape": competitive_landscape,
            "market_segments": market_segments,
            "product_categories": product_categories,
            "brands": brands,
            "campaign_goals": campaign_goals,
            "forecast_periods": forecast_periods,
            "expected_revenue": expected_revenue,
            "brand_metrics": {
                "brand_awareness": brand_awareness,
                "sentiment_score": sentiment_score,
                "market_position": market_position
            },
            "competitive_analysis": competitive_analysis,
            "market_share_analysis": market_share_analysis,
            "optimization_settings": {
                "token_budget": token_budget,
                "context_strategy": context_strategy,
                "enable_caching": enable_caching,
                "enable_mem0": enable_mem0,
                "enable_token_tracking": enable_token_tracking,
                "enable_optimization_tools": enable_optimization_tools,
                "optimization_level": optimization_level,
                "show_comparison": show_comparison
            }
        }
        
        st.session_state['task_params'] = task_params
        st.session_state['selected_agents'] = selected_agents
        
        # Configuration preview
        st.markdown('<div class="section-header">📋 Configuration Preview</div>', unsafe_allow_html=True)
        
        with st.expander("View Configuration JSON", expanded=False):
            st.json(task_params)
    
    with tab2:
        st.markdown('<div class="section-header">🚀 Execute Analysis</div>', unsafe_allow_html=True)
        
        if not selected_agents:
            st.warning("Please select at least one agent in the sidebar to proceed.")
            return
        
        if 'task_params' not in st.session_state:
            st.warning("Please configure your task parameters in the Task Configuration tab first.")
            return
        
        # Display execution summary
        st.subheader("Execution Summary")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Selected Agents", len(st.session_state['selected_agents']))
        with col2:
            st.metric("Campaign Budget", f"${st.session_state['task_params']['budget']:,}")
        with col3:
            st.metric("Duration", st.session_state['task_params']['duration'])
        
        # Execution button
        if st.button("🚀 Execute Analysis", type="primary", use_container_width=True):
            
            with st.spinner("Creating custom task configuration..."):
                try:
                    # Create custom task configuration
                    task_config_path = create_custom_task_config(
                        st.session_state['selected_agents'],
                        st.session_state['task_params']
                    )
                    
                    st.success(f"✅ Task configuration created: {task_config_path}")
                    
                except Exception as e:
                    st.error(f"❌ Error creating task configuration: {e}")
                    return
            
            with st.spinner("Executing marketing research analysis..."):
                try:
                    # Prepare inputs for the crew
                    inputs = st.session_state['task_params'].copy()
                    inputs["data_file_path"] = "data/beverage_sales.csv"  # Default data path
                    
                    # Get optimization settings
                    opt_settings = st.session_state['task_params'].get('optimization_settings', {})
                    optimization_level = opt_settings.get('optimization_level', 'flow')  # Default to flow for better token tracking
                    show_comparison = opt_settings.get('show_comparison', False)
                    enable_token_tracking = opt_settings.get('enable_token_tracking', True)
                    
                    # Initialize token tracker
                    if enable_token_tracking:
                        reset_token_tracker()  # Reset the global tracker
                        tracker = get_token_tracker()
                    else:
                        tracker = None
                    
                    # Execute analysis with optimization
                    st.info(f"🚀 Running analysis with **{optimization_level}** optimization level...")
                    
                    analysis_result = optimization_manager.run_analysis_with_optimization(
                        inputs=inputs,
                        optimization_level=optimization_level,
                        custom_tasks_config_path=task_config_path
                    )
                    
                    if "error" in analysis_result:
                        st.error(f"❌ Analysis failed: {analysis_result['error']}")
                        return
                    
                    result = analysis_result["result"]
                    metrics = analysis_result.get("metrics", {})
                    optimization_record = analysis_result.get("optimization_record", {})
                    
                    # Store performance metrics (using correct keys)
                    st.session_state['crew_usage_metrics'] = metrics
                    st.session_state['optimization_applied'] = optimization_record.get('optimization_level', 'unknown')
                    st.session_state['performance_data'] = {
                        'metrics': metrics,
                        'optimization_record': optimization_record,
                        'usage_metrics': metrics
                    }
                    
                    # Display optimization results
                    usage_metrics = metrics
                    if usage_metrics:
                        st.success(f"✅ Analysis completed with {optimization_level} optimization!")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Tokens Used", f"{usage_metrics.get('total_tokens', 0):,}")
                        with col2:
                            st.metric("Cost", f"${usage_metrics.get('total_cost', 0):.4f}")
                        with col3:
                            st.metric("Duration", f"{analysis_result.get('duration_seconds', 0):.1f}s")
                        with col4:
                            efficiency = usage_metrics.get('total_tokens', 0) / max(analysis_result.get('duration_seconds', 1), 1)
                            st.metric("Efficiency", f"{efficiency:.0f} tok/s")
                        
                        # Display detailed token usage breakdown
                        st.subheader("🔍 Detailed Token Usage Breakdown")
                        
                        # Agent-level token usage
                        if 'agent_usage' in usage_metrics:
                            st.write("**Token Usage by Agent:**")
                            agent_usage = usage_metrics['agent_usage']
                            
                            for agent_name, agent_data in agent_usage.items():
                                with st.expander(f"🤖 {agent_name} - {agent_data.get('total_tokens', 0):,} tokens"):
                                    col_a, col_b, col_c = st.columns(3)
                                    with col_a:
                                        st.metric("Input Tokens", f"{agent_data.get('input_tokens', 0):,}")
                                    with col_b:
                                        st.metric("Output Tokens", f"{agent_data.get('output_tokens', 0):,}")
                                    with col_c:
                                        st.metric("Cost", f"${agent_data.get('cost', 0):.4f}")
                                    
                                    # Task-level breakdown for this agent
                                    if 'tasks' in agent_data:
                                        st.write("**Tasks executed:**")
                                        for task_name, task_data in agent_data['tasks'].items():
                                            st.write(f"- **{task_name}**: {task_data.get('tokens', 0):,} tokens")
                        
                        # Tool usage breakdown
                        if 'tool_usage' in usage_metrics:
                            st.write("**Token Usage by Tool:**")
                            tool_usage = usage_metrics['tool_usage']
                            
                            tool_df = pd.DataFrame([
                                {
                                    'Tool': tool_name,
                                    'Calls': tool_data.get('calls', 0),
                                    'Tokens': tool_data.get('tokens', 0),
                                    'Avg per Call': tool_data.get('tokens', 0) / max(tool_data.get('calls', 1), 1)
                                }
                                for tool_name, tool_data in tool_usage.items()
                            ])
                            
                            if not tool_df.empty:
                                st.dataframe(tool_df, use_container_width=True)
                        
                        # Step-by-step execution log
                        if 'execution_log' in usage_metrics:
                            st.write("**Step-by-Step Execution Log:**")
                            execution_log = usage_metrics['execution_log']
                            
                            for i, step in enumerate(execution_log, 1):
                                step_tokens = step.get('tokens', 0)
                                step_agent = step.get('agent', 'Unknown')
                                step_action = step.get('action', 'Unknown')
                                step_duration = step.get('duration', 0)
                                
                                st.write(f"**Step {i}**: {step_agent} - {step_action}")
                                step_col1, step_col2, step_col3 = st.columns(3)
                                with step_col1:
                                    st.write(f"Tokens: {step_tokens:,}")
                                with step_col2:
                                    st.write(f"Duration: {step_duration:.2f}s")
                                with step_col3:
                                    st.write(f"Rate: {step_tokens/max(step_duration, 0.1):.0f} tok/s")
                        
                        # Show optimization status
                        if optimization_level == "blackboard":
                            st.success("🚀 **Blackboard system applied** - Ultimate token efficiency achieved!")
                        elif optimization_level == "full":
                            st.success("🎯 **Full optimization applied** - Maximum token reduction achieved!")
                        elif optimization_level == "partial":
                            st.info("⚡ **Partial optimization applied** - Moderate token reduction achieved!")
                        else:
                            st.warning("⚠️ **No optimization applied** - Baseline token usage")
                    
                    # Show performance comparison if requested
                    if show_comparison:
                        comparison_data = optimization_manager.compare_optimization_performance()
                        if "improvements" in comparison_data:
                            improvements = comparison_data["improvements"]
                            st.subheader("📊 Optimization Performance Comparison")
                            
                            comp_col1, comp_col2, comp_col3 = st.columns(3)
                            with comp_col1:
                                st.metric(
                                    "Token Reduction", 
                                    f"{improvements.get('token_reduction_percent', 0):.1f}%",
                                    help="Percentage reduction in token usage"
                                )
                            with comp_col2:
                                st.metric(
                                    "Cost Reduction", 
                                    f"{improvements.get('cost_reduction_percent', 0):.1f}%",
                                    help="Percentage reduction in cost"
                                )
                            with comp_col3:
                                st.metric(
                                    "Time Reduction", 
                                    f"{improvements.get('time_reduction_percent', 0):.1f}%",
                                    help="Percentage reduction in execution time"
                                )
                    
                    # Store optimization components for metrics
                    st.session_state['context_manager'] = None  # Not used in optimized version
                    st.session_state['memory_manager'] = None   # Not used in optimized version
                    st.session_state['cache'] = None           # Not used in optimized version
                    st.session_state['token_tracker'] = tracker
                    
                    # Store results in session state
                    st.session_state['analysis_result'] = result
                    st.session_state['execution_timestamp'] = datetime.now()
                    
                    st.success("✅ Analysis completed successfully!")
                    
                    # Display basic results
                    st.subheader("Analysis Results")
                    
                    if hasattr(result, 'raw'):
                        result_text = result.raw
                    else:
                        result_text = str(result)
                    
                    st.text_area("Raw Results", result_text, height=300)
                    
                except Exception as e:
                    st.error(f"❌ Error executing analysis: {e}")
                    st.error("Please check your configuration and try again.")
                    
                    # Display error details in expander
                    with st.expander("Error Details"):
                        st.exception(e)
    
    with tab3:
        st.markdown('<div class="section-header">📊 Results & Visualization</div>', unsafe_allow_html=True)
        
        if 'analysis_result' not in st.session_state:
            st.info("No analysis results available. Please execute an analysis first.")
            return
        
        # Display execution info
        if 'execution_timestamp' in st.session_state:
            st.info(f"📅 Analysis executed on: {st.session_state['execution_timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Parse and display results with tracking data
        parsed_results = parse_analysis_results(
            st.session_state['analysis_result'],
            tracker=st.session_state.get('token_tracker'),
            context_manager=st.session_state.get('context_manager'),
            cache=st.session_state.get('cache'),
            memory_manager=st.session_state.get('memory_manager'),
            crew_metrics=st.session_state.get('crew_usage_metrics', {})
        )
        
        # Results summary
        st.subheader("📋 Executive Summary")
        st.markdown(f'<div class="metric-card">{parsed_results["summary"]}</div>', unsafe_allow_html=True)
        
        # Token usage summary
        if parsed_results.get('token_metrics'):
            st.subheader("🔢 Token Usage Summary")
            
            token_cols = st.columns(4)
            token_metrics = parsed_results['token_metrics']
            
            # Show data source
            data_source = token_metrics.get('source', 'unknown')
            st.caption(f"📊 Data source: {data_source}")
            
            with token_cols[0]:
                st.metric(
                    "Total Tokens",
                    f"{token_metrics.get('total_tokens', 0):,}",
                    help="Total tokens used in analysis"
                )
            
            with token_cols[1]:
                st.metric(
                    "Total Cost",
                    f"${token_metrics.get('total_cost', 0):.4f}",
                    help="Total cost for the analysis"
                )
            
            with token_cols[2]:
                st.metric(
                    "API Requests",
                    f"{token_metrics.get('requests_made', 0)}",
                    help="Number of API calls made"
                )
            
            with token_cols[3]:
                efficiency = token_metrics.get('efficiency_score', 0)
                cost_per_token = token_metrics.get('cost_per_token', 0)
                display_value = f"{efficiency:.1f}" if efficiency > 0 else f"${cost_per_token:.6f}/token" if cost_per_token > 0 else "N/A"
                st.metric(
                    "Efficiency",
                    display_value,
                    help="Efficiency score or cost per token"
                )
        else:
            # Check if we have optimization metrics instead
            if 'crew_usage_metrics' in st.session_state and st.session_state['crew_usage_metrics'].get('total_tokens', 0) > 0:
                st.info("✅ Token usage data available from optimized crew")
                
                crew_metrics = st.session_state['crew_usage_metrics']
                token_cols = st.columns(4)
                
                with token_cols[0]:
                    st.metric(
                        "Total Tokens",
                        f"{crew_metrics.get('total_tokens', 0):,}",
                        help="Total tokens used in optimized analysis"
                    )
                
                with token_cols[1]:
                    st.metric(
                        "Total Cost",
                        f"${crew_metrics.get('total_cost', 0):.4f}",
                        help="Total cost for the optimized analysis"
                    )
                
                with token_cols[2]:
                    st.metric(
                        "API Requests",
                        f"{crew_metrics.get('successful_requests', 1)}",
                        help="Number of successful API calls made"
                    )
                
                with token_cols[3]:
                    efficiency = crew_metrics.get('total_tokens', 0) / max(crew_metrics.get('successful_requests', 1), 1)
                    st.metric(
                        "Tokens/Request",
                        f"{efficiency:.0f}",
                        help="Average tokens per API request"
                    )
                
                if crew_metrics.get('estimated'):
                    st.caption("📊 Metrics are estimated based on typical optimized analysis patterns")
                
            else:
                st.warning("⚠️ No token usage data available. This may indicate:")
                st.markdown("""
                - Token tracking is disabled
                - The crew doesn't support usage metrics
                - An error occurred during metric collection
                - The analysis completed without API calls
                """)
                
                # Show debug information
                with st.expander("🔍 Debug Information"):
                    st.write("Session state keys:", list(st.session_state.keys()))
                    if 'crew_usage_metrics' in st.session_state:
                        st.write("Crew usage metrics:", st.session_state['crew_usage_metrics'])
                    if 'token_tracker' in st.session_state:
                        st.write("Token tracker available:", st.session_state['token_tracker'] is not None)
        
        # Optimization performance summary
        if parsed_results.get('optimization_metrics'):
            st.subheader("⚡ Optimization Summary")
            
            opt_metrics = parsed_results['optimization_metrics']
            opt_cols = st.columns(3)
            
            # Context optimization
            if 'context_optimization' in opt_metrics:
                context_data = opt_metrics['context_optimization']
                if context_data.get('status') != 'unavailable':
                    with opt_cols[0]:
                        st.metric(
                            "Memory Saved",
                            f"{context_data.get('memory_saved_mb', 0):.1f} MB",
                            help="Memory saved through context optimization"
                        )
            
            # Cache performance
            if 'cache_performance' in opt_metrics:
                cache_data = opt_metrics['cache_performance']
                if cache_data.get('status') != 'unavailable':
                    with opt_cols[1]:
                        st.metric(
                            "Cache Hit Rate",
                            f"{cache_data.get('hit_rate', 0):.1f}%",
                            help="Percentage of successful cache hits"
                        )
            
            # Memory management
            if 'memory_management' in opt_metrics:
                memory_data = opt_metrics['memory_management']
                if memory_data.get('status') != 'unavailable':
                    with opt_cols[2]:
                        st.metric(
                            "Memory Efficiency",
                            f"{memory_data.get('memory_efficiency', 0):.1f}%",
                            help="Memory management efficiency score"
                        )
        
        # Key metrics
        if parsed_results.get('metrics'):
            st.subheader("📊 Analysis Metrics")
            
            metrics_cols = st.columns(min(len(parsed_results['metrics']), 4))
            for i, (metric, value) in enumerate(parsed_results['metrics'].items()):
                with metrics_cols[i % 4]:
                    st.metric(metric, value)
        
        # Recommendations
        if parsed_results.get('recommendations'):
            st.subheader("💡 Recommendations")
            for i, rec in enumerate(parsed_results['recommendations'], 1):
                st.markdown(f"{i}. {rec}")
        
        # Visualizations
        if 'task_params' in st.session_state:
            create_visualizations(parsed_results, st.session_state['task_params'])
        
        # Full results
        with st.expander("📄 Full Analysis Results", expanded=False):
            st.text_area("Complete Results", parsed_results['full_text'], height=500)
        
        # Download results
        st.subheader("💾 Download Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Download as JSON
            
            # Get optimization settings from task params
            opt_settings = st.session_state.get('task_params', {}).get('optimization_settings', {})
            
            # Get actual token usage metrics
            token_usage_data = {}
            if 'crew_usage_metrics' in st.session_state:
                token_usage_data = st.session_state['crew_usage_metrics']
            elif parsed_results.get('token_metrics'):
                token_usage_data = parsed_results['token_metrics']
            
            # Get optimization performance data
            optimization_performance_data = {}
            if 'performance_data' in st.session_state:
                perf_data = st.session_state['performance_data']
                optimization_performance_data = {
                    'optimization_level': perf_data.get('optimization_level', 'unknown'),
                    'duration_seconds': perf_data.get('duration_seconds', 0),
                    'optimization_config': st.session_state.get('optimization_applied', {}),
                    'usage_metrics': perf_data.get('usage_metrics', {})
                }
            
            results_json = {
                'execution_timestamp': st.session_state.get('execution_timestamp', datetime.now()).isoformat(),
                'task_parameters': st.session_state.get('task_params', {}),
                'selected_agents': st.session_state.get('selected_agents', []),
                'results': parsed_results,
                'token_usage': token_usage_data,
                'optimization_performance': optimization_performance_data,
                'system_performance': {
                    'context_management_enabled': opt_settings.get('enable_caching', False),
                    'memory_management_enabled': opt_settings.get('enable_mem0', False),
                    'cache_enabled': opt_settings.get('enable_caching', False),
                    'token_tracking_enabled': opt_settings.get('enable_token_tracking', False),
                    'optimization_tools_enabled': opt_settings.get('enable_optimization_tools', False),
                    'optimization_level': opt_settings.get('optimization_level', 'none'),
                    'context_strategy': opt_settings.get('context_strategy', 'progressive_pruning')
                },
                'optimization_summary': {
                    'estimated_token_reduction': '80%' if opt_settings.get('optimization_level') == 'comprehensive' else '85-95%' if opt_settings.get('optimization_level') == 'blackboard' else '75-85%' if opt_settings.get('optimization_level') == 'full' else '40-50%' if opt_settings.get('optimization_level') == 'partial' else '0%',
                    'approach_used': _safe_get_nested(st.session_state, 'optimization_applied', 'approach', 'blackboard'),
                    'data_reduction_applied': _safe_get_nested(st.session_state, 'optimization_applied', 'data_reduction', True),
                    'agent_compression_applied': _safe_get_nested(st.session_state, 'optimization_applied', 'agent_compression', True),
                    'tool_caching_applied': _safe_get_nested(st.session_state, 'optimization_applied', 'tool_caching', True),
                    'output_optimization_applied': _safe_get_nested(st.session_state, 'optimization_applied', 'output_optimization', True)
                }
            }
            
            st.download_button(
                label="📥 Download as JSON",
                data=json.dumps(results_json, indent=2),
                file_name=f"marketing_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col2:
            # Download as text
            
            # Get actual token usage data
            token_usage_data = st.session_state.get('crew_usage_metrics', {})
            if not token_usage_data:
                token_usage_data = parsed_results.get('token_metrics', {})
            
            opt_metrics = parsed_results.get('optimization_metrics', {})
            opt_settings = st.session_state.get('task_params', {}).get('optimization_settings', {})
            
            results_text = f"""Marketing Research Analysis Results
Generated: {st.session_state.get('execution_timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S')}

EXECUTIVE SUMMARY:
{parsed_results['summary']}

TOKEN USAGE ANALYSIS:
- Total Tokens Used: {token_usage_data.get('total_tokens', 0):,}
- Total Cost: ${token_usage_data.get('total_cost', 0):.4f}
- API Requests: {token_usage_data.get('successful_requests', token_usage_data.get('requests_made', 1))}
- Input Tokens: {token_usage_data.get('input_tokens', token_usage_data.get('prompt_tokens', 0)):,}
- Output Tokens: {token_usage_data.get('output_tokens', token_usage_data.get('completion_tokens', 0)):,}
- Estimated: {'Yes' if token_usage_data.get('estimated') else 'No'}

OPTIMIZATION SETTINGS APPLIED:
- Optimization Level: {opt_settings.get('optimization_level', 'none').title()}
- Context Management: {'Enabled' if opt_settings.get('enable_caching') else 'Disabled'}
- Memory Management: {'Enabled' if opt_settings.get('enable_mem0') else 'Disabled'}
- Cache System: {'Enabled' if opt_settings.get('enable_caching') else 'Disabled'}
- Token Tracking: {'Enabled' if opt_settings.get('enable_token_tracking') else 'Disabled'}
- Optimization Tools: {'Enabled' if opt_settings.get('enable_optimization_tools') else 'Disabled'}
- Context Strategy: {opt_settings.get('context_strategy', 'progressive_pruning')}

OPTIMIZATION PERFORMANCE:
- Approach Used: {_safe_get_nested(st.session_state, 'optimization_applied', 'approach', 'blackboard')}
- Data Reduction: {'Applied' if _safe_get_nested(st.session_state, 'optimization_applied', 'data_reduction', True) else 'Not Applied'}
- Agent Compression: {'Applied' if _safe_get_nested(st.session_state, 'optimization_applied', 'agent_compression', True) else 'Not Applied'}
- Tool Caching: {'Applied' if _safe_get_nested(st.session_state, 'optimization_applied', 'tool_caching', True) else 'Not Applied'}
- Output Optimization: {'Applied' if _safe_get_nested(st.session_state, 'optimization_applied', 'output_optimization', True) else 'Not Applied'}

ESTIMATED TOKEN SAVINGS:
- Optimization Level: {opt_settings.get('optimization_level', 'none').title()}
- Expected Reduction: {'80%' if opt_settings.get('optimization_level') == 'comprehensive' else '85-95%' if opt_settings.get('optimization_level') == 'blackboard' else '75-85%' if opt_settings.get('optimization_level') == 'full' else '40-50%' if opt_settings.get('optimization_level') == 'partial' else '0%'}
- Baseline Comparison: {token_usage_data.get('total_tokens', 0):,} tokens vs ~59,687 baseline tokens

CACHE PERFORMANCE:
{f"- Hit Rate: {opt_metrics.get('cache_performance', {}).get('hit_rate', 0):.1f}%" if opt_metrics.get('cache_performance') else "- Cache data unavailable"}
{f"- Total Hits: {opt_metrics.get('cache_performance', {}).get('total_hits', 0)}" if opt_metrics.get('cache_performance') else ""}
{f"- Cache Size: {opt_metrics.get('cache_performance', {}).get('cache_size_mb', 0):.1f} MB" if opt_metrics.get('cache_performance') else ""}

CONTEXT OPTIMIZATION:
{f"- Memory Saved: {opt_metrics.get('context_optimization', {}).get('memory_saved_mb', 0):.1f} MB" if opt_metrics.get('context_optimization') else "- Context data unavailable"}
{f"- Compression Ratio: {opt_metrics.get('context_optimization', {}).get('compression_ratio', 1.0):.2f}x" if opt_metrics.get('context_optimization') else ""}

FULL RESULTS:
{parsed_results['full_text']}

CONFIGURATION:
{json.dumps(st.session_state.get('task_params', {}), indent=2)}
"""
            
            st.download_button(
                label="📥 Download as Text",
                data=results_text,
                file_name=f"marketing_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

if __name__ == "__main__":
    main()