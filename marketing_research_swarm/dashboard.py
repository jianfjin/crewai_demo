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

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from marketing_research_swarm.crew import MarketingResearchCrew
    from marketing_research_swarm.main import run_specific_analysis
except ImportError as e:
    st.error(f"Import error: {e}")
    st.error("Please ensure you're running from the correct directory and all dependencies are installed.")

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
        agents_path = 'src/marketing_research_swarm/config/agents.yaml'
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
        }
    }
    
    # Create tasks for selected agents
    for i, agent in enumerate(selected_agents):
        if agent in agent_task_mapping:
            task_name = f"{agent}_task_{task_id}"
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

def parse_analysis_results(result: Any) -> Dict[str, Any]:
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
        'recommendations': []
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

def create_visualizations(parsed_results: Dict[str, Any], task_params: Dict[str, Any]):
    """Create visualizations based on analysis results"""
    
    st.markdown('<div class="section-header">📊 Analysis Visualizations</div>', unsafe_allow_html=True)
    
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
    
    # Sidebar for agent selection
    st.sidebar.markdown("## 🤖 Agent Selection")
    st.sidebar.markdown("Select the agents you want to include in your analysis:")
    
    available_agents = list(agents_config.keys())
    selected_agents = st.sidebar.multiselect(
        "Choose Agents",
        available_agents,
        default=available_agents[:3],  # Select first 3 by default
        help="Select one or more agents to participate in the analysis"
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
            "market_share_analysis": market_share_analysis
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
                    
                    # Execute the analysis
                    agents_config_path = 'src/marketing_research_swarm/config/agents.yaml'
                    
                    # Initialize and run the crew
                    crew = MarketingResearchCrew(agents_config_path, task_config_path)
                    result = crew.kickoff(inputs)
                    
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
        
        # Parse and display results
        parsed_results = parse_analysis_results(st.session_state['analysis_result'])
        
        # Results summary
        st.subheader("📋 Executive Summary")
        st.markdown(f'<div class="metric-card">{parsed_results["summary"]}</div>', unsafe_allow_html=True)
        
        # Key metrics
        if parsed_results.get('metrics'):
            st.subheader("📊 Key Metrics")
            
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
            results_json = {
                'execution_timestamp': st.session_state.get('execution_timestamp', datetime.now()).isoformat(),
                'task_parameters': st.session_state.get('task_params', {}),
                'selected_agents': st.session_state.get('selected_agents', []),
                'results': parsed_results
            }
            
            st.download_button(
                label="📥 Download as JSON",
                data=json.dumps(results_json, indent=2),
                file_name=f"marketing_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        with col2:
            # Download as text
            results_text = f"""Marketing Research Analysis Results
Generated: {st.session_state.get('execution_timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M:%S')}

EXECUTIVE SUMMARY:
{parsed_results['summary']}

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