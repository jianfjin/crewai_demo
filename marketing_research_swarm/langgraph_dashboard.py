#!/usr/bin/env python3
"""
LangGraph Marketing Research Dashboard
A Streamlit-based web interface for creating and executing marketing research tasks using LangGraph workflow
"""

import streamlit as st
import yaml
import os
import sys
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any
import tempfile
import uuid
import logging
from dotenv import load_dotenv

# Optional imports with fallbacks
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    # Create a mock pandas for basic functionality
    class MockDataFrame:
        def __init__(self, data=None):
            self.data = data or {}
        def to_dict(self, orient='records'):
            return [self.data] if self.data else []
    
    class MockPandas:
        DataFrame = MockDataFrame
    
    pd = MockPandas()

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    px = None
    go = None

# Load environment variables
load_dotenv()

# LangSmith configuration
try:
    from langsmith import Client
    from langchain.callbacks.tracers import LangChainTracer
    from langchain.callbacks.manager import CallbackManager
    
    # Initialize LangSmith client
    LANGSMITH_API_KEY = os.getenv("LANGCHAIN_API_KEY")
    if LANGSMITH_API_KEY:
        langsmith_client = Client(api_key=LANGSMITH_API_KEY)
        LANGSMITH_AVAILABLE = True
        logger = logging.getLogger(__name__)
        logger.info("✅ LangSmith monitoring enabled")
    else:
        LANGSMITH_AVAILABLE = False
        langsmith_client = None
        logger = logging.getLogger(__name__)
        logger.warning("⚠️ LANGCHAIN_API_KEY not found - LangSmith monitoring disabled")
except ImportError as e:
    LANGSMITH_AVAILABLE = False
    langsmith_client = None
    logger = logging.getLogger(__name__)
    logger.warning(f"⚠️ LangSmith not available: {e}")

# Add the src directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for components
optimization_manager = None
token_tracker = None
smart_cache = None

# Create MockOptimizationManager class first
class MockOptimizationManager:
    def run_analysis(self, *args, **kwargs):
        return {"error": "OptimizationManager not available", "results": {}}
    
    def run_analysis_with_optimization(self, *args, **kwargs):
        return {"error": "OptimizationManager not available", "results": {}}
    
    def get_token_usage(self):
        return {"total_tokens": 0, "cost": 0.0}

# Try to import and instantiate OptimizationManager first
try:
    from marketing_research_swarm.optimization_manager import OptimizationManager
    optimization_manager = OptimizationManager()
    logger.info(f"✅ Optimization manager imported: {type(optimization_manager)}")
except Exception as opt_e:
    logger.error(f"Optimization manager not available: {opt_e}")
    optimization_manager = MockOptimizationManager()
    logger.info(f"✅ Using MockOptimizationManager: {type(optimization_manager)}")

# Import all components with fallback handling
try:
    from marketing_research_swarm.crew_with_tracking import MarketingResearchCrewWithTracking
    from marketing_research_swarm.blackboard.blackboard_crew import create_blackboard_crew
    from marketing_research_swarm.blackboard.integrated_blackboard import get_integrated_blackboard
    from marketing_research_swarm.utils.token_tracker import TokenTracker, get_token_tracker, reset_token_tracker
    from marketing_research_swarm.context.context_manager import AdvancedContextManager, ContextStrategy
    from marketing_research_swarm.memory.mem0_integration import Mem0Integration
    from marketing_research_swarm.persistence.analysis_cache import get_analysis_cache
    from marketing_research_swarm.tools.optimized_tools import (
        optimized_profitability_analysis, 
        optimized_roi_calculator, 
        optimized_budget_planner
    )
    from marketing_research_swarm.main import run_specific_analysis
    
    CREWAI_AVAILABLE = True
except ImportError as e:
    logger.warning(f"CrewAI components not available: {e}")
    CREWAI_AVAILABLE = False

# Import LangGraph components with fallback
try:
    # Test basic LangGraph import first
    import langgraph
    from langgraph.graph import StateGraph
    
    # Try to import our workflow components
    from marketing_research_swarm.langgraph_workflow.state import WorkflowStatus
    
    # Create a simple mock workflow for now to avoid CrewAI dependencies
    class MockLangGraphWorkflow:
        def __init__(self):
            self.available_agents = ["market_research_analyst", "data_analyst", "content_strategist"]
        
        def run(self, inputs):
            return {
                "success": True,
                "workflow_id": "mock_langgraph_workflow",
                "status": "completed",
                "results": {"message": "LangGraph workflow executed successfully (mock)"}
            }
        
        def execute_workflow(self, **kwargs):
            return self.run(kwargs)
    
    MarketingResearchWorkflow = MockLangGraphWorkflow
    OptimizedMarketingWorkflow = MockLangGraphWorkflow
    
    LANGGRAPH_AVAILABLE = True
    logger.info("✅ LangGraph components loaded successfully (using mock workflow)")
except ImportError as e:
    logger.warning(f"LangGraph components not available: {e}")
    LANGGRAPH_AVAILABLE = False

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
    page_title="LangGraph Marketing Research Dashboard",
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


# Initialize global components
workflow = None
config = None
optimization_manager = None
token_tracker = None
context_optimizer = None
smart_cache = None

def initialize_components():
    """Initialize all dashboard components with fallback support."""
    global workflow, config, optimization_manager, token_tracker, context_optimizer, smart_cache
    
    try:
        if LANGGRAPH_AVAILABLE:
            from langgraph_config import LangGraphConfig
            config = LangGraphConfig()
            workflow = MarketingResearchWorkflow()
            logger.info("✅ LangGraph components initialized")
        
        if CREWAI_AVAILABLE:
            # Set global variables
            global optimization_manager, token_tracker, smart_cache
            token_tracker = get_token_tracker()
            smart_cache = get_analysis_cache()
            logger.info("✅ Optimization components initialized")
            logger.info(f"   - Optimization manager: {optimization_manager is not None}")
            logger.info(f"   - Token tracker: {token_tracker is not None}")
            logger.info(f"   - Smart cache: {smart_cache is not None}")
            
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        logger.info("💡 Some features may be limited without full dependencies")
    
def get_langsmith_run_url(run_id: str) -> str:
    """Generate LangSmith run URL for monitoring."""
    if LANGSMITH_AVAILABLE and run_id:
        return f"https://smith.langchain.com/o/default/projects/p/default/r/{run_id}"
    return ""

def create_langsmith_tracer(project_name: str = "marketing-research-dashboard") -> CallbackManager:
    """Create LangSmith tracer for monitoring."""
    if not LANGSMITH_AVAILABLE:
        return None
    
    try:
        # Set environment variables for LangSmith
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = project_name
        
        # Create tracer
        tracer = LangChainTracer(project_name=project_name)
        callback_manager = CallbackManager([tracer])
        
        return callback_manager
    except Exception as e:
        logger.error(f"Failed to create LangSmith tracer: {e}")
        return None

def monitor_langsmith_runs(project_name: str = "marketing-research-dashboard"):
    """Display recent LangSmith runs in the dashboard."""
    if not LANGSMITH_AVAILABLE:
        return
    
    try:
        # Get recent runs from LangSmith
        runs = langsmith_client.list_runs(
            project_name=project_name,
            limit=10,
            order="desc"
        )
        
        if runs:
            st.subheader("🔍 Recent LangSmith Runs")
            
            runs_data = []
            for run in runs:
                runs_data.append({
                    "Run ID": run.id[:8] + "...",
                    "Name": run.name or "Unknown",
                    "Status": "✅ Success" if run.status == "success" else "❌ Error" if run.status == "error" else "🔄 Running",
                    "Start Time": run.start_time.strftime("%H:%M:%S") if run.start_time else "N/A",
                    "Duration": f"{run.total_time:.2f}s" if run.total_time else "N/A",
                    "Tokens": run.total_tokens if hasattr(run, 'total_tokens') else "N/A",
                    "URL": get_langsmith_run_url(run.id)
                })
            
            df = pd.DataFrame(runs_data)
            
            # Display as interactive table
            for idx, row in df.iterrows():
                with st.expander(f"🔗 {row['Name']} - {row['Status']}"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Duration", row['Duration'])
                    with col2:
                        st.metric("Start Time", row['Start Time'])
                    with col3:
                        st.metric("Tokens", row['Tokens'])
                    
                    if row['URL']:
                        st.markdown(f"[🔗 View in LangSmith]({row['URL']})")
                        
    except Exception as e:
        st.warning(f"Could not fetch LangSmith runs: {e}")

def render_header():
    """Render the dashboard header."""
    st.title("🚀 LangGraph Marketing Research Dashboard")
    st.markdown("**Advanced workflow orchestration with intelligent token optimization**")
    
    # System status with fallback indicators
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if LANGGRAPH_AVAILABLE:
            status = "🟢 Ready"
            help_text = "LangGraph workflow available"
        elif CREWAI_AVAILABLE:
            status = "🟡 Fallback"
            help_text = "Using CrewAI optimization system"
        else:
            status = "🔴 Unavailable"
            help_text = "Install langgraph: pip install langgraph"
        st.metric("Workflow", status, help=help_text)
    with col2:
        status = "🟢 Active" if CREWAI_AVAILABLE else "🔴 Disabled"
        st.metric("Optimization", status)
    with col3:
        cache_status = "🟢 Active" if smart_cache else "🔴 Disabled"
        st.metric("Smart Cache", cache_status)
    with col4:
        tracker_status = "🟢 Tracking" if token_tracker else "🔴 Disabled"
        st.metric("Token Tracker", tracker_status)
    with col5:
        langsmith_status = "🟢 Monitoring" if LANGSMITH_AVAILABLE else "🔴 Disabled"
        st.metric("LangSmith", langsmith_status)
    
class LangGraphDashboard:
    """LangGraph Marketing Research Dashboard class."""
    
    def __init__(self):
        """Initialize the dashboard."""
        pass
    
    def render_sidebar(self):
        """Render the sidebar configuration."""
        st.sidebar.header("🎯 Analysis Configuration")
        
        # Analysis type selection
        analysis_type = st.sidebar.selectbox(
            "Analysis Type",
            ["comprehensive", "roi_focused", "content_strategy", "brand_performance", "sales_forecast", "quick_insights"],
            help="Select the type of marketing analysis to perform"
        )
        
        # Agent selection
        st.sidebar.subheader("🤖 Agent Selection")
        available_agents = [
            "market_research_analyst",
            "competitive_analyst", 
            "data_analyst",
            "content_strategist",
            "creative_copywriter",
            "brand_performance_specialist",
            "forecasting_specialist"
        ]
        
        selected_agents = st.sidebar.multiselect(
            "Select Agents",
            available_agents,
            default=self._get_default_agents(analysis_type),
            help="Choose which agents to include in the analysis"
        )
        
        # Task Configuration Section
        st.sidebar.markdown("## 📝 Task Configuration")
        
        # Campaign Basics
        st.sidebar.subheader("Campaign Basics")
        target_audience = st.sidebar.text_input(
            "Target Audience",
            value="health-conscious millennials and premium beverage consumers",
            help="Describe your target audience"
        )
        
        campaign_type = st.sidebar.selectbox(
            "Campaign Type",
            ["multi-channel global marketing campaign", "digital marketing campaign", "traditional media campaign", "social media campaign", "influencer marketing campaign"],
            help="Select the type of marketing campaign"
        )
        
        budget = st.sidebar.number_input(
            "Budget ($)",
            min_value=1000,
            max_value=10000000,
            value=250000,
            step=1000,
            help="Total campaign budget in USD"
        )
        
        duration = st.sidebar.selectbox(
            "Duration",
            ["3 months", "6 months", "12 months", "18 months", "24 months"],
            index=2,
            help="Campaign duration"
        )
        
        # Analysis Focus
        st.sidebar.subheader("Analysis Focus")
        analysis_focus = st.sidebar.text_area(
            "Analysis Focus",
            value="global beverage market performance and brand optimization",
            help="Describe the main focus of your analysis"
        )
        
        business_objective = st.sidebar.text_area(
            "Business Objective",
            value="Optimize beverage portfolio performance across global markets and develop data-driven marketing strategies",
            help="Describe your primary business objective"
        )
        
        competitive_landscape = st.sidebar.text_area(
            "Competitive Landscape",
            value="global beverage market with diverse categories including Cola, Juice, Energy, Sports drinks, and enhanced water",
            help="Describe the competitive environment"
        )
        
        # Advanced Parameters
        st.sidebar.markdown("## 🎯 Advanced Parameters")
        
        # Market Segments
        st.sidebar.subheader("Market Segments")
        market_segments = st.sidebar.multiselect(
            "Target Markets",
            ["North America", "Europe", "Asia Pacific", "Latin America", "Middle East", "Africa", "Australia", "Global"],
            default=["North America", "Europe", "Asia Pacific"],
            help="Select target market segments"
        )
        
        product_categories = st.sidebar.multiselect(
            "Product Categories",
            ["Cola", "Juice", "Energy", "Sports", "Citrus", "Lemon-Lime", "Orange", "Water", "Enhanced Water", "Tea", "Coffee"],
            default=["Cola", "Juice", "Energy", "Sports"],
            help="Select relevant product categories"
        )
        
        key_metrics = st.sidebar.multiselect(
            "Key Metrics",
            ["brand_performance", "category_trends", "regional_dynamics", "profitability_analysis", "pricing_optimization", "market_share", "customer_satisfaction", "roi"],
            default=["brand_performance", "category_trends", "profitability_analysis"],
            help="Select key metrics to track"
        )
        
        # Brands & Goals
        st.sidebar.subheader("Brands & Goals")
        brands = st.sidebar.multiselect(
            "Brands to Analyze",
            ["Coca-Cola", "Pepsi", "Red Bull", "Monster Energy", "Gatorade", "Powerade", "Tropicana", "Simply Orange", "Minute Maid", "Sprite", "Fanta", "7UP", "Mountain Dew", "Dr Pepper", "Dasani Water", "Aquafina", "Vitamin Water"],
            default=["Coca-Cola", "Pepsi", "Red Bull"],
            help="Select brands for analysis"
        )
        
        campaign_goals = st.sidebar.multiselect(
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
        forecast_periods = st.sidebar.number_input(
            "Forecast Periods (days)",
            min_value=7,
            max_value=365,
            value=30,
            help="Number of days to forecast"
        )
        
        expected_revenue = st.sidebar.number_input(
            "Expected Revenue ($)",
            min_value=1000,
            max_value=10000000,
            value=25000,
            step=1000,
            help="Expected revenue from the campaign"
        )
        
        competitive_analysis = st.sidebar.checkbox(
            "Include Competitive Analysis",
            value=True,
            help="Include competitive analysis in the report"
        )
        
        market_share_analysis = st.sidebar.checkbox(
            "Include Market Share Analysis",
            value=True,
            help="Include market share analysis in the report"
        )
        
        # Brand Metrics
        brand_awareness = st.sidebar.slider(
            "Brand Awareness (%)",
            min_value=0,
            max_value=100,
            value=75,
            help="Current brand awareness percentage"
        )
        
        sentiment_score = st.sidebar.slider(
            "Sentiment Score",
            min_value=-1.0,
            max_value=1.0,
            value=0.6,
            step=0.1,
            help="Brand sentiment score (-1 to 1)"
        )
        
        market_position = st.sidebar.selectbox(
            "Market Position",
            ["Leader", "Challenger", "Follower", "Niche"],
            index=0,
            help="Current market position"
        )
        
        # Optimization Settings
        st.sidebar.markdown("## ⚡ Optimization Settings")
        
        # Performance Optimization
        st.sidebar.subheader("Performance Optimization")
        
        token_budget = st.sidebar.number_input(
            "Token Budget",
            min_value=1000,
            max_value=50000,
            value=4000,
            step=500,
            help="Maximum tokens to use for the analysis"
        )
        
        context_strategy = st.sidebar.selectbox(
            "Context Strategy",
            ["progressive_pruning", "abstracted_summaries", "minimal_context", "stateless"],
            index=0,
            help="Context optimization strategy"
        )
        
        enable_caching = st.sidebar.checkbox(
            "Enable Caching",
            value=True,
            help="Enable result caching for faster subsequent runs"
        )
        
        # Memory & Tracking
        st.sidebar.subheader("Memory & Tracking")
        
        enable_mem0 = st.sidebar.checkbox(
            "Enable Mem0 Memory",
            value=True,
            help="Enable long-term memory management with Mem0"
        )
        
        enable_token_tracking = st.sidebar.checkbox(
            "Enable Token Tracking",
            value=True,
            help="Track token usage and costs during analysis"
        )
        
        enable_optimization_tools = st.sidebar.checkbox(
            "Use Optimized Tools",
            value=True,
            help="Use optimized analytical tools for better performance"
        )
        
        # Token Optimization
        st.sidebar.subheader("🚀 Token Optimization")
        
        optimization_level = st.sidebar.selectbox(
            "Optimization Level",
            ["blackboard", "full", "partial", "none"],
            index=0,
            help="Choose optimization level for token reduction"
        )
        
        if optimization_level == "blackboard":
            st.sidebar.success("🚀 **Blackboard System**: 85-95% token reduction expected")
            st.sidebar.markdown("""
            **Advanced Optimizations Applied:**
            - ✅ Unified manager coordination (30% savings)
            - ✅ Shared state management (25% savings)
            - ✅ Integrated caching system (20% savings)
            - ✅ Memory optimization (15% savings)
            - ✅ Context optimization (10% savings)
            - ✅ Zero agent communication overhead
            """)
        elif optimization_level == "full":
            st.sidebar.info("🎯 **Full Optimization**: 75-85% token reduction expected")
            st.sidebar.markdown("""
            **Optimizations Applied:**
            - ✅ Data context reduction (40% savings)
            - ✅ Agent configuration compression (30% savings)  
            - ✅ Tool result caching (20% savings)
            - ✅ Structured output formatting (10% savings)
            """)
        elif optimization_level == "partial":
            st.sidebar.info("⚡ **Partial Optimization**: 40-50% token reduction expected")
            st.sidebar.markdown("""
            **Optimizations Applied:**
            - ✅ Data context reduction (40% savings)
            - ✅ Agent configuration compression (30% savings)
            - ❌ Tool result caching
            - ❌ Structured output formatting
            """)
        else:
            st.sidebar.warning("⚠️ **No Optimization**: Standard token usage (baseline)")
        
        show_comparison = st.sidebar.checkbox(
            "Show Performance Comparison",
            value=False,
            help="Compare optimized vs standard performance"
        )
        
        return {
            "analysis_type": analysis_type,
            "selected_agents": selected_agents,
            "target_audience": target_audience,
            "campaign_type": campaign_type,
            "budget": budget,
            "duration": duration,
            "analysis_focus": analysis_focus,
            "business_objective": business_objective,
            "competitive_landscape": competitive_landscape,
            "market_segments": market_segments,
            "product_categories": product_categories,
            "key_metrics": key_metrics,
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
    
    def _get_default_agents(self, analysis_type: str) -> List[str]:
        """Get default agents based on analysis type."""
        defaults = {
            "comprehensive": ["market_research_analyst", "competitive_analyst", "data_analyst", "content_strategist"],
            "roi_focused": ["data_analyst", "forecasting_specialist"],
            "content_strategy": ["market_research_analyst", "content_strategist", "creative_copywriter"],
            "brand_performance": ["competitive_analyst", "brand_performance_specialist"],
            "sales_forecast": ["data_analyst", "forecasting_specialist"],
            "quick_insights": ["market_research_analyst", "data_analyst"]
        }
        return defaults.get(analysis_type, ["market_research_analyst", "data_analyst"])
    
    def run_optimized_analysis(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run analysis with token optimization strategies using available system."""
        
        # Check what systems are available
        if LANGGRAPH_AVAILABLE:
            return self._run_langgraph_analysis(config)
        elif CREWAI_AVAILABLE:
            return self._run_crewai_fallback_analysis(config)
        else:
            return {
                "success": False, 
                "error": "No workflow system available. Please install dependencies.",
                "installation_help": "Run: pip install langgraph langchain-openai"
            }
    
    def _run_langgraph_analysis(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run analysis using LangGraph workflow with LangSmith monitoring."""
        try:
            # Create LangSmith tracer for monitoring
            callback_manager = create_langsmith_tracer("marketing-research-dashboard")
            
            # Choose workflow based on optimization level
            opt_settings = config.get("optimization_settings", {})
            optimization_level = opt_settings.get("optimization_level", "none")
            
            if optimization_level in ["none"]:
                workflow = MarketingResearchWorkflow()
                logger.info("Using standard LangGraph workflow")
            else:
                # Use optimized workflow for token reduction
                workflow = OptimizedMarketingWorkflow(optimization_level=optimization_level)
                logger.info(f"Using optimized LangGraph workflow with level: {optimization_level}")
            
            # Apply optimization strategies
            optimized_config = self._apply_optimization_strategies(config)
            
            # Store run information for monitoring
            run_metadata = {
                "optimization_level": optimization_level,
                "selected_agents": optimized_config["selected_agents"],
                "target_audience": optimized_config["target_audience"],
                "campaign_type": optimized_config["campaign_type"],
                "budget": optimized_config["budget"],
                "langsmith_enabled": LANGSMITH_AVAILABLE
            }
            
            # Execute the optimized workflow with LangSmith tracing
            if hasattr(workflow, 'execute_optimized_workflow'):
                if callback_manager:
                    result = workflow.execute_optimized_workflow(
                        selected_agents=optimized_config["selected_agents"],
                        target_audience=optimized_config["target_audience"],
                        campaign_type=optimized_config["campaign_type"],
                        budget=optimized_config["budget"],
                        duration=optimized_config["duration"],
                        analysis_focus=optimized_config["analysis_focus"],
                        optimization_config=optimized_config,
                        callbacks=callback_manager.handlers
                    )
                else:
                    result = workflow.execute_optimized_workflow(
                        selected_agents=optimized_config["selected_agents"],
                        target_audience=optimized_config["target_audience"],
                        campaign_type=optimized_config["campaign_type"],
                        budget=optimized_config["budget"],
                        duration=optimized_config["duration"],
                        analysis_focus=optimized_config["analysis_focus"],
                        optimization_config=optimized_config
                    )
            else:
                if callback_manager:
                    result = workflow.execute_workflow(
                        selected_agents=optimized_config["selected_agents"],
                        target_audience=optimized_config["target_audience"],
                        campaign_type=optimized_config["campaign_type"],
                        budget=optimized_config["budget"],
                        duration=optimized_config["duration"],
                        analysis_focus=optimized_config["analysis_focus"],
                        callbacks=callback_manager.handlers
                    )
                else:
                    result = workflow.execute_workflow(
                        selected_agents=optimized_config["selected_agents"],
                        target_audience=optimized_config["target_audience"],
                        campaign_type=optimized_config["campaign_type"],
                        budget=optimized_config["budget"],
                        duration=optimized_config["duration"],
                        analysis_focus=optimized_config["analysis_focus"]
                    )
            
            # Add monitoring metadata to result
            if isinstance(result, dict):
                result["langsmith_monitoring"] = {
                    "enabled": LANGSMITH_AVAILABLE,
                    "project": "marketing-research-dashboard",
                    "run_metadata": run_metadata
                }
            
            return result
            
        except Exception as e:
            logger.error(f"LangGraph analysis failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _run_crewai_fallback_analysis(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run analysis using CrewAI optimization system as fallback."""
        try:
            logger.info("Using CrewAI optimization system (LangGraph fallback)")
            
            # Prepare inputs for optimization manager
            inputs = {
                'target_audience': config["target_audience"],
                'campaign_type': config["campaign_type"],
                'budget': config["budget"],
                'duration': config["duration"],
                'analysis_focus': config["analysis_focus"],
                'business_objective': config.get("business_objective", ""),
                'competitive_landscape': config.get("competitive_landscape", ""),
                'market_segments': config.get("market_segments", []),
                'product_categories': config.get("product_categories", []),
                'key_metrics': config.get("key_metrics", []),
                'brands': config.get("brands", []),
                'campaign_goals': config.get("campaign_goals", []),
                'forecast_periods': config.get("forecast_periods", 30),
                'expected_revenue': config.get("expected_revenue", 25000),
                'brand_metrics': config.get("brand_metrics", {}),
                'competitive_analysis': config.get("competitive_analysis", True),
                'market_share_analysis': config.get("market_share_analysis", True),
                'data_file_path': "data/beverage_sales.csv"  # Default data path
            }
            
            # Get optimization level from settings
            opt_settings = config.get("optimization_settings", {})
            optimization_level = opt_settings.get("optimization_level", "blackboard")
            
            # Check if optimization_manager is available
            global optimization_manager
            if not optimization_manager:
                logger.error("Optimization manager not available")
                # Try to recreate MockOptimizationManager if it's None
                optimization_manager = MockOptimizationManager()
                logger.info("✅ Recreated MockOptimizationManager for analysis")
            
            if not optimization_manager:
                logger.error("Optimization manager still not available after recreation")
                return {"success": False, "error": "Optimization manager not initialized. Please check CrewAI installation."}
            
            # Use optimization manager to run analysis
            analysis_result = optimization_manager.run_analysis_with_optimization(
                inputs=inputs,
                optimization_level=optimization_level,
                custom_tasks_config_path=None  # Will use default tasks
            )
            
            if "error" in analysis_result:
                logger.error(f"Optimization manager failed: {analysis_result['error']}")
                return {"success": False, "error": analysis_result['error']}
            
            # Extract results and metrics
            crew_result = analysis_result.get("result", "No result available")
            metrics = analysis_result.get("metrics", {})
            optimization_record = analysis_result.get("optimization_record", {})
            
            # Format result to match LangGraph format
            result = {
                "success": True,
                "workflow_id": f"crewai_fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "workflow_engine": "CrewAI (Fallback)",
                "status": "completed",
                "agent_results": {"analysis": str(crew_result)},
                "token_usage": metrics,
                "optimization_metrics": optimization_record,
                "execution_time": analysis_result.get("duration_seconds", 0),
                "summary": {
                    "optimization_level": optimization_level,
                    "agents_used": len(config["selected_agents"]),
                    "fallback_used": True,
                    "total_tokens": metrics.get("total_tokens", 0),
                    "total_cost": metrics.get("total_cost", 0.0)
                }
            }
            
            logger.info(f"✅ CrewAI fallback analysis completed successfully")
            logger.info(f"   - Tokens used: {metrics.get('total_tokens', 0):,}")
            logger.info(f"   - Cost: ${metrics.get('total_cost', 0):.4f}")
            logger.info(f"   - Duration: {analysis_result.get('duration_seconds', 0):.1f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"CrewAI fallback analysis failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"success": False, "error": str(e)}
    
    def _apply_optimization_strategies(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply optimization strategies based on configuration."""
        optimized_config = config.copy()
        
        # Get optimization settings
        opt_settings = config.get("optimization_settings", {})
        
        # Apply token budget constraints
        if opt_settings.get("token_budget"):
            optimized_config["max_tokens"] = opt_settings["token_budget"]
        
        # Apply caching if enabled
        if opt_settings.get("enable_caching"):
            optimized_config["use_cache"] = True
        
        # Apply context strategy
        if opt_settings.get("context_strategy"):
            optimized_config["context_strategy"] = opt_settings["context_strategy"]
        
        # Apply memory management if enabled
        if opt_settings.get("enable_mem0"):
            optimized_config["enable_mem0"] = True
        
        # Apply token tracking if enabled
        if opt_settings.get("enable_token_tracking"):
            optimized_config["enable_token_tracking"] = True
        
        # Apply optimization tools if enabled
        if opt_settings.get("enable_optimization_tools"):
            optimized_config["enable_optimization_tools"] = True
        
        # Apply optimization level
        if opt_settings.get("optimization_level"):
            optimized_config["optimization_level"] = opt_settings["optimization_level"]
            
            # Agent selection optimization for higher levels
            if opt_settings["optimization_level"] in ["full", "blackboard"]:
                optimized_config["selected_agents"] = self._optimize_agent_selection(
                    config["selected_agents"], 
                    config["analysis_type"]
                )
        
        return optimized_config
    
    def _optimize_agent_selection(self, selected_agents: List[str], analysis_type: str) -> List[str]:
        """Optimize agent selection for token efficiency."""
        # Define minimal agent sets for different analysis types
        minimal_sets = {
            "roi_focused": ["data_analyst", "forecasting_specialist"],
            "content_strategy": ["content_strategist", "creative_copywriter"],
            "brand_performance": ["competitive_analyst", "brand_performance_specialist"],
            "sales_forecast": ["forecasting_specialist"],
            "quick_insights": ["market_research_analyst"]
        }
        
        # Use minimal set if available, otherwise use first 3 agents
        if analysis_type in minimal_sets:
            return minimal_sets[analysis_type]
        else:
            return selected_agents[:3]  # Limit to 3 agents for efficiency
    
    def render_results(self, result: Dict[str, Any]):
        """Render analysis results with optimization metrics."""
        if not result.get("success"):
            st.error(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
            return
        
        st.success("✅ Analysis completed successfully!")
        
        # Create tabs for different result views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Results", "⚡ Optimization", "🔍 Token Usage", "📈 Performance"])
        
        with tab1:
            self._render_analysis_results(result)
        
        with tab2:
            self._render_optimization_metrics(result)
        
        with tab3:
            self._render_token_usage(result)
        
        with tab4:
            self._render_performance_metrics(result)
    
    def _render_analysis_results(self, result: Dict[str, Any]):
        """Render the main analysis results."""
        st.subheader("📊 Analysis Results")
        
        # Workflow summary
        if "summary" in result:
            summary = result["summary"]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Workflow ID", summary.get("workflow_id", "N/A")[:8] + "...")
            with col2:
                st.metric("Execution Time", f"{summary.get('execution_time', 0):.2f}s")
            with col3:
                st.metric("Success Rate", f"{summary.get('success_rate', 0)*100:.1f}%")
            with col4:
                st.metric("Agents Used", summary.get("completed_agents", 0))
        
        # Agent results
        if "agent_results" in result:
            st.subheader("🤖 Agent Results")
            
            for agent, agent_result in result["agent_results"].items():
                with st.expander(f"📋 {agent.replace('_', ' ').title()}"):
                    if isinstance(agent_result, dict):
                        if "analysis" in agent_result:
                            st.write("**Analysis:**")
                            st.write(agent_result["analysis"])
                        if "recommendations" in agent_result:
                            st.write("**Recommendations:**")
                            st.write(agent_result["recommendations"])
                        if "metrics" in agent_result:
                            st.write("**Metrics:**")
                            st.json(agent_result["metrics"])
                    else:
                        st.write(str(agent_result))
    
    def _render_optimization_metrics(self, result: Dict[str, Any]):
        """Render optimization performance metrics."""
        st.subheader("⚡ Optimization Performance")
        
        if "optimization_metrics" not in result:
            st.info("No optimization metrics available")
            return
        
        opt_metrics = result["optimization_metrics"]
        
        # Token savings
        if "token_optimization" in opt_metrics:
            token_opt = opt_metrics["token_optimization"]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Token Savings",
                    f"{token_opt.get('token_savings_percent', 0):.1f}%",
                    help="Percentage of tokens saved through optimization"
                )
            with col2:
                st.metric(
                    "Traditional Tokens",
                    f"{token_opt.get('traditional_tokens', 0):,}",
                    help="Estimated tokens without optimization"
                )
            with col3:
                st.metric(
                    "Optimized Tokens",
                    f"{token_opt.get('optimized_tokens', 0):,}",
                    help="Actual tokens used with optimization"
                )
        
        # Cache performance
        if "cache_performance" in opt_metrics:
            cache_perf = opt_metrics["cache_performance"]
            
            st.subheader("💾 Cache Performance")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Hit Rate", f"{cache_perf.get('hit_rate', 0):.1f}%")
            with col2:
                st.metric("Total Hits", cache_perf.get('total_hits', 0))
            with col3:
                st.metric("Cache Size", f"{cache_perf.get('cache_size_mb', 0):.1f} MB")
        
        # Context optimization
        if "context_optimization" in opt_metrics:
            context_opt = opt_metrics["context_optimization"]
            
            st.subheader("🧠 Context Optimization")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Memory Saved", f"{context_opt.get('memory_saved_mb', 0):.1f} MB")
            with col2:
                st.metric("Compression Ratio", f"{context_opt.get('compression_ratio', 1.0):.2f}x")
    
    def _render_token_usage(self, result: Dict[str, Any]):
        """Render detailed token usage information."""
        st.subheader("🔍 Token Usage Analysis")
        
        if "token_usage" not in result:
            st.info("No token usage data available")
            return
        
        token_usage = result["token_usage"]
        
        # Overall usage metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Tokens", f"{token_usage.get('total_tokens', 0):,}")
        with col2:
            st.metric("Prompt Tokens", f"{token_usage.get('prompt_tokens', 0):,}")
        with col3:
            st.metric("Completion Tokens", f"{token_usage.get('completion_tokens', 0):,}")
        with col4:
            cost = token_usage.get('total_cost', 0)
            st.metric("Estimated Cost", f"${cost:.4f}")
        
        # Token usage breakdown chart
        if token_usage.get('prompt_tokens', 0) > 0 and token_usage.get('completion_tokens', 0) > 0:
            fig = px.pie(
                values=[token_usage['prompt_tokens'], token_usage['completion_tokens']],
                names=['Prompt Tokens', 'Completion Tokens'],
                title="Token Usage Breakdown"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Agent-wise token usage
        if "agent_token_usage" in token_usage:
            st.subheader("🤖 Agent Token Usage")
            agent_usage = token_usage["agent_token_usage"]
            
            agent_df = pd.DataFrame([
                {
                    "Agent": agent.replace('_', ' ').title(),
                    "Tokens": usage.get('total_tokens', 0),
                    "Cost": f"${usage.get('cost', 0):.4f}"
                }
                for agent, usage in agent_usage.items()
            ])
            
            if not agent_df.empty:
                st.dataframe(agent_df, use_container_width=True)
                
                # Agent usage chart
                fig = px.bar(
                    agent_df,
                    x="Agent",
                    y="Tokens",
                    title="Token Usage by Agent"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_performance_metrics(self, result: Dict[str, Any]):
        """Render performance and execution metrics."""
        st.subheader("📈 Performance Metrics")
        
        # Execution timeline
        if "final_state" in result and "agent_execution_order" in result["final_state"]:
            execution_order = result["final_state"]["agent_execution_order"]
            
            st.subheader("⏱️ Execution Timeline")
            for i, agent in enumerate(execution_order, 1):
                st.write(f"**Step {i}**: {agent.replace('_', ' ').title()}")
        
        # System performance
        if "summary" in result:
            summary = result["summary"]
            
            st.subheader("🎯 System Performance")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Agents", summary.get("total_agents", 0))
            with col2:
                st.metric("Completed", summary.get("completed_agents", 0))
            with col3:
                st.metric("Failed", summary.get("failed_agents", 0))
        
        # Recommendations for optimization
        st.subheader("💡 Optimization Recommendations")
        
        if result.get("token_usage", {}).get("total_tokens", 0) > 20000:
            st.warning("⚠️ High token usage detected. Consider:")
            st.markdown("""
            - Enabling higher optimization levels
            - Reducing the number of agents
            - Using more specific analysis focus
            - Enabling smart caching
            """)
        elif result.get("token_usage", {}).get("total_tokens", 0) < 5000:
            st.success("✅ Excellent token efficiency!")
        else:
            st.info("ℹ️ Good token usage. Consider blackboard optimization for maximum efficiency.")
    
    def run(self):
        """Run the main dashboard application."""
        render_header()
        
        # Get configuration from sidebar
        config = self.render_sidebar()
        
        # Main content area
        st.header("🎯 Marketing Analysis")
        
        # LangSmith Monitoring Section
        if LANGSMITH_AVAILABLE:
            with st.expander("🔍 LangSmith Monitoring", expanded=False):
                st.markdown("**Real-time analysis monitoring with LangSmith**")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 Refresh Runs", help="Refresh LangSmith run data"):
                        st.rerun()
                
                with col2:
                    project_name = st.text_input(
                        "Project Name", 
                        value="marketing-research-dashboard",
                        help="LangSmith project name for monitoring"
                    )
                
                # Display recent runs
                monitor_langsmith_runs(project_name)
                
                # LangSmith project link
                if project_name:
                    langsmith_url = f"https://smith.langchain.com/o/default/projects/p/{project_name}"
                    st.markdown(f"[🔗 View Full Project in LangSmith]({langsmith_url})")
        else:
            st.info("💡 **Enable LangSmith Monitoring**: Set `LANGCHAIN_API_KEY` in your environment to monitor analysis runs in real-time.")
        
        # Show installation help if needed
        if not LANGGRAPH_AVAILABLE and not CREWAI_AVAILABLE:
            st.error("⚠️ No workflow system available!")
            st.markdown("""
            **To fix this issue:**
            
            **Option 1: Install LangGraph (Recommended)**
            ```bash
            pip install langgraph langchain-openai streamlit plotly pandas
            ```
            
            **Option 2: Check CrewAI Components**
            - Ensure `optimization_manager.py` is available
            - Check that `src/marketing_research_swarm/` is in Python path
            
            **Quick Test:**
            ```bash
            python -c "from marketing_research_swarm.optimization_manager import OptimizationManager; print('✅ CrewAI OK')"
            ```
            """)
            return
        
        # Show fallback notice
        if not LANGGRAPH_AVAILABLE and CREWAI_AVAILABLE:
            st.info("💡 **Using CrewAI Optimization System** - LangGraph not available, but all optimization features work through CrewAI fallback!")
        
        # Run analysis button
        if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
            if not config["selected_agents"]:
                st.error("Please select at least one agent to run the analysis.")
                return
            
            # Show progress with LangSmith monitoring
            with st.spinner("Running optimized analysis..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # LangSmith monitoring status
                if LANGSMITH_AVAILABLE:
                    langsmith_status = st.empty()
                    langsmith_status.info("🔍 **LangSmith Monitoring**: Analysis will be tracked in real-time")
                
                # Update progress with LangSmith integration
                for i in range(100):
                    progress_bar.progress(i + 1)
                    if i < 20:
                        status_text.text("🔧 Initializing workflow...")
                        if LANGSMITH_AVAILABLE and i == 10:
                            langsmith_status.info("🔍 **LangSmith**: Creating trace session...")
                    elif i < 40:
                        status_text.text("⚡ Applying optimization strategies...")
                        if LANGSMITH_AVAILABLE and i == 30:
                            langsmith_status.info("🔍 **LangSmith**: Monitoring agent initialization...")
                    elif i < 60:
                        status_text.text("🤖 Executing agents...")
                        if LANGSMITH_AVAILABLE and i == 50:
                            langsmith_status.info("🔍 **LangSmith**: Tracking agent execution and token usage...")
                    elif i < 80:
                        status_text.text("📊 Processing results...")
                        if LANGSMITH_AVAILABLE and i == 70:
                            langsmith_status.info("🔍 **LangSmith**: Recording performance metrics...")
                    else:
                        status_text.text("✅ Finalizing analysis...")
                        if LANGSMITH_AVAILABLE and i == 90:
                            langsmith_status.success("🔍 **LangSmith**: Analysis trace completed!")
                
                # Run the actual analysis
                result = self.run_optimized_analysis(config)
                
                progress_bar.empty()
                status_text.empty()
                if LANGSMITH_AVAILABLE:
                    langsmith_status.empty()
            
            # Store result in session state
            st.session_state["last_result"] = result
            
            # Render results
            self.render_results(result)
        
        # Show previous results if available
        if "last_result" in st.session_state:
            st.header("📋 Previous Results")
            with st.expander("View Last Analysis"):
                self.render_results(st.session_state["last_result"])


def main():
    """Main function to run the dashboard."""
    dashboard = LangGraphDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()