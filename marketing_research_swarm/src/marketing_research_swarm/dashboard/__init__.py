"""
Dashboard components for the LangGraph Marketing Research Dashboard.
"""

from .monitoring.enhanced_token_tracker import EnhancedTokenTracker
from .monitoring.enhanced_langsmith_monitor import EnhancedLangSmithMonitor, create_langsmith_tracer
from .visualization.state_graph_visualizer import StateGraphVisualizer
from .mock_optimization_manager import MockOptimizationManager
from .components.rag_document_monitor import RAGDocumentMonitor
from .utils.dashboard_utils import monitor_langsmith_runs

# Utility functions
def load_agents_config():
    """Load available agents from agents.yaml"""
    import yaml
    import streamlit as st
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

def create_custom_task_config(selected_agents, task_params):
    """Create a custom task configuration YAML file"""
    import yaml
    import os
    import uuid
    from datetime import datetime
    
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
        'data_analyst': {
            'description': f"""Perform comprehensive data analysis on the sales data from {{data_file_path}} focusing on {', '.join(task_params.get('key_metrics', []))}. 
            Analyze {', '.join(task_params.get('product_categories', []))} categories across {', '.join(task_params.get('market_segments', []))} regions.
            Generate forecasts for {task_params.get('forecast_periods', 30)} periods with expected revenue of ${task_params.get('expected_revenue', 25000):,}.
            Include competitive analysis: {task_params.get('competitive_analysis', True)} and market share analysis: {task_params.get('market_share_analysis', True)}.""",
            'expected_output': f"A detailed data analysis report with forecasts, trend analysis, performance metrics for {', '.join(task_params.get('key_metrics', []))}, and recommendations for optimization."
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
    
    return config_filename

def initialize_components():
    """Initialize all dashboard components with fallback support."""
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize enhanced components
        enhanced_token_tracker = EnhancedTokenTracker()
        enhanced_langsmith_monitor = EnhancedLangSmithMonitor()
        state_graph_visualizer = StateGraphVisualizer()
        
        logger.info("✅ Enhanced dashboard components initialized")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        logger.info("💡 Some features may be limited without full dependencies")
        return False

def render_header():
    """Render the dashboard header."""
    import streamlit as st
    
    st.title("🚀 LangGraph Marketing Research Dashboard")
    st.markdown("**Advanced workflow orchestration with intelligent token optimization**")
    
    # System status indicators would go here
    # This is a simplified version - the full implementation is in the main dashboard

__all__ = [
    'EnhancedTokenTracker',
    'EnhancedLangSmithMonitor', 
    'StateGraphVisualizer',
    'MockOptimizationManager',
    'RAGDocumentMonitor',
    'create_langsmith_tracer',
    'monitor_langsmith_runs',
    'load_agents_config',
    'create_custom_task_config',
    'initialize_components',
    'render_header'
]