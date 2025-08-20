"""
Dashboard Utility Functions
"""

import os
import yaml
import uuid
import logging
from datetime import datetime
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


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


def load_agents_config():
    """Load available agents from agents.yaml"""
    try:
        agents_path = 'src/marketing_research_swarm/config/agents.yaml'
        with open(agents_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logger.error(f"Agents configuration file not found at {agents_path}")
        return {}
    except Exception as e:
        logger.error(f"Error loading agents configuration: {e}")
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
            Focus on {task_params.get('target_audience', 'general market trends')} and provide detailed insights on market dynamics, consumer behavior patterns, and competitive landscape analysis.""",
            'expected_output': f"A comprehensive market research report covering market size, growth trends, consumer segments, competitive analysis, and strategic opportunities for {task_params.get('analysis_focus', 'the beverage industry')}"
        },
        'competitive_analyst': {
            'description': f"""Analyze competitive landscape and positioning for {task_params.get('analysis_focus', 'beverage brands')} using the sales data from {{data_file_path}}. 
            Compare market share, pricing strategies, and performance metrics across major competitors with focus on {task_params.get('target_audience', 'market positioning')}.""",
            'expected_output': f"A detailed competitive analysis report with market share analysis, competitor positioning, pricing strategies, and competitive advantages for {task_params.get('analysis_focus', 'beverage brands')}"
        },
        'data_analyst': {
            'description': f"""Perform advanced data analysis on the sales dataset from {{data_file_path}} focusing on {task_params.get('analysis_focus', 'performance metrics')}. 
            Calculate key performance indicators, identify trends, and provide statistical insights relevant to {task_params.get('target_audience', 'business stakeholders')}.""",
            'expected_output': f"A comprehensive data analysis report with statistical insights, trend analysis, KPI calculations, and data-driven recommendations for {task_params.get('analysis_focus', 'business performance')}"
        },
        'forecasting_specialist': {
            'description': f"""Develop sales forecasts and predictive models for {task_params.get('analysis_focus', 'beverage sales')} using historical data from {{data_file_path}}. 
            Create forecasts for {task_params.get('forecast_periods', '12')} periods ahead with confidence intervals and scenario planning for {task_params.get('target_audience', 'strategic planning')}.""",
            'expected_output': f"A detailed sales forecast report with predictive models, confidence intervals, scenario planning, and strategic recommendations for {task_params.get('forecast_periods', '12')} periods ahead"
        },
        'brand_performance_specialist': {
            'description': f"""Analyze brand performance metrics and market positioning for {task_params.get('analysis_focus', 'beverage brands')} using data from {{data_file_path}}. 
            Evaluate brand strength, market share evolution, and performance indicators with insights for {task_params.get('target_audience', 'brand managers')}.""",
            'expected_output': f"A comprehensive brand performance analysis with market share trends, brand strength metrics, positioning analysis, and strategic recommendations for {task_params.get('analysis_focus', 'brand optimization')}"
        },
        'campaign_optimizer': {
            'description': f"""Optimize marketing campaigns and budget allocation for {task_params.get('analysis_focus', 'beverage marketing')} based on performance data from {{data_file_path}}. 
            Analyze ROI, channel effectiveness, and recommend optimization strategies for {task_params.get('target_audience', 'marketing teams')}.""",
            'expected_output': f"A detailed campaign optimization report with ROI analysis, budget allocation recommendations, channel performance insights, and optimization strategies for {task_params.get('analysis_focus', 'marketing campaigns')}"
        },
        'content_strategist': {
            'description': f"""Develop content strategy and messaging frameworks for {task_params.get('analysis_focus', 'beverage marketing')} based on market insights from {{data_file_path}}. 
            Create content recommendations targeting {task_params.get('target_audience', 'target demographics')} with focus on engagement and conversion optimization.""",
            'expected_output': f"A comprehensive content strategy with messaging frameworks, content recommendations, audience targeting strategies, and engagement optimization tactics for {task_params.get('target_audience', 'target demographics')}"
        },
        'creative_copywriter': {
            'description': f"""Create compelling marketing copy and creative content for {task_params.get('analysis_focus', 'beverage campaigns')} based on strategic insights from {{data_file_path}}. 
            Develop copy that resonates with {task_params.get('target_audience', 'target audience')} and drives engagement and conversions.""",
            'expected_output': f"Creative marketing copy and content pieces including headlines, taglines, campaign concepts, and messaging that effectively targets {task_params.get('target_audience', 'the intended audience')} for {task_params.get('analysis_focus', 'beverage marketing')}"
        }
    }
    
    # Create tasks for selected agents
    for agent in selected_agents:
        if agent in agent_task_mapping:
            task_config = agent_task_mapping[agent].copy()
            tasks_config[f"{agent}_task"] = task_config
    
    # Create the YAML content
    yaml_content = yaml.dump(tasks_config, default_flow_style=False, sort_keys=False)
    
    # Save to file
    filename = f"tasks_custom_{timestamp}_{task_id}.yaml"
    filepath = f"src/marketing_research_swarm/config/{filename}"
    
    try:
        with open(filepath, 'w') as file:
            file.write(yaml_content)
        logger.info(f"✅ Created custom task configuration: {filename}")
        return filename
    except Exception as e:
        logger.error(f"❌ Error creating task configuration: {e}")
        return ""


def initialize_components():
    """Initialize dashboard components and return their status"""
    components_status = {
        'token_tracker': False,
        'langsmith_monitor': False,
        'state_visualizer': False,
        'rag_monitor': False
    }
    
    try:
        # Import and initialize components
        from ..monitoring.token_tracker import EnhancedTokenTracker
        from ..monitoring.langsmith_monitor import EnhancedLangSmithMonitor
        from ..visualization.state_graph_visualizer import StateGraphVisualizer
        from ..components.rag_document_monitor import RAGDocumentMonitor
        
        components_status['token_tracker'] = True
        components_status['langsmith_monitor'] = True
        components_status['state_visualizer'] = True
        components_status['rag_monitor'] = True
        
        logger.info("✅ All dashboard components initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Error initializing components: {e}")
    
    return components_status


def get_langsmith_run_url(run_id: str) -> str:
    """Generate LangSmith run URL"""
    return f"https://smith.langchain.com/public/{run_id}/r"


def create_langsmith_tracer(project_name: str = "marketing-research-dashboard"):
    """Create LangSmith tracer with callback manager"""
    try:
        from langchain.callbacks import CallbackManager
        from langchain.callbacks.tracers import LangChainTracer
        
        tracer = LangChainTracer(project_name=project_name)
        callback_manager = CallbackManager([tracer])
        return callback_manager
    except Exception as e:
        logger.error(f"❌ Error creating LangSmith tracer: {e}")
        return None


def monitor_langsmith_runs(project_name: str = "marketing-research-dashboard"):
    """Monitor LangSmith runs and return recent activity"""
    try:
        from langsmith import Client
        client = Client()
        
        runs = list(client.list_runs(
            project_name=project_name,
            limit=10
        ))
        
        return [
            {
                'id': str(run.id),
                'name': run.name or 'Unknown',
                'status': run.status or 'unknown',
                'start_time': run.start_time,
                'end_time': run.end_time,
                'url': get_langsmith_run_url(str(run.id))
            }
            for run in runs
        ]
    except Exception as e:
        logger.error(f"❌ Error monitoring LangSmith runs: {e}")
        return []


def _compute_context_quality_summary(metrics: List[str] = None):
    """Compute context quality summary metrics"""
    if metrics is None:
        metrics = ["relevance", "completeness", "accuracy", "timeliness"]
    
    # Mock implementation - in real scenario this would analyze actual context
    import random
    
    quality_scores = {}
    for metric in metrics:
        quality_scores[metric] = random.uniform(0.7, 0.95)
    
    overall_score = sum(quality_scores.values()) / len(quality_scores)
    
    return {
        'overall_score': overall_score,
        'individual_scores': quality_scores,
        'recommendation': 'Good' if overall_score > 0.8 else 'Needs Improvement'
    }


def render_header():
    """Render the dashboard header"""
    return """
    <div class="main-header">
        📊 LangGraph Marketing Research Dashboard
    </div>
    <div style="text-align: center; margin-bottom: 2rem; color: #7f8c8d;">
        Advanced AI-Powered Marketing Analytics with Real-time Monitoring
    </div>
    """