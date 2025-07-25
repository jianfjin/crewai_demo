"""
LangGraph Agent Nodes for Marketing Research Workflow

This module defines the agent nodes that replace CrewAI agents with LangGraph nodes.
Each agent is implemented as a LangGraph node function that processes the state.
"""

import yaml
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, SystemMessage
import requests

from .state import MarketingResearchState, AgentStatus, store_agent_result, store_agent_error, get_agent_context
from ..tools.advanced_tools import (
    calculate_roi, analyze_kpis, forecast_sales, plan_budget,
    analyze_brand_performance, calculate_market_share, time_series_analysis,
    cross_sectional_analysis, beverage_market_analysis, profitability_analysis
)

logger = logging.getLogger(__name__)


class LangGraphAgent:
    """Base class for LangGraph agent nodes."""
    
    def __init__(self, role: str, goal: str, backstory: str, tools: List[str], llm_config: str = "openai/gpt-4o-mini"):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.tools = tools
        self.llm_config = llm_config
        self.llm = self._initialize_llm()
        
        # Initialize tools
        self.available_tools = {
            "calculate_roi": calculate_roi,
            "analyze_kpis": analyze_kpis,
            "forecast_sales": forecast_sales,
            "plan_budget": plan_budget,
            "analyze_brand_performance": analyze_brand_performance,
            "calculate_market_share": calculate_market_share,
            "time_series_analysis": time_series_analysis,
            "cross_sectional_analysis": cross_sectional_analysis,
            "beverage_market_analysis": beverage_market_analysis,
            "profitability_analysis": profitability_analysis,
        }
        
    def _initialize_llm(self):
        """Initialize the LLM for the agent."""
        try:
            if 'OPENAI_API_KEY' in os.environ and os.environ['OPENAI_API_KEY'] != 'your_openai_api_key_here':
                return ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
            else:
                # Try Ollama as fallback
                try:
                    response = requests.get("http://localhost:11434/api/health", timeout=5)
                    response.raise_for_status()
                    return ChatOllama(model="gemma", base_url="http://localhost:11434")
                except requests.exceptions.RequestException:
                    logger.warning(f"No LLM configured for agent {self.role}")
                    return None
        except Exception as e:
            logger.error(f"Failed to initialize LLM for {self.role}: {e}")
            return None
    
    def execute_task(self, state: MarketingResearchState, task_description: str) -> Dict[str, Any]:
        """Execute the agent's task with the given state and task description."""
        if not self.llm:
            raise Exception(f"No LLM available for agent {self.role}")
        
        # Get context for this agent
        context = get_agent_context(state, self.role)
        
        # Create system message with agent's role and context
        system_message = SystemMessage(content=f"""
You are a {self.role} with the following background:
{self.backstory}

Your goal is: {self.goal}

You have access to the following tools: {', '.join(self.tools)}

Current workflow context:
- Target Audience: {context.get('target_audience', 'Not specified')}
- Campaign Type: {context.get('campaign_type', 'Not specified')}
- Budget: ${context.get('budget', 0):,}
- Duration: {context.get('duration', 'Not specified')}
- Analysis Focus: {context.get('analysis_focus', 'Not specified')}
- Business Objective: {context.get('business_objective', 'Not specified')}

Available data from previous agents:
{self._format_previous_results(context)}

Please provide a comprehensive analysis following the task description.
Format your response as a structured analysis with clear sections and actionable insights.
""")
        
        # Create human message with task
        human_message = HumanMessage(content=task_description)
        
        try:
            # Execute the LLM call
            response = self.llm.invoke([system_message, human_message])
            
            # Process the response and extract structured data
            result = {
                'agent_role': self.role,
                'task_description': task_description,
                'analysis': response.content,
                'timestamp': datetime.now().isoformat(),
                'context_used': list(context.keys()),
                'tools_available': self.tools
            }
            
            # Try to execute relevant tools if mentioned in the response
            tool_results = self._execute_relevant_tools(context, response.content)
            if tool_results:
                result['tool_results'] = tool_results
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing task for {self.role}: {e}")
            raise
    
    def _format_previous_results(self, context: Dict[str, Any]) -> str:
        """Format previous agent results for context."""
        previous_results = []
        for key, value in context.items():
            if key.endswith('_results') and value:
                agent_name = key.replace('_results', '').replace('_', ' ').title()
                previous_results.append(f"- {agent_name}: Available")
        
        if not previous_results:
            return "No previous results available."
        
        return "\n".join(previous_results)
    
    def _execute_relevant_tools(self, context: Dict[str, Any], response_content: str) -> Dict[str, Any]:
        """Execute tools that are relevant to the agent's analysis."""
        tool_results = {}
        
        # Simple tool execution based on agent role and available tools
        try:
            if self.role == 'data_analyst' and 'profitability_analysis' in self.tools:
                # Execute profitability analysis
                data_file = context.get('data_file_path', 'data/beverage_sales_data.csv')
                if os.path.exists(data_file):
                    result = profitability_analysis.invoke({
                        'data_file_path': data_file,
                        'analysis_dimension': 'brand'
                    })
                    tool_results['profitability_analysis'] = result
            
            elif self.role == 'market_research_analyst' and 'beverage_market_analysis' in self.tools:
                # Execute market analysis
                data_file = context.get('data_file_path', 'data/beverage_sales_data.csv')
                if os.path.exists(data_file):
                    result = beverage_market_analysis.invoke({
                        'data_file_path': data_file
                    })
                    tool_results['beverage_market_analysis'] = result
            
            elif self.role == 'forecasting_specialist' and 'forecast_sales' in self.tools:
                # Execute sales forecasting
                data_file = context.get('data_file_path', 'data/beverage_sales_data.csv')
                if os.path.exists(data_file):
                    result = forecast_sales.invoke({
                        'data_file_path': data_file,
                        'forecast_periods': context.get('forecast_periods', 30)
                    })
                    tool_results['forecast_sales'] = result
            
            elif self.role == 'campaign_optimizer' and 'calculate_roi' in self.tools:
                # Execute ROI calculation
                result = calculate_roi.invoke({
                    'investment': context.get('budget', 250000),
                    'revenue': context.get('expected_revenue', 25000),
                    'time_period': context.get('duration', '12 months')
                })
                tool_results['calculate_roi'] = result
                
        except Exception as e:
            logger.warning(f"Tool execution error for {self.role}: {e}")
        
        return tool_results


def load_agent_configs(config_path: str) -> Dict[str, Dict[str, Any]]:
    """Load agent configurations from YAML file."""
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Failed to load agent configs from {config_path}: {e}")
        return {}


def create_agent_from_config(agent_name: str, agent_config: Dict[str, Any]) -> LangGraphAgent:
    """Create a LangGraph agent from configuration."""
    return LangGraphAgent(
        role=agent_config.get('role', agent_name),
        goal=agent_config.get('goal', ''),
        backstory=agent_config.get('backstory', ''),
        tools=agent_config.get('tools', []),
        llm_config=agent_config.get('llm', 'openai/gpt-4o-mini')
    )


# Agent node functions for LangGraph
def market_research_analyst_node(state: MarketingResearchState) -> MarketingResearchState:
    """Market Research Analyst node."""
    agent_role = 'market_research_analyst'
    
    try:
        # Load agent config
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'agents.yaml')
        agent_configs = load_agent_configs(config_path)
        agent_config = agent_configs.get(agent_role, {})
        
        # Create agent
        agent = create_agent_from_config(agent_role, agent_config)
        
        # Define task
        task_description = f"""
        Conduct comprehensive market research on the beverage industry using the enhanced sales data.
        Focus on identifying opportunities for {state['target_audience']} and provide insights on:
        - Market structure and dynamics
        - Seasonal patterns and trends
        - Pricing strategies and profit margins
        - Regional performance analysis
        - Competitive landscape overview
        - Strategic recommendations for the global beverage market
        
        Target audience: {state['target_audience']}
        Market segments: {', '.join(state['market_segments'])}
        Product categories: {', '.join(state['product_categories'])}
        """
        
        # Execute task
        result = agent.execute_task(state, task_description)
        
        # Store result in state
        state = store_agent_result(state, agent_role, result)
        
    except Exception as e:
        logger.error(f"Error in {agent_role} node: {e}")
        state = store_agent_error(state, agent_role, str(e))
    
    return state


def competitive_analyst_node(state: MarketingResearchState) -> MarketingResearchState:
    """Competitive Analyst node."""
    agent_role = 'competitive_analyst'
    
    try:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'agents.yaml')
        agent_configs = load_agent_configs(config_path)
        agent_config = agent_configs.get(agent_role, {})
        
        agent = create_agent_from_config(agent_role, agent_config)
        
        task_description = f"""
        Analyze competitive landscape, market positioning, and brand dynamics to provide competitive intelligence.
        Focus on:
        - Competitive positioning analysis
        - Brand market share evaluation
        - Competitive threats and opportunities
        - Market structure analysis
        - Strategic competitive insights
        
        Competitive landscape: {state['competitive_landscape']}
        Key brands: {', '.join(state['brands'])}
        Market segments: {', '.join(state['market_segments'])}
        """
        
        result = agent.execute_task(state, task_description)
        state = store_agent_result(state, agent_role, result)
        
    except Exception as e:
        logger.error(f"Error in {agent_role} node: {e}")
        state = store_agent_error(state, agent_role, str(e))
    
    return state


def data_analyst_node(state: MarketingResearchState) -> MarketingResearchState:
    """Data Analyst node."""
    agent_role = 'data_analyst'
    
    try:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'agents.yaml')
        agent_configs = load_agent_configs(config_path)
        agent_config = agent_configs.get(agent_role, {})
        
        agent = create_agent_from_config(agent_role, agent_config)
        
        task_description = f"""
        Perform comprehensive data analysis on beverage sales to uncover profitability insights, trends, and performance patterns.
        Focus on:
        - Profitability analysis across brands, categories, and regions
        - Time series analysis for trend identification
        - Cross-sectional analysis for performance comparison
        - KPI analysis and performance metrics
        - Statistical insights and data interpretation
        
        Key metrics: {', '.join(state['key_metrics'])}
        Product categories: {', '.join(state['product_categories'])}
        Market segments: {', '.join(state['market_segments'])}
        """
        
        result = agent.execute_task(state, task_description)
        state = store_agent_result(state, agent_role, result)
        
    except Exception as e:
        logger.error(f"Error in {agent_role} node: {e}")
        state = store_agent_error(state, agent_role, str(e))
    
    return state


def content_strategist_node(state: MarketingResearchState) -> MarketingResearchState:
    """Content Strategist node."""
    agent_role = 'content_strategist'
    
    try:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'agents.yaml')
        agent_configs = load_agent_configs(config_path)
        agent_config = agent_configs.get(agent_role, {})
        
        agent = create_agent_from_config(agent_role, agent_config)
        
        task_description = f"""
        Develop a comprehensive content strategy for beverage brands based on market research insights.
        Focus on:
        - Channel-specific content recommendations
        - Seasonal campaign strategies
        - Brand positioning strategies
        - Social media campaign ideas
        - Influencer collaboration strategies
        - Regional content adaptation
        
        Target audience: {state['target_audience']}
        Campaign type: {state['campaign_type']}
        Campaign goals: {', '.join(state['campaign_goals'])}
        """
        
        result = agent.execute_task(state, task_description)
        state = store_agent_result(state, agent_role, result)
        
    except Exception as e:
        logger.error(f"Error in {agent_role} node: {e}")
        state = store_agent_error(state, agent_role, str(e))
    
    return state


def creative_copywriter_node(state: MarketingResearchState) -> MarketingResearchState:
    """Creative Copywriter node."""
    agent_role = 'creative_copywriter'
    
    try:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'agents.yaml')
        agent_configs = load_agent_configs(config_path)
        agent_config = agent_configs.get(agent_role, {})
        
        agent = create_agent_from_config(agent_role, agent_config)
        
        task_description = f"""
        Create compelling marketing copy for beverage campaigns targeting {state['target_audience']}.
        Develop:
        - Social media posts and content
        - Product descriptions and messaging
        - Campaign taglines and slogans
        - Promotional content
        - Brand voice and messaging guidelines
        - Regional content variations
        
        Target audience: {state['target_audience']}
        Key brands: {', '.join(state['brands'])}
        Product categories: {', '.join(state['product_categories'])}
        """
        
        result = agent.execute_task(state, task_description)
        state = store_agent_result(state, agent_role, result)
        
    except Exception as e:
        logger.error(f"Error in {agent_role} node: {e}")
        state = store_agent_error(state, agent_role, str(e))
    
    return state


def campaign_optimizer_node(state: MarketingResearchState) -> MarketingResearchState:
    """Campaign Optimizer node."""
    agent_role = 'campaign_optimizer'
    
    try:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'agents.yaml')
        agent_configs = load_agent_configs(config_path)
        agent_config = agent_configs.get(agent_role, {})
        
        agent = create_agent_from_config(agent_role, agent_config)
        
        task_description = f"""
        Develop comprehensive optimization strategies for beverage marketing campaigns.
        Focus on:
        - Budget allocation recommendations across channels and regions
        - ROI projections and optimization strategies
        - Channel strategy optimization
        - Regional focus strategies
        - Performance improvement recommendations
        - Cost optimization suggestions
        
        Budget: ${state['budget']:,}
        Duration: {state['duration']}
        Campaign goals: {', '.join(state['campaign_goals'])}
        """
        
        result = agent.execute_task(state, task_description)
        state = store_agent_result(state, agent_role, result)
        
    except Exception as e:
        logger.error(f"Error in {agent_role} node: {e}")
        state = store_agent_error(state, agent_role, str(e))
    
    return state


def brand_performance_specialist_node(state: MarketingResearchState) -> MarketingResearchState:
    """Brand Performance Specialist node."""
    agent_role = 'brand_performance_specialist'
    
    try:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'agents.yaml')
        agent_configs = load_agent_configs(config_path)
        agent_config = agent_configs.get(agent_role, {})
        
        agent = create_agent_from_config(agent_role, agent_config)
        
        task_description = f"""
        Analyze brand performance in the beverage market using comprehensive sales data.
        Focus on:
        - Brand positioning and market share analysis
        - Competitive landscape evaluation
        - Brand health metrics and performance indicators
        - Growth opportunities identification
        - Strategic brand recommendations
        - Market penetration analysis
        
        Key brands: {', '.join(state['brands'])}
        Market segments: {', '.join(state['market_segments'])}
        Product categories: {', '.join(state['product_categories'])}
        """
        
        result = agent.execute_task(state, task_description)
        state = store_agent_result(state, agent_role, result)
        
    except Exception as e:
        logger.error(f"Error in {agent_role} node: {e}")
        state = store_agent_error(state, agent_role, str(e))
    
    return state


def forecasting_specialist_node(state: MarketingResearchState) -> MarketingResearchState:
    """Forecasting Specialist node."""
    agent_role = 'forecasting_specialist'
    
    try:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'agents.yaml')
        agent_configs = load_agent_configs(config_path)
        agent_config = agent_configs.get(agent_role, {})
        
        agent = create_agent_from_config(agent_role, agent_config)
        
        task_description = f"""
        Generate accurate sales forecasts and predictive models based on market analysis insights.
        Focus on:
        - Sales forecasting with confidence intervals
        - Predictive modeling and trend analysis
        - Scenario planning and forecasting
        - Market trend predictions
        - Revenue projections
        - Risk assessment and forecasting accuracy
        
        Forecast periods: {state.get('forecast_periods', 30)} days
        Market segments: {', '.join(state['market_segments'])}
        Product categories: {', '.join(state['product_categories'])}
        """
        
        result = agent.execute_task(state, task_description)
        state = store_agent_result(state, agent_role, result)
        
    except Exception as e:
        logger.error(f"Error in {agent_role} node: {e}")
        state = store_agent_error(state, agent_role, str(e))
    
    return state


# Agent node mapping for easy access
AGENT_NODES = {
    'market_research_analyst': market_research_analyst_node,
    'competitive_analyst': competitive_analyst_node,
    'data_analyst': data_analyst_node,
    'content_strategist': content_strategist_node,
    'creative_copywriter': creative_copywriter_node,
    'campaign_optimizer': campaign_optimizer_node,
    'brand_performance_specialist': brand_performance_specialist_node,
    'forecasting_specialist': forecasting_specialist_node,
}