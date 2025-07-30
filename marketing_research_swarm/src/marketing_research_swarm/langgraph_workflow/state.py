"""
LangGraph State Management for Marketing Research Workflow

This module defines the state structure and management for the LangGraph-based
marketing research workflow, replacing the CrewAI implementation.
"""

from typing import Dict, Any, List, Optional, TypedDict, Annotated
from datetime import datetime
import uuid
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentStatus(Enum):
    """Individual agent execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class MarketingResearchState(TypedDict):
    """
    Main state structure for the marketing research workflow.
    This replaces the CrewAI crew state with LangGraph state management.
    """
    # Workflow metadata
    workflow_id: str
    workflow_type: str
    status: WorkflowStatus
    created_at: datetime
    updated_at: datetime
    
    # Input data
    initial_inputs: Dict[str, Any]
    target_audience: str
    campaign_type: str
    budget: float
    duration: str
    analysis_focus: str
    business_objective: str
    competitive_landscape: str
    market_segments: List[str]
    product_categories: List[str]
    key_metrics: List[str]
    brands: List[str]
    campaign_goals: List[str]
    
    # Agent selection and configuration
    selected_agents: List[str]
    agent_configs: Dict[str, Dict[str, Any]]
    optimization_level: str
    
    # Agent execution tracking
    agent_execution_order: List[str]
    current_agent: Optional[str]
    agent_status: Dict[str, AgentStatus]
    agent_results: Dict[str, Dict[str, Any]]
    agent_errors: Dict[str, str]
    agent_token_usage: Dict[str, Dict[str, int]]
    
    # Shared data and context
    shared_data: Dict[str, Any]
    shared_context: Dict[str, Any]
    cached_results: Dict[str, Any]
    
    # Optional completion tracking
    completed_at: Optional[datetime]
    final_summary: Optional[Dict[str, Any]]
    
    # Geographic and temporal data
    regions: List[str]
    data_file_path: str
    forecast_periods: int
    
    # Analysis results
    market_research_results: Optional[Dict[str, Any]]
    competitive_analysis_results: Optional[Dict[str, Any]]
    data_analysis_results: Optional[Dict[str, Any]]
    content_strategy_results: Optional[Dict[str, Any]]
    copywriting_results: Optional[Dict[str, Any]]
    campaign_optimization_results: Optional[Dict[str, Any]]
    brand_performance_results: Optional[Dict[str, Any]]
    forecasting_results: Optional[Dict[str, Any]]
    
    # Final outputs
    final_report: Optional[Dict[str, Any]]
    recommendations: Optional[List[str]]
    next_steps: Optional[List[str]]
    
    # Error handling
    errors: List[str]
    warnings: List[str]
    
    # Performance metrics
    total_token_usage: Dict[str, int]
    execution_time: Optional[float]
    cache_hits: int
    cache_misses: int


def create_initial_state(
    workflow_type: str,
    selected_agents: List[str],
    inputs: Dict[str, Any]
) -> MarketingResearchState:
    """
    Create initial state for a marketing research workflow.
    
    Args:
        workflow_type: Type of workflow to execute
        selected_agents: List of agent roles to include
        inputs: Input parameters for the workflow
        
    Returns:
        Initial state dictionary
    """
    workflow_id = str(uuid.uuid4())
    now = datetime.now()
    
    # Initialize agent statuses
    agent_status = {agent: AgentStatus.PENDING for agent in selected_agents}
    
    # Extract input parameters with defaults
    target_audience = inputs.get('target_audience', 'health-conscious millennials and premium beverage consumers')
    campaign_type = inputs.get('campaign_type', 'multi-channel global marketing campaign')
    budget = inputs.get('budget', 250000)
    duration = inputs.get('duration', '12 months')
    analysis_focus = inputs.get('analysis_focus', 'global beverage market performance and brand optimization')
    business_objective = inputs.get('business_objective', 'Optimize beverage portfolio performance across global markets')
    competitive_landscape = inputs.get('competitive_landscape', 'global beverage market with diverse categories')
    market_segments = inputs.get('market_segments', ['North America', 'Europe', 'Asia Pacific'])
    product_categories = inputs.get('product_categories', ['Cola', 'Juice', 'Energy', 'Sports'])
    key_metrics = inputs.get('key_metrics', ['brand_performance', 'category_trends', 'profitability_analysis'])
    brands = inputs.get('brands', ['Coca-Cola', 'Pepsi', 'Red Bull'])
    campaign_goals = inputs.get('campaign_goals', [
        'Optimize brand portfolio performance across global markets',
        'Identify high-margin opportunities by category and region',
        'Develop pricing strategies based on profitability analysis'
    ])
    optimization_level = inputs.get('optimization_level', 'partial')
    
    return MarketingResearchState(
        # Workflow metadata
        workflow_id=workflow_id,
        workflow_type=workflow_type,
        status=WorkflowStatus.PENDING,
        created_at=now,
        updated_at=now,
        
        # Input data
        initial_inputs=inputs,
        target_audience=target_audience,
        campaign_type=campaign_type,
        budget=budget,
        duration=duration,
        analysis_focus=analysis_focus,
        business_objective=business_objective,
        competitive_landscape=competitive_landscape,
        market_segments=market_segments,
        product_categories=product_categories,
        key_metrics=key_metrics,
        brands=brands,
        campaign_goals=campaign_goals,
        
        # Agent configuration
        selected_agents=selected_agents,
        agent_configs={},
        optimization_level=optimization_level,
        
        # Agent execution tracking
        agent_execution_order=[],
        current_agent=None,
        agent_status=agent_status,
        agent_results={},
        agent_errors={},
        agent_token_usage={},
        
        # Shared data and context
        shared_data={},
        shared_context={},
        cached_results={},
        
        # Optional completion tracking
        completed_at=None,
        final_summary=None,
        
        # Geographic and temporal data
        regions=inputs.get('regions', ['North', 'South', 'East', 'West']),
        data_file_path=inputs.get('data_file_path', 'data/beverage_sales_data.csv'),
        forecast_periods=inputs.get('forecast_periods', 30),
        
        # Analysis results (all start as None)
        market_research_results=None,
        competitive_analysis_results=None,
        data_analysis_results=None,
        content_strategy_results=None,
        copywriting_results=None,
        campaign_optimization_results=None,
        brand_performance_results=None,
        forecasting_results=None,
        
        # Final outputs
        final_report=None,
        recommendations=None,
        next_steps=None,
        
        # Error handling
        errors=[],
        warnings=[],
        
        # Performance metrics
        total_token_usage={'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
        execution_time=None,
        cache_hits=0,
        cache_misses=0
    )


def update_agent_status(state: MarketingResearchState, agent_role: str, status: AgentStatus) -> MarketingResearchState:
    """Update the status of a specific agent."""
    state['agent_status'][agent_role] = status
    state['updated_at'] = datetime.now()
    return state


def store_agent_result(
    state: MarketingResearchState, 
    agent_role: str, 
    result: Dict[str, Any],
    token_usage: Optional[Dict[str, int]] = None
) -> MarketingResearchState:
    """Store results from an agent execution."""
    state['agent_results'][agent_role] = result
    state['agent_status'][agent_role] = AgentStatus.COMPLETED
    state['updated_at'] = datetime.now()
    
    # Store token usage if provided
    if token_usage:
        state['agent_token_usage'][agent_role] = token_usage
        # Update total token usage
        for key in ['prompt_tokens', 'completion_tokens', 'total_tokens']:
            if key in token_usage:
                state['total_token_usage'][key] += token_usage[key]
    
    # Store result in appropriate field based on agent role
    if agent_role == 'market_research_analyst':
        state['market_research_results'] = result
    elif agent_role == 'competitive_analyst':
        state['competitive_analysis_results'] = result
    elif agent_role == 'data_analyst':
        state['data_analysis_results'] = result
    elif agent_role == 'content_strategist':
        state['content_strategy_results'] = result
    elif agent_role == 'creative_copywriter':
        state['copywriting_results'] = result
    elif agent_role == 'campaign_optimizer':
        state['campaign_optimization_results'] = result
    elif agent_role == 'brand_performance_specialist':
        state['brand_performance_results'] = result
    elif agent_role == 'forecasting_specialist':
        state['forecasting_results'] = result
    
    return state


def store_agent_error(state: MarketingResearchState, agent_role: str, error: str) -> MarketingResearchState:
    """Store error from an agent execution."""
    state['agent_errors'][agent_role] = error
    state['agent_status'][agent_role] = AgentStatus.FAILED
    state['errors'].append(f"{agent_role}: {error}")
    state['updated_at'] = datetime.now()
    return state


def update_shared_data(state: MarketingResearchState, key: str, value: Any) -> MarketingResearchState:
    """Update shared data that can be accessed by all agents."""
    if 'shared_data' not in state:
        state['shared_data'] = {}
    state['shared_data'][key] = value
    state['updated_at'] = datetime.now()
    return state


def get_agent_context(state: MarketingResearchState, agent_role: str) -> Dict[str, Any]:
    """
    Get context data for a specific agent, including results from previous agents.
    This enables agent interdependency similar to the blackboard system.
    """
    context = {
        'workflow_id': state['workflow_id'],
        'workflow_type': state['workflow_type'],
        'target_audience': state['target_audience'],
        'campaign_type': state['campaign_type'],
        'budget': state['budget'],
        'duration': state['duration'],
        'analysis_focus': state['analysis_focus'],
        'business_objective': state['business_objective'],
        'competitive_landscape': state['competitive_landscape'],
        'market_segments': state['market_segments'],
        'product_categories': state['product_categories'],
        'key_metrics': state['key_metrics'],
        'brands': state['brands'],
        'campaign_goals': state['campaign_goals'],
        'shared_data': state.get('shared_data', {}),
        'cached_results': state['cached_results']
    }
    
    # Add results from completed agents
    for completed_agent, result in state['agent_results'].items():
        if completed_agent != agent_role:  # Don't include own results
            context[f'{completed_agent}_results'] = result
    
    # Add specific context based on agent role
    if agent_role == 'competitive_analyst':
        # Competitive analyst can use market research results
        if state['market_research_results']:
            context['market_research_insights'] = state['market_research_results']
    
    elif agent_role == 'data_analyst':
        # Data analyst can use market research and competitive analysis
        if state['market_research_results']:
            context['market_context'] = state['market_research_results']
        if state['competitive_analysis_results']:
            context['competitive_context'] = state['competitive_analysis_results']
    
    elif agent_role == 'content_strategist':
        # Content strategist can use all analytical results
        for result_key in ['market_research_results', 'competitive_analysis_results', 'data_analysis_results']:
            if state[result_key]:
                context[result_key] = state[result_key]
    
    elif agent_role == 'creative_copywriter':
        # Copywriter can use strategy and analytical results
        if state['content_strategy_results']:
            context['strategy_guidance'] = state['content_strategy_results']
        for result_key in ['market_research_results', 'data_analysis_results']:
            if state[result_key]:
                context[result_key] = state[result_key]
    
    elif agent_role == 'campaign_optimizer':
        # Campaign optimizer can use all previous results
        for result_key in ['market_research_results', 'competitive_analysis_results', 
                          'data_analysis_results', 'content_strategy_results']:
            if state[result_key]:
                context[result_key] = state[result_key]
    
    elif agent_role == 'brand_performance_specialist':
        # Brand specialist can use market research and competitive analysis
        if state['market_research_results']:
            context['market_insights'] = state['market_research_results']
        if state['competitive_analysis_results']:
            context['competitive_insights'] = state['competitive_analysis_results']
    
    elif agent_role == 'forecasting_specialist':
        # Forecasting specialist can use market research and data analysis
        if state['market_research_results']:
            context['market_trends'] = state['market_research_results']
        if state['data_analysis_results']:
            context['historical_data'] = state['data_analysis_results']
    
    return context


def is_workflow_complete(state: MarketingResearchState) -> bool:
    """Check if the workflow is complete (all selected agents have finished)."""
    for agent in state['selected_agents']:
        status = state['agent_status'].get(agent, AgentStatus.PENDING)
        if status not in [AgentStatus.COMPLETED, AgentStatus.FAILED, AgentStatus.SKIPPED]:
            return False
    return True


def get_next_agent(state: MarketingResearchState) -> Optional[str]:
    """
    Get the next agent to execute based on dependencies and current state.
    This implements the agent execution order logic.
    """
    # Define agent dependencies (which agents should run before others)
    dependencies = {
        'market_research_analyst': [],  # Can run first
        'competitive_analyst': [],  # Can run in parallel with market research
        'data_analyst': ['market_research_analyst'],  # Needs market context
        'content_strategist': ['market_research_analyst', 'competitive_analyst'],  # Needs market and competitive insights
        'creative_copywriter': ['content_strategist'],  # Needs strategy
        'campaign_optimizer': ['data_analyst', 'content_strategist'],  # Needs data and strategy
        'brand_performance_specialist': ['market_research_analyst', 'competitive_analyst'],  # Needs market insights
        'forecasting_specialist': ['market_research_analyst', 'data_analyst'],  # Needs market trends and data
    }
    
    for agent in state['selected_agents']:
        # Skip if already completed, failed, or running
        status = state['agent_status'].get(agent, AgentStatus.PENDING)
        if status != AgentStatus.PENDING:
            continue
        
        # Check if all dependencies are satisfied
        agent_dependencies = dependencies.get(agent, [])
        dependencies_satisfied = True
        
        for dep_agent in agent_dependencies:
            if dep_agent in state['selected_agents']:
                dep_status = state['agent_status'].get(dep_agent, AgentStatus.PENDING)
                if dep_status != AgentStatus.COMPLETED:
                    dependencies_satisfied = False
                    break
        
        if dependencies_satisfied:
            return agent
    
    return None