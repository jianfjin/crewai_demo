"""
Dynamic CrewAI Flow Implementation

This module implements a dynamic workflow system using CrewAI Flow that:
1. Creates workflows dynamically based on selected agents
2. Tracks token usage accurately through Flow's built-in mechanisms
3. Implements proper agent interdependency
4. Provides real-time execution monitoring
"""

import time
from typing import Dict, Any, List, Optional
from datetime import datetime
from crewai.flow.flow import Flow, listen, start
from crewai import Agent, Task, Crew
from pydantic import BaseModel

from ..blackboard.state_aware_agents import StateAwareAgent
from ..blackboard.result_reference_system import get_reference_manager
from ..blackboard.enhanced_token_tracker import get_blackboard_tracker
from ..tools.advanced_tools import (
    calculate_roi, analyze_kpis, forecast_sales, plan_budget,
    analyze_brand_performance, calculate_market_share,
    time_series_analysis, cross_sectional_analysis,
    beverage_market_analysis, profitability_analysis
)


class WorkflowState(BaseModel):
    """State model for the dynamic workflow."""
    workflow_id: str
    selected_agents: List[str]
    task_params: Dict[str, Any]
    agent_results: Dict[str, Any] = {}
    current_step: int = 0
    total_steps: int = 0
    start_time: datetime = None
    token_usage: Dict[str, Any] = {}


class DynamicMarketingFlow(Flow[WorkflowState]):
    """
    Dynamic CrewAI Flow that creates workflows based on selected agents.
    
    Features:
    - Dynamic agent and task creation
    - Automatic token tracking
    - Agent interdependency through reference system
    - Real-time progress monitoring
    """
    
    def __init__(self):
        super().__init__()
        self.reference_manager = get_reference_manager()
        self.token_tracker = get_blackboard_tracker()
        
        # Agent configurations
        self.agent_configs = {
            'data_analyst': {
                'role': 'Data Analyst',
                'goal': 'Perform comprehensive data analysis on sales data',
                'backstory': 'Expert in data analysis and statistical modeling',
                'tools': [profitability_analysis, time_series_analysis, cross_sectional_analysis, analyze_kpis],
                'task_type': 'data_analysis',
                'dependencies': []
            },
            'competitive_analyst': {
                'role': 'Competitive Analyst',
                'goal': 'Analyze competitive landscape and market positioning',
                'backstory': 'Expert in competitive intelligence and market dynamics',
                'tools': [beverage_market_analysis, calculate_market_share, cross_sectional_analysis],
                'task_type': 'competitive_analysis',
                'dependencies': []
            },
            'brand_strategist': {
                'role': 'Brand Strategist',
                'goal': 'Develop strategic brand recommendations',
                'backstory': 'Expert in brand strategy and market positioning',
                'tools': [analyze_brand_performance, profitability_analysis],
                'task_type': 'brand_strategy',
                'dependencies': ['competitive_analysis', 'data_analysis']
            },
            'campaign_optimizer': {
                'role': 'Campaign Optimizer',
                'goal': 'Optimize marketing campaigns and budget allocation',
                'backstory': 'Expert in campaign optimization and ROI maximization',
                'tools': [plan_budget, calculate_roi],
                'task_type': 'campaign_optimization',
                'dependencies': ['data_analysis']
            },
            'forecasting_specialist': {
                'role': 'Forecasting Specialist',
                'goal': 'Generate accurate sales forecasts and predictive models',
                'backstory': 'Expert in predictive analytics and forecasting',
                'tools': [forecast_sales, analyze_kpis],
                'task_type': 'sales_forecast',
                'dependencies': ['data_analysis']
            }
        }
    
    @start()
    def initialize_workflow(self, selected_agents: List[str], task_params: Dict[str, Any]) -> WorkflowState:
        """Initialize the dynamic workflow based on selected agents."""
        workflow_id = f"flow_{int(time.time())}"
        
        # Start token tracking
        self.token_tracker.start_workflow_tracking(workflow_id)
        
        # Create workflow state
        state = WorkflowState(
            workflow_id=workflow_id,
            selected_agents=selected_agents,
            task_params=task_params,
            total_steps=len(selected_agents),
            start_time=datetime.now()
        )
        
        print(f"🚀 Initialized dynamic workflow: {workflow_id}")
        print(f"📋 Selected agents: {selected_agents}")
        
        return state
    
    @listen(initialize_workflow)
    def execute_data_analysis(self, state: WorkflowState) -> WorkflowState:
        """Execute data analysis if data_analyst is selected."""
        if 'data_analyst' not in state.selected_agents:
            return state
        
        print("📊 Executing data analysis...")
        
        # Create data analyst agent
        agent = self._create_agent('data_analyst', state.workflow_id)
        
        # Create data analysis task
        task = Task(
            description=f"""Perform comprehensive data analysis on the sales data from {state.task_params.get('data_file_path', 'data/beverage_sales.csv')}.
            Analyze {', '.join(state.task_params.get('product_categories', []))} categories across {', '.join(state.task_params.get('market_segments', []))} regions.
            Focus on key metrics: {', '.join(state.task_params.get('key_metrics', []))}.
            Generate insights for {state.task_params.get('target_audience', 'target market')}.""",
            expected_output="A detailed data analysis report with insights, trends, and recommendations.",
            agent=agent
        )
        
        # Execute task
        result = self._execute_task_with_tracking(agent, task, state.workflow_id)
        
        # Store result with reference system
        reference_key = self.reference_manager.store_agent_result(
            agent_role='data_analyst',
            task_type='data_analysis',
            result_data=result
        )
        
        state.agent_results['data_analyst'] = {
            'result': result,
            'reference_key': reference_key,
            'completed_at': datetime.now().isoformat()
        }
        state.current_step += 1
        
        print(f"✅ Data analysis completed: {reference_key}")
        return state
    
    @listen(execute_data_analysis)
    def execute_competitive_analysis(self, state: WorkflowState) -> WorkflowState:
        """Execute competitive analysis if competitive_analyst is selected."""
        if 'competitive_analyst' not in state.selected_agents:
            return state
        
        print("🏆 Executing competitive analysis...")
        
        # Create competitive analyst agent
        agent = self._create_agent('competitive_analyst', state.workflow_id)
        
        # Create competitive analysis task
        task = Task(
            description=f"""Analyze competitive landscape for {', '.join(state.task_params.get('brands', []))} in the {state.task_params.get('analysis_focus', 'beverage industry')}.
            Examine competitive dynamics across {', '.join(state.task_params.get('market_segments', []))} markets.
            Assess market share, competitive threats, and positioning opportunities.
            Focus on {state.task_params.get('target_audience', 'target market')}.""",
            expected_output="A detailed competitive analysis report with market positioning insights and strategic recommendations.",
            agent=agent
        )
        
        # Execute task
        result = self._execute_task_with_tracking(agent, task, state.workflow_id)
        
        # Store result with reference system
        reference_key = self.reference_manager.store_agent_result(
            agent_role='competitive_analyst',
            task_type='competitive_analysis',
            result_data=result
        )
        
        state.agent_results['competitive_analyst'] = {
            'result': result,
            'reference_key': reference_key,
            'completed_at': datetime.now().isoformat()
        }
        state.current_step += 1
        
        print(f"✅ Competitive analysis completed: {reference_key}")
        return state
    
    @listen(execute_competitive_analysis)
    def execute_brand_strategy(self, state: WorkflowState) -> WorkflowState:
        """Execute brand strategy if brand_strategist is selected."""
        if 'brand_strategist' not in state.selected_agents:
            return state
        
        print("🎯 Executing brand strategy...")
        
        # Get relevant references for dependencies
        relevant_refs = self.reference_manager.get_relevant_references('brand_strategist', 'brand_strategy')
        
        # Create brand strategist agent
        agent = self._create_agent('brand_strategist', state.workflow_id)
        
        # Create context with previous results
        context_info = ""
        if relevant_refs:
            context_info = f"\n\nPrevious analysis results available via reference keys: {[ref.reference_key for ref in relevant_refs]}"
        
        # Create brand strategy task
        task = Task(
            description=f"""Develop strategic brand recommendations for {', '.join(state.task_params.get('brands', []))}.
            Create optimization strategies for {', '.join(state.task_params.get('market_segments', []))} markets.
            Consider budget allocation of ${state.task_params.get('budget', 0):,} over {state.task_params.get('duration', '6 months')}.
            {context_info}""",
            expected_output="A comprehensive brand strategy document with optimization recommendations and actionable plans.",
            agent=agent
        )
        
        # Execute task
        result = self._execute_task_with_tracking(agent, task, state.workflow_id)
        
        # Store result with reference system
        reference_key = self.reference_manager.store_agent_result(
            agent_role='brand_strategist',
            task_type='brand_strategy',
            result_data=result,
            dependencies=[ref.reference_key for ref in relevant_refs]
        )
        
        state.agent_results['brand_strategist'] = {
            'result': result,
            'reference_key': reference_key,
            'completed_at': datetime.now().isoformat()
        }
        state.current_step += 1
        
        print(f"✅ Brand strategy completed: {reference_key}")
        return state
    
    @listen(execute_brand_strategy)
    def execute_campaign_optimization(self, state: WorkflowState) -> WorkflowState:
        """Execute campaign optimization if campaign_optimizer is selected."""
        if 'campaign_optimizer' not in state.selected_agents:
            return state
        
        print("⚡ Executing campaign optimization...")
        
        # Get relevant references for dependencies
        relevant_refs = self.reference_manager.get_relevant_references('campaign_optimizer', 'campaign_optimization')
        
        # Create campaign optimizer agent
        agent = self._create_agent('campaign_optimizer', state.workflow_id)
        
        # Create context with previous results
        context_info = ""
        if relevant_refs:
            context_info = f"\n\nPrevious analysis results available via reference keys: {[ref.reference_key for ref in relevant_refs]}"
        
        # Create campaign optimization task
        task = Task(
            description=f"""Optimize marketing campaign with budget of ${state.task_params.get('budget', 0):,} over {state.task_params.get('duration', '6 months')}.
            Focus on {', '.join(state.task_params.get('campaign_goals', []))} across {', '.join(state.task_params.get('market_segments', []))} markets.
            {context_info}""",
            expected_output="A comprehensive optimization strategy with budget allocation and performance projections.",
            agent=agent
        )
        
        # Execute task
        result = self._execute_task_with_tracking(agent, task, state.workflow_id)
        
        # Store result with reference system
        reference_key = self.reference_manager.store_agent_result(
            agent_role='campaign_optimizer',
            task_type='campaign_optimization',
            result_data=result,
            dependencies=[ref.reference_key for ref in relevant_refs]
        )
        
        state.agent_results['campaign_optimizer'] = {
            'result': result,
            'reference_key': reference_key,
            'completed_at': datetime.now().isoformat()
        }
        state.current_step += 1
        
        print(f"✅ Campaign optimization completed: {reference_key}")
        return state
    
    @listen(execute_campaign_optimization)
    def finalize_workflow(self, state: WorkflowState) -> WorkflowState:
        """Finalize the workflow and compile results."""
        print("🎯 Finalizing workflow...")
        
        # Complete token tracking
        final_stats = self.token_tracker.complete_workflow_tracking(state.workflow_id)
        state.token_usage = final_stats
        
        # Calculate execution time
        end_time = datetime.now()
        duration = (end_time - state.start_time).total_seconds()
        
        print(f"✅ Workflow completed in {duration:.2f} seconds")
        print(f"📊 Token usage: {final_stats.get('total_tokens', 0)} tokens")
        print(f"📋 Completed {len(state.agent_results)} agent tasks")
        
        return state
    
    def _create_agent(self, agent_type: str, workflow_id: str) -> StateAwareAgent:
        """Create a state-aware agent for the workflow."""
        config = self.agent_configs[agent_type]
        
        agent = StateAwareAgent(
            role=config['role'],
            goal=config['goal'],
            backstory=config['backstory'],
            tools=config['tools'],
            blackboard_system=None,  # We'll manage state through Flow
            workflow_id=workflow_id,
            verbose=True
        )
        
        return agent
    
    def _execute_task_with_tracking(self, agent: StateAwareAgent, task: Task, workflow_id: str) -> str:
        """Execute a task with proper token tracking."""
        # Start agent tracking
        self.token_tracker.start_agent_tracking(workflow_id, agent.role, task.description[:50])
        
        # Create a single-agent crew for execution
        crew = Crew(
            agents=[agent],
            tasks=[task],
            verbose=True
        )
        
        # Execute the crew
        result = crew.kickoff()
        
        # Complete agent tracking
        agent_stats = self.token_tracker.complete_agent_tracking(workflow_id, agent.role)
        
        print(f"🔍 Agent {agent.role} used {agent_stats.get('total_tokens', 0)} tokens")
        
        return str(result)


# Factory function for easy integration
def create_dynamic_flow() -> DynamicMarketingFlow:
    """Create a new dynamic marketing flow instance."""
    return DynamicMarketingFlow()


# Integration function for the dashboard
def run_dynamic_workflow(selected_agents: List[str], task_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a dynamic workflow with selected agents and return results.
    
    Args:
        selected_agents: List of agent names to include in workflow
        task_params: Parameters for task execution
        
    Returns:
        Dictionary with workflow results and metrics
    """
    try:
        # Create and run the flow
        flow = create_dynamic_flow()
        
        # Execute the workflow
        final_state = flow.kickoff(
            selected_agents=selected_agents,
            task_params=task_params
        )
        
        return {
            'success': True,
            'workflow_id': final_state.workflow_id,
            'agent_results': final_state.agent_results,
            'token_usage': final_state.token_usage,
            'execution_time': (datetime.now() - final_state.start_time).total_seconds(),
            'completed_steps': final_state.current_step,
            'total_steps': final_state.total_steps
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'workflow_id': None,
            'agent_results': {},
            'token_usage': {},
            'execution_time': 0
        }