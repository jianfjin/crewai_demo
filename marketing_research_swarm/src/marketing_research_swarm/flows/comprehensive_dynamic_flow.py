"""
Comprehensive Dynamic CrewAI Flow Implementation

This module implements a complete workflow system using CrewAI Flow that includes
all 9 agents from agents.yaml with proper dependency management and execution order.
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
from crewai_tools import SerperDevTool, WebsiteSearchTool


class ComprehensiveWorkflowState(BaseModel):
    """State model for the comprehensive workflow with all agents."""
    workflow_id: str
    selected_agents: List[str]
    task_params: Dict[str, Any]
    agent_results: Dict[str, Any] = {}
    execution_order: List[str] = []
    current_step: int = 0
    total_steps: int = 0
    start_time: datetime = None
    token_usage: Dict[str, Any] = {}
    completed_phases: List[str] = []


class ComprehensiveDynamicFlow(Flow[ComprehensiveWorkflowState]):
    """
    Comprehensive CrewAI Flow that includes all 9 agents with proper dependencies.
    
    Features:
    - All 9 agents from agents.yaml
    - Proper dependency management
    - Phase-based execution (Foundation → Analysis → Strategy → Content)
    - Automatic token tracking
    - Agent interdependency through reference system
    """
    
    def __init__(self):
        super().__init__()
        self.reference_manager = get_reference_manager()
        self.token_tracker = get_blackboard_tracker()
        
        # Initialize search tools
        try:
            self.search_tool = SerperDevTool()
        except:
            self.search_tool = None
        
        try:
            self.web_search_tool = WebsiteSearchTool()
        except:
            self.web_search_tool = None
        
        # Comprehensive agent configurations with proper dependencies
        self.agent_configs = {
            # PHASE 1: FOUNDATION (Independent agents that provide base insights)
            'market_research_analyst': {
                'role': 'Market Research Analyst',
                'goal': 'Analyze current market structure, trends, and seasonal patterns to provide context for all other analyses',
                'backstory': 'An expert in beverage market research with deep knowledge of market dynamics, seasonal consumption patterns, and competitive landscape.',
                'tools': [beverage_market_analysis, time_series_analysis, cross_sectional_analysis],
                'task_type': 'market_research',
                'dependencies': [],
                'phase': 1,
                'execution_priority': 1
            },
            
            # PHASE 2: CORE ANALYSIS (Can run in parallel, may use foundation insights)
            'data_analyst': {
                'role': 'Data Analyst',
                'goal': 'Perform comprehensive data analysis on beverage sales to uncover profitability insights and performance patterns',
                'backstory': 'A quantitative expert specializing in beverage industry analytics with focus on extracting insights from sales data.',
                'tools': [profitability_analysis, time_series_analysis, cross_sectional_analysis, analyze_kpis],
                'task_type': 'data_analysis',
                'dependencies': ['market_research'],  # Benefits from market context
                'phase': 2,
                'execution_priority': 2
            },
            'competitive_analyst': {
                'role': 'Competitive Analyst',
                'goal': 'Analyze competitive landscape, market positioning, and brand dynamics',
                'backstory': 'An expert in competitive intelligence and market dynamics with deep knowledge of beverage industry competition.',
                'tools': [beverage_market_analysis, calculate_market_share, cross_sectional_analysis],
                'task_type': 'competitive_analysis',
                'dependencies': ['market_research'],  # Uses market structure insights
                'phase': 2,
                'execution_priority': 2
            },
            'brand_performance_specialist': {
                'role': 'Brand Performance Specialist',
                'goal': 'Monitor beverage brand metrics, market positioning, and competitive landscape',
                'backstory': 'An analytical professional specializing in beverage brand management and market dynamics.',
                'tools': [analyze_brand_performance, calculate_market_share, beverage_market_analysis] + ([self.search_tool, self.web_search_tool] if self.search_tool and self.web_search_tool else []),
                'task_type': 'brand_performance',
                'dependencies': ['market_research', 'competitive_analysis'],  # Uses market and competitive insights
                'phase': 2,
                'execution_priority': 3
            },
            
            # PHASE 3: STRATEGIC PLANNING (Depends on analysis results)
            'brand_strategist': {
                'role': 'Brand Strategist',
                'goal': 'Evaluate brand performance metrics and develop strategic recommendations for brand growth',
                'backstory': 'A strategic brand management expert specializing in beverage brand optimization.',
                'tools': [analyze_brand_performance, profitability_analysis],
                'task_type': 'brand_strategy',
                'dependencies': ['competitive_analysis', 'data_analysis', 'brand_performance'],
                'phase': 3,
                'execution_priority': 1
            },
            'campaign_optimizer': {
                'role': 'Campaign Optimizer',
                'goal': 'Develop strategic budget allocation and campaign optimization recommendations',
                'backstory': 'A strategic marketing expert who transforms analytical insights into actionable campaign strategies.',
                'tools': [plan_budget, calculate_roi],
                'task_type': 'campaign_optimization',
                'dependencies': ['data_analysis', 'brand_strategy'],
                'phase': 3,
                'execution_priority': 2
            },
            'forecasting_specialist': {
                'role': 'Forecasting Specialist',
                'goal': 'Generate accurate sales forecasts and predictive models based on market analysis insights',
                'backstory': 'A predictive analytics expert specializing in beverage industry forecasting.',
                'tools': [forecast_sales, analyze_kpis],
                'task_type': 'sales_forecast',
                'dependencies': ['data_analysis', 'market_research'],
                'phase': 3,
                'execution_priority': 3
            },
            
            # PHASE 4: CONTENT CREATION (Depends on strategic insights)
            'content_strategist': {
                'role': 'Content Strategist',
                'goal': 'Create personalized content strategies and campaign frameworks for beverage brands',
                'backstory': 'A creative thinker with expertise in beverage marketing and consumer behavior.',
                'tools': [self.search_tool, self.web_search_tool] if self.search_tool and self.web_search_tool else [],
                'task_type': 'content_strategy',
                'dependencies': ['market_research', 'brand_strategy', 'competitive_analysis'],
                'phase': 4,
                'execution_priority': 1
            },
            'creative_copywriter': {
                'role': 'Creative Copywriter',
                'goal': 'Generate high-quality, targeted marketing copy and creative assets',
                'backstory': 'A master of words who crafts persuasive and engaging copy for various marketing channels.',
                'tools': [self.search_tool, self.web_search_tool] if self.search_tool and self.web_search_tool else [],
                'task_type': 'creative_copywriting',
                'dependencies': ['content_strategy', 'brand_strategy'],
                'phase': 4,
                'execution_priority': 2
            }
        }
        
        # Define execution phases for proper dependency management
        self.execution_phases = {
            1: ['market_research_analyst'],  # Foundation
            2: ['data_analyst', 'competitive_analyst', 'brand_performance_specialist'],  # Core Analysis
            3: ['brand_strategist', 'campaign_optimizer', 'forecasting_specialist'],  # Strategic Planning
            4: ['content_strategist', 'creative_copywriter']  # Content Creation
        }
    
    @start()
    def initialize_comprehensive_workflow(self, selected_agents: List[str], task_params: Dict[str, Any]) -> ComprehensiveWorkflowState:
        """Initialize the comprehensive workflow with proper execution order."""
        workflow_id = f"comprehensive_flow_{int(time.time())}"
        
        # Start token tracking
        self.token_tracker.start_workflow_tracking(workflow_id)
        
        # Determine execution order based on dependencies
        execution_order = self._calculate_execution_order(selected_agents)
        
        # Create workflow state
        state = ComprehensiveWorkflowState(
            workflow_id=workflow_id,
            selected_agents=selected_agents,
            task_params=task_params,
            execution_order=execution_order,
            total_steps=len(selected_agents),
            start_time=datetime.now()
        )
        
        print(f"🚀 Initialized comprehensive workflow: {workflow_id}")
        print(f"📋 Selected agents: {selected_agents}")
        print(f"🔄 Execution order: {execution_order}")
        print(f"📊 Execution phases: {self._get_phase_breakdown(selected_agents)}")
        
        return state
    
    def _calculate_execution_order(self, selected_agents: List[str]) -> List[str]:
        """Calculate proper execution order based on dependencies."""
        execution_order = []
        remaining_agents = selected_agents.copy()
        
        # Process agents phase by phase
        for phase in sorted(self.execution_phases.keys()):
            phase_agents = []
            
            # Find agents in this phase that are selected
            for agent in self.execution_phases[phase]:
                if agent in remaining_agents:
                    # Check if dependencies are satisfied
                    config = self.agent_configs[agent]
                    dependencies_met = all(
                        any(dep in self.agent_configs[completed_agent]['task_type'] 
                            for completed_agent in execution_order)
                        for dep in config['dependencies']
                    ) if config['dependencies'] else True
                    
                    if dependencies_met or not config['dependencies']:
                        phase_agents.append(agent)
            
            # Sort by execution priority within phase
            phase_agents.sort(key=lambda x: self.agent_configs[x]['execution_priority'])
            execution_order.extend(phase_agents)
            
            # Remove processed agents
            for agent in phase_agents:
                remaining_agents.remove(agent)
        
        # Add any remaining agents (shouldn't happen with proper config)
        execution_order.extend(remaining_agents)
        
        return execution_order
    
    def _get_phase_breakdown(self, selected_agents: List[str]) -> Dict[int, List[str]]:
        """Get breakdown of selected agents by execution phase."""
        phase_breakdown = {}
        
        for agent in selected_agents:
            if agent in self.agent_configs:
                phase = self.agent_configs[agent]['phase']
                if phase not in phase_breakdown:
                    phase_breakdown[phase] = []
                phase_breakdown[phase].append(agent)
        
        return phase_breakdown
    
    @listen(initialize_comprehensive_workflow)
    def execute_foundation_phase(self, state: ComprehensiveWorkflowState) -> ComprehensiveWorkflowState:
        """Execute Phase 1: Foundation agents (market_research_analyst)."""
        phase_1_agents = [agent for agent in state.execution_order 
                         if agent in self.execution_phases[1]]
        
        if not phase_1_agents:
            print("📋 Phase 1 (Foundation): No agents selected")
            state.completed_phases.append("foundation")
            return state
        
        print(f"🏗️  Phase 1 (Foundation): Executing {phase_1_agents}")
        
        for agent_type in phase_1_agents:
            if agent_type in state.selected_agents:
                print(f"🔍 Executing foundation agent: {agent_type}")
                
                # Create and execute agent
                agent = self._create_state_aware_agent(agent_type, state.workflow_id)
                task = self._create_dynamic_task(agent_type, state.task_params, [])
                
                # Execute task
                crew = Crew(agents=[agent], tasks=[task], verbose=True)
                result = crew.kickoff()
                
                # Store result with reference
                ref_key = self.reference_manager.store_agent_result(
                    agent_role=agent_type,
                    task_type=self.agent_configs[agent_type]['task_type'],
                    result_data=result,
                    workflow_id=state.workflow_id
                )
                
                state.agent_results[agent_type] = {
                    'result': result,
                    'reference_key': ref_key,
                    'task_type': self.agent_configs[agent_type]['task_type'],
                    'phase': 1
                }
                
                state.current_step += 1
                print(f"✅ Completed {agent_type} - Reference: {ref_key}")
        
        state.completed_phases.append("foundation")
        print(f"🏁 Phase 1 (Foundation) completed")
        return state
    
    @listen(execute_foundation_phase)
    def execute_analysis_phase(self, state: ComprehensiveWorkflowState) -> ComprehensiveWorkflowState:
        """Execute Phase 2: Core Analysis agents."""
        phase_2_agents = [agent for agent in state.execution_order 
                         if agent in self.execution_phases[2]]
        
        if not phase_2_agents:
            print("📋 Phase 2 (Analysis): No agents selected")
            state.completed_phases.append("analysis")
            return state
        
        print(f"🔬 Phase 2 (Analysis): Executing {phase_2_agents}")
        
        # Get available references from previous phases
        available_references = self._get_available_references(state, ['foundation'])
        
        for agent_type in phase_2_agents:
            if agent_type in state.selected_agents:
                print(f"📊 Executing analysis agent: {agent_type}")
                
                # Create and execute agent
                agent = self._create_state_aware_agent(agent_type, state.workflow_id)
                task = self._create_dynamic_task(agent_type, state.task_params, available_references)
                
                # Execute task
                crew = Crew(agents=[agent], tasks=[task], verbose=True)
                result = crew.kickoff()
                
                # Store result with reference
                ref_key = self.reference_manager.store_agent_result(
                    agent_role=agent_type,
                    task_type=self.agent_configs[agent_type]['task_type'],
                    result_data=result,
                    workflow_id=state.workflow_id
                )
                
                state.agent_results[agent_type] = {
                    'result': result,
                    'reference_key': ref_key,
                    'task_type': self.agent_configs[agent_type]['task_type'],
                    'phase': 2
                }
                
                state.current_step += 1
                print(f"✅ Completed {agent_type} - Reference: {ref_key}")
        
        state.completed_phases.append("analysis")
        print(f"🏁 Phase 2 (Analysis) completed")
        return state
    
    @listen(execute_analysis_phase)
    def execute_strategy_phase(self, state: ComprehensiveWorkflowState) -> ComprehensiveWorkflowState:
        """Execute Phase 3: Strategic Planning agents."""
        phase_3_agents = [agent for agent in state.execution_order 
                         if agent in self.execution_phases[3]]
        
        if not phase_3_agents:
            print("📋 Phase 3 (Strategy): No agents selected")
            state.completed_phases.append("strategy")
            return state
        
        print(f"🎯 Phase 3 (Strategy): Executing {phase_3_agents}")
        
        # Get available references from previous phases
        available_references = self._get_available_references(state, ['foundation', 'analysis'])
        
        for agent_type in phase_3_agents:
            if agent_type in state.selected_agents:
                print(f"🎯 Executing strategy agent: {agent_type}")
                
                # Create and execute agent
                agent = self._create_state_aware_agent(agent_type, state.workflow_id)
                task = self._create_dynamic_task(agent_type, state.task_params, available_references)
                
                # Execute task
                crew = Crew(agents=[agent], tasks=[task], verbose=True)
                result = crew.kickoff()
                
                # Store result with reference
                ref_key = self.reference_manager.store_agent_result(
                    agent_role=agent_type,
                    task_type=self.agent_configs[agent_type]['task_type'],
                    result_data=result,
                    workflow_id=state.workflow_id
                )
                
                state.agent_results[agent_type] = {
                    'result': result,
                    'reference_key': ref_key,
                    'task_type': self.agent_configs[agent_type]['task_type'],
                    'phase': 3
                }
                
                state.current_step += 1
                print(f"✅ Completed {agent_type} - Reference: {ref_key}")
        
        state.completed_phases.append("strategy")
        print(f"🏁 Phase 3 (Strategy) completed")
        return state
    
    @listen(execute_strategy_phase)
    def execute_content_phase(self, state: ComprehensiveWorkflowState) -> ComprehensiveWorkflowState:
        """Execute Phase 4: Content Creation agents."""
        phase_4_agents = [agent for agent in state.execution_order 
                         if agent in self.execution_phases[4]]
        
        if not phase_4_agents:
            print("📋 Phase 4 (Content): No agents selected")
            state.completed_phases.append("content")
            return state
        
        print(f"✍️  Phase 4 (Content): Executing {phase_4_agents}")
        
        # Get available references from all previous phases
        available_references = self._get_available_references(state, ['foundation', 'analysis', 'strategy'])
        
        for agent_type in phase_4_agents:
            if agent_type in state.selected_agents:
                print(f"✍️  Executing content agent: {agent_type}")
                
                # Create and execute agent
                agent = self._create_state_aware_agent(agent_type, state.workflow_id)
                task = self._create_dynamic_task(agent_type, state.task_params, available_references)
                
                # Execute task
                crew = Crew(agents=[agent], tasks=[task], verbose=True)
                result = crew.kickoff()
                
                # Store result with reference
                ref_key = self.reference_manager.store_agent_result(
                    agent_role=agent_type,
                    task_type=self.agent_configs[agent_type]['task_type'],
                    result_data=result,
                    workflow_id=state.workflow_id
                )
                
                state.agent_results[agent_type] = {
                    'result': result,
                    'reference_key': ref_key,
                    'task_type': self.agent_configs[agent_type]['task_type'],
                    'phase': 4
                }
                
                state.current_step += 1
                print(f"✅ Completed {agent_type} - Reference: {ref_key}")
        
        state.completed_phases.append("content")
        print(f"🏁 Phase 4 (Content) completed")
        return state
    
    @listen(execute_content_phase)
    def finalize_comprehensive_workflow(self, state: ComprehensiveWorkflowState) -> ComprehensiveWorkflowState:
        """Finalize the comprehensive workflow and generate summary."""
        end_time = datetime.now()
        duration = (end_time - state.start_time).total_seconds()
        
        # Complete token tracking
        final_stats = self.token_tracker.complete_workflow_tracking(state.workflow_id)
        
        # Generate comprehensive summary
        summary = {
            'workflow_id': state.workflow_id,
            'total_agents': len(state.selected_agents),
            'completed_phases': state.completed_phases,
            'execution_order': state.execution_order,
            'duration_seconds': duration,
            'agent_results': {
                agent: {
                    'task_type': result['task_type'],
                    'reference_key': result['reference_key'],
                    'phase': result['phase']
                }
                for agent, result in state.agent_results.items()
            },
            'token_stats': final_stats,
            'phase_breakdown': self._get_phase_breakdown(state.selected_agents)
        }
        
        state.token_usage = final_stats
        
        print(f"🎉 Comprehensive workflow completed!")
        print(f"⏱️  Total duration: {duration:.2f} seconds")
        print(f"🔢 Total agents executed: {len(state.agent_results)}")
        print(f"📊 Phases completed: {state.completed_phases}")
        print(f"🎯 Final summary: {summary}")
        
        return state
    
    def _get_available_references(self, state: ComprehensiveWorkflowState, completed_phases: List[str]) -> List[Dict[str, str]]:
        """Get available reference keys from completed phases."""
        references = []
        
        for agent, result in state.agent_results.items():
            phase_name = {1: 'foundation', 2: 'analysis', 3: 'strategy', 4: 'content'}[result['phase']]
            if phase_name in completed_phases:
                references.append({
                    'agent': agent,
                    'task_type': result['task_type'],
                    'reference_key': result['reference_key']
                })
        
        return references
    
    def _create_state_aware_agent(self, agent_type: str, workflow_id: str) -> StateAwareAgent:
        """Create a state-aware agent for the workflow."""
        config = self.agent_configs[agent_type]
        
        # Filter out None tools
        tools = [tool for tool in config['tools'] if tool is not None]
        
        return StateAwareAgent(
            role=config['role'],
            goal=config['goal'],
            backstory=config['backstory'],
            workflow_id=workflow_id,
            tools=tools,
            allow_delegation=False
        )
    
    def _create_dynamic_task(self, agent_type: str, task_params: Dict[str, Any], available_references: List[Dict[str, str]]) -> Task:
        """Create a dynamic task with proper context and references."""
        config = self.agent_configs[agent_type]
        
        # Build context from available references
        reference_context = ""
        if available_references:
            reference_context = "\n\nAvailable insights from previous agents:\n"
            for ref in available_references:
                reference_context += f"- {ref['agent']} ({ref['task_type']}): Reference key {ref['reference_key']}\n"
            reference_context += "\nUse the retrieve_by_reference tool to access specific insights when needed.\n"
        
        # Create task description based on agent type
        task_descriptions = {
            'market_research': f"""
            Conduct comprehensive market research on the beverage industry using the enhanced sales data from {task_params.get('data_file_path', 'data/beverage_sales.csv')}. 
            Use the beverage market analysis tool to understand market structure, then apply time series analysis to identify temporal patterns 
            and cross-sectional analysis to compare performance across regions, brands, and categories. 
            Focus on identifying opportunities for {task_params.get('target_audience', 'target consumers')} and provide insights on pricing strategies, profit margins, and market share dynamics.
            This analysis will provide the foundation for all subsequent agents.
            """,
            
            'data_analysis': f"""
            Perform comprehensive data analysis on the beverage sales data from {task_params.get('data_file_path', 'data/beverage_sales.csv')} using the specialized analytical tools. 
            Use the profitability analysis tool to examine profit margins across brands, categories, and regions. 
            Use the time series analysis tool to identify seasonal patterns and growth trends.
            {reference_context}
            Focus on quantitative insights that will support strategic decision-making.
            """,
            
            'competitive_analysis': f"""
            Analyze competitive landscape, market positioning, and brand dynamics in the beverage industry.
            Use market share analysis and competitive positioning tools to understand the competitive environment.
            {reference_context}
            Provide insights on competitive threats, opportunities, and market positioning strategies.
            """,
            
            'brand_performance': f"""
            Analyze brand performance in the beverage market using comprehensive sales data and market insights.
            Evaluate brand performance across different categories, regions, and price points.
            {reference_context}
            Track key metrics such as revenue share, profit margins, pricing strategies, and market penetration.
            """,
            
            'brand_strategy': f"""
            Develop strategic brand recommendations based on competitive analysis and data insights.
            {reference_context}
            Create actionable strategies for brand growth, positioning, and market expansion.
            """,
            
            'campaign_optimization': f"""
            Develop strategic budget allocation and campaign optimization recommendations based on data analysis insights.
            Consider the {task_params.get('budget', '$100,000')} total budget and {task_params.get('duration', '3 months')} campaign duration.
            {reference_context}
            Focus on {task_params.get('campaign_goals', 'ROI maximization')} and provide specific recommendations for campaign improvement.
            """,
            
            'sales_forecast': f"""
            Generate accurate sales forecasts and predictive models based on market analysis insights and historical data patterns.
            {reference_context}
            Provide reliable sales predictions, confidence intervals, and scenario planning for strategic decision-making.
            """,
            
            'content_strategy': f"""
            Develop a comprehensive content strategy for beverage brands based on market research and strategic insights.
            The strategy should target {task_params.get('target_audience', 'target consumers')} and leverage insights about regional preferences, category trends, and competitive positioning.
            {reference_context}
            Include recommendations for social media campaigns, influencer collaborations, seasonal promotions, and brand positioning strategies.
            """,
            
            'creative_copywriting': f"""
            Create compelling marketing copy for beverage campaigns targeting {task_params.get('target_audience', 'target consumers')}.
            Develop copy that resonates with beverage consumers, highlights product benefits, and addresses category-specific needs.
            {reference_context}
            Include social media posts, product descriptions, and campaign taglines that reflect strategic insights and market research findings.
            """
        }
        
        task_type = config['task_type']
        description = task_descriptions.get(task_type, f"Execute {task_type} task with available insights.")
        
        return Task(
            description=description,
            expected_output=f"A comprehensive {task_type} report with actionable insights and recommendations. Include reference keys for any detailed data that other agents might need.",
            agent=None  # Will be set by CrewAI
        )


def create_comprehensive_flow() -> ComprehensiveDynamicFlow:
    """Create a comprehensive dynamic flow instance."""
    return ComprehensiveDynamicFlow()


# Example usage
if __name__ == "__main__":
    # Example of running the comprehensive flow
    flow = create_comprehensive_flow()
    
    # Example parameters
    task_params = {
        'data_file_path': 'data/beverage_sales.csv',
        'target_audience': 'health-conscious millennials',
        'budget': '$100,000',
        'duration': '3 months',
        'campaign_goals': 'increase brand awareness and market share'
    }
    
    # Run with all agents
    all_agents = [
        'market_research_analyst', 'data_analyst', 'competitive_analyst',
        'brand_performance_specialist', 'brand_strategist', 'campaign_optimizer',
        'forecasting_specialist', 'content_strategist', 'creative_copywriter'
    ]
    
    print("🚀 Starting comprehensive workflow with all 9 agents...")
    result = flow.kickoff(selected_agents=all_agents, task_params=task_params)
    print("✅ Comprehensive workflow completed!")