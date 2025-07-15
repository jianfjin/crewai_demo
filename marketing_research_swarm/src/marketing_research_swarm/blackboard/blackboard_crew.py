"""
Blackboard-Integrated CrewAI Implementation

This module provides a CrewAI implementation that uses the integrated blackboard system
to minimize token usage through shared state management and coordinated execution.
"""

import yaml
import os
import requests
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from pydantic import Field, ConfigDict
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

# Load environment variables
load_dotenv()

from ..tools import read_file_tool, read_csv_tool
from ..tools.advanced_tools import (
    calculate_roi,
    analyze_kpis,
    forecast_sales,
    plan_budget,
    analyze_brand_performance,
    calculate_market_share,
    time_series_analysis,
    cross_sectional_analysis,
    beverage_market_analysis,
    profitability_analysis,
)
from ..tools.context_aware_tools import create_context_aware_tools, reference_retriever, set_agent_context_for_tools
from crewai_tools import SerperDevTool, WebsiteSearchTool
from .integrated_blackboard import get_integrated_blackboard
from .agent_dependency_manager import get_dependency_manager
from .enhanced_agent_dependencies import get_enhanced_dependency_manager
from .state_aware_agents import StateAwareAgent
from .smart_cache import get_smart_cache


def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


class BlackboardMarketingResearchCrew:
    """
    CrewAI implementation that uses the integrated blackboard system
    for maximum token efficiency through shared state management.
    """
    
    def __init__(self, agents_config_path, tasks_config_path, selected_agents=None):
        self.agents_config = load_yaml(agents_config_path)
        self.tasks_config = load_yaml(tasks_config_path)
        self.selected_agents = selected_agents  # List of selected agent names
        self.blackboard = get_integrated_blackboard()
        self.dependency_manager = get_dependency_manager()
        self.enhanced_dependency_manager = get_enhanced_dependency_manager()
        self.smart_cache = get_smart_cache()
        
        # Resolve dependencies and optimize agent order if selected_agents provided
        if self.selected_agents:
            try:
                # Use enhanced dependency manager for better optimization
                self.optimized_agent_order = self.enhanced_dependency_manager.get_optimal_execution_order(self.selected_agents)
                print(f"[ENHANCED_DEPENDENCY] Optimized agent order: {self.optimized_agent_order}")
                
                # Get tool sharing opportunities
                self.tool_sharing_plan = self.enhanced_dependency_manager.get_shared_tool_opportunities(self.selected_agents)
                print(f"[TOOL_SHARING] Found {len(self.tool_sharing_plan)} tool sharing opportunities")
                
                # Get data sharing plan
                self.data_sharing_plan = self.enhanced_dependency_manager.get_data_sharing_plan(self.selected_agents)
                print(f"[DATA_SHARING] Created data sharing plan for {len(self.data_sharing_plan)} agents")
                
                # Calculate efficiency score
                efficiency_analysis = self.enhanced_dependency_manager.calculate_efficiency_score(self.selected_agents)
                print(f"[EFFICIENCY] Score: {efficiency_analysis['score']:.2f}")
                if efficiency_analysis['recommendations']:
                    print(f"[EFFICIENCY] Recommendations: {efficiency_analysis['recommendations']}")
                    
            except Exception as e:
                print(f"[ENHANCED_DEPENDENCY] Error with enhanced optimization, falling back: {e}")
                try:
                    self.optimized_agent_order = self.dependency_manager.resolve_dependencies(self.selected_agents)
                    print(f"[FALLBACK_DEPENDENCY] Using fallback order: {self.optimized_agent_order}")
                except Exception as e2:
                    print(f"[FALLBACK_DEPENDENCY] Fallback failed: {e2}")
                    self.optimized_agent_order = self.selected_agents
                
                self.tool_sharing_plan = {}
                self.data_sharing_plan = {}
        else:
            self.optimized_agent_order = []
            self.tool_sharing_plan = {}
            self.data_sharing_plan = {}
        
        # Remove duplicate blackboard initialization
        
        # Initialize search tools (with fallback if API keys not available)
        try:
            search_tool = SerperDevTool()
        except:
            search_tool = None
        
        try:
            web_search_tool = WebsiteSearchTool()
        except:
            web_search_tool = None
        
        # Create context-aware tools with smart caching
        context_aware_tools = create_context_aware_tools()
        
        self.tools = {
            "read_file_tool": read_file_tool,
            "read_csv_tool": read_csv_tool,
            "retrieve_by_reference": reference_retriever,
            # Use context-aware versions of analytical tools with smart caching
            "calculate_roi": context_aware_tools['calculate_roi'],
            "analyze_kpis": context_aware_tools['analyze_kpis'],
            "forecast_sales": context_aware_tools['forecast_sales'],
            "plan_budget": context_aware_tools['plan_budget'],
            "analyze_brand_performance": context_aware_tools['analyze_brand_performance'],
            "calculate_market_share": context_aware_tools['calculate_market_share'],
            "time_series_analysis": context_aware_tools['time_series_analysis'],
            "cross_sectional_analysis": context_aware_tools['cross_sectional_analysis'],
            "beverage_market_analysis": context_aware_tools['beverage_market_analysis'],
            "profitability_analysis": context_aware_tools['profitability_analysis'],
        }
        
        # Add optional tools if available
        if search_tool:
            self.tools["search"] = search_tool
        if web_search_tool:
            self.tools["web_search"] = web_search_tool
    
    def _create_blackboard_agent(self, agent_config: Dict[str, Any], workflow_id: str) -> StateAwareAgent:
        """Create a state-aware agent that uses the blackboard system with smart caching."""
        agent_role = agent_config['role']
        
        # Get tools for this agent
        agent_tools = [
            self.tools[tool_name] 
            for tool_name in agent_config.get('tools', []) 
            if tool_name in self.tools
        ]
        
        # Set agent context for tools to enable proper cache tracking
        tool_dict = {tool.name: tool for tool in agent_tools if hasattr(tool, 'set_current_agent')}
        set_agent_context_for_tools(tool_dict, agent_role)
        
        llm = self._configure_llm(agent_config)
        
        # Create state-aware agent with enhanced capabilities
        state_aware_agent = StateAwareAgent(
            role=agent_role,
            goal=agent_config['goal'],
            backstory=agent_config['backstory'],
            workflow_id=workflow_id,
            blackboard_system=self.blackboard,
            tools=agent_tools,
            llm=llm,
            allow_delegation=agent_config.get('allow_delegation', False)
        )
        
        # Add data sharing context if available
        if agent_role in self.data_sharing_plan:
            sharing_info = self.data_sharing_plan[agent_role]
            if sharing_info['available_data']:
                print(f"[DATA_SHARING] {agent_role} can access data from: {[d['source_agent'] for d in sharing_info['available_data']]}")
        
        return state_aware_agent
    
    def _configure_llm(self, agent_config: Dict[str, Any]):
        """Configure LLM for the agent."""
        llm = None
        
        # Try to configure LLM based on available API keys and configuration
        if 'OPENAI_API_KEY' in os.environ and os.environ['OPENAI_API_KEY'] != 'your_openai_api_key_here':
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.7
            )
        else:
            # Try Ollama as fallback
            try:
                response = requests.get("http://localhost:11434/api/health", timeout=5)
                response.raise_for_status()
                llm = ChatOllama(model="gemma", base_url="http://localhost:11434")
            except requests.exceptions.RequestException:
                # If no LLM is available, the agent will use default behavior
                print(f"Warning: No LLM configured for agent {agent_config.get('role', 'unknown')}")
                pass
        
        return llm
    
    def _create_blackboard_task(self, task_config: Dict[str, Any], agents: List[StateAwareAgent]) -> Task:
        """Create a task that works with the blackboard system."""
        agent_role = task_config['agent'].strip()
        agent = next((a for a in agents if a.role.strip() == agent_role), None)
        
        if not agent:
            raise ValueError(f"Agent with role '{agent_role}' not found")
        
        return Task(
            description=task_config['description'],
            expected_output=task_config['expected_output'],
            agent=agent
        )
    
    def kickoff(self, inputs: Dict[str, Any]) -> Any:
        """
        Execute the crew using the blackboard system for coordinated execution.
        
        Args:
            inputs: Input data for the workflow
            
        Returns:
            Workflow results
        """
        # Determine workflow type based on inputs or tasks
        workflow_type = self._determine_workflow_type(inputs)
        
        # Create integrated workflow in blackboard
        workflow_id = self.blackboard.create_integrated_workflow(
            workflow_type=workflow_type,
            initial_data=inputs
        )
        
        try:
            # Create state-aware agents in optimal execution order
            agents = []
            agent_execution_order = self.optimized_agent_order if self.optimized_agent_order else list(self.agents_config.keys())
            
            print(f"[EXECUTION_ORDER] Creating agents in order: {agent_execution_order}")
            
            for agent_name in agent_execution_order:
                if agent_name in self.agents_config:
                    agent_config = self.agents_config[agent_name]
                    agent = self._create_blackboard_agent(agent_config, workflow_id)
                    agents.append(agent)
                    print(f"[AGENT_CREATED] {agent_name} with {len(agent.tools)} tools")
            
            # Create tasks from the custom tasks config in the same order
            tasks = []
            for agent_name in agent_execution_order:
                # Find task for this agent
                for task_config in self.tasks_config.values():
                    if task_config.get('agent', '').strip() == agent_name:
                        task = self._create_blackboard_task(task_config, agents)
                        tasks.append(task)
                        print(f"[TASK_CREATED] Task for {agent_name}")
                        break
            
            # Create crew with blackboard coordination
            crew = BlackboardCoordinatedCrew(
                agents=agents,
                tasks=tasks,
                workflow_id=workflow_id,
                blackboard_system=self.blackboard,
                process=Process.sequential,
                verbose=True
            )
            
            # Execute with blackboard coordination
            result = crew.kickoff(inputs=inputs)
            
            # Get workflow summary with cache statistics
            workflow_summary = self.blackboard.get_workflow_summary(workflow_id)
            cache_stats = self.smart_cache.get_cache_stats()
            
            print(f"[CACHE_STATS] Hit ratio: {cache_stats['hit_ratio']:.2f}, Entries: {cache_stats['total_entries']}")
            
            # Add cache and optimization info to summary
            enhanced_summary = {
                **workflow_summary,
                'cache_performance': cache_stats,
                'tool_sharing_plan': self.tool_sharing_plan,
                'data_sharing_plan': self.data_sharing_plan,
                'agent_execution_order': self.optimized_agent_order
            }
            
            return {
                'result': result,
                'workflow_id': workflow_id,
                'workflow_summary': enhanced_summary
            }
            
        finally:
            # Cleanup workflow resources and cache
            cleanup_stats = self.blackboard.cleanup_workflow(workflow_id)
            
            # Clean up expired cache entries
            self.smart_cache.cleanup_expired()
            
            print(f"[CLEANUP] Workflow cleanup completed: {cleanup_stats}")
            print(f"[CLEANUP] Final cache stats: {self.smart_cache.get_cache_stats()}")
    
    def _determine_workflow_type(self, inputs: Dict[str, Any]) -> str:
        """Determine the workflow type based on inputs."""
        # Check for specific workflow indicators
        if 'roi' in str(inputs).lower():
            return 'roi_analysis'
        elif 'brand' in str(inputs).lower():
            return 'brand_performance'
        elif 'forecast' in str(inputs).lower():
            return 'sales_forecast'
        else:
            return 'general_analysis'


class BlackboardCoordinatedCrew:
    """
    Enhanced Crew that coordinates execution through the blackboard system.
    Uses composition instead of inheritance to avoid Pydantic field conflicts.
    """
    
    def __init__(self, 
                 agents: List[StateAwareAgent],
                 tasks: List[Task],
                 workflow_id: str,
                 blackboard_system,
                 **kwargs):
        """
        Initialize blackboard-coordinated crew.
        
        Args:
            agents: List of state-aware agents
            tasks: List of tasks
            workflow_id: Workflow identifier
            blackboard_system: Integrated blackboard system
            **kwargs: Additional crew arguments
        """
        # Store blackboard-specific attributes
        self.workflow_id = workflow_id
        self.blackboard_system = blackboard_system
        
        # Create the actual Crew instance
        self._crew = Crew(
            agents=agents, 
            tasks=tasks, 
            **kwargs
        )
        
        # Set up agents with workflow context
        for agent in agents:
            if hasattr(agent, 'set_workflow_context'):
                agent.set_workflow_context(workflow_id, blackboard_system)
    
    def kickoff(self, inputs: Dict[str, Any]) -> Any:
        """
        Execute crew with blackboard coordination.
        
        Args:
            inputs: Input data
            
        Returns:
            Execution results
        """
        print(f"Starting blackboard-coordinated execution for workflow {self.workflow_id}")
        
        # Pre-execution: Set up shared context
        self._setup_shared_context(inputs)
        
        # Execute with coordination
        try:
            result = self._crew.kickoff(inputs=inputs)
            
            # Post-execution: Update final results
            self._finalize_results(result)
            
            return result
            
        except Exception as e:
            print(f"Execution error in workflow {self.workflow_id}: {e}")
            raise
    
    def _setup_shared_context(self, inputs: Dict[str, Any]):
        """Set up shared context for all agents."""
        print(f"Setting up shared context for workflow {self.workflow_id}")
        
        # Store initial inputs in blackboard
        self.blackboard_system.update_agent_results(
            workflow_id=self.workflow_id,
            agent_role='system',
            results={'initial_inputs': inputs}
        )
    
    def _finalize_results(self, result: Any):
        """Finalize and store results in blackboard."""
        print(f"Finalizing results for workflow {self.workflow_id}")
        
        # Store final results
        self.blackboard_system.update_agent_results(
            workflow_id=self.workflow_id,
            agent_role='system',
            results={'final_result': result}
        )


def create_blackboard_crew(agents_config_path: str, tasks_config_path: str, selected_agents: List[str] = None) -> BlackboardMarketingResearchCrew:
    """
    Factory function to create a blackboard-integrated marketing research crew.
    
    Args:
        agents_config_path: Path to agents configuration
        tasks_config_path: Path to tasks configuration
        
    Returns:
        BlackboardMarketingResearchCrew instance
    """
    return BlackboardMarketingResearchCrew(agents_config_path, tasks_config_path, selected_agents)