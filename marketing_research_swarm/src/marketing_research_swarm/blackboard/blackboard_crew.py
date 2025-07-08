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
from crewai_tools import SerperDevTool, WebsiteSearchTool
from .integrated_blackboard import get_integrated_blackboard
from .state_aware_agents import StateAwareAgent


def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


class BlackboardMarketingResearchCrew:
    """
    CrewAI implementation that uses the integrated blackboard system
    for maximum token efficiency through shared state management.
    """
    
    def __init__(self, agents_config_path, tasks_config_path):
        self.agents_config = load_yaml(agents_config_path)
        self.tasks_config = load_yaml(tasks_config_path)
        
        # Get the integrated blackboard system
        self.blackboard = get_integrated_blackboard()
        
        # Initialize search tools (with fallback if API keys not available)
        try:
            search_tool = SerperDevTool()
        except:
            search_tool = None
        
        try:
            web_search_tool = WebsiteSearchTool()
        except:
            web_search_tool = None
        
        self.tools = {
            "read_file_tool": read_file_tool,
            "read_csv_tool": read_csv_tool,
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
        
        # Add optional tools if available
        if search_tool:
            self.tools["search"] = search_tool
        if web_search_tool:
            self.tools["web_search"] = web_search_tool
    
    def _create_blackboard_agent(self, agent_config: Dict[str, Any], workflow_id: str) -> StateAwareAgent:
        """Create a state-aware agent that uses the blackboard system."""
        agent_tools = [
            self.tools[tool_name] 
            for tool_name in agent_config.get('tools', []) 
            if tool_name in self.tools
        ]
        
        llm = self._configure_llm(agent_config)
        
        # Create state-aware agent
        state_aware_agent = StateAwareAgent(
            role=agent_config['role'],
            goal=agent_config['goal'],
            backstory=agent_config['backstory'],
            workflow_id=workflow_id,
            blackboard_system=self.blackboard,
            tools=agent_tools,
            llm=llm,
            allow_delegation=agent_config.get('allow_delegation', False)
        )
        
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
            # Create state-aware agents
            agents = [
                self._create_blackboard_agent(agent_config, workflow_id)
                for agent_config in self.agents_config.values()
            ]
            
            # Create tasks
            tasks = [
                self._create_blackboard_task(task_config, agents)
                for task_config in self.tasks_config.values()
            ]
            
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
            
            # Get workflow summary
            workflow_summary = self.blackboard.get_workflow_summary(workflow_id)
            
            return {
                'result': result,
                'workflow_id': workflow_id,
                'workflow_summary': workflow_summary
            }
            
        finally:
            # Cleanup workflow resources
            cleanup_stats = self.blackboard.cleanup_workflow(workflow_id)
            print(f"Workflow cleanup completed: {cleanup_stats}")
    
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


class BlackboardCoordinatedCrew(Crew):
    """
    Enhanced Crew that coordinates execution through the blackboard system.
    """
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # Define additional fields for Pydantic
    workflow_id: str = Field(default="", description="Workflow identifier")
    blackboard_system: Any = Field(default=None, description="Blackboard system instance")
    
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
        super().__init__(
            agents=agents, 
            tasks=tasks, 
            workflow_id=workflow_id,
            blackboard_system=blackboard_system,
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
            result = super().kickoff(inputs=inputs)
            
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


def create_blackboard_crew(agents_config_path: str, tasks_config_path: str) -> BlackboardMarketingResearchCrew:
    """
    Factory function to create a blackboard-integrated marketing research crew.
    
    Args:
        agents_config_path: Path to agents configuration
        tasks_config_path: Path to tasks configuration
        
    Returns:
        BlackboardMarketingResearchCrew instance
    """
    return BlackboardMarketingResearchCrew(agents_config_path, tasks_config_path)