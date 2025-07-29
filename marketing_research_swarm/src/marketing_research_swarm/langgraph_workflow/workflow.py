"""
LangGraph Workflow Implementation for Marketing Research

This module implements the main LangGraph workflow that replaces the CrewAI
crew-based approach with a dynamic, state-managed workflow system.
"""

import os
import logging
from typing import Dict, Any, List, Optional, Literal
from datetime import datetime
import uuid

from langgraph.graph import StateGraph, END
# SqliteSaver temporarily disabled - using in-memory state for now
# from langgraph_checkpoint.sqlite import SqliteSaver

from .state import MarketingResearchState, WorkflowStatus, AgentStatus
from .agents import AGENT_NODES
from ..blackboard.integrated_blackboard import get_global_blackboard

logger = logging.getLogger(__name__)


class MarketingResearchWorkflow:
    """
    LangGraph-based marketing research workflow that dynamically selects
    and executes agents based on configuration and user selection.
    """
    
    def __init__(self, checkpoint_path: Optional[str] = None):
        """Initialize the workflow with optional checkpointing."""
        self.checkpoint_path = checkpoint_path or "cache/workflow_checkpoints.db"
        # self.checkpointer = SqliteSaver.from_conn_string(self.checkpoint_path)
        self.checkpointer = None
        self.blackboard = get_global_blackboard()
        
        # Available agent types
        self.available_agents = list(AGENT_NODES.keys())
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
        
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow with dynamic agent selection."""
        
        # Create the state graph
        workflow = StateGraph(MarketingResearchState)
        
        # Add the start node
        workflow.add_node("start", self._start_node)
        
        # Add all available agent nodes
        for agent_name, agent_function in AGENT_NODES.items():
            workflow.add_node(agent_name, agent_function)
        
        # Add routing and completion nodes
        workflow.add_node("route_next_agent", self._route_next_agent)
        workflow.add_node("finalize_results", self._finalize_results)
        
        # Set entry point
        workflow.set_entry_point("start")
        
        # Add edges from start to routing
        workflow.add_edge("start", "route_next_agent")
        
        # Add conditional edges from routing to agents
        workflow.add_conditional_edges(
            "route_next_agent",
            self._should_continue,
            {
                "continue": "route_next_agent",
                "finalize": "finalize_results",
                **{agent: agent for agent in self.available_agents}
            }
        )
        
        # Add edges from agents back to routing
        for agent_name in self.available_agents:
            workflow.add_edge(agent_name, "route_next_agent")
        
        # Add edge from finalize to end
        workflow.add_edge("finalize_results", END)
        
        return workflow.compile(checkpointer=self.checkpointer)
    
    def _start_node(self, state: MarketingResearchState) -> MarketingResearchState:
        """Initialize the workflow state."""
        logger.info(f"Starting workflow {state['workflow_id']}")
        
        # Update workflow status
        state["status"] = WorkflowStatus.RUNNING
        state["updated_at"] = datetime.now()
        
        # Initialize agent execution tracking
        if "agent_execution_order" not in state:
            state["agent_execution_order"] = []
        
        if "agent_results" not in state:
            state["agent_results"] = {}
        
        if "agent_status" not in state:
            state["agent_status"] = {}
        
        # Set all selected agents to pending
        for agent in state["selected_agents"]:
            state["agent_status"][agent] = AgentStatus.PENDING
        
        # Store initial state in blackboard
        self.blackboard.store_workflow_state(state["workflow_id"], state)
        
        logger.info(f"Workflow initialized with agents: {state['selected_agents']}")
        return state
    
    def _route_next_agent(self, state: MarketingResearchState) -> MarketingResearchState:
        """Route to the next agent that should be executed."""
        
        # Find the next agent to execute
        next_agent = self._get_next_agent(state)
        
        if next_agent:
            logger.info(f"Routing to agent: {next_agent}")
            state["current_agent"] = next_agent
            state["agent_status"][next_agent] = AgentStatus.RUNNING
        else:
            logger.info("No more agents to execute, preparing to finalize")
            state["current_agent"] = None
        
        state["updated_at"] = datetime.now()
        return state
    
    def _get_next_agent(self, state: MarketingResearchState) -> Optional[str]:
        """Determine the next agent to execute based on dependencies and status."""
        
        # Get agents that haven't been executed yet
        pending_agents = [
            agent for agent in state["selected_agents"]
            if state["agent_status"].get(agent) == AgentStatus.PENDING
        ]
        
        if not pending_agents:
            return None
        
        # Define agent dependencies and execution order
        agent_dependencies = {
            "market_research_analyst": [],  # Can run first
            "competitive_analyst": [],  # Can run first
            "data_analyst": [],  # Can run first
            "content_strategist": ["market_research_analyst"],  # Needs market research
            "creative_copywriter": ["content_strategist"],  # Needs content strategy
            "brand_performance_specialist": ["competitive_analyst", "data_analyst"],  # Needs competitive and data analysis
            "forecasting_specialist": ["market_research_analyst", "data_analyst"],  # Needs market research and data analysis
            "campaign_optimizer": ["data_analyst", "content_strategist"],  # Needs data and content strategy
        }
        
        # Find agents whose dependencies are satisfied
        for agent in pending_agents:
            dependencies = agent_dependencies.get(agent, [])
            
            # Check if all dependencies are completed
            dependencies_met = all(
                state["agent_status"].get(dep) == AgentStatus.COMPLETED
                for dep in dependencies
                if dep in state["selected_agents"]
            )
            
            if dependencies_met:
                return agent
        
        # If no dependencies are met, return the first pending agent
        # (this handles cases where dependencies aren't selected)
        return pending_agents[0] if pending_agents else None
    
    def _should_continue(self, state: MarketingResearchState) -> Literal["continue", "finalize"] | str:
        """Determine whether to continue with another agent or finalize."""
        
        current_agent = state.get("current_agent")
        
        # If we just routed and have a current agent, execute it
        if current_agent and current_agent in self.available_agents:
            return current_agent
        
        # Check if there are more agents to execute
        pending_agents = [
            agent for agent in state["selected_agents"]
            if state["agent_status"].get(agent) == AgentStatus.PENDING
        ]
        
        if pending_agents:
            return "continue"
        else:
            return "finalize"
    
    def _finalize_results(self, state: MarketingResearchState) -> MarketingResearchState:
        """Finalize the workflow results and update status."""
        logger.info(f"Finalizing workflow {state['workflow_id']}")
        
        # Update workflow status
        state["status"] = WorkflowStatus.COMPLETED
        state["updated_at"] = datetime.now()
        state["completed_at"] = datetime.now()
        
        # Generate final summary
        state["final_summary"] = self._generate_final_summary(state)
        
        # Store final state in blackboard
        self.blackboard.store_workflow_state(state["workflow_id"], state)
        
        logger.info(f"Workflow {state['workflow_id']} completed successfully")
        return state
    
    def _generate_final_summary(self, state: MarketingResearchState) -> Dict[str, Any]:
        """Generate a summary of the workflow execution."""
        
        completed_agents = [
            agent for agent in state["selected_agents"]
            if state["agent_status"].get(agent) == AgentStatus.COMPLETED
        ]
        
        failed_agents = [
            agent for agent in state["selected_agents"]
            if state["agent_status"].get(agent) == AgentStatus.FAILED
        ]
        
        summary = {
            "workflow_id": state["workflow_id"],
            "workflow_type": state["workflow_type"],
            "execution_time": (state["updated_at"] - state["created_at"]).total_seconds(),
            "total_agents": len(state["selected_agents"]),
            "completed_agents": len(completed_agents),
            "failed_agents": len(failed_agents),
            "success_rate": len(completed_agents) / len(state["selected_agents"]) if state["selected_agents"] else 0,
            "agent_execution_order": state["agent_execution_order"],
            "key_insights": self._extract_key_insights(state),
            "recommendations": self._extract_recommendations(state)
        }
        
        return summary
    
    def _extract_key_insights(self, state: MarketingResearchState) -> List[str]:
        """Extract key insights from agent results."""
        insights = []
        
        for agent, result in state.get("agent_results", {}).items():
            if isinstance(result, dict) and "analysis" in result:
                # Extract key points from analysis (simplified)
                analysis = result["analysis"]
                if "key insights" in analysis.lower():
                    insights.append(f"{agent}: Key insights identified")
                if "recommendation" in analysis.lower():
                    insights.append(f"{agent}: Strategic recommendations provided")
        
        return insights
    
    def _extract_recommendations(self, state: MarketingResearchState) -> List[str]:
        """Extract recommendations from agent results."""
        recommendations = []
        
        # Extract recommendations from campaign optimizer
        if "campaign_optimizer" in state.get("agent_results", {}):
            recommendations.append("Budget allocation strategy developed")
            recommendations.append("ROI optimization recommendations provided")
        
        # Extract recommendations from content strategist
        if "content_strategist" in state.get("agent_results", {}):
            recommendations.append("Content strategy framework created")
            recommendations.append("Channel-specific recommendations provided")
        
        return recommendations
    
    def create_initial_state(
        self,
        selected_agents: List[str],
        target_audience: str,
        campaign_type: str,
        budget: float,
        duration: str,
        analysis_focus: str,
        **kwargs
    ) -> MarketingResearchState:
        """Create the initial state for a new workflow execution."""
        
        workflow_id = str(uuid.uuid4())
        current_time = datetime.now()
        
        # Validate selected agents
        invalid_agents = [agent for agent in selected_agents if agent not in self.available_agents]
        if invalid_agents:
            raise ValueError(f"Invalid agents selected: {invalid_agents}")
        
        state = MarketingResearchState(
            # Workflow metadata
            workflow_id=workflow_id,
            workflow_type=kwargs.get("workflow_type", "marketing_research"),
            status=WorkflowStatus.PENDING,
            created_at=current_time,
            updated_at=current_time,
            
            # Input data
            initial_inputs=kwargs,
            target_audience=target_audience,
            campaign_type=campaign_type,
            budget=budget,
            duration=duration,
            analysis_focus=analysis_focus,
            business_objective=kwargs.get("business_objective", ""),
            
            # Agent selection and execution
            selected_agents=selected_agents,
            agent_execution_order=[],
            current_agent=None,
            
            # Results and state
            agent_results={},
            agent_status={},
            shared_context={},
            
            # Configuration
            market_segments=kwargs.get("market_segments", ["premium", "mass_market"]),
            product_categories=kwargs.get("product_categories", ["soft_drinks", "energy_drinks", "water"]),
            brands=kwargs.get("brands", ["Brand_A", "Brand_B", "Brand_C"]),
            regions=kwargs.get("regions", ["North", "South", "East", "West"]),
            key_metrics=kwargs.get("key_metrics", ["revenue", "profit_margin", "market_share"]),
            campaign_goals=kwargs.get("campaign_goals", ["increase_awareness", "drive_sales"]),
            competitive_landscape=kwargs.get("competitive_landscape", "highly_competitive"),
            
            # Optional fields
            data_file_path=kwargs.get("data_file_path", "data/beverage_sales_data.csv"),
            forecast_periods=kwargs.get("forecast_periods", 30),
        )
        
        return state
    
    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run the workflow with input dictionary (for compatibility)."""
        return self.execute_workflow(
            selected_agents=inputs.get("selected_agents", ["market_research_analyst", "data_analyst"]),
            target_audience=inputs.get("target_audience", "general audience"),
            campaign_type=inputs.get("campaign_type", "digital marketing"),
            budget=inputs.get("budget", 50000),
            duration=inputs.get("duration", "3 months"),
            analysis_focus=inputs.get("analysis_focus", "marketing analysis"),
            **{k: v for k, v in inputs.items() if k not in [
                "selected_agents", "target_audience", "campaign_type", 
                "budget", "duration", "analysis_focus"
            ]}
        )
    
    def execute_workflow(
        self,
        selected_agents: List[str],
        target_audience: str,
        campaign_type: str,
        budget: float,
        duration: str,
        analysis_focus: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute the complete workflow with the given parameters."""
        
        try:
            # Create initial state
            initial_state = self.create_initial_state(
                selected_agents=selected_agents,
                target_audience=target_audience,
                campaign_type=campaign_type,
                budget=budget,
                duration=duration,
                analysis_focus=analysis_focus,
                **kwargs
            )
            
            logger.info(f"Starting workflow execution: {initial_state['workflow_id']}")
            
            # Execute the workflow
            config = {"configurable": {"thread_id": initial_state["workflow_id"]}}
            final_state = self.workflow.invoke(initial_state, config=config)
            
            # Return the final state and summary
            return {
                "success": True,
                "workflow_id": final_state["workflow_id"],
                "status": final_state["status"].value,
                "final_state": final_state,
                "summary": final_state.get("final_summary", {}),
                "agent_results": final_state.get("agent_results", {}),
                "execution_time": (final_state["updated_at"] - final_state["created_at"]).total_seconds()
            }
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "workflow_id": initial_state.get("workflow_id") if 'initial_state' in locals() else None
            }
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get the current status of a workflow."""
        try:
            state = self.blackboard.get_workflow_state(workflow_id)
            if state:
                return {
                    "workflow_id": workflow_id,
                    "status": state["status"].value,
                    "progress": self._calculate_progress(state),
                    "current_agent": state.get("current_agent"),
                    "completed_agents": [
                        agent for agent in state["selected_agents"]
                        if state["agent_status"].get(agent) == AgentStatus.COMPLETED
                    ],
                    "updated_at": state["updated_at"].isoformat()
                }
            return None
        except Exception as e:
            logger.error(f"Failed to get workflow status: {e}")
            return None
    
    def _calculate_progress(self, state: MarketingResearchState) -> float:
        """Calculate workflow progress as a percentage."""
        if not state["selected_agents"]:
            return 0.0
        
        completed_count = sum(
            1 for agent in state["selected_agents"]
            if state["agent_status"].get(agent) == AgentStatus.COMPLETED
        )
        
        return (completed_count / len(state["selected_agents"])) * 100