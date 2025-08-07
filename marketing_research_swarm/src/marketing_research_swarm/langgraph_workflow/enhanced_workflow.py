"""
Enhanced LangGraph Workflow with Advanced Context Engineering

This module integrates the enhanced context engineering techniques from the article
with your existing LangGraph workflow to achieve optimal token usage and performance.

Key Features:
- Scratchpads for short-term memory
- Enhanced checkpointing with state persistence
- InMemoryStore for long-term memory across threads
- Context compression and isolation
- Dynamic context management per agent
"""

import os
import logging
from typing import Dict, Any, List, Optional, Literal
from datetime import datetime
import uuid

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import MarketingResearchState, WorkflowStatus, AgentStatus
from .agents import AGENT_NODES
from ..blackboard.integrated_blackboard import get_integrated_blackboard
from ..context.enhanced_context_engineering import get_enhanced_context_engineering
from ..utils.token_tracker import TokenTracker

logger = logging.getLogger(__name__)


class EnhancedMarketingWorkflow:
    """
    Enhanced LangGraph workflow implementing advanced context engineering techniques
    from the "Optimizing LangChain AI Agents with Contextual Engineering" article.
    
    This workflow provides:
    1. Scratchpads for agent short-term memory
    2. Enhanced checkpointing at each step
    3. InMemoryStore for long-term memory across threads
    4. Context compression and isolation
    5. Dynamic context management
    """
    
    def __init__(self, checkpoint_path: Optional[str] = None, context_strategy: str = "smart"):
        """
        Initialize the enhanced workflow.
        
        Args:
            checkpoint_path: Path for checkpoint storage
            context_strategy: Context optimization strategy ('smart', 'isolated', 'compressed', 'minimal')
        """
        self.checkpoint_path = checkpoint_path or "cache/enhanced_workflow_checkpoints.db"
        self.context_strategy = context_strategy
        
        # Initialize enhanced context engineering
        self.context_engine = get_enhanced_context_engineering()
        
        # Initialize checkpointer with memory saver
        self.checkpointer = MemorySaver()
        
        # Initialize other components
        self.blackboard = get_integrated_blackboard()
        self.token_tracker = TokenTracker()
        
        # Available agent types
        self.available_agents = list(AGENT_NODES.keys())
        
        # Build the enhanced workflow graph
        self.workflow = self._build_enhanced_workflow()
        
        logger.info(f"🚀 Enhanced Marketing Workflow initialized with {context_strategy} context strategy")
    
    def _build_enhanced_workflow(self) -> StateGraph:
        """Build the enhanced LangGraph workflow with context engineering."""
        
        # Create the state graph
        workflow = StateGraph(MarketingResearchState)
        
        # Add enhanced nodes
        workflow.add_node("start", self._enhanced_start_node)
        workflow.add_node("context_preparation", self._context_preparation_node)
        workflow.add_node("agent_router", self._enhanced_agent_router)
        workflow.add_node("checkpoint_manager", self._checkpoint_manager_node)
        workflow.add_node("finalize", self._enhanced_finalize_node)
        
        # Add enhanced agent nodes with context engineering
        for agent_name in self.available_agents:
            workflow.add_node(agent_name, self._create_enhanced_agent_node(agent_name))
        
        # Define enhanced edges
        workflow.set_entry_point("start")
        workflow.add_edge("start", "context_preparation")
        workflow.add_edge("context_preparation", "agent_router")
        
        # Add conditional routing from agent_router
        workflow.add_conditional_edges(
            "agent_router",
            self._route_next_enhanced_agent,
            {
                **{agent: agent for agent in self.available_agents},
                "checkpoint": "checkpoint_manager",
                "finalize": "finalize"
            }
        )
        
        # Add edges from agents to checkpoint manager
        for agent_name in self.available_agents:
            workflow.add_edge(agent_name, "checkpoint_manager")
        
        workflow.add_edge("checkpoint_manager", "agent_router")
        workflow.add_edge("finalize", END)
        
        return workflow.compile(checkpointer=self.checkpointer)
    
    def _enhanced_start_node(self, state: MarketingResearchState) -> MarketingResearchState:
        """Enhanced start node with context engineering initialization."""
        
        # Initialize workflow
        workflow_id = state.get("workflow_id") or str(uuid.uuid4())
        state["workflow_id"] = workflow_id
        state["status"] = WorkflowStatus.RUNNING
        state["updated_at"] = datetime.now()
        
        # Initialize enhanced context tracking
        state["context_strategy"] = self.context_strategy
        state["current_step"] = 0
        state["agent_steps"] = {}
        
        # Initialize agent status tracking
        if "agent_status" not in state:
            state["agent_status"] = {}
        
        for agent in state["selected_agents"]:
            state["agent_status"][agent] = AgentStatus.PENDING
            state["agent_steps"][agent] = 0
        
        # Initialize token tracking
        self.token_tracker.start_tracking(workflow_id=workflow_id)
        
        # Store initial long-term memory context
        workflow_context = {
            "workflow_id": workflow_id,
            "selected_agents": state["selected_agents"],
            "target_audience": state.get("target_audience", ""),
            "campaign_type": state.get("campaign_type", ""),
            "analysis_focus": state.get("analysis_focus", ""),
            "started_at": datetime.now().isoformat()
        }
        
        self.context_engine.store_long_term_memory(
            key=f"workflow_{workflow_id}",
            value=workflow_context,
            namespace="workflows"
        )
        
        logger.info(f"🚀 Enhanced workflow started: {workflow_id}")
        return state
    
    def _context_preparation_node(self, state: MarketingResearchState) -> MarketingResearchState:
        """Prepare context for the workflow using enhanced context engineering."""
        
        workflow_id = state["workflow_id"]
        
        # Prepare global context for all agents
        global_context = {
            "workflow_metadata": {
                "workflow_id": workflow_id,
                "workflow_type": state.get("workflow_type", "marketing_research"),
                "selected_agents": state["selected_agents"],
                "context_strategy": self.context_strategy
            },
            "input_data": {
                "target_audience": state.get("target_audience", ""),
                "campaign_type": state.get("campaign_type", ""),
                "budget": state.get("budget", 0),
                "duration": state.get("duration", ""),
                "analysis_focus": state.get("analysis_focus", "")
            },
            "execution_context": {
                "current_step": state["current_step"],
                "agent_steps": state["agent_steps"],
                "agent_status": state["agent_status"]
            }
        }
        
        # Store global context in long-term memory
        self.context_engine.store_long_term_memory(
            key=f"global_context_{workflow_id}",
            value=global_context,
            namespace="contexts"
        )
        
        # Initialize scratchpads for all agents
        for agent in state["selected_agents"]:
            self.context_engine.create_scratchpad_entry(
                agent_role=agent,
                step=0,
                content={"status": "initialized", "workflow_id": workflow_id},
                reasoning=f"Initialized scratchpad for {agent} in workflow {workflow_id}"
            )
        
        state["context_prepared"] = True
        state["global_context_key"] = f"global_context_{workflow_id}"
        
        logger.info(f"🧠 Context preparation completed for workflow {workflow_id}")
        return state
    
    def _enhanced_agent_router(self, state: MarketingResearchState) -> MarketingResearchState:
        """Enhanced agent router with context-aware routing."""
        
        # Find the next agent to execute
        next_agent = self._get_next_agent_with_context(state)
        
        if next_agent:
            logger.info(f"🎯 Routing to agent: {next_agent}")
            state["current_agent"] = next_agent
            state["agent_status"][next_agent] = AgentStatus.RUNNING
            
            # Increment step for this agent
            state["agent_steps"][next_agent] += 1
            state["current_step"] += 1
            
        else:
            logger.info("✅ All agents completed, preparing to finalize")
            state["current_agent"] = None
        
        state["updated_at"] = datetime.now()
        return state
    
    def _get_next_agent_with_context(self, state: MarketingResearchState) -> Optional[str]:
        """Get next agent considering context and dependencies."""
        
        # Get agents that haven't been executed yet
        pending_agents = [
            agent for agent in state["selected_agents"]
            if state["agent_status"].get(agent) == AgentStatus.PENDING
        ]
        
        if not pending_agents:
            return None
        
        # Enhanced dependency resolution with context awareness
        agent_dependencies = {
            "market_research_analyst": [],
            "competitive_analyst": [],
            "data_analyst": [],
            "content_strategist": ["market_research_analyst"],
            "creative_copywriter": ["content_strategist"],
            "brand_performance_specialist": ["competitive_analyst", "data_analyst"],
            "forecasting_specialist": ["market_research_analyst", "data_analyst"],
            "campaign_optimizer": ["data_analyst", "content_strategist"]
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
                # Additional context-based prioritization
                agent_priority = self._calculate_agent_priority(agent, state)
                logger.debug(f"Agent {agent} priority: {agent_priority}")
                return agent
        
        # If no dependencies are met, return the first pending agent
        return pending_agents[0] if pending_agents else None
    
    def _calculate_agent_priority(self, agent: str, state: MarketingResearchState) -> float:
        """Calculate agent priority based on context and workflow state."""
        
        base_priorities = {
            "market_research_analyst": 1.0,
            "data_analyst": 0.9,
            "competitive_analyst": 0.8,
            "forecasting_specialist": 0.7,
            "content_strategist": 0.6,
            "brand_performance_specialist": 0.5,
            "creative_copywriter": 0.4,
            "campaign_optimizer": 0.3
        }
        
        priority = base_priorities.get(agent, 0.5)
        
        # Adjust based on analysis focus
        analysis_focus = state.get("analysis_focus", "").lower()
        if "data" in analysis_focus and agent == "data_analyst":
            priority += 0.2
        elif "competitive" in analysis_focus and agent == "competitive_analyst":
            priority += 0.2
        elif "content" in analysis_focus and agent == "content_strategist":
            priority += 0.2
        
        return priority
    
    def _create_enhanced_agent_node(self, agent_name: str):
        """Create an enhanced agent node with context engineering."""
        
        def enhanced_agent_node(state: MarketingResearchState) -> MarketingResearchState:
            """Enhanced agent execution with context engineering."""
            
            try:
                workflow_id = state["workflow_id"]
                current_step = state["agent_steps"][agent_name]
                
                logger.info(f"🤖 Executing {agent_name} (step {current_step})")
                
                # Get global context from long-term memory
                global_context_key = state.get("global_context_key")
                global_context = self.context_engine.retrieve_long_term_memory(
                    key=global_context_key.split("_", 2)[2],  # Extract workflow_id
                    namespace="contexts"
                ) if global_context_key else {}
                
                # Get optimized context for this agent
                optimized_context = self.context_engine.get_context_for_agent(
                    agent_role=agent_name,
                    thread_id=workflow_id,
                    step=current_step,
                    full_context={**state, **global_context},
                    strategy=self.context_strategy
                )
                
                # Create scratchpad entry for this execution step
                self.context_engine.create_scratchpad_entry(
                    agent_role=agent_name,
                    step=current_step,
                    content={
                        "action": "starting_execution",
                        "context_size": len(str(optimized_context)),
                        "strategy": self.context_strategy
                    },
                    reasoning=f"Starting execution of {agent_name} with {self.context_strategy} context strategy"
                )
                
                # Track tokens before execution
                tokens_before = self.token_tracker.get_current_usage()
                
                # Execute the original agent with optimized context
                agent_node = AGENT_NODES[agent_name]
                
                # Create a modified state with optimized context
                enhanced_state = state.copy()
                enhanced_state["optimized_context"] = optimized_context
                enhanced_state["scratchpad"] = self.context_engine.get_scratchpad_context(agent_name)
                
                # Execute the agent
                result_state = agent_node(enhanced_state)
                
                # Track tokens after execution
                tokens_after = self.token_tracker.get_current_usage()
                agent_tokens = tokens_after - tokens_before
                
                # Update state with results
                if agent_name in result_state.get("agent_results", {}):
                    agent_result = result_state["agent_results"][agent_name]
                    
                    # Store result in state
                    if "agent_results" not in state:
                        state["agent_results"] = {}
                    state["agent_results"][agent_name] = agent_result
                    
                    # Create scratchpad entry for completion
                    self.context_engine.create_scratchpad_entry(
                        agent_role=agent_name,
                        step=current_step,
                        content={
                            "action": "completed_execution",
                            "result_size": len(str(agent_result)),
                            "tokens_used": agent_tokens
                        },
                        reasoning=f"Completed execution of {agent_name} with {agent_tokens} tokens used"
                    )
                    
                    # Update agent's long-term memory with insights
                    if isinstance(agent_result, dict) and "insights" in agent_result:
                        self.context_engine.update_agent_memory(
                            agent_role=agent_name,
                            new_insights={
                                "latest_insights": agent_result["insights"],
                                "execution_step": current_step,
                                "tokens_used": agent_tokens,
                                "timestamp": datetime.now().isoformat()
                            }
                        )
                
                # Update agent status
                state["agent_status"][agent_name] = AgentStatus.COMPLETED
                if "agent_execution_order" not in state:
                    state["agent_execution_order"] = []
                state["agent_execution_order"].append(agent_name)
                
                logger.info(f"✅ {agent_name} completed successfully (tokens: {agent_tokens})")
                
            except Exception as e:
                logger.error(f"❌ {agent_name} failed: {e}")
                state["agent_status"][agent_name] = AgentStatus.FAILED
                
                # Create scratchpad entry for failure
                self.context_engine.create_scratchpad_entry(
                    agent_role=agent_name,
                    step=state["agent_steps"][agent_name],
                    content={
                        "action": "execution_failed",
                        "error": str(e)
                    },
                    reasoning=f"Execution of {agent_name} failed: {str(e)}"
                )
                
                if "agent_errors" not in state:
                    state["agent_errors"] = {}
                state["agent_errors"][agent_name] = str(e)
            
            return state
        
        return enhanced_agent_node
    
    def _checkpoint_manager_node(self, state: MarketingResearchState) -> MarketingResearchState:
        """Manage checkpoints after each agent execution."""
        
        workflow_id = state["workflow_id"]
        current_agent = state.get("current_agent")
        
        if current_agent and state["agent_status"].get(current_agent) in [AgentStatus.COMPLETED, AgentStatus.FAILED]:
            # Create checkpoint after agent execution
            token_usage = self.token_tracker.get_usage_summary()
            
            checkpoint = self.context_engine.create_checkpoint(
                thread_id=workflow_id,
                agent_role=current_agent,
                step=state["agent_steps"][current_agent],
                state=state,
                token_usage=token_usage
            )
            
            # Store checkpoint reference in state
            if "checkpoints" not in state:
                state["checkpoints"] = []
            state["checkpoints"].append(checkpoint.checkpoint_id)
            
            logger.info(f"💾 Created checkpoint after {current_agent} execution")
        
        return state
    
    def _route_next_enhanced_agent(self, state: MarketingResearchState) -> str:
        """Route to next agent or finalization with enhanced logic."""
        
        # Check if we have a current agent to checkpoint
        current_agent = state.get("current_agent")
        if current_agent and state["agent_status"].get(current_agent) in [AgentStatus.COMPLETED, AgentStatus.FAILED]:
            return "checkpoint"
        
        # Check if all agents are done
        pending_agents = [
            agent for agent in state["selected_agents"]
            if state["agent_status"].get(agent) == AgentStatus.PENDING
        ]
        
        if not pending_agents:
            return "finalize"
        
        # Continue routing
        return "agent_router"
    
    def _enhanced_finalize_node(self, state: MarketingResearchState) -> MarketingResearchState:
        """Enhanced finalization with context engineering summary."""
        
        workflow_id = state["workflow_id"]
        
        # Update workflow status
        state["status"] = WorkflowStatus.COMPLETED
        state["updated_at"] = datetime.now()
        state["completed_at"] = datetime.now()
        
        # Generate enhanced summary with context engineering metrics
        context_stats = self.context_engine.get_system_stats()
        token_usage = self.token_tracker.get_usage_summary()
        
        enhanced_summary = {
            "workflow_id": workflow_id,
            "execution_summary": self._generate_execution_summary(state),
            "context_engineering_stats": context_stats,
            "token_usage": token_usage,
            "optimization_metrics": {
                "context_strategy": self.context_strategy,
                "total_steps": state["current_step"],
                "agents_executed": len([a for a in state["selected_agents"] 
                                      if state["agent_status"].get(a) == AgentStatus.COMPLETED]),
                "checkpoints_created": len(state.get("checkpoints", [])),
                "scratchpad_entries": sum(
                    len(self.context_engine.scratchpads.get(agent, []))
                    for agent in state["selected_agents"]
                )
            }
        }
        
        state["enhanced_summary"] = enhanced_summary
        
        # Store final workflow state in long-term memory
        self.context_engine.store_long_term_memory(
            key=f"completed_workflow_{workflow_id}",
            value={
                "summary": enhanced_summary,
                "final_state": state,
                "completed_at": datetime.now().isoformat()
            },
            namespace="completed_workflows"
        )
        
        # Stop token tracking
        self.token_tracker.stop_tracking(workflow_id)
        
        logger.info(f"🎉 Enhanced workflow completed: {workflow_id}")
        logger.info(f"📊 Context engineering stats: {context_stats}")
        
        return state
    
    def _generate_execution_summary(self, state: MarketingResearchState) -> Dict[str, Any]:
        """Generate execution summary."""
        
        completed_agents = [
            agent for agent in state["selected_agents"]
            if state["agent_status"].get(agent) == AgentStatus.COMPLETED
        ]
        
        failed_agents = [
            agent for agent in state["selected_agents"]
            if state["agent_status"].get(agent) == AgentStatus.FAILED
        ]
        
        return {
            "total_agents": len(state["selected_agents"]),
            "completed_agents": len(completed_agents),
            "failed_agents": len(failed_agents),
            "success_rate": len(completed_agents) / len(state["selected_agents"]) if state["selected_agents"] else 0,
            "execution_time": (state["updated_at"] - state["created_at"]).total_seconds(),
            "agent_execution_order": state.get("agent_execution_order", [])
        }
    
    def execute_enhanced_workflow(
        self,
        selected_agents: List[str],
        target_audience: str,
        campaign_type: str,
        budget: float,
        duration: str,
        analysis_focus: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute the enhanced workflow with context engineering."""
        
        try:
            # Create initial state
            initial_state = {
                "workflow_id": str(uuid.uuid4()),
                "workflow_type": "enhanced_marketing_research",
                "status": WorkflowStatus.PENDING,
                "selected_agents": selected_agents,
                "target_audience": target_audience,
                "campaign_type": campaign_type,
                "budget": budget,
                "duration": duration,
                "analysis_focus": analysis_focus,
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
                **kwargs
            }
            
            logger.info(f"🚀 Starting enhanced workflow: {initial_state['workflow_id']}")
            
            # Execute the workflow
            config = {"configurable": {"thread_id": initial_state["workflow_id"]}}
            final_state = self.workflow.invoke(initial_state, config=config)
            
            # Return comprehensive result
            return {
                "success": True,
                "workflow_id": final_state["workflow_id"],
                "status": final_state["status"].value,
                "final_state": final_state,
                "enhanced_summary": final_state.get("enhanced_summary", {}),
                "agent_results": final_state.get("agent_results", {}),
                "execution_time": (final_state["updated_at"] - final_state["created_at"]).total_seconds(),
                "context_engineering_applied": True,
                "context_strategy": self.context_strategy
            }
            
        except Exception as e:
            logger.error(f"❌ Enhanced workflow execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "workflow_id": initial_state.get("workflow_id") if 'initial_state' in locals() else None,
                "context_engineering_applied": True
            }
    
    def restore_workflow_from_checkpoint(self, thread_id: str, checkpoint_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Restore workflow from a checkpoint."""
        
        checkpoint = self.context_engine.restore_from_checkpoint(thread_id, checkpoint_id)
        if checkpoint:
            logger.info(f"🔄 Restored workflow from checkpoint: {checkpoint.checkpoint_id}")
            return {
                "checkpoint_id": checkpoint.checkpoint_id,
                "thread_id": checkpoint.thread_id,
                "agent_role": checkpoint.agent_role,
                "step": checkpoint.step,
                "state": checkpoint.state,
                "timestamp": checkpoint.timestamp.isoformat(),
                "token_usage": checkpoint.token_usage
            }
        
        return None
    
    def get_workflow_context_stats(self, workflow_id: str) -> Dict[str, Any]:
        """Get context engineering statistics for a specific workflow."""
        
        # Get workflow context from long-term memory
        workflow_context = self.context_engine.retrieve_long_term_memory(
            key=f"workflow_{workflow_id}",
            namespace="workflows"
        )
        
        if not workflow_context:
            return {"error": "Workflow not found"}
        
        # Get system stats
        system_stats = self.context_engine.get_system_stats()
        
        # Get workflow-specific stats
        workflow_agents = workflow_context.get("selected_agents", [])
        workflow_scratchpad_stats = {
            agent: len(self.context_engine.scratchpads.get(agent, []))
            for agent in workflow_agents
        }
        
        return {
            "workflow_id": workflow_id,
            "workflow_context": workflow_context,
            "scratchpad_stats": workflow_scratchpad_stats,
            "system_stats": system_stats,
            "context_strategy": self.context_strategy
        }

# Global instance
_global_enhanced_workflow = None

def get_enhanced_workflow(context_strategy: str = "smart") -> EnhancedMarketingWorkflow:
    """Get the global enhanced workflow instance."""
    global _global_enhanced_workflow
    if _global_enhanced_workflow is None:
        _global_enhanced_workflow = EnhancedMarketingWorkflow(context_strategy=context_strategy)
    return _global_enhanced_workflow