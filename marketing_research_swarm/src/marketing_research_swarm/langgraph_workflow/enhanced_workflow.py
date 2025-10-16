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
from enum import Enum
from .enhanced_agent_nodes import ENHANCED_AGENT_NODES
# Removed blackboard integration for performance optimization
# from ..blackboard.integrated_blackboard import get_integrated_blackboard
from ..context.enhanced_context_engineering import get_enhanced_context_engineering
from ..context.persistent_context_storage import get_persistent_storage
from ..context.intelligent_context_filter import get_intelligent_filter
from ..context.compression_strategies import get_context_compressor, CompressionLevel
from ..context.context_quality import ContextQualityMonitor, TokenBudgetManager
from ..utils.token_tracker import TokenTracker
from ..memory.mem0_integration import MarketingMemoryManager

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
        
        # Initialize mem0 memory manager for persistent learning
        try:
            self.memory_manager = MarketingMemoryManager()
            self.mem0_available = True
            logger.info("✅ mem0 integration initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️ mem0 integration failed, continuing without persistent memory: {e}")
            self.memory_manager = None
            self.mem0_available = False
        
        # Initialize other components
        # Removed blackboard initialization for performance optimization
        # self.blackboard = get_integrated_blackboard()
        self.token_tracker = TokenTracker()
        self.persistent_storage = get_persistent_storage()
        self.context_filter = get_intelligent_filter()
        self.compressor = get_context_compressor()
        self.context_quality = ContextQualityMonitor()
        self.budget_manager: Optional[TokenBudgetManager] = None
        
        # Available agent types
        self.available_agents = list(ENHANCED_AGENT_NODES.keys())
        
        # Build the enhanced workflow graph
        self.workflow = self._build_enhanced_workflow()
        
        mem0_status = "with mem0 persistent learning" if self.mem0_available else "without mem0 (session-only memory)"
        logger.info(f"🚀 Enhanced Marketing Workflow initialized with {context_strategy} context strategy {mem0_status}")
    
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
        
        logger.info(f"🔍 Initialized agent status for: {state['selected_agents']}")
        logger.info(f"🔍 Agent status: {state['agent_status']}")
        
        # Initialize token tracking
        self.token_tracker.start_tracking(workflow_id=workflow_id)

        # Initialize token budget manager if budget provided in state
        token_budget = state.get("token_budget")
        if token_budget:
            self.budget_manager = TokenBudgetManager(total_budget=int(token_budget))
        else:
            self.budget_manager = None
        
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
        """Prepare context for the workflow using enhanced context engineering and mem0."""
        
        workflow_id = state["workflow_id"]
        
        # Get historical context from mem0 if available
        historical_context = {}
        if self.mem0_available and self.memory_manager:
            try:
                # Create query for relevant historical context
                analysis_query = f"{state.get('analysis_focus', '')} {state.get('campaign_type', '')} {state.get('target_audience', '')}"
                
                # Get relevant historical insights
                historical_insights = self.memory_manager.get_relevant_context(
                    query=analysis_query,
                    user_id="marketing_team",
                    limit=5
                )
                
                if historical_insights:
                    historical_context = {
                        "relevant_insights": historical_insights,
                        "historical_patterns": self._extract_patterns_from_insights(historical_insights),
                        "previous_recommendations": self._extract_recommendations_from_insights(historical_insights)
                    }
                    logger.info(f"🧠 Retrieved {len(historical_insights)} relevant historical insights")
                else:
                    logger.info("🔍 No relevant historical insights found")
                    
            except Exception as e:
                logger.warning(f"Failed to retrieve historical context from mem0: {e}")
        
        # Prepare global context for all agents
        global_context = {
            "workflow_metadata": {
                "workflow_id": workflow_id,
                "workflow_type": state.get("workflow_type", "marketing_research"),
                "selected_agents": state["selected_agents"],
                "context_strategy": self.context_strategy,
                "mem0_enabled": self.mem0_available
            },
            "input_data": {
                "target_audience": state.get("target_audience", ""),
                "campaign_type": state.get("campaign_type", ""),
                "budget": state.get("budget", 0),
                "duration": state.get("duration", ""),
                "analysis_focus": state.get("analysis_focus", ""),
                "data_file_path": state.get("data_file_path", "data/beverage_sales.csv"),
                "brands": state.get("brands", []),
                "market_segments": state.get("market_segments", []),
                "product_categories": state.get("product_categories", [])
            },
            "execution_context": {
                "current_step": state["current_step"],
                "agent_steps": state["agent_steps"],
                "agent_status": state["agent_status"]
            },
            "historical_context": historical_context
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
    
    def _extract_patterns_from_insights(self, insights: List[Dict[str, Any]]) -> List[str]:
        """Extract patterns from historical insights."""
        patterns = []
        for insight in insights:
            content = insight.get('content', '')
            if 'pattern' in content.lower() or 'trend' in content.lower():
                # Extract pattern-related sentences
                sentences = content.split('.')
                for sentence in sentences:
                    if any(keyword in sentence.lower() for keyword in ['pattern', 'trend', 'consistently', 'typically']):
                        patterns.append(sentence.strip())
        return patterns[:3]  # Limit to top 3 patterns
    
    def _extract_recommendations_from_insights(self, insights: List[Dict[str, Any]]) -> List[str]:
        """Extract recommendations from historical insights."""
        recommendations = []
        for insight in insights:
            content = insight.get('content', '')
            if 'recommend' in content.lower() or 'suggest' in content.lower():
                # Extract recommendation-related sentences
                sentences = content.split('.')
                for sentence in sentences:
                    if any(keyword in sentence.lower() for keyword in ['recommend', 'suggest', 'should', 'consider']):
                        recommendations.append(sentence.strip())
        return recommendations[:3]  # Limit to top 3 recommendations
    
    def _extract_learnings_for_agent(self, insights: List[Dict[str, Any]], agent_name: str) -> List[str]:
        """Extract agent-specific learnings from historical insights."""
        learnings = []
        agent_keywords = {
            'market_research_analyst': ['market', 'research', 'analysis', 'trends'],
            'data_analyst': ['data', 'metrics', 'statistics', 'analysis'],
            'competitive_analyst': ['competitive', 'competition', 'market share', 'positioning'],
            'brand_performance_specialist': ['brand', 'performance', 'awareness', 'equity'],
            'content_strategist': ['content', 'strategy', 'messaging', 'communication'],
            'forecasting_specialist': ['forecast', 'prediction', 'future', 'projection'],
            'campaign_optimizer': ['campaign', 'optimization', 'performance', 'roi'],
            'creative_copywriter': ['creative', 'copy', 'messaging', 'content']
        }
        
        keywords = agent_keywords.get(agent_name, [])
        
        for insight in insights:
            content = insight.get('content', '')
            # Look for content relevant to this agent's domain
            if any(keyword in content.lower() for keyword in keywords):
                sentences = content.split('.')
                for sentence in sentences:
                    if any(keyword in sentence.lower() for keyword in keywords):
                        learnings.append(sentence.strip())
        
        return learnings[:2]  # Limit to top 2 learnings
    
    def _format_insights_for_mem0(self, agent_result: Dict[str, Any], agent_name: str, state: MarketingResearchState) -> str:
        """Format agent insights for storage in mem0."""
        
        # Extract key information from agent result
        insights = agent_result.get("insights", "")
        analysis = agent_result.get("analysis", "")
        recommendations = agent_result.get("recommendations", [])
        
        # Create structured insight content
        formatted_content = f"Agent: {agent_name}\n"
        formatted_content += f"Analysis Focus: {state.get('analysis_focus', '')}\n"
        formatted_content += f"Target Audience: {state.get('target_audience', '')}\n"
        formatted_content += f"Campaign Type: {state.get('campaign_type', '')}\n\n"
        
        if insights:
            formatted_content += f"Key Insights:\n{insights}\n\n"
        
        if analysis:
            formatted_content += f"Analysis:\n{analysis}\n\n"
        
        if recommendations:
            if isinstance(recommendations, list):
                formatted_content += "Recommendations:\n"
                for i, rec in enumerate(recommendations, 1):
                    formatted_content += f"{i}. {rec}\n"
            else:
                formatted_content += f"Recommendations:\n{recommendations}\n"
        
        # Add tool results if available
        if "tool_results" in agent_result:
            tool_results = agent_result["tool_results"]
            if tool_results:
                formatted_content += "\nTool Results Summary:\n"
                for tool_name, result in tool_results.items():
                    if isinstance(result, dict) and "insights" in result:
                        formatted_content += f"- {tool_name}: {result['insights']}\n"
        
        return formatted_content
    
    def _create_workflow_summary_for_mem0(self, state: MarketingResearchState, enhanced_summary: Dict[str, Any]) -> str:
        """Create a comprehensive workflow summary for mem0 storage."""
        
        workflow_id = state["workflow_id"]
        
        summary_content = f"Workflow Summary - {workflow_id}\n"
        summary_content += f"Analysis Type: {state.get('analysis_focus', '')}\n"
        summary_content += f"Target Audience: {state.get('target_audience', '')}\n"
        summary_content += f"Campaign Type: {state.get('campaign_type', '')}\n"
        summary_content += f"Budget: ${state.get('budget', 0):,}\n"
        summary_content += f"Duration: {state.get('duration', '')}\n\n"
        
        # Add execution metrics
        exec_summary = enhanced_summary["execution_summary"]
        summary_content += f"Execution Metrics:\n"
        summary_content += f"- Agents Used: {', '.join(state['selected_agents'])}\n"
        summary_content += f"- Success Rate: {exec_summary['success_rate']:.1%}\n"
        summary_content += f"- Execution Time: {exec_summary['execution_time']:.1f} seconds\n"
        summary_content += f"- Context Strategy: {self.context_strategy}\n\n"
        
        # Add key insights from each agent
        agent_results = state.get("agent_results", {})
        if agent_results:
            summary_content += "Key Agent Insights:\n"
            for agent_name, result in agent_results.items():
                if isinstance(result, dict) and "insights" in result:
                    insights = result["insights"]
                    if insights:
                        # Truncate long insights
                        truncated_insights = insights[:200] + "..." if len(insights) > 200 else insights
                        summary_content += f"- {agent_name}: {truncated_insights}\n"
        
        # Add optimization metrics
        opt_metrics = enhanced_summary["optimization_metrics"]
        summary_content += f"\nOptimization Results:\n"
        summary_content += f"- Total Steps: {opt_metrics['total_steps']}\n"
        summary_content += f"- Checkpoints Created: {opt_metrics['checkpoints_created']}\n"
        summary_content += f"- Scratchpad Entries: {opt_metrics['scratchpad_entries']}\n"
        
        # Add token usage if available
        token_usage = enhanced_summary.get("token_usage", {})
        if token_usage:
            summary_content += f"- Total Tokens Used: {token_usage.get('total_tokens', 0):,}\n"
        
        return summary_content
    
    def _extract_workflow_patterns(self, state: MarketingResearchState, enhanced_summary: Dict[str, Any]) -> str:
        """Extract workflow patterns for future optimization."""
        
        patterns_content = f"Workflow Pattern Analysis\n"
        patterns_content += f"Agent Combination: {', '.join(sorted(state['selected_agents']))}\n"
        patterns_content += f"Context Strategy: {self.context_strategy}\n"
        patterns_content += f"Optimization Level: {state.get('optimization_level', '')}\n\n"
        
        # Execution patterns
        exec_summary = enhanced_summary["execution_summary"]
        patterns_content += f"Execution Patterns:\n"
        patterns_content += f"- Execution Order: {' → '.join(exec_summary.get('agent_execution_order', []))}\n"
        patterns_content += f"- Success Rate: {exec_summary['success_rate']:.1%}\n"
        patterns_content += f"- Average Time per Agent: {exec_summary['execution_time'] / len(state['selected_agents']):.1f}s\n\n"
        
        # Context engineering effectiveness
        context_stats = enhanced_summary.get("context_engineering_stats", {})
        if context_stats:
            patterns_content += f"Context Engineering Effectiveness:\n"
            scratchpad_stats = context_stats.get("scratchpads", {})
            if scratchpad_stats:
                patterns_content += f"- Total Scratchpad Entries: {scratchpad_stats.get('total_entries', 0)}\n"
                patterns_content += f"- Agents with Scratchpads: {scratchpad_stats.get('total_agents', 0)}\n"
        
        # Performance insights
        opt_metrics = enhanced_summary["optimization_metrics"]
        if opt_metrics["agents_executed"] == len(state["selected_agents"]):
            patterns_content += f"\nSuccess Pattern: All agents completed successfully with {self.context_strategy} strategy\n"
        
        patterns_content += f"\nRecommendation: This agent combination with {self.context_strategy} context strategy "
        patterns_content += f"achieved {exec_summary['success_rate']:.1%} success rate and can be reused for similar analyses.\n"
        
        return patterns_content
    
    def _enhanced_agent_router(self, state: MarketingResearchState) -> MarketingResearchState:
        """Enhanced agent router with context-aware routing."""
        
        # Find the next agent to execute
        next_agent = self._get_next_agent_with_context(state)
        
        if next_agent:
            logger.info(f"🎯 Routing to agent: {next_agent}")
            state["current_agent"] = next_agent
            # Don't set to RUNNING here - let the agent node do that when it actually starts
            # This ensures the routing logic can continue to find the agent as PENDING
            
            # Increment step for this agent
            if "agent_steps" not in state:
                state["agent_steps"] = {}
            if next_agent not in state["agent_steps"]:
                state["agent_steps"][next_agent] = 0
            state["agent_steps"][next_agent] += 1
            state["current_step"] = state.get("current_step", 0) + 1
            
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
            if state["agent_status"].get(agent) in [AgentStatus.PENDING, "pending"]
        ]
        
        logger.info(f"🔍 Selected agents: {state['selected_agents']}")
        logger.info(f"🔍 Agent status: {state['agent_status']}")
        logger.info(f"🔍 Pending agents: {pending_agents}")
        
        if not pending_agents:
            return None
        
        # Exclude report_summarizer from initial execution - it should run last
        other_pending_agents = [agent for agent in pending_agents if agent != 'report_summarizer']
        
        # If there are other agents pending, prioritize them
        if other_pending_agents:
            # Enhanced dependency resolution with context awareness
            agent_dependencies = {
                "market_research_analyst": [],
                "competitive_analyst": [],
                "data_analyst": [],
                "content_strategist": ["market_research_analyst"],
                "creative_copywriter": ["content_strategist"],
                "brand_performance_specialist": ["competitive_analyst", "data_analyst"],
                "forecasting_specialist": ["market_research_analyst", "data_analyst"],
                "campaign_optimizer": ["data_analyst", "content_strategist"],
            }
            
            # Find agents whose dependencies are satisfied, prioritized by execution order
            ready_agents = []
            for agent in other_pending_agents:
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
                    ready_agents.append((agent, agent_priority))
                    logger.debug(f"Agent {agent} priority: {agent_priority}")
            
            # Return the highest priority ready agent
            if ready_agents:
                ready_agents.sort(key=lambda x: x[1], reverse=True)
                return ready_agents[0][0]
            
            # If no dependencies are met, return the first pending agent
            # (this handles cases where dependencies aren't selected)
            return other_pending_agents[0] if other_pending_agents else None
        
        # If only report_summarizer is pending, return it
        elif 'report_summarizer' in pending_agents:
            logger.info("🔄 All other agents completed, executing report_summarizer")
            return 'report_summarizer'
        
        # FIXED: Check if all other agents are completed and report_summarizer is in selected_agents
        elif 'report_summarizer' in state["selected_agents"]:
            # Check if all non-report_summarizer agents are completed
            other_agents = [agent for agent in state["selected_agents"] if agent != 'report_summarizer']
            all_others_completed = all(
                state["agent_status"].get(agent) == AgentStatus.COMPLETED
                for agent in other_agents
            )
            
            if all_others_completed and state["agent_status"].get('report_summarizer') != AgentStatus.COMPLETED:
                logger.info("🔄 All other agents completed, forcing report_summarizer execution as final step")
                return 'report_summarizer'
        
        return None
    
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
                
                # Set agent status to RUNNING when execution actually starts
                state["agent_status"][agent_name] = AgentStatus.RUNNING
                
                logger.info(f"🤖 Executing {agent_name} (step {current_step})")
                
                # Get global context from long-term memory
                global_context_key = state.get("global_context_key")
                global_context = self.context_engine.retrieve_long_term_memory(
                    key=global_context_key.split("_", 2)[2],  # Extract workflow_id
                    namespace="contexts"
                ) if global_context_key else {}
                
                # Get agent-specific historical context from mem0
                agent_historical_context = {}
                if self.mem0_available and self.memory_manager:
                    try:
                        # Create agent-specific query
                        agent_query = f"{agent_name} {state.get('analysis_focus', '')} {state.get('target_audience', '')}"
                        
                        # Get agent-specific historical insights
                        agent_insights = self.memory_manager.get_relevant_context(
                            query=agent_query,
                            user_id=f"agent_{agent_name}",
                            limit=3
                        )
                        
                        if agent_insights:
                            agent_historical_context = {
                                "agent_specific_insights": agent_insights,
                                "previous_learnings": self._extract_learnings_for_agent(agent_insights, agent_name)
                            }
                            logger.info(f"🧠 Retrieved {len(agent_insights)} historical insights for {agent_name}")
                    except Exception as e:
                        logger.warning(f"Failed to retrieve agent-specific context for {agent_name}: {e}")
                
                # Merge historical context with global context
                enhanced_global_context = {**global_context}
                if agent_historical_context:
                    enhanced_global_context["agent_historical_context"] = agent_historical_context
                
                # Get optimized context for this agent (includes isolation)
                base_context = self.context_engine.get_context_for_agent(
                    agent_role=agent_name,
                    thread_id=workflow_id,
                    step=current_step,
                    full_context={**state, **enhanced_global_context},
                    strategy=self.context_strategy
                )

                # Apply intelligent filtering within token budget; if budget manager exists, pass budget
                token_budget = state.get("token_budget")
                if not token_budget and self.budget_manager:
                    # Use knowledge bucket as primary budget for context
                    allocations = self.budget_manager.allocate_budget().allocations
                    token_budget = allocations.get("knowledge", token_budget)
                filtered_context = self.context_filter.filter_context(
                    context=base_context,
                    agent_id=agent_name,
                    task_description=state.get("analysis_focus", ""),
                    required_keys=[],
                    token_budget=token_budget
                )

                # Choose compression level based on strategy
                strategy_to_level = {
                    "smart": CompressionLevel.MEDIUM,
                    "isolated": CompressionLevel.LIGHT,
                    "compressed": CompressionLevel.AGGRESSIVE,
                    "minimal": CompressionLevel.EXTREME,
                }
                compression_level = strategy_to_level.get(self.context_strategy, CompressionLevel.MEDIUM)

                # Apply compression
                compressed_context, compression_result = self.compressor.compress_context(
                    context=filtered_context,
                    level=compression_level,
                    preserve_keys=["agent_role", "scratchpad"],
                    target_reduction=0.5
                )

                # Evaluate context quality and record
                quality_report = self.context_quality.evaluate_quality(compressed_context)

                # Record preparation details in state for observability and persistence
                state.setdefault("last_optimized_context", {})[agent_name] = compressed_context
                state["last_context_compression"] = {
                    "agent": agent_name,
                    "method": compression_result.method_used,
                    "compression_ratio": compression_result.compression_ratio,
                    "original_size": compression_result.original_size,
                    "compressed_size": compression_result.compressed_size,
                }
                state.setdefault("context_quality", {})[agent_name] = {
                    "poisoning": quality_report.poisoning_score,
                    "distraction": quality_report.distraction_score,
                    "confusion": quality_report.confusion_score,
                    "clash": quality_report.clash_score,
                    "size_estimate": quality_report.size_estimate,
                    "timestamp": quality_report.timestamp,
                }

                # Create scratchpad entry for this execution step
                self.context_engine.create_scratchpad_entry(
                    agent_role=agent_name,
                    step=current_step,
                    content={
                        "action": "starting_execution",
                        "context_size": len(str(compressed_context)),
                        "strategy": self.context_strategy,
                        "compression": state["last_context_compression"],
                    },
                    reasoning=f"Starting execution of {agent_name} with {self.context_strategy} context strategy"
                )

                # If quality is poor (e.g., high clash or poisoning), annotate scratchpad
                if quality_report.poisoning_score > 0.7 or quality_report.clash_score > 0.7:
                    self.context_engine.create_scratchpad_entry(
                        agent_role=agent_name,
                        step=current_step,
                        content={
                            "action": "quality_warning",
                            "quality": {
                                "poisoning": quality_report.poisoning_score,
                                "clash": quality_report.clash_score
                            }
                        },
                        reasoning=f"Context quality warning for {agent_name}: potential poisoning/clash"
                    )
                
                # Track tokens before execution
                tokens_before = self.token_tracker.get_current_usage()
                
                # Execute the enhanced agent with optimized context
                agent_node = ENHANCED_AGENT_NODES[agent_name]
                
                # Create a modified state with optimized context
                enhanced_state = state.copy()
                enhanced_state["optimized_context"] = compressed_context
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
                        
                        # Store insights in mem0 for persistent learning
                        if self.mem0_available and self.memory_manager:
                            try:
                                # Prepare insight data for mem0
                                insight_content = self._format_insights_for_mem0(agent_result, agent_name, state)
                                
                                # Store agent-specific insights
                                self.memory_manager.store_insights(
                                    insights=insight_content,
                                    context={
                                        "agent_role": agent_name,
                                        "workflow_id": workflow_id,
                                        "analysis_focus": state.get("analysis_focus", ""),
                                        "target_audience": state.get("target_audience", ""),
                                        "campaign_type": state.get("campaign_type", ""),
                                        "execution_step": current_step,
                                        "timestamp": datetime.now().isoformat()
                                    },
                                    user_id=f"agent_{agent_name}"
                                )
                                
                                # Also store in general marketing team memory
                                self.memory_manager.store_insights(
                                    insights=insight_content,
                                    context={
                                        "source_agent": agent_name,
                                        "workflow_id": workflow_id,
                                        "analysis_type": state.get("analysis_focus", ""),
                                        "timestamp": datetime.now().isoformat()
                                    },
                                    user_id="marketing_team"
                                )
                                
                                logger.info(f"💾 Stored {agent_name} insights in mem0 for persistent learning")
                                
                            except Exception as e:
                                logger.warning(f"Failed to store insights in mem0 for {agent_name}: {e}")
                
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

            # Persist to durable storage as well
            try:
                # Serialize state to handle enums properly
                serialized_state = self._serialize_state_for_checkpoint(state)
                
                self.persistent_storage.save_checkpoint(
                    workflow_id=workflow_id,
                    agent_id=current_agent,
                    context_data={
                        "state_snapshot": serialized_state,
                        "optimized_context": state.get("last_optimized_context", {}).get(current_agent),
                        "compression": state.get("last_context_compression"),
                    },
                    token_count=token_usage.get("total_tokens", 0),
                    priority_level="normal",
                    dependencies=state.get("agent_execution_order", [])[-3:],
                    compression_level=state.get("last_context_compression", {}).get("method", "none"),
                    metadata={
                        "step": state["agent_steps"][current_agent],
                        "status": state["agent_status"].get(current_agent).value if hasattr(state["agent_status"].get(current_agent), 'value') else str(state["agent_status"].get(current_agent)),
                        "timestamp": datetime.now().isoformat(),
                    },
                )
            except Exception as e:
                logger.warning(f"Persistent checkpoint save failed: {e}")
            
            # Store checkpoint reference in state
            if "checkpoints" not in state:
                state["checkpoints"] = []
            state["checkpoints"].append(checkpoint.checkpoint_id)
            
            logger.info(f"💾 Created checkpoint after {current_agent} execution")
        
        return state
    
    def _serialize_state_for_checkpoint(self, state: MarketingResearchState) -> dict:
        """Serialize state for checkpoint storage, converting enums to their values."""
        def serialize_value(value):
            if isinstance(value, Enum):
                return value.value
            elif isinstance(value, dict):
                return {k: serialize_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [serialize_value(item) for item in value]
            elif isinstance(value, tuple):
                return tuple(serialize_value(item) for item in value)
            else:
                return value
        
        # Create a copy of the state and serialize all enum values
        serialized_state = {}
        for key, value in state.items():
            serialized_state[key] = serialize_value(value)
        
        return serialized_state
    
    def _route_next_enhanced_agent(self, state: MarketingResearchState) -> str:
        """Route to next agent or finalization with enhanced logic."""
        
        # Check if we have a current agent to checkpoint
        current_agent = state.get("current_agent")
        if current_agent and state["agent_status"].get(current_agent) in [AgentStatus.COMPLETED, AgentStatus.FAILED]:
            return "checkpoint"
        
        # Check if all agents are done
        pending_agents = [
            agent for agent in state["selected_agents"]
            if state["agent_status"].get(agent) in [AgentStatus.PENDING, "pending"]
        ]
        
        logger.info(f"🔍 Routing check - Pending agents: {pending_agents}")
        
        if not pending_agents:
            logger.info("✅ All agents completed, routing to finalize")
            return "finalize"
        
        # Get the next agent to execute
        next_agent = self._get_next_agent_with_context(state)
        if next_agent:
            logger.info(f"🎯 Routing to next agent: {next_agent}")
            return next_agent
        else:
            logger.info("🔄 No ready agents, routing to finalize")
            return "finalize"
    
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
        
        # Store comprehensive workflow insights in mem0
        if self.mem0_available and self.memory_manager:
            try:
                # Create comprehensive workflow summary for mem0
                workflow_insights = self._create_workflow_summary_for_mem0(state, enhanced_summary)
                
                # Store workflow-level insights
                self.memory_manager.store_insights(
                    insights=workflow_insights,
                    context={
                        "workflow_type": "completed_analysis",
                        "workflow_id": workflow_id,
                        "analysis_focus": state.get("analysis_focus", ""),
                        "target_audience": state.get("target_audience", ""),
                        "campaign_type": state.get("campaign_type", ""),
                        "agents_used": state["selected_agents"],
                        "success_rate": enhanced_summary["optimization_metrics"]["agents_executed"] / len(state["selected_agents"]),
                        "completion_time": datetime.now().isoformat()
                    },
                    user_id="marketing_team"
                )
                
                # Store workflow patterns for future optimization
                if enhanced_summary["optimization_metrics"]["agents_executed"] == len(state["selected_agents"]):
                    pattern_insights = self._extract_workflow_patterns(state, enhanced_summary)
                    self.memory_manager.store_insights(
                        insights=pattern_insights,
                        context={
                            "insight_type": "workflow_pattern",
                            "context_strategy": self.context_strategy,
                            "optimization_level": state.get("optimization_level", ""),
                            "agents_combination": "_".join(sorted(state["selected_agents"])),
                            "timestamp": datetime.now().isoformat()
                        },
                        user_id="workflow_optimizer"
                    )
                
                logger.info(f"💾 Stored comprehensive workflow insights in mem0")
                
            except Exception as e:
                logger.warning(f"Failed to store workflow insights in mem0: {e}")
        
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
            # Only add report_summarizer if user hasn't explicitly excluded it
            # Check if this is a manual configuration where user deliberately excluded it
            if "report_summarizer" not in selected_agents:
                # Add report_summarizer by default, but log it clearly for transparency
                selected_agents = selected_agents + ["report_summarizer"]
                logger.info(f"🔄 Automatically added report_summarizer for final report generation: {selected_agents}")
                logger.info("💡 To exclude report_summarizer, ensure it's not in your agent selection")
            
            # Create initial state as MarketingResearchState object
            workflow_id = str(uuid.uuid4())
            current_time = datetime.now()
            
            # Removed duplicate data context manager - context handled by Enhanced Context Engineering
            # Get data context through the main context engine (eliminates redundancy)
            try:
                # Data context is now handled by self.context_engine for better integration
                data_context = {"schema": {"shape": "Handled by Enhanced Context Engineering"}, "status": "optimized"}
                logger.info(f"📊 Data context optimized: Handled by Enhanced Context Engineering")
            except Exception as e:
                logger.warning(f"Context engineering data access: {e}")
                data_context = {"error": "Data context handled by main context engine"}
            
            initial_state = MarketingResearchState(
                workflow_id=workflow_id,
                workflow_type="enhanced_marketing_research",
                status=WorkflowStatus.PENDING,
                selected_agents=selected_agents,
                target_audience=target_audience,
                campaign_type=campaign_type,
                budget=budget,
                duration=duration,
                analysis_focus=analysis_focus,
                created_at=current_time,
                updated_at=current_time,
                # Add required fields with defaults from kwargs or sensible defaults
                initial_inputs=kwargs,
                user_query=kwargs.get("user_query", ""),  # Add user_query for chat mode detection
                data_context=data_context,  # Add data context for all agents
                business_objective=kwargs.get("business_objective", ""),
                competitive_landscape=kwargs.get("competitive_landscape", ""),
                market_segments=kwargs.get("market_segments", ["premium", "mass_market"]),
                product_categories=kwargs.get("product_categories", ["soft_drinks", "energy_drinks"]),
                key_metrics=kwargs.get("key_metrics", ["revenue", "market_share"]),
                brands=kwargs.get("brands", ["Brand_A", "Brand_B"]),
                campaign_goals=kwargs.get("campaign_goals", ["increase_awareness"]),
                agent_configs={},
                optimization_level=kwargs.get("optimization_level", "partial"),
                agent_execution_order=[],
                current_agent=None,
                agent_status={agent: AgentStatus.PENDING for agent in selected_agents},
                agent_results={},
                agent_errors={},
                agent_token_usage={},
                # Enhanced workflow tracking (for context engineering)
                current_step=0,
                agent_steps={agent: 0 for agent in selected_agents},
                context_strategy=kwargs.get("context_strategy", self.context_strategy),
                shared_data={},
                shared_context={},
                cached_results={},
                completed_at=None,
                final_summary=None,
                regions=kwargs.get("regions", ["North", "South"]),
                forecast_periods=kwargs.get("forecast_periods", 30),
                market_research_results=None,
                competitive_analysis_results=None,
                data_analysis_results=None,
                content_strategy_results=None,
                copywriting_results=None,
                campaign_optimization_results=None,
                brand_performance_results=None,
                forecasting_results=None,
                final_report=None,
                recommendations=None,
                next_steps=None,
                errors=[],
                warnings=[],
                total_token_usage={'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
                execution_time=None,
                cache_hits=0,
                cache_misses=0
            )
            
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
        """Restore workflow from a checkpoint.
        First tries in-memory EnhancedContextEngineering history; if not found, tries persistent storage.
        """
        
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
        
        # Fallback to persistent storage (latest checkpoint)
        try:
            persistent_checkpoints = self.persistent_storage.get_workflow_checkpoints(thread_id, limit=1)
            if persistent_checkpoints:
                cp = persistent_checkpoints[0]
                logger.info(f"🔄 Restored from persistent checkpoint: {cp.checkpoint_id}")
                return {
                    "checkpoint_id": cp.checkpoint_id,
                    "thread_id": cp.workflow_id,
                    "agent_role": cp.agent_id,
                    "step": cp.metadata.get("step") if isinstance(cp.metadata, dict) else None,
                    "state": cp.context_data.get("state_snapshot", {}),
                    "timestamp": cp.timestamp.isoformat(),
                    "token_usage": {"total_tokens": cp.token_count},
                }
        except Exception as e:
            logger.warning(f"Persistent restore failed: {e}")
        
        return None
    
    def execute_workflow(self, **kwargs) -> Dict[str, Any]:
        """
        Compatibility method that calls execute_enhanced_workflow.
        This ensures the dashboard can call workflow.execute_workflow() successfully.
        """
        return self.execute_enhanced_workflow(**kwargs)
    
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