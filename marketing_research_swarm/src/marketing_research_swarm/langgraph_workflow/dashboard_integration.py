"""
Dashboard Integration for LangGraph Marketing Research Workflow

This module provides the interface between the Streamlit dashboard and the
LangGraph workflow system, replacing the CrewAI integration.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable
import streamlit as st
from datetime import datetime
import json

from .workflow import MarketingResearchWorkflow
from .state import MarketingResearchState, WorkflowStatus, AgentStatus
from ..blackboard.integrated_blackboard import get_integrated_blackboard

logger = logging.getLogger(__name__)


class DashboardWorkflowManager:
    """
    Manages workflow execution and provides real-time updates for the dashboard.
    """
    
    def __init__(self):
        self.workflow = MarketingResearchWorkflow()
        self.blackboard = get_integrated_blackboard()
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
    
    def get_available_agents(self) -> List[Dict[str, Any]]:
        """Get list of available agents with their configurations."""
        import yaml
        import os
        
        config_path = os.path.join(
            os.path.dirname(__file__), '..', 'config', 'agents.yaml'
        )
        
        try:
            with open(config_path, 'r') as file:
                agent_configs = yaml.safe_load(file)
            
            agents = []
            for agent_name, config in agent_configs.items():
                agents.append({
                    'name': agent_name,
                    'role': config.get('role', agent_name),
                    'goal': config.get('goal', ''),
                    'backstory': config.get('backstory', ''),
                    'tools': config.get('tools', []),
                    'allow_delegation': config.get('allow_delegation', False)
                })
            
            return agents
            
        except Exception as e:
            logger.error(f"Failed to load agent configurations: {e}")
            return []
    
    def start_workflow(
        self,
        selected_agents: List[str],
        workflow_params: Dict[str, Any],
        progress_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Start a new workflow execution.
        
        Args:
            selected_agents: List of agent names to include
            workflow_params: Parameters for the workflow
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with workflow start result
        """
        try:
            # Validate inputs
            if not selected_agents:
                return {
                    "success": False,
                    "error": "No agents selected"
                }
            
            # Extract parameters
            target_audience = workflow_params.get('target_audience', 'health-conscious consumers')
            campaign_type = workflow_params.get('campaign_type', 'digital marketing')
            budget = float(workflow_params.get('budget', 250000))
            duration = workflow_params.get('duration', '12 months')
            analysis_focus = workflow_params.get('analysis_focus', 'market analysis')
            
            # Start workflow execution
            result = self.workflow.execute_workflow(
                selected_agents=selected_agents,
                target_audience=target_audience,
                campaign_type=campaign_type,
                budget=budget,
                duration=duration,
                analysis_focus=analysis_focus,
                **workflow_params
            )
            
            if result["success"]:
                workflow_id = result["workflow_id"]
                
                # Store workflow info for tracking
                self.active_workflows[workflow_id] = {
                    "start_time": datetime.now(),
                    "selected_agents": selected_agents,
                    "params": workflow_params,
                    "progress_callback": progress_callback
                }
                
                logger.info(f"Workflow {workflow_id} started successfully")
                
            return result
            
        except Exception as e:
            logger.error(f"Failed to start workflow: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_workflow_progress(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get current progress of a workflow."""
        try:
            status = self.workflow.get_workflow_status(workflow_id)
            if status:
                # Add additional progress information
                workflow_info = self.active_workflows.get(workflow_id, {})
                status.update({
                    "start_time": workflow_info.get("start_time"),
                    "selected_agents": workflow_info.get("selected_agents", []),
                    "elapsed_time": (
                        datetime.now() - workflow_info["start_time"]
                    ).total_seconds() if workflow_info.get("start_time") else 0
                })
                
            return status
            
        except Exception as e:
            logger.error(f"Failed to get workflow progress: {e}")
            return None
    
    def get_workflow_results(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get final results of a completed workflow."""
        try:
            state = self.blackboard.get_workflow_state(workflow_id)
            if state and state["status"] == WorkflowStatus.COMPLETED:
                return {
                    "workflow_id": workflow_id,
                    "status": "completed",
                    "final_summary": state.get("final_summary", {}),
                    "agent_results": state.get("agent_results", {}),
                    "recommendations": state.get("recommendations", []),
                    "execution_time": state.get("execution_time", 0),
                    "completed_at": state.get("completed_at")
                }
            return None
            
        except Exception as e:
            logger.error(f"Failed to get workflow results: {e}")
            return None
    
    def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a running workflow."""
        try:
            # Note: LangGraph doesn't have built-in cancellation
            # This would need to be implemented with additional state management
            logger.warning(f"Workflow cancellation not yet implemented for {workflow_id}")
            return False
            
        except Exception as e:
            logger.error(f"Failed to cancel workflow: {e}")
            return False


def create_dashboard_interface():
    """
    Create the Streamlit dashboard interface for the LangGraph workflow.
    """
    st.set_page_config(
        page_title="Marketing Research Workflow",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("🚀 Marketing Research Workflow - LangGraph Edition")
    st.markdown("*Powered by LangGraph State Management*")
    
    # Initialize workflow manager
    if 'workflow_manager' not in st.session_state:
        st.session_state.workflow_manager = DashboardWorkflowManager()
    
    workflow_manager = st.session_state.workflow_manager
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("🔧 Workflow Configuration")
        
        # Agent Selection
        st.subheader("Select Agents")
        available_agents = workflow_manager.get_available_agents()
        
        agent_options = {agent['name']: agent['role'] for agent in available_agents}
        selected_agent_names = st.multiselect(
            "Choose agents for your workflow:",
            options=list(agent_options.keys()),
            default=['market_research_analyst', 'data_analyst'],
            format_func=lambda x: f"{agent_options[x]} ({x})"
        )
        
        # Workflow Parameters
        st.subheader("Campaign Parameters")
        
        target_audience = st.text_input(
            "Target Audience",
            value="health-conscious millennials and premium beverage consumers"
        )
        
        campaign_type = st.selectbox(
            "Campaign Type",
            ["multi-channel global marketing campaign", "digital marketing", "brand awareness", "product launch"]
        )
        
        budget = st.number_input(
            "Budget ($)",
            min_value=1000,
            max_value=10000000,
            value=250000,
            step=1000
        )
        
        duration = st.selectbox(
            "Duration",
            ["3 months", "6 months", "12 months", "18 months", "24 months"]
        )
        
        analysis_focus = st.text_area(
            "Analysis Focus",
            value="global beverage market performance and brand optimization"
        )
        
        # Advanced Options
        with st.expander("Advanced Options"):
            market_segments = st.multiselect(
                "Market Segments",
                ["North America", "Europe", "Asia Pacific", "Latin America", "Middle East & Africa"],
                default=["North America", "Europe", "Asia Pacific"]
            )
            
            product_categories = st.multiselect(
                "Product Categories",
                ["Cola", "Juice", "Energy", "Sports", "Water", "Tea", "Coffee"],
                default=["Cola", "Juice", "Energy", "Sports"]
            )
            
            data_file_path = st.text_input(
                "Data File Path",
                value="data/beverage_sales.csv"
            )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🎯 Workflow Execution")
        
        # Start workflow button
        if st.button("🚀 Start Workflow", type="primary", disabled=not selected_agent_names):
            workflow_params = {
                'target_audience': target_audience,
                'campaign_type': campaign_type,
                'budget': budget,
                'duration': duration,
                'analysis_focus': analysis_focus,
                'market_segments': market_segments,
                'product_categories': product_categories,
                'data_file_path': data_file_path
            }
            
            with st.spinner("Starting workflow..."):
                result = workflow_manager.start_workflow(
                    selected_agents=selected_agent_names,
                    workflow_params=workflow_params
                )
                
                if result["success"]:
                    st.session_state.current_workflow_id = result["workflow_id"]
                    st.success(f"Workflow started! ID: {result['workflow_id']}")
                    st.rerun()
                else:
                    st.error(f"Failed to start workflow: {result.get('error', 'Unknown error')}")
        
        # Display workflow progress
        if 'current_workflow_id' in st.session_state:
            workflow_id = st.session_state.current_workflow_id
            
            st.subheader(f"📈 Workflow Progress: {workflow_id}")
            
            progress_info = workflow_manager.get_workflow_progress(workflow_id)
            
            if progress_info:
                # Progress bar
                progress = progress_info.get('progress', 0)
                st.progress(progress / 100)
                
                # Status information
                col_status, col_agent, col_time = st.columns(3)
                
                with col_status:
                    status = progress_info.get('status', 'unknown')
                    st.metric("Status", status.title())
                
                with col_agent:
                    current_agent = progress_info.get('current_agent', 'None')
                    st.metric("Current Agent", current_agent or 'Finalizing')
                
                with col_time:
                    elapsed = progress_info.get('elapsed_time', 0)
                    st.metric("Elapsed Time", f"{elapsed:.1f}s")
                
                # Completed agents
                completed_agents = progress_info.get('completed_agents', [])
                if completed_agents:
                    st.write("✅ **Completed Agents:**")
                    for agent in completed_agents:
                        st.write(f"  • {agent}")
                
                # Auto-refresh for running workflows
                if status in ['running', 'pending']:
                    import time
                    time.sleep(2)
                    st.rerun()
                
                # Show results if completed
                elif status == 'completed':
                    results = workflow_manager.get_workflow_results(workflow_id)
                    if results:
                        st.success("🎉 Workflow completed successfully!")
                        
                        # Display summary
                        summary = results.get('final_summary', {})
                        if summary:
                            st.subheader("📋 Execution Summary")
                            
                            col_metrics1, col_metrics2, col_metrics3 = st.columns(3)
                            
                            with col_metrics1:
                                st.metric("Total Agents", summary.get('total_agents', 0))
                            
                            with col_metrics2:
                                st.metric("Success Rate", f"{summary.get('success_rate', 0):.1%}")
                            
                            with col_metrics3:
                                st.metric("Execution Time", f"{summary.get('execution_time', 0):.1f}s")
                        
                        # Display agent results
                        agent_results = results.get('agent_results', {})
                        if agent_results:
                            st.subheader("🔍 Agent Results")
                            
                            for agent_name, result in agent_results.items():
                                with st.expander(f"📊 {agent_name.replace('_', ' ').title()} Results"):
                                    if isinstance(result, dict):
                                        if 'analysis' in result:
                                            st.write("**Analysis:**")
                                            st.write(result['analysis'])
                                        
                                        if 'tool_results' in result:
                                            st.write("**Tool Results:**")
                                            st.json(result['tool_results'])
                                    else:
                                        st.write(str(result))
            
            else:
                st.warning("Unable to retrieve workflow progress")
    
    with col2:
        st.header("ℹ️ Selected Agents")
        
        if selected_agent_names:
            for agent_name in selected_agent_names:
                agent_info = next((a for a in available_agents if a['name'] == agent_name), None)
                if agent_info:
                    with st.expander(f"🤖 {agent_info['role']}"):
                        st.write(f"**Goal:** {agent_info['goal']}")
                        st.write(f"**Tools:** {', '.join(agent_info['tools'])}")
        else:
            st.info("Select agents from the sidebar to see their details")
        
        # Workflow history
        st.header("📚 Recent Workflows")
        
        # This would be implemented with persistent storage
        st.info("Workflow history will be displayed here")


if __name__ == "__main__":
    create_dashboard_interface()