#!/usr/bin/env python3
"""
LangGraph Marketing Research Dashboard

A Streamlit dashboard that uses the LangGraph workflow instead of CrewAI,
with integrated token optimization strategies for efficient LLM usage.
"""

import streamlit as st
import sys
import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Add the project root to the Python path
sys.path.append('src')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import LangGraph components with fallback
try:
    from marketing_research_swarm.langgraph_workflow.workflow import MarketingResearchWorkflow
    from marketing_research_swarm.langgraph_workflow.optimized_workflow import OptimizedMarketingWorkflow
    from marketing_research_swarm.langgraph_workflow.state import WorkflowStatus
    from langgraph_config import LangGraphConfig
    LANGGRAPH_AVAILABLE = True
    logger.info("✅ LangGraph components loaded successfully")
except ImportError as e:
    logger.warning(f"LangGraph components not available: {e}")
    logger.info("💡 Falling back to CrewAI optimization system")
    LANGGRAPH_AVAILABLE = False
    
    # Import CrewAI fallback components
    try:
        from marketing_research_swarm.optimization_manager import OptimizationManager as FallbackOptimizationManager
        CREWAI_FALLBACK_AVAILABLE = True
    except ImportError:
        CREWAI_FALLBACK_AVAILABLE = False

# Import optimization components
try:
    from marketing_research_swarm.optimization_manager import OptimizationManager
    from marketing_research_swarm.utils.token_tracker import TokenTracker
    from marketing_research_swarm.performance.context_optimizer import ContextOptimizer
    from marketing_research_swarm.cache.smart_cache import SmartCache
    OPTIMIZATION_AVAILABLE = True
except ImportError as e:
    logger.error(f"Optimization components not available: {e}")
    OPTIMIZATION_AVAILABLE = False


class LangGraphDashboard:
    """Streamlit dashboard for LangGraph workflow with token optimization."""
    
    def __init__(self):
        """Initialize the dashboard."""
        self.workflow = None
        self.config = None
        self.optimization_manager = None
        self.token_tracker = None
        self.context_optimizer = None
        self.smart_cache = None
        
        # Initialize components
        self.initialize_components()
    
    def initialize_components(self):
        """Initialize all dashboard components with fallback support."""
        try:
            if LANGGRAPH_AVAILABLE:
                self.config = LangGraphConfig()
                self.workflow = MarketingResearchWorkflow()
                self.optimized_workflow = OptimizedMarketingWorkflow()
                logger.info("✅ LangGraph components initialized")
            elif CREWAI_FALLBACK_AVAILABLE:
                self.fallback_manager = FallbackOptimizationManager()
                logger.info("✅ CrewAI fallback components initialized")
            
            if OPTIMIZATION_AVAILABLE:
                self.optimization_manager = OptimizationManager()
                self.token_tracker = TokenTracker()
                self.context_optimizer = ContextOptimizer()
                self.smart_cache = SmartCache()
                logger.info("✅ Optimization components initialized")
                
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            logger.info("💡 Some features may be limited without full dependencies")
    
    def render_header(self):
        """Render the dashboard header."""
        st.set_page_config(
            page_title="LangGraph Marketing Research",
            page_icon="🚀",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🚀 LangGraph Marketing Research Dashboard")
        st.markdown("**Advanced workflow orchestration with intelligent token optimization**")
        
        # System status with fallback indicators
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if LANGGRAPH_AVAILABLE:
                status = "🟢 Ready"
                help_text = "LangGraph workflow available"
            elif CREWAI_FALLBACK_AVAILABLE:
                status = "🟡 Fallback"
                help_text = "Using CrewAI optimization system"
            else:
                status = "🔴 Unavailable"
                help_text = "Install langgraph: pip install langgraph"
            st.metric("Workflow", status, help=help_text)
        with col2:
            status = "🟢 Active" if OPTIMIZATION_AVAILABLE else "🔴 Disabled"
            st.metric("Optimization", status)
        with col3:
            cache_status = "🟢 Active" if self.smart_cache else "🔴 Disabled"
            st.metric("Smart Cache", cache_status)
        with col4:
            tracker_status = "🟢 Tracking" if self.token_tracker else "🔴 Disabled"
            st.metric("Token Tracker", tracker_status)
    
    def render_sidebar(self):
        """Render the sidebar configuration."""
        st.sidebar.header("🎯 Analysis Configuration")
        
        # Analysis type selection
        analysis_type = st.sidebar.selectbox(
            "Analysis Type",
            ["comprehensive", "roi_focused", "content_strategy", "brand_performance", "sales_forecast", "quick_insights"],
            help="Select the type of marketing analysis to perform"
        )
        
        # Agent selection
        st.sidebar.subheader("🤖 Agent Selection")
        available_agents = [
            "market_research_analyst",
            "competitive_analyst", 
            "data_analyst",
            "content_strategist",
            "creative_copywriter",
            "brand_performance_specialist",
            "forecasting_specialist"
        ]
        
        selected_agents = st.sidebar.multiselect(
            "Select Agents",
            available_agents,
            default=self._get_default_agents(analysis_type),
            help="Choose which agents to include in the analysis"
        )
        
        # Campaign parameters
        st.sidebar.subheader("📊 Campaign Parameters")
        target_audience = st.sidebar.text_input(
            "Target Audience",
            value="Tech-savvy millennials interested in sustainable products",
            help="Describe your target audience"
        )
        
        campaign_type = st.sidebar.selectbox(
            "Campaign Type",
            ["Digital Marketing", "Content Marketing", "Social Media", "Email Marketing", "Traditional Media"],
            help="Select the type of marketing campaign"
        )
        
        budget = st.sidebar.number_input(
            "Budget ($)",
            min_value=1000,
            max_value=1000000,
            value=50000,
            step=1000,
            help="Campaign budget in USD"
        )
        
        duration = st.sidebar.selectbox(
            "Campaign Duration",
            ["1 month", "3 months", "6 months", "1 year"],
            index=1,
            help="Duration of the marketing campaign"
        )
        
        analysis_focus = st.sidebar.text_area(
            "Analysis Focus",
            value="ROI optimization and market penetration strategies",
            help="Specific areas to focus the analysis on"
        )
        
        # Token optimization settings
        st.sidebar.subheader("⚡ Token Optimization")
        
        optimization_level = st.sidebar.selectbox(
            "Optimization Level",
            ["none", "partial", "full", "blackboard"],
            index=2,
            help="Choose optimization level for token reduction"
        )
        
        enable_caching = st.sidebar.checkbox(
            "Enable Smart Caching",
            value=True,
            help="Use intelligent caching to reduce redundant API calls"
        )
        
        enable_context_optimization = st.sidebar.checkbox(
            "Enable Context Optimization",
            value=True,
            help="Optimize context to reduce token usage"
        )
        
        token_budget = st.sidebar.number_input(
            "Token Budget",
            min_value=1000,
            max_value=100000,
            value=10000,
            step=1000,
            help="Maximum tokens to use for this analysis"
        )
        
        # Advanced settings
        with st.sidebar.expander("🔧 Advanced Settings"):
            enable_parallel_execution = st.checkbox(
                "Parallel Execution",
                value=True,
                help="Execute independent agents in parallel"
            )
            
            enable_dependency_optimization = st.checkbox(
                "Dependency Optimization",
                value=True,
                help="Optimize agent execution order based on dependencies"
            )
            
            enable_result_compression = st.checkbox(
                "Result Compression",
                value=True,
                help="Compress intermediate results to save memory"
            )
        
        return {
            "analysis_type": analysis_type,
            "selected_agents": selected_agents,
            "target_audience": target_audience,
            "campaign_type": campaign_type,
            "budget": budget,
            "duration": duration,
            "analysis_focus": analysis_focus,
            "optimization_level": optimization_level,
            "enable_caching": enable_caching,
            "enable_context_optimization": enable_context_optimization,
            "token_budget": token_budget,
            "enable_parallel_execution": enable_parallel_execution,
            "enable_dependency_optimization": enable_dependency_optimization,
            "enable_result_compression": enable_result_compression
        }
    
    def _get_default_agents(self, analysis_type: str) -> List[str]:
        """Get default agents based on analysis type."""
        defaults = {
            "comprehensive": ["market_research_analyst", "competitive_analyst", "data_analyst", "content_strategist"],
            "roi_focused": ["data_analyst", "forecasting_specialist"],
            "content_strategy": ["market_research_analyst", "content_strategist", "creative_copywriter"],
            "brand_performance": ["competitive_analyst", "brand_performance_specialist"],
            "sales_forecast": ["data_analyst", "forecasting_specialist"],
            "quick_insights": ["market_research_analyst", "data_analyst"]
        }
        return defaults.get(analysis_type, ["market_research_analyst", "data_analyst"])
    
    def run_optimized_analysis(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run analysis with token optimization strategies using available system."""
        
        # Check what systems are available
        if LANGGRAPH_AVAILABLE:
            return self._run_langgraph_analysis(config)
        elif CREWAI_FALLBACK_AVAILABLE:
            return self._run_crewai_fallback_analysis(config)
        else:
            return {
                "success": False, 
                "error": "No workflow system available. Please install dependencies.",
                "installation_help": "Run: pip install langgraph langchain-openai"
            }
    
    def _run_langgraph_analysis(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run analysis using LangGraph workflow."""
        try:
            # Choose workflow based on optimization level
            if config["optimization_level"] in ["none"]:
                workflow = self.workflow
                logger.info("Using standard LangGraph workflow")
            else:
                # Use optimized workflow for token reduction
                workflow = OptimizedMarketingWorkflow(optimization_level=config["optimization_level"])
                logger.info(f"Using optimized LangGraph workflow with level: {config['optimization_level']}")
            
            # Apply optimization strategies
            optimized_config = self._apply_optimization_strategies(config)
            
            # Execute the optimized workflow
            if hasattr(workflow, 'execute_optimized_workflow'):
                result = workflow.execute_optimized_workflow(
                    selected_agents=optimized_config["selected_agents"],
                    target_audience=optimized_config["target_audience"],
                    campaign_type=optimized_config["campaign_type"],
                    budget=optimized_config["budget"],
                    duration=optimized_config["duration"],
                    analysis_focus=optimized_config["analysis_focus"],
                    optimization_config=optimized_config
                )
            else:
                result = workflow.execute_workflow(
                    selected_agents=optimized_config["selected_agents"],
                    target_audience=optimized_config["target_audience"],
                    campaign_type=optimized_config["campaign_type"],
                    budget=optimized_config["budget"],
                    duration=optimized_config["duration"],
                    analysis_focus=optimized_config["analysis_focus"]
                )
            
            return result
            
        except Exception as e:
            logger.error(f"LangGraph analysis failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _run_crewai_fallback_analysis(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Run analysis using CrewAI optimization system as fallback."""
        try:
            logger.info("Using CrewAI optimization system (LangGraph fallback)")
            
            # Get optimized crew instance
            crew = self.fallback_manager.get_crew_instance(
                mode=config["optimization_level"],
                selected_agents=config["selected_agents"]
            )
            
            # Prepare task parameters
            task_params = {
                'target_audience': config["target_audience"],
                'campaign_type': config["campaign_type"],
                'budget': config["budget"],
                'duration': config["duration"],
                'analysis_focus': config["analysis_focus"]
            }
            
            # Run the crew analysis
            crew_result = crew.kickoff(inputs=task_params)
            
            # Extract metrics
            metrics = self.fallback_manager.extract_metrics_from_output(crew_result)
            
            # Format result to match LangGraph format
            result = {
                "success": True,
                "workflow_id": f"crewai_fallback_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "workflow_engine": "CrewAI (Fallback)",
                "status": "completed",
                "agent_results": {"analysis": str(crew_result)},
                "token_usage": metrics.get("token_usage", {}),
                "optimization_metrics": metrics.get("optimization_metrics", {}),
                "execution_time": metrics.get("execution_time", 0),
                "summary": {
                    "optimization_level": config["optimization_level"],
                    "agents_used": len(config["selected_agents"]),
                    "fallback_used": True
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"CrewAI fallback analysis failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _apply_optimization_strategies(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply token optimization strategies to the configuration."""
        optimized_config = config.copy()
        
        # Context optimization
        if config["enable_context_optimization"] and self.context_optimizer:
            optimized_config = self.context_optimizer.optimize_context(
                optimized_config, 
                token_budget=config["token_budget"]
            )
        
        # Agent selection optimization
        if config["optimization_level"] in ["full", "blackboard"]:
            optimized_config["selected_agents"] = self._optimize_agent_selection(
                config["selected_agents"], 
                config["analysis_type"]
            )
        
        # Enable caching
        if config["enable_caching"] and self.smart_cache:
            optimized_config["cache_enabled"] = True
            optimized_config["cache_config"] = self.smart_cache.get_cache_config()
        
        return optimized_config
    
    def _optimize_agent_selection(self, selected_agents: List[str], analysis_type: str) -> List[str]:
        """Optimize agent selection for token efficiency."""
        # Define minimal agent sets for different analysis types
        minimal_sets = {
            "roi_focused": ["data_analyst", "forecasting_specialist"],
            "content_strategy": ["content_strategist", "creative_copywriter"],
            "brand_performance": ["competitive_analyst", "brand_performance_specialist"],
            "sales_forecast": ["forecasting_specialist"],
            "quick_insights": ["market_research_analyst"]
        }
        
        # Use minimal set if available, otherwise use first 3 agents
        if analysis_type in minimal_sets:
            return minimal_sets[analysis_type]
        else:
            return selected_agents[:3]  # Limit to 3 agents for efficiency
    
    def render_results(self, result: Dict[str, Any]):
        """Render analysis results with optimization metrics."""
        if not result.get("success"):
            st.error(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
            return
        
        st.success("✅ Analysis completed successfully!")
        
        # Create tabs for different result views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Results", "⚡ Optimization", "🔍 Token Usage", "📈 Performance"])
        
        with tab1:
            self._render_analysis_results(result)
        
        with tab2:
            self._render_optimization_metrics(result)
        
        with tab3:
            self._render_token_usage(result)
        
        with tab4:
            self._render_performance_metrics(result)
    
    def _render_analysis_results(self, result: Dict[str, Any]):
        """Render the main analysis results."""
        st.subheader("📊 Analysis Results")
        
        # Workflow summary
        if "summary" in result:
            summary = result["summary"]
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Workflow ID", summary.get("workflow_id", "N/A")[:8] + "...")
            with col2:
                st.metric("Execution Time", f"{summary.get('execution_time', 0):.2f}s")
            with col3:
                st.metric("Success Rate", f"{summary.get('success_rate', 0)*100:.1f}%")
            with col4:
                st.metric("Agents Used", summary.get("completed_agents", 0))
        
        # Agent results
        if "agent_results" in result:
            st.subheader("🤖 Agent Results")
            
            for agent, agent_result in result["agent_results"].items():
                with st.expander(f"📋 {agent.replace('_', ' ').title()}"):
                    if isinstance(agent_result, dict):
                        if "analysis" in agent_result:
                            st.write("**Analysis:**")
                            st.write(agent_result["analysis"])
                        if "recommendations" in agent_result:
                            st.write("**Recommendations:**")
                            st.write(agent_result["recommendations"])
                        if "metrics" in agent_result:
                            st.write("**Metrics:**")
                            st.json(agent_result["metrics"])
                    else:
                        st.write(str(agent_result))
    
    def _render_optimization_metrics(self, result: Dict[str, Any]):
        """Render optimization performance metrics."""
        st.subheader("⚡ Optimization Performance")
        
        if "optimization_metrics" not in result:
            st.info("No optimization metrics available")
            return
        
        opt_metrics = result["optimization_metrics"]
        
        # Token savings
        if "token_optimization" in opt_metrics:
            token_opt = opt_metrics["token_optimization"]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Token Savings",
                    f"{token_opt.get('token_savings_percent', 0):.1f}%",
                    help="Percentage of tokens saved through optimization"
                )
            with col2:
                st.metric(
                    "Traditional Tokens",
                    f"{token_opt.get('traditional_tokens', 0):,}",
                    help="Estimated tokens without optimization"
                )
            with col3:
                st.metric(
                    "Optimized Tokens",
                    f"{token_opt.get('optimized_tokens', 0):,}",
                    help="Actual tokens used with optimization"
                )
        
        # Cache performance
        if "cache_performance" in opt_metrics:
            cache_perf = opt_metrics["cache_performance"]
            
            st.subheader("💾 Cache Performance")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Hit Rate", f"{cache_perf.get('hit_rate', 0):.1f}%")
            with col2:
                st.metric("Total Hits", cache_perf.get('total_hits', 0))
            with col3:
                st.metric("Cache Size", f"{cache_perf.get('cache_size_mb', 0):.1f} MB")
        
        # Context optimization
        if "context_optimization" in opt_metrics:
            context_opt = opt_metrics["context_optimization"]
            
            st.subheader("🧠 Context Optimization")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Memory Saved", f"{context_opt.get('memory_saved_mb', 0):.1f} MB")
            with col2:
                st.metric("Compression Ratio", f"{context_opt.get('compression_ratio', 1.0):.2f}x")
    
    def _render_token_usage(self, result: Dict[str, Any]):
        """Render detailed token usage information."""
        st.subheader("🔍 Token Usage Analysis")
        
        if "token_usage" not in result:
            st.info("No token usage data available")
            return
        
        token_usage = result["token_usage"]
        
        # Overall usage metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Tokens", f"{token_usage.get('total_tokens', 0):,}")
        with col2:
            st.metric("Prompt Tokens", f"{token_usage.get('prompt_tokens', 0):,}")
        with col3:
            st.metric("Completion Tokens", f"{token_usage.get('completion_tokens', 0):,}")
        with col4:
            cost = token_usage.get('total_cost', 0)
            st.metric("Estimated Cost", f"${cost:.4f}")
        
        # Token usage breakdown chart
        if token_usage.get('prompt_tokens', 0) > 0 and token_usage.get('completion_tokens', 0) > 0:
            fig = px.pie(
                values=[token_usage['prompt_tokens'], token_usage['completion_tokens']],
                names=['Prompt Tokens', 'Completion Tokens'],
                title="Token Usage Breakdown"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Agent-wise token usage
        if "agent_token_usage" in token_usage:
            st.subheader("🤖 Agent Token Usage")
            agent_usage = token_usage["agent_token_usage"]
            
            agent_df = pd.DataFrame([
                {
                    "Agent": agent.replace('_', ' ').title(),
                    "Tokens": usage.get('total_tokens', 0),
                    "Cost": f"${usage.get('cost', 0):.4f}"
                }
                for agent, usage in agent_usage.items()
            ])
            
            if not agent_df.empty:
                st.dataframe(agent_df, use_container_width=True)
                
                # Agent usage chart
                fig = px.bar(
                    agent_df,
                    x="Agent",
                    y="Tokens",
                    title="Token Usage by Agent"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_performance_metrics(self, result: Dict[str, Any]):
        """Render performance and execution metrics."""
        st.subheader("📈 Performance Metrics")
        
        # Execution timeline
        if "final_state" in result and "agent_execution_order" in result["final_state"]:
            execution_order = result["final_state"]["agent_execution_order"]
            
            st.subheader("⏱️ Execution Timeline")
            for i, agent in enumerate(execution_order, 1):
                st.write(f"**Step {i}**: {agent.replace('_', ' ').title()}")
        
        # System performance
        if "summary" in result:
            summary = result["summary"]
            
            st.subheader("🎯 System Performance")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Agents", summary.get("total_agents", 0))
            with col2:
                st.metric("Completed", summary.get("completed_agents", 0))
            with col3:
                st.metric("Failed", summary.get("failed_agents", 0))
        
        # Recommendations for optimization
        st.subheader("💡 Optimization Recommendations")
        
        if result.get("token_usage", {}).get("total_tokens", 0) > 20000:
            st.warning("⚠️ High token usage detected. Consider:")
            st.markdown("""
            - Enabling higher optimization levels
            - Reducing the number of agents
            - Using more specific analysis focus
            - Enabling smart caching
            """)
        elif result.get("token_usage", {}).get("total_tokens", 0) < 5000:
            st.success("✅ Excellent token efficiency!")
        else:
            st.info("ℹ️ Good token usage. Consider blackboard optimization for maximum efficiency.")
    
    def run(self):
        """Run the main dashboard application."""
        self.render_header()
        
        # Get configuration from sidebar
        config = self.render_sidebar()
        
        # Main content area
        st.header("🎯 Marketing Analysis")
        
        # Show installation help if needed
        if not LANGGRAPH_AVAILABLE and not CREWAI_FALLBACK_AVAILABLE:
            st.error("⚠️ No workflow system available!")
            st.markdown("""
            **To fix this issue:**
            
            **Option 1: Install LangGraph (Recommended)**
            ```bash
            pip install langgraph langchain-openai streamlit plotly pandas
            ```
            
            **Option 2: Check CrewAI Components**
            - Ensure `optimization_manager.py` is available
            - Check that `src/marketing_research_swarm/` is in Python path
            
            **Quick Test:**
            ```bash
            python -c "from marketing_research_swarm.optimization_manager import OptimizationManager; print('✅ CrewAI OK')"
            ```
            """)
            return
        
        # Show fallback notice
        if not LANGGRAPH_AVAILABLE and CREWAI_FALLBACK_AVAILABLE:
            st.info("💡 **Using CrewAI Optimization System** - LangGraph not available, but all optimization features work through CrewAI fallback!")
        
        # Run analysis button
        if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
            if not config["selected_agents"]:
                st.error("Please select at least one agent to run the analysis.")
                return
            
            # Show progress
            with st.spinner("Running optimized analysis..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Update progress (simulated for now)
                for i in range(100):
                    progress_bar.progress(i + 1)
                    if i < 20:
                        status_text.text("Initializing workflow...")
                    elif i < 40:
                        status_text.text("Applying optimization strategies...")
                    elif i < 60:
                        status_text.text("Executing agents...")
                    elif i < 80:
                        status_text.text("Processing results...")
                    else:
                        status_text.text("Finalizing analysis...")
                
                # Run the actual analysis
                result = self.run_optimized_analysis(config)
                
                progress_bar.empty()
                status_text.empty()
            
            # Store result in session state
            st.session_state["last_result"] = result
            
            # Render results
            self.render_results(result)
        
        # Show previous results if available
        if "last_result" in st.session_state:
            st.header("📋 Previous Results")
            with st.expander("View Last Analysis"):
                self.render_results(st.session_state["last_result"])


def main():
    """Main function to run the dashboard."""
    dashboard = LangGraphDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()