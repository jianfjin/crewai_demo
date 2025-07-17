#!/usr/bin/env python
"""
Marketing Research Swarm - API-Based Dashboard
A Streamlit-based web interface that communicates with FastAPI backend
for better architecture separation and enhanced user experience.
"""

import streamlit as st
import requests
import json
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import uuid
from dataclasses import dataclass
import asyncio
import yaml
import sys
import os

# Page configuration
st.set_page_config(
    page_title="Marketing Research API Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f2f6, #ffffff);
        border-radius: 10px;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin: 1.5rem 0 1rem 0;
        padding: 0.5rem;
        border-left: 4px solid #1f77b4;
        background-color: #f8f9fa;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .status-running {
        color: #ff9800;
        font-weight: bold;
    }
    .status-completed {
        color: #4caf50;
        font-weight: bold;
    }
    .status-failed {
        color: #f44336;
        font-weight: bold;
    }
    .progress-container {
        margin: 1rem 0;
        padding: 1rem;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        background-color: #fafafa;
    }
</style>
""", unsafe_allow_html=True)

@dataclass
class AnalysisConfig:
    """Configuration for analysis request"""
    selected_agents: List[str]
    optimization_level: str
    enable_mem0: bool = False
    enable_caching: bool = True
    max_workers: int = 4
    task_parameters: Dict[str, Any] = None

class APIClient:
    """Client for communicating with FastAPI backend"""
    
    def __init__(self, base_url: str = None):
        # Auto-detect environment and set appropriate base URL
        if base_url is None:
            base_url = self._detect_backend_url()
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json"
        })
    
    def _detect_backend_url(self) -> str:
        """Auto-detect the backend URL based on environment"""
        import os
        
        # Check for environment variable override
        if os.environ.get('API_BASE_URL'):
            return os.environ.get('API_BASE_URL')
        
        # Check if we're in GitHub Codespaces
        if os.environ.get('CODESPACES'):
            codespace_name = os.environ.get('CODESPACE_NAME')
            github_codespaces_port_forwarding_domain = os.environ.get('GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN')
            
            if codespace_name and github_codespaces_port_forwarding_domain:
                # GitHub Codespaces URL format
                backend_url = f"https://{codespace_name}-8000.{github_codespaces_port_forwarding_domain}"
                print(f"🔍 Detected GitHub Codespaces environment")
                print(f"🔗 Backend URL: {backend_url}")
                return backend_url
        
        # Check if we're in Gitpod
        if os.environ.get('GITPOD_WORKSPACE_URL'):
            workspace_url = os.environ.get('GITPOD_WORKSPACE_URL')
            backend_url = workspace_url.replace('https://', 'https://8000-')
            print(f"🔍 Detected Gitpod environment")
            print(f"🔗 Backend URL: {backend_url}")
            return backend_url
        
        # Default to localhost for local development
        print(f"🔍 Using localhost for backend connection")
        return "http://localhost:8000"
    
    def health_check(self) -> Dict[str, Any]:
        """Check if API is available"""
        try:
            response = self.session.get(f"{self.base_url}/", timeout=5)
            return {"status": "healthy", "response": response.json()}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    def get_available_agents(self) -> Dict[str, Any]:
        """Get list of available agents"""
        try:
            # Use the actual backend endpoint that exists
            response = self.session.get(f"{self.base_url}/", timeout=10)
            response.raise_for_status()
            
            # Return hardcoded agents list since backend doesn't have this endpoint yet
            # These are the agents from the actual system
            agents = [
                "market_research_analyst",
                "competitive_analyst", 
                "data_analyst",
                "content_strategist",
                "brand_performance_specialist",
                "campaign_optimizer",
                "forecasting_specialist",
                "creative_copywriter"
            ]
            
            return {"agents": agents, "status": "success"}
        except Exception as e:
            st.error(f"Failed to connect to backend: {e}")
            # Return default agents list as fallback
            return {
                "agents": [
                    "market_research_analyst",
                    "competitive_analyst", 
                    "brand_performance_specialist",
                    "campaign_optimizer"
                ], 
                "error": str(e)
            }
    
    def get_analysis_types(self) -> Dict[str, Any]:
        """Get available analysis types"""
        try:
            # Try the actual API endpoint first
            response = self.session.get(f"{self.base_url}/api/analysis/types", timeout=10)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            # Log the connection issue for debugging
            print(f"⚠️ API call failed: {e}")
            print(f"🔗 Trying to connect to: {self.base_url}/api/analysis/types")
            
            # Return predefined analysis types as fallback
            analysis_types = {
                "roi_analysis": {
                    "name": "ROI Analysis",
                    "description": "Comprehensive ROI and profitability analysis for marketing campaigns",
                    "agents": ["market_research_analyst", "data_analyst", "campaign_optimizer"]
                },
                "brand_performance": {
                    "name": "Brand Performance Analysis", 
                    "description": "Analyze brand performance metrics and market positioning",
                    "agents": ["market_research_analyst", "competitive_analyst", "brand_performance_specialist"]
                },
                "sales_forecast": {
                    "name": "Sales Forecast Analysis",
                    "description": "Predict future sales trends and market opportunities",
                    "agents": ["market_research_analyst", "data_analyst", "forecasting_specialist"]
                },
                "comprehensive": {
                    "name": "Comprehensive Analysis",
                    "description": "Full marketing research analysis with all available agents",
                    "agents": ["market_research_analyst", "competitive_analyst", "data_analyst", "content_strategist", "brand_performance_specialist", "campaign_optimizer"]
                },
                "custom": {
                    "name": "Custom Analysis",
                    "description": "Select your own combination of agents for custom analysis",
                    "agents": []
                }
            }
            
            print(f"✅ Using fallback analysis types")
            return {"types": analysis_types, "status": "fallback"}
    
    def start_analysis(self, config: AnalysisConfig) -> Dict[str, Any]:
        """Start new analysis using the actual backend API"""
        try:
            # Prepare request data for the backend API
            request_data = {
                "analysis_type": "custom",
                "selected_agents": config.selected_agents,
                "optimization_level": config.optimization_level,
                "target_audience": config.task_parameters.get("target_audience", "beverage consumers"),
                "campaign_type": config.task_parameters.get("campaign_type", "Brand Awareness"),
                "budget": config.task_parameters.get("budget", 100000),
                "duration": config.task_parameters.get("duration", "3 months"),
                "enable_mem0": config.enable_mem0,
                "enable_caching": config.enable_caching,
                "max_workers": config.max_workers
            }
            
            # Call the actual backend API
            response = self.session.post(
                f"{self.base_url}/api/analysis",
                json=request_data,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            if result.get("success"):
                return {
                    "success": True,
                    "analysis_id": result["analysis_id"],
                    "message": "Analysis started successfully",
                    "status": "running"
                }
            else:
                return {"success": False, "error": result.get("error", "Unknown error")}
            
        except requests.exceptions.Timeout:
            return {"success": False, "error": "Request timeout - analysis may still be starting"}
        except requests.exceptions.RequestException as e:
            return {"success": False, "error": f"Network error: {str(e)}"}
        except Exception as e:
            return {"success": False, "error": f"Unexpected error: {str(e)}"}
    
    def get_analysis_status(self, analysis_id: str) -> Dict[str, Any]:
        """Get analysis status"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/analysis/{analysis_id}/status",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def get_analysis_result(self, analysis_id: str) -> Dict[str, Any]:
        """Get analysis results"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/analysis/{analysis_id}/result",
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def cancel_analysis(self, analysis_id: str) -> Dict[str, Any]:
        """Cancel running analysis"""
        try:
            response = self.session.delete(
                f"{self.base_url}/api/analysis/{analysis_id}/cancel",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_analysis_history(self) -> Dict[str, Any]:
        """Get analysis history"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/analysis/history",
                timeout=15
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"analyses": [], "error": str(e)}
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/system/metrics",
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

def initialize_session_state():
    """Initialize session state variables"""
    if 'api_client' not in st.session_state:
        st.session_state.api_client = APIClient()
    
    if 'current_analysis_id' not in st.session_state:
        st.session_state.current_analysis_id = None
    
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    
    if 'auto_refresh' not in st.session_state:
        st.session_state.auto_refresh = False

def check_api_connection():
    """Check API connection and display status"""
    health = st.session_state.api_client.health_check()
    
    if health["status"] == "healthy":
        st.sidebar.success("🟢 API Connected")
        st.sidebar.info(f"🔗 Backend: {st.session_state.api_client.base_url}")
        return True
    else:
        st.sidebar.error("🔴 API Disconnected")
        st.sidebar.error(f"Error: {health.get('error', 'Unknown error')}")
        
        # Show current backend URL and allow override
        st.sidebar.warning(f"🔗 Trying to connect to: {st.session_state.api_client.base_url}")
        
        # Allow manual backend URL override
        with st.sidebar.expander("🔧 Backend Configuration"):
            new_url = st.text_input(
                "Backend URL Override",
                value=st.session_state.api_client.base_url,
                help="Enter the correct backend URL for your environment"
            )
            
            if st.button("🔄 Update Backend URL"):
                st.session_state.api_client = APIClient(base_url=new_url)
                st.rerun()
            
            # Test specific endpoints
            if st.button("🧪 Test API Endpoints"):
                st.write("**Testing API Endpoints:**")
                
                # Test base endpoint
                try:
                    response = st.session_state.api_client.session.get(f"{st.session_state.api_client.base_url}/", timeout=5)
                    st.write(f"✅ Base endpoint: {response.status_code}")
                except Exception as e:
                    st.write(f"❌ Base endpoint: {e}")
                
                # Test agents endpoint
                try:
                    response = st.session_state.api_client.session.get(f"{st.session_state.api_client.base_url}/api/agents/available", timeout=5)
                    st.write(f"✅ Agents endpoint: {response.status_code}")
                except Exception as e:
                    st.write(f"❌ Agents endpoint: {e}")
                
                # Test analysis types endpoint
                try:
                    response = st.session_state.api_client.session.get(f"{st.session_state.api_client.base_url}/api/analysis/types", timeout=5)
                    st.write(f"✅ Analysis types endpoint: {response.status_code}")
                except Exception as e:
                    st.write(f"❌ Analysis types endpoint: {e}")
            
            # Show environment detection info
            st.write("**Environment Detection:**")
            import os
            if os.environ.get('CODESPACES'):
                st.write("🔍 GitHub Codespaces detected")
                st.write(f"📝 Codespace: {os.environ.get('CODESPACE_NAME', 'Unknown')}")
                st.write(f"🌐 Domain: {os.environ.get('GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN', 'Unknown')}")
                
                # Show expected URLs
                codespace_name = os.environ.get('CODESPACE_NAME', 'your-codespace')
                domain = os.environ.get('GITHUB_CODESPACES_PORT_FORWARDING_DOMAIN', 'app.github.dev')
                expected_backend = f"https://{codespace_name}-8000.{domain}"
                expected_frontend = f"https://{codespace_name}-8501.{domain}"
                
                st.write("**Expected URLs:**")
                st.write(f"🔗 Backend: {expected_backend}")
                st.write(f"🔗 Frontend: {expected_frontend}")
                
                # Port forwarding reminder
                st.write("**Port Forwarding Checklist:**")
                st.write("1. ✅ Port 8000 forwarded as Public")
                st.write("2. ✅ Port 8501 forwarded as Public")
                st.write("3. ✅ Backend running with --host 0.0.0.0")
                
            elif os.environ.get('GITPOD_WORKSPACE_URL'):
                st.write("🔍 Gitpod detected")
                st.write(f"🌐 Workspace: {os.environ.get('GITPOD_WORKSPACE_URL')}")
            else:
                st.write("🔍 Local development environment")
                st.write("🔗 Expected backend: http://localhost:8000")
        
        st.error("⚠️ Cannot connect to backend API. Please check the backend URL and ensure the FastAPI server is running.")
        return False

def render_configuration_form() -> Optional[AnalysisConfig]:
    """Render the analysis configuration form"""
    
    st.markdown('<div class="section-header">🤖 Analysis Configuration</div>', unsafe_allow_html=True)
    
    # Get available agents from API
    agents_data = st.session_state.api_client.get_available_agents()
    available_agents = agents_data.get("agents", [])
    
    if not available_agents:
        st.error("No agents available. Please check the backend configuration.")
        return None
    
    # Get analysis types
    types_data = st.session_state.api_client.get_analysis_types()
    analysis_types = types_data.get("types", {})
    
    with st.form("analysis_configuration"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Analysis Type")
            
            if analysis_types:
                analysis_type = st.selectbox(
                    "Select Analysis Type",
                    options=list(analysis_types.keys()),
                    help="Choose a pre-configured analysis type or select 'custom' for manual configuration"
                )
                
                if analysis_type != "custom":
                    selected_agents = analysis_types[analysis_type]["agents"]
                    st.info(f"**{analysis_types[analysis_type]['name']}**\n\n{analysis_types[analysis_type]['description']}")
                    st.write("**Auto-selected agents:**")
                    for agent in selected_agents:
                        st.write(f"• {agent}")
                else:
                    selected_agents = st.multiselect(
                        "Select Agents",
                        options=available_agents,
                        help="Choose agents for your custom analysis"
                    )
            else:
                selected_agents = st.multiselect(
                    "Select Agents",
                    options=available_agents,
                    help="Choose agents for your analysis"
                )
        
        with col2:
            st.subheader("⚙️ Optimization Settings")
            
            optimization_level = st.selectbox(
                "Optimization Level",
                options=["none", "partial", "full", "blackboard"],
                index=3,  # Default to blackboard
                help="Higher levels provide better performance"
            )
            
            # Performance preview
            if selected_agents and optimization_level:
                expected_duration = len(selected_agents) * 30 * (1 - (["none", "partial", "full", "blackboard"].index(optimization_level) * 0.2))
                expected_tokens = len(selected_agents) * 3000 * (1 - (["none", "partial", "full", "blackboard"].index(optimization_level) * 0.15))
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Expected Duration", f"{expected_duration:.0f}s")
                with col_b:
                    st.metric("Estimated Tokens", f"{expected_tokens:.0f}")
        
        # Advanced parameters
        with st.expander("🔧 Advanced Parameters"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                enable_mem0 = st.checkbox(
                    "Enable Mem0 Memory",
                    value=False,
                    help="⚠️ Adds 200-800ms delay but enables long-term memory"
                )
            
            with col2:
                enable_caching = st.checkbox(
                    "Enable Result Caching",
                    value=True,
                    help="✅ Recommended: Speeds up repeated analyses"
                )
            
            with col3:
                max_workers = st.slider(
                    "Parallel Workers",
                    min_value=1,
                    max_value=8,
                    value=4,
                    help="Number of parallel workers for agent execution"
                )
        
        # Task parameters
        st.subheader("📝 Task Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            target_audience = st.text_input("Target Audience", value="beverage consumers")
            budget = st.number_input("Budget ($)", min_value=1000, value=100000, step=1000)
        
        with col2:
            duration = st.selectbox("Campaign Duration", ["1 month", "3 months", "6 months", "1 year"])
            campaign_type = st.selectbox("Campaign Type", ["Brand Awareness", "Product Launch", "Market Expansion"])
        
        # Submit button
        submitted = st.form_submit_button("🚀 Start Analysis", type="primary")
        
        if submitted:
            if not selected_agents:
                st.error("Please select at least one agent")
                return None
            
            task_parameters = {
                "target_audience": target_audience,
                "budget": budget,
                "duration": duration,
                "campaign_type": campaign_type,
                "analysis_focus": "beverage industry",
                "business_objective": "Optimize marketing performance"
            }
            
            return AnalysisConfig(
                selected_agents=selected_agents,
                optimization_level=optimization_level,
                enable_mem0=enable_mem0,
                enable_caching=enable_caching,
                max_workers=max_workers,
                task_parameters=task_parameters
            )
    
    return None

def render_real_time_monitoring(analysis_id: str):
    """Render real-time analysis monitoring"""
    
    st.markdown('<div class="section-header">📊 Real-time Analysis Monitoring</div>', unsafe_allow_html=True)
    
    # Create containers for live updates
    status_container = st.empty()
    progress_container = st.empty()
    metrics_container = st.empty()
    
    # Control buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Refresh Status"):
            st.rerun()
    
    with col2:
        auto_refresh = st.checkbox("🔄 Auto Refresh", value=st.session_state.auto_refresh)
        st.session_state.auto_refresh = auto_refresh
    
    with col3:
        if st.button("❌ Cancel Analysis"):
            result = st.session_state.api_client.cancel_analysis(analysis_id)
            if result.get("success"):
                st.success("Analysis cancelled successfully")
                st.session_state.current_analysis_id = None
                st.rerun()
            else:
                st.error(f"Failed to cancel: {result.get('error')}")
    
    # Get current status
    status_data = st.session_state.api_client.get_analysis_status(analysis_id)
    
    if "error" in status_data:
        st.error(f"Error getting status: {status_data['error']}")
        return
    
    # Display status
    status = status_data.get("status", "unknown")
    
    with status_container.container():
        if status == "running":
            st.markdown(f'<div class="status-running">🔄 Status: {status.upper()}</div>', unsafe_allow_html=True)
        elif status == "completed":
            st.markdown(f'<div class="status-completed">✅ Status: {status.upper()}</div>', unsafe_allow_html=True)
        elif status == "failed":
            st.markdown(f'<div class="status-failed">❌ Status: {status.upper()}</div>', unsafe_allow_html=True)
        else:
            st.info(f"Status: {status}")
        
        if "current_agent" in status_data:
            st.info(f"Current Agent: {status_data['current_agent']}")
    
    # Display progress
    if "progress" in status_data:
        with progress_container.container():
            progress = status_data["progress"]
            st.progress(progress)
            st.write(f"Progress: {progress:.1%}")
    
    # Display metrics
    if "current_metrics" in status_data:
        metrics = status_data["current_metrics"]
        
        with metrics_container.container():
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Tokens Used", f"{metrics.get('tokens_used', 0):,}")
            
            with col2:
                st.metric("Agents Completed", f"{metrics.get('agents_completed', 0)}/{metrics.get('agents_total', 0)}")
            
            with col3:
                st.metric("Elapsed Time", f"{metrics.get('elapsed_time', 0):.1f}s")
            
            with col4:
                if metrics.get('estimated_remaining'):
                    st.metric("Est. Remaining", f"{metrics.get('estimated_remaining', 0):.1f}s")
    
    # Check if analysis is complete
    if status == "completed":
        st.success("🎉 Analysis completed successfully!")
        if st.button("📊 View Results"):
            st.session_state.show_results = True
            st.rerun()
    elif status == "failed":
        st.error(f"❌ Analysis failed: {status_data.get('error', 'Unknown error')}")
        st.session_state.current_analysis_id = None
    
    # Auto refresh
    if auto_refresh and status == "running":
        time.sleep(2)
        st.rerun()

def render_analysis_results(analysis_id: str):
    """Render analysis results"""
    
    st.markdown('<div class="section-header">📊 Analysis Results</div>', unsafe_allow_html=True)
    
    # Get results from API
    results_data = st.session_state.api_client.get_analysis_result(analysis_id)
    
    if "error" in results_data:
        st.error(f"Error getting results: {results_data['error']}")
        return
    
    if not results_data.get("success"):
        st.error("Failed to retrieve results")
        return
    
    results = results_data.get("result", {})
    
    # Display performance metrics
    if "performance_metrics" in results:
        st.subheader("⚡ Performance Metrics")
        
        metrics = results["performance_metrics"]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Duration", f"{metrics.get('total_duration', 0):.1f}s")
        
        with col2:
            st.metric("Total Tokens", f"{metrics.get('total_tokens', 0):,}")
        
        with col3:
            st.metric("Total Cost", f"${metrics.get('total_cost', 0):.4f}")
        
        with col4:
            st.metric("Optimization Level", metrics.get('optimization_level', 'unknown'))
        
        # Performance breakdown
        if "agent_breakdown" in metrics:
            st.subheader("🤖 Agent Performance Breakdown")
            
            agent_data = []
            for agent, data in metrics["agent_breakdown"].items():
                agent_data.append({
                    "Agent": agent,
                    "Tokens": data.get("tokens", 0),
                    "Duration": data.get("duration", 0),
                    "Cost": data.get("cost", 0)
                })
            
            if agent_data:
                df = pd.DataFrame(agent_data)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_tokens = px.bar(df, x="Agent", y="Tokens", title="Token Usage by Agent")
                    st.plotly_chart(fig_tokens, use_container_width=True)
                
                with col2:
                    fig_duration = px.bar(df, x="Agent", y="Duration", title="Execution Time by Agent")
                    st.plotly_chart(fig_duration, use_container_width=True)
    
    # Display analysis results
    if "analysis_result" in results:
        st.subheader("📋 Analysis Results")
        
        analysis_result = results["analysis_result"]
        
        if isinstance(analysis_result, str):
            st.text_area("Results", analysis_result, height=400)
        else:
            st.json(analysis_result)
    
    # Export options
    st.subheader("📥 Export Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Export as Text"):
            results_text = json.dumps(results, indent=2)
            st.download_button(
                label="Download Text File",
                data=results_text,
                file_name=f"analysis_results_{analysis_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    
    with col2:
        if st.button("📊 Export as JSON"):
            results_json = json.dumps(results, indent=2)
            st.download_button(
                label="Download JSON File",
                data=results_json,
                file_name=f"analysis_results_{analysis_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col3:
        if st.button("🔄 Run Similar Analysis"):
            # Reset to configuration with same parameters
            st.session_state.current_analysis_id = None
            st.session_state.show_results = False
            st.rerun()

def render_system_metrics():
    """Render system performance metrics"""
    
    st.markdown('<div class="section-header">📊 System Performance</div>', unsafe_allow_html=True)
    
    # Get system metrics from API
    metrics_data = st.session_state.api_client.get_system_metrics()
    
    if "error" in metrics_data:
        st.error(f"Error getting system metrics: {metrics_data['error']}")
        return
    
    # Overall system health
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        health_score = metrics_data.get("health_score", 0)
        health_status = "🟢 Healthy" if health_score > 80 else "🟡 Warning" if health_score > 60 else "🔴 Critical"
        st.metric("System Health", health_status, f"{health_score:.0f}%")
    
    with col2:
        st.metric("Active Analyses", metrics_data.get("active_analyses", 0))
    
    with col3:
        st.metric("Avg Response Time", f"{metrics_data.get('avg_response_time', 0):.2f}s")
    
    with col4:
        st.metric("Cache Hit Rate", f"{metrics_data.get('cache_hit_rate', 0):.1f}%")
    
    # Optimization metrics
    if "optimization_metrics" in metrics_data:
        st.subheader("⚡ Optimization Performance")
        
        opt_metrics = metrics_data["optimization_metrics"]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Token Savings", f"{opt_metrics.get('token_savings_percent', 0):.0f}%")
        
        with col2:
            st.metric("Execution Speedup", f"{opt_metrics.get('speedup_factor', 1):.1f}x")
        
        with col3:
            st.metric("Memory Efficiency", f"{opt_metrics.get('memory_efficiency', 0):.0f}%")

def render_analysis_history():
    """Render analysis history"""
    
    st.markdown('<div class="section-header">📚 Analysis History</div>', unsafe_allow_html=True)
    
    # Get history from API
    history_data = st.session_state.api_client.get_analysis_history()
    
    if "error" in history_data:
        st.error(f"Error getting analysis history: {history_data['error']}")
        return
    
    analyses = history_data.get("analyses", [])
    
    if not analyses:
        st.info("No previous analyses found. Start your first analysis above!")
        return
    
    # Display analysis cards
    for analysis in analyses[:10]:  # Show last 10 analyses
        with st.expander(f"📋 {analysis['id']} - {analysis['created_at']} ({analysis['status']})"):
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Configuration:**")
                st.json(analysis.get("configuration", {}))
            
            with col2:
                st.write("**Performance:**")
                if analysis["status"] == "completed" and "performance_metrics" in analysis:
                    perf = analysis["performance_metrics"]
                    st.metric("Duration", f"{perf.get('duration', 0):.1f}s")
                    st.metric("Tokens Used", f"{perf.get('tokens', 0):,}")
                    st.metric("Optimization", perf.get('optimization_level', 'unknown'))
                else:
                    st.error(f"Status: {analysis['status']}")
                    if "error" in analysis:
                        st.error(f"Error: {analysis['error']}")
            
            # Action buttons
            if analysis["status"] == "completed":
                if st.button(f"📊 View Results", key=f"view_{analysis['id']}"):
                    st.session_state.current_analysis_id = analysis['id']
                    st.session_state.show_results = True
                    st.rerun()

def main():
    """Main dashboard application"""
    
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">🚀 Marketing Research API Dashboard</div>', unsafe_allow_html=True)
    st.markdown("**API-Based Architecture with Real-time Monitoring**")
    
    # Check API connection
    if not check_api_connection():
        st.stop()
    
    # Sidebar navigation
    st.sidebar.markdown("## 📊 Dashboard Navigation")
    
    page = st.sidebar.selectbox(
        "Select Page",
        ["🚀 Start Analysis", "📊 System Metrics", "📚 Analysis History"],
        index=0
    )
    
    # Main content based on page selection
    if page == "🚀 Start Analysis":
        
        # Check if we have a running analysis
        if st.session_state.current_analysis_id:
            if st.session_state.get('show_results', False):
                render_analysis_results(st.session_state.current_analysis_id)
                
                if st.button("🔙 Back to Monitoring"):
                    st.session_state.show_results = False
                    st.rerun()
            else:
                render_real_time_monitoring(st.session_state.current_analysis_id)
        else:
            # Configuration form
            config = render_configuration_form()
            
            if config:
                # Start analysis
                with st.spinner("Starting analysis..."):
                    result = st.session_state.api_client.start_analysis(config)
                
                if result.get("success"):
                    st.session_state.current_analysis_id = result["analysis_id"]
                    st.success(f"✅ Analysis started successfully! ID: {result['analysis_id']}")
                    st.rerun()
                else:
                    st.error(f"❌ Failed to start analysis: {result.get('error')}")
    
    elif page == "📊 System Metrics":
        render_system_metrics()
        
        # Auto-refresh option
        if st.checkbox("🔄 Auto Refresh Metrics"):
            time.sleep(5)
            st.rerun()
    
    elif page == "📚 Analysis History":
        render_analysis_history()
    
    # Footer
    st.markdown("---")
    st.markdown("**🔗 API-Based Dashboard** | Connected to FastAPI Backend | Real-time Updates Enabled")

if __name__ == "__main__":
    main()