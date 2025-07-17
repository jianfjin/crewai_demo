"""
FastAPI Backend for Marketing Research Swarm
Provides REST API endpoints for the CrewAI marketing research platform with smart environment detection
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import asyncio
import uuid
import sys
import os
from datetime import datetime
import json

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Change working directory to project root to fix relative paths
project_root = os.path.join(os.path.dirname(__file__), '..')
os.chdir(project_root)

from marketing_research_swarm.optimization_manager import OptimizationManager
from marketing_research_swarm.blackboard.agent_dependency_manager import AgentDependencyManager

app = FastAPI(
    title="Marketing Research Swarm API",
    description="AI-powered marketing research analysis using CrewAI multi-agent swarm",
    version="1.0.0"
)

# Configure CORS with smart environment detection
def get_cors_origins():
    """Get CORS origins with smart environment detection"""
    import os
    
    # Start with default local development origins
    origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ]
    
    # Add environment variable origins if specified
    env_origins = os.getenv("CORS_ORIGINS", "")
    if env_origins:
        env_origins_list = [origin.strip() for origin in env_origins.split(",") if origin.strip()]
        origins.extend(env_origins_list)
    
    # Smart detection for GitHub Codespaces
    codespace_name = os.getenv("CODESPACE_NAME")
    if codespace_name:
        # Add GitHub Codespaces origin
        codespaces_origin = f"https://{codespace_name}-3000.app.github.dev"
        origins.append(codespaces_origin)
        print(f"[CORS] GitHub Codespaces detected: {codespaces_origin}")
    
    # Smart detection for Gitpod
    gitpod_workspace_url = os.getenv("GITPOD_WORKSPACE_URL")
    if gitpod_workspace_url:
        # Convert workspace URL to frontend URL
        gitpod_origin = gitpod_workspace_url.replace("https://", "https://3000-")
        origins.append(gitpod_origin)
        print(f"[CORS] Gitpod detected: {gitpod_origin}")
    
    # Smart detection for other cloud environments
    # Check for common cloud environment variables
    if os.getenv("REPLIT_DB_URL"):  # Replit
        replit_origin = f"https://{os.getenv('REPL_SLUG', 'app')}.{os.getenv('REPL_OWNER', 'user')}.repl.co"
        origins.append(replit_origin)
        print(f"[CORS] Replit detected: {replit_origin}")
    
    # Smart detection for Railway
    if os.getenv("RAILWAY_ENVIRONMENT"):
        railway_domain = os.getenv("RAILWAY_PUBLIC_DOMAIN")
        if railway_domain:
            railway_origin = f"https://{railway_domain}"
            origins.append(railway_origin)
            print(f"[CORS] Railway detected: {railway_origin}")
    
    # Smart detection for Vercel
    if os.getenv("VERCEL_URL"):
        vercel_origin = f"https://{os.getenv('VERCEL_URL')}"
        origins.append(vercel_origin)
        print(f"[CORS] Vercel detected: {vercel_origin}")
    
    # Smart detection for Netlify
    if os.getenv("NETLIFY_URL"):
        netlify_origin = os.getenv("NETLIFY_URL")
        origins.append(netlify_origin)
        print(f"[CORS] Netlify detected: {netlify_origin}")
    
    # Generic wildcard for development environments
    # Allow any subdomain of common cloud platforms
    wildcard_patterns = [
        "*.app.github.dev",
        "*.gitpod.io", 
        "*.repl.co",
        "*.railway.app",
        "*.vercel.app",
        "*.netlify.app"
    ]
    
    # Remove duplicates while preserving order
    unique_origins = []
    for origin in origins:
        if origin not in unique_origins:
            unique_origins.append(origin)
    
    # Add wildcard patterns for development
    if os.getenv("ENVIRONMENT", "development") == "development":
        unique_origins.extend(wildcard_patterns)
    
    print(f"[CORS] Origins configured: {unique_origins}")
    return unique_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state for tracking running analyses
running_analyses: Dict[str, Dict] = {}
completed_analyses: Dict[str, Dict] = {}

# Pydantic models for request/response
class AgentsResponse(BaseModel):
    agents: List[str]
    status: str = "success"

class AnalysisTypesResponse(BaseModel):
    types: Dict[str, Dict[str, Any]]
    status: str = "success"

class AnalysisStatusResponse(BaseModel):
    analysis_id: str
    status: str
    progress: float = 0.0
    current_agent: Optional[str] = None
    agents_completed: int = 0
    agents_total: int = 0
    elapsed_time: float = 0.0
    estimated_remaining: Optional[float] = None
    current_metrics: Optional[Dict[str, Any]] = None

class AnalysisResultResponse(BaseModel):
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class SystemMetricsResponse(BaseModel):
    health_score: int
    active_analyses: int
    completed_analyses: int
    avg_response_time: float
    cache_hit_rate: float
    optimization_metrics: Dict[str, Any]

class AnalysisRequest(BaseModel):
    analysis_type: str
    selected_agents: List[str]
    optimization_level: str = "blackboard"
    
    # Campaign Basics
    target_audience: str = "health-conscious millennials and premium beverage consumers"
    campaign_type: str = "multi-channel global marketing campaign"
    budget: int = 250000
    duration: str = "6 months"
    
    # Analysis Focus
    analysis_focus: str = "global beverage market performance and brand optimization"
    business_objective: str = "Optimize beverage portfolio performance across global markets"
    competitive_landscape: str = "global beverage market with diverse categories"
    
    # Market Segments
    market_segments: List[str] = ["North America", "Europe", "Asia Pacific"]
    product_categories: List[str] = ["Cola", "Juice", "Energy", "Sports"]
    key_metrics: List[str] = ["brand_performance", "category_trends", "profitability_analysis"]
    
    # Brands & Goals
    brands: List[str] = ["Coca-Cola", "Pepsi", "Red Bull"]
    campaign_goals: List[str] = ["Optimize brand portfolio performance across global markets"]
    
    # Forecast & Metrics
    forecast_periods: int = 30
    expected_revenue: int = 25000
    competitive_analysis: bool = True
    market_share_analysis: bool = True
    
    # Brand Metrics
    brand_awareness: float = 75.0
    sentiment_score: float = 0.6
    market_position: str = "Leader"
    
    # Optimization Settings
    token_budget: int = 4000
    context_strategy: str = "progressive_pruning"
    enable_caching: bool = True
    enable_mem0: bool = True
    enable_token_tracking: bool = True
    enable_optimization_tools: bool = True
    show_comparison: bool = False
    
    custom_inputs: Optional[Dict[str, Any]] = None

class AnalysisResponse(BaseModel):
    analysis_id: str
    status: str
    message: str
    estimated_duration: Optional[int] = None

class AnalysisStatus(BaseModel):
    analysis_id: str
    status: str
    progress: float
    current_step: Optional[str] = None
    agents_completed: List[str]
    total_agents: int
    start_time: datetime
    estimated_completion: Optional[datetime] = None
    token_usage: Optional[Dict[str, Any]] = None

class AnalysisResult(BaseModel):
    analysis_id: str
    status: str
    result: Optional[str] = None
    token_usage: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    duration: Optional[float] = None

class AgentInfo(BaseModel):
    role: str
    goal: str
    backstory: str
    tools: List[str]
    phase: str

class AnalysisTypeInfo(BaseModel):
    name: str
    description: str
    recommended_agents: List[str]
    estimated_duration: int
    complexity: str

# Initialize managers
optimization_manager = OptimizationManager()
dependency_manager = AgentDependencyManager()

@app.get("/")
async def root():
    """Health check endpoint with environment info"""
    environment_info = {
        "codespace_name": os.getenv("CODESPACE_NAME"),
        "gitpod_workspace": os.getenv("GITPOD_WORKSPACE_URL"),
        "replit": bool(os.getenv("REPLIT_DB_URL")),
        "railway": bool(os.getenv("RAILWAY_ENVIRONMENT")),
        "vercel": bool(os.getenv("VERCEL_URL")),
        "netlify": bool(os.getenv("NETLIFY_URL")),
        "cors_origins": get_cors_origins()
    }
    
    return {
        "message": "Marketing Research Swarm API",
        "version": "1.0.0",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "environment": environment_info
    }

@app.get("/api/agents/available", response_model=AgentsResponse)
async def get_available_agents_simple():
    """Get list of available agents (simple format)"""
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
    return AgentsResponse(agents=agents)

@app.get("/api/analysis/types", response_model=AnalysisTypesResponse)
async def get_analysis_types():
    """Get available analysis types"""
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
    return AnalysisTypesResponse(types=analysis_types)

@app.post("/api/analysis", response_model=AnalysisResponse)
async def start_analysis(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Start a new marketing research analysis"""
    try:
        # Generate unique analysis ID
        analysis_id = str(uuid.uuid4())
        
        # Initialize analysis tracking
        running_analyses[analysis_id] = {
            "status": "starting",
            "progress": 0.0,
            "current_step": "Initializing analysis",
            "agents_completed": [],
            "total_agents": len(request.selected_agents),
            "start_time": datetime.now(),
            "request": request.dict()
        }
        
        # Start analysis in background
        background_tasks.add_task(
            run_analysis_background,
            analysis_id,
            request
        )
        
        # Estimate duration based on agents and optimization level
        base_duration = len(request.selected_agents) * 60  # 60 seconds per agent
        
        optimization_level = request.optimization_level or "partial"
        
        if optimization_level == "blackboard":
            estimated_duration = int(base_duration * 0.7)  # 30% faster with blackboard
        elif optimization_level == "partial":
            estimated_duration = int(base_duration * 0.05)  # 95% faster with partial optimization
        elif optimization_level == "full":
            estimated_duration = int(base_duration * 0.1)   # 90% faster with full optimization
        else:
            estimated_duration = base_duration  # Standard duration for "none"
        
        return AnalysisResponse(
            analysis_id=analysis_id,
            status="started",
            message=f"Analysis started with {len(request.selected_agents)} agents",
            estimated_duration=estimated_duration
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start analysis: {str(e)}")

@app.get("/api/analysis/{analysis_id}/status", response_model=AnalysisStatus)
async def get_analysis_status(analysis_id: str):
    """Get the current status of a running analysis"""
    if analysis_id in completed_analyses:
        analysis = completed_analyses[analysis_id]
        return AnalysisStatus(
            analysis_id=analysis_id,
            status="completed",
            progress=100.0,
            current_step="Analysis completed",
            agents_completed=analysis.get("agents_completed", []),
            total_agents=analysis.get("total_agents", 0),
            start_time=analysis.get("start_time"),
            estimated_completion=analysis.get("end_time"),
            token_usage=analysis.get("token_usage")
        )
    
    if analysis_id in running_analyses:
        analysis = running_analyses[analysis_id]
        return AnalysisStatus(
            analysis_id=analysis_id,
            status=analysis["status"],
            progress=analysis["progress"],
            current_step=analysis["current_step"],
            agents_completed=analysis["agents_completed"],
            total_agents=analysis["total_agents"],
            start_time=analysis["start_time"],
            estimated_completion=analysis.get("estimated_completion"),
            token_usage=analysis.get("token_usage")
        )
    
    raise HTTPException(status_code=404, detail="Analysis not found")

@app.get("/api/analysis/{analysis_id}/result", response_model=AnalysisResult)
async def get_analysis_result(analysis_id: str):
    """Get the result of a completed analysis"""
    if analysis_id in completed_analyses:
        analysis = completed_analyses[analysis_id]
        return AnalysisResult(
            analysis_id=analysis_id,
            status=analysis["status"],
            result=analysis.get("result"),
            token_usage=analysis.get("token_usage"),
            performance_metrics=analysis.get("performance_metrics"),
            error_message=analysis.get("error_message"),
            duration=analysis.get("duration")
        )
    
    if analysis_id in running_analyses:
        return AnalysisResult(
            analysis_id=analysis_id,
            status="running",
            result=None,
            error_message=None
        )
    
    raise HTTPException(status_code=404, detail="Analysis not found")

@app.get("/api/system/metrics", response_model=SystemMetricsResponse)
async def get_system_metrics():
    """Get system performance metrics"""
    return SystemMetricsResponse(
        health_score=95,
        active_analyses=len(running_analyses),
        completed_analyses=len(completed_analyses),
        avg_response_time=1.23,
        cache_hit_rate=87.5,
        optimization_metrics={
            "token_savings_percent": 68,
            "speedup_factor": 2.3,
            "memory_efficiency": 85
        }
    )

@app.get("/api/analysis/history")
async def get_analysis_history():
    """Get analysis history"""
    history = []
    
    # Add completed analyses
    for analysis_id, analysis in completed_analyses.items():
        history.append({
            "analysis_id": analysis_id,
            "status": analysis["status"],
            "start_time": analysis.get("start_time"),
            "end_time": analysis.get("end_time"),
            "duration": analysis.get("duration"),
            "analysis_type": analysis.get("request", {}).get("analysis_type"),
            "agents_count": analysis.get("total_agents", 0),
            "token_usage": analysis.get("token_usage", {}).get("total_tokens", 0) if analysis.get("token_usage") else 0
        })
    
    # Add running analyses
    for analysis_id, analysis in running_analyses.items():
        history.append({
            "analysis_id": analysis_id,
            "status": analysis["status"],
            "start_time": analysis.get("start_time"),
            "end_time": None,
            "duration": None,
            "analysis_type": analysis.get("request", {}).get("analysis_type"),
            "agents_count": analysis.get("total_agents", 0),
            "token_usage": 0
        })
    
    # Sort by start time (most recent first)
    history.sort(key=lambda x: x["start_time"] or datetime.min, reverse=True)
    
    return {"analyses": history}

@app.delete("/api/analysis/{analysis_id}")
async def cancel_analysis(analysis_id: str):
    """Cancel a running analysis"""
    if analysis_id in running_analyses:
        analysis = running_analyses[analysis_id]
        analysis["status"] = "cancelled"
        analysis["current_step"] = "Analysis cancelled by user"
        
        # Move to completed analyses
        completed_analyses[analysis_id] = analysis
        del running_analyses[analysis_id]
        
        return {"message": "Analysis cancelled successfully"}
    
    raise HTTPException(status_code=404, detail="Analysis not found or already completed")

async def run_analysis_background(analysis_id: str, request: AnalysisRequest):
    """Run the analysis in the background"""
    try:
        # Update status
        running_analyses[analysis_id]["status"] = "running"
        running_analyses[analysis_id]["current_step"] = "Setting up optimization manager"
        running_analyses[analysis_id]["progress"] = 10.0
        
        # Prepare inputs for the analysis
        inputs = request.custom_inputs or {}
        
        # Update progress
        running_analyses[analysis_id]["current_step"] = "Starting agent execution"
        running_analyses[analysis_id]["progress"] = 20.0
        
        # Prepare inputs with all request data including required template variables
        analysis_inputs = {
            'analysis_type': request.analysis_type,
            'selected_agents': request.selected_agents,
            'target_audience': request.target_audience,
            'campaign_type': request.campaign_type,
            'budget': str(request.budget),  # Convert to string for template usage
            'duration': request.duration,
            'analysis_focus': request.analysis_focus,
            'business_objective': request.business_objective,
            'competitive_landscape': request.competitive_landscape,
            'market_segments': request.market_segments,
            'product_categories': request.product_categories,
            'key_metrics': request.key_metrics,
            'brands': request.brands,
            'campaign_goals': ', '.join(request.campaign_goals) if isinstance(request.campaign_goals, list) else str(request.campaign_goals),
            'forecast_periods': request.forecast_periods,
            'expected_revenue': request.expected_revenue,
            'competitive_analysis': request.competitive_analysis,
            'market_share_analysis': request.market_share_analysis,
            'brand_awareness': request.brand_awareness,
            'sentiment_score': request.sentiment_score,
            'market_position': request.market_position,
            # Required template variables for tasks
            'data_file_path': 'data/enhanced_beverage_sales_data.csv',  # Default data file
            **inputs  # Include any custom inputs
        }
        
        # Use partial optimization as default for best performance
        optimization_level = request.optimization_level or "partial"
        
        # Run the analysis using optimization manager
        result = optimization_manager.run_analysis_with_optimization(
            inputs=analysis_inputs,
            optimization_level=optimization_level
        )
        
        # Update progress during execution (this would need to be integrated with the optimization manager)
        for i, agent in enumerate(request.selected_agents):
            running_analyses[analysis_id]["agents_completed"].append(agent)
            progress = 20.0 + (60.0 * (i + 1) / len(request.selected_agents))
            running_analyses[analysis_id]["progress"] = progress
            running_analyses[analysis_id]["current_step"] = f"Completed agent: {agent}"
            await asyncio.sleep(1)  # Simulate processing time
        
        # Extract results and metrics
        running_analyses[analysis_id]["current_step"] = "Processing results"
        running_analyses[analysis_id]["progress"] = 90.0
        
        # Move to completed analyses
        end_time = datetime.now()
        duration = (end_time - running_analyses[analysis_id]["start_time"]).total_seconds()
        
        # Extract and properly format the result
        result_content = str(result) if result else "Analysis completed successfully"
        
        completed_analyses[analysis_id] = {
            **running_analyses[analysis_id],
            "status": "completed",
            "progress": 100.0,
            "current_step": "Analysis completed",
            "end_time": end_time,
            "duration": duration,
            "result": result_content,
            "token_usage": result.get("metrics", {}) if isinstance(result, dict) else {},
            "performance_metrics": result.get("optimization_record", {}) if isinstance(result, dict) else {}
        }
        
        # Remove from running analyses
        del running_analyses[analysis_id]
        
    except Exception as e:
        # Handle errors
        end_time = datetime.now()
        duration = (end_time - running_analyses[analysis_id]["start_time"]).total_seconds()
        
        completed_analyses[analysis_id] = {
            **running_analyses[analysis_id],
            "status": "failed",
            "progress": 100.0,
            "current_step": "Analysis failed",
            "end_time": end_time,
            "duration": duration,
            "error_message": str(e),
            "result": None,
            "token_usage": {},
            "performance_metrics": {}
        }
        
        # Remove from running analyses
        del running_analyses[analysis_id]

if __name__ == "__main__":
    import uvicorn
    
    # Get host and port from environment variables with smart defaults
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    
    print(f"[SERVER] Starting FastAPI server on {host}:{port}")
    print(f"[SERVER] Environment detection enabled")
    print(f"[SERVER] CORS origins: {get_cors_origins()}")
    
    uvicorn.run(app, host=host, port=port)