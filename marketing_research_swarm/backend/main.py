"""
FastAPI Backend for Marketing Research Swarm
Provides REST API endpoints for the CrewAI marketing research platform
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

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://127.0.0.1:3000",
        "https://super-space-guide-jxg7rrvxg72jr56-3000.app.github.dev"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state for tracking running analyses
running_analyses: Dict[str, Dict] = {}
completed_analyses: Dict[str, Dict] = {}

# Pydantic models for request/response
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
    """Health check endpoint"""
    return {
        "message": "Marketing Research Swarm API",
        "version": "1.0.0",
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/agents", response_model=List[AgentInfo])
async def get_available_agents():
    """Get list of available agents with their configurations"""
    try:
        import yaml
        import os
        
        # Load agents configuration directly
        agents_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'marketing_research_swarm', 'config', 'agents.yaml')
        with open(agents_path, 'r') as file:
            agents_config = yaml.safe_load(file)
        
        agents = []
        
        for agent_key, agent_config in agents_config.items():
            # Get phase information from dependency manager
            try:
                phase = dependency_manager.get_agent_phase(agent_key)
            except:
                # Fallback phase mapping
                phase_mapping = {
                    'market_research_analyst': 'FOUNDATION',
                    'data_analyst': 'FOUNDATION',
                    'competitive_analyst': 'ANALYSIS',
                    'brand_performance_specialist': 'ANALYSIS',
                    'brand_strategist': 'STRATEGY',
                    'campaign_optimizer': 'STRATEGY',
                    'forecasting_specialist': 'STRATEGY',
                    'content_strategist': 'CONTENT',
                    'creative_copywriter': 'CONTENT'
                }
                phase = phase_mapping.get(agent_key, 'UNKNOWN')
            
            agent_info = AgentInfo(
                role=agent_config.get('role', agent_key),
                goal=agent_config.get('goal', ''),
                backstory=agent_config.get('backstory', ''),
                tools=agent_config.get('tools', []),
                phase=phase
            )
            agents.append(agent_info)
        
        return agents
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load agents: {str(e)}")

@app.get("/api/analysis-types", response_model=List[AnalysisTypeInfo])
async def get_analysis_types():
    """Get available analysis types with recommended agent combinations"""
    try:
        # Try to get analysis types from dependency manager
        from marketing_research_swarm.blackboard.agent_dependency_manager import get_dependency_manager
        dependency_manager = get_dependency_manager()
        analysis_types_config = dependency_manager.get_analysis_types()
        
        analysis_types = []
        for key, config in analysis_types_config.items():
            if key != "custom":
                analysis_types.append(AnalysisTypeInfo(
                    name=key,
                    description=config["description"],
                    recommended_agents=config["agents"],
                    estimated_duration=int(config["expected_duration"].replace("min", "")) if "min" in config["expected_duration"] else 180,
                    complexity=config.get("complexity", "Medium")
                ))
        
        # Add custom option
        analysis_types.append(AnalysisTypeInfo(
            name="custom",
            description="Custom analysis with user-selected agents",
            recommended_agents=[],
            estimated_duration=120,
            complexity="Variable"
        ))
        
    except Exception as e:
        print(f"Error loading dynamic analysis types: {e}")
        # Fallback to static analysis types
        analysis_types = [
            AnalysisTypeInfo(
                name="roi_analysis",
                description="Comprehensive ROI and profitability analysis",
                recommended_agents=["market_research_analyst", "data_analyst", "campaign_optimizer"],
                estimated_duration=180,
                complexity="Medium"
            ),
            AnalysisTypeInfo(
                name="brand_performance",
                description="Brand performance and competitive analysis",
                recommended_agents=["brand_performance_specialist", "competitive_analyst", "market_research_analyst"],
                estimated_duration=240,
                complexity="High"
            ),
            AnalysisTypeInfo(
                name="sales_forecast",
                description="Sales forecasting and trend analysis",
                recommended_agents=["forecasting_specialist", "data_analyst", "market_research_analyst"],
                estimated_duration=200,
                complexity="Medium"
            ),
            AnalysisTypeInfo(
                name="content_strategy",
                description="Content strategy and creative campaign development",
                recommended_agents=["content_strategist", "creative_copywriter", "market_research_analyst"],
                estimated_duration=160,
                complexity="Medium"
            ),
            AnalysisTypeInfo(
                name="comprehensive",
                description="Complete marketing research with all agents",
                recommended_agents=["market_research_analyst", "data_analyst", "competitive_analyst", "brand_performance_specialist", "content_strategist"],
                estimated_duration=300,
                complexity="High"
            ),
            AnalysisTypeInfo(
                name="custom",
                description="Custom analysis with user-selected agents",
                recommended_agents=[],
                estimated_duration=120,
                complexity="Variable"
            )
        ]
    
    return analysis_types

@app.post("/api/analysis/start", response_model=AnalysisResponse)
async def start_analysis(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """Start a new marketing research analysis"""
    try:
        # Generate unique analysis ID
        analysis_id = str(uuid.uuid4())
        
        # Validate agents - load from config file directly
        import yaml
        import os
        try:
            agents_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'marketing_research_swarm', 'config', 'agents.yaml')
            with open(agents_path, 'r') as file:
                agents_config = yaml.safe_load(file)
            available_agents = list(agents_config.keys())
            invalid_agents = [agent for agent in request.selected_agents if agent not in available_agents]
            if invalid_agents:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Invalid agents: {invalid_agents}. Available agents: {available_agents}"
                )
        except Exception as e:
            print(f"Warning: Could not validate agents: {e}")
            # Continue without validation
        
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
        
        # Use partial optimization as default for best performance
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

@app.get("/api/analysis/history")
async def get_analysis_history():
    """Get history of all analyses"""
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

def extract_blackboard_result(result):
    """Extract result content from various CrewAI and blackboard output formats"""
    try:
        if isinstance(result, dict):
            # Handle blackboard crew output format
            if 'result' in result:
                crew_result = result['result']
                
                # Handle CrewAI CrewOutput objects
                if hasattr(crew_result, 'raw'):
                    return crew_result.raw
                elif hasattr(crew_result, 'result'):
                    return crew_result.result
                elif hasattr(crew_result, 'tasks_output'):
                    # Extract from tasks output
                    tasks_output = crew_result.tasks_output
                    if tasks_output and len(tasks_output) > 0:
                        # Get the last task output (final result)
                        final_task = tasks_output[-1]
                        if hasattr(final_task, 'raw'):
                            return final_task.raw
                        elif hasattr(final_task, 'result'):
                            return final_task.result
                        else:
                            return str(final_task)
                elif isinstance(crew_result, dict):
                    # Format as readable JSON
                    return json.dumps(crew_result, indent=2, default=str)
                else:
                    return str(crew_result)
            
            # Handle workflow summary format
            elif 'workflow_summary' in result:
                workflow_summary = result['workflow_summary']
                
                # Try to extract agent results
                if 'agent_results' in workflow_summary:
                    agent_results = workflow_summary['agent_results']
                    
                    # Combine all agent results into a comprehensive report
                    combined_results = []
                    for agent_name, agent_data in agent_results.items():
                        if 'results' in agent_data:
                            results = agent_data['results']
                            combined_results.append(f"## {agent_name.replace('_', ' ').title()} Results\n")
                            
                            for key, value in results.items():
                                if isinstance(value, str) and len(value) > 100:
                                    combined_results.append(f"**{key}**: {value}\n")
                                elif isinstance(value, dict):
                                    combined_results.append(f"**{key}**: {json.dumps(value, indent=2)}\n")
                                else:
                                    combined_results.append(f"**{key}**: {value}\n")
                            combined_results.append("\n---\n")
                    
                    if combined_results:
                        return "\n".join(combined_results)
                
                # Fallback to workflow summary
                return json.dumps(workflow_summary, indent=2, default=str)
            
            # Handle direct result dictionary
            else:
                return json.dumps(result, indent=2, default=str)
        
        # Handle string results
        elif isinstance(result, str):
            return result
        
        # Handle other object types
        else:
            # Try to extract from CrewAI objects
            if hasattr(result, 'raw'):
                return result.raw
            elif hasattr(result, 'result'):
                return result.result
            elif hasattr(result, 'tasks_output'):
                tasks_output = result.tasks_output
                if tasks_output and len(tasks_output) > 0:
                    final_task = tasks_output[-1]
                    if hasattr(final_task, 'raw'):
                        return final_task.raw
                    else:
                        return str(final_task)
            else:
                return str(result)
                
    except Exception as e:
        print(f"Error extracting blackboard result: {e}")
        return f"Analysis completed but result extraction failed: {str(e)}\n\nRaw result: {str(result)[:1000]}..."

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
        result_content = extract_blackboard_result(result)
        
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
    uvicorn.run(app, host="0.0.0.0", port=8000)