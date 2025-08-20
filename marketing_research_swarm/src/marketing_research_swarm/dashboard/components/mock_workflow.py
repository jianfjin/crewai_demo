"""
Mock Workflow Classes for Dashboard
Contains mock implementations of workflow classes used in the dashboard.
"""

import time
import random
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class MockLangGraphWorkflow:
    """Mock LangGraph workflow for testing and fallback scenarios."""
    
    def __init__(self, checkpoint_path=None, optimization_level="full", **kwargs):
        self.available_agents = ["market_research_analyst", "data_analyst", "content_strategist", 
                               "competitive_analyst", "brand_performance_specialist", "forecasting_specialist"]
        self.checkpoint_path = checkpoint_path
        self.optimization_level = optimization_level
    
    def run(self, inputs):
        """Run the mock workflow with simulated processing time."""
        # Simulate actual processing time
        time.sleep(2)
        
        return {
            "success": True,
            "workflow_id": f"langgraph_workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "status": "completed",
            "results": {
                "message": "LangGraph workflow executed successfully",
                "optimization_level": self.optimization_level,
                "agents_used": self.available_agents[:2],
                "analysis_summary": f"Analysis completed for {inputs.get('target_audience', 'target audience')}"
            },
            "agent_results": {
                "market_research_analyst": {
                    "analysis": f"Market research analysis for {inputs.get('target_audience', 'target audience')} in {inputs.get('campaign_type', 'campaign')} with budget ${inputs.get('budget', 0):,}"
                },
                "data_analyst": {
                    "analysis": f"Data analysis shows potential ROI of {random.randint(15, 35)}% for the {inputs.get('duration', 'campaign duration')} campaign"
                }
            },
            "execution_time": 2.0
        }
    
    def execute_workflow(self, selected_agents=None, target_audience="", campaign_type="", 
                       budget=0, duration="", analysis_focus="", analysis_type="marketing_research", **kwargs):
        """Execute the mock workflow with detailed results."""
        # Simulate actual processing time
        logger.info(f"Starting LangGraph workflow analysis for {target_audience}")
        time.sleep(3)  # Simulate real processing
        
        agents_to_use = selected_agents or self.available_agents[:2]
        
        # Generate more realistic results
        workflow_result = {
            "success": True,
            "workflow_id": f"langgraph_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "status": "completed",
            "final_state": {
                "selected_agents": agents_to_use,
                "target_audience": target_audience,
                "campaign_type": campaign_type,
                "budget": budget,
                "duration": duration,
                "analysis_focus": analysis_focus,
                "analysis_type": analysis_type
            },
            "summary": {
                "workflow_type": analysis_type,
                "execution_time": 3.0,
                "total_agents": len(agents_to_use),
                "completed_agents": len(agents_to_use),
                "success_rate": 1.0,
                "key_insights": [
                    f"Target audience '{target_audience}' shows high engagement potential",
                    f"Recommended budget allocation for {campaign_type} campaign",
                    f"Expected ROI: {random.randint(15, 35)}% over {duration}"
                ]
            },
            "agent_results": {}
        }
        
        # Generate realistic agent results
        for agent in agents_to_use:
            if agent == "market_research_analyst":
                workflow_result["agent_results"][agent] = {
                    "analysis": f"Market research for {target_audience}: High potential in {campaign_type} segment. Recommended budget: ${budget:,}",
                    "recommendations": ["Focus on digital channels", "Target 25-45 age group", "Emphasize value proposition"]
                }
            elif agent == "data_analyst":
                workflow_result["agent_results"][agent] = {
                    "analysis": f"Data analysis shows {random.randint(15, 35)}% potential ROI for {duration} campaign targeting {target_audience}",
                    "metrics": {"conversion_rate": f"{random.randint(3, 8)}%", "engagement_rate": f"{random.randint(12, 25)}%"}
                }
            elif agent == "content_strategist":
                workflow_result["agent_results"][agent] = {
                    "analysis": f"Content strategy for {target_audience}: Focus on {analysis_focus} messaging",
                    "content_recommendations": ["Educational content", "Case studies", "Social proof"]
                }
            else:
                workflow_result["agent_results"][agent] = {
                    "analysis": f"Analysis from {agent} for {analysis_type} targeting {target_audience}",
                    "insights": [f"Key insight from {agent}", f"Recommendation for {campaign_type}"]
                }
        
        logger.info(f"LangGraph workflow completed: {workflow_result['workflow_id']}")
        return workflow_result
    
    def create_initial_state(self, **kwargs):
        """Create initial state for the workflow."""
        return {"workflow_id": f"langgraph_{datetime.now().strftime('%Y%m%d_%H%M%S')}", **kwargs}