"""
Optimized Workflow Wrapper for Dashboard
Contains the OptimizedWorkflowWrapper class that wraps the optimized marketing workflow.
"""

import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class OptimizedWorkflowWrapper:
    """Wrapper for optimized marketing workflow."""
    
    def __init__(self, checkpoint_path=None, optimization_level="full", **kwargs):
        try:
            from marketing_research_swarm.langgraph_workflow.optimized_workflow import OptimizedMarketingWorkflow
            self.workflow = OptimizedMarketingWorkflow(checkpoint_path=checkpoint_path, optimization_level=optimization_level)
        except ImportError as e:
            logger.error(f"Failed to import OptimizedMarketingWorkflow: {e}")
            self.workflow = None
            
        self.available_agents = ["market_research_analyst", "data_analyst", "content_strategist", 
                               "competitive_analyst", "brand_performance_specialist", "forecasting_specialist"]
        self.checkpoint_path = checkpoint_path
        self.optimization_level = optimization_level
    
    def run(self, inputs, optimization_level="none"):
        """Run the optimized workflow with formatted inputs."""
        if not self.workflow:
            raise RuntimeError("OptimizedMarketingWorkflow not available")
            
        # Convert inputs to proper format for optimized workflow
        formatted_inputs = self._format_inputs(inputs)
        return self.workflow.execute_optimized_workflow(**formatted_inputs, optimization_level=optimization_level)
    
    def execute_workflow(self, selected_agents=None, target_audience="", campaign_type="", 
                       budget=0, duration="", analysis_focus="", optimization_level="none", **kwargs):
        """Execute the optimized workflow with all parameters."""
        if not self.workflow:
            raise RuntimeError("OptimizedMarketingWorkflow not available")
            
        # Use the optimized workflow's execute_optimized_workflow method
        return self.workflow.execute_optimized_workflow(
            selected_agents=selected_agents or ["market_research_analyst", "data_analyst"],
            target_audience=target_audience,
            campaign_type=campaign_type,
            budget=budget,
            duration=duration,
            analysis_focus=analysis_focus,
            optimization_level=optimization_level,
            **kwargs
        )
    
    def _format_inputs(self, inputs):
        """Format inputs to match optimized workflow expectations."""
        if isinstance(inputs, dict):
            # Ensure required keys exist with defaults
            formatted = {
                'target_audience': inputs.get('target_audience', 'target audience'),
                'campaign_type': inputs.get('campaign_type', 'marketing campaign'),
                'budget': inputs.get('budget', 50000),
                'duration': inputs.get('duration', '6 months'),
                'analysis_focus': inputs.get('analysis_focus', 'market analysis'),
                'selected_agents': inputs.get('selected_agents', ["market_research_analyst", "data_analyst"])
            }
            # Add any additional keys from original inputs
            for key, value in inputs.items():
                if key not in formatted:
                    formatted[key] = value
            return formatted
        else:
            # If inputs is not a dict, create default structure
            return {
                'target_audience': 'target audience',
                'campaign_type': 'marketing campaign',
                'budget': 50000,
                'duration': '6 months',
                'analysis_focus': 'market analysis',
                'selected_agents': ["market_research_analyst", "data_analyst"]
            }
    
    def create_initial_state(self, **kwargs):
        """Create initial state for the workflow."""
        from datetime import datetime
        return {"workflow_id": f"langgraph_{datetime.now().strftime('%Y%m%d_%H%M%S')}", **kwargs}