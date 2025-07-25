#!/usr/bin/env python3
"""
Enhanced LangGraph Workflow Runner for Marketing Research

This script provides an improved interface to run the LangGraph-based
marketing research workflow with better error handling and monitoring.
"""

import sys
import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional

# Add the project root to the Python path
sys.path.append('src')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedWorkflowRunner:
    """Enhanced runner for LangGraph marketing research workflows."""
    
    def __init__(self):
        """Initialize the enhanced workflow runner."""
        self.workflow = None
        self.results_history = []
        
    def initialize_workflow(self):
        """Initialize the LangGraph workflow with error handling."""
        try:
            # Try to import and initialize the workflow
            from marketing_research_swarm.langgraph_workflow.workflow import MarketingResearchWorkflow
            self.workflow = MarketingResearchWorkflow()
            logger.info("✅ LangGraph workflow initialized successfully")
            return True
        except ImportError as e:
            logger.error(f"❌ Failed to import LangGraph workflow: {e}")
            logger.info("💡 This might be due to missing LangGraph dependencies")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to initialize workflow: {e}")
            return False
    
    def run_workflow(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run the workflow with enhanced error handling and monitoring."""
        
        if not self.workflow:
            if not self.initialize_workflow():
                return {
                    "success": False,
                    "error": "Failed to initialize workflow",
                    "fallback_available": True
                }
        
        try:
            logger.info("🚀 Starting LangGraph workflow execution")
            logger.info(f"Selected agents: {inputs.get('selected_agents', 'auto-select')}")
            
            # Execute the workflow
            result = self.workflow.run(inputs)
            
            # Store result in history
            self.results_history.append({
                "timestamp": datetime.now(),
                "inputs": inputs,
                "result": result
            })
            
            if result.get("success"):
                logger.info("✅ LangGraph workflow completed successfully")
                return result
            else:
                logger.error(f"❌ LangGraph workflow failed: {result.get('error')}")
                return result
                
        except Exception as e:
            logger.error(f"❌ Workflow execution error: {e}")
            return {
                "success": False,
                "error": str(e),
                "fallback_available": True
            }
    
    def run_with_crewai_fallback(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run with CrewAI fallback if LangGraph fails."""
        
        # First try LangGraph
        result = self.run_workflow(inputs)
        
        if result.get("success"):
            return result
        
        # If LangGraph fails, try CrewAI fallback
        logger.info("🔄 LangGraph failed, attempting CrewAI fallback...")
        
        try:
            from marketing_research_swarm.crew_with_tracking import MarketingResearchCrew
            
            # Configuration paths
            agents_config_path = 'src/marketing_research_swarm/config/agents.yaml'
            tasks_config_path = 'src/marketing_research_swarm/config/tasks.yaml'
            
            # Initialize and run CrewAI crew
            crew = MarketingResearchCrew(agents_config_path, tasks_config_path)
            crew_result = crew.kickoff(inputs)
            
            logger.info("✅ CrewAI fallback completed successfully")
            
            return {
                "success": True,
                "workflow_type": "crewai_fallback",
                "result": crew_result,
                "original_error": result.get("error")
            }
            
        except Exception as e:
            logger.error(f"❌ CrewAI fallback also failed: {e}")
            return {
                "success": False,
                "error": f"Both LangGraph and CrewAI failed. LangGraph: {result.get('error')}, CrewAI: {str(e)}"
            }
    
    def get_available_agents(self) -> List[str]:
        """Get list of available agents."""
        try:
            if self.workflow:
                return self.workflow.available_agents
            else:
                # Return default agent list
                return [
                    'market_research_analyst',
                    'competitive_analyst', 
                    'data_analyst',
                    'content_strategist',
                    'creative_copywriter',
                    'campaign_optimizer',
                    'brand_performance_specialist',
                    'forecasting_specialist'
                ]
        except Exception as e:
            logger.error(f"Error getting available agents: {e}")
            return []
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow status if available."""
        try:
            if self.workflow:
                return self.workflow.get_workflow_status(workflow_id)
            return None
        except Exception as e:
            logger.error(f"Error getting workflow status: {e}")
            return None
    
    def export_results_history(self, filename: str = None) -> str:
        """Export results history to JSON file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"langgraph_results_history_{timestamp}.json"
        
        try:
            # Convert datetime objects to strings for JSON serialization
            serializable_history = []
            for entry in self.results_history:
                serializable_entry = {
                    "timestamp": entry["timestamp"].isoformat(),
                    "inputs": entry["inputs"],
                    "result": entry["result"]
                }
                serializable_history.append(serializable_entry)
            
            with open(filename, 'w') as f:
                json.dump(serializable_history, f, indent=2)
            
            logger.info(f"✅ Results history exported to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"❌ Failed to export results history: {e}")
            return ""


def main():
    """Main function for command-line usage."""
    
    print("🚀 Enhanced LangGraph Marketing Research Workflow Runner")
    print("=" * 60)
    
    # Initialize the runner
    runner = EnhancedWorkflowRunner()
    
    # Test inputs
    test_inputs = {
        "target_audience": "health-conscious millennials interested in premium beverages",
        "campaign_type": "multi-channel digital marketing",
        "budget": 100000,
        "duration": "6 months",
        "data_file_path": "data/beverage_sales.csv",
        "analysis_focus": "beverage market performance and optimization",
        "business_objective": "Launch new organic beverage line and optimize existing product performance",
        "key_metrics": ["sales_trends", "regional_performance", "product_performance", "roi_analysis"],
        "competitive_landscape": "premium beverage market with focus on organic and health-conscious products",
        "market_segments": ["North", "South"],
        "product_lines": ["Green Tea", "Latte"],
        "selected_agents": ["market_research_analyst", "data_analyst", "content_strategist"]
    }
    
    print("📋 Test Configuration:")
    print(f"  Target Audience: {test_inputs['target_audience']}")
    print(f"  Campaign Budget: ${test_inputs['budget']:,}")
    print(f"  Selected Agents: {', '.join(test_inputs['selected_agents'])}")
    print()
    
    # Run the workflow with fallback
    print("🔄 Running workflow with fallback support...")
    result = runner.run_with_crewai_fallback(test_inputs)
    
    print("\n" + "=" * 60)
    print("📊 EXECUTION RESULTS")
    print("=" * 60)
    
    if result.get("success"):
        print("✅ Workflow completed successfully!")
        print(f"Workflow Type: {result.get('workflow_type', 'langgraph')}")
        
        if result.get('workflow_id'):
            print(f"Workflow ID: {result['workflow_id']}")
        
        if result.get('result'):
            print("📈 Analysis Results Available")
            
        # Export results
        if runner.results_history:
            export_file = runner.export_results_history()
            if export_file:
                print(f"📁 Results exported to: {export_file}")
    else:
        print("❌ Workflow execution failed")
        print(f"Error: {result.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 60)
    print("Available Agents:", ", ".join(runner.get_available_agents()))
    print("✅ Enhanced LangGraph runner test completed!")


if __name__ == "__main__":
    main()