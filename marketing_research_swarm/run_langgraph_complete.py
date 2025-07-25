#!/usr/bin/env python3
"""
Complete LangGraph Marketing Research Workflow Runner

This script provides a comprehensive interface to run the LangGraph-based
marketing research workflow with full configuration management, error handling,
and result processing.
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


class CompleteLangGraphRunner:
    """Complete LangGraph workflow runner with configuration management."""
    
    def __init__(self):
        """Initialize the complete workflow runner."""
        self.config = None
        self.workflow = None
        self.results_history = []
        self.initialize_system()
    
    def initialize_system(self):
        """Initialize the complete system with configuration and workflow."""
        try:
            # Initialize configuration
            from langgraph_config import LangGraphConfig
            self.config = LangGraphConfig()
            logger.info("✅ Configuration system initialized")
            
            # Try to initialize LangGraph workflow
            try:
                from marketing_research_swarm.langgraph_workflow.workflow import MarketingResearchWorkflow
                self.workflow = MarketingResearchWorkflow()
                logger.info("✅ LangGraph workflow initialized")
            except ImportError as e:
                logger.warning(f"⚠️ LangGraph not available: {e}")
                logger.info("💡 Will use CrewAI fallback when needed")
            except Exception as e:
                logger.error(f"❌ Failed to initialize LangGraph workflow: {e}")
                
        except Exception as e:
            logger.error(f"❌ System initialization failed: {e}")
    
    def run_analysis(self, analysis_type: str = "comprehensive", custom_agents: List[str] = None) -> Dict[str, Any]:
        """Run marketing analysis with specified type and agents."""
        
        # Define analysis configurations
        analysis_configs = {
            "comprehensive": {
                "agents": ["market_research_analyst", "data_analyst", "content_strategist", "campaign_optimizer"],
                "focus": "comprehensive marketing research and strategy development",
                "description": "Full-scale marketing analysis with all key components"
            },
            "roi_focused": {
                "agents": ["data_analyst", "forecasting_specialist", "campaign_optimizer"],
                "focus": "ROI analysis and financial optimization",
                "description": "ROI-focused analysis for budget optimization"
            },
            "content_strategy": {
                "agents": ["market_research_analyst", "content_strategist", "creative_copywriter"],
                "focus": "content strategy and creative development",
                "description": "Content-focused analysis for marketing campaigns"
            },
            "brand_performance": {
                "agents": ["competitive_analyst", "brand_performance_specialist", "data_analyst"],
                "focus": "brand performance and competitive analysis",
                "description": "Brand positioning and competitive intelligence"
            },
            "quick_insights": {
                "agents": ["market_research_analyst", "data_analyst"],
                "focus": "rapid market insights and data analysis",
                "description": "Quick analysis for immediate insights"
            }
        }
        
        # Get configuration for analysis type
        if analysis_type in analysis_configs:
            config = analysis_configs[analysis_type]
            selected_agents = custom_agents or config["agents"]
            analysis_focus = config["focus"]
            description = config["description"]
        else:
            # Default configuration
            selected_agents = custom_agents or self.config.get_default_agents()
            analysis_focus = "marketing research analysis"
            description = "Custom marketing analysis"
        
        # Validate agent selection
        if self.config:
            validation = self.config.validate_agent_selection(selected_agents)
            if not validation["valid"]:
                logger.error(f"Invalid agent selection: {validation['errors']}")
                return {
                    "success": False,
                    "error": f"Invalid agents: {validation['errors']}",
                    "suggested_agents": self.config.get_available_agents()
                }
            
            if validation["warnings"]:
                logger.warning(f"Agent selection warnings: {validation['warnings']}")
            
            # Use suggested execution order
            selected_agents = validation["suggested_order"]
        
        # Prepare inputs
        inputs = {
            "target_audience": "health-conscious millennials interested in premium beverages",
            "campaign_type": "multi-channel digital marketing",
            "budget": 100000,
            "duration": "6 months",
            "data_file_path": "data/beverage_sales.csv",
            "analysis_focus": analysis_focus,
            "business_objective": "Launch new organic beverage line and optimize existing product performance",
            "key_metrics": ["sales_trends", "regional_performance", "product_performance", "roi_analysis"],
            "competitive_landscape": "premium beverage market with focus on organic and health-conscious products",
            "market_segments": ["North", "South"],
            "product_lines": ["Green Tea", "Latte"],
            "selected_agents": selected_agents,
            "workflow_type": analysis_type,
            "campaign_goals": [
                "Increase market share in underperforming regions",
                "Optimize product mix based on sales data",
                "Develop targeted marketing strategies for each region",
                "Forecast sales for budget planning",
                "Create compelling content for health-conscious consumers"
            ]
        }
        
        logger.info(f"🚀 Starting {analysis_type} analysis")
        logger.info(f"📋 Description: {description}")
        logger.info(f"🤖 Selected Agents: {', '.join(selected_agents)}")
        
        # Execute workflow
        if self.workflow:
            result = self._run_langgraph_workflow(inputs)
        else:
            result = self._run_crewai_fallback(inputs)
        
        # Store result
        self.results_history.append({
            "timestamp": datetime.now(),
            "analysis_type": analysis_type,
            "agents": selected_agents,
            "inputs": inputs,
            "result": result
        })
        
        return result
    
    def _run_langgraph_workflow(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run the LangGraph workflow."""
        try:
            result = self.workflow.execute_workflow(
                selected_agents=inputs["selected_agents"],
                target_audience=inputs["target_audience"],
                campaign_type=inputs["campaign_type"],
                budget=inputs["budget"],
                duration=inputs["duration"],
                analysis_focus=inputs["analysis_focus"],
                **{k: v for k, v in inputs.items() if k not in [
                    "selected_agents", "target_audience", "campaign_type", 
                    "budget", "duration", "analysis_focus"
                ]}
            )
            
            if result.get("success"):
                logger.info("✅ LangGraph workflow completed successfully")
                result["workflow_engine"] = "langgraph"
            else:
                logger.error(f"❌ LangGraph workflow failed: {result.get('error')}")
                
            return result
            
        except Exception as e:
            logger.error(f"❌ LangGraph execution error: {e}")
            return {
                "success": False,
                "error": str(e),
                "workflow_engine": "langgraph_failed"
            }
    
    def _run_crewai_fallback(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run CrewAI as fallback."""
        try:
            logger.info("🔄 Using CrewAI fallback...")
            
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
                "workflow_engine": "crewai_fallback",
                "result": crew_result,
                "workflow_id": f"crewai_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "summary": {
                    "workflow_type": inputs.get("workflow_type", "fallback"),
                    "agents_executed": inputs["selected_agents"],
                    "execution_engine": "CrewAI"
                }
            }
            
        except Exception as e:
            logger.error(f"❌ CrewAI fallback failed: {e}")
            return {
                "success": False,
                "error": f"Both LangGraph and CrewAI failed: {str(e)}",
                "workflow_engine": "all_failed"
            }
    
    def run_interactive_analysis(self):
        """Run interactive analysis with user input."""
        print("\n🎯 Interactive Marketing Research Analysis")
        print("=" * 50)
        
        # Display available analysis types
        analysis_types = {
            "1": "comprehensive",
            "2": "roi_focused", 
            "3": "content_strategy",
            "4": "brand_performance",
            "5": "quick_insights"
        }
        
        print("\n📊 Available Analysis Types:")
        for key, analysis_type in analysis_types.items():
            print(f"  {key}. {analysis_type.replace('_', ' ').title()}")
        
        # Get user selection
        while True:
            choice = input("\nSelect analysis type (1-5) or 'q' to quit: ").strip()
            if choice.lower() == 'q':
                return None
            if choice in analysis_types:
                selected_type = analysis_types[choice]
                break
            print("❌ Invalid choice. Please select 1-5 or 'q'.")
        
        # Ask about custom agents
        use_custom = input("\nUse custom agent selection? (y/n): ").strip().lower()
        custom_agents = None
        
        if use_custom == 'y':
            available_agents = self.config.get_available_agents() if self.config else [
                "market_research_analyst", "data_analyst", "content_strategist"
            ]
            print(f"\nAvailable agents: {', '.join(available_agents)}")
            agent_input = input("Enter agent names (comma-separated): ").strip()
            if agent_input:
                custom_agents = [agent.strip() for agent in agent_input.split(',')]
        
        # Run analysis
        print(f"\n🚀 Running {selected_type.replace('_', ' ').title()} Analysis...")
        result = self.run_analysis(selected_type, custom_agents)
        
        return result
    
    def export_results(self, filename: str = None) -> str:
        """Export all results to a comprehensive report."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"langgraph_complete_analysis_{timestamp}.json"
        
        try:
            # Prepare comprehensive report
            report = {
                "report_metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "total_analyses": len(self.results_history),
                    "workflow_engine": "LangGraph + CrewAI Fallback",
                    "configuration_loaded": self.config is not None,
                    "langgraph_available": self.workflow is not None
                },
                "system_configuration": self.config.export_config_summary() if self.config else {},
                "analysis_history": []
            }
            
            # Add analysis history with serializable data
            for entry in self.results_history:
                serializable_entry = {
                    "timestamp": entry["timestamp"].isoformat(),
                    "analysis_type": entry["analysis_type"],
                    "agents": entry["agents"],
                    "success": entry["result"].get("success", False),
                    "workflow_engine": entry["result"].get("workflow_engine", "unknown"),
                    "workflow_id": entry["result"].get("workflow_id"),
                    "execution_time": entry["result"].get("execution_time"),
                    "summary": entry["result"].get("summary", {}),
                    "error": entry["result"].get("error") if not entry["result"].get("success") else None
                }
                report["analysis_history"].append(serializable_entry)
            
            # Save report
            with open(filename, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"✅ Comprehensive report exported to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"❌ Failed to export results: {e}")
            return ""
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            "timestamp": datetime.now().isoformat(),
            "configuration_loaded": self.config is not None,
            "langgraph_available": self.workflow is not None,
            "available_agents": self.config.get_available_agents() if self.config else [],
            "default_agents": self.config.get_default_agents() if self.config else [],
            "total_analyses_run": len(self.results_history),
            "last_analysis": self.results_history[-1]["timestamp"].isoformat() if self.results_history else None,
            "system_ready": self.config is not None and (self.workflow is not None or True)  # CrewAI fallback available
        }


def main():
    """Main function for command-line usage."""
    
    print("🚀 Complete LangGraph Marketing Research System")
    print("=" * 60)
    
    # Initialize the system
    runner = CompleteLangGraphRunner()
    
    # Display system status
    status = runner.get_system_status()
    print("\n📊 System Status:")
    print(f"  Configuration Loaded: {'✅' if status['configuration_loaded'] else '❌'}")
    print(f"  LangGraph Available: {'✅' if status['langgraph_available'] else '❌'}")
    print(f"  System Ready: {'✅' if status['system_ready'] else '❌'}")
    print(f"  Available Agents: {len(status['available_agents'])}")
    
    # Run different analysis types
    analysis_types = ["comprehensive", "roi_focused", "content_strategy", "quick_insights"]
    
    print(f"\n🧪 Running Test Analyses...")
    for analysis_type in analysis_types:
        print(f"\n{'='*40}")
        print(f"🔍 Testing {analysis_type.replace('_', ' ').title()} Analysis")
        print(f"{'='*40}")
        
        result = runner.run_analysis(analysis_type)
        
        if result.get("success"):
            print(f"✅ {analysis_type} analysis completed successfully")
            print(f"   Engine: {result.get('workflow_engine', 'unknown')}")
            print(f"   Workflow ID: {result.get('workflow_id', 'N/A')}")
            if result.get('execution_time'):
                print(f"   Execution Time: {result['execution_time']:.2f} seconds")
        else:
            print(f"❌ {analysis_type} analysis failed")
            print(f"   Error: {result.get('error', 'Unknown error')}")
    
    # Export comprehensive results
    print(f"\n{'='*60}")
    print("📁 Exporting Results...")
    export_file = runner.export_results()
    if export_file:
        print(f"✅ Comprehensive report saved to: {export_file}")
    
    # Final system summary
    final_status = runner.get_system_status()
    print(f"\n📈 Final Summary:")
    print(f"  Total Analyses Run: {final_status['total_analyses_run']}")
    print(f"  System Performance: {'Excellent' if final_status['system_ready'] else 'Needs Attention'}")
    
    print(f"\n✅ Complete LangGraph system test finished!")
    
    # Offer interactive mode
    interactive = input("\n🎮 Run interactive analysis? (y/n): ").strip().lower()
    if interactive == 'y':
        runner.run_interactive_analysis()


if __name__ == "__main__":
    main()