#!/usr/bin/env python3
"""
LangGraph Configuration and Setup for Marketing Research

This module provides configuration management and setup utilities
for the LangGraph marketing research workflow.
"""

import os
import yaml
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class LangGraphConfig:
    """Configuration manager for LangGraph workflows."""
    
    def __init__(self, config_dir: str = "src/marketing_research_swarm/config"):
        """Initialize configuration manager."""
        self.config_dir = config_dir
        self.agents_config = None
        self.tasks_config = None
        self.workflow_config = None
        
        # Load configurations
        self.load_configurations()
    
    def load_configurations(self):
        """Load all configuration files."""
        try:
            # Load agents configuration
            agents_path = os.path.join(self.config_dir, "agents.yaml")
            if os.path.exists(agents_path):
                with open(agents_path, 'r') as f:
                    self.agents_config = yaml.safe_load(f)
                logger.info(f"✅ Loaded agents config from {agents_path}")
            
            # Load tasks configuration
            tasks_path = os.path.join(self.config_dir, "tasks.yaml")
            if os.path.exists(tasks_path):
                with open(tasks_path, 'r') as f:
                    self.tasks_config = yaml.safe_load(f)
                logger.info(f"✅ Loaded tasks config from {tasks_path}")
            
            # Load workflow-specific configuration
            workflow_path = os.path.join(self.config_dir, "workflow.yaml")
            if os.path.exists(workflow_path):
                with open(workflow_path, 'r') as f:
                    self.workflow_config = yaml.safe_load(f)
            else:
                # Create default workflow configuration
                self.workflow_config = self.create_default_workflow_config()
                self.save_workflow_config()
            
        except Exception as e:
            logger.error(f"❌ Error loading configurations: {e}")
            # Use default configurations
            self.create_default_configurations()
    
    def create_default_workflow_config(self) -> Dict[str, Any]:
        """Create default workflow configuration."""
        return {
            "workflow": {
                "name": "marketing_research_workflow",
                "version": "1.0.0",
                "description": "LangGraph-based marketing research workflow",
                "max_retries": 2,
                "timeout_minutes": 30,
                "checkpoint_enabled": True,
                "parallel_execution": False
            },
            "agent_selection": {
                "default_agents": [
                    "market_research_analyst",
                    "data_analyst", 
                    "content_strategist"
                ],
                "available_agents": [
                    "market_research_analyst",
                    "competitive_analyst",
                    "data_analyst",
                    "content_strategist",
                    "creative_copywriter",
                    "campaign_optimizer",
                    "brand_performance_specialist",
                    "forecasting_specialist"
                ],
                "agent_dependencies": {
                    "content_strategist": ["market_research_analyst"],
                    "creative_copywriter": ["content_strategist"],
                    "campaign_optimizer": ["data_analyst", "content_strategist"],
                    "brand_performance_specialist": ["market_research_analyst", "data_analyst"],
                    "forecasting_specialist": ["data_analyst"]
                }
            },
            "execution": {
                "sequential_mode": True,
                "error_handling": "retry_with_fallback",
                "result_aggregation": "comprehensive",
                "progress_tracking": True
            },
            "output": {
                "format": "structured_json",
                "include_metadata": True,
                "save_intermediate_results": True,
                "export_formats": ["json", "markdown", "csv"]
            }
        }
    
    def create_default_configurations(self):
        """Create default configurations if loading fails."""
        logger.info("Creating default configurations...")
        
        # Default agents config
        self.agents_config = {
            "market_research_analyst": {
                "role": "Market Research Analyst",
                "goal": "Analyze market trends and consumer behavior",
                "backstory": "Expert in market research with deep understanding of consumer psychology",
                "llm": "openai/gpt-4o-mini",
                "tools": ["read_csv_tool", "analyze_kpis", "time_series_analysis"]
            },
            "data_analyst": {
                "role": "Data Analyst", 
                "goal": "Process and analyze marketing data",
                "backstory": "Skilled data analyst specializing in marketing analytics",
                "llm": "openai/gpt-4o-mini",
                "tools": ["read_csv_tool", "calculate_roi", "forecast_sales"]
            },
            "content_strategist": {
                "role": "Content Strategist",
                "goal": "Develop content strategy based on research insights",
                "backstory": "Creative strategist with expertise in content marketing",
                "llm": "openai/gpt-4o-mini",
                "tools": ["analyze_brand_performance"]
            }
        }
        
        # Default tasks config
        self.tasks_config = {
            "market_analysis": {
                "description": "Analyze market trends and opportunities",
                "expected_output": "Comprehensive market analysis report",
                "agent": "market_research_analyst"
            },
            "data_processing": {
                "description": "Process and analyze marketing data",
                "expected_output": "Data analysis with insights and recommendations", 
                "agent": "data_analyst"
            },
            "content_strategy": {
                "description": "Develop content strategy based on insights",
                "expected_output": "Content strategy recommendations",
                "agent": "content_strategist"
            }
        }
    
    def save_workflow_config(self):
        """Save workflow configuration to file."""
        try:
            workflow_path = os.path.join(self.config_dir, "workflow.yaml")
            os.makedirs(self.config_dir, exist_ok=True)
            
            with open(workflow_path, 'w') as f:
                yaml.dump(self.workflow_config, f, default_flow_style=False, indent=2)
            
            logger.info(f"✅ Saved workflow config to {workflow_path}")
            
        except Exception as e:
            logger.error(f"❌ Error saving workflow config: {e}")
    
    def get_agent_config(self, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific agent."""
        if self.agents_config and agent_name in self.agents_config:
            return self.agents_config[agent_name]
        return None
    
    def get_available_agents(self) -> List[str]:
        """Get list of available agents."""
        if self.workflow_config:
            return self.workflow_config.get("agent_selection", {}).get("available_agents", [])
        return list(self.agents_config.keys()) if self.agents_config else []
    
    def get_default_agents(self) -> List[str]:
        """Get default agent selection."""
        if self.workflow_config:
            return self.workflow_config.get("agent_selection", {}).get("default_agents", [])
        return ["market_research_analyst", "data_analyst", "content_strategist"]
    
    def get_agent_dependencies(self) -> Dict[str, List[str]]:
        """Get agent dependency mapping."""
        if self.workflow_config:
            return self.workflow_config.get("agent_selection", {}).get("agent_dependencies", {})
        return {}
    
    def validate_agent_selection(self, selected_agents: List[str]) -> Dict[str, Any]:
        """Validate agent selection against dependencies."""
        available_agents = self.get_available_agents()
        dependencies = self.get_agent_dependencies()
        
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "suggested_order": []
        }
        
        # Check if all selected agents are available
        for agent in selected_agents:
            if agent not in available_agents:
                validation_result["valid"] = False
                validation_result["errors"].append(f"Agent '{agent}' is not available")
        
        # Check dependencies
        for agent in selected_agents:
            if agent in dependencies:
                required_deps = dependencies[agent]
                missing_deps = [dep for dep in required_deps if dep not in selected_agents]
                if missing_deps:
                    validation_result["warnings"].append(
                        f"Agent '{agent}' recommends having {missing_deps} but they are not selected"
                    )
        
        # Suggest execution order based on dependencies
        validation_result["suggested_order"] = self._calculate_execution_order(selected_agents, dependencies)
        
        return validation_result
    
    def _calculate_execution_order(self, agents: List[str], dependencies: Dict[str, List[str]]) -> List[str]:
        """Calculate optimal execution order based on dependencies."""
        ordered = []
        remaining = agents.copy()
        
        while remaining:
            # Find agents with no unmet dependencies
            ready = []
            for agent in remaining:
                deps = dependencies.get(agent, [])
                if all(dep in ordered or dep not in agents for dep in deps):
                    ready.append(agent)
            
            if not ready:
                # No agents ready, add remaining in original order to break cycle
                ready = remaining
            
            # Add ready agents to order
            for agent in ready:
                ordered.append(agent)
                remaining.remove(agent)
        
        return ordered
    
    def export_config_summary(self) -> Dict[str, Any]:
        """Export configuration summary."""
        return {
            "timestamp": datetime.now().isoformat(),
            "config_dir": self.config_dir,
            "available_agents": self.get_available_agents(),
            "default_agents": self.get_default_agents(),
            "agent_dependencies": self.get_agent_dependencies(),
            "workflow_settings": self.workflow_config.get("workflow", {}) if self.workflow_config else {},
            "agents_loaded": len(self.agents_config) if self.agents_config else 0,
            "tasks_loaded": len(self.tasks_config) if self.tasks_config else 0
        }


def main():
    """Test the configuration system."""
    print("🔧 LangGraph Configuration System Test")
    print("=" * 50)
    
    # Initialize configuration
    config = LangGraphConfig()
    
    # Display configuration summary
    summary = config.export_config_summary()
    print("📋 Configuration Summary:")
    print(f"  Available Agents: {len(summary['available_agents'])}")
    print(f"  Default Agents: {', '.join(summary['default_agents'])}")
    print(f"  Agents Loaded: {summary['agents_loaded']}")
    print(f"  Tasks Loaded: {summary['tasks_loaded']}")
    
    # Test agent selection validation
    print("\n🧪 Testing Agent Selection Validation:")
    test_selections = [
        ["market_research_analyst", "data_analyst"],
        ["content_strategist", "creative_copywriter"],  # Missing dependency
        ["invalid_agent", "data_analyst"],  # Invalid agent
        ["market_research_analyst", "data_analyst", "content_strategist", "campaign_optimizer"]
    ]
    
    for selection in test_selections:
        print(f"\nTesting: {selection}")
        validation = config.validate_agent_selection(selection)
        print(f"  Valid: {validation['valid']}")
        if validation['errors']:
            print(f"  Errors: {validation['errors']}")
        if validation['warnings']:
            print(f"  Warnings: {validation['warnings']}")
        print(f"  Suggested Order: {validation['suggested_order']}")
    
    print("\n✅ Configuration system test completed!")


if __name__ == "__main__":
    main()