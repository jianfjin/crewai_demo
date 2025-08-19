"""
Workflow Integration for RAG Chat System

This module integrates the RAG chat system with the actual workflow execution,
allowing users to go from chat recommendations to running analyses seamlessly.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import json
import re
from datetime import datetime

logger = logging.getLogger(__name__)

class ChatWorkflowIntegrator:
    """
    Integrates chat recommendations with workflow execution.
    
    This class bridges the gap between chat recommendations and actual
    workflow execution by:
    - Parsing chat responses for workflow parameters
    - Building workflow configurations
    - Providing execution interfaces
    - Managing workflow state
    """
    
    def __init__(self):
        """Initialize the workflow integrator."""
        self.pending_workflows = {}
        self.execution_history = []
        
        # Parameter extraction patterns
        self.parameter_patterns = {
            "agents": r"(?:recommended agents?|agents?)[:\s]*([^.]+)",
            "budget": r"budget[:\s]*\$?([0-9,]+)",
            "duration": r"duration[:\s]*([0-9]+)\s*days?",
            "target_audience": r"target audience[:\s]*([^.]+)",
            "brands": r"brands?[:\s]*([^.]+)",
            "regions": r"regions?[:\s]*([^.]+)",
            "analysis_type": r"analysis type[:\s]*([^.]+)",
            "forecast_periods": r"forecast[:\s]*([0-9]+)\s*(?:days?|periods?)"
        }
        
        # Agent name mappings
        self.agent_mappings = {
            "market research": "market_research_analyst",
            "data analyst": "data_analyst", 
            "competitive": "competitive_analyst",
            "forecasting": "forecasting_specialist",
            "campaign optimizer": "campaign_optimizer",
            "brand performance": "brand_performance_specialist",
            "content strategist": "content_strategist",
            "copywriter": "creative_copywriter"
        }
    
    def extract_workflow_parameters(self, chat_response: str) -> Dict[str, Any]:
        """
        Extract workflow parameters from a chat response.
        
        Args:
            chat_response: The formatted chat response text
            
        Returns:
            Dictionary with extracted workflow parameters
        """
        try:
            parameters = {}
            response_lower = chat_response.lower()
            
            # Extract agents
            agents = []
            for agent_key, agent_id in self.agent_mappings.items():
                if agent_key in response_lower:
                    agents.append(agent_id)
            
            if agents:
                parameters["selected_agents"] = agents
            
            # Extract other parameters using patterns
            for param, pattern in self.parameter_patterns.items():
                if param == "agents":
                    continue  # Already handled above
                
                match = re.search(pattern, response_lower)
                if match:
                    value = match.group(1).strip()
                    
                    # Clean up the value based on parameter type
                    if param == "budget":
                        # Extract numeric value
                        numeric_value = re.sub(r'[^\d]', '', value)
                        if numeric_value:
                            parameters[param] = int(numeric_value)
                    elif param == "duration":
                        parameters[param] = f"{value} days"
                    elif param == "forecast_periods":
                        parameters[param] = int(value)
                    else:
                        parameters[param] = value
            
            # Set defaults for missing parameters
            defaults = {
                "target_audience": "General market audience",
                "campaign_type": "Marketing analysis campaign",
                "budget": 25000,
                "duration": "30 days",
                "analysis_focus": "Comprehensive market analysis",
                "forecast_periods": 30,
                "business_objective": "Market performance analysis",
                "market_segments": ["premium", "mass_market"],
                "product_categories": ["beverages"],
                "key_metrics": ["brand_performance", "market_share", "profitability"],
                "regions": ["North America"]
            }
            
            for key, default_value in defaults.items():
                if key not in parameters:
                    parameters[key] = default_value
            
            return parameters
            
        except Exception as e:
            logger.error(f"❌ Error extracting workflow parameters: {e}")
            return {}
    
    def detect_analysis_intent(self, user_query: str) -> Dict[str, Any]:
        """
        Detect if the user query contains analysis intent and extract key information.
        
        Args:
            user_query: The user's original query
            
        Returns:
            Dictionary with analysis intent and extracted information
        """
        try:
            query_lower = user_query.lower()
            
            # Analysis intent patterns
            analysis_patterns = [
                r"analyze.*performance",
                r"compare.*(?:against|vs|versus)",
                r"analysis.*of",
                r"study.*market",
                r"research.*brand",
                r"forecast.*sales",
                r"predict.*performance",
                r"evaluate.*campaign"
            ]
            
            has_analysis_intent = any(re.search(pattern, query_lower) for pattern in analysis_patterns)
            
            if not has_analysis_intent:
                return {"has_intent": False}
            
            # Extract entities
            entities = {
                "brands": [],
                "regions": [],
                "analysis_type": "general",
                "comparison": False
            }
            
            # Extract brand names (common beverage brands)
            brand_patterns = [
                r"coca[- ]?cola", r"pepsi", r"sprite", r"fanta", r"dr[. ]pepper",
                r"mountain dew", r"red bull", r"monster", r"rockstar",
                r"gatorade", r"powerade", r"vitaminwater", r"smartwater"
            ]
            
            for pattern in brand_patterns:
                if re.search(pattern, query_lower):
                    brand_name = re.search(pattern, query_lower).group(0)
                    entities["brands"].append(brand_name.title().replace(" ", " "))
            
            # Extract regions
            region_patterns = [
                r"north america", r"usa", r"united states", r"canada",
                r"europe", r"asia", r"global", r"worldwide"
            ]
            
            for pattern in region_patterns:
                if re.search(pattern, query_lower):
                    region_name = re.search(pattern, query_lower).group(0)
                    entities["regions"].append(region_name.title())
            
            # Detect comparison intent
            comparison_words = ["against", "vs", "versus", "compared to", "compare"]
            entities["comparison"] = any(word in query_lower for word in comparison_words)
            
            # Determine analysis type
            if "performance" in query_lower:
                entities["analysis_type"] = "brand_performance"
            elif "forecast" in query_lower or "predict" in query_lower:
                entities["analysis_type"] = "forecasting"
            elif "market" in query_lower:
                entities["analysis_type"] = "market_research"
            elif "campaign" in query_lower:
                entities["analysis_type"] = "campaign_optimization"
            
            return {
                "has_intent": True,
                "entities": entities,
                "original_query": user_query
            }
            
        except Exception as e:
            logger.error(f"❌ Error detecting analysis intent: {e}")
            return {"has_intent": False, "error": str(e)}
    
    def build_workflow_config(self, chat_parameters: Dict[str, Any], 
                            analysis_intent: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a complete workflow configuration from chat parameters and analysis intent.
        
        Args:
            chat_parameters: Parameters extracted from chat response
            analysis_intent: Analysis intent detected from user query
            
        Returns:
            Complete workflow configuration
        """
        try:
            # Start with chat parameters
            config = chat_parameters.copy()
            
            # Enhance with analysis intent
            if analysis_intent.get("has_intent", False):
                entities = analysis_intent.get("entities", {})
                
                # Update brands if detected
                if entities.get("brands"):
                    config["brands"] = entities["brands"]
                
                # Update regions if detected
                if entities.get("regions"):
                    config["regions"] = entities["regions"]
                
                # Update analysis focus based on type
                analysis_type = entities.get("analysis_type", "general")
                if analysis_type == "brand_performance":
                    config["analysis_focus"] = "Brand performance and competitive analysis"
                    config["key_metrics"] = ["brand_performance", "market_share", "competitive_position"]
                elif analysis_type == "forecasting":
                    config["analysis_focus"] = "Sales forecasting and trend analysis"
                    config["key_metrics"] = ["sales_forecast", "trend_analysis", "demand_prediction"]
                elif analysis_type == "market_research":
                    config["analysis_focus"] = "Market research and consumer analysis"
                    config["key_metrics"] = ["market_size", "consumer_behavior", "market_trends"]
                elif analysis_type == "campaign_optimization":
                    config["analysis_focus"] = "Campaign optimization and ROI analysis"
                    config["key_metrics"] = ["campaign_performance", "roi", "budget_efficiency"]
                
                # Adjust agents based on analysis type
                if analysis_type == "brand_performance":
                    config["selected_agents"] = ["brand_performance_specialist", "competitive_analyst", "data_analyst"]
                elif analysis_type == "forecasting":
                    config["selected_agents"] = ["forecasting_specialist", "data_analyst", "market_research_analyst"]
                elif analysis_type == "market_research":
                    config["selected_agents"] = ["market_research_analyst", "data_analyst", "competitive_analyst"]
                elif analysis_type == "campaign_optimization":
                    config["selected_agents"] = ["campaign_optimizer", "data_analyst", "content_strategist"]
            
            # Ensure required fields are present
            required_fields = {
                "selected_agents": ["market_research_analyst", "data_analyst"],
                "target_audience": "General market audience",
                "campaign_type": "Marketing analysis campaign",
                "budget": 25000,
                "duration": "30 days",
                "analysis_focus": "Comprehensive market analysis"
            }
            
            for field, default_value in required_fields.items():
                if field not in config or not config[field]:
                    config[field] = default_value
            
            # Add workflow metadata
            config["workflow_source"] = "chat_integration"
            config["created_at"] = datetime.now().isoformat()
            config["workflow_id"] = f"chat_workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            return config
            
        except Exception as e:
            logger.error(f"❌ Error building workflow config: {e}")
            return {}
    
    def create_workflow_summary(self, config: Dict[str, Any]) -> str:
        """
        Create a human-readable summary of the workflow configuration.
        
        Args:
            config: Workflow configuration dictionary
            
        Returns:
            Formatted summary string
        """
        try:
            summary_parts = []
            
            # Header
            summary_parts.append("🎯 **Workflow Configuration Ready**")
            summary_parts.append("")
            
            # Analysis details
            summary_parts.append("📊 **Analysis Details:**")
            summary_parts.append(f"• **Focus**: {config.get('analysis_focus', 'General analysis')}")
            summary_parts.append(f"• **Target Audience**: {config.get('target_audience', 'General audience')}")
            summary_parts.append(f"• **Budget**: ${config.get('budget', 25000):,}")
            summary_parts.append(f"• **Duration**: {config.get('duration', '30 days')}")
            summary_parts.append("")
            
            # Agents
            agents = config.get("selected_agents", [])
            if agents:
                summary_parts.append("🤖 **Selected Agents:**")
                for agent in agents:
                    agent_name = agent.replace("_", " ").title()
                    summary_parts.append(f"• {agent_name}")
                summary_parts.append("")
            
            # Brands and regions
            brands = config.get("brands", [])
            if brands:
                summary_parts.append(f"🏷️ **Brands**: {', '.join(brands)}")
            
            regions = config.get("regions", [])
            if regions:
                summary_parts.append(f"🌍 **Regions**: {', '.join(regions)}")
            
            if brands or regions:
                summary_parts.append("")
            
            # Key metrics
            metrics = config.get("key_metrics", [])
            if metrics:
                summary_parts.append("📈 **Key Metrics:**")
                for metric in metrics:
                    metric_name = metric.replace("_", " ").title()
                    summary_parts.append(f"• {metric_name}")
                summary_parts.append("")
            
            # Call to action
            summary_parts.append("🚀 **Ready to Execute!**")
            summary_parts.append("Click the 'Run Analysis' button below to start the workflow with these parameters.")
            
            return "\n".join(summary_parts)
            
        except Exception as e:
            logger.error(f"❌ Error creating workflow summary: {e}")
            return "Error creating workflow summary. Please check the configuration."
    
    def store_pending_workflow(self, workflow_id: str, config: Dict[str, Any]):
        """Store a workflow configuration for later execution."""
        self.pending_workflows[workflow_id] = {
            "config": config,
            "created_at": datetime.now().isoformat(),
            "status": "pending"
        }
    
    def get_pending_workflow(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a pending workflow configuration."""
        return self.pending_workflows.get(workflow_id)
    
    def mark_workflow_executed(self, workflow_id: str, result: Dict[str, Any]):
        """Mark a workflow as executed and store the result."""
        if workflow_id in self.pending_workflows:
            self.pending_workflows[workflow_id]["status"] = "executed"
            self.pending_workflows[workflow_id]["executed_at"] = datetime.now().isoformat()
            self.pending_workflows[workflow_id]["result"] = result
            
            # Add to execution history
            self.execution_history.append({
                "workflow_id": workflow_id,
                "executed_at": datetime.now().isoformat(),
                "config": self.pending_workflows[workflow_id]["config"],
                "result_summary": {
                    "success": result.get("success", False),
                    "agents_executed": len(result.get("agent_results", {})),
                    "total_tokens": result.get("total_token_usage", {}).get("total_tokens", 0)
                }
            })

# Global integrator instance
_workflow_integrator = None

def get_workflow_integrator() -> ChatWorkflowIntegrator:
    """Get the global workflow integrator instance."""
    global _workflow_integrator
    if _workflow_integrator is None:
        _workflow_integrator = ChatWorkflowIntegrator()
    return _workflow_integrator