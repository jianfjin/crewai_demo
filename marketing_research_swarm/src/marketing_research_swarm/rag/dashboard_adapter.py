"""
Dashboard Adapter for RAG Chat Agent
Provides compatibility between RAG chat agent and dashboard expectations
"""

from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class RAGChatAgentAdapter:
    """
    Adapter class that wraps the RAG chat agent to provide the interface
    expected by the dashboard while leveraging RAG capabilities.
    """
    
    def __init__(self, rag_chat_agent):
        self.rag_agent = rag_chat_agent
        self.knowledge_base = rag_chat_agent.knowledge_base
        self.conversation_history = []
        self.workflow_ready = False
        self.recommended_agents = []
        self.extracted_requirements = {}
        
        # Metadata cache for compatibility
        self.metadata_cache = None
        self.metadata_retrieved = False
        
    def chat(self, user_message: str) -> Dict[str, Any]:
        """
        Process user message using RAG capabilities and return dashboard-compatible response.
        """
        try:
            # Add to conversation history
            self.conversation_history.append({"role": "user", "content": user_message})
            
            # Use RAG agent to process the query
            rag_response = self.rag_agent.process_query(user_message)
            
            # Extract information from RAG response
            intent = rag_response.get("intent", "general_question")
            
            # Get agent recommendations from the RAG response
            agent_recommendations = rag_response.get("recommendations", {}).get("agents", [])
            
            # Convert RAG agent recommendations to agent names
            agent_names = []
            for agent_rec in agent_recommendations:
                if isinstance(agent_rec, dict):
                    agent_names.append(agent_rec.get("agent", agent_rec.get("name", "")))
                else:
                    agent_names.append(str(agent_rec))
            
            self.recommended_agents = agent_names
            
            # Extract parameters from RAG response (if any)
            extracted_params = rag_response.get("extracted_parameters", {})
            
            # Also extract parameters directly from the user message
            additional_params = self._extract_parameters_from_message(user_message)
            extracted_params.update(additional_params)
            
            self.extracted_requirements.update(extracted_params)
            
            # Determine if workflow is ready
            workflow_ready = self._determine_workflow_readiness(rag_response, extracted_params)
            self.workflow_ready = workflow_ready
            
            # Generate response text using the RAG agent's format_response method
            response_text = self.rag_agent.format_response(rag_response)
            
            # Add to conversation history
            self.conversation_history.append({"role": "assistant", "content": response_text})
            
            return {
                "response": response_text,
                "needs_parameters": not workflow_ready and intent == "analysis_request",
                "parameter_options": self._get_parameter_options() if not workflow_ready else {},
                "workflow_ready": workflow_ready,
                "recommended_agents": self.recommended_agents,
                "extracted_requirements": self.extracted_requirements
            }
            
        except Exception as e:
            logger.error(f"RAG chat processing failed: {e}")
            # Fallback response
            return {
                "response": "I encountered an issue processing your request. Could you please rephrase your question?",
                "needs_parameters": False,
                "parameter_options": {},
                "workflow_ready": False,
                "recommended_agents": [],
                "extracted_requirements": {}
            }
    
    def _extract_parameters_from_message(self, message: str) -> Dict[str, Any]:
        """Extract marketing analysis parameters from user message."""
        import re
        
        params = {}
        message_lower = message.lower()
        
        # Extract brands - improved pattern matching
        brand_patterns = {
            r"\bcoca[- ]?cola\b": "Coca-Cola",
            r"\bpepsi\b": "Pepsi", 
            r"\bred bull\b": "Red Bull",
            r"\bmonster( energy)?\b": "Monster Energy",
            r"\bgatorade\b": "Gatorade",
            r"\bpowerade\b": "Powerade",
            r"\bsprite\b": "Sprite",
            r"\bfanta\b": "Fanta", 
            r"\bmountain dew\b": "Mountain Dew",
            r"\bdr\.? pepper\b": "Dr Pepper",
            r"\btropicana\b": "Tropicana",
            r"\bsimply orange\b": "Simply Orange",
            r"\bminute maid\b": "Minute Maid",
            r"\b7up\b": "7UP",
            r"\bdasani\b": "Dasani Water",
            r"\baquafina\b": "Aquafina",
            r"\bvitamin water\b": "Vitamin Water"
        }
        
        brands = []
        for pattern, brand_name in brand_patterns.items():
            if re.search(pattern, message_lower):
                if brand_name not in brands:  # Avoid duplicates
                    brands.append(brand_name)
        
        if brands:
            params["brands"] = brands
        
        # Extract regions
        region_patterns = {
            r"north america": "North America",
            r"europe": "Europe", 
            r"asia pacific": "Asia Pacific",
            r"latin america": "Latin America",
            r"middle east": "Middle East",
            r"africa": "Africa"
        }
        regions = []
        for pattern, region in region_patterns.items():
            if re.search(pattern, message_lower):
                regions.append(region)
        if regions:
            params["target_markets"] = regions
        
        # Extract categories
        category_patterns = {
            r"cola": "Cola",
            r"energy drink": "Energy",
            r"sports drink": "Sports", 
            r"juice": "Juice",
            r"water": "Water"
        }
        categories = []
        for pattern, category in category_patterns.items():
            if re.search(pattern, message_lower):
                categories.append(category)
        if categories:
            params["product_categories"] = categories
        
        # Extract analysis type
        if any(word in message_lower for word in ["roi", "return", "investment", "profitability", "profit", "margin"]):
            params["key_metrics"] = params.get("key_metrics", []) + ["roi", "profitability_analysis"]
        
        if any(word in message_lower for word in ["forecast", "predict", "future", "trend"]):
            params["key_metrics"] = params.get("key_metrics", []) + ["forecasting"]
        
        if any(word in message_lower for word in ["market share", "share", "competition"]):
            params["key_metrics"] = params.get("key_metrics", []) + ["market_share", "competitive_analysis"]
        
        # For general analytical questions, provide default context
        if any(word in message_lower for word in ["which", "what", "show me", "highest", "best", "top"]):
            # If asking about categories and no specific categories mentioned, include common ones
            if "categor" in message_lower and not params.get("product_categories"):
                params["product_categories"] = ["Cola", "Energy", "Sports", "Juice"]
            
            # If asking about regions and no specific regions mentioned, include major markets
            if any(word in message_lower for word in ["region", "market", "area"]) and not params.get("target_markets"):
                params["target_markets"] = ["North America", "Europe", "Asia Pacific"]
            
            # If asking about brands and no specific brands mentioned, include major brands
            if "brand" in message_lower and not params.get("brands"):
                params["brands"] = ["Coca-Cola", "Pepsi", "Red Bull", "Gatorade"]
        
        return params
    
    def _determine_workflow_readiness(self, rag_response: Dict[str, Any], extracted_params: Dict[str, Any]) -> bool:
        """Determine if we have enough information to proceed with workflow."""
        
        # Check if RAG agent indicates readiness
        if rag_response.get("workflow_ready", False):
            return True
        
        # Check if we have essential parameters
        has_agents = bool(self.recommended_agents)
        has_context = bool(extracted_params.get("brands") or extracted_params.get("target_markets") or extracted_params.get("product_categories"))
        has_analysis_type = bool(extracted_params.get("key_metrics"))
        
        # For analysis requests, we need agents and either context or analysis type
        intent = rag_response.get("intent", "")
        if "analysis" in intent or "inquiry" in intent:
            # If we have agents and analysis type (like profitability), that's enough
            if has_agents and has_analysis_type:
                return True
            # Or if we have agents and context
            if has_agents and has_context:
                return True
            # For general analytical questions with agents, provide default context
            if has_agents and any(word in rag_response.get("query", "").lower() for word in ["which", "what", "show me", "highest", "best", "top"]):
                return True
        
        return has_agents or has_context
    
    def _get_parameter_options(self) -> Dict[str, List[str]]:
        """Get parameter options for user selection."""
        return {
            "target_markets": ["North America", "Europe", "Asia Pacific", "Latin America", "Middle East", "Africa", "Australia", "Global"],
            "product_categories": ["Cola", "Juice", "Energy", "Sports", "Citrus", "Lemon-Lime", "Orange", "Water", "Enhanced Water", "Tea", "Coffee"],
            "key_metrics": ["brand_performance", "category_trends", "regional_dynamics", "profitability_analysis", "pricing_optimization", "market_share", "customer_satisfaction", "roi"],
            "brands": ["Coca-Cola", "Pepsi", "Red Bull", "Monster Energy", "Gatorade", "Powerade", "Tropicana", "Simply Orange", "Minute Maid", "Sprite", "Fanta", "7UP", "Mountain Dew", "Dr Pepper", "Dasani Water", "Aquafina", "Vitamin Water"]
        }
    
    def set_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Set parameters and update workflow readiness."""
        
        # Update extracted requirements
        self.extracted_requirements.update(parameters)
        
        # Check if workflow is now ready
        essential_params = ["target_markets", "product_categories", "key_metrics"]
        has_essential = any(self.extracted_requirements.get(param) for param in essential_params)
        
        if has_essential or self.recommended_agents:
            self.workflow_ready = True
            response_text = "Great! I've updated the parameters. The workflow is now ready to run."
        else:
            response_text = "Parameters updated. Please provide more information to complete the workflow setup."
        
        return {
            "response": response_text,
            "needs_parameters": not self.workflow_ready,
            "parameter_options": {} if self.workflow_ready else self._get_parameter_options(),
            "workflow_ready": self.workflow_ready,
            "recommended_agents": self.recommended_agents,
            "extracted_requirements": self.extracted_requirements
        }
    
    def get_workflow_config(self) -> Dict[str, Any]:
        """Get complete workflow configuration."""
        
        if not self.workflow_ready:
            return {}
        
        # Use RAG-enhanced agent selection if available
        selected_agents = self.recommended_agents if self.recommended_agents else ["market_research_analyst", "data_analyst"]
        
        # Build configuration with extracted requirements
        config = {
            "analysis_type": "rag_enhanced",
            "selected_agents": selected_agents,
            "target_audience": "beverage consumers",
            "campaign_type": "comprehensive_analysis",
            "budget": self.extracted_requirements.get("budget", 25000),
            "duration": self.extracted_requirements.get("duration", 30),
            "analysis_focus": "RAG-enhanced beverage industry analysis",
            "business_objective": "Optimize marketing performance using knowledge base insights",
            "competitive_landscape": "global beverage market",
            "market_segments": self.extracted_requirements.get("target_markets", ["North America", "Europe"]),
            "product_categories": self.extracted_requirements.get("product_categories", ["Cola", "Energy", "Sports"]),
            "key_metrics": self.extracted_requirements.get("key_metrics", ["brand_performance", "category_trends"]),
            "brands": self.extracted_requirements.get("brands", ["Coca-Cola", "Pepsi"]),
            "campaign_goals": self.extracted_requirements.get("campaign_goals", [
                "Optimize brand portfolio performance across global markets",
                "Leverage knowledge base for strategic insights"
            ]),
            "forecast_periods": self.extracted_requirements.get("forecast_periods", 30),
            "expected_revenue": self.extracted_requirements.get("expected_revenue", 25000),
            "brand_metrics": {
                "brand_awareness": 75,
                "sentiment_score": 0.7,
                "market_position": "Premium"
            },
            "competitive_analysis": True,
            "market_share_analysis": True,
            "optimization_settings": {
                "token_budget": 50000,
                "context_strategy": "rag_enhanced",
                "enable_caching": True,
                "enable_mem0": False,
                "enable_token_tracking": True,
                "enable_optimization_tools": True,
                "optimization_level": "full",
                "show_comparison": False
            }
        }
        
        return config
    
    def reset(self):
        """Reset chat agent state."""
        self.conversation_history = []
        self.extracted_requirements = {}
        self.recommended_agents = []
        self.workflow_ready = False
        self.metadata_cache = None
        self.metadata_retrieved = False
    
    # RAG-specific methods for dashboard integration
    def get_data_context(self) -> Dict[str, Any]:
        """Get data context from knowledge base."""
        try:
            # Try to get metadata from knowledge base
            if hasattr(self.knowledge_base, 'get_metadata'):
                return self.knowledge_base.get_metadata()
            else:
                # Return basic context
                return {
                    "distinct_values": {
                        "region": ["North America", "Europe", "Asia Pacific", "Latin America"],
                        "brand": ["Coca-Cola", "Pepsi", "Red Bull", "Monster Energy", "Gatorade"],
                        "category": ["Cola", "Energy", "Sports", "Juice", "Water"]
                    },
                    "insights": [
                        "Knowledge base contains comprehensive marketing research data",
                        "RAG system provides intelligent agent recommendations",
                        "Historical analysis patterns available for reference"
                    ]
                }
        except Exception as e:
            logger.error(f"Error getting data context: {e}")
            return {}
    
    def get_selection_reasoning(self) -> str:
        """Get reasoning for agent selection."""
        if self.recommended_agents:
            return f"Selected {len(self.recommended_agents)} agents based on knowledge base analysis and query intent matching."
        return ""
    
    def get_workflow_insights(self) -> List[str]:
        """Get RAG-enhanced workflow insights."""
        insights = []
        
        if self.knowledge_base:
            insights.append("Knowledge base search enabled for intelligent recommendations")
        
        if self.recommended_agents:
            insights.append(f"RAG system recommended {len(self.recommended_agents)} specialized agents")
        
        if self.extracted_requirements:
            insights.append("Parameters extracted using natural language understanding")
        
        return insights


def get_rag_chat_agent_adapter():
    """Get RAG chat agent wrapped with dashboard compatibility."""
    try:
        from .chat_integration import get_chat_agent
        rag_agent = get_chat_agent()
        return RAGChatAgentAdapter(rag_agent)
    except Exception as e:
        logger.error(f"Failed to create RAG chat agent adapter: {e}")
        raise