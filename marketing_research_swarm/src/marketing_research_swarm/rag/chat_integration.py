"""
Chat Integration for RAG Knowledge Base

This module integrates the knowledge base with the chat agent to provide
intelligent responses about the marketing research tool's capabilities,
agents, tools, and features.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import json
import re

from .knowledge_base import get_knowledge_base
from .self_corrective_rag import SelfCorrectiveRAG

logger = logging.getLogger(__name__)

class RAGChatAgent:
    """
    Enhanced chat agent with RAG capabilities for the Marketing Research Tool.
    
    This agent can:
    - Answer questions about system features and capabilities
    - Recommend appropriate agents for specific tasks
    - Suggest relevant tools for analysis needs
    - Provide implementation guidance
    - Explain workflow options
    """
    
    def __init__(self):
        """Initialize the RAG chat agent."""
        self.knowledge_base = get_knowledge_base()
        self.self_corrective_rag = SelfCorrectiveRAG(self.knowledge_base)
        
        # Intent patterns for query classification
        self.intent_patterns = {
            "agent_inquiry": [
                r"what.*agent.*do",
                r"which.*agent.*for",
                r"agent.*capabilities",
                r"who.*analyze",
                r"best.*agent.*for"
            ],
            "tool_inquiry": [
                r"what.*tool.*use",
                r"which.*tool.*for",
                r"tool.*analysis",
                r"function.*available",
                r"how.*analyze"
            ],
            "workflow_inquiry": [
                r"how.*workflow",
                r"what.*workflow",
                r"langgraph.*crewai",
                r"execution.*process",
                r"run.*analysis"
            ],
            "feature_inquiry": [
                r"what.*feature",
                r"capabilities.*system",
                r"what.*can.*do",
                r"features.*available",
                r"system.*capabilities"
            ],
            "implementation_help": [
                r"how.*implement",
                r"setup.*guide",
                r"installation.*help",
                r"getting.*started",
                r"configuration.*help"
            ],
            "optimization_help": [
                r"optimize.*performance",
                r"improve.*speed",
                r"token.*usage",
                r"cache.*memory",
                r"performance.*tips"
            ]
        }
        
        # Agent specializations
        self.agent_specializations = {
            "market_research_analyst": [
                "market analysis", "consumer behavior", "market trends", 
                "competitive landscape", "market segmentation"
            ],
            "data_analyst": [
                "data analysis", "statistical analysis", "profitability analysis",
                "cross-sectional analysis", "time series analysis", "KPI analysis"
            ],
            "competitive_analyst": [
                "competitive analysis", "competitor research", "market positioning",
                "competitive intelligence", "brand comparison"
            ],
            "forecasting_specialist": [
                "sales forecasting", "demand forecasting", "predictive analysis",
                "trend prediction", "future projections"
            ],
            "campaign_optimizer": [
                "campaign optimization", "budget allocation", "ROI optimization",
                "marketing efficiency", "resource planning"
            ],
            "brand_performance_specialist": [
                "brand analysis", "brand performance", "market share analysis",
                "brand positioning", "brand metrics"
            ],
            "content_strategist": [
                "content strategy", "marketing content", "messaging strategy",
                "content planning", "strategic communication"
            ],
            "creative_copywriter": [
                "copywriting", "creative content", "marketing copy",
                "advertising content", "creative strategy"
            ]
        }
        
        # Tool categories
        self.tool_categories = {
            "financial_analysis": [
                "profitability_analysis", "calculate_roi", "plan_budget"
            ],
            "market_analysis": [
                "beverage_market_analysis", "calculate_market_share", 
                "cross_sectional_analysis"
            ],
            "forecasting": [
                "forecast_sales", "time_series_analysis"
            ],
            "performance_analysis": [
                "analyze_kpis", "time_series_analysis"
            ]
        }
    
    def classify_intent(self, query: str) -> str:
        """
        Classify the user's intent based on their query.
        
        Args:
            query: User's query string
            
        Returns:
            Intent classification
        """
        query_lower = query.lower()
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return intent
        
        return "general_inquiry"
    
    def recommend_agents(self, task_description: str) -> List[Dict[str, Any]]:
        """
        Recommend appropriate agents for a given task.
        
        Args:
            task_description: Description of the task or analysis needed
            
        Returns:
            List of recommended agents with explanations
        """
        task_lower = task_description.lower()
        recommendations = []
        
        # Check for comparative analysis keywords
        comparative_keywords = ["compare", "against", "vs", "versus", "competition", "competitive"]
        is_comparative = any(keyword in task_lower for keyword in comparative_keywords)
        
        # Extract brands to check for multi-brand analysis
        brand_patterns = [
            r"\bcoca[- ]?cola\b", r"\bpepsi\b", r"\bred bull\b", r"\bmonster\b", 
            r"\bgatorade\b", r"\bpowerade\b", r"\bsprite\b", r"\bfanta\b"
        ]
        brands_found = sum(1 for pattern in brand_patterns if re.search(pattern, task_lower))
        is_multi_brand = brands_found > 1
        
        for agent, specializations in self.agent_specializations.items():
            relevance_score = 0
            matching_specializations = []
            
            for spec in specializations:
                if any(word in task_lower for word in spec.split()):
                    relevance_score += 1
                    matching_specializations.append(spec)
            
            # Boost competitive analyst for comparative/multi-brand analysis
            if agent == "competitive_analyst" and (is_comparative or is_multi_brand):
                relevance_score += 3
                matching_specializations.append("comparative brand analysis")
            
            # Boost brand performance specialist for brand-related queries
            if agent == "brand_performance_specialist" and any(word in task_lower for word in ["brand", "performance"]):
                relevance_score += 2
                matching_specializations.append("brand performance analysis")
            
            # Boost market research analyst for general performance and regional queries
            if agent == "market_research_analyst" and any(word in task_lower for word in ["performing", "best", "top", "region", "market"]):
                relevance_score += 2
                matching_specializations.append("market performance analysis")
            
            # Boost data analyst for analysis requests and financial queries
            if agent == "data_analyst" and ("analyz" in task_lower or any(word in task_lower for word in ["profit", "margin", "revenue", "cost", "financial", "calculate"])):
                relevance_score += 2
                if any(word in task_lower for word in ["profit", "margin", "revenue", "cost"]):
                    matching_specializations.append("profitability analysis")
                else:
                    matching_specializations.append("data analysis")
            
            # Boost forecasting specialist for revenue/financial projections
            if agent == "forecasting_specialist" and any(word in task_lower for word in ["revenue", "profit", "forecast", "predict", "projection"]):
                relevance_score += 1
                matching_specializations.append("financial forecasting")
            
            # Boost campaign optimizer for ROI and profitability optimization
            if agent == "campaign_optimizer" and any(word in task_lower for word in ["roi", "profit", "margin", "optimize", "efficiency"]):
                relevance_score += 1
                matching_specializations.append("profitability optimization")
            
            # Boost content strategist for content strategy requests
            if agent == "content_strategist" and any(word in task_lower for word in ["content", "strategy", "develop", "create", "messaging", "communication", "millennial", "consumer", "target", "audience"]):
                relevance_score += 3
                matching_specializations.append("content strategy development")
            
            # Boost market research analyst for consumer and demographic analysis
            if agent == "market_research_analyst" and any(word in task_lower for word in ["consumer", "millennial", "demographic", "target", "audience", "segment", "behavior"]):
                relevance_score += 2
                matching_specializations.append("consumer analysis")
            
            if relevance_score > 0:
                # Get additional info from knowledge base
                agent_info = self.knowledge_base.get_agent_information(agent)
                
                recommendations.append({
                    "agent": agent,
                    "relevance_score": relevance_score,
                    "matching_specializations": matching_specializations,
                    "description": f"Specialized in {', '.join(matching_specializations)}",
                    "knowledge_base_info": agent_info
                })
        
        # Sort by relevance score
        recommendations.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return recommendations[:3]  # Return top 3 recommendations
    
    def recommend_tools(self, analysis_type: str) -> List[Dict[str, Any]]:
        """
        Recommend appropriate tools for a given analysis type.
        
        Args:
            analysis_type: Type of analysis needed
            
        Returns:
            List of recommended tools with explanations
        """
        analysis_lower = analysis_type.lower()
        recommendations = []
        
        for category, tools in self.tool_categories.items():
            if any(word in analysis_lower for word in category.split('_')):
                for tool in tools:
                    # Get tool info from knowledge base
                    tool_info = self.knowledge_base.get_tool_information(tool)
                    
                    recommendations.append({
                        "tool": tool,
                        "category": category,
                        "description": f"Tool for {category.replace('_', ' ')}",
                        "knowledge_base_info": tool_info
                    })
        
        return recommendations
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query using self-corrective RAG with hallucination detection and web search fallback.
        
        Args:
            query: User's query string
            
        Returns:
            Dictionary with response information
        """
        try:
            # Step 1: Use self-corrective RAG to get the best possible answer
            corrective_result = self.self_corrective_rag.process_query_with_correction(query)
            
            # Classify intent for additional processing
            intent = self.classify_intent(query)
            
            # Check if this looks like an analysis request
            analysis_keywords = [
                "analyze", "analysis", "compare", "performance", "forecast", "roi", "market share", "trends",
                "profit", "profitability", "margin", "margins", "revenue", "cost", "which", "what", "show me",
                "calculate", "identify", "find", "determine", "evaluate", "assess", "measure", "develop", 
                "create", "strategy", "content strategy", "plan", "planning", "design", "build", "recommend",
                "suggest", "optimize", "improve", "enhance", "target", "targeting", "segment", "segmentation"
            ]
            is_analysis_request = any(keyword in query.lower() for keyword in analysis_keywords)
            
            # Initialize response with corrective RAG results
            response = {
                "query": query,
                "intent": intent if not is_analysis_request else "analysis_request",
                "recommendations": {},
                "knowledge_results": corrective_result.get("documents", []),
                "suggested_actions": [],
                "confidence": corrective_result.get("confidence", 0.5),
                "corrective_rag_result": corrective_result,  # Include full corrective result
                "answer": corrective_result.get("answer", ""),
                "source": corrective_result.get("source", "unknown")
            }
            
            # Add agent recommendations for analysis requests
            if response["intent"] == "analysis_request" or intent in ["agent_inquiry", "tool_inquiry"]:
                agent_recommendations = self.recommend_agents(query)
                response["recommendations"]["agents"] = agent_recommendations
                
                response["suggested_actions"] = [
                    "Review the recommended agents for your analysis",
                    "Confirm the analysis parameters", 
                    "Run the analysis with the selected configuration"
                ]
            
            # Add tool recommendations for tool inquiries
            if intent == "tool_inquiry":
                tool_recommendations = self.recommend_tools(query)
                response["recommendations"]["tools"] = tool_recommendations
                
                response["suggested_actions"] = [
                    "Use recommended tools for your analysis",
                    "Check tool parameters and requirements",
                    "Review tool output formats and capabilities"
                ]
            
            # Add workflow information for workflow inquiries
            elif intent == "workflow_inquiry":
                workflow_info = self.knowledge_base.get_workflow_information()
                response["recommendations"]["workflows"] = workflow_info
                
                response["suggested_actions"] = [
                    "Choose appropriate workflow type (LangGraph vs CrewAI)",
                    "Configure workflow parameters",
                    "Review execution options and optimization settings"
                ]
            
            # Add feature information for feature inquiries
            elif intent == "feature_inquiry":
                feature_info = self.knowledge_base.get_feature_capabilities()
                response["recommendations"]["features"] = feature_info
                
                response["suggested_actions"] = [
                    "Explore system capabilities and features",
                    "Review implementation guides",
                    "Check optimization and performance features"
                ]
            
            # Default suggested actions for other intents
            else:
                response["suggested_actions"] = [
                    "Review the provided information",
                    "Ask more specific questions for detailed guidance",
                    "Explore related features and capabilities"
                ]
            
            # Add web search indicator if used
            if corrective_result.get("source") == "web_search":
                response["suggested_actions"].append("Note: Information sourced from web search due to limited knowledge base coverage")
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Error processing query: {e}")
            return {
                "query": query,
                "error": str(e),
                "confidence": 0.1,
                "suggested_actions": ["Please try rephrasing your question"],
                "answer": "I encountered an error processing your request. Please try rephrasing your question.",
                "source": "error"
            }
    
    def format_response(self, response_data: Dict[str, Any]) -> str:
        """
        Format the response data into a user-friendly string with self-corrective RAG results.
        
        Args:
            response_data: Response data from process_query
            
        Returns:
            Formatted response string
        """
        try:
            query = response_data.get("query", "")
            intent = response_data.get("intent", "general_inquiry")
            confidence = response_data.get("confidence", 0.5)
            answer = response_data.get("answer", "")
            source = response_data.get("source", "unknown")
            corrective_result = response_data.get("corrective_rag_result", {})
            
            # Start building response
            response_parts = []
            
            # Add greeting and intent recognition
            response_parts.append(f"🤖 **Marketing Research Assistant** (Confidence: {confidence:.0%})")
            response_parts.append(f"I understand you're asking about: **{intent.replace('_', ' ').title()}**\n")
            
            # Add the main answer from self-corrective RAG
            if answer and answer.strip():
                response_parts.append("📋 **Answer:**")
                response_parts.append(answer)
                response_parts.append("")
                
                # Add source information
                if source == "knowledge_base":
                    response_parts.append("✅ *Answer generated from knowledge base with hallucination detection*")
                elif source == "web_search":
                    response_parts.append("🌐 *Answer generated from web search (knowledge base insufficient)*")
                    # Show web search results if available
                    web_results = corrective_result.get("web_results", [])
                    if web_results:
                        response_parts.append("\n🔍 **Web Search Sources:**")
                        for i, result in enumerate(web_results[:2], 1):
                            if isinstance(result, dict):
                                title = result.get("title", f"Source {i}")
                                url = result.get("url", "")
                                response_parts.append(f"{i}. {title}")
                                if url:
                                    response_parts.append(f"   {url}")
                elif source == "error":
                    response_parts.append("⚠️ *Error occurred during processing*")
                
                response_parts.append("")
            
            # Add quality metrics if available
            corrections_made = corrective_result.get("corrections_made", 0)
            if corrections_made > 0:
                response_parts.append(f"🔧 *Self-corrected {corrections_made} time(s) for accuracy*")
                response_parts.append("")
            
            # Add knowledge results for context
            knowledge_results = response_data.get("knowledge_results", [])
            if knowledge_results and source == "knowledge_base":
                response_parts.append("📚 **Supporting Documents:**")
                for i, result in enumerate(knowledge_results[:2], 1):
                    metadata = result.get("metadata", {})
                    source_name = metadata.get("file_name", f"Document {i}")
                    response_parts.append(f"{i}. {source_name}")
                response_parts.append("")
            
            # Add recommendations
            recommendations = response_data.get("recommendations", {})
            
            # Agent recommendations
            if "agents" in recommendations:
                response_parts.append("🤖 **Recommended Agents:**")
                for agent_rec in recommendations["agents"][:3]:
                    agent = agent_rec["agent"].replace("_", " ").title()
                    specializations = ", ".join(agent_rec["matching_specializations"])
                    response_parts.append(f"• **{agent}**: {specializations}")
                response_parts.append("")
            
            # Tool recommendations
            if "tools" in recommendations:
                response_parts.append("🔧 **Recommended Tools:**")
                for tool_rec in recommendations["tools"][:3]:
                    tool = tool_rec["tool"].replace("_", " ").title()
                    category = tool_rec["category"].replace("_", " ").title()
                    response_parts.append(f"• **{tool}**: {category}")
                response_parts.append("")
            
            # Feature information
            if "features" in recommendations:
                features = recommendations["features"]
                response_parts.append("✨ **System Capabilities:**")
                for category, results in features.get("capabilities", {}).items():
                    if results:
                        response_parts.append(f"• **{category.title()}**: {len(results)} features available")
                response_parts.append("")
            
            # Workflow information
            if "workflows" in recommendations:
                workflow_info = recommendations["workflows"]
                workflow_docs = workflow_info.get("workflow_docs", [])
                if workflow_docs:
                    response_parts.append("🔄 **Workflow Options:**")
                    for doc in workflow_docs[:2]:
                        metadata = doc.get("metadata", {})
                        title = metadata.get("title", "Workflow Documentation")
                        response_parts.append(f"• {title}")
                    response_parts.append("")
            
            # Add suggested actions
            suggested_actions = response_data.get("suggested_actions", [])
            if suggested_actions:
                response_parts.append("💡 **Suggested Next Steps:**")
                for i, action in enumerate(suggested_actions, 1):
                    response_parts.append(f"{i}. {action}")
                response_parts.append("")
            
            # Add help footer only if no specific answer was provided
            if not answer or intent == "general_inquiry":
                response_parts.append("❓ **Need More Help?**")
                response_parts.append("Ask me about:")
                response_parts.append("• Specific agents and their capabilities")
                response_parts.append("• Available analysis tools and functions")
                response_parts.append("• Workflow setup and execution")
                response_parts.append("• System features and optimization")
            
            return "\n".join(response_parts)
            
        except Exception as e:
            logger.error(f"❌ Error formatting response: {e}")
            return f"I encountered an error while formatting my response: {str(e)}\n\nPlease try asking your question again."

# Global chat agent instance
_chat_agent = None

def get_chat_agent() -> RAGChatAgent:
    """Get the global RAG chat agent instance."""
    global _chat_agent
    if _chat_agent is None:
        _chat_agent = RAGChatAgent()
    return _chat_agent