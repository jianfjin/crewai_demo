"""
Chat Agent for Marketing Research Dashboard
Handles conversational interactions and dynamic workflow building
"""

import json
import yaml
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.tools.tavily_search import TavilySearchResults
import logging

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class ChatAgent:
    """
    Intelligent chat agent that can:
    1. Engage in conversation with users
    2. Extract requirements and parameters
    3. Build dynamic workflows
    4. Select appropriate agents
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
        self.conversation_history = []
        self.extracted_requirements = {}
        self.recommended_agents = []
        self.workflow_ready = False
        
        # Initialize web search tool for fallback
        try:
            self.web_search_tool = TavilySearchResults(max_results=3)
        except Exception as e:
            logger.warning(f"Web search tool not available: {e}")
            self.web_search_tool = None
        
        # Load available agents and their capabilities
        self.available_agents = self._load_agents_config()
        
        # Parameter options from dashboard (matching render_sidebar exactly)
        self.parameter_options = {
            "target_markets": ["North America", "Europe", "Asia Pacific", "Latin America", "Middle East", "Africa", "Australia", "Global"],
            "product_categories": ["Cola", "Juice", "Energy", "Sports", "Citrus", "Lemon-Lime", "Orange", "Water", "Enhanced Water", "Tea", "Coffee"],
            "key_metrics": ["brand_performance", "category_trends", "regional_dynamics", "profitability_analysis", "pricing_optimization", "market_share", "customer_satisfaction", "roi"],
            "brands": ["Coca-Cola", "Pepsi", "Red Bull", "Monster Energy", "Gatorade", "Powerade", "Tropicana", "Simply Orange", "Minute Maid", "Sprite", "Fanta", "7UP", "Mountain Dew", "Dr Pepper", "Dasani Water", "Aquafina", "Vitamin Water"],
            "campaign_goals": [
                "Optimize brand portfolio performance across global markets",
                "Identify high-margin opportunities by category and region",
                "Develop pricing strategies based on profitability analysis",
                "Create targeted marketing strategies for different beverage categories",
                "Forecast sales and revenue for strategic planning",
                "Enhance brand positioning in competitive categories",
                "Increase market share in key segments",
                "Improve customer acquisition and retention"
            ]
        }
        
        # Metadata cache
        self.metadata_cache = None
        self.metadata_retrieved = False
        
        # Initialize RAG system for enhanced responses
        self.rag_system = None
        self._initialize_rag_system()
        
        # Default values
        self.default_parameters = {
            "target_markets": ["North America", "Europe", "Asia Pacific"],
            "product_categories": ["Cola", "Juice", "Energy", "Sports"],
            "key_metrics": ["brand_performance", "category_trends", "profitability_analysis"],
            "brands": ["Coca-Cola", "Pepsi", "Red Bull"],
            "campaign_goals": [
                "Optimize brand portfolio performance across global markets",
                "Identify high-margin opportunities by category and region",
                "Develop pricing strategies based on profitability analysis"
            ],
            "budget": 25000,
            "duration": 30,
            "forecast_periods": 30,
            "expected_revenue": 25000
        }
    
    def _load_agents_config(self) -> Dict[str, Any]:
        """Load available agents from configuration."""
        try:
            with open('src/marketing_research_swarm/config/agents.yaml', 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Error loading agents config: {e}")
            return {}
    
    def _initialize_rag_system(self):
        """Initialize the RAG system for enhanced responses."""
        try:
            from ..rag.chat_integration import get_chat_agent
            self.rag_system = get_chat_agent()
            logger.info("✅ RAG system initialized successfully")
        except Exception as e:
            logger.warning(f"⚠️ Could not initialize RAG system: {e}")
            self.rag_system = None
    
    def chat(self, user_message: str) -> Dict[str, Any]:
        """
        Process user message using self-corrective RAG and return response with workflow recommendations.
        
        Returns:
            Dict containing:
            - response: Chat response text
            - needs_parameters: Whether parameter selection is needed
            - parameter_options: Available parameter options if needed
            - workflow_ready: Whether workflow can be built
            - recommended_agents: List of recommended agents
            - extracted_requirements: Extracted parameters
        """
        
        # Add user message to conversation history
        self.conversation_history.append({"role": "user", "content": user_message})
        
        # Retrieve metadata if not already done
        if not self.metadata_retrieved:
            self._retrieve_metadata()
        
        # Try to use RAG system first for enhanced responses
        if self.rag_system:
            try:
                rag_response = self.rag_system.process_query(user_message)
                
                # Check if RAG provided a good answer
                if rag_response.get("answer") and rag_response.get("confidence", 0) > 0.3:
                    # Use RAG response but enhance with workflow capabilities
                    enhanced_response = self._enhance_rag_response(user_message, rag_response)
                    
                    # Add assistant response to history
                    self.conversation_history.append({"role": "assistant", "content": enhanced_response["response"]})
                    
                    return enhanced_response
                else:
                    logger.info("RAG response insufficient, falling back to standard analysis")
            except Exception as e:
                logger.warning(f"RAG system error, falling back to standard analysis: {e}")
        
        # Fallback to standard analysis if RAG is not available or insufficient
        analysis_result = self._analyze_user_intent(user_message)
        
        # Generate response based on analysis
        response = self._generate_response(analysis_result)
        
        # Add assistant response to history
        self.conversation_history.append({"role": "assistant", "content": response["response"]})
        
        return response
    
    def _enhance_rag_response(self, user_message: str, rag_response: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance RAG response with workflow capabilities and parameter extraction.
        """
        try:
            # Extract the formatted response from RAG
            rag_formatted_response = self.rag_system.format_response(rag_response)
            
            # Extract parameters from the user message for workflow building
            extracted_params = self._extract_parameters_from_message(user_message)
            
            # Get agent recommendations from RAG response
            agent_recommendations = rag_response.get("recommendations", {}).get("agents", [])
            
            # Convert RAG agent recommendations to our format
            recommended_agents = []
            for agent_rec in agent_recommendations:
                if isinstance(agent_rec, dict):
                    agent_name = agent_rec.get("agent", "")
                    if agent_name:
                        recommended_agents.append(agent_name)
                else:
                    recommended_agents.append(str(agent_rec))
            
            # Update internal state
            self.extracted_requirements.update(extracted_params)
            self.recommended_agents = recommended_agents
            
            # Determine if workflow is ready
            workflow_ready = self._determine_workflow_readiness(extracted_params, recommended_agents)
            self.workflow_ready = workflow_ready
            
            # Determine if parameters are needed
            needs_parameters = not workflow_ready and rag_response.get("intent") == "analysis_request"
            
            return {
                "response": rag_formatted_response,
                "needs_parameters": needs_parameters,
                "parameter_options": self.parameter_options if needs_parameters else {},
                "workflow_ready": workflow_ready,
                "recommended_agents": recommended_agents,
                "extracted_requirements": self.extracted_requirements,
                "rag_enhanced": True,
                "rag_source": rag_response.get("source", "unknown"),
                "rag_confidence": rag_response.get("confidence", 0.5)
            }
            
        except Exception as e:
            logger.error(f"Error enhancing RAG response: {e}")
            # Fallback to basic response
            return {
                "response": rag_response.get("answer", "I encountered an error processing your request."),
                "needs_parameters": False,
                "parameter_options": {},
                "workflow_ready": False,
                "recommended_agents": [],
                "extracted_requirements": {},
                "rag_enhanced": False
            }
    
    def _extract_parameters_from_message(self, message: str) -> Dict[str, Any]:
        """Extract marketing analysis parameters from user message."""
        import re
        
        params = {}
        message_lower = message.lower()
        
        # Extract brands
        brand_patterns = {
            r"\bcoca[- ]?cola\b": "Coca-Cola",
            r"\bpepsi\b": "Pepsi", 
            r"\bred bull\b": "Red Bull",
            r"\bmonster( energy)?\b": "Monster Energy",
            r"\bgatorade\b": "Gatorade",
            r"\bpowerade\b": "Powerade",
            r"\bsprite\b": "Sprite",
            r"\bfanta\b": "Fanta"
        }
        
        brands = []
        for pattern, brand_name in brand_patterns.items():
            if re.search(pattern, message_lower):
                if brand_name not in brands:
                    brands.append(brand_name)
        
        if brands:
            params["brands"] = brands
        
        # Extract regions
        region_patterns = {
            r"north america": "North America",
            r"europe": "Europe", 
            r"asia pacific": "Asia Pacific"
        }
        regions = []
        for pattern, region in region_patterns.items():
            if re.search(pattern, message_lower):
                regions.append(region)
        if regions:
            params["target_markets"] = regions
        
        # Extract analysis type
        if any(word in message_lower for word in ["roi", "profitability", "profit"]):
            params["key_metrics"] = ["roi", "profitability_analysis"]
        elif any(word in message_lower for word in ["forecast", "predict"]):
            params["key_metrics"] = ["forecasting"]
        elif any(word in message_lower for word in ["market share", "competition"]):
            params["key_metrics"] = ["market_share", "competitive_analysis"]
        
        return params
    
    def _determine_workflow_readiness(self, extracted_params: Dict[str, Any], recommended_agents: List[str]) -> bool:
        """Determine if we have enough information to proceed with workflow."""
        has_agents = bool(recommended_agents)
        has_context = bool(extracted_params.get("brands") or extracted_params.get("target_markets"))
        has_analysis_type = bool(extracted_params.get("key_metrics"))
        
        # If we have agents and either context or analysis type, we're ready
        return has_agents and (has_context or has_analysis_type)
    
    def _retrieve_metadata(self):
        """Retrieve metadata from the metadata agent and update parameter options."""
        try:
            from ..tools.advanced_tools import meta_analysis_tool
            
            # Get metadata from the tool
            metadata_result = meta_analysis_tool._run()
            metadata = json.loads(metadata_result)
            
            self.metadata_cache = metadata
            self.metadata_retrieved = True
            
            # Update parameter options with actual data values
            self._update_parameter_options_from_metadata(metadata)
            
            logger.info("✅ Metadata retrieved and parameter options updated")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not retrieve metadata: {e}")
            # Continue with default parameter options
            self.metadata_retrieved = True
    
    def _update_parameter_options_from_metadata(self, metadata: Dict[str, Any]):
        """Update parameter options based on actual data metadata."""
        try:
            distinct_values = metadata.get("distinct_values", {})
            
            # Update target markets (regions from data)
            if "region" in distinct_values and distinct_values["region"]:
                data_regions = distinct_values["region"]
                # Keep dashboard options but prioritize data regions
                combined_regions = list(set(data_regions + self.parameter_options["target_markets"]))
                self.parameter_options["target_markets"] = sorted(combined_regions)
            
            # Update product categories from data
            if "category" in distinct_values and distinct_values["category"]:
                data_categories = distinct_values["category"]
                # Keep dashboard options but prioritize data categories
                combined_categories = list(set(data_categories + self.parameter_options["product_categories"]))
                self.parameter_options["product_categories"] = sorted(combined_categories)
            
            # Update brands from data
            if "brand" in distinct_values and distinct_values["brand"]:
                data_brands = distinct_values["brand"]
                # Keep dashboard options but prioritize data brands
                combined_brands = list(set(data_brands + self.parameter_options["brands"]))
                self.parameter_options["brands"] = sorted(combined_brands)
            
            # Update default parameters with data-driven values
            if "region" in distinct_values and distinct_values["region"]:
                self.default_parameters["target_markets"] = distinct_values["region"][:3]
            
            if "category" in distinct_values and distinct_values["category"]:
                self.default_parameters["product_categories"] = distinct_values["category"][:4]
            
            if "brand" in distinct_values and distinct_values["brand"]:
                self.default_parameters["brands"] = distinct_values["brand"][:3]
                
        except Exception as e:
            logger.warning(f"⚠️ Could not update parameter options from metadata: {e}")
    
    def _get_metadata_insights(self) -> str:
        """Get insights from metadata for better user guidance."""
        if not self.metadata_cache:
            return ""
        
        try:
            insights = self.metadata_cache.get("insights", [])
            distinct_values = self.metadata_cache.get("distinct_values", {})
            
            insight_text = "Based on the available data:\n"
            
            # Add key insights
            for insight in insights[:3]:  # Limit to top 3 insights
                insight_text += f"• {insight}\n"
            
            # Add available options summary
            if "region" in distinct_values:
                regions = distinct_values["region"]
                insight_text += f"• Available regions: {', '.join(regions[:5])}{'...' if len(regions) > 5 else ''}\n"
            
            if "brand" in distinct_values:
                brands = distinct_values["brand"]
                insight_text += f"• Available brands: {', '.join(brands[:5])}{'...' if len(brands) > 5 else ''}\n"
            
            if "category" in distinct_values:
                categories = distinct_values["category"]
                insight_text += f"• Available categories: {', '.join(categories)}\n"
            
            return insight_text
            
        except Exception as e:
            logger.warning(f"⚠️ Could not generate metadata insights: {e}")
            return ""
    
    def _analyze_user_intent(self, user_message: str) -> Dict[str, Any]:
        """Analyze user intent and extract requirements."""
        
        # Get metadata insights for better analysis
        metadata_insights = self._get_metadata_insights()
        
        system_prompt = f"""
        You are an expert marketing research consultant. Analyze the user's message to understand their marketing research needs.
        
        Available agents and their capabilities:
        {json.dumps({k: v.get('goal', '') for k, v in self.available_agents.items()}, indent=2)}
        
        Current conversation history:
        {json.dumps(self.conversation_history[-5:], indent=2)}
        
        {metadata_insights}
        
        Available parameter options (use these exact values):
        - Target Markets: {self.parameter_options['target_markets']}
        - Product Categories: {self.parameter_options['product_categories']}
        - Key Metrics: {self.parameter_options['key_metrics']}
        - Brands: {self.parameter_options['brands']}
        - Campaign Goals: {self.parameter_options['campaign_goals']}
        
        Extract the following information from the user's message:
        1. Analysis type/goal
        2. Target markets (if mentioned) - ONLY use values from the available options above
        3. Product categories (if mentioned) - ONLY use values from the available options above
        4. Brands (if mentioned) - ONLY use values from the available options above
        5. Key metrics of interest - ONLY use values from the available options above
        6. Campaign goals - ONLY use values from the available options above
        7. Budget and timeline (if mentioned)
        8. Specific requirements or constraints
        
        Also determine:
        - What agents would be most suitable for their needs
        - What parameters are missing and need to be clarified
        - Whether the user is ready to proceed with analysis
        
        IMPORTANT: You MUST respond with ONLY valid JSON. Do not include any text before or after the JSON.
        Do not use markdown code blocks. Return only the raw JSON object.
        
        Respond with this exact JSON structure:
        {{
            "intent": "analysis_request|parameter_clarification|general_question",
            "analysis_type": "comprehensive|roi_focused|brand_performance|sales_forecast|content_strategy|custom",
            "extracted_parameters": {{
                "target_markets": [],
                "product_categories": [],
                "brands": [],
                "key_metrics": [],
                "campaign_goals": [],
                "budget": null,
                "duration": null,
                "specific_requirements": ""
            }},
            "recommended_agents": [],
            "missing_parameters": [],
            "confidence": 0.0,
            "next_action": "ask_parameters|build_workflow|continue_conversation"
        }}
        """
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_message)
            ]
            
            response = self.llm.invoke(messages)
            
            # Clean and parse the response
            response_content = response.content.strip()
            
            # Try to extract JSON from the response
            if response_content.startswith("```json"):
                # Remove markdown code blocks
                response_content = response_content.replace("```json", "").replace("```", "").strip()
            elif response_content.startswith("```"):
                # Remove any code blocks
                response_content = response_content.replace("```", "").strip()
            
            # Find JSON content if it's embedded in text
            import re
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                response_content = json_match.group()
            
            analysis = json.loads(response_content)
            
            # Validate the analysis structure
            analysis = self._validate_analysis_structure(analysis)
            
            # Update extracted requirements
            for key, value in analysis.get("extracted_parameters", {}).items():
                if value:  # Only update if value is not empty
                    self.extracted_requirements[key] = value
            
            return analysis
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in user intent analysis: {e}")
            logger.error(f"Raw response: {response.content if 'response' in locals() else 'No response'}")
            return self._get_fallback_analysis(user_message)
            
        except Exception as e:
            logger.error(f"Error analyzing user intent: {e}")
            return self._get_fallback_analysis(user_message)
    
    def _validate_analysis_structure(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and fix analysis structure."""
        
        # Ensure required keys exist
        required_keys = {
            "intent": "general_question",
            "analysis_type": "comprehensive", 
            "extracted_parameters": {},
            "recommended_agents": ["market_research_analyst"],
            "missing_parameters": [],
            "confidence": 0.0,
            "next_action": "continue_conversation"
        }
        
        for key, default_value in required_keys.items():
            if key not in analysis:
                analysis[key] = default_value
        
        # Ensure extracted_parameters has the right structure
        if not isinstance(analysis["extracted_parameters"], dict):
            analysis["extracted_parameters"] = {}
        
        param_defaults = {
            "target_markets": [],
            "product_categories": [],
            "brands": [],
            "key_metrics": [],
            "campaign_goals": [],
            "budget": None,
            "duration": None,
            "specific_requirements": ""
        }
        
        for key, default_value in param_defaults.items():
            if key not in analysis["extracted_parameters"]:
                analysis["extracted_parameters"][key] = default_value
        
        # Ensure lists are actually lists
        list_keys = ["target_markets", "product_categories", "brands", "key_metrics", "campaign_goals", "recommended_agents", "missing_parameters"]
        for key in list_keys:
            if key in analysis and not isinstance(analysis[key], list):
                analysis[key] = []
            if key in analysis["extracted_parameters"] and not isinstance(analysis["extracted_parameters"][key], list):
                analysis["extracted_parameters"][key] = []
        
        return analysis
    
    def _get_fallback_analysis(self, user_message: str) -> Dict[str, Any]:
        """Generate fallback analysis when LLM fails."""
        
        # Simple keyword-based analysis
        message_lower = user_message.lower()
        
        # Determine intent
        if any(word in message_lower for word in ["analyze", "analysis", "performance", "compare", "roi", "forecast"]):
            intent = "analysis_request"
            next_action = "ask_parameters"
        else:
            intent = "general_question"
            next_action = "continue_conversation"
        
        # Determine analysis type
        analysis_type = "comprehensive"
        if "roi" in message_lower or "profitability" in message_lower:
            analysis_type = "roi_focused"
        elif "brand" in message_lower or "performance" in message_lower:
            analysis_type = "brand_performance"
        elif "forecast" in message_lower or "predict" in message_lower:
            analysis_type = "sales_forecast"
        elif "content" in message_lower or "marketing" in message_lower or "campaign" in message_lower:
            analysis_type = "content_strategy"
        
        # Extract basic parameters using keyword matching
        extracted_params = {
            "target_markets": [],
            "product_categories": [],
            "brands": [],
            "key_metrics": [],
            "campaign_goals": [],
            "budget": None,
            "duration": None,
            "specific_requirements": user_message
        }
        
        # Simple brand extraction
        for brand in self.parameter_options.get("brands", []):
            if brand.lower() in message_lower:
                extracted_params["brands"].append(brand)
        
        # Simple region extraction
        for region in self.parameter_options.get("target_markets", []):
            if region.lower() in message_lower:
                extracted_params["target_markets"].append(region)
        
        # Simple category extraction
        for category in self.parameter_options.get("product_categories", []):
            if category.lower() in message_lower:
                extracted_params["product_categories"].append(category)
        
        # Recommend agents based on analysis type
        agent_mapping = {
            "comprehensive": ["market_research_analyst", "competitive_analyst", "data_analyst"],
            "roi_focused": ["data_analyst", "campaign_optimizer"],
            "brand_performance": ["competitive_analyst", "brand_performance_specialist"],
            "sales_forecast": ["forecasting_specialist", "data_analyst"],
            "content_strategy": ["content_strategist", "creative_copywriter"]
        }
        
        recommended_agents = agent_mapping.get(analysis_type, ["market_research_analyst"])
        
        # Determine missing parameters
        missing_params = []
        if intent == "analysis_request":
            # Only require parameters if they weren't extracted from the message
            if not extracted_params["target_markets"]:
                missing_params.append("target_markets")
            if not extracted_params["product_categories"]:
                missing_params.append("product_categories")
            # Don't require key_metrics if we have brands or regions - we can infer
            if not extracted_params["key_metrics"] and not extracted_params["brands"] and not extracted_params["target_markets"]:
                missing_params.append("key_metrics")
            
            # If we have enough info (brands or regions), we can proceed
            if extracted_params["brands"] or extracted_params["target_markets"]:
                missing_params = []  # Clear missing params if we have enough context
                next_action = "build_workflow"
        
        return {
            "intent": intent,
            "analysis_type": analysis_type,
            "extracted_parameters": extracted_params,
            "recommended_agents": recommended_agents,
            "missing_parameters": missing_params,
            "confidence": 0.5,  # Lower confidence for fallback
            "next_action": next_action
        }
    
    def _generate_response(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate appropriate response based on analysis."""
        
        intent = analysis.get("intent", "general_question")
        next_action = analysis.get("next_action", "continue_conversation")
        missing_params = analysis.get("missing_parameters", [])
        
        if next_action == "ask_parameters" and missing_params:
            return self._ask_for_parameters(missing_params, analysis)
        elif next_action == "build_workflow":
            return self._prepare_workflow(analysis)
        else:
            return self._continue_conversation(analysis)
    
    def _ask_for_parameters(self, missing_params: List[str], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Ask user to select missing parameters."""
        
        # Include metadata insights in the response
        metadata_insights = self._get_metadata_insights()
        
        response_text = "I understand you'd like to run a marketing analysis. To provide the best results, I need some additional information:\n\n"
        
        # Add metadata insights if available
        if metadata_insights:
            response_text += f"{metadata_insights}\n"
        
        parameter_options = {}
        
        for param in missing_params:
            if param in self.parameter_options:
                parameter_options[param] = self.parameter_options[param]
                response_text += f"**{param.replace('_', ' ').title()}**: Please select from the available options.\n"
        
        response_text += "\nYou can either:\n"
        response_text += "1. Select specific options for each parameter\n"
        response_text += "2. Tell me to use default values (recommended based on data)\n"
        response_text += "3. Provide your own custom values\n\n"
        response_text += "What would you prefer?"
        
        return {
            "response": response_text,
            "needs_parameters": True,
            "parameter_options": parameter_options,
            "workflow_ready": False,
            "recommended_agents": analysis.get("recommended_agents", []),
            "extracted_requirements": self.extracted_requirements
        }
    
    def _prepare_workflow(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare workflow configuration."""
        
        # Merge extracted requirements with defaults
        final_config = self.default_parameters.copy()
        final_config.update(self.extracted_requirements)
        
        # Auto-fill missing parameters with intelligent defaults based on what we have
        extracted_params = analysis.get("extracted_parameters", {})
        
        # If we have brands but no categories, infer categories
        if extracted_params.get("brands") and not final_config.get("product_categories"):
            brand_category_mapping = {
                "Coca-Cola": ["Cola", "Citrus"],
                "Pepsi": ["Cola", "Citrus"], 
                "Red Bull": ["Energy"],
                "Monster Energy": ["Energy"],
                "Gatorade": ["Sports"],
                "Powerade": ["Sports"],
                "Tropicana": ["Juice"],
                "Simply Orange": ["Juice"]
            }
            inferred_categories = []
            for brand in extracted_params["brands"]:
                if brand in brand_category_mapping:
                    inferred_categories.extend(brand_category_mapping[brand])
            if inferred_categories:
                final_config["product_categories"] = list(set(inferred_categories))
        
        # If we have no key metrics, infer from analysis type and brands
        if not final_config.get("key_metrics"):
            analysis_type = analysis.get("analysis_type", "comprehensive")
            if analysis_type == "brand_performance" or extracted_params.get("brands"):
                final_config["key_metrics"] = ["brand_performance", "category_trends", "profitability_analysis"]
            elif analysis_type == "roi_focused":
                final_config["key_metrics"] = ["profitability_analysis", "roi"]
            else:
                final_config["key_metrics"] = ["brand_performance", "category_trends", "profitability_analysis"]
        
        # Set recommended agents based on final config
        self.recommended_agents = self._select_agents_for_requirements_with_config(final_config, analysis)
        
        response_text = f"""
Perfect! I have all the information needed to build your marketing analysis workflow.

**Analysis Summary:**
- **Type**: {analysis.get('analysis_type', 'comprehensive').title()}
- **Target Markets**: {', '.join(final_config.get('target_markets', []))}
- **Product Categories**: {', '.join(final_config.get('product_categories', []))}
- **Key Metrics**: {', '.join(final_config.get('key_metrics', []))}
- **Recommended Agents**: {', '.join(self.recommended_agents)}

**Configuration:**
- Budget: ${final_config.get('budget', 25000):,}
- Duration: {final_config.get('duration', 30)} days
- Forecast Period: {final_config.get('forecast_periods', 30)} days

The workflow is ready to execute! Click "Run Analysis" to start the analysis with these agents and parameters.
        """
        
        self.workflow_ready = True
        
        return {
            "response": response_text.strip(),
            "needs_parameters": False,
            "parameter_options": {},
            "workflow_ready": True,
            "recommended_agents": self.recommended_agents,
            "extracted_requirements": final_config
        }
    
    def _continue_conversation(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Continue general conversation."""
        
        # Get metadata insights and query examples
        metadata_insights = self._get_metadata_insights()
        query_examples = self._get_query_examples()
        
        system_prompt = f"""
        You are a helpful marketing research consultant. Provide a conversational response that:
        1. Acknowledges the user's message
        2. Provides relevant information about marketing research capabilities
        3. Guides them toward defining their analysis needs with specific examples
        4. Is friendly and professional
        
        {metadata_insights}
        
        Include some of these example queries to help guide the user:
        {query_examples}
        
        Keep responses concise but informative and helpful.
        """
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"User said: {self.conversation_history[-1]['content']}")
            ]
            
            response = self.llm.invoke(messages)
            response_text = response.content
            
        except Exception as e:
            logger.error(f"Error generating conversation response: {e}")
            response_text = self._get_fallback_response_with_examples()
        
        return {
            "response": response_text,
            "needs_parameters": False,
            "parameter_options": {},
            "workflow_ready": False,
            "recommended_agents": analysis.get("recommended_agents", []),
            "extracted_requirements": self.extracted_requirements
        }
    
    def _get_query_examples(self) -> str:
        """Generate query examples based on available data."""
        examples = []
        
        if self.metadata_cache:
            distinct_values = self.metadata_cache.get("distinct_values", {})
            
            # Brand performance examples
            brands = distinct_values.get("brand", ["Coca-Cola", "Pepsi", "Red Bull"])[:3]
            regions = distinct_values.get("region", ["North America", "Europe"])[:2]
            categories = distinct_values.get("category", ["Cola", "Energy", "Sports"])[:3]
            
            examples.extend([
                f"• 'I want to analyze {brands[0]}'s performance against {brands[1]} in {regions[0]}'",
                f"• 'How is {brands[2] if len(brands) > 2 else brands[0]} performing in the {categories[1] if len(categories) > 1 else categories[0]} category?'",
                f"• 'Compare market share across {regions[0]} and {regions[1] if len(regions) > 1 else regions[0]}'",
                f"• 'Analyze {categories[0]} category trends and profitability'",
                f"• 'Calculate ROI for {categories[1] if len(categories) > 1 else categories[0]} campaigns'",
                f"• 'Forecast sales for {brands[0]} in the next quarter'",
                f"• 'What are the top performing brands in {regions[0]}?'",
                f"• 'Show me {categories[2] if len(categories) > 2 else categories[0]} category performance across all regions'"
            ])
        else:
            # Fallback examples
            examples.extend([
                "• 'I want to analyze Coca-Cola's performance against Pepsi in North America'",
                "• 'How is Red Bull performing in the Energy drink category?'",
                "• 'Compare Gatorade vs Powerade market share in Sports drinks'",
                "• 'Analyze Cola category trends and profitability'",
                "• 'Calculate ROI for Energy drink campaigns'",
                "• 'Forecast sales for premium water brands'",
                "• 'What are the top performing brands in Europe?'",
                "• 'Show me Sports category performance across all regions'"
            ])
        
        return "\n".join(examples[:6])  # Limit to 6 examples
    
    def _get_fallback_response_with_examples(self) -> str:
        """Provide a helpful fallback response with examples."""
        query_examples = self._get_query_examples()
        metadata_insights = self._get_metadata_insights()
        
        response = "I'm here to help you with marketing research analysis! "
        
        if metadata_insights:
            response += f"\n\n{metadata_insights}"
        
        response += f"""

Here are some example queries you can try:

{query_examples}

What specific analysis would you like to explore?"""
        
        return response
    
    def set_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Set parameters provided by user and check if workflow is ready."""
        
        # Update extracted requirements
        self.extracted_requirements.update(parameters)
        
        # Check if we have enough information to build workflow
        essential_params = ["target_markets", "product_categories", "key_metrics"]
        has_essential = all(self.extracted_requirements.get(param) for param in essential_params)
        
        if has_essential:
            # Auto-select agents based on requirements
            self.recommended_agents = self._select_agents_for_requirements()
            
            response_text = f"""
Great! I've updated the parameters. Here's what we have:

**Selected Parameters:**
- **Target Markets**: {', '.join(self.extracted_requirements.get('target_markets', []))}
- **Product Categories**: {', '.join(self.extracted_requirements.get('product_categories', []))}
- **Key Metrics**: {', '.join(self.extracted_requirements.get('key_metrics', []))}

**Recommended Agents**: {', '.join(self.recommended_agents)}

The workflow is ready! You can now run the analysis.
            """
            
            self.workflow_ready = True
            
            return {
                "response": response_text.strip(),
                "needs_parameters": False,
                "parameter_options": {},
                "workflow_ready": True,
                "recommended_agents": self.recommended_agents,
                "extracted_requirements": self.extracted_requirements
            }
        else:
            missing = [param for param in essential_params if not self.extracted_requirements.get(param)]
            return self._ask_for_parameters(missing, {"recommended_agents": []})
    
    def _select_agents_for_requirements(self) -> List[str]:
        """Select appropriate agents based on requirements."""
        
        agents = []
        key_metrics = self.extracted_requirements.get('key_metrics', [])
        
        # Always include market research analyst for comprehensive analysis
        agents.append('market_research_analyst')
        
        # Add agents based on key metrics
        if any(metric in key_metrics for metric in ['brand_performance', 'market_share']):
            agents.extend(['competitive_analyst', 'brand_performance_specialist'])
        
        if any(metric in key_metrics for metric in ['profitability_analysis', 'roi']):
            agents.append('data_analyst')
        
        if 'pricing_optimization' in key_metrics:
            agents.append('campaign_optimizer')
        
        # Add forecasting if needed
        if any(goal for goal in self.extracted_requirements.get('campaign_goals', []) if 'forecast' in goal.lower()):
            agents.append('forecasting_specialist')
        
        # Add content strategy if marketing goals are present
        if any(goal for goal in self.extracted_requirements.get('campaign_goals', []) if 'marketing' in goal.lower()):
            agents.extend(['content_strategist', 'creative_copywriter'])
        
        # Remove duplicates and ensure we have valid agents
        agents = list(set(agents))
        valid_agents = [agent for agent in agents if agent in self.available_agents]
        
        return valid_agents[:6]  # Limit to 6 agents for performance
    
    def _select_agents_for_requirements_with_config(self, config: Dict[str, Any], analysis: Dict[str, Any]) -> List[str]:
        """Select appropriate agents based on requirements and config."""
        
        agents = []
        key_metrics = config.get('key_metrics', [])
        analysis_type = analysis.get('analysis_type', 'comprehensive')
        brands = config.get('brands', [])
        
        # Always include market research analyst for comprehensive analysis
        agents.append('market_research_analyst')
        
        # Add agents based on analysis type and content
        if analysis_type == 'brand_performance' or brands:
            agents.extend(['competitive_analyst', 'brand_performance_specialist'])
        
        if analysis_type == 'roi_focused' or any(metric in key_metrics for metric in ['profitability_analysis', 'roi']):
            agents.append('data_analyst')
        
        if analysis_type == 'sales_forecast':
            agents.append('forecasting_specialist')
        
        if analysis_type == 'content_strategy':
            agents.extend(['content_strategist', 'creative_copywriter'])
        
        # Add agents based on key metrics
        if any(metric in key_metrics for metric in ['brand_performance', 'market_share']):
            if 'competitive_analyst' not in agents:
                agents.append('competitive_analyst')
            if 'brand_performance_specialist' not in agents:
                agents.append('brand_performance_specialist')
        
        if any(metric in key_metrics for metric in ['profitability_analysis', 'roi']):
            if 'data_analyst' not in agents:
                agents.append('data_analyst')
        
        if 'pricing_optimization' in key_metrics:
            agents.append('campaign_optimizer')
        
        # Add forecasting if needed
        if any(goal for goal in config.get('campaign_goals', []) if 'forecast' in goal.lower()):
            if 'forecasting_specialist' not in agents:
                agents.append('forecasting_specialist')
        
        # Add content strategy if marketing goals are present
        if any(goal for goal in config.get('campaign_goals', []) if 'marketing' in goal.lower()):
            if 'content_strategist' not in agents:
                agents.append('content_strategist')
            if 'creative_copywriter' not in agents:
                agents.append('creative_copywriter')
        
        # Remove duplicates and ensure we have valid agents
        agents = list(set(agents))
        valid_agents = [agent for agent in agents if agent in self.available_agents]
        
        return valid_agents[:6]  # Limit to 6 agents for performance
    
    def get_workflow_config(self) -> Dict[str, Any]:
        """Get the complete workflow configuration."""
        
        if not self.workflow_ready:
            return {}
        
        # Merge with defaults for any missing values
        config = self.default_parameters.copy()
        config.update(self.extracted_requirements)
        
        return {
            "analysis_type": "custom",
            "selected_agents": self.recommended_agents,
            "target_audience": "beverage consumers",
            "campaign_type": "comprehensive_analysis",
            "budget": config.get('budget', 25000),
            "duration": config.get('duration', 30),
            "analysis_focus": "beverage industry",
            "business_objective": "Optimize marketing performance based on data analysis",
            "competitive_landscape": "global beverage market",
            "market_segments": config.get('target_markets', self.default_parameters['target_markets']),
            "product_categories": config.get('product_categories', self.default_parameters['product_categories']),
            "key_metrics": config.get('key_metrics', self.default_parameters['key_metrics']),
            "brands": config.get('brands', self.default_parameters['brands']),
            "campaign_goals": config.get('campaign_goals', self.default_parameters['campaign_goals']),
            "forecast_periods": config.get('forecast_periods', 30),
            "expected_revenue": config.get('expected_revenue', 25000),
            "brand_metrics": {
                "brand_awareness": 75,
                "sentiment_score": 0.7,
                "market_position": "Premium"
            },
            "competitive_analysis": True,
            "market_share_analysis": True,
            "optimization_settings": {
                "token_budget": 50000,
                "context_strategy": "adaptive",
                "enable_caching": True,
                "enable_mem0": False,
                "enable_token_tracking": True,
                "enable_optimization_tools": True,
                "optimization_level": "full",
                "show_comparison": False
            }
        }
    
    def reset(self):
        """Reset chat agent state."""
        self.conversation_history = []
        self.extracted_requirements = {}
        self.recommended_agents = []
        self.workflow_ready = False
        self.metadata_cache = None
        self.metadata_retrieved = False