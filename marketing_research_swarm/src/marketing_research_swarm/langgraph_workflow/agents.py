"""
LangGraph Agent Nodes for Marketing Research Workflow

This module defines the agent nodes that replace CrewAI agents with LangGraph nodes.
Each agent is implemented as a LangGraph node function that processes the state.
"""

import yaml
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, SystemMessage
import requests

from .state import MarketingResearchState, AgentStatus, store_agent_result, store_agent_error, get_agent_context
from ..tools.advanced_tools import (
    calculate_roi, analyze_kpis, forecast_sales, plan_budget,
    analyze_brand_performance, calculate_market_share, time_series_analysis,
    cross_sectional_analysis, beverage_market_analysis, profitability_analysis
)

logger = logging.getLogger(__name__)


class LangGraphAgent:
    """Base class for LangGraph agent nodes."""
    
    def __init__(self, role: str, goal: str, backstory: str, tools: List[str], llm_config: str = "openai/gpt-4o-mini"):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.tools = tools
        self.llm_config = llm_config
        self.llm = self._initialize_llm()
        
        # Initialize tools
        self.available_tools = {
            "calculate_roi": calculate_roi,
            "analyze_kpis": analyze_kpis,
            "forecast_sales": forecast_sales,
            "plan_budget": plan_budget,
            "analyze_brand_performance": analyze_brand_performance,
            "calculate_market_share": calculate_market_share,
            "time_series_analysis": time_series_analysis,
            "cross_sectional_analysis": cross_sectional_analysis,
            "beverage_market_analysis": beverage_market_analysis,
            "profitability_analysis": profitability_analysis,
        }
        
    def _initialize_llm(self):
        """Initialize the LLM for the agent."""
        try:
            if 'OPENAI_API_KEY' in os.environ and os.environ['OPENAI_API_KEY'] != 'your_openai_api_key_here':
                return ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
            else:
                # Try Ollama as fallback
                try:
                    response = requests.get("http://localhost:11434/api/health", timeout=5)
                    response.raise_for_status()
                    return ChatOllama(model="gemma", base_url="http://localhost:11434")
                except requests.exceptions.RequestException:
                    logger.warning(f"No LLM configured for agent {self.role}")
                    return None
        except Exception as e:
            logger.error(f"Failed to initialize LLM for {self.role}: {e}")
            return None
    
    def execute_task(self, state: MarketingResearchState, task_description: str) -> Dict[str, Any]:
        """Execute the agent's task with the given state and task description."""
        if not self.llm:
            raise Exception(f"No LLM available for agent {self.role}")
        
        # Get context for this agent
        context = get_agent_context(state, self.role)
        
        # Create system message with agent's role and context
        system_message = SystemMessage(content=f"""
You are a {self.role} with the following background:
{self.backstory}

Your goal is: {self.goal}

You have access to the following tools: {', '.join(self.tools)}

Current workflow context:
- Target Audience: {context.get('target_audience', 'Not specified')}
- Campaign Type: {context.get('campaign_type', 'Not specified')}
- Budget: ${context.get('budget', 0):,}
- Duration: {context.get('duration', 'Not specified')}
- Analysis Focus: {context.get('analysis_focus', 'Not specified')}
- Business Objective: {context.get('business_objective', 'Not specified')}

Previous Agent Results (use these to inform your analysis):
{self._format_previous_results(context)}

IMPORTANT: If previous agent results are available above, you should:
1. Reference and build upon their findings in your analysis
2. Avoid duplicating work already completed by other agents
3. Focus on your specific expertise while integrating previous insights
4. Cite specific findings from other agents when relevant

OUTPUT FORMAT REQUIREMENTS:
Please structure your response with clear sections to enable downstream agents to easily extract key information:

1. **Key Insights**: Start with 3-5 bullet points of your main findings
2. **Recommendations**: Provide 2-4 specific, actionable recommendations
3. **Brand-Specific Analysis**: Include insights for each mentioned brand ({', '.join(context.get('brands', []))})
4. **Numerical Data**: Include specific percentages, dollar amounts, and metrics where applicable
5. **Integration Points**: Explicitly reference how your analysis builds on previous agent findings

Format your response as a comprehensive analysis with these clearly marked sections.
""")
        
        # Create human message with task
        human_message = HumanMessage(content=task_description)
        
        try:
            # Execute the LLM call
            response = self.llm.invoke([system_message, human_message])
            
            # Process the response and extract structured data
            result = {
                'agent_role': self.role,
                'task_description': task_description,
                'analysis': response.content,
                'timestamp': datetime.now().isoformat(),
                'context_used': list(context.keys()),
                'tools_available': self.tools
            }
            
            # Extract structured insights from the analysis
            structured_insights = self._extract_structured_insights(response.content, context)
            result.update(structured_insights)
            
            # Attach structured tool parameter suggestions based on role and context
            suggestions = self._build_tool_param_suggestions(context)
            if suggestions:
                result['tool_param_suggestions'] = suggestions
            
            # Execute relevant tools BEFORE LLM analysis so they can be included
            tool_results = self._execute_relevant_tools(context, "")  # Execute regardless of response
            
            # If we have tool results, re-run the LLM with tool data included
            if tool_results:
                result['tool_results'] = tool_results
                
                # Create enhanced human message with tool results for analysis
                enhanced_task = self._create_enhanced_task_with_tools(task_description, tool_results)
                enhanced_human_message = HumanMessage(content=enhanced_task)
                
                # Re-run LLM with tool data for deeper analysis
                enhanced_response = self.llm.invoke([system_message, enhanced_human_message])
                result['analysis'] = enhanced_response.content
                
                # Re-extract structured insights from enhanced analysis
                enhanced_insights = self._extract_structured_insights(enhanced_response.content, context)
                result.update(enhanced_insights)
            else:
                result['tool_results'] = {}
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing task for {self.role}: {e}")
            raise
    
    def _format_previous_results(self, context: Dict[str, Any]) -> str:
        """Format previous agent results for context with actual insights."""
        previous_results = []
        
        for key, value in context.items():
            if key.endswith('_results') and value:
                agent_name = key.replace('_results', '').replace('_', ' ').title()
                
                # Extract key insights from the result
                if isinstance(value, dict):
                    insights = []
                    
                    # Extract analysis summary
                    if 'analysis' in value:
                        analysis = str(value['analysis'])[:300] + "..." if len(str(value['analysis'])) > 300 else str(value['analysis'])
                        insights.append(f"Analysis: {analysis}")
                    
                    # Extract recommendations
                    if 'recommendations' in value:
                        recommendations = value['recommendations']
                        if isinstance(recommendations, list):
                            insights.append(f"Recommendations: {', '.join(recommendations[:3])}")
                        else:
                            rec_text = str(recommendations)[:200] + "..." if len(str(recommendations)) > 200 else str(recommendations)
                            insights.append(f"Recommendations: {rec_text}")
                    
                    # Extract key metrics or insights
                    if 'key_insights' in value:
                        insights.append(f"Key Insights: {', '.join(value['key_insights'][:3])}")
                    elif 'insights' in value:
                        insights.append(f"Insights: {', '.join(value['insights'][:3])}")
                    
                    # Extract tool results summaries
                    if 'tool_results' in value:
                        tool_count = len(value['tool_results'])
                        insights.append(f"Tool Analysis: {tool_count} analytical tools executed")
                    
                    # Extract structured insights for easier consumption
                    if 'brand_specific_insights' in value:
                        brand_count = len(value['brand_specific_insights'])
                        insights.append(f"Brand Insights: Analysis for {brand_count} brands")
                    
                    if 'numerical_insights' in value:
                        numerical = value['numerical_insights']
                        if 'roi_percentage' in numerical:
                            insights.append(f"ROI: {numerical['roi_percentage']}%")
                        if 'financial_figures' in numerical:
                            insights.append(f"Financial Data: {len(numerical['financial_figures'])} figures analyzed")
                    
                    # Add role-specific structured data previews
                    if 'performance_metrics' in value:
                        metrics_count = len(value['performance_metrics'])
                        insights.append(f"Performance Metrics: {metrics_count} KPIs measured")
                    
                    if 'budget_allocation' in value:
                        allocation_count = len(value['budget_allocation'])
                        insights.append(f"Budget Allocation: {allocation_count} categories planned")
                    
                    if 'forecast_values' in value:
                        forecast_count = len(value['forecast_values'])
                        insights.append(f"Forecasts: {forecast_count} projections generated")
                    
                    if insights:
                        previous_results.append(f"**{agent_name} Results:**")
                        for insight in insights:
                            previous_results.append(f"  • {insight}")
                    else:
                        previous_results.append(f"**{agent_name}:** Analysis completed")
                else:
                    previous_results.append(f"**{agent_name}:** {str(value)[:200]}...")
        
        if not previous_results:
            return "No previous results available."
        
        return "\n".join(previous_results)
    
    def _extract_structured_insights(self, analysis_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structured data from LLM analysis text for easy downstream consumption."""
        import re
        
        structured = {}
        
        # Extract key insights (look for bullet points, numbered lists, or "key insights" sections)
        key_insights = []
        insight_patterns = [
            r"(?:key insights?|main findings?|important points?):\s*(.+?)(?=\n\n|\n[A-Z]|\Z)",
            r"[•\-*]\s*(.+?)(?=\n[•\-*]|\n\n|\Z)",
            r"^\d+\.\s*(.+?)(?=\n\d+\.|\n\n|\Z)"
        ]
        
        for pattern in insight_patterns:
            matches = re.findall(pattern, analysis_text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            for match in matches[:5]:  # Limit to top 5 insights
                cleaned = match.strip().replace('\n', ' ')[:200]  # Clean and truncate
                if len(cleaned) > 20:  # Only meaningful insights
                    key_insights.append(cleaned)
        
        if key_insights:
            structured['key_insights'] = key_insights
        
        # Extract recommendations (look for recommendation sections)
        recommendations = []
        rec_patterns = [
            r"(?:recommendations?|suggestions?|next steps?):\s*(.+?)(?=\n\n|\n[A-Z]|\Z)",
            r"(?:recommend|suggest)(?:s|ed)?:?\s*(.+?)(?=\n\n|\Z)"
        ]
        
        for pattern in rec_patterns:
            matches = re.findall(pattern, analysis_text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            for match in matches[:3]:  # Limit to top 3 recommendations
                cleaned = match.strip().replace('\n', ' ')[:200]
                if len(cleaned) > 20:
                    recommendations.append(cleaned)
        
        if recommendations:
            structured['recommendations'] = recommendations
        
        # Extract brand-specific insights (based on context brands)
        brands = context.get('brands', [])
        if brands:
            brand_insights = {}
            for brand in brands:
                brand_mentions = re.findall(rf"{brand}.{0,150}", analysis_text, re.IGNORECASE)
                if brand_mentions:
                    # Get the most relevant mention
                    best_mention = max(brand_mentions, key=len)[:200]
                    brand_insights[brand] = best_mention.strip()
            
            if brand_insights:
                structured['brand_specific_insights'] = brand_insights
        
        # Extract numerical data (ROI, percentages, dollar amounts)
        numerical_data = {}
        
        # ROI percentages
        roi_matches = re.findall(r"roi\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*%", analysis_text, re.IGNORECASE)
        if roi_matches:
            numerical_data['roi_percentage'] = float(roi_matches[0])
        
        # Revenue/cost figures
        money_matches = re.findall(r"\$\s*([\d,]+(?:\.\d+)?)", analysis_text)
        if money_matches:
            amounts = [float(m.replace(',', '')) for m in money_matches]
            numerical_data['financial_figures'] = amounts
        
        # Percentage figures
        percent_matches = re.findall(r"(\d+(?:\.\d+)?)\s*%", analysis_text)
        if percent_matches:
            percentages = [float(p) for p in percent_matches[:5]]  # Top 5 percentages
            numerical_data['percentages'] = percentages
        
        if numerical_data:
            structured['numerical_insights'] = numerical_data
        
        # Role-specific structured extraction
        if self.role == 'market_research_analyst':
            structured.update(self._extract_market_research_structure(analysis_text, context))
        elif self.role == 'data_analyst':
            structured.update(self._extract_data_analysis_structure(analysis_text, context))
        elif self.role == 'content_strategist':
            structured.update(self._extract_content_strategy_structure(analysis_text, context))
        elif self.role == 'campaign_optimizer':
            structured.update(self._extract_campaign_optimization_structure(analysis_text, context))
        elif self.role == 'forecasting_specialist':
            structured.update(self._extract_forecasting_structure(analysis_text, context))
        
        return structured
    
    def _extract_market_research_structure(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract market research specific structured data."""
        import re
        
        structure = {}
        
        # Market opportunities
        opportunities = re.findall(r"(?:opportunity|potential):?\s*(.+?)(?=\n|\.|,)", text, re.IGNORECASE)
        if opportunities:
            structure['market_opportunities'] = opportunities[:3]
        
        # Competitive landscape insights
        competitive_insights = re.findall(r"(?:competitor|competition|competitive):?\s*(.+?)(?=\n\n|\Z)", text, re.IGNORECASE)
        if competitive_insights:
            structure['competitive_landscape_insights'] = competitive_insights[:3]
        
        # Market size/value
        market_values = re.findall(r"market\s+(?:size|value|worth)\s*(?:of\s*)?\$?\s*([\d,]+(?:\.\d+)?)", text, re.IGNORECASE)
        if market_values:
            structure['market_size_estimates'] = [float(v.replace(',', '')) for v in market_values]
        
        return structure
    
    def _extract_data_analysis_structure(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data analysis specific structured data."""
        import re
        
        structure = {}
        
        # Performance metrics
        metrics = {}
        metric_patterns = [
            (r"revenue:?\s*\$?([\d,]+(?:\.\d+)?)", 'revenue'),
            (r"profit:?\s*\$?([\d,]+(?:\.\d+)?)", 'profit'),
            (r"margin:?\s*([\d,]+(?:\.\d+)?)\s*%", 'profit_margin'),
            (r"growth:?\s*([\d,]+(?:\.\d+)?)\s*%", 'growth_rate'),
            (r"market\s+share:?\s*([\d,]+(?:\.\d+)?)\s*%", 'market_share')
        ]
        
        for pattern, metric_name in metric_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                metrics[metric_name] = float(matches[0].replace(',', ''))
        
        if metrics:
            structure['performance_metrics'] = metrics
        
        # Top performing brands/categories
        top_performers = re.findall(r"(?:top|best|leading|highest)\s+(?:performing\s+)?(.+?)(?:in|with|for)", text, re.IGNORECASE)
        if top_performers:
            structure['top_performers'] = top_performers[:3]
        
        return structure
    
    def _extract_content_strategy_structure(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract content strategy specific structured data."""
        import re
        
        structure = {}
        
        # Channel recommendations
        channels = re.findall(r"(?:channel|platform):?\s*(.+?)(?=\n|,|\.|for)", text, re.IGNORECASE)
        if channels:
            structure['recommended_channels'] = channels[:5]
        
        # Content types
        content_types = re.findall(r"(?:content|campaign)\s+(?:type|format):?\s*(.+?)(?=\n|,|\.|for)", text, re.IGNORECASE)
        if content_types:
            structure['content_types'] = content_types[:5]
        
        # Messaging themes
        themes = re.findall(r"(?:theme|message|messaging):?\s*(.+?)(?=\n\n|\Z)", text, re.IGNORECASE)
        if themes:
            structure['messaging_themes'] = themes[:3]
        
        return structure
    
    def _extract_campaign_optimization_structure(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract campaign optimization specific structured data."""
        import re
        
        structure = {}
        
        # Budget allocation
        budget_items = re.findall(r"(?:allocate|budget|spend):?\s*\$?([\d,]+(?:\.\d+)?)\s*(?:for|on|to)\s*(.+?)(?=\n|,|\.|and)", text, re.IGNORECASE)
        if budget_items:
            budget_allocation = {}
            for amount, item in budget_items:
                budget_allocation[item.strip()] = float(amount.replace(',', ''))
            structure['budget_allocation'] = budget_allocation
        
        # ROI projections
        roi_projections = re.findall(r"(?:projected|expected)\s+roi:?\s*([\d,]+(?:\.\d+)?)\s*%", text, re.IGNORECASE)
        if roi_projections:
            structure['roi_projections'] = [float(r.replace(',', '')) for r in roi_projections]
        
        return structure
    
    def _extract_forecasting_structure(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract forecasting specific structured data."""
        import re
        
        structure = {}
        
        # Forecast values
        forecast_values = re.findall(r"(?:forecast|predicted|projected):?\s*\$?([\d,]+(?:\.\d+)?)", text, re.IGNORECASE)
        if forecast_values:
            structure['forecast_values'] = [float(v.replace(',', '')) for v in forecast_values]
        
        # Confidence intervals
        confidence = re.findall(r"confidence:?\s*([\d,]+(?:\.\d+)?)\s*%", text, re.IGNORECASE)
        if confidence:
            structure['confidence_levels'] = [float(c.replace(',', '')) for c in confidence]
        
        # Risk factors
        risks = re.findall(r"(?:risk|concern|challenge):?\s*(.+?)(?=\n|\.|,)", text, re.IGNORECASE)
        if risks:
            structure['risk_factors'] = risks[:3]
        
        return structure
    
    def _create_enhanced_task_with_tools(self, task_description: str, tool_results: Dict[str, Any]) -> str:
        """Create enhanced task description that includes tool results for LLM analysis."""
        
        enhanced_task = f"{task_description}\n\n"
        enhanced_task += "**ANALYTICAL DATA AVAILABLE:**\n"
        enhanced_task += "The following analytical tools have been executed with results. Please analyze this data thoroughly and integrate findings into your response:\n\n"
        
        for tool_name, tool_data in tool_results.items():
            enhanced_task += f"**{tool_name.replace('_', ' ').title()} Results:**\n"
            
            # Parse tool data if it's a JSON string
            try:
                if isinstance(tool_data, str):
                    import json
                    parsed_data = json.loads(tool_data)
                else:
                    parsed_data = tool_data
                
                # Extract key insights from tool data
                if isinstance(parsed_data, dict):
                    # Add revenue/financial data
                    if 'total_revenue' in parsed_data:
                        enhanced_task += f"- Total Revenue: ${parsed_data['total_revenue']:,.2f}\n"
                    if 'total_profit' in parsed_data:
                        enhanced_task += f"- Total Profit: ${parsed_data['total_profit']:,.2f}\n"
                    if 'average_profit_margin' in parsed_data:
                        enhanced_task += f"- Average Profit Margin: {parsed_data['average_profit_margin']:.2f}%\n"
                    if 'roi_percentage' in parsed_data:
                        enhanced_task += f"- ROI: {parsed_data['roi_percentage']:.2f}%\n"
                    
                    # Add performance metrics
                    if 'top_performers' in parsed_data and parsed_data['top_performers']:
                        enhanced_task += f"- Top Performers: {list(parsed_data['top_performers'].keys())[:3]}\n"
                    if 'top_brands' in parsed_data and parsed_data['top_brands']:
                        top_brands = list(parsed_data['top_brands'].keys())[:3] if isinstance(parsed_data['top_brands'], dict) else parsed_data['top_brands'][:3]
                        enhanced_task += f"- Top Brands: {top_brands}\n"
                    
                    # Add forecasting data
                    if 'forecast_values' in parsed_data:
                        avg_forecast = sum(parsed_data['forecast_values']) / len(parsed_data['forecast_values']) if parsed_data['forecast_values'] else 0
                        enhanced_task += f"- Average Forecast Value: ${avg_forecast:,.2f}\n"
                    
                    # Add market insights
                    if 'market_opportunities' in parsed_data:
                        enhanced_task += f"- Market Opportunities: {parsed_data['market_opportunities'][:2]}\n"
                    
                    # Add budget allocation
                    if 'budget_allocation' in parsed_data:
                        enhanced_task += f"- Budget Allocation: {parsed_data['budget_allocation']}\n"
                    
                    # Add any insights or summary
                    insight_fields = ['profitability_insights', 'market_overview', 'forecast_insights', 'kpi_insights', 'brand_insights']
                    for field in insight_fields:
                        if field in parsed_data:
                            enhanced_task += f"- {field.replace('_', ' ').title()}: {parsed_data[field]}\n"
                
                enhanced_task += "\n"
                
            except Exception as e:
                # If parsing fails, include raw data
                enhanced_task += f"Raw data: {str(tool_data)[:300]}...\n\n"
        
        enhanced_task += "**ANALYSIS REQUIREMENTS:**\n"
        enhanced_task += "1. Thoroughly analyze the above tool results and extract key insights\n"
        enhanced_task += "2. Reference specific numbers, percentages, and metrics from the tool data\n"
        enhanced_task += "3. Compare performance across brands, regions, or categories using the data\n"
        enhanced_task += "4. Identify trends and patterns from the analytical results\n"
        enhanced_task += "5. Provide data-driven recommendations based on the tool findings\n"
        enhanced_task += "6. Integrate tool insights with your domain expertise\n\n"
        
        return enhanced_task
    
    def _build_tool_param_suggestions(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create structured tool parameter suggestions per agent role consistent with tool signatures."""
        suggestions = {}
        data_file = context.get('data_file_path', 'data/beverage_sales.csv')
        # Resolve to an existing path if possible (prefer absolute workspace path)
        data_path = data_file
        try:
            import os
            candidates = [
                data_file,
                '/workspaces/crewai_demo/marketing_research_swarm/data/beverage_sales.csv',
                'data/beverage_sales.csv'
            ]
            for p in candidates:
                if p and os.path.exists(p):
                    data_path = p
                    break
        except Exception:
            pass
        
        # Get specific analysis parameters from context
        brands = context.get('brands', [])
        market_segments = context.get('market_segments', [])
        product_categories = context.get('product_categories', [])
        if self.role == 'data_analyst':
            if 'profitability_analysis' in self.tools:
                suggestions['profitability_analysis'] = {
                    'data_path': data_path,
                    'analysis_dimension': 'brand',
                    'brands': brands if brands else None,
                    'market_segments': market_segments if market_segments else None
                }
            if 'cross_sectional_analysis' in self.tools:
                suggestions['cross_sectional_analysis'] = {
                    'data_path': data_path,
                    'segment_column': 'brand',
                    'value_column': 'total_revenue',
                    'brands': brands if brands else None
                }
            if 'time_series_analysis' in self.tools:
                suggestions['time_series_analysis'] = {
                    'data_path': data_path,
                    'date_column': 'sale_date',
                    'value_column': 'total_revenue',
                    'brands': brands if brands else None,
                    'categories': product_categories if product_categories else None
                }
            if 'analyze_kpis' in self.tools:
                suggestions['analyze_kpis'] = {
                    'data_path': data_path,
                    'brands': brands if brands else None,
                    'market_segments': market_segments if market_segments else None
                }
        elif self.role == 'market_research_analyst' and 'beverage_market_analysis' in self.tools:
            suggestions['beverage_market_analysis'] = {
                'data_path': data_path,
                'brands': brands if brands else None,
                'market_segments': market_segments if market_segments else None,
                'product_categories': product_categories if product_categories else None
            }
        elif self.role == 'forecasting_specialist' and 'forecast_sales' in self.tools:
            suggestions['forecast_sales'] = {
                'data_path': data_path,
                'periods': context.get('forecast_periods', 30),
                'brands': brands if brands else None,
                'market_segments': market_segments if market_segments else None
            }
        elif self.role == 'campaign_optimizer':
            if 'calculate_roi' in self.tools:
                suggestions['calculate_roi'] = {
                    'investment': context.get('budget', 250000),
                    'revenue': context.get('expected_revenue', 25000),
                    'brands': brands if brands else None,
                    'campaign_type': context.get('campaign_type', 'marketing campaign')
                }
            if 'plan_budget' in self.tools:
                suggestions['plan_budget'] = {
                    'total_budget': context.get('budget', 250000),
                    'market_segments': market_segments if market_segments else None,
                    'campaign_goals': context.get('campaign_goals', [])
                }
        elif self.role == 'brand_performance_specialist':
            if 'calculate_market_share' in self.tools:
                suggestions['calculate_market_share'] = {
                    'company_revenue': None,
                    'total_market_revenue': None,
                    'brands': brands if brands else None
                }
            if 'analyze_brand_performance' in self.tools:
                suggestions['analyze_brand_performance'] = {
                    'data_path': data_path,
                    'brands': brands if brands else None,
                    'market_segments': market_segments if market_segments else None
                }
        # Only return if non-empty
        return suggestions
    
    def _execute_relevant_tools(self, context: Dict[str, Any], response_content: str) -> Dict[str, Any]:
        """Execute tools that are relevant to the agent's analysis with enhanced parameters."""
        tool_results = {}
        
        # Get data path with fallback handling
        data_file = context.get('data_file_path', 'data/beverage_sales.csv')
        
        # Get specific parameters from context
        brands = context.get('brands', [])
        market_segments = context.get('market_segments', [])
        product_categories = context.get('product_categories', [])
        
        try:
            if self.role == 'data_analyst':
                # Execute multiple analytical tools with specific parameters
                if 'profitability_analysis' in self.tools:
                    result = profitability_analysis.invoke({
                        'data_path': data_file,
                        'analysis_dimension': 'brand',
                        'brands': brands if brands else None,
                        'market_segments': market_segments if market_segments else None
                    })
                    tool_results['profitability_analysis'] = result
                
                if 'cross_sectional_analysis' in self.tools:
                    result = cross_sectional_analysis.invoke({
                        'data_path': data_file,
                        'segment_column': 'brand',
                        'value_column': 'total_revenue',
                        'brands': brands if brands else None
                    })
                    tool_results['cross_sectional_analysis'] = result
                
                if 'analyze_kpis' in self.tools:
                    result = analyze_kpis.invoke({
                        'data_path': data_file,
                        'brands': brands if brands else None,
                        'market_segments': market_segments if market_segments else None
                    })
                    tool_results['analyze_kpis'] = result
            
            elif self.role == 'market_research_analyst':
                if 'beverage_market_analysis' in self.tools:
                    result = beverage_market_analysis.invoke({
                        'data_path': data_file,
                        'brands': brands if brands else None,
                        'market_segments': market_segments if market_segments else None,
                        'product_categories': product_categories if product_categories else None
                    })
                    tool_results['beverage_market_analysis'] = result
            
            elif self.role == 'forecasting_specialist':
                if 'forecast_sales' in self.tools:
                    result = forecast_sales.invoke({
                        'data_path': data_file,
                        'periods': context.get('forecast_periods', 30),
                        'brands': brands if brands else None,
                        'market_segments': market_segments if market_segments else None
                    })
                    tool_results['forecast_sales'] = result
            
            elif self.role == 'campaign_optimizer':
                if 'calculate_roi' in self.tools:
                    result = calculate_roi.invoke({
                        'investment': context.get('budget', 250000),
                        'revenue': context.get('expected_revenue', 25000)
                    })
                    tool_results['calculate_roi'] = result
                
                if 'plan_budget' in self.tools:
                    result = plan_budget.invoke({
                        'total_budget': context.get('budget', 250000),
                        'channels': ['Digital Marketing', 'Social Media', 'Traditional Media', 'Content Marketing'],
                        'priorities': [1.5, 1.3, 1.0, 1.2]  # Higher priority for digital and social
                    })
                    tool_results['plan_budget'] = result
            
            elif self.role == 'brand_performance_specialist':
                if 'analyze_brand_performance' in self.tools:
                    result = analyze_brand_performance.invoke({
                        'data_path': data_file,
                        'brands': brands if brands else None,
                        'market_segments': market_segments if market_segments else None
                    })
                    tool_results['analyze_brand_performance'] = result
                
                if 'calculate_market_share' in self.tools:
                    result = calculate_market_share.invoke({
                        'company_revenue': None,  # Will use defaults from tool
                        'total_market_revenue': None,
                        'brands': brands if brands else None
                    })
                    tool_results['calculate_market_share'] = result
                
        except Exception as e:
            logger.warning(f"Tool execution error for {self.role}: {e}")
        
        return tool_results


def load_agent_configs(config_path: str) -> Dict[str, Dict[str, Any]]:
    """Load agent configurations from YAML file."""
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Failed to load agent configs from {config_path}: {e}")
        return {}


def create_agent_from_config(agent_name: str, agent_config: Dict[str, Any]) -> LangGraphAgent:
    """Create a LangGraph agent from configuration."""
    return LangGraphAgent(
        role=agent_config.get('role', agent_name),
        goal=agent_config.get('goal', ''),
        backstory=agent_config.get('backstory', ''),
        tools=agent_config.get('tools', []),
        llm_config=agent_config.get('llm', 'openai/gpt-4o-mini')
    )


# Agent node functions for LangGraph
def market_research_analyst_node(state: MarketingResearchState) -> MarketingResearchState:
    """Market Research Analyst node."""
    agent_role = 'market_research_analyst'
    
    try:
        # Load agent config
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'agents.yaml')
        agent_configs = load_agent_configs(config_path)
        agent_config = agent_configs.get(agent_role, {})
        
        # Create agent
        agent = create_agent_from_config(agent_role, agent_config)
        
        # Define task
        task_description = f"""
        Conduct comprehensive market research on the beverage industry using the enhanced sales data from {state.get('data_file_path', 'data/beverage_sales.csv')}.
        
        **Specific Research Focus**:
        - Target audience: {state['target_audience']}
        - Campaign type: {state.get('campaign_type', 'marketing campaign')}
        - Budget: ${state.get('budget', 0):,} over {state.get('duration', 'campaign period')}
        - Brands to research: {', '.join(state.get('brands', ['all brands']))}
        
        **Market Analysis Requirements**:
        - Market structure and dynamics for {', '.join(state['market_segments'])} segments
        - Seasonal patterns and trends in {', '.join(state['product_categories'])} categories
        - Pricing strategies and profit margins for {', '.join(state.get('brands', ['selected brands']))}
        - Regional performance analysis across {', '.join(state['market_segments'])} markets
        - Competitive landscape overview for {', '.join(state.get('brands', ['target brands']))}
        - Strategic recommendations for {state.get('analysis_focus', 'beverage market')}
        
        **Expected Deliverables**:
        - Market opportunity assessment for {state['target_audience']}
        - Brand positioning insights for: {', '.join(state.get('brands', ['selected brands']))}
        - Regional market dynamics in: {', '.join(state['market_segments'])}
        - Competitive intelligence and strategic recommendations
        """
        
        # Execute task
        result = agent.execute_task(state, task_description)
        
        # Store result in state
        state = store_agent_result(state, agent_role, result)
        
    except Exception as e:
        logger.error(f"Error in {agent_role} node: {e}")
        state = store_agent_error(state, agent_role, str(e))
    
    return state


def competitive_analyst_node(state: MarketingResearchState) -> MarketingResearchState:
    """Competitive Analyst node."""
    agent_role = 'competitive_analyst'
    
    try:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'agents.yaml')
        agent_configs = load_agent_configs(config_path)
        agent_config = agent_configs.get(agent_role, {})
        
        agent = create_agent_from_config(agent_role, agent_config)
        
        task_description = f"""
        Analyze competitive landscape, market positioning, and brand dynamics using data from {state.get('data_file_path', 'data/beverage_sales.csv')}.
        
        **Specific Competitive Analysis Requirements**:
        - Target brands for analysis: {', '.join(state.get('brands', ['all brands']))}
        - Market segments to examine: {', '.join(state['market_segments'])}
        - Product categories focus: {', '.join(state['product_categories'])}
        - Target audience context: {state['target_audience']}
        - Budget context: ${state.get('budget', 0):,} over {state.get('duration', 'campaign period')}
        
        **Competitive Intelligence Focus**:
        - Competitive positioning analysis for {', '.join(state.get('brands', ['selected brands']))}
        - Brand market share evaluation in {', '.join(state['market_segments'])} markets
        - Competitive threats and opportunities for {', '.join(state.get('brands', ['target brands']))}
        - Market structure analysis in {', '.join(state['product_categories'])} categories
        - Strategic competitive insights for {state.get('campaign_type', 'marketing campaign')}
        
        **Competitive Landscape Context**: {state['competitive_landscape']}
        
        **Expected Deliverables**:
        - Brand positioning analysis for: {', '.join(state.get('brands', ['selected brands']))}
        - Market share insights in: {', '.join(state['market_segments'])}
        - Competitive threat assessment and strategic recommendations
        """
        
        result = agent.execute_task(state, task_description)
        state = store_agent_result(state, agent_role, result)
        
    except Exception as e:
        logger.error(f"Error in {agent_role} node: {e}")
        state = store_agent_error(state, agent_role, str(e))
    
    return state


def data_analyst_node(state: MarketingResearchState) -> MarketingResearchState:
    """Data Analyst node."""
    agent_role = 'data_analyst'
    
    try:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'agents.yaml')
        agent_configs = load_agent_configs(config_path)
        agent_config = agent_configs.get(agent_role, {})
        
        agent = create_agent_from_config(agent_role, agent_config)
        
        task_description = f"""
        Perform comprehensive data analysis on beverage sales to uncover profitability insights, trends, and performance patterns.
        
        **Data Source**: Use data from {state.get('data_file_path', 'data/beverage_sales.csv')}
        
        **Specific Analysis Requirements**:
        - Analyze specifically these brands: {', '.join(state.get('brands', ['all brands']))}
        - Focus on these market segments: {', '.join(state['market_segments'])}
        - Examine these product categories: {', '.join(state['product_categories'])}
        - Track these key metrics: {', '.join(state['key_metrics'])}
        - Budget context: ${state.get('budget', 0):,} over {state.get('duration', 'campaign period')}
        
        **Analysis Focus Areas**:
        - Profitability analysis across the specified brands, categories, and regions
        - Time series analysis for trend identification in selected markets
        - Cross-sectional analysis for performance comparison between chosen brands
        - KPI analysis and performance metrics for {', '.join(state.get('brands', ['selected brands']))}
        - Statistical insights and data interpretation for {state.get('target_audience', 'target audience')}
        
        **Expected Deliverables**:
        - Brand-specific performance analysis for: {', '.join(state.get('brands', ['selected brands']))}
        - Market segment performance in: {', '.join(state['market_segments'])}
        - Actionable insights for {state.get('target_audience', 'target audience')} campaign
        """
        
        result = agent.execute_task(state, task_description)
        state = store_agent_result(state, agent_role, result)
        
    except Exception as e:
        logger.error(f"Error in {agent_role} node: {e}")
        state = store_agent_error(state, agent_role, str(e))
    
    return state


def content_strategist_node(state: MarketingResearchState) -> MarketingResearchState:
    """Content Strategist node."""
    agent_role = 'content_strategist'
    
    try:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'agents.yaml')
        agent_configs = load_agent_configs(config_path)
        agent_config = agent_configs.get(agent_role, {})
        
        agent = create_agent_from_config(agent_role, agent_config)
        
        task_description = f"""
        Develop a comprehensive content strategy for beverage brands based on market research insights.
        
        **Specific Content Strategy Requirements**:
        - Target audience: {state['target_audience']}
        - Campaign type: {state.get('campaign_type', 'marketing campaign')}
        - Budget allocation: ${state.get('budget', 0):,} over {state.get('duration', 'campaign period')}
        - Target brands: {', '.join(state.get('brands', ['selected brands']))}
        - Market segments: {', '.join(state['market_segments'])}
        - Product categories: {', '.join(state['product_categories'])}
        
        **Campaign Goals**: {', '.join(state.get('campaign_goals', ['brand awareness', 'engagement']))}
        
        **Content Strategy Focus Areas**:
        - Channel-specific content recommendations for {state.get('campaign_type', 'multi-channel campaign')}
        - Seasonal campaign strategies for {', '.join(state['product_categories'])} categories
        - Brand positioning strategies for {', '.join(state.get('brands', ['target brands']))}
        - Social media campaign ideas targeting {state['target_audience']}
        - Influencer collaboration strategies in {', '.join(state['market_segments'])} markets
        - Regional content adaptation for {', '.join(state['market_segments'])} regions
        
        **Content Requirements**:
        - Budget-conscious strategies within ${state.get('budget', 0):,} allocation
        - {state.get('duration', 'Campaign duration')}-specific content calendar
        - Brand-specific messaging for: {', '.join(state.get('brands', ['selected brands']))}
        
        **Expected Deliverables**:
        - Content strategy framework for {', '.join(state.get('brands', ['target brands']))}
        - Channel recommendations for {state.get('campaign_type', 'campaign')}
        - Regional content adaptation strategy for {', '.join(state['market_segments'])}
        
        **Integration Requirements**:
        - Build upon market research insights from previous agents
        - Reference competitive analysis findings when developing positioning strategies
        - Integrate data analysis results into content recommendations
        """
        
        result = agent.execute_task(state, task_description)
        state = store_agent_result(state, agent_role, result)
        
    except Exception as e:
        logger.error(f"Error in {agent_role} node: {e}")
        state = store_agent_error(state, agent_role, str(e))
    
    return state


def creative_copywriter_node(state: MarketingResearchState) -> MarketingResearchState:
    """Creative Copywriter node."""
    agent_role = 'creative_copywriter'
    
    try:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'agents.yaml')
        agent_configs = load_agent_configs(config_path)
        agent_config = agent_configs.get(agent_role, {})
        
        agent = create_agent_from_config(agent_role, agent_config)
        
        task_description = f"""
        Create compelling marketing copy for beverage campaigns targeting {state['target_audience']}.
        
        **Specific Copywriting Requirements**:
        - Target audience: {state['target_audience']}
        - Campaign type: {state.get('campaign_type', 'marketing campaign')}
        - Budget context: ${state.get('budget', 0):,} over {state.get('duration', 'campaign period')}
        - Key brands to write for: {', '.join(state.get('brands', ['selected brands']))}
        - Product categories: {', '.join(state['product_categories'])}
        - Market segments: {', '.join(state['market_segments'])}
        
        **Campaign Goals**: {', '.join(state.get('campaign_goals', ['brand awareness', 'engagement']))}
        
        **Copy Development Focus**:
        - Social media posts and content for {', '.join(state.get('brands', ['target brands']))}
        - Product descriptions and messaging for {', '.join(state['product_categories'])} categories
        - Campaign taglines and slogans for {state.get('campaign_type', 'campaign')}
        - Promotional content targeting {state['target_audience']}
        - Brand voice and messaging guidelines for {', '.join(state.get('brands', ['selected brands']))}
        - Regional content variations for {', '.join(state['market_segments'])} markets
        
        **Copy Requirements**:
        - Align with {state.get('analysis_focus', 'campaign focus')} messaging
        - Budget-appropriate content for ${state.get('budget', 0):,} campaign
        - {state.get('duration', 'Campaign duration')}-specific messaging
        
        **Expected Deliverables**:
        - Brand-specific copy for: {', '.join(state.get('brands', ['selected brands']))}
        - Regional copy variations for: {', '.join(state['market_segments'])}
        - Campaign messaging aligned with {', '.join(state.get('campaign_goals', ['objectives']))}
        
        **Integration Requirements**:
        - Use content strategy guidance from the Content Strategist
        - Reference market research findings for audience-appropriate messaging
        - Align copy with data analysis insights on brand performance
        """
        
        result = agent.execute_task(state, task_description)
        state = store_agent_result(state, agent_role, result)
        
    except Exception as e:
        logger.error(f"Error in {agent_role} node: {e}")
        state = store_agent_error(state, agent_role, str(e))
    
    return state


def campaign_optimizer_node(state: MarketingResearchState) -> MarketingResearchState:
    """Campaign Optimizer node."""
    agent_role = 'campaign_optimizer'
    
    try:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'agents.yaml')
        agent_configs = load_agent_configs(config_path)
        agent_config = agent_configs.get(agent_role, {})
        
        agent = create_agent_from_config(agent_role, agent_config)
        
        task_description = f"""
        Develop comprehensive optimization strategies for beverage marketing campaigns.
        
        **Specific Campaign Optimization Requirements**:
        - Total budget: ${state['budget']:,}
        - Campaign duration: {state['duration']}
        - Target audience: {state['target_audience']}
        - Campaign type: {state.get('campaign_type', 'marketing campaign')}
        - Target brands: {', '.join(state.get('brands', ['selected brands']))}
        - Market segments: {', '.join(state['market_segments'])}
        - Product categories: {', '.join(state['product_categories'])}
        - Expected revenue: ${state.get('expected_revenue', 25000):,}
        
        **Campaign Goals**: {', '.join(state.get('campaign_goals', ['optimization', 'ROI improvement']))}
        
        **Optimization Focus Areas**:
        - Budget allocation recommendations across {', '.join(state['market_segments'])} regions
        - ROI projections for {', '.join(state.get('brands', ['target brands']))} brands
        - Channel strategy optimization for {state.get('campaign_type', 'multi-channel campaign')}
        - Regional focus strategies for {', '.join(state['market_segments'])} markets
        - Performance improvement recommendations for {', '.join(state['product_categories'])} categories
        - Cost optimization suggestions within ${state['budget']:,} budget
        
        **Key Metrics to Optimize**: {', '.join(state.get('key_metrics', ['ROI', 'efficiency']))}
        
        **Expected Deliverables**:
        - Budget allocation strategy for ${state['budget']:,} across {', '.join(state['market_segments'])}
        - ROI optimization plan for {', '.join(state.get('brands', ['target brands']))}
        - Channel performance optimization for {state.get('campaign_type', 'campaign')}
        - Regional investment priorities and cost optimization recommendations
        
        **Integration Requirements**:
        - Use data analysis results to inform budget allocation decisions
        - Reference content strategy recommendations for channel optimization
        - Build upon market research insights for regional prioritization
        """
        
        result = agent.execute_task(state, task_description)
        state = store_agent_result(state, agent_role, result)
        
    except Exception as e:
        logger.error(f"Error in {agent_role} node: {e}")
        state = store_agent_error(state, agent_role, str(e))
    
    return state


def brand_performance_specialist_node(state: MarketingResearchState) -> MarketingResearchState:
    """Brand Performance Specialist node."""
    agent_role = 'brand_performance_specialist'
    
    try:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'agents.yaml')
        agent_configs = load_agent_configs(config_path)
        agent_config = agent_configs.get(agent_role, {})
        
        agent = create_agent_from_config(agent_role, agent_config)
        
        task_description = f"""
        Analyze brand performance in the beverage market using comprehensive sales data from {state.get('data_file_path', 'data/beverage_sales.csv')}.
        
        **Specific Brand Performance Analysis Requirements**:
        - Key brands to analyze: {', '.join(state.get('brands', ['selected brands']))}
        - Market segments focus: {', '.join(state['market_segments'])}
        - Product categories: {', '.join(state['product_categories'])}
        - Target audience context: {state['target_audience']}
        - Campaign budget context: ${state.get('budget', 0):,} over {state.get('duration', 'period')}
        - Brand metrics to track: {', '.join(state.get('key_metrics', ['brand_performance', 'market_share']))}
        
        **Brand Performance Focus Areas**:
        - Brand positioning and market share analysis for {', '.join(state.get('brands', ['target brands']))}
        - Competitive landscape evaluation in {', '.join(state['market_segments'])} markets
        - Brand health metrics and performance indicators for {', '.join(state.get('brands', ['selected brands']))}
        - Growth opportunities identification in {', '.join(state['product_categories'])} categories
        - Strategic brand recommendations for {state.get('campaign_type', 'marketing initiatives')}
        - Market penetration analysis across {', '.join(state['market_segments'])} regions
        
        **Brand Context**:
        - Brand awareness: {state.get('brand_metrics', {}).get('brand_awareness', 'N/A')}%
        - Sentiment score: {state.get('brand_metrics', {}).get('sentiment_score', 'N/A')}
        - Market position: {state.get('brand_metrics', {}).get('market_position', 'N/A')}
        
        **Expected Deliverables**:
        - Brand performance dashboard for: {', '.join(state.get('brands', ['selected brands']))}
        - Market share analysis in: {', '.join(state['market_segments'])}
        - Brand growth opportunities and strategic recommendations
        - Competitive positioning insights for target brands
        
        **Integration Requirements**:
        - Use competitive analysis results to assess brand positioning
        - Reference data analysis findings for performance metrics validation
        - Build upon market research insights for growth opportunity identification
        """
        
        result = agent.execute_task(state, task_description)
        state = store_agent_result(state, agent_role, result)
        
    except Exception as e:
        logger.error(f"Error in {agent_role} node: {e}")
        state = store_agent_error(state, agent_role, str(e))
    
    return state


def forecasting_specialist_node(state: MarketingResearchState) -> MarketingResearchState:
    """Forecasting Specialist node."""
    agent_role = 'forecasting_specialist'
    
    try:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'agents.yaml')
        agent_configs = load_agent_configs(config_path)
        agent_config = agent_configs.get(agent_role, {})
        
        agent = create_agent_from_config(agent_role, agent_config)
        
        task_description = f"""
        Generate accurate sales forecasts and predictive models using data from {state.get('data_file_path', 'data/beverage_sales.csv')}.
        
        **Specific Forecasting Requirements**:
        - Forecast periods: {state.get('forecast_periods', 30)} days
        - Target brands for forecasting: {', '.join(state.get('brands', ['selected brands']))}
        - Market segments: {', '.join(state['market_segments'])}
        - Product categories: {', '.join(state['product_categories'])}
        - Expected revenue target: ${state.get('expected_revenue', 25000):,}
        - Campaign context: ${state.get('budget', 0):,} over {state.get('duration', 'forecast period')}
        - Target audience: {state['target_audience']}
        
        **Forecasting Focus Areas**:
        - Sales forecasting with confidence intervals for {', '.join(state.get('brands', ['target brands']))}
        - Predictive modeling and trend analysis in {', '.join(state['market_segments'])} markets
        - Scenario planning for {state.get('campaign_type', 'marketing campaign')} impact
        - Market trend predictions for {', '.join(state['product_categories'])} categories
        - Revenue projections for {', '.join(state.get('brands', ['selected brands']))}
        - Risk assessment and forecasting accuracy for ${state.get('budget', 0):,} investment
        
        **Key Metrics to Forecast**: {', '.join(state.get('key_metrics', ['revenue', 'sales']))}
        
        **Expected Deliverables**:
        - {state.get('forecast_periods', 30)}-day sales forecast for: {', '.join(state.get('brands', ['selected brands']))}
        - Revenue projections across: {', '.join(state['market_segments'])}
        - Risk assessment for ${state.get('expected_revenue', 25000):,} revenue target
        - Market trend predictions and scenario analysis
        
        **Integration Requirements**:
        - Use market research findings to inform trend predictions
        - Reference data analysis results for historical pattern validation
        - Consider competitive analysis insights for scenario planning
        """
        
        result = agent.execute_task(state, task_description)
        state = store_agent_result(state, agent_role, result)
        
    except Exception as e:
        logger.error(f"Error in {agent_role} node: {e}")
        state = store_agent_error(state, agent_role, str(e))
    
    return state


# Agent node mapping for easy access
AGENT_NODES = {
    'market_research_analyst': market_research_analyst_node,
    'competitive_analyst': competitive_analyst_node,
    'data_analyst': data_analyst_node,
    'content_strategist': content_strategist_node,
    'creative_copywriter': creative_copywriter_node,
    'campaign_optimizer': campaign_optimizer_node,
    'brand_performance_specialist': brand_performance_specialist_node,
    'forecasting_specialist': forecasting_specialist_node,
}