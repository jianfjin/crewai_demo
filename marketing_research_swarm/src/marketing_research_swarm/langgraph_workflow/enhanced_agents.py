"""
Enhanced LangGraph Agent with Smart Tool Selection

This module extends the base LangGraph agent with smart tool selection capabilities,
implementing the hybrid approach for optimal tool execution.
"""

import logging
import time
from typing import Dict, Any, List, Optional
from datetime import datetime

from .agents import LangGraphAgent
from .smart_tool_selector import SmartToolSelector
from .state import MarketingResearchState
from ..tools.langgraph_tools import (
    calculate_roi,
    analyze_kpis,
    forecast_sales,
    plan_budget,
    analyze_brand_performance,
    calculate_market_share,
    time_series_analysis,
    cross_sectional_analysis,
    beverage_market_analysis,
    profitability_analysis,
)
# Temporarily disable context_aware_tools due to circular import
# from ..tools.context_aware_tools import get_context_aware_tools

logger = logging.getLogger(__name__)


class EnhancedLangGraphAgent(LangGraphAgent):
    """
    Enhanced LangGraph agent with smart tool selection capabilities.
    
    This agent extends the base LangGraphAgent with:
    - Query-aware tool selection
    - Tiered tool execution strategy
    - Dynamic tool filtering
    - Performance tracking
    """
    
    def __init__(self, role: str, goal: str, backstory: str, tools: List[str], llm_config: str = "openai/gpt-4o-mini"):
        """Initialize the enhanced agent with smart tool selection."""
        super().__init__(role, goal, backstory, tools, llm_config)
        
        # Initialize smart tool selector
        self.tool_selector = SmartToolSelector()
        
        # Initialize available tools - CRITICAL FIX for tool retrieval
        self.available_tools = {}
        self._initialize_available_tools()
        
        # Performance tracking
        self.execution_history = []
        self.tool_performance = {}
        
        logger.info(f"Enhanced agent {role} initialized with {len(tools)} tools and {len(self.available_tools)} available tools")
    
    def _initialize_available_tools(self):
        """Initialize all available tools for fallback access."""
        try:
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
            logger.info(f"Initialized {len(self.available_tools)} available tools for enhanced agent")
        except Exception as e:
            logger.error(f"Failed to initialize available tools: {e}")
            self.available_tools = {}
    
    def execute_task_with_smart_tools(self, state: MarketingResearchState, task_description: str) -> Dict[str, Any]:
        """
        Execute the agent's task with smart tool selection.
        
        This method implements the hybrid approach:
        1. Analyze query to determine tool requirements
        2. Select tools using tiered strategy
        3. Execute tools in optimal order
        4. Track performance for future optimization
        """
        
        if not self.llm:
            raise Exception(f"No LLM available for agent {self.role}")
        
        start_time = time.time()
        
        # Get context for this agent
        context = self._get_enhanced_context(state)
        
        # Analyze query and select tools
        tool_selection = self._smart_tool_selection(task_description, context)
        
        # Execute selected tools
        tool_results = self._execute_selected_tools(tool_selection, context)
        
        # Create enhanced system message with smart tool context
        system_message = self._create_enhanced_system_message(context, tool_results)
        
        # Create enhanced task description with tool results
        enhanced_task = self._create_enhanced_task_with_smart_tools(task_description, tool_results, tool_selection)
        
        try:
            # Execute the LLM call with enhanced context
            from langchain.schema import HumanMessage
            human_message = HumanMessage(content=enhanced_task)
            response = self.llm.invoke([system_message, human_message])
            
            # Process the response and extract structured data
            result = {
                'agent_role': self.role,
                'task_description': task_description,
                'analysis': response.content,
                'timestamp': datetime.now().isoformat(),
                'context_used': list(context.keys()),
                'tools_available': self.tools,
                'tool_selection': tool_selection,
                'tool_results': tool_results,
                'execution_time': time.time() - start_time
            }
            
            # Extract structured insights from the analysis
            structured_insights = self._extract_structured_insights(response.content, context)
            result.update(structured_insights)
            
            # Track performance
            self._track_execution_performance(tool_selection, tool_results, result, start_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing enhanced task for {self.role}: {e}")
            raise
    
    def _get_enhanced_context(self, state: MarketingResearchState) -> Dict[str, Any]:
        """Get enhanced context with additional metadata for smart tool selection."""
        
        # Get base context
        from .state import get_agent_context
        context = get_agent_context(state, self.role)
        
        # Add enhanced context information
        context.update({
            'agent_role': self.role,
            'available_tools': self.tools,
            'execution_timestamp': datetime.now().isoformat(),
            'workflow_id': state.get('workflow_id', 'unknown'),
            'optimization_level': state.get('optimization_level', 'none')
        })
        
        return context
    
    def _smart_tool_selection(self, task_description: str, context: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Perform smart tool selection using the tiered approach with fallback.
        
        Returns:
            Dict with 'essential', 'contextual', 'supplementary' tool lists
        """
        
        try:
            # Use smart tool selector to determine optimal tools
            tool_selection = self.tool_selector.select_tools_for_execution(
                agent_role=self.role,
                query_text=task_description,
                available_tools=self.tools,
                context=context,
                max_tools=4  # Limit for performance
            )
            
            logger.info(f"Smart tool selection for {self.role}:")
            logger.info(f"  Essential: {tool_selection['essential']}")
            logger.info(f"  Contextual: {tool_selection['contextual']}")
            logger.info(f"  Supplementary: {tool_selection['supplementary']}")
            
            return tool_selection
            
        except Exception as e:
            logger.warning(f"Smart tool selector failed for {self.role}: {e}")
            logger.info("Falling back to default tool selection based on agent role")
            
            # FALLBACK: Use role-based tool selection
            return self._fallback_tool_selection(task_description, context)
    
    def _fallback_tool_selection(self, task_description: str, context: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        Fallback tool selection based on agent role when smart selector fails.
        """
        # Role-based tool mapping
        role_tools = {
            'Market Research Analyst': ['web_search_tool', 'data_analysis_tool', 'market_data_processor'],
            'Data Analyst': ['data_analysis_tool', 'statistical_analysis_tool', 'visualization_tool'],
            'Brand Performance Analyst': ['brand_analysis_tool', 'competitive_analysis_tool', 'performance_metrics_tool'],
            'Sales Forecast Analyst': ['forecasting_tool', 'trend_analysis_tool', 'sales_data_processor'],
            'ROI Analysis Expert': ['roi_calculator', 'financial_analysis_tool', 'cost_benefit_analyzer']
        }
        
        # Get tools for this agent's role
        essential_tools = role_tools.get(self.role, ['web_search_tool', 'data_analysis_tool'])
        
        # Ensure tools exist in available tools
        available_essential = [tool for tool in essential_tools if tool in self.available_tools]
        
        if not available_essential:
            # Last resort: use any available tools
            available_essential = list(self.available_tools.keys())[:2]
        
        logger.info(f"Fallback tool selection for {self.role}: {available_essential}")
        
        return {
            'essential': available_essential,
            'contextual': [],
            'supplementary': []
        }
    
    def _execute_selected_tools(self, tool_selection: Dict[str, List[str]], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute selected tools in optimal order using tiered strategy.
        """
        
        tool_results = {}
        execution_order = []
        
        # Phase 1: Execute essential tools
        essential_tools = tool_selection.get('essential', [])
        if essential_tools:
            essential_order = self.tool_selector.get_execution_order(essential_tools)
            execution_order.extend(essential_order)
            
            for tool_name in essential_order:
                result = self._execute_single_tool(tool_name, context)
                if result:
                    tool_results[tool_name] = result
        
        # Phase 2: Execute contextual tools
        contextual_tools = tool_selection.get('contextual', [])
        if contextual_tools:
            contextual_order = self.tool_selector.get_execution_order(contextual_tools)
            execution_order.extend(contextual_order)
            
            for tool_name in contextual_order:
                result = self._execute_single_tool(tool_name, context)
                if result:
                    tool_results[tool_name] = result
        
        # Phase 3: Execute supplementary tools (if needed)
        supplementary_tools = tool_selection.get('supplementary', [])
        if supplementary_tools:
            # Check if supplementary tools are needed
            should_execute_supplementary = self.tool_selector.should_execute_supplementary_tools(
                essential_results={k: v for k, v in tool_results.items() if k in essential_tools},
                contextual_results={k: v for k, v in tool_results.items() if k in contextual_tools},
                query_analysis=context.get('analysis_focus', ''),
                context=context
            )
            
            if should_execute_supplementary:
                supplementary_order = self.tool_selector.get_execution_order(supplementary_tools)
                execution_order.extend(supplementary_order)
                
                for tool_name in supplementary_order:
                    result = self._execute_single_tool(tool_name, context)
                    if result:
                        tool_results[tool_name] = result
                        
                logger.info(f"Executed supplementary tools: {supplementary_order}")
            else:
                logger.info("Supplementary tools not needed based on initial results")
        
        # Add execution metadata
        tool_results['_execution_metadata'] = {
            'execution_order': execution_order,
            'total_tools_executed': len([t for t in tool_results.keys() if not t.startswith('_')]),
            'essential_count': len([t for t in essential_tools if t in tool_results]),
            'contextual_count': len([t for t in contextual_tools if t in tool_results]),
            'supplementary_count': len([t for t in supplementary_tools if t in tool_results]),
            'execution_timestamp': datetime.now().isoformat()
        }
        
        return tool_results
    
    def _execute_single_tool(self, tool_name: str, context: Dict[str, Any]) -> Optional[Any]:
        """
        Execute a single tool with error handling and performance tracking.
        """
        
        start_time = time.time()
        success = False
        result = None
        
        try:
            # Get tool parameters based on context and tool type
            tool_params = self._get_tool_parameters(tool_name, context)
            
            # Execute the tool
            if tool_name in self.available_tools:
                tool_func = self.available_tools[tool_name]
                
                result = tool_func(**tool_params)
                
                success = True
                logger.debug(f"Successfully executed {tool_name}")
                
            else:
                logger.warning(f"Tool {tool_name} not found in available tools")
                
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            result = {"error": str(e), "tool": tool_name}
        
        finally:
            # Track performance
            execution_time = time.time() - start_time
            self.tool_selector.update_execution_stats(tool_name, success, execution_time)
        
        return result
    
    def _get_tool_parameters(self, tool_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get optimized parameters for tool execution based on context.
        """
        
        # Base parameters
        data_file = context.get('data_file_path', 'data/beverage_sales.csv')
        brands = context.get('brands', [])
        market_segments = context.get('market_segments', [])
        product_categories = context.get('product_categories', [])
        
        # Tool-specific parameter mapping
        tool_params = {
            'data_path': data_file,
            'brands': brands if brands else None,
            'market_segments': market_segments if market_segments else None,
            'product_categories': product_categories if product_categories else None
        }
        
        # Add tool-specific parameters
        if tool_name == 'profitability_analysis':
            tool_params.update({
                'analysis_dimension': 'brand',
                'brands': brands if brands else None,
                'market_segments': market_segments if market_segments else None
            })
        
        elif tool_name == 'cross_sectional_analysis':
            tool_params.update({
                'segment_column': 'brand',
                'value_column': 'total_revenue',
                'brands': brands if brands else None
            })
        
        elif tool_name == 'time_series_analysis':
            tool_params.update({
                'date_column': 'sale_date',
                'value_column': 'total_revenue',
                'brands': brands if brands else None,
                'categories': product_categories if product_categories else None
            })
        
        elif tool_name == 'forecast_sales':
            tool_params.update({
                'periods': context.get('forecast_periods', 30),
                'brands': brands if brands else None,
                'market_segments': market_segments if market_segments else None
            })
        
        elif tool_name == 'calculate_roi':
            tool_params.update({
                'investment': context.get('budget', 250000),
                'revenue': context.get('expected_revenue', 25000)
            })
        
        elif tool_name == 'plan_budget':
            tool_params.update({
                'total_budget': context.get('budget', 250000),
                'channels': ['Digital Marketing', 'Social Media', 'Traditional Media', 'Content Marketing'],
                'priorities': [1.5, 1.3, 1.0, 1.2]
            })
        
        elif tool_name == 'calculate_market_share':
            tool_params.update({
                'company_revenue': None,  # Will use defaults from tool
                'total_market_revenue': None,
                'brands': brands if brands else None
            })
        
        elif tool_name == 'analyze_brand_performance':
            tool_params.update({
                'brands': brands if brands else None,
                'market_segments': market_segments if market_segments else None
            })
        
        elif tool_name == 'beverage_market_analysis':
            tool_params.update({
                'brands': brands if brands else None,
                'market_segments': market_segments if market_segments else None,
                'product_categories': product_categories if product_categories else None
            })
        
        elif tool_name == 'analyze_kpis':
            tool_params.update({
                'brands': brands if brands else None,
                'market_segments': market_segments if market_segments else None
            })
        
        # Remove None values to avoid parameter issues
        return {k: v for k, v in tool_params.items() if v is not None}
    
    def _create_enhanced_system_message(self, context: Dict[str, Any], tool_results: Dict[str, Any]):
        """Create enhanced system message with smart tool context."""
        
        from langchain.schema import SystemMessage
        
        # Get execution metadata
        exec_metadata = tool_results.get('_execution_metadata', {})
        tools_executed = exec_metadata.get('total_tools_executed', 0)
        execution_order = exec_metadata.get('execution_order', [])
        
        system_content = f"""
You are a {self.role} with the following background:
{self.backstory}

Your goal is: {self.goal}

SMART TOOL EXECUTION SUMMARY:
- Tools executed: {tools_executed} tools in optimized order
- Execution sequence: {' → '.join(execution_order) if execution_order else 'None'}
- Essential tools: {exec_metadata.get('essential_count', 0)}
- Contextual tools: {exec_metadata.get('contextual_count', 0)}
- Supplementary tools: {exec_metadata.get('supplementary_count', 0)}

Current workflow context:
- Target Audience: {context.get('target_audience', 'Not specified')}
- Campaign Type: {context.get('campaign_type', 'Not specified')}
- Budget: ${context.get('budget', 0):,}
- Duration: {context.get('duration', 'Not specified')}
- Analysis Focus: {context.get('analysis_focus', 'Not specified')}
- Business Objective: {context.get('business_objective', 'Not specified')}
- Brands: {', '.join(context.get('brands', []))}
- Market Segments: {', '.join(context.get('market_segments', []))}
- Product Categories: {', '.join(context.get('product_categories', []))}

Previous Agent Results (use these to inform your analysis):
{self._format_previous_results(context)}

IMPORTANT: You have access to comprehensive analytical data from the executed tools.
Use this data to provide specific, data-driven insights and recommendations.

OUTPUT FORMAT REQUIREMENTS:
Please structure your response with clear sections to enable downstream agents to easily extract key information:

1. **Key Insights**: Start with 3-5 bullet points of your main findings from the tool data
2. **Data-Driven Analysis**: Reference specific numbers, percentages, and metrics from the tool results
3. **Brand-Specific Insights**: Include insights for each mentioned brand
4. **Recommendations**: Provide 2-4 specific, actionable recommendations based on the data
5. **Integration Points**: Explicitly reference how your analysis builds on previous agent findings

Format your response as a comprehensive analysis with these clearly marked sections.
"""
        
        return SystemMessage(content=system_content)
    
    def _create_enhanced_task_with_smart_tools(
        self, 
        task_description: str, 
        tool_results: Dict[str, Any], 
        tool_selection: Dict[str, List[str]]
    ) -> str:
        """Create enhanced task description with smart tool results."""
        
        enhanced_task = f"{task_description}\n\n"
        enhanced_task += "**COMPREHENSIVE ANALYTICAL DATA AVAILABLE:**\n"
        enhanced_task += "The following analytical tools have been intelligently selected and executed based on your query requirements:\n\n"
        
        # Group results by tier
        essential_tools = tool_selection.get('essential', [])
        contextual_tools = tool_selection.get('contextual', [])
        supplementary_tools = tool_selection.get('supplementary', [])
        
        # Display essential tool results
        if essential_tools:
            enhanced_task += "**ESSENTIAL ANALYSIS (Core Data):**\n"
            for tool_name in essential_tools:
                if tool_name in tool_results:
                    enhanced_task += self._format_tool_result_for_llm(tool_name, tool_results[tool_name])
            enhanced_task += "\n"
        
        # Display contextual tool results
        if contextual_tools:
            enhanced_task += "**CONTEXTUAL ANALYSIS (Query-Specific):**\n"
            for tool_name in contextual_tools:
                if tool_name in tool_results:
                    enhanced_task += self._format_tool_result_for_llm(tool_name, tool_results[tool_name])
            enhanced_task += "\n"
        
        # Display supplementary tool results
        if supplementary_tools:
            enhanced_task += "**SUPPLEMENTARY ANALYSIS (Additional Insights):**\n"
            for tool_name in supplementary_tools:
                if tool_name in tool_results:
                    enhanced_task += self._format_tool_result_for_llm(tool_name, tool_results[tool_name])
            enhanced_task += "\n"
        
        enhanced_task += "**ANALYSIS REQUIREMENTS:**\n"
        enhanced_task += "1. Thoroughly analyze the above tool results and extract key insights\n"
        enhanced_task += "2. Reference specific numbers, percentages, and metrics from the tool data\n"
        enhanced_task += "3. Compare performance across brands, regions, or categories using the data\n"
        enhanced_task += "4. Identify trends and patterns from the analytical results\n"
        enhanced_task += "5. Provide data-driven recommendations based on the tool findings\n"
        enhanced_task += "6. Integrate tool insights with your domain expertise\n"
        enhanced_task += "7. Highlight the most significant findings from each tier of analysis\n\n"
        
        return enhanced_task
    
    def _format_tool_result_for_llm(self, tool_name: str, tool_data: Any) -> str:
        """Format tool result for LLM consumption."""
        
        formatted_result = f"**{tool_name.replace('_', ' ').title()} Results:**\n"
        
        try:
            if isinstance(tool_data, str):
                import json
                parsed_data = json.loads(tool_data)
            else:
                parsed_data = tool_data
            
            if isinstance(parsed_data, dict):
                # Extract key insights from tool data
                if 'total_revenue' in parsed_data:
                    formatted_result += f"- Total Revenue: ${parsed_data['total_revenue']:,.2f}\n"
                if 'total_profit' in parsed_data:
                    formatted_result += f"- Total Profit: ${parsed_data['total_profit']:,.2f}\n"
                if 'average_profit_margin' in parsed_data:
                    formatted_result += f"- Average Profit Margin: {parsed_data['average_profit_margin']:.2f}%\n"
                if 'roi_percentage' in parsed_data:
                    formatted_result += f"- ROI: {parsed_data['roi_percentage']:.2f}%\n"
                
                # Add performance metrics
                if 'top_performers' in parsed_data and parsed_data['top_performers']:
                    top_performers = list(parsed_data['top_performers'].keys())[:3] if isinstance(parsed_data['top_performers'], dict) else parsed_data['top_performers'][:3]
                    formatted_result += f"- Top Performers: {top_performers}\n"
                
                if 'top_brands' in parsed_data and parsed_data['top_brands']:
                    top_brands = list(parsed_data['top_brands'].keys())[:3] if isinstance(parsed_data['top_brands'], dict) else parsed_data['top_brands'][:3]
                    formatted_result += f"- Top Brands: {top_brands}\n"
                
                # Add forecasting data
                if 'forecast_values' in parsed_data and parsed_data['forecast_values']:
                    avg_forecast = sum(parsed_data['forecast_values']) / len(parsed_data['forecast_values'])
                    formatted_result += f"- Average Forecast Value: ${avg_forecast:,.2f}\n"
                
                # Add market insights
                if 'market_opportunities' in parsed_data:
                    formatted_result += f"- Market Opportunities: {parsed_data['market_opportunities'][:2]}\n"
                
                # Add budget allocation
                if 'budget_allocation' in parsed_data:
                    formatted_result += f"- Budget Allocation: {parsed_data['budget_allocation']}\n"
                
                # Add any insights or summary
                insight_fields = ['profitability_insights', 'market_overview', 'forecast_insights', 'kpi_insights', 'brand_insights']
                for field in insight_fields:
                    if field in parsed_data:
                        formatted_result += f"- {field.replace('_', ' ').title()}: {parsed_data[field]}\n"
            
            formatted_result += "\n"
            
        except Exception as e:
            # If parsing fails, include raw data summary
            formatted_result += f"Raw data summary: {str(tool_data)[:200]}...\n\n"
        
        return formatted_result
    
    def _track_execution_performance(
        self, 
        tool_selection: Dict[str, List[str]], 
        tool_results: Dict[str, Any], 
        agent_result: Dict[str, Any], 
        start_time: float
    ):
        """Track execution performance for optimization."""
        
        execution_record = {
            'timestamp': datetime.now().isoformat(),
            'agent_role': self.role,
            'execution_time': time.time() - start_time,
            'tool_selection': tool_selection,
            'tools_executed': len([k for k in tool_results.keys() if not k.startswith('_')]),
            'success': 'error' not in agent_result,
            'result_quality': self._assess_result_quality(agent_result)
        }
        
        self.execution_history.append(execution_record)
        
        # Keep only last 100 executions
        if len(self.execution_history) > 100:
            self.execution_history = self.execution_history[-100:]
        
        logger.debug(f"Tracked execution performance for {self.role}: {execution_record}")
    
    def _assess_result_quality(self, result: Dict[str, Any]) -> float:
        """Assess the quality of the analysis result."""
        
        quality_score = 0.0
        
        # Check if analysis is present and substantial
        analysis = result.get('analysis', '')
        if len(analysis) > 500:
            quality_score += 0.3
        elif len(analysis) > 200:
            quality_score += 0.2
        
        # Check for structured insights
        if result.get('key_insights'):
            quality_score += 0.2
        
        if result.get('recommendations'):
            quality_score += 0.2
        
        if result.get('numerical_insights'):
            quality_score += 0.2
        
        # Check for tool results integration
        if result.get('tool_results'):
            quality_score += 0.1
        
        return min(quality_score, 1.0)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for this agent."""
        
        if not self.execution_history:
            return {'message': 'No execution history available'}
        
        total_executions = len(self.execution_history)
        successful_executions = sum(1 for record in self.execution_history if record['success'])
        avg_execution_time = sum(record['execution_time'] for record in self.execution_history) / total_executions
        avg_quality = sum(record['result_quality'] for record in self.execution_history) / total_executions
        
        return {
            'agent_role': self.role,
            'total_executions': total_executions,
            'success_rate': successful_executions / total_executions,
            'average_execution_time': avg_execution_time,
            'average_result_quality': avg_quality,
            'tool_performance': self.tool_selector.get_tool_performance_summary(),
            'last_execution': self.execution_history[-1] if self.execution_history else None
        }
