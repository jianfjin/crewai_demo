"""
Reflective Report Summarizer with Multi-Step Retry Logic

This module implements an enhanced report summarizer that uses reflection to verify
answer relevance and implements a multi-step retry strategy for better results.
"""

import os
import json
import logging
import duckdb
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from langchain.schema import SystemMessage, HumanMessage

from .state import MarketingResearchState, store_agent_result, store_agent_error
from .enhanced_agents import EnhancedLangGraphAgent

logger = logging.getLogger(__name__)

class ReflectiveReportSummarizer:
    """Enhanced report summarizer with reflection and retry capabilities."""
    
    def __init__(self, agent_config: Dict[str, Any]):
        self.agent = EnhancedLangGraphAgent(
            role=agent_config.get('role', 'report_summarizer'),
            goal=agent_config.get('goal', ''),
            backstory=agent_config.get('backstory', ''),
            tools=agent_config.get('tools', []),
            llm_config=agent_config.get('llm', 'openai/gpt-4o-mini')
        )
        self.max_retries = 3
        self.available_tools = [
            'beverage_market_analysis', 'analyze_brand_performance', 'customer_churn_analysis',
            'profitability_analysis', 'time_series_analysis', 'cross_sectional_analysis',
            'forecast_sales', 'calculate_roi', 'analyze_kpis', 'calculate_market_share'
        ]
        self.available_agents = [
            'market_research_analyst', 'competitive_analyst', 'data_analyst',
            'brand_performance_specialist', 'forecasting_specialist', 'campaign_optimizer'
        ]
    
    def process_with_reflection(self, state: MarketingResearchState) -> MarketingResearchState:
        """Main processing method with reflection and retry logic."""
        
        # Extract user query and context - check multiple possible locations
        initial_inputs = state.get('initial_inputs', {})
        user_query = (
            state.get('user_query', '') or 
            initial_inputs.get('user_query', '') or 
            initial_inputs.get('query', '') or
            ''
        )
        is_chat_mode = bool(user_query.strip())
        
        logger.info(f"🔍 Chat mode detection: user_query='{user_query[:50]}...', is_chat_mode={is_chat_mode}")
        logger.info(f"🔍 State keys: {list(state.keys())}")
        logger.info(f"🔍 Initial inputs keys: {list(initial_inputs.keys())}")
        
        if not is_chat_mode:
            # For manual mode, use standard processing
            logger.info("📋 Using standard processing for manual configuration mode")
            return self._standard_processing(state)
        
        logger.info(f"🔍 Starting reflective processing for query: {user_query[:100]}...")
        
        # Check if we should skip to intelligent SQL generation for data-specific queries
        should_skip_to_sql = self._should_skip_to_intelligent_sql(user_query, state)
        
        # Step 1: Initial attempt with current agent results
        attempt_count = 0
        final_result = None
        
        while attempt_count < self.max_retries and final_result is None:
            attempt_count += 1
            logger.info(f"🔄 Attempt {attempt_count}/{self.max_retries}")
            
            if should_skip_to_sql and attempt_count == 1:
                # Skip directly to intelligent SQL for data-specific queries
                logger.info("🚀 Skipping to intelligent SQL generation for data-specific query")
                result = self._attempt_intelligent_sql(state, user_query, attempt_count)
            elif attempt_count == 1:
                # Step 1: Re-summarize existing agent outputs
                result = self._attempt_resummary(state, user_query, attempt_count)
            elif attempt_count == 2:
                # Step 2: Try intelligent SQL if not done yet, otherwise different tools
                if not should_skip_to_sql:
                    result = self._attempt_intelligent_sql(state, user_query, attempt_count)
                else:
                    result = self._attempt_different_tools(state, user_query, attempt_count)
            elif attempt_count == 3:
                # Step 3: Try different tools or fallback SQL
                if should_skip_to_sql:
                    result = self._attempt_different_tools(state, user_query, attempt_count)
                else:
                    result = self._attempt_different_agents_or_sql(state, user_query, attempt_count)
            
            # Reflect on the result
            is_relevant = self._reflect_on_relevance(user_query, result)
            
            if is_relevant:
                logger.info(f"✅ Relevant answer found on attempt {attempt_count}")
                final_result = result
                break
            else:
                logger.warning(f"❌ Answer not relevant on attempt {attempt_count}, retrying...")
        
        # If all attempts failed, create final summary
        if final_result is None:
            logger.warning("🚨 All attempts failed, creating failure summary")
            final_result = self._create_failure_summary(state, user_query, attempt_count)
        
        # Store the final result
        final_report = {
            'final_summary': final_result,
            'timestamp': datetime.now().isoformat(),
            'mode': 'chat_reflective',
            'user_query': user_query,
            'attempts_made': attempt_count,
            'reflection_enabled': True,
            'success': final_result is not None
        }
        
        state = store_agent_result(state, 'report_summarizer', final_report)
        state['final_report'] = final_report
        
        return state
    
    def _should_skip_to_intelligent_sql(self, user_query: str, state: MarketingResearchState) -> bool:
        """Determine if we should skip directly to intelligent SQL generation."""
        
        # Check if data context is available
        data_context = state.get('data_context', {})
        if "error" in data_context:
            return False
        
        query_lower = user_query.lower()
        
        # Keywords that indicate data-specific queries that would benefit from direct SQL
        sql_indicators = [
            'which', 'what', 'how many', 'how much', 'top', 'bottom', 'highest', 'lowest',
            'greatest', 'least', 'compare', 'comparison', 'versus', 'vs', 'between',
            'specific', 'exact', 'precise', 'list', 'show me', 'find', 'identify',
            'net loss', 'customer loss', 'churn', 'decline', 'decrease', 'drop',
            'performance', 'metrics', 'numbers', 'data', 'statistics', 'analysis'
        ]
        
        # Check if query contains SQL-friendly indicators
        has_sql_indicators = any(indicator in query_lower for indicator in sql_indicators)
        
        # Check if query asks for specific brands, numbers, or comparisons
        asks_for_specifics = any(word in query_lower for word in [
            'brand', 'brands', 'company', 'companies', 'quarter', 'month', 'year',
            'revenue', 'profit', 'sales', 'customers', 'market share', 'percentage'
        ])
        
        should_skip = has_sql_indicators and asks_for_specifics
        
        if should_skip:
            logger.info(f"🎯 Query identified as data-specific: will skip to intelligent SQL")
        
        return should_skip
    
    def _attempt_intelligent_sql(self, state: MarketingResearchState, user_query: str, attempt: int) -> str:
        """Use intelligent SQL generation with LLM and data context."""
        
        logger.info(f"🧠 Step {attempt}: Using intelligent SQL generation with data context")
        
        try:
            # Import intelligent SQL generator
            from .intelligent_sql_generator import get_intelligent_sql_generator
            
            # Get SQL generator with LLM
            sql_generator = get_intelligent_sql_generator(self.agent.llm)
            
            # Generate and execute SQL
            sql_result = sql_generator.generate_sql_from_query(user_query, "beverage_sales")
            
            if "error" in sql_result:
                logger.error(f"Intelligent SQL generation failed: {sql_result['error']}")
                return f"Error in intelligent SQL generation: {sql_result['error']}"
            
            # Format SQL results for analysis
            formatted_results = sql_generator.format_sql_results_for_analysis(sql_result)
            
            # Get existing agent analysis for context
            all_results = state.get('agent_results', {})
            detailed_analysis = self._extract_detailed_analysis(all_results)
            
            # Create enhanced prompt with SQL results
            enhanced_prompt = f"""
            You are an expert marketing analyst. The user asked: "{user_query}"
            
            I've used intelligent SQL generation with full data context awareness to query the database directly.
            
            INTELLIGENT SQL QUERY EXECUTED:
            ```sql
            {sql_result.get('query', 'N/A')}
            ```
            
            EXPLANATION: {sql_result.get('explanation', 'N/A')}
            
            DIRECT DATABASE RESULTS:
            {formatted_results}
            
            SUPPORTING AGENT ANALYSIS:
            {detailed_analysis}
            
            CRITICAL INSTRUCTIONS:
            - Use the DIRECT SQL RESULTS as your primary source of truth
            - The SQL was generated with full knowledge of the data schema and context
            - Provide specific brand names, numbers, and metrics from the SQL results
            - Answer the user's EXACT question with concrete data
            - Combine SQL insights with agent analysis for comprehensive understanding
            - Be specific and data-driven, not generic
            
            Provide a definitive, data-backed answer to: "{user_query}"
            """
            
            try:
                from langchain.schema import SystemMessage, HumanMessage
                
                system_message = SystemMessage(content="You are an expert marketing research analyst with direct database access and intelligent SQL generation capabilities.")
                human_message = HumanMessage(content=enhanced_prompt)
                
                response = self.agent.llm.invoke([system_message, human_message])
                return response.content
                
            except Exception as e:
                logger.error(f"Error in intelligent SQL analysis: {e}")
                return f"Error in intelligent SQL analysis: {str(e)}"
                
        except Exception as e:
            logger.error(f"Error in intelligent SQL attempt: {e}")
            return f"Error in intelligent SQL generation: {str(e)}"
    
    def _reflect_on_relevance(self, user_query: str, response: str) -> bool:
        """Reflect on whether the response is relevant to the user query."""
        
        reflection_prompt = f"""
        You are a quality assurance expert evaluating whether an AI response adequately answers a user's question.
        
        USER QUERY: "{user_query}"
        
        AI RESPONSE: "{response[:2000]}..."
        
        Evaluate if the AI response:
        1. Directly addresses the user's specific question
        2. Provides relevant, specific information (not generic statements)
        3. Includes concrete data, numbers, or specific insights when requested
        4. Answers what was actually asked (not something tangentially related)
        
        For queries about:
        - "Which brands..." - Response should list specific brand names
        - "Customer churn rates..." - Response should include churn percentages/rates
        - "Performance analysis..." - Response should include specific metrics
        - "Comparison..." - Response should provide direct comparisons
        
        Respond with ONLY "RELEVANT" or "NOT_RELEVANT" followed by a brief reason.
        """
        
        try:
            system_message = SystemMessage(content="You are a quality assurance expert.")
            human_message = HumanMessage(content=reflection_prompt)
            
            reflection_response = self.agent.llm.invoke([system_message, human_message])
            result = reflection_response.content.strip().upper()
            
            is_relevant = result.startswith("RELEVANT")
            logger.info(f"🔍 Reflection result: {result[:100]}...")
            
            return is_relevant
            
        except Exception as e:
            logger.error(f"Error in reflection: {e}")
            return False  # Conservative approach - assume not relevant if reflection fails
    
    def _attempt_resummary(self, state: MarketingResearchState, user_query: str, attempt: int) -> str:
        """Step 1: Re-analyze and re-summarize existing agent outputs."""
        
        logger.info(f"📝 Step 1 (Attempt {attempt}): Re-summarizing existing agent outputs")
        
        # Get all agent results
        all_results = state.get('agent_results', {})
        
        # Create detailed analysis of existing results
        detailed_analysis = self._extract_detailed_analysis(all_results)
        
        enhanced_prompt = f"""
        You are an expert marketing analyst. The user asked: "{user_query}"
        
        Previous agents provided analysis, but the summary wasn't relevant enough. 
        Re-analyze their outputs and provide a DIRECT, SPECIFIC answer to the user's question.
        
        AGENT ANALYSIS RESULTS:
        {detailed_analysis}
        
        CRITICAL INSTRUCTIONS:
        - Answer the user's EXACT question, not a general summary
        - If they asked for specific brands, LIST THE BRANDS with data
        - If they asked for rates/percentages, PROVIDE THE NUMBERS
        - If they asked for comparisons, MAKE DIRECT COMPARISONS
        - Use specific data points from the agent results
        - Be concrete and actionable, not generic
        
        Focus on extracting the most relevant information that directly answers: "{user_query}"
        """
        
        try:
            system_message = SystemMessage(content="You are an expert marketing research analyst focused on providing specific, relevant answers.")
            human_message = HumanMessage(content=enhanced_prompt)
            
            response = self.agent.llm.invoke([system_message, human_message])
            return response.content
            
        except Exception as e:
            logger.error(f"Error in re-summary attempt: {e}")
            return f"Error in re-analysis: {str(e)}"
    
    def _extract_detailed_analysis(self, all_results: Dict[str, Any]) -> str:
        """Extract detailed analysis from agent results."""
        
        detailed_analysis = ""
        
        for agent_name, result in all_results.items():
            if agent_name == 'report_summarizer':
                continue  # Skip previous summarizer results
                
            detailed_analysis += f"\n### {agent_name.replace('_', ' ').title()} Results:\n"
            
            if isinstance(result, dict):
                # Extract key information
                if 'analysis' in result:
                    analysis = result['analysis']
                    if isinstance(analysis, str):
                        detailed_analysis += f"Analysis: {analysis}\n"
                    elif isinstance(analysis, dict):
                        detailed_analysis += f"Analysis: {json.dumps(analysis, indent=2)}\n"
                
                # Extract tool results
                if 'tool_results' in result:
                    tool_results = result['tool_results']
                    for tool_name, tool_result in tool_results.items():
                        if tool_name.startswith('_'):
                            continue  # Skip metadata
                        detailed_analysis += f"Tool {tool_name}: {str(tool_result)[:500]}...\n"
                
                # Extract other relevant data
                for key in ['insights', 'recommendations', 'findings', 'summary']:
                    if key in result:
                        detailed_analysis += f"{key.title()}: {result[key]}\n"
                        
            elif isinstance(result, str):
                detailed_analysis += f"Result: {result}\n"
            
            detailed_analysis += "\n"
        
        return detailed_analysis
    
    def _attempt_different_tools(self, state: MarketingResearchState, user_query: str, attempt: int) -> str:
        """Step 2: Try different relevant tools if available."""
        
        logger.info(f"🔧 Step 2 (Attempt {attempt}): Trying different relevant tools")
        
        # Analyze query to determine most relevant tools
        relevant_tools = self._select_relevant_tools_for_query(user_query)
        
        # Get tools already used by agents
        used_tools = self._get_used_tools(state)
        
        # Find unused relevant tools
        unused_tools = [tool for tool in relevant_tools if tool not in used_tools]
        
        if not unused_tools:
            logger.info("No unused relevant tools found, falling back to re-analysis")
            return self._attempt_resummary(state, user_query, attempt)
        
        logger.info(f"🔧 Trying unused tools: {unused_tools}")
        
        # Execute the most relevant unused tool
        tool_results = {}
        for tool_name in unused_tools[:2]:  # Limit to 2 tools to avoid overload
            try:
                tool_result = self._execute_tool(tool_name, state)
                tool_results[tool_name] = tool_result
                logger.info(f"✅ Executed tool {tool_name}")
            except Exception as e:
                logger.error(f"❌ Failed to execute tool {tool_name}: {e}")
        
        # Combine new tool results with existing analysis
        all_results = state.get('agent_results', {})
        detailed_analysis = self._extract_detailed_analysis(all_results)
        
        # Add new tool results
        new_tool_analysis = ""
        for tool_name, tool_result in tool_results.items():
            new_tool_analysis += f"\n### Additional {tool_name} Analysis:\n{tool_result}\n"
        
        enhanced_prompt = f"""
        You are an expert marketing analyst. The user asked: "{user_query}"
        
        Previous analysis wasn't specific enough. I've run additional relevant tools to get better data.
        
        ORIGINAL AGENT ANALYSIS:
        {detailed_analysis}
        
        NEW TOOL RESULTS:
        {new_tool_analysis}
        
        CRITICAL INSTRUCTIONS:
        - Use the NEW TOOL RESULTS to provide a more specific answer
        - Focus on concrete data from the additional analysis
        - If the new tools provide specific brand names, churn rates, or performance metrics, USE THEM
        - Answer the user's EXACT question with the enhanced data
        - Be specific and data-driven, not generic
        
        Provide a direct answer to: "{user_query}"
        """
        
        try:
            system_message = SystemMessage(content="You are an expert marketing research analyst with access to enhanced analytical tools.")
            human_message = HumanMessage(content=enhanced_prompt)
            
            response = self.agent.llm.invoke([system_message, human_message])
            return response.content
            
        except Exception as e:
            logger.error(f"Error in different tools attempt: {e}")
            return f"Error in enhanced tool analysis: {str(e)}"
    
    def _attempt_different_agents_or_sql(self, state: MarketingResearchState, user_query: str, attempt: int) -> str:
        """Step 3: Try different agents or custom SQL query."""
        
        logger.info(f"🤖 Step 3 (Attempt {attempt}): Trying different agents or SQL query")
        
        # First, try to identify if we need different agents
        current_agents = list(state.get('agent_results', {}).keys())
        relevant_agents = self._select_relevant_agents_for_query(user_query)
        
        missing_agents = [agent for agent in relevant_agents if agent not in current_agents]
        
        if missing_agents:
            logger.info(f"🤖 Missing relevant agents: {missing_agents}")
            # For now, we'll note this but proceed to SQL since running new agents is complex
        
        # Try custom SQL query approach
        logger.info("🗃️ Attempting custom SQL query for direct data analysis")
        
        sql_result = self._execute_custom_sql_query(user_query)
        
        # Combine SQL results with existing analysis
        all_results = state.get('agent_results', {})
        detailed_analysis = self._extract_detailed_analysis(all_results)
        
        enhanced_prompt = f"""
        You are an expert marketing analyst. The user asked: "{user_query}"
        
        Previous approaches weren't specific enough. I've run a custom SQL query directly on the data.
        
        ORIGINAL AGENT ANALYSIS:
        {detailed_analysis}
        
        DIRECT SQL QUERY RESULTS:
        {sql_result}
        
        CRITICAL INSTRUCTIONS:
        - Prioritize the DIRECT SQL RESULTS as they come straight from the data
        - Use specific numbers, brand names, and metrics from the SQL results
        - If SQL shows specific churn rates, performance metrics, or comparisons, USE THEM
        - Combine SQL insights with agent analysis for a complete answer
        - Be concrete and data-driven
        
        Provide a definitive answer to: "{user_query}"
        """
        
        try:
            system_message = SystemMessage(content="You are an expert marketing research analyst with direct database access.")
            human_message = HumanMessage(content=enhanced_prompt)
            
            response = self.agent.llm.invoke([system_message, human_message])
            return response.content
            
        except Exception as e:
            logger.error(f"Error in different agents/SQL attempt: {e}")
            return f"Error in advanced analysis: {str(e)}"
    
    def _select_relevant_tools_for_query(self, user_query: str) -> List[str]:
        """Select the most relevant tools based on the user query."""
        
        query_lower = user_query.lower()
        relevant_tools = []
        
        # Map query keywords to relevant tools
        if any(word in query_lower for word in ['churn', 'retention', 'customer loss']):
            relevant_tools.extend(['customer_churn_analysis', 'analyze_brand_performance'])
        
        if any(word in query_lower for word in ['brand', 'performance', 'compare']):
            relevant_tools.extend(['analyze_brand_performance', 'beverage_market_analysis'])
        
        if any(word in query_lower for word in ['profit', 'roi', 'revenue']):
            relevant_tools.extend(['profitability_analysis', 'calculate_roi'])
        
        if any(word in query_lower for word in ['forecast', 'predict', 'future']):
            relevant_tools.extend(['forecast_sales', 'time_series_analysis'])
        
        if any(word in query_lower for word in ['market', 'share', 'competition']):
            relevant_tools.extend(['calculate_market_share', 'beverage_market_analysis'])
        
        if any(word in query_lower for word in ['trend', 'time', 'period']):
            relevant_tools.extend(['time_series_analysis', 'cross_sectional_analysis'])
        
        # Remove duplicates and ensure we have fallback tools
        relevant_tools = list(set(relevant_tools))
        
        if not relevant_tools:
            relevant_tools = ['beverage_market_analysis', 'analyze_brand_performance']
        
        return relevant_tools
    
    def _select_relevant_agents_for_query(self, user_query: str) -> List[str]:
        """Select the most relevant agents based on the user query."""
        
        query_lower = user_query.lower()
        relevant_agents = []
        
        if any(word in query_lower for word in ['churn', 'retention', 'customer']):
            relevant_agents.extend(['data_analyst', 'brand_performance_specialist'])
        
        if any(word in query_lower for word in ['brand', 'performance', 'compare']):
            relevant_agents.extend(['brand_performance_specialist', 'competitive_analyst'])
        
        if any(word in query_lower for word in ['market', 'research', 'analysis']):
            relevant_agents.extend(['market_research_analyst', 'competitive_analyst'])
        
        if any(word in query_lower for word in ['forecast', 'predict', 'future']):
            relevant_agents.extend(['forecasting_specialist', 'data_analyst'])
        
        if any(word in query_lower for word in ['data', 'metrics', 'kpi']):
            relevant_agents.extend(['data_analyst'])
        
        # Remove duplicates
        relevant_agents = list(set(relevant_agents))
        
        if not relevant_agents:
            relevant_agents = ['market_research_analyst', 'data_analyst']
        
        return relevant_agents
    
    def _get_used_tools(self, state: MarketingResearchState) -> List[str]:
        """Get list of tools already used by agents."""
        
        used_tools = []
        all_results = state.get('agent_results', {})
        
        for agent_name, result in all_results.items():
            if isinstance(result, dict) and 'tool_results' in result:
                tool_results = result['tool_results']
                for tool_name in tool_results.keys():
                    if not tool_name.startswith('_'):  # Skip metadata
                        used_tools.append(tool_name)
        
        return list(set(used_tools))
    
    def _execute_tool(self, tool_name: str, state: MarketingResearchState) -> str:
        """Execute a specific tool and return results."""
        
        try:
            # Import the tool dynamically
            from ..tools.langgraph_tools import (
                beverage_market_analysis, analyze_brand_performance, customer_churn_analysis,
                profitability_analysis, time_series_analysis, cross_sectional_analysis,
                forecast_sales, calculate_roi, analyze_kpis, calculate_market_share
            )
            
            tool_mapping = {
                'beverage_market_analysis': beverage_market_analysis,
                'analyze_brand_performance': analyze_brand_performance,
                'customer_churn_analysis': customer_churn_analysis,
                'profitability_analysis': profitability_analysis,
                'time_series_analysis': time_series_analysis,
                'cross_sectional_analysis': cross_sectional_analysis,
                'forecast_sales': forecast_sales,
                'calculate_roi': calculate_roi,
                'analyze_kpis': analyze_kpis,
                'calculate_market_share': calculate_market_share
            }
            
            if tool_name not in tool_mapping:
                return f"Tool {tool_name} not available"
            
            tool = tool_mapping[tool_name]
            
            # Execute tool with appropriate parameters
            if tool_name == 'customer_churn_analysis':
                result = tool.invoke({'time_period': 'quarter'})
            elif tool_name == 'profitability_analysis':
                result = tool.invoke({'analysis_dimension': 'brand'})
            elif tool_name == 'time_series_analysis':
                result = tool.invoke({'date_column': 'sale_date', 'value_column': 'total_revenue'})
            elif tool_name == 'cross_sectional_analysis':
                result = tool.invoke({'value_column': 'total_revenue'})
            elif tool_name == 'forecast_sales':
                result = tool.invoke({'periods': 30, 'forecast_column': 'total_revenue'})
            else:
                result = tool.invoke({})
            
            return str(result)
            
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return f"Error executing {tool_name}: {str(e)}"
    
    def _execute_custom_sql_query(self, user_query: str) -> str:
        """Execute custom SQL query based on user query."""
        
        logger.info("🗃️ Generating custom SQL query for direct data analysis")
        
        try:
            # Load the data
            import pandas as pd
            
            # Try to load from the CSV file
            data_path = "marketing_research_swarm/data/beverage_sales.csv"
            if os.path.exists(data_path):
                df = pd.read_csv(data_path)
            else:
                # Fallback to sample data
                from ..tools.langgraph_tools import get_cached_data
                df = get_cached_data()
            
            if df.empty:
                return "No data available for SQL analysis"
            
            # Create DuckDB connection
            conn = duckdb.connect(':memory:')
            
            # Register the dataframe
            conn.register('beverage_sales', df)
            
            # Generate SQL query based on user query
            sql_query = self._generate_sql_for_query(user_query)
            
            logger.info(f"🗃️ Executing SQL: {sql_query}")
            
            # Execute the query
            result = conn.execute(sql_query).fetchall()
            columns = [desc[0] for desc in conn.description]
            
            # Format results
            if not result:
                return "No results found from SQL query"
            
            # Convert to readable format
            formatted_result = f"SQL Query Results:\n"
            formatted_result += f"Columns: {', '.join(columns)}\n\n"
            
            for i, row in enumerate(result[:20]):  # Limit to 20 rows
                formatted_result += f"Row {i+1}: {dict(zip(columns, row))}\n"
            
            if len(result) > 20:
                formatted_result += f"\n... and {len(result) - 20} more rows"
            
            conn.close()
            return formatted_result
            
        except Exception as e:
            logger.error(f"Error in SQL query execution: {e}")
            return f"SQL query error: {str(e)}"
    
    def _generate_sql_for_query(self, user_query: str) -> str:
        """Generate appropriate SQL query based on user query."""
        
        query_lower = user_query.lower()
        
        # Customer churn analysis
        if any(word in query_lower for word in ['churn', 'retention', 'customer loss']):
            return """
            WITH quarterly_revenue AS (
                SELECT 
                    brand,
                    CONCAT(year, '-Q', quarter) as period,
                    SUM(total_revenue) as revenue
                FROM beverage_sales 
                GROUP BY brand, year, quarter, CONCAT(year, '-Q', quarter)
                ORDER BY brand, year, quarter
            ),
            revenue_changes AS (
                SELECT 
                    brand,
                    period,
                    revenue,
                    LAG(revenue) OVER (PARTITION BY brand ORDER BY period) as prev_revenue,
                    CASE 
                        WHEN LAG(revenue) OVER (PARTITION BY brand ORDER BY period) > 0 
                        THEN (revenue - LAG(revenue) OVER (PARTITION BY brand ORDER BY period)) / LAG(revenue) OVER (PARTITION BY brand ORDER BY period) * 100
                        ELSE 0 
                    END as revenue_change_pct
                FROM quarterly_revenue
            )
            SELECT 
                brand,
                COUNT(CASE WHEN revenue_change_pct < -15 THEN 1 END) as severe_declines,
                COUNT(*) as total_periods,
                ROUND(COUNT(CASE WHEN revenue_change_pct < -15 THEN 1 END) * 100.0 / COUNT(*), 2) as estimated_churn_rate,
                ROUND(AVG(revenue_change_pct), 2) as avg_revenue_change_pct
            FROM revenue_changes 
            WHERE prev_revenue IS NOT NULL
            GROUP BY brand
            ORDER BY estimated_churn_rate DESC;
            """
        
        # Brand performance comparison
        elif any(word in query_lower for word in ['brand', 'performance', 'compare']):
            return """
            SELECT 
                brand,
                COUNT(*) as total_transactions,
                SUM(units_sold) as total_units,
                ROUND(SUM(total_revenue), 2) as total_revenue,
                ROUND(AVG(total_revenue), 2) as avg_revenue_per_transaction,
                ROUND(SUM(profit), 2) as total_profit,
                ROUND(AVG(profit_margin), 2) as avg_profit_margin,
                COUNT(DISTINCT region) as regions_present
            FROM beverage_sales 
            GROUP BY brand
            ORDER BY total_revenue DESC;
            """
        
        # Market share analysis
        elif any(word in query_lower for word in ['market', 'share']):
            return """
            WITH total_market AS (
                SELECT SUM(total_revenue) as market_total FROM beverage_sales
            )
            SELECT 
                brand,
                ROUND(SUM(total_revenue), 2) as brand_revenue,
                ROUND(SUM(total_revenue) * 100.0 / (SELECT market_total FROM total_market), 2) as market_share_pct
            FROM beverage_sales
            GROUP BY brand
            ORDER BY market_share_pct DESC;
            """
        
        # Regional analysis
        elif any(word in query_lower for word in ['region', 'geographic', 'location']):
            return """
            SELECT 
                region,
                brand,
                ROUND(SUM(total_revenue), 2) as revenue,
                SUM(units_sold) as units_sold,
                ROUND(AVG(profit_margin), 2) as avg_profit_margin
            FROM beverage_sales 
            GROUP BY region, brand
            ORDER BY region, revenue DESC;
            """
        
        # Time-based trends
        elif any(word in query_lower for word in ['trend', 'time', 'quarter', 'month']):
            return """
            SELECT 
                year,
                quarter,
                brand,
                ROUND(SUM(total_revenue), 2) as quarterly_revenue,
                SUM(units_sold) as quarterly_units
            FROM beverage_sales 
            GROUP BY year, quarter, brand
            ORDER BY year, quarter, quarterly_revenue DESC;
            """
        
        # Default comprehensive query
        else:
            return """
            SELECT 
                brand,
                region,
                category,
                COUNT(*) as transactions,
                ROUND(SUM(total_revenue), 2) as total_revenue,
                ROUND(AVG(profit_margin), 2) as avg_profit_margin,
                SUM(units_sold) as total_units
            FROM beverage_sales 
            GROUP BY brand, region, category
            ORDER BY total_revenue DESC
            LIMIT 50;
            """
    
    def _create_failure_summary(self, state: MarketingResearchState, user_query: str, attempts: int) -> str:
        """Create a summary when all retry attempts have failed."""
        
        logger.warning(f"🚨 Creating failure summary after {attempts} attempts")
        
        # Get all previous results and attempts
        all_results = state.get('agent_results', {})
        detailed_analysis = self._extract_detailed_analysis(all_results)
        
        failure_summary = f"""
        I attempted to provide a specific answer to your query "{user_query}" through {attempts} different approaches, but was unable to generate a sufficiently relevant response.
        
        **Attempts Made:**
        1. **Re-analysis of Agent Results**: Tried to extract more specific insights from existing agent outputs
        2. **Additional Tool Execution**: Ran supplementary analytical tools for enhanced data
        3. **Direct SQL Analysis**: Queried the raw data directly using custom SQL
        
        **Available Analysis Results:**
        {detailed_analysis}
        
        **What We Found:**
        Based on the available data and analysis, here are the key insights that may partially address your query:
        
        - The analysis covered {len(all_results)} different analytical perspectives
        - Data includes beverage sales across multiple brands, regions, and time periods
        - Analysis tools examined market performance, brand metrics, and competitive dynamics
        
        **Limitations Encountered:**
        - The specific metrics or comparisons you requested may not be directly available in the current dataset
        - The analysis may require additional data sources or different analytical approaches
        - The query may need to be refined or broken down into more specific sub-questions
        
        **Recommendations:**
        1. Try rephrasing your question with more specific parameters
        2. Consider asking for related metrics that might be available in the data
        3. Break down complex queries into simpler, more focused questions
        
        **Next Steps:**
        - Refine the query to focus on available data dimensions (brands, regions, time periods, revenue, profit)
        - Consider alternative approaches to get the insights you need
        - Review the detailed agent analysis above for any relevant partial answers
        """
        
        return failure_summary
    
    def _standard_processing(self, state: MarketingResearchState) -> MarketingResearchState:
        """Standard processing for manual configuration mode (non-chat)."""
        
        logger.info("📋 Using standard processing for manual configuration mode")
        
        # Extract configuration and context information
        target_audience = state.get('target_audience', '')
        campaign_type = state.get('campaign_type', '')
        budget = state.get('budget', 0)
        duration = state.get('duration', '')
        analysis_focus = state.get('analysis_focus', '')
        business_objective = state.get('business_objective', '')
        competitive_landscape = state.get('competitive_landscape', '')
        campaign_goals = state.get('campaign_goals', [])
        brands = state.get('brands', [])
        market_segments = state.get('market_segments', [])
        product_categories = state.get('product_categories', [])
        
        # Consolidate all previous agent results
        all_results = state.get('agent_results', {})
        agent_errors = state.get('agent_errors', {})
        
        # Build detailed analysis summary
        detailed_analysis = self._extract_detailed_analysis(all_results)
        
        # Add error information if any
        if agent_errors:
            detailed_analysis += "### Analysis Limitations\n"
            for agent_name, error in agent_errors.items():
                detailed_analysis += f"- **{agent_name.replace('_', ' ').title()}**: {error}\n"
            detailed_analysis += "\n"
        
        # Manual configuration mode: Check against configuration parameters
        config_summary = f"""
        **TASK CONFIGURATION**:
        - Target Audience: {target_audience}
        - Campaign Type: {campaign_type}
        - Budget: ${budget:,.2f} if budget else 'Not specified'
        - Duration: {duration}
        
        **ANALYSIS FOCUS**:
        - Business Objective: {business_objective}
        - Competitive Landscape: {competitive_landscape}
        - Analysis Focus: {analysis_focus}
        
        **ADVANCED PARAMETERS**:
        - Campaign Goals: {', '.join(campaign_goals) if campaign_goals else 'Not specified'}
        - Target Brands: {', '.join(brands) if brands else 'Not specified'}
        - Market Segments: {', '.join(market_segments) if market_segments else 'Not specified'}
        - Product Categories: {', '.join(product_categories) if product_categories else 'Not specified'}
        """
        
        task_description = f"""
        You are an expert marketing research analyst tasked with creating a comprehensive final report that validates whether the analysis results satisfy the specified configuration parameters and objectives.
        
        {config_summary}
        
        **ANALYSIS RESULTS FROM SPECIALIZED AGENTS**:
        {detailed_analysis}
        
        **YOUR TASK**: 
        Create a comprehensive final report that:
        
        1. **Executive Summary**: Provide a high-level overview of whether the analysis meets the specified objectives
        2. **Configuration Alignment**: Explicitly assess how well the analysis results align with:
           - The target audience requirements
           - The campaign type and goals
           - The budget constraints and expectations
           - The specified business objectives
           - The competitive landscape analysis needs
        3. **Integrated Insights**: Synthesize insights from all agents into a cohesive narrative
        4. **Gap Analysis**: Identify any gaps between what was requested and what was delivered
        5. **Strategic Recommendations**: Provide actionable recommendations that specifically address the configuration parameters
        6. **Success Metrics**: Define how success should be measured based on the campaign goals and business objectives
        7. **Next Steps**: Outline specific next steps that align with the campaign duration and budget
        
        **CRITICAL REQUIREMENTS**:
        - Validate that the analysis addresses the specified target audience
        - Ensure recommendations fit within the specified budget and duration
        - Confirm that the competitive landscape analysis meets the requirements
        - Verify that the business objectives are adequately addressed
        - Provide specific, measurable recommendations aligned with campaign goals
        
        Your report should demonstrate clear value and ROI for the specified budget and objectives.
        """
        
        # Generate the intelligent response
        from langchain.schema import SystemMessage, HumanMessage
        
        enhanced_system_prompt = f"""You are an expert marketing research analyst and strategic consultant with deep expertise in:
        - Market analysis and competitive intelligence
        - Brand performance optimization
        - Sales forecasting and revenue prediction
        - Campaign strategy and ROI optimization
        - Consumer behavior and market segmentation
        
        Your role is to provide intelligent, actionable insights that go beyond simple data aggregation. You synthesize complex information into clear, strategic recommendations that drive business results.
        
        Your goal: {self.agent.goal}
        Your expertise: {self.agent.backstory}"""
        
        try:
            system_message = SystemMessage(content=enhanced_system_prompt)
            human_message = HumanMessage(content=task_description)
            
            response = self.agent.llm.invoke([system_message, human_message])
            
            # Create enhanced final report with metadata
            final_report = {
                'final_summary': response.content,
                'timestamp': datetime.now().isoformat(),
                'mode': 'manual',
                'user_query': None,
                'configuration_validated': True,
                'agents_analyzed': len(all_results),
                'analysis_completeness': len(all_results) / max(len(state.get('selected_agents', [])), 1) * 100,
                'reflection_enabled': False
            }
            
            state = store_agent_result(state, 'report_summarizer', final_report)
            state['final_report'] = final_report
            
            logger.info(f"Standard report summarizer completed - Agents analyzed: {len(all_results)}")
            
        except Exception as e:
            logger.error(f"Error in standard processing: {e}")
            state = store_agent_error(state, 'report_summarizer', str(e))
        
        return state


def load_agent_configs(config_path: str) -> Dict[str, Dict[str, Any]]:
    """Load agent configurations from YAML file."""
    try:
        import yaml
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Failed to load agent configs from {config_path}: {e}")
        return {}


def enhanced_reflective_report_summarizer_node(state: MarketingResearchState) -> MarketingResearchState:
    """Enhanced Report Summarizer node with reflection and retry capabilities."""
    
    agent_role = 'report_summarizer'
    
    try:
        # Load agent config
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'agents.yaml')
        agent_configs = load_agent_configs(config_path)
        agent_config = agent_configs.get(agent_role, {})
        
        # Create reflective summarizer
        reflective_summarizer = ReflectiveReportSummarizer(agent_config)
        
        # Process with reflection
        state = reflective_summarizer.process_with_reflection(state)
        
        logger.info(f"Enhanced reflective {agent_role} completed")
        
    except Exception as e:
        logger.error(f"Error in enhanced reflective {agent_role} node: {e}")
        state = store_agent_error(state, agent_role, str(e))
    
    return state