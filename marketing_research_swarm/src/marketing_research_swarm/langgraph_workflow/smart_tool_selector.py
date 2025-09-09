"""
Smart Tool Selector for LangGraph Marketing Research Workflow

This module implements a hybrid approach that combines static tool assignment
with dynamic, query-aware tool selection for optimal performance.
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ToolRelevanceScore:
    """Represents a tool's relevance score for a given query."""
    tool_name: str
    score: float
    matched_keywords: List[str]
    tier: int
    execution_priority: int


class SmartToolSelector:
    """
    Smart tool selector that implements tiered tool execution strategy
    with query-aware selection and contextual filtering.
    """
    
    def __init__(self):
        """Initialize the smart tool selector."""
        self.tool_relevance_mapping = self._initialize_tool_relevance_mapping()
        self.agent_essential_tools = self._initialize_essential_tools()
        self.tool_dependencies = self._initialize_tool_dependencies()
        self.execution_stats = {}  # Track tool execution success rates
        
    def _initialize_tool_relevance_mapping(self) -> Dict[str, Dict[str, Any]]:
        """Initialize tool relevance mapping with keywords and metadata."""
        return {
            "profitability_analysis": {
                "keywords": ["profit", "margin", "revenue", "cost", "financial", "profitability", "earnings", "income"],
                "tier": 1,  # Essential for data analysis
                "weight": 1.0,
                "execution_time": "medium",
                "data_requirements": ["sales_data", "cost_data"]
            },
            "beverage_market_analysis": {
                "keywords": ["market", "segment", "brand", "category", "beverage", "industry", "competitive", "landscape"],
                "tier": 1,  # Essential for market research
                "weight": 1.0,
                "execution_time": "medium",
                "data_requirements": ["market_data", "brand_data"]
            },
            "forecast_sales": {
                "keywords": ["forecast", "predict", "future", "trend", "projection", "prediction", "outlook", "estimate"],
                "tier": 2,  # Contextual
                "weight": 0.9,
                "execution_time": "high",
                "data_requirements": ["historical_data", "trend_data"]
            },
            "time_series_analysis": {
                "keywords": ["trend", "seasonal", "time", "pattern", "series", "temporal", "historical", "evolution"],
                "tier": 1,  # Contextual
                "weight": 1.0,
                "execution_time": "medium",
                "data_requirements": ["time_series_data"]
            },
            "cross_sectional_analysis": {
                "keywords": ["compare", "segment", "performance", "cross", "comparison", "benchmark", "relative"],
                "tier": 2,  # Contextual
                "weight": 0.8,
                "execution_time": "medium",
                "data_requirements": ["segment_data"]
            },
            "analyze_kpis": {
                "keywords": ["kpi", "metric", "performance", "indicator", "measurement", "tracking", "dashboard"],
                "tier": 2,  # Contextual
                "weight": 0.7,
                "execution_time": "low",
                "data_requirements": ["performance_data"]
            },
            "calculate_roi": {
                "keywords": ["roi", "return", "investment", "efficiency", "effectiveness", "value", "benefit"],
                "tier": 3,  # Supplementary
                "weight": 0.6,
                "execution_time": "low",
                "data_requirements": ["investment_data", "return_data"]
            },
            "plan_budget": {
                "keywords": ["budget", "allocation", "spend", "investment", "planning", "resource", "distribution"],
                "tier": 3,  # Supplementary
                "weight": 0.6,
                "execution_time": "low",
                "data_requirements": ["budget_data"]
            },
            "calculate_market_share": {
                "keywords": ["share", "market", "competitive", "position", "dominance", "penetration", "coverage"],
                "tier": 3,  # Supplementary
                "weight": 0.5,
                "execution_time": "medium",
                "data_requirements": ["market_data", "competitor_data"]
            },
            "analyze_brand_performance": {
                "keywords": ["brand", "performance", "awareness", "sentiment", "positioning", "equity", "strength"],
                "tier": 2,  # Contextual
                "weight": 0.7,
                "execution_time": "medium",
                "data_requirements": ["brand_data", "performance_data"]
            }
        }
    
    def _initialize_essential_tools(self) -> Dict[str, List[str]]:
        """Initialize essential tools for each agent role."""
        return {
            "data_analyst": ["profitability_analysis", "analyze_kpis", "time_series_analysis", "customer_churn_analysis"],
            "market_research_analyst": ["beverage_market_analysis"],
            "forecasting_specialist": ["forecast_sales"],
            "competitive_analyst": ["beverage_market_analysis", "analyze_brand_performance", "customer_churn_analysis"],
            "content_strategist": ["analyze_kpis"],
            "creative_copywriter": [],  # No essential analytical tools
            "brand_performance_specialist": ["analyze_brand_performance", "customer_churn_analysis"],
            "campaign_optimizer": ["calculate_roi", "plan_budget"]
        }
    
    def _initialize_tool_dependencies(self) -> Dict[str, List[str]]:
        """Initialize tool dependencies for optimal execution order."""
        return {
            "calculate_roi": ["profitability_analysis"],  # ROI needs profit data
            "plan_budget": ["profitability_analysis", "forecast_sales"],  # Budget needs profit and forecast
            "calculate_market_share": ["beverage_market_analysis"],  # Market share needs market data
            "analyze_brand_performance": ["beverage_market_analysis"],  # Brand performance needs market context
        }
    
    def extract_query_keywords(self, query_text: str, context: Dict[str, Any] = None) -> List[str]:
        """Extract relevant keywords from query text and context."""
        keywords = []
        
        # Clean and normalize query text
        cleaned_query = re.sub(r'[^\w\s]', ' ', query_text.lower())
        words = cleaned_query.split()
        
        # Extract meaningful words (filter out common words)
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those'
        }
        
        keywords.extend([word for word in words if len(word) > 2 and word not in stop_words])
        
        # Extract keywords from context
        if context:
            # Add context-specific keywords
            if context.get('analysis_focus'):
                focus_words = re.sub(r'[^\w\s]', ' ', context['analysis_focus'].lower()).split()
                keywords.extend([word for word in focus_words if len(word) > 2 and word not in stop_words])
            
            # Add brand and market segment keywords
            if context.get('brands'):
                keywords.extend([brand.lower() for brand in context['brands']])
            
            if context.get('market_segments'):
                keywords.extend([segment.lower().replace(' ', '_') for segment in context['market_segments']])
            
            if context.get('product_categories'):
                keywords.extend([cat.lower().replace(' ', '_') for cat in context['product_categories']])
        
        return list(set(keywords))  # Remove duplicates
    
    def calculate_tool_relevance_scores(
        self, 
        query_keywords: List[str], 
        available_tools: List[str],
        context: Dict[str, Any] = None
    ) -> List[ToolRelevanceScore]:
        """Calculate relevance scores for tools based on query keywords."""
        
        scores = []
        
        for tool_name in available_tools:
            if tool_name not in self.tool_relevance_mapping:
                continue
                
            tool_info = self.tool_relevance_mapping[tool_name]
            tool_keywords = tool_info["keywords"]
            
            # Calculate base score from keyword matching
            matched_keywords = list(set(query_keywords) & set(tool_keywords))
            base_score = len(matched_keywords) / len(tool_keywords) if tool_keywords else 0
            
            # Apply weight and tier adjustments
            weighted_score = base_score * tool_info["weight"]
            
            # Boost score for tier 1 tools
            if tool_info["tier"] == 1:
                weighted_score *= 1.2
            
            # Context-based score adjustments
            if context:
                # Boost forecasting tools if duration is mentioned
                if tool_name == "forecast_sales" and context.get('duration'):
                    weighted_score *= 1.3
                
                # Boost brand analysis if specific brands are mentioned
                if "brand" in tool_name and context.get('brands'):
                    weighted_score *= 1.2
                
                # Boost market analysis if market segments are specified
                if "market" in tool_name and context.get('market_segments'):
                    weighted_score *= 1.2
            
            # Only include tools with meaningful scores
            if weighted_score > 0.1 or tool_info["tier"] == 1:
                scores.append(ToolRelevanceScore(
                    tool_name=tool_name,
                    score=weighted_score,
                    matched_keywords=matched_keywords,
                    tier=tool_info["tier"],
                    execution_priority=self._calculate_execution_priority(tool_name, tool_info)
                ))
        
        # Sort by score (descending) and then by execution priority
        scores.sort(key=lambda x: (x.score, -x.execution_priority), reverse=True)
        return scores
    
    def _calculate_execution_priority(self, tool_name: str, tool_info: Dict[str, Any]) -> int:
        """Calculate execution priority based on dependencies and execution time."""
        priority = 100  # Base priority
        
        # Higher priority for tier 1 tools
        if tool_info["tier"] == 1:
            priority += 50
        elif tool_info["tier"] == 2:
            priority += 25
        
        # Adjust for execution time (faster tools get higher priority)
        execution_time = tool_info.get("execution_time", "medium")
        if execution_time == "low":
            priority += 20
        elif execution_time == "high":
            priority -= 20
        
        # Adjust for dependencies (tools with no dependencies get higher priority)
        if tool_name not in self.tool_dependencies:
            priority += 10
        
        return priority
    
    def select_tools_for_execution(
        self,
        agent_role: str,
        query_text: str,
        available_tools: List[str],
        context: Dict[str, Any] = None,
        max_tools: int = 4
    ) -> Dict[str, List[str]]:
        """
        Select tools for execution using the tiered approach.
        
        Returns:
            Dict with keys: 'essential', 'contextual', 'supplementary'
        """
        
        # Extract query keywords
        query_keywords = self.extract_query_keywords(query_text, context)
        
        # Get essential tools for this agent
        essential_tools = self.agent_essential_tools.get(agent_role, [])
        essential_tools = [tool for tool in essential_tools if tool in available_tools]
        
        # Calculate relevance scores for all available tools
        relevance_scores = self.calculate_tool_relevance_scores(
            query_keywords, available_tools, context
        )
        
        # Boost scores for tools explicitly mentioned in the query text
        query_lower = query_text.lower()
        boosted_scores = []
        for score in relevance_scores:
            # Boost score if tool name is explicitly mentioned
            tool_name_parts = score.tool_name.lower().replace('_', ' ').split()
            if any(part in query_lower for part in tool_name_parts):
                score.score *= 1.5  # 50% boost for explicit mentions
            boosted_scores.append(score)
        
        # Sort boosted scores
        boosted_scores.sort(key=lambda x: (x.score, -x.execution_priority), reverse=True)
        relevance_scores = boosted_scores
        
        # Separate tools by tier
        tier_1_tools = [score.tool_name for score in relevance_scores if score.tier == 1]
        tier_2_tools = [score.tool_name for score in relevance_scores if score.tier == 2]
        tier_3_tools = [score.tool_name for score in relevance_scores if score.tier == 3]
        
        # Select contextual tools (tier 2) based on relevance
        contextual_tools = []
        for score in relevance_scores:
            if score.tier == 2 and score.score > 0.3:  # Threshold for contextual relevance
                contextual_tools.append(score.tool_name)
                if len(contextual_tools) >= 2:  # Limit contextual tools
                    break
        
        # Select supplementary tools (tier 3) based on high relevance
        supplementary_tools = []
        for score in relevance_scores:
            if score.tier == 3 and score.score > 0.5:  # Higher threshold for supplementary
                supplementary_tools.append(score.tool_name)
                if len(supplementary_tools) >= 1:  # Limit supplementary tools
                    break
        
        # Ensure time series analysis is included if explicitly mentioned
        if "time series" in query_lower or "trend identification" in query_lower:
            time_series_tool = "time_series_analysis"
            if time_series_tool in available_tools and time_series_tool not in essential_tools and time_series_tool not in contextual_tools:
                # Add to contextual tools if there's space
                if len(contextual_tools) < 2:
                    contextual_tools.append(time_series_tool)
                # Otherwise replace the lowest scoring contextual tool
                elif relevance_scores:
                    # Find the time_series_analysis score
                    ts_score = next((s for s in relevance_scores if s.tool_name == time_series_tool), None)
                    if ts_score:
                        # Replace lowest scoring contextual tool if time_series_analysis has higher score
                        contextual_tools[-1] = time_series_tool
        
        # Ensure we don't exceed max_tools
        total_selected = len(essential_tools) + len(contextual_tools) + len(supplementary_tools)
        if total_selected > max_tools:
            # Prioritize essential, then contextual, then supplementary
            if len(essential_tools) + len(contextual_tools) > max_tools:
                contextual_tools = contextual_tools[:max(0, max_tools - len(essential_tools))]
                supplementary_tools = []
            else:
                remaining = max_tools - len(essential_tools) - len(contextual_tools)
                supplementary_tools = supplementary_tools[:remaining]
        
        selection_result = {
            'essential': essential_tools,
            'contextual': contextual_tools,
            'supplementary': supplementary_tools
        }
        
        # Log selection decision
        logger.info(f"Tool selection for {agent_role}:")
        logger.info(f"  Query keywords: {query_keywords[:5]}...")  # Show first 5 keywords
        logger.info(f"  Essential: {essential_tools}")
        logger.info(f"  Contextual: {contextual_tools}")
        logger.info(f"  Supplementary: {supplementary_tools}")
        
        return selection_result
    
    def get_execution_order(self, selected_tools: List[str]) -> List[str]:
        """Determine optimal execution order based on dependencies."""
        
        execution_order = []
        remaining_tools = set(selected_tools)
        
        # Execute tools in dependency order
        while remaining_tools:
            # Find tools with no unmet dependencies
            ready_tools = []
            for tool in remaining_tools:
                dependencies = self.tool_dependencies.get(tool, [])
                unmet_deps = [dep for dep in dependencies if dep in remaining_tools]
                if not unmet_deps:
                    ready_tools.append(tool)
            
            if not ready_tools:
                # Circular dependency or error - add remaining tools
                ready_tools = list(remaining_tools)
            
            # Sort ready tools by priority
            ready_tools.sort(key=lambda x: self._get_tool_priority(x), reverse=True)
            
            # Add to execution order and remove from remaining
            for tool in ready_tools:
                execution_order.append(tool)
                remaining_tools.remove(tool)
        
        return execution_order
    
    def _get_tool_priority(self, tool_name: str) -> int:
        """Get tool priority for execution ordering."""
        if tool_name in self.tool_relevance_mapping:
            tool_info = self.tool_relevance_mapping[tool_name]
            return self._calculate_execution_priority(tool_name, tool_info)
        return 0
    
    def should_execute_supplementary_tools(
        self,
        essential_results: Dict[str, Any],
        contextual_results: Dict[str, Any],
        query_analysis: str,
        context: Dict[str, Any] = None
    ) -> bool:
        """
        Determine if supplementary tools should be executed based on
        initial results and query requirements.
        """
        
        # Check if essential results are insufficient
        if not essential_results and not contextual_results:
            return True  # Need more data
        
        # Check for specific query patterns that require supplementary analysis
        query_lower = query_analysis.lower()
        
        # ROI-related queries
        if any(keyword in query_lower for keyword in ['roi', 'return', 'investment', 'efficiency']):
            return 'calculate_roi' not in essential_results and 'calculate_roi' not in contextual_results
        
        # Budget-related queries
        if any(keyword in query_lower for keyword in ['budget', 'allocation', 'spend', 'planning']):
            return 'plan_budget' not in essential_results and 'plan_budget' not in contextual_results
        
        # Market share queries
        if any(keyword in query_lower for keyword in ['share', 'market share', 'competitive position']):
            return 'calculate_market_share' not in essential_results and 'calculate_market_share' not in contextual_results
        
        # Check if results are sparse (low data quality indicator)
        total_results = len(essential_results) + len(contextual_results)
        if total_results < 2:
            return True
        
        return False
    
    def update_execution_stats(self, tool_name: str, success: bool, execution_time: float):
        """Update tool execution statistics for future optimization."""
        
        if tool_name not in self.execution_stats:
            self.execution_stats[tool_name] = {
                'total_executions': 0,
                'successful_executions': 0,
                'total_time': 0.0,
                'average_time': 0.0,
                'success_rate': 0.0
            }
        
        stats = self.execution_stats[tool_name]
        stats['total_executions'] += 1
        stats['total_time'] += execution_time
        
        if success:
            stats['successful_executions'] += 1
        
        stats['average_time'] = stats['total_time'] / stats['total_executions']
        stats['success_rate'] = stats['successful_executions'] / stats['total_executions']
        
        logger.debug(f"Updated stats for {tool_name}: {stats}")
    
    def get_tool_performance_summary(self) -> Dict[str, Any]:
        """Get summary of tool performance statistics."""
        return {
            'execution_stats': self.execution_stats,
            'total_tools_tracked': len(self.execution_stats),
            'generated_at': datetime.now().isoformat()
        }