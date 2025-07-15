"""
Context-Aware Tool Wrapper System
Intercepts tool outputs and stores them by reference instead of dumping to context.
Enhanced with smart caching to prevent redundant tool executions.
"""

from typing import Any, Dict, Optional
import json
import time
from crewai.tools import BaseTool
from pydantic import Field
# Import optimization_manager dynamically to avoid circular imports
from ..blackboard.integrated_blackboard import get_integrated_blackboard
from ..blackboard.smart_cache import get_smart_cache

class ContextAwareToolWrapper(BaseTool):
    """Wrapper that stores tool outputs by reference instead of returning raw data."""
    
    name: str = Field(default="context_aware_tool")
    description: str = Field(default="Context-aware tool wrapper")
    
    def __init__(self, original_tool, tool_name: str, **kwargs):
        # Initialize BaseTool with proper attributes
        super().__init__(
            name=tool_name,
            description=f"Context-aware version of {tool_name} with smart caching and reference storage",
            **kwargs
        )
        # Set attributes after initialization to avoid field validation
        object.__setattr__(self, 'original_tool', original_tool)
        object.__setattr__(self, 'tool_name', tool_name)
        object.__setattr__(self, 'blackboard', get_integrated_blackboard())
        object.__setattr__(self, 'smart_cache', get_smart_cache())
        object.__setattr__(self, '_optimization_manager', None)
        object.__setattr__(self, '_current_agent', None)
        
    def _run(self, *args, **kwargs):
        """Execute tool with smart caching and store output by reference."""
        try:
            # Prepare parameters for caching
            cache_params = {
                'args': args,
                'kwargs': kwargs
            }
            
            # Get current agent name for cache tracking
            agent_name = getattr(self, '_current_agent', 'unknown_agent')
            
            # Check cache first
            start_time = time.time()
            cached_result = self.smart_cache.get_cached_result(self.tool_name, cache_params)
            
            if cached_result is not None:
                # Return cached result with cache indicator
                if isinstance(cached_result, dict) and 'reference' in cached_result:
                    cached_result['cached'] = True
                    cached_result['cache_hit'] = True
                return cached_result
            
            # Execute the original tool if not cached
            if hasattr(self.original_tool, '_run'):
                raw_output = self.original_tool._run(*args, **kwargs)
            elif hasattr(self.original_tool, '__call__'):
                raw_output = self.original_tool(*args, **kwargs)
            else:
                # Fallback for other tool types
                raw_output = self.original_tool.run(*args, **kwargs)
            
            execution_time = time.time() - start_time
            
            # Check if output is large enough to warrant reference storage
            output_size = len(str(raw_output))
            
            result_to_cache = None
            
            if output_size > 500:  # Store by reference if output > 500 characters
                # Get optimization manager dynamically
                if self._optimization_manager is None:
                    from ..optimization_manager import optimization_manager
                    self._optimization_manager = optimization_manager
                
                # Store in optimization manager
                ref_key = self._optimization_manager.store_tool_output(
                    tool_name=self.tool_name,
                    output=raw_output,
                    context_key=f"{self.tool_name}_latest"
                )
                
                # Return reference with summary
                if isinstance(raw_output, dict):
                    summary = self._create_dict_summary(raw_output)
                elif isinstance(raw_output, str):
                    summary = raw_output[:200] + "..." if len(raw_output) > 200 else raw_output
                else:
                    summary = f"{type(raw_output).__name__} with {output_size} characters"
                
                result_to_cache = {
                    "reference": ref_key,
                    "summary": summary,
                    "tool_name": self.tool_name,
                    "output_size": output_size,
                    "cached": False,
                    "execution_time": execution_time,
                    "note": f"Full output stored by reference. Use reference key to retrieve: {ref_key}"
                }
            else:
                # Return small outputs directly
                result_to_cache = raw_output
            
            # Store result in cache for future use
            self.smart_cache.store_result(
                tool_name=self.tool_name,
                parameters=cache_params,
                result=result_to_cache,
                agent_name=agent_name,
                execution_time=execution_time
            )
            
            return result_to_cache
                
        except Exception as e:
            error_msg = f"Error executing {self.tool_name}: {str(e)}"
            print(f"ERROR: {error_msg}")
            return error_msg
    
    def set_current_agent(self, agent_name: str):
        """Set the current agent name for cache tracking"""
        object.__setattr__(self, '_current_agent', agent_name)
    
    def _create_dict_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of dictionary data."""
        summary = {}
        
        for key, value in data.items():
            if isinstance(value, (dict, list)) and len(str(value)) > 100:
                if isinstance(value, dict):
                    summary[key] = f"<Dict with {len(value)} keys>"
                else:
                    summary[key] = f"<List with {len(value)} items>"
            elif isinstance(value, str) and len(value) > 100:
                summary[key] = value[:100] + "..."
            else:
                summary[key] = value
        
        return summary

def wrap_tool_for_context_isolation(tool, tool_name: str):
    """Wrap a tool to enable context isolation and smart caching."""
    return ContextAwareToolWrapper(tool, tool_name)

def set_agent_context_for_tools(tools_dict: Dict[str, ContextAwareToolWrapper], agent_name: str):
    """Set the current agent context for all tools"""
    for tool in tools_dict.values():
        if hasattr(tool, 'set_current_agent'):
            tool.set_current_agent(agent_name)

class ReferenceRetrieverTool(BaseTool):
    """Tool that allows agents to retrieve data by reference key."""
    
    name: str = Field(default="retrieve_by_reference")
    description: str = Field(default="Retrieve stored analysis results by reference key")
    
    def __init__(self, **kwargs):
        super().__init__(
            name="retrieve_by_reference",
            description="Retrieve stored analysis results by reference key",
            **kwargs
        )
    
    def _run(self, reference_key: str) -> Any:
        """Retrieve data by reference key."""
        try:
            from ..optimization_manager import optimization_manager
            return optimization_manager.retrieve_by_reference(reference_key)
        except Exception as e:
            return f"Error retrieving reference {reference_key}: {str(e)}"

# Global reference retriever
reference_retriever = ReferenceRetrieverTool()

# Context-aware versions of the main tools
def create_context_aware_tools():
    """Create context-aware versions of all tools with smart caching."""
    from .advanced_tools import (
        beverage_market_analysis, time_series_analysis, cross_sectional_analysis,
        profitability_analysis, calculate_roi, analyze_kpis, forecast_sales,
        plan_budget, analyze_brand_performance, calculate_market_share
    )
    
    return {
        "beverage_market_analysis": wrap_tool_for_context_isolation(
            beverage_market_analysis, "beverage_market_analysis"
        ),
        "time_series_analysis": wrap_tool_for_context_isolation(
            time_series_analysis, "time_series_analysis"
        ),
        "cross_sectional_analysis": wrap_tool_for_context_isolation(
            cross_sectional_analysis, "cross_sectional_analysis"
        ),
        "profitability_analysis": wrap_tool_for_context_isolation(
            profitability_analysis, "profitability_analysis"
        ),
        "calculate_roi": wrap_tool_for_context_isolation(
            calculate_roi, "calculate_roi"
        ),
        "analyze_kpis": wrap_tool_for_context_isolation(
            analyze_kpis, "analyze_kpis"
        ),
        "forecast_sales": wrap_tool_for_context_isolation(
            forecast_sales, "forecast_sales"
        ),
        "plan_budget": wrap_tool_for_context_isolation(
            plan_budget, "plan_budget"
        ),
        "analyze_brand_performance": wrap_tool_for_context_isolation(
            analyze_brand_performance, "analyze_brand_performance"
        ),
        "calculate_market_share": wrap_tool_for_context_isolation(
            calculate_market_share, "calculate_market_share"
        )
    }