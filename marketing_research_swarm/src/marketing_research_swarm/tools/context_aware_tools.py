"""
Context-Aware Tool Wrapper System
Intercepts tool outputs and stores them by reference instead of dumping to context.
"""

from typing import Any, Dict, Optional
import json
# Import optimization_manager dynamically to avoid circular imports
from ..blackboard.integrated_blackboard import get_integrated_blackboard

class ContextAwareToolWrapper:
    """Wrapper that stores tool outputs by reference instead of returning raw data."""
    
    def __init__(self, original_tool, tool_name: str):
        self.original_tool = original_tool
        self.tool_name = tool_name
        self.blackboard = get_integrated_blackboard()
        self._optimization_manager = None
        
    def __call__(self, *args, **kwargs):
        """Execute tool and store output by reference."""
        try:
            # Execute the original tool
            raw_output = self.original_tool(*args, **kwargs)
            
            # Check if output is large enough to warrant reference storage
            output_size = len(str(raw_output))
            
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
                
                return {
                    "reference": ref_key,
                    "summary": summary,
                    "tool_name": self.tool_name,
                    "output_size": output_size,
                    "note": f"Full output stored by reference. Use reference key to retrieve: {ref_key}"
                }
            else:
                # Return small outputs directly
                return raw_output
                
        except Exception as e:
            return f"Error executing {self.tool_name}: {str(e)}"
    
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
    """Wrap a tool to enable context isolation."""
    return ContextAwareToolWrapper(tool, tool_name)

# Context-aware versions of the main tools
def create_context_aware_tools():
    """Create context-aware versions of all tools."""
    from .advanced_tools import (
        beverage_market_analysis, time_series_analysis, cross_sectional_analysis,
        profitability_analysis, calculate_roi, analyze_kpis, forecast_sales,
        plan_budget, analyze_brand_performance, calculate_market_share
    )
    
    return {
        'beverage_market_analysis': wrap_tool_for_context_isolation(
            beverage_market_analysis, 'beverage_market_analysis'
        ),
        'time_series_analysis': wrap_tool_for_context_isolation(
            time_series_analysis, 'time_series_analysis'
        ),
        'cross_sectional_analysis': wrap_tool_for_context_isolation(
            cross_sectional_analysis, 'cross_sectional_analysis'
        ),
        'profitability_analysis': wrap_tool_for_context_isolation(
            profitability_analysis, 'profitability_analysis'
        ),
        'calculate_roi': wrap_tool_for_context_isolation(
            calculate_roi, 'calculate_roi'
        ),
        'analyze_kpis': wrap_tool_for_context_isolation(
            analyze_kpis, 'analyze_kpis'
        ),
        'forecast_sales': wrap_tool_for_context_isolation(
            forecast_sales, 'forecast_sales'
        ),
        'plan_budget': wrap_tool_for_context_isolation(
            plan_budget, 'plan_budget'
        ),
        'analyze_brand_performance': wrap_tool_for_context_isolation(
            analyze_brand_performance, 'analyze_brand_performance'
        ),
        'calculate_market_share': wrap_tool_for_context_isolation(
            calculate_market_share, 'calculate_market_share'
        )
    }

class ReferenceRetrieverTool:
    """Tool that allows agents to retrieve data by reference key."""
    
    def __init__(self):
        self.name = "retrieve_by_reference"
        self.description = "Retrieve stored analysis results by reference key"
    
    def __call__(self, reference_key: str) -> Any:
        """Retrieve data by reference key."""
        try:
            from ..optimization_manager import optimization_manager
            return optimization_manager.retrieve_by_reference(reference_key)
        except Exception as e:
            return f"Error retrieving reference {reference_key}: {str(e)}"

# Global reference retriever
reference_retriever = ReferenceRetrieverTool()