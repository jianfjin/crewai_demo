"""
Optimization Manager for Token Usage Reduction
Manages switching between original and optimized crews with performance tracking.
"""

from typing import Dict, Any, Optional
from datetime import datetime
import json

class OptimizationManager:
    """Manages optimization strategies and performance tracking."""
    
    def __init__(self):
        self.optimization_history = []
        self.current_mode = "standard"  # or "optimized"
    
    def get_crew_instance(self, mode: str = "optimized", **kwargs):
        """Get crew instance based on optimization mode."""
        
        if mode == "blackboard":
            from .blackboard.blackboard_crew import create_blackboard_crew
            # Create blackboard crew with config paths
            agents_config = kwargs.get('agents_config_path', 'src/marketing_research_swarm/config/agents.yaml')
            tasks_config = kwargs.get('tasks_config_path', 'src/marketing_research_swarm/config/tasks.yaml')
            return create_blackboard_crew(agents_config, tasks_config)
        elif mode == "optimized":
            from .crew_optimized import OptimizedMarketingResearchCrew
            return OptimizedMarketingResearchCrew(**kwargs)
        elif mode == "simple_optimized":
            from .crew_simple_optimized import SimpleOptimizedCrew
            return SimpleOptimizedCrew(**kwargs)
        else:
            from .crew_with_tracking import MarketingResearchCrewWithTracking
            return MarketingResearchCrewWithTracking(**kwargs)
    
    def run_analysis_with_optimization(self, inputs: Dict[str, Any], 
                                     optimization_level: str = "full",
                                     custom_tasks_config_path: str = None) -> Dict[str, Any]:
        """
        Run analysis with specified optimization level.
        
        Args:
            inputs: Analysis inputs
            optimization_level: "none", "partial", "full", "blackboard"
        """
        
        start_time = datetime.now()
        
        # Configure optimization based on level
        if optimization_level == "full":
            # Use simple optimized crew to avoid iteration issues
            try:
                crew = self.get_crew_instance("simple_optimized")
                optimization_config = {
                    "data_reduction": True,
                    "agent_compression": True,
                    "tool_caching": False,  # Disabled in simple version
                    "output_optimization": True,
                    "approach": "simple_optimized"
                }
            except Exception as e:
                print(f"Warning: Using fallback crew due to: {e}")
                crew = self.get_crew_instance("standard", 
                    agents_config_path='src/marketing_research_swarm/config/agents.yaml',
                    tasks_config_path='src/marketing_research_swarm/config/tasks.yaml'
                )
                optimization_config = {"fallback": True}
        elif optimization_level == "partial":
            crew = self.get_crew_instance("simple_optimized")
            optimization_config = {
                "data_reduction": True,
                "agent_compression": True,
                "tool_caching": False,
                "output_optimization": False,
                "approach": "simple_optimized"
            }
        elif optimization_level == "blackboard":
            # Use integrated blackboard system for maximum efficiency
            try:
                # Use custom tasks config if provided
                blackboard_kwargs = {}
                if custom_tasks_config_path:
                    blackboard_kwargs['tasks_config_path'] = custom_tasks_config_path
                crew = self.get_crew_instance("blackboard", **blackboard_kwargs)
                optimization_config = {
                    "unified_coordination": True,
                    "shared_state_management": True,
                    "integrated_caching": True,
                    "memory_optimization": True,
                    "context_optimization": True,
                    "token_efficiency": "maximum",
                    "approach": "integrated_blackboard"
                }
            except Exception as e:
                print(f"Warning: Blackboard system unavailable, falling back to optimized crew: {e}")
                crew = self.get_crew_instance("simple_optimized")
                optimization_config = {"fallback_from_blackboard": True, "approach": "simple_optimized"}
        else:  # "none"
            crew = self.get_crew_instance("standard",
                agents_config_path='src/marketing_research_swarm/config/agents.yaml',
                tasks_config_path='src/marketing_research_swarm/config/tasks.yaml'
            )
            optimization_config = {
                "data_reduction": False,
                "agent_compression": False,
                "tool_caching": False,
                "output_optimization": False,
                "approach": "standard"
            }
        
        # Execute analysis
        try:
            result = crew.kickoff(inputs)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Extract metrics with enhanced detection
            usage_metrics = self._extract_metrics_from_result(result, crew)
            
            # Debug: Print extracted metrics
            print(f"Optimization Manager - Extracted metrics: {usage_metrics}")
            
            # Record optimization performance
            performance_record = {
                "timestamp": start_time.isoformat(),
                "optimization_level": optimization_level,
                "optimization_config": optimization_config,
                "duration_seconds": duration,
                "usage_metrics": usage_metrics,
                "inputs_summary": self._summarize_inputs(inputs)
            }
            
            self.optimization_history.append(performance_record)
            
            return {
                "result": result,
                "performance": performance_record,
                "optimization_applied": optimization_config
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "optimization_level": optimization_level,
                "duration": (datetime.now() - start_time).total_seconds()
            }
    
    def _extract_metrics_from_result(self, result, crew) -> Dict[str, Any]:
        """Extract usage metrics from result or crew with enhanced detection."""
        
        # Handle blackboard crew results
        if isinstance(result, dict) and 'workflow_summary' in result:
            try:
                workflow_summary = result['workflow_summary']
                managers_status = workflow_summary.get('managers_status', {})
                
                # Extract token metrics from blackboard system
                token_tracker_stats = managers_status.get('token_tracker', {}).get('stats', {})
                if token_tracker_stats:
                    metrics = {
                        'total_tokens': token_tracker_stats.get('total_tokens', 0),
                        'input_tokens': token_tracker_stats.get('input_tokens', 0),
                        'output_tokens': token_tracker_stats.get('output_tokens', 0),
                        'total_cost': token_tracker_stats.get('total_cost', 0.0),
                        'blackboard_efficiency': True,
                        'managers_active': len([m for m, s in managers_status.items() 
                                              if isinstance(s, dict) and s.get('active', False)])
                    }
                    if metrics['total_tokens'] > 0:
                        print(f"Got metrics from blackboard system: {metrics}")
                        return metrics
            except Exception as e:
                print(f"Warning: Could not extract blackboard metrics: {e}")
        
        # Try to extract from crew's own method first (for simple optimized crew)
        if hasattr(crew, '_extract_usage_metrics'):
            try:
                metrics = crew._extract_usage_metrics(crew)
                if metrics and metrics.get('total_tokens', 0) > 0:
                    print(f"Got metrics from crew method: {metrics}")
                    return metrics
            except Exception as e:
                print(f"Warning: Could not use crew's extract method: {e}")
        
        # Try to extract from crew usage_metrics
        if hasattr(crew, 'usage_metrics') and crew.usage_metrics:
            usage = crew.usage_metrics
            metrics = {
                'total_tokens': getattr(usage, 'total_tokens', 0),
                'input_tokens': getattr(usage, 'prompt_tokens', 0),
                'output_tokens': getattr(usage, 'completion_tokens', 0),
                'total_cost': getattr(usage, 'total_cost', 0.0),
                'successful_requests': getattr(usage, 'successful_requests', 0)
            }
            if metrics['total_tokens'] > 0:
                print(f"Got metrics from crew.usage_metrics: {metrics}")
                return metrics
        
        # Try to extract from result string
        if isinstance(result, str):
            # Look for token usage in the formatted result
            if "Total Tokens:" in result:
                try:
                    lines = result.split('\n')
                    metrics = {}
                    for line in lines:
                        if "Total Tokens:" in line:
                            token_str = line.split(':')[1].strip().replace(',', '')
                            metrics['total_tokens'] = int(token_str)
                        elif "Cost:" in line and "$" in line:
                            cost_str = line.split('$')[1].strip()
                            metrics['total_cost'] = float(cost_str)
                    if metrics.get('total_tokens', 0) > 0:
                        print(f"Got metrics from result parsing: {metrics}")
                        return metrics
                except Exception as e:
                    print(f"Warning: Could not parse metrics from result: {e}")
        
        # Try to get actual token usage from global tracker
        try:
            from .utils.token_tracker import get_token_tracker
            tracker = get_token_tracker()
            if tracker and hasattr(tracker, 'crew_usage') and tracker.crew_usage:
                actual_usage = tracker.crew_usage.total_token_usage
                print(f"Got actual token usage: {actual_usage.total_tokens}")
                return {
                    'total_tokens': actual_usage.total_tokens,
                    'input_tokens': actual_usage.prompt_tokens,
                    'output_tokens': actual_usage.completion_tokens,
                    'total_cost': actual_usage.total_tokens * 0.0000025,  # Rough estimate
                    'successful_requests': 1,
                    'estimated': False,
                    'source': 'actual_tracking'
                }
        except Exception as e:
            print(f"Could not get actual token usage: {e}")
        
        # Final fallback - provide estimated metrics based on optimization level
        print("Using fallback estimated metrics")
        return {
            'total_tokens': 8000,  # Conservative estimate for optimized analysis
            'input_tokens': 5600,
            'output_tokens': 2400,
            'total_cost': 0.0025,
            'successful_requests': 1,
            'estimated': True,
            'source': 'fallback_estimate'
        }
    
    def compare_optimization_performance(self) -> Dict[str, Any]:
        """Compare performance between different optimization levels."""
        
        if len(self.optimization_history) < 2:
            return {"error": "Need at least 2 analysis runs for comparison"}
        
        # Group by optimization level
        performance_by_level = {}
        for record in self.optimization_history:
            level = record["optimization_level"]
            if level not in performance_by_level:
                performance_by_level[level] = []
            performance_by_level[level].append(record)
        
        # Calculate averages
        comparison = {}
        for level, records in performance_by_level.items():
            avg_tokens = sum(r["usage_metrics"].get("total_tokens", 0) for r in records) / len(records)
            avg_cost = sum(r["usage_metrics"].get("total_cost", 0) for r in records) / len(records)
            avg_duration = sum(r["duration_seconds"] for r in records) / len(records)
            
            comparison[level] = {
                "avg_tokens": avg_tokens,
                "avg_cost": avg_cost,
                "avg_duration": avg_duration,
                "runs_count": len(records)
            }
        
        # Calculate improvements
        if "none" in comparison and "full" in comparison:
            baseline = comparison["none"]
            optimized = comparison["full"]
            
            token_reduction = ((baseline["avg_tokens"] - optimized["avg_tokens"]) / baseline["avg_tokens"]) * 100
            cost_reduction = ((baseline["avg_cost"] - optimized["avg_cost"]) / baseline["avg_cost"]) * 100
            time_reduction = ((baseline["avg_duration"] - optimized["avg_duration"]) / baseline["avg_duration"]) * 100
            
            comparison["improvements"] = {
                "token_reduction_percent": token_reduction,
                "cost_reduction_percent": cost_reduction,
                "time_reduction_percent": time_reduction
            }
        
        return comparison
    
    def _summarize_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of inputs for comparison."""
        return {
            "target_audience": inputs.get("target_audience", ""),
            "budget": inputs.get("budget", 0),
            "analysis_focus": inputs.get("analysis_focus", ""),
            "has_data_file": bool(inputs.get("data_file_path"))
        }

# Global optimization manager instance
optimization_manager = OptimizationManager()