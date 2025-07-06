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
        
        if mode == "optimized":
            from .crew_optimized import OptimizedMarketingResearchCrew
            return OptimizedMarketingResearchCrew(**kwargs)
        else:
            from .crew_with_tracking import MarketingResearchCrewWithTracking
            return MarketingResearchCrewWithTracking(**kwargs)
    
    def run_analysis_with_optimization(self, inputs: Dict[str, Any], 
                                     optimization_level: str = "full") -> Dict[str, Any]:
        """
        Run analysis with specified optimization level.
        
        Args:
            inputs: Analysis inputs
            optimization_level: "none", "partial", "full"
        """
        
        start_time = datetime.now()
        
        # Configure optimization based on level
        if optimization_level == "full":
            crew = self.get_crew_instance("optimized")
            optimization_config = {
                "data_reduction": True,
                "agent_compression": True,
                "tool_caching": True,
                "output_optimization": True
            }
        elif optimization_level == "partial":
            crew = self.get_crew_instance("optimized")
            optimization_config = {
                "data_reduction": True,
                "agent_compression": True,
                "tool_caching": False,
                "output_optimization": False
            }
        else:  # "none"
            crew = self.get_crew_instance("standard")
            optimization_config = {
                "data_reduction": False,
                "agent_compression": False,
                "tool_caching": False,
                "output_optimization": False
            }
        
        # Execute analysis
        try:
            result = crew.kickoff(inputs)
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Extract metrics
            usage_metrics = self._extract_metrics_from_result(result, crew)
            
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
    
    def get_optimization_recommendations(self, current_usage: Dict[str, Any]) -> list[str]:
        """Get optimization recommendations based on current usage patterns."""
        
        recommendations = []
        
        total_tokens = current_usage.get("total_tokens", 0)
        input_tokens = current_usage.get("input_tokens", 0)
        output_tokens = current_usage.get("output_tokens", 0)
        
        # Analyze token distribution
        if total_tokens > 0:
            input_ratio = input_tokens / total_tokens
            
            if input_ratio > 0.85:
                recommendations.append("🔥 HIGH PRIORITY: Input tokens >85% - implement data context reduction")
            
            if total_tokens > 50000:
                recommendations.append("💰 COST OPTIMIZATION: High token usage - enable full optimization mode")
            
            if output_tokens > input_tokens * 0.2:
                recommendations.append("📝 OUTPUT OPTIMIZATION: High output tokens - use structured outputs")
        
        # Check for caching opportunities
        if len(self.optimization_history) > 1:
            recent_runs = self.optimization_history[-3:]
            similar_inputs = sum(1 for run in recent_runs if self._are_inputs_similar(
                run["inputs_summary"], current_usage.get("inputs_summary", {})
            ))
            
            if similar_inputs > 1:
                recommendations.append("🔄 CACHING OPPORTUNITY: Similar inputs detected - enable tool caching")
        
        return recommendations
    
    def _extract_metrics_from_result(self, result, crew) -> Dict[str, Any]:
        """Extract usage metrics from result or crew."""
        
        # Try to extract from crew first
        if hasattr(crew, 'usage_metrics') and crew.usage_metrics:
            usage = crew.usage_metrics
            return {
                'total_tokens': getattr(usage, 'total_tokens', 0),
                'input_tokens': getattr(usage, 'prompt_tokens', 0),
                'output_tokens': getattr(usage, 'completion_tokens', 0),
                'total_cost': getattr(usage, 'total_cost', 0.0),
                'successful_requests': getattr(usage, 'successful_requests', 0)
            }
        
        # Try to extract from result string
        if isinstance(result, str) and "Total Tokens:" in result:
            try:
                lines = result.split('\n')
                metrics = {}
                for line in lines:
                    if "Total Tokens:" in line:
                        metrics['total_tokens'] = int(line.split(':')[1].strip().replace(',', ''))
                    elif "Input Tokens:" in line:
                        metrics['input_tokens'] = int(line.split(':')[1].strip().split()[0].replace(',', ''))
                    elif "Output Tokens:" in line:
                        metrics['output_tokens'] = int(line.split(':')[1].strip().split()[0].replace(',', ''))
                    elif "Total Cost:" in line:
                        metrics['total_cost'] = float(line.split('$')[1].strip())
                return metrics
            except:
                pass
        
        return {'total_tokens': 0, 'total_cost': 0.0}
    
    def _summarize_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of inputs for comparison."""
        return {
            "target_audience": inputs.get("target_audience", ""),
            "budget": inputs.get("budget", 0),
            "analysis_focus": inputs.get("analysis_focus", ""),
            "has_data_file": bool(inputs.get("data_file_path"))
        }
    
    def _are_inputs_similar(self, inputs1: Dict[str, Any], inputs2: Dict[str, Any]) -> bool:
        """Check if two input summaries are similar."""
        if not inputs1 or not inputs2:
            return False
        
        # Simple similarity check
        similar_fields = 0
        total_fields = 0
        
        for key in inputs1.keys():
            if key in inputs2:
                total_fields += 1
                if inputs1[key] == inputs2[key]:
                    similar_fields += 1
        
        return (similar_fields / max(total_fields, 1)) > 0.7
    
    def export_optimization_report(self) -> str:
        """Export detailed optimization performance report."""
        
        if not self.optimization_history:
            return "No optimization data available"
        
        comparison = self.compare_optimization_performance()
        
        report = f"""# 🚀 Token Optimization Performance Report

## 📊 Summary Statistics
- Total Analysis Runs: {len(self.optimization_history)}
- Optimization Levels Tested: {list(set(r['optimization_level'] for r in self.optimization_history))}
- Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 🎯 Performance Comparison
"""
        
        for level, stats in comparison.items():
            if level != "improvements":
                report += f"""
### {level.title()} Mode
- Average Tokens: {stats['avg_tokens']:,.0f}
- Average Cost: ${stats['avg_cost']:.4f}
- Average Duration: {stats['avg_duration']:.1f}s
- Runs: {stats['runs_count']}
"""
        
        if "improvements" in comparison:
            improvements = comparison["improvements"]
            report += f"""
## 🏆 Optimization Results
- **Token Reduction**: {improvements['token_reduction_percent']:.1f}%
- **Cost Reduction**: {improvements['cost_reduction_percent']:.1f}%
- **Time Reduction**: {improvements['time_reduction_percent']:.1f}%

## 💡 Status
{'✅ OPTIMIZATION SUCCESSFUL' if improvements['token_reduction_percent'] > 50 else '⚠️ OPTIMIZATION NEEDS IMPROVEMENT'}
"""
        
        return report

# Global optimization manager instance
optimization_manager = OptimizationManager()