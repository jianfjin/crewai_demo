"""
Token Usage Tracking and Analysis Utilities
Provides comprehensive tracking of LLM token consumption across agents and tasks.
"""

import time
import tiktoken
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import json

@dataclass
class TokenUsage:
    """Represents token usage for a single operation."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    def __add__(self, other):
        """Add two TokenUsage objects together."""
        if not isinstance(other, TokenUsage):
            return self
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens
        )

@dataclass
class TaskTokenUsage:
    """Represents token usage for a complete task."""
    task_name: str
    agent_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    tool_calls: int = 0
    llm_calls: int = 0
    status: str = "running"
    error_message: Optional[str] = None
    
    def complete(self, token_usage: TokenUsage, status: str = "completed", error_message: Optional[str] = None):
        """Mark the task as completed and record final metrics."""
        self.end_time = datetime.now()
        self.duration_seconds = (self.end_time - self.start_time).total_seconds()
        self.token_usage = token_usage
        self.status = status
        self.error_message = error_message

@dataclass
class CrewTokenUsage:
    """Represents token usage for the entire crew execution."""
    crew_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_duration_seconds: float = 0.0
    total_token_usage: TokenUsage = field(default_factory=TokenUsage)
    task_usages: List[TaskTokenUsage] = field(default_factory=list)
    model_name: str = "unknown"
    
    def add_task_usage(self, task_usage: TaskTokenUsage):
        """Add a task's token usage to the crew total."""
        self.task_usages.append(task_usage)
        self.total_token_usage += task_usage.token_usage
    
    def complete(self):
        """Mark the crew execution as completed."""
        self.end_time = datetime.now()
        self.total_duration_seconds = (self.end_time - self.start_time).total_seconds()

class TokenTracker:
    """Tracks token usage across crew execution."""
    
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model_name = model_name
        self.crew_usage: Optional[CrewTokenUsage] = None
        self.current_task: Optional[TaskTokenUsage] = None
        self.workflow_tracking = {}  # Track workflow-level metrics
        
        # Initialize tokenizer for the model
        try:
            self.tokenizer = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # Fallback to cl100k_base for unknown models
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def start_tracking(self, workflow_id: str, optimization_level: str = "none") -> Dict[str, Any]:
        """Start tracking token usage for a workflow."""
        self.workflow_tracking[workflow_id] = {
            "start_time": datetime.now(),
            "optimization_level": optimization_level,
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "llm_calls": 0,
            "tool_calls": 0,
            "status": "running"
        }
        return self.workflow_tracking[workflow_id]
    
    def stop_tracking(self, workflow_id: str) -> Dict[str, Any]:
        """Stop tracking and return final metrics for a workflow."""
        if workflow_id not in self.workflow_tracking:
            return {"error": f"No tracking data found for workflow {workflow_id}"}
        
        tracking_data = self.workflow_tracking[workflow_id]
        tracking_data["end_time"] = datetime.now()
        tracking_data["duration_seconds"] = (
            tracking_data["end_time"] - tracking_data["start_time"]
        ).total_seconds()
        tracking_data["status"] = "completed"
        
        # Calculate cost estimate
        total_tokens = tracking_data["total_tokens"]
        tracking_data["estimated_cost"] = total_tokens * 0.0000025  # Rough estimate for gpt-4o-mini
        
        return tracking_data
    
    def get_current_usage(self) -> int:
        """Get current total token usage across all workflows."""
        total = 0
        for workflow_data in self.workflow_tracking.values():
            total += workflow_data.get("total_tokens", 0)
        
        # Also include crew usage if available
        if self.crew_usage:
            total += self.crew_usage.total_token_usage.total_tokens
            
        return total
    
    def get_usage_summary(self) -> Dict[str, Any]:
        """Get comprehensive usage summary across all tracking."""
        summary = {
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_cost": 0.0,
            "workflow_count": len(self.workflow_tracking),
            "active_workflows": 0
        }
        
        for workflow_data in self.workflow_tracking.values():
            summary["total_tokens"] += workflow_data.get("total_tokens", 0)
            summary["prompt_tokens"] += workflow_data.get("prompt_tokens", 0)
            summary["completion_tokens"] += workflow_data.get("completion_tokens", 0)
            if workflow_data.get("status") == "running":
                summary["active_workflows"] += 1
        
        # Include crew usage if available
        if self.crew_usage:
            summary["total_tokens"] += self.crew_usage.total_token_usage.total_tokens
            summary["prompt_tokens"] += self.crew_usage.total_token_usage.prompt_tokens
            summary["completion_tokens"] += self.crew_usage.total_token_usage.completion_tokens
        
        # Calculate estimated cost
        summary["total_cost"] = summary["total_tokens"] * 0.0000025
        
        return summary
    
    def start_crew_tracking(self, crew_id: str) -> CrewTokenUsage:
        """Start tracking token usage for a crew execution."""
        self.crew_usage = CrewTokenUsage(
            crew_id=crew_id,
            start_time=datetime.now(),
            model_name=self.model_name
        )
        return self.crew_usage
    
    def start_task_tracking(self, task_name: str, agent_name: str) -> TaskTokenUsage:
        """Start tracking token usage for a task."""
        self.current_task = TaskTokenUsage(
            task_name=task_name,
            agent_name=agent_name,
            start_time=datetime.now()
        )
        return self.current_task
    
    def record_llm_usage(self, prompt: str, response: str, actual_usage: Optional[Dict] = None) -> TokenUsage:
        """Record token usage for an LLM call."""
        if actual_usage:
            # Use actual usage from API response if available
            token_usage = TokenUsage(
                prompt_tokens=actual_usage.get('prompt_tokens', 0),
                completion_tokens=actual_usage.get('completion_tokens', 0),
                total_tokens=actual_usage.get('total_tokens', 0)
            )
        else:
            # Estimate token usage using tiktoken
            prompt_tokens = len(self.tokenizer.encode(prompt))
            completion_tokens = len(self.tokenizer.encode(response))
            token_usage = TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens
            )
        
        if self.current_task:
            self.current_task.llm_calls += 1
            self.current_task.token_usage += token_usage
        
        return token_usage
    
    def record_tool_usage(self):
        """Record a tool call."""
        if self.current_task:
            self.current_task.tool_calls += 1
    
    def complete_task(self, status: str = "completed", error_message: Optional[str] = None):
        """Complete the current task tracking."""
        if self.current_task and self.crew_usage:
            self.current_task.complete(
                token_usage=self.current_task.token_usage,
                status=status,
                error_message=error_message
            )
            self.crew_usage.add_task_usage(self.current_task)
            self.current_task = None
    
    def complete_crew_tracking(self) -> CrewTokenUsage:
        """Complete crew tracking and return final usage."""
        if self.crew_usage:
            self.crew_usage.complete()
        return self.crew_usage
    
    def estimate_cost(self, token_usage: TokenUsage, model_name: str = None) -> Dict[str, float]:
        """Estimate cost based on token usage and model pricing."""
        model = model_name or self.model_name
        
        # Pricing per 1K tokens (as of 2024)
        pricing = {
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
            "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004},
        }
        
        model_pricing = pricing.get(model, pricing["gpt-4o-mini"])  # Default fallback
        
        input_cost = (token_usage.prompt_tokens / 1000) * model_pricing["input"]
        output_cost = (token_usage.completion_tokens / 1000) * model_pricing["output"]
        total_cost = input_cost + output_cost
        
        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "currency": "USD"
        }

class TokenAnalyzer:
    """Analyzes token usage patterns and provides insights."""
    
    @staticmethod
    def analyze_crew_usage(crew_usage: CrewTokenUsage) -> Dict[str, Any]:
        """Analyze crew token usage and provide insights."""
        if not crew_usage or not crew_usage.task_usages:
            return {"error": "No usage data available"}
        
        tracker = TokenTracker(crew_usage.model_name)
        total_cost = tracker.estimate_cost(crew_usage.total_token_usage)
        
        # Task-level analysis
        task_analysis = []
        for task in crew_usage.task_usages:
            task_cost = tracker.estimate_cost(task.token_usage)
            efficiency = task.token_usage.total_tokens / max(task.duration_seconds, 1)  # tokens per second
            
            task_analysis.append({
                "task_name": task.task_name,
                "agent_name": task.agent_name,
                "duration_seconds": task.duration_seconds,
                "token_usage": {
                    "prompt_tokens": task.token_usage.prompt_tokens,
                    "completion_tokens": task.token_usage.completion_tokens,
                    "total_tokens": task.token_usage.total_tokens
                },
                "cost_analysis": task_cost,
                "efficiency_metrics": {
                    "tokens_per_second": round(efficiency, 2),
                    "llm_calls": task.llm_calls,
                    "tool_calls": task.tool_calls,
                    "avg_tokens_per_call": round(task.token_usage.total_tokens / max(task.llm_calls, 1), 2)
                },
                "status": task.status
            })
        
        # Agent-level aggregation
        agent_stats = {}
        for task in crew_usage.task_usages:
            agent = task.agent_name
            if agent not in agent_stats:
                agent_stats[agent] = {
                    "total_tokens": 0,
                    "total_cost": 0.0,
                    "total_duration": 0.0,
                    "task_count": 0,
                    "llm_calls": 0,
                    "tool_calls": 0
                }
            
            agent_cost = tracker.estimate_cost(task.token_usage)
            agent_stats[agent]["total_tokens"] += task.token_usage.total_tokens
            agent_stats[agent]["total_cost"] += agent_cost["total_cost"]
            agent_stats[agent]["total_duration"] += task.duration_seconds
            agent_stats[agent]["task_count"] += 1
            agent_stats[agent]["llm_calls"] += task.llm_calls
            agent_stats[agent]["tool_calls"] += task.tool_calls
        
        # Overall efficiency metrics
        overall_efficiency = {
            "tokens_per_second": round(crew_usage.total_token_usage.total_tokens / max(crew_usage.total_duration_seconds, 1), 2),
            "cost_per_minute": round(total_cost["total_cost"] / max(crew_usage.total_duration_seconds / 60, 1), 4),
            "avg_tokens_per_task": round(crew_usage.total_token_usage.total_tokens / len(crew_usage.task_usages), 2)
        }
        
        return {
            "summary": {
                "total_duration_seconds": crew_usage.total_duration_seconds,
                "total_duration_minutes": round(crew_usage.total_duration_seconds / 60, 2),
                "total_tokens": crew_usage.total_token_usage.total_tokens,
                "prompt_tokens": crew_usage.total_token_usage.prompt_tokens,
                "completion_tokens": crew_usage.total_token_usage.completion_tokens,
                "model_used": crew_usage.model_name,
                "total_cost": total_cost,
                "task_count": len(crew_usage.task_usages)
            },
            "efficiency_metrics": overall_efficiency,
            "agent_breakdown": agent_stats,
            "task_breakdown": task_analysis,
            "recommendations": TokenAnalyzer._generate_recommendations(crew_usage, agent_stats, overall_efficiency)
        }
    
    @staticmethod
    def _generate_recommendations(crew_usage: CrewTokenUsage, agent_stats: Dict, efficiency: Dict) -> List[str]:
        """Generate optimization recommendations based on usage patterns."""
        recommendations = []
        
        # Cost optimization
        if crew_usage.total_token_usage.total_tokens > 50000:
            recommendations.append("Consider using a smaller model like gpt-4o-mini for cost optimization")
        
        # Efficiency recommendations
        if efficiency["tokens_per_second"] < 100:
            recommendations.append("Low token processing rate detected - consider optimizing prompts for brevity")
        
        # Agent-specific recommendations
        max_cost_agent = max(agent_stats.items(), key=lambda x: x[1]["total_cost"])
        if max_cost_agent[1]["total_cost"] > sum(stats["total_cost"] for stats in agent_stats.values()) * 0.4:
            recommendations.append(f"Agent '{max_cost_agent[0]}' consumes {max_cost_agent[1]['total_cost']:.4f} USD - consider prompt optimization")
        
        # Task duration recommendations
        long_tasks = [task for task in crew_usage.task_usages if task.duration_seconds > 120]
        if long_tasks:
            recommendations.append(f"{len(long_tasks)} tasks took over 2 minutes - consider breaking down complex tasks")
        
        return recommendations


# Global token tracker instance
_global_tracker = None

def get_token_tracker(model_name: str = "gpt-4o-mini") -> TokenTracker:
    """Get or create the global token tracker instance."""
    global _global_tracker
    if _global_tracker is None or _global_tracker.model_name != model_name:
        _global_tracker = TokenTracker(model_name)
    return _global_tracker

def reset_token_tracker():
    """Reset the global token tracker."""
    global _global_tracker
    _global_tracker = None