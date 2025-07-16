"""
Performance Profiler for Marketing Research Swarm

This module provides comprehensive performance profiling and monitoring
to identify bottlenecks and measure optimization effectiveness.
"""

import time
import psutil
import threading
import json
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import logging
import traceback
import functools

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Individual performance metric measurement."""
    name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    memory_start_mb: Optional[float] = None
    memory_end_mb: Optional[float] = None
    memory_delta_mb: Optional[float] = None
    cpu_percent: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class WorkflowProfile:
    """Complete workflow performance profile."""
    workflow_id: str
    workflow_type: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_duration_seconds: Optional[float] = None
    agents_executed: List[str] = None
    metrics: List[PerformanceMetric] = None
    optimization_level: str = "none"
    token_usage: Dict[str, Any] = None
    cache_stats: Dict[str, Any] = None
    context_stats: Dict[str, Any] = None

class PerformanceProfiler:
    """
    Comprehensive performance profiler for the marketing research system.
    Tracks timing, memory usage, CPU utilization, and optimization effectiveness.
    """
    
    def __init__(self, enable_detailed_profiling: bool = True):
        """
        Initialize the performance profiler.
        
        Args:
            enable_detailed_profiling: Whether to enable detailed system metrics
        """
        self.enable_detailed_profiling = enable_detailed_profiling
        self.active_metrics: Dict[str, PerformanceMetric] = {}
        self.completed_metrics: List[PerformanceMetric] = []
        self.workflow_profiles: List[WorkflowProfile] = []
        self._lock = threading.RLock()
        
        # System monitoring
        self.process = psutil.Process()
        self.baseline_memory = self._get_memory_usage()
        self.baseline_cpu = self._get_cpu_usage()
        
        logger.info(f"📊 PerformanceProfiler initialized (detailed={enable_detailed_profiling})")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            return self.process.memory_info().rss / (1024 * 1024)
        except:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            return self.process.cpu_percent()
        except:
            return 0.0
    
    @contextmanager
    def profile_operation(self, operation_name: str, metadata: Dict[str, Any] = None):
        """
        Context manager for profiling operations.
        
        Args:
            operation_name: Name of the operation being profiled
            metadata: Additional metadata to store
        """
        metric_id = f"{operation_name}_{int(time.time() * 1000)}"
        
        try:
            # Start profiling
            start_time = datetime.now()
            memory_start = self._get_memory_usage() if self.enable_detailed_profiling else None
            
            metric = PerformanceMetric(
                name=operation_name,
                start_time=start_time,
                memory_start_mb=memory_start,
                metadata=metadata or {}
            )
            
            with self._lock:
                self.active_metrics[metric_id] = metric
            
            logger.debug(f"🔍 Started profiling: {operation_name}")
            
            yield metric
            
        except Exception as e:
            # Handle errors
            metric.success = False
            metric.error_message = str(e)
            logger.error(f"❌ Error in profiled operation {operation_name}: {e}")
            raise
            
        finally:
            # Complete profiling
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            memory_end = self._get_memory_usage() if self.enable_detailed_profiling else None
            cpu_percent = self._get_cpu_usage() if self.enable_detailed_profiling else None
            
            metric.end_time = end_time
            metric.duration_seconds = duration
            metric.memory_end_mb = memory_end
            metric.cpu_percent = cpu_percent
            
            if memory_start and memory_end:
                metric.memory_delta_mb = memory_end - memory_start
            
            with self._lock:
                if metric_id in self.active_metrics:
                    del self.active_metrics[metric_id]
                self.completed_metrics.append(metric)
            
            logger.debug(f"✅ Completed profiling: {operation_name} ({duration:.3f}s)")
    
    def profile_function(self, operation_name: str = None, include_args: bool = False):
        """
        Decorator for profiling functions.
        
        Args:
            operation_name: Custom name for the operation
            include_args: Whether to include function arguments in metadata
        """
        def decorator(func: Callable):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                name = operation_name or f"{func.__module__}.{func.__name__}"
                metadata = {}
                
                if include_args:
                    metadata['args_count'] = len(args)
                    metadata['kwargs_keys'] = list(kwargs.keys())
                
                with self.profile_operation(name, metadata):
                    return func(*args, **kwargs)
            
            return wrapper
        return decorator
    
    def start_workflow_profiling(self, workflow_id: str, workflow_type: str, 
                                optimization_level: str = "none") -> WorkflowProfile:
        """
        Start profiling a complete workflow.
        
        Args:
            workflow_id: Unique workflow identifier
            workflow_type: Type of workflow
            optimization_level: Optimization level being used
            
        Returns:
            WorkflowProfile object
        """
        profile = WorkflowProfile(
            workflow_id=workflow_id,
            workflow_type=workflow_type,
            start_time=datetime.now(),
            agents_executed=[],
            metrics=[],
            optimization_level=optimization_level
        )
        
        with self._lock:
            self.workflow_profiles.append(profile)
        
        logger.info(f"🚀 Started workflow profiling: {workflow_id} ({workflow_type})")
        return profile
    
    def complete_workflow_profiling(self, workflow_id: str, 
                                  agents_executed: List[str] = None,
                                  token_usage: Dict[str, Any] = None,
                                  cache_stats: Dict[str, Any] = None,
                                  context_stats: Dict[str, Any] = None) -> Optional[WorkflowProfile]:
        """
        Complete workflow profiling and calculate final metrics.
        
        Args:
            workflow_id: Workflow identifier
            agents_executed: List of agents that were executed
            token_usage: Token usage statistics
            cache_stats: Cache performance statistics
            context_stats: Context optimization statistics
            
        Returns:
            Completed WorkflowProfile or None if not found
        """
        with self._lock:
            # Find the workflow profile
            profile = None
            for p in self.workflow_profiles:
                if p.workflow_id == workflow_id:
                    profile = p
                    break
            
            if not profile:
                logger.warning(f"⚠️ Workflow profile not found: {workflow_id}")
                return None
            
            # Complete the profile
            profile.end_time = datetime.now()
            profile.total_duration_seconds = (profile.end_time - profile.start_time).total_seconds()
            profile.agents_executed = agents_executed or []
            profile.token_usage = token_usage
            profile.cache_stats = cache_stats
            profile.context_stats = context_stats
            
            # Collect relevant metrics
            profile.metrics = [
                m for m in self.completed_metrics
                if m.start_time >= profile.start_time and 
                   (profile.end_time is None or m.start_time <= profile.end_time)
            ]
        
        logger.info(f"✅ Completed workflow profiling: {workflow_id} ({profile.total_duration_seconds:.2f}s)")
        return profile
    
    def get_performance_summary(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """
        Get comprehensive performance summary.
        
        Args:
            time_window_hours: Time window for analysis
            
        Returns:
            Performance summary dictionary
        """
        cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
        
        with self._lock:
            # Filter recent metrics
            recent_metrics = [
                m for m in self.completed_metrics
                if m.start_time >= cutoff_time and m.success
            ]
            
            recent_workflows = [
                w for w in self.workflow_profiles
                if w.start_time >= cutoff_time and w.end_time is not None
            ]
        
        # Calculate operation statistics
        operation_stats = {}
        for metric in recent_metrics:
            if metric.name not in operation_stats:
                operation_stats[metric.name] = {
                    'count': 0,
                    'total_duration': 0.0,
                    'min_duration': float('inf'),
                    'max_duration': 0.0,
                    'total_memory_delta': 0.0,
                    'avg_cpu_percent': 0.0
                }
            
            stats = operation_stats[metric.name]
            stats['count'] += 1
            stats['total_duration'] += metric.duration_seconds or 0
            stats['min_duration'] = min(stats['min_duration'], metric.duration_seconds or 0)
            stats['max_duration'] = max(stats['max_duration'], metric.duration_seconds or 0)
            
            if metric.memory_delta_mb:
                stats['total_memory_delta'] += metric.memory_delta_mb
            if metric.cpu_percent:
                stats['avg_cpu_percent'] += metric.cpu_percent
        
        # Calculate averages
        for stats in operation_stats.values():
            if stats['count'] > 0:
                stats['avg_duration'] = stats['total_duration'] / stats['count']
                stats['avg_memory_delta'] = stats['total_memory_delta'] / stats['count']
                stats['avg_cpu_percent'] = stats['avg_cpu_percent'] / stats['count']
        
        # Calculate workflow statistics
        workflow_stats = {}
        if recent_workflows:
            total_workflow_time = sum(w.total_duration_seconds or 0 for w in recent_workflows)
            avg_workflow_time = total_workflow_time / len(recent_workflows)
            
            # Group by optimization level
            optimization_performance = {}
            for workflow in recent_workflows:
                opt_level = workflow.optimization_level
                if opt_level not in optimization_performance:
                    optimization_performance[opt_level] = {
                        'count': 0,
                        'total_duration': 0.0,
                        'total_token_usage': 0
                    }
                
                optimization_performance[opt_level]['count'] += 1
                optimization_performance[opt_level]['total_duration'] += workflow.total_duration_seconds or 0
                
                if workflow.token_usage and 'total_tokens' in workflow.token_usage:
                    optimization_performance[opt_level]['total_token_usage'] += workflow.token_usage['total_tokens']
            
            # Calculate averages
            for opt_stats in optimization_performance.values():
                if opt_stats['count'] > 0:
                    opt_stats['avg_duration'] = opt_stats['total_duration'] / opt_stats['count']
                    opt_stats['avg_token_usage'] = opt_stats['total_token_usage'] / opt_stats['count']
            
            workflow_stats = {
                'total_workflows': len(recent_workflows),
                'avg_workflow_duration': avg_workflow_time,
                'optimization_performance': optimization_performance
            }
        
        # System resource usage
        current_memory = self._get_memory_usage()
        memory_increase = current_memory - self.baseline_memory
        
        summary = {
            'time_window_hours': time_window_hours,
            'total_operations': len(recent_metrics),
            'successful_operations': len([m for m in recent_metrics if m.success]),
            'failed_operations': len([m for m in recent_metrics if not m.success]),
            'operation_stats': operation_stats,
            'workflow_stats': workflow_stats,
            'system_resources': {
                'current_memory_mb': current_memory,
                'baseline_memory_mb': self.baseline_memory,
                'memory_increase_mb': memory_increase,
                'current_cpu_percent': self._get_cpu_usage()
            },
            'top_slowest_operations': self._get_slowest_operations(recent_metrics, 5),
            'top_memory_consumers': self._get_memory_consumers(recent_metrics, 5)
        }
        
        return summary
    
    def _get_slowest_operations(self, metrics: List[PerformanceMetric], limit: int) -> List[Dict[str, Any]]:
        """Get the slowest operations."""
        sorted_metrics = sorted(
            [m for m in metrics if m.duration_seconds],
            key=lambda x: x.duration_seconds,
            reverse=True
        )
        
        return [
            {
                'name': m.name,
                'duration_seconds': m.duration_seconds,
                'start_time': m.start_time.isoformat(),
                'memory_delta_mb': m.memory_delta_mb
            }
            for m in sorted_metrics[:limit]
        ]
    
    def _get_memory_consumers(self, metrics: List[PerformanceMetric], limit: int) -> List[Dict[str, Any]]:
        """Get the biggest memory consumers."""
        sorted_metrics = sorted(
            [m for m in metrics if m.memory_delta_mb and m.memory_delta_mb > 0],
            key=lambda x: x.memory_delta_mb,
            reverse=True
        )
        
        return [
            {
                'name': m.name,
                'memory_delta_mb': m.memory_delta_mb,
                'duration_seconds': m.duration_seconds,
                'start_time': m.start_time.isoformat()
            }
            for m in sorted_metrics[:limit]
        ]
    
    def export_performance_data(self, file_path: str, time_window_hours: int = 24):
        """
        Export performance data to a JSON file.
        
        Args:
            file_path: Path to export file
            time_window_hours: Time window for data export
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
            
            with self._lock:
                # Filter recent data
                recent_metrics = [
                    asdict(m) for m in self.completed_metrics
                    if m.start_time >= cutoff_time
                ]
                
                recent_workflows = [
                    asdict(w) for w in self.workflow_profiles
                    if w.start_time >= cutoff_time
                ]
            
            # Convert datetime objects to strings
            for metric in recent_metrics:
                metric['start_time'] = metric['start_time'].isoformat() if metric['start_time'] else None
                metric['end_time'] = metric['end_time'].isoformat() if metric['end_time'] else None
            
            for workflow in recent_workflows:
                workflow['start_time'] = workflow['start_time'].isoformat() if workflow['start_time'] else None
                workflow['end_time'] = workflow['end_time'].isoformat() if workflow['end_time'] else None
                
                # Convert nested metrics
                if workflow['metrics']:
                    for metric in workflow['metrics']:
                        metric['start_time'] = metric['start_time'].isoformat() if metric['start_time'] else None
                        metric['end_time'] = metric['end_time'].isoformat() if metric['end_time'] else None
            
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'time_window_hours': time_window_hours,
                'performance_summary': self.get_performance_summary(time_window_hours),
                'metrics': recent_metrics,
                'workflows': recent_workflows
            }
            
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"📁 Exported performance data to {file_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to export performance data: {e}")
    
    def get_optimization_effectiveness(self) -> Dict[str, Any]:
        """
        Analyze the effectiveness of different optimization strategies.
        
        Returns:
            Dictionary with optimization effectiveness metrics
        """
        with self._lock:
            workflows_by_optimization = {}
            
            for workflow in self.workflow_profiles:
                if workflow.end_time is None:
                    continue
                
                opt_level = workflow.optimization_level
                if opt_level not in workflows_by_optimization:
                    workflows_by_optimization[opt_level] = []
                
                workflows_by_optimization[opt_level].append(workflow)
        
        effectiveness = {}
        
        for opt_level, workflows in workflows_by_optimization.items():
            if not workflows:
                continue
            
            durations = [w.total_duration_seconds for w in workflows if w.total_duration_seconds]
            token_usages = []
            cache_hit_rates = []
            context_compressions = []
            
            for w in workflows:
                if w.token_usage and 'total_tokens' in w.token_usage:
                    token_usages.append(w.token_usage['total_tokens'])
                
                if w.cache_stats and 'hit_rate_percentage' in w.cache_stats:
                    cache_hit_rates.append(w.cache_stats['hit_rate_percentage'])
                
                if w.context_stats and 'compression_ratio' in w.context_stats:
                    context_compressions.append(w.context_stats['compression_ratio'])
            
            effectiveness[opt_level] = {
                'workflow_count': len(workflows),
                'avg_duration_seconds': sum(durations) / len(durations) if durations else 0,
                'avg_token_usage': sum(token_usages) / len(token_usages) if token_usages else 0,
                'avg_cache_hit_rate': sum(cache_hit_rates) / len(cache_hit_rates) if cache_hit_rates else 0,
                'avg_context_compression': sum(context_compressions) / len(context_compressions) if context_compressions else 0
            }
        
        # Calculate relative improvements
        if 'none' in effectiveness and len(effectiveness) > 1:
            baseline = effectiveness['none']
            
            for opt_level, stats in effectiveness.items():
                if opt_level == 'none':
                    continue
                
                stats['duration_improvement_percent'] = (
                    (baseline['avg_duration_seconds'] - stats['avg_duration_seconds']) / 
                    baseline['avg_duration_seconds'] * 100
                ) if baseline['avg_duration_seconds'] > 0 else 0
                
                stats['token_reduction_percent'] = (
                    (baseline['avg_token_usage'] - stats['avg_token_usage']) / 
                    baseline['avg_token_usage'] * 100
                ) if baseline['avg_token_usage'] > 0 else 0
        
        return effectiveness
    
    def clear_old_data(self, max_age_hours: int = 168):  # 1 week default
        """Clear old performance data to prevent memory buildup."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        with self._lock:
            # Remove old metrics
            old_metrics_count = len(self.completed_metrics)
            self.completed_metrics = [
                m for m in self.completed_metrics
                if m.start_time >= cutoff_time
            ]
            
            # Remove old workflows
            old_workflows_count = len(self.workflow_profiles)
            self.workflow_profiles = [
                w for w in self.workflow_profiles
                if w.start_time >= cutoff_time
            ]
            
            metrics_removed = old_metrics_count - len(self.completed_metrics)
            workflows_removed = old_workflows_count - len(self.workflow_profiles)
        
        if metrics_removed > 0 or workflows_removed > 0:
            logger.info(f"🧹 Cleaned up {metrics_removed} old metrics and {workflows_removed} old workflows")

# Global profiler instance
_global_profiler = None

def get_profiler() -> PerformanceProfiler:
    """Get the global performance profiler instance."""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler()
    return _global_profiler