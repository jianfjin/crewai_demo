"""
Performance Optimization Module

This module provides performance optimizations for the marketing research swarm:
- Shared data cache to eliminate redundant loading
- Parallel execution flows for independent agents
- Context isolation optimization
- Performance profiling and monitoring
"""

from .shared_data_cache import SharedDataCache, get_shared_cache, clear_global_cache
from .parallel_execution import ParallelExecutionManager, create_parallel_flow
from .context_optimizer import ContextOptimizer, get_context_optimizer
# Optional import - make psutil dependency optional
try:
    from .performance_profiler import PerformanceProfiler, get_profiler
except ImportError as e:
    # Create mock classes if psutil is not available
    class MockPerformanceProfiler:
        def __init__(self):
            pass
        def start_profiling(self, *args, **kwargs):
            pass
        def stop_profiling(self, *args, **kwargs):
            return {}
    
    PerformanceProfiler = MockPerformanceProfiler
    
    def get_profiler():
        return MockPerformanceProfiler()

__all__ = [
    'SharedDataCache',
    'get_shared_cache', 
    'clear_global_cache',
    'ParallelExecutionManager',
    'create_parallel_flow',
    'ContextOptimizer',
    'get_context_optimizer',
    'PerformanceProfiler',
    'get_profiler'
]