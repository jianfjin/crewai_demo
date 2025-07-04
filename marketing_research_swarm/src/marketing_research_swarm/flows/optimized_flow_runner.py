"""
Optimized Flow Runner with Token Management and Performance Monitoring
"""

from typing import Dict, Any, Optional
import time
from datetime import datetime

from .optimized_roi_flow import OptimizedROIFlow
from .base_flow import FlowState
from ..context.context_manager import AdvancedContextManager, ContextStrategy, ContextPriority
from ..memory.mem0_integration import MarketingMemoryManager
from ..cache.smart_cache import get_cache

class OptimizedFlowRunner:
    """
    Enhanced flow runner with comprehensive token optimization and performance monitoring
    """
    
    def __init__(self, token_budget: int = 4000, use_mem0: bool = False):
        self.token_budget = token_budget
        self.context_manager = AdvancedContextManager(token_budget)
        self.memory_manager = MarketingMemoryManager(use_mock=not use_mem0)
        self.cache = get_cache()
        self.execution_history = []
        
    def run_roi_analysis(self, data_file_path: str, 
                        context_strategy: ContextStrategy = ContextStrategy.PROGRESSIVE_PRUNING,
                        **kwargs) -> Dict[str, Any]:
        """
        Run optimized ROI analysis with comprehensive token management
        
        Args:
            data_file_path: Path to source data
            context_strategy: Context optimization strategy
            **kwargs: Additional parameters
            
        Returns:
            Analysis results with optimization metrics
        """
        print("Starting Optimized ROI Analysis Flow...")
        start_time = time.time()
        
        # Initialize flow
        flow = OptimizedROIFlow()
        
        # Configure flow state
        flow.state.data_file_path = data_file_path
        flow.state.analysis_type = "roi_analysis"
        flow.state.context_budget = self.token_budget
        
        # Setup context management
        self._setup_context_management(flow, context_strategy, data_file_path)
        
        try:
            # Execute optimized flow
            print(f"Executing with {context_strategy.value} strategy...")
            result = flow.kickoff()
            
            # Calculate execution metrics
            execution_time = time.time() - start_time
            optimization_metrics = self._calculate_comprehensive_metrics(
                flow, execution_time, context_strategy
            )
            
            # Store execution in history
            execution_record = {
                'timestamp': datetime.now().isoformat(),
                'analysis_type': 'roi_analysis',
                'execution_time': execution_time,
                'context_strategy': context_strategy.value,
                'optimization_metrics': optimization_metrics
            }
            self.execution_history.append(execution_record)
            
            # Store insights in long-term memory
            if isinstance(result, dict) and 'profitability_insights' in result:
                memory_id = self.memory_manager.store_analysis_insights(
                    analysis_type="roi_analysis",
                    insights=result['profitability_insights'],
                    metadata={
                        'data_source': data_file_path,
                        'context_strategy': context_strategy.value,
                        'execution_time': execution_time,
                        'optimization_score': optimization_metrics.get('optimization_score', 0)
                    }
                )
                result['memory_id'] = memory_id
            
            # Combine results with optimization data
            final_result = {
                'analysis_results': result,
                'optimization_metrics': optimization_metrics,
                'execution_metadata': {
                    'execution_time_seconds': execution_time,
                    'context_strategy': context_strategy.value,
                    'token_budget': self.token_budget,
                    'timestamp': datetime.now().isoformat(),
                    'flow_type': 'optimized_roi_flow'
                },
                'performance_summary': self._create_performance_summary(optimization_metrics)
            }
            
            print(f"ROI Analysis completed in {execution_time:.2f}s")
            print(f"Token savings: {optimization_metrics.get('estimated_token_savings', 0):.1f}%")
            print(f"Optimization score: {optimization_metrics.get('optimization_score', 0):.1f}/100")
            
            return final_result
            
        except Exception as e:
            print(f"Flow execution failed: {e}")
            raise
    
    def run_sales_forecast(self, data_file_path: str, **kwargs) -> Dict[str, Any]:
        """Run sales forecast analysis (placeholder)"""
        return {
            'status': 'not_implemented',
            'message': 'Sales forecast flow will be implemented in next phase',
            'optimization_ready': True
        }
    
    def run_brand_performance(self, data_file_path: str, **kwargs) -> Dict[str, Any]:
        """Run brand performance analysis (placeholder)"""
        return {
            'status': 'not_implemented',
            'message': 'Brand performance flow will be implemented in next phase',
            'optimization_ready': True
        }
    
    def _setup_context_management(self, flow: OptimizedROIFlow, 
                                 strategy: ContextStrategy, 
                                 data_file_path: str) -> None:
        """Setup context management for the flow"""
        
        # Add critical context elements
        flow.context_manager.add_context(
            key="data_file_path",
            value=data_file_path,
            priority=ContextPriority.CRITICAL
        )
        
        flow.context_manager.add_context(
            key="analysis_type",
            value="roi_analysis",
            priority=ContextPriority.CRITICAL
        )
        
        flow.context_manager.add_context(
            key="optimization_strategy",
            value=strategy.value,
            priority=ContextPriority.IMPORTANT
        )
        
        # Get relevant historical context (limited tokens)
        historical_context = self.memory_manager.get_relevant_context(
            query="roi analysis profitability budget optimization",
            analysis_type="roi_analysis",
            max_tokens=300
        )
        
        if historical_context.get('insights'):
            flow.context_manager.add_context(
                key="historical_insights",
                value=historical_context,
                priority=ContextPriority.USEFUL
            )
    
    def _calculate_comprehensive_metrics(self, flow: OptimizedROIFlow, 
                                       execution_time: float,
                                       strategy: ContextStrategy) -> Dict[str, Any]:
        """Calculate comprehensive optimization metrics"""
        
        # Get component stats
        context_stats = flow.context_manager.get_context_stats()
        cache_stats = self.cache.get_cache_stats()
        memory_stats = self.memory_manager.get_memory_stats()
        
        # Calculate token savings
        traditional_tokens = 38000  # Estimated traditional approach
        optimized_tokens = context_stats['total_tokens']
        token_savings = max(0, ((traditional_tokens - optimized_tokens) / traditional_tokens) * 100)
        
        # Calculate cost savings
        cost_per_1k_tokens = 0.0002  # gpt-4o-mini pricing
        traditional_cost = (traditional_tokens / 1000) * cost_per_1k_tokens
        optimized_cost = (optimized_tokens / 1000) * cost_per_1k_tokens
        cost_savings = traditional_cost - optimized_cost
        
        # Calculate optimization score
        optimization_score = self._calculate_optimization_score(
            context_stats, cache_stats, memory_stats, execution_time
        )
        
        return {
            'token_optimization': {
                'traditional_tokens': traditional_tokens,
                'optimized_tokens': optimized_tokens,
                'token_savings_percent': token_savings,
                'tokens_per_second': optimized_tokens / execution_time if execution_time > 0 else 0
            },
            'cost_optimization': {
                'traditional_cost_usd': traditional_cost,
                'optimized_cost_usd': optimized_cost,
                'cost_savings_usd': cost_savings,
                'cost_per_minute': (optimized_cost / (execution_time / 60)) if execution_time > 0 else 0
            },
            'context_management': {
                'strategy_used': strategy.value,
                'total_elements': context_stats['total_elements'],
                'context_tokens': context_stats['total_tokens'],
                'budget_utilization': context_stats['budget_utilization'],
                'aged_elements': context_stats.get('aged_elements', 0)
            },
            'caching_performance': {
                'total_cache_items': cache_stats['total_items'],
                'cache_size_mb': cache_stats['total_size_mb'],
                'memory_items': cache_stats['memory_items'],
                'disk_items': cache_stats['disk_items']
            },
            'memory_management': {
                'local_cache_size': memory_stats['local_cache_size'],
                'total_stored_insights': memory_stats['total_stored_insights'],
                'mem0_available': memory_stats['mem0_available']
            },
            'performance_metrics': {
                'execution_time_seconds': execution_time,
                'optimization_score': optimization_score,
                'efficiency_rating': self._get_efficiency_rating(optimization_score)
            }
        }
    
    def _calculate_optimization_score(self, context_stats: Dict, cache_stats: Dict, 
                                    memory_stats: Dict, execution_time: float) -> float:
        """Calculate overall optimization score (0-100)"""
        
        # Token efficiency (25% weight)
        budget_utilization = float(context_stats['budget_utilization'].rstrip('%'))
        token_score = max(0, 100 - abs(budget_utilization - 75))  # Optimal around 75%
        
        # Cache efficiency (25% weight)
        cache_score = min(100, cache_stats['total_items'] * 10)  # More cache usage = better
        
        # Memory efficiency (20% weight)
        memory_score = min(100, memory_stats['total_stored_insights'] * 5)
        
        # Execution efficiency (20% weight)
        execution_score = max(0, 100 - execution_time * 2)  # Faster = better
        
        # Context management (10% weight)
        context_score = min(100, 100 - context_stats['total_elements'])  # Fewer elements = better
        
        # Weighted average
        overall_score = (
            token_score * 0.25 +
            cache_score * 0.25 +
            memory_score * 0.20 +
            execution_score * 0.20 +
            context_score * 0.10
        )
        
        return max(0, min(100, overall_score))
    
    def _get_efficiency_rating(self, score: float) -> str:
        """Get efficiency rating based on score"""
        if score >= 90:
            return "Excellent"
        elif score >= 80:
            return "Very Good"
        elif score >= 70:
            return "Good"
        elif score >= 60:
            return "Fair"
        else:
            return "Needs Improvement"
    
    def _create_performance_summary(self, metrics: Dict[str, Any]) -> Dict[str, str]:
        """Create human-readable performance summary"""
        token_savings = metrics['token_optimization']['token_savings_percent']
        cost_savings = metrics['cost_optimization']['cost_savings_usd']
        optimization_score = metrics['performance_metrics']['optimization_score']
        execution_time = metrics['performance_metrics']['execution_time_seconds']
        
        return {
            'token_efficiency': f"{token_savings:.1f}% reduction in token usage",
            'cost_efficiency': f"${cost_savings:.4f} saved per analysis",
            'performance': f"{optimization_score:.1f}/100 optimization score",
            'speed': f"Completed in {execution_time:.2f} seconds",
            'overall_rating': metrics['performance_metrics']['efficiency_rating']
        }
    
    def get_optimization_recommendations(self) -> Dict[str, Any]:
        """Get recommendations for further optimization"""
        
        context_stats = self.context_manager.get_context_stats()
        cache_stats = self.cache.get_cache_stats()
        memory_stats = self.memory_manager.get_memory_stats()
        
        recommendations = []
        
        # Context optimization recommendations
        budget_utilization = float(context_stats['budget_utilization'].rstrip('%'))
        
        if budget_utilization > 90:
            recommendations.append({
                'type': 'context_optimization',
                'priority': 'high',
                'message': 'Context budget nearly exceeded',
                'action': 'Use MINIMAL_CONTEXT or STATELESS strategy for better efficiency'
            })
        elif budget_utilization < 40:
            recommendations.append({
                'type': 'context_optimization',
                'priority': 'low',
                'message': 'Context budget underutilized',
                'action': 'Consider including more historical context for richer analysis'
            })
        
        # Cache optimization recommendations
        if cache_stats['total_size_mb'] > 100:
            recommendations.append({
                'type': 'cache_optimization',
                'priority': 'medium',
                'message': 'Large cache detected',
                'action': 'Run cache cleanup to free up space'
            })
        
        # Memory optimization recommendations
        if memory_stats['local_cache_size'] > 50:
            recommendations.append({
                'type': 'memory_optimization',
                'priority': 'medium',
                'message': 'Large memory cache detected',
                'action': 'Consider running cleanup_old_memories()'
            })
        
        # Performance recommendations
        if len(self.execution_history) > 1:
            recent_times = [exec['execution_time'] for exec in self.execution_history[-5:]]
            avg_time = sum(recent_times) / len(recent_times)
            
            if avg_time > 60:  # More than 1 minute
                recommendations.append({
                    'type': 'performance',
                    'priority': 'high',
                    'message': 'Slow execution times detected',
                    'action': 'Consider using more aggressive context pruning strategies'
                })
        
        return {
            'recommendations': recommendations,
            'current_stats': {
                'context': context_stats,
                'cache': cache_stats,
                'memory': memory_stats
            },
            'execution_history_count': len(self.execution_history),
            'optimization_trends': self._analyze_optimization_trends()
        }
    
    def _analyze_optimization_trends(self) -> Dict[str, Any]:
        """Analyze optimization trends over time"""
        if len(self.execution_history) < 2:
            return {'message': 'Insufficient data for trend analysis'}
        
        recent_executions = self.execution_history[-5:]  # Last 5 executions
        
        execution_times = [exec['execution_time'] for exec in recent_executions]
        optimization_scores = [
            exec['optimization_metrics']['performance_metrics']['optimization_score'] 
            for exec in recent_executions
        ]
        
        return {
            'average_execution_time': sum(execution_times) / len(execution_times),
            'average_optimization_score': sum(optimization_scores) / len(optimization_scores),
            'execution_time_trend': 'improving' if execution_times[-1] < execution_times[0] else 'stable',
            'optimization_trend': 'improving' if optimization_scores[-1] > optimization_scores[0] else 'stable',
            'total_analyses': len(self.execution_history)
        }
    
    def cleanup_resources(self) -> Dict[str, Any]:
        """Cleanup optimization resources"""
        print("Cleaning up optimization resources...")
        
        # Cleanup memory
        memory_cleanup = self.memory_manager.cleanup_old_memories(retention_days=7)
        
        # Cleanup cache
        cache_cleanup = self.cache.cleanup_expired()
        
        # Reset context manager
        initial_elements = len(self.context_manager.context_elements)
        self.context_manager = AdvancedContextManager(self.token_budget)
        
        cleanup_stats = {
            'memories_cleaned': memory_cleanup,
            'cache_items_cleaned': cache_cleanup,
            'context_elements_reset': initial_elements,
            'cleanup_timestamp': datetime.now().isoformat()
        }
        
        print(f"Cleanup complete: {memory_cleanup} memories, {cache_cleanup} cache items, {initial_elements} context elements")
        
        return cleanup_stats