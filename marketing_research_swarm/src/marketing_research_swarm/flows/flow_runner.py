"""
Flow Runner with Token Optimization
Orchestrates flow execution with context management and caching
"""

from typing import Dict, Any, Optional
import time
from datetime import datetime

from .roi_analysis_flow import ROIAnalysisFlow
from .base_flow import FlowState
from ..context.context_manager import AdvancedContextManager, ContextStrategy, ContextPriority
from ..memory.mem0_integration import MarketingMemoryManager

class OptimizedFlowRunner:
    """
    Flow runner with advanced token optimization
    """
    
    def __init__(self, token_budget: int = 4000, use_mem0: bool = True):
        self.token_budget = token_budget
        self.context_manager = AdvancedContextManager(token_budget)
        self.memory_manager = MarketingMemoryManager(use_mock=not use_mem0)
        self.execution_stats = {}
        
    def run_roi_analysis(self, data_file_path: str, 
                        context_strategy: ContextStrategy = ContextStrategy.PROGRESSIVE_PRUNING,
                        **kwargs) -> Dict[str, Any]:
        """
        Run ROI analysis with token optimization
        
        Args:
            data_file_path: Path to source data
            context_strategy: Context optimization strategy
            **kwargs: Additional parameters
            
        Returns:
            Analysis results with optimization metrics
        """
        print("🚀 Starting Optimized ROI Analysis Flow...")
        start_time = time.time()
        
        # Initialize flow state
        flow_state = FlowState(
            data_file_path=data_file_path,
            analysis_type="roi_analysis",
            context_budget=self.token_budget
        )
        
        # Create and configure flow
        flow = ROIAnalysisFlow()
        flow.state = flow_state
        
        # Add context management
        self._setup_context_management(flow, context_strategy)
        
        try:
            # Execute flow
            print(f"📊 Executing with {context_strategy.value} strategy...")
            result = flow.kickoff()
            
            # Store insights in long-term memory
            if isinstance(result, dict):
                memory_id = self.memory_manager.store_analysis_insights(
                    analysis_type="roi_analysis",
                    insights=result,
                    metadata={
                        'data_source': data_file_path,
                        'context_strategy': context_strategy.value,
                        'execution_time': time.time() - start_time
                    }
                )
                result['memory_id'] = memory_id
            
            # Calculate optimization metrics
            execution_time = time.time() - start_time
            optimization_metrics = self._calculate_optimization_metrics(
                flow, execution_time, context_strategy
            )
            
            # Combine results
            final_result = {
                'analysis_results': result,
                'optimization_metrics': optimization_metrics,
                'execution_metadata': {
                    'execution_time': execution_time,
                    'context_strategy': context_strategy.value,
                    'token_budget': self.token_budget,
                    'timestamp': datetime.now().isoformat()
                }
            }
            
            print(f"✅ ROI Analysis completed in {execution_time:.2f}s")
            print(f"💾 Token savings: {optimization_metrics.get('estimated_token_savings', 0)}%")
            
            return final_result
            
        except Exception as e:
            print(f"❌ Flow execution failed: {e}")
            raise
    
    def run_sales_forecast(self, data_file_path: str, **kwargs) -> Dict[str, Any]:
        """Run sales forecast analysis (placeholder for future implementation)"""
        print("🔮 Sales Forecast Flow - Coming Soon!")
        return {
            'status': 'not_implemented',
            'message': 'Sales forecast flow will be implemented in next phase'
        }
    
    def run_brand_performance(self, data_file_path: str, **kwargs) -> Dict[str, Any]:
        """Run brand performance analysis (placeholder for future implementation)"""
        print("🏆 Brand Performance Flow - Coming Soon!")
        return {
            'status': 'not_implemented', 
            'message': 'Brand performance flow will be implemented in next phase'
        }
    
    def _setup_context_management(self, flow: ROIAnalysisFlow, 
                                 strategy: ContextStrategy) -> None:
        """Setup context management for the flow"""
        
        # Add critical context elements
        self.context_manager.add_context(
            key="data_file_path",
            value=flow.state.data_file_path,
            priority=ContextPriority.CRITICAL
        )
        
        self.context_manager.add_context(
            key="analysis_type", 
            value="roi_analysis",
            priority=ContextPriority.CRITICAL
        )
        
        self.context_manager.add_context(
            key="context_strategy",
            value=strategy.value,
            priority=ContextPriority.IMPORTANT
        )
        
        # Get relevant historical context
        historical_context = self.memory_manager.get_relevant_context(
            query="roi analysis profitability budget optimization",
            analysis_type="roi_analysis",
            max_tokens=500
        )
        
        if historical_context.get('insights'):
            self.context_manager.add_context(
                key="historical_insights",
                value=historical_context,
                priority=ContextPriority.USEFUL
            )
    
    def _calculate_optimization_metrics(self, flow: ROIAnalysisFlow, 
                                      execution_time: float,
                                      strategy: ContextStrategy) -> Dict[str, Any]:
        """Calculate optimization metrics"""
        
        # Get context stats
        context_stats = self.context_manager.get_context_stats()
        
        # Get memory stats
        memory_stats = self.memory_manager.get_memory_stats()
        
        # Estimate token savings (compared to traditional approach)
        traditional_tokens = 40000  # Estimated traditional approach
        optimized_tokens = context_stats['total_tokens']
        token_savings = ((traditional_tokens - optimized_tokens) / traditional_tokens) * 100
        
        # Calculate cost savings
        cost_per_1k_tokens = 0.0002  # gpt-4o-mini pricing
        traditional_cost = (traditional_tokens / 1000) * cost_per_1k_tokens
        optimized_cost = (optimized_tokens / 1000) * cost_per_1k_tokens
        cost_savings = traditional_cost - optimized_cost
        
        return {
            'estimated_token_savings': max(0, token_savings),
            'cost_savings_usd': cost_savings,
            'execution_time_seconds': execution_time,
            'context_optimization': {
                'strategy_used': strategy.value,
                'total_context_elements': context_stats['total_elements'],
                'context_tokens': context_stats['total_tokens'],
                'budget_utilization': context_stats['budget_utilization']
            },
            'memory_management': {
                'local_cache_size': memory_stats['local_cache_size'],
                'total_stored_insights': memory_stats['total_stored_insights'],
                'mem0_available': memory_stats['mem0_available']
            },
            'performance_metrics': {
                'tokens_per_second': optimized_tokens / execution_time if execution_time > 0 else 0,
                'cost_per_minute': (optimized_cost / (execution_time / 60)) if execution_time > 0 else 0,
                'efficiency_score': min(100, (token_savings + (60 - execution_time)) / 2)  # Combined metric
            }
        }
    
    def get_optimization_recommendations(self, analysis_type: str = "roi_analysis") -> Dict[str, Any]:
        """Get recommendations for further optimization"""
        
        context_stats = self.context_manager.get_context_stats()
        memory_stats = self.memory_manager.get_memory_stats()
        
        recommendations = []
        
        # Context optimization recommendations
        budget_utilization = float(context_stats['budget_utilization'].rstrip('%'))
        
        if budget_utilization > 90:
            recommendations.append({
                'type': 'context_optimization',
                'priority': 'high',
                'message': 'Context budget nearly exceeded. Consider more aggressive pruning.',
                'action': 'Use MINIMAL_CONTEXT or STATELESS strategy'
            })
        elif budget_utilization < 50:
            recommendations.append({
                'type': 'context_optimization',
                'priority': 'low',
                'message': 'Context budget underutilized. Could include more historical context.',
                'action': 'Consider including more historical insights'
            })
        
        # Memory optimization recommendations
        if memory_stats['local_cache_size'] > 100:
            recommendations.append({
                'type': 'memory_optimization',
                'priority': 'medium',
                'message': 'Large local cache detected. Consider cleanup.',
                'action': 'Run cleanup_old_memories() method'
            })
        
        # Performance recommendations
        aged_elements = context_stats.get('aged_elements', 0)
        if aged_elements > 10:
            recommendations.append({
                'type': 'performance',
                'priority': 'medium', 
                'message': f'{aged_elements} aged context elements detected.',
                'action': 'Automatic aging is working well'
            })
        
        return {
            'recommendations': recommendations,
            'current_stats': {
                'context': context_stats,
                'memory': memory_stats
            },
            'optimization_score': self._calculate_optimization_score(context_stats, memory_stats)
        }
    
    def _calculate_optimization_score(self, context_stats: Dict, memory_stats: Dict) -> float:
        """Calculate overall optimization score (0-100)"""
        
        # Context efficiency (30% weight)
        budget_utilization = float(context_stats['budget_utilization'].rstrip('%'))
        context_score = 100 - abs(budget_utilization - 75)  # Optimal around 75%
        
        # Memory efficiency (20% weight) 
        memory_score = min(100, 100 - memory_stats['local_cache_size'])  # Lower cache = better
        
        # Element management (25% weight)
        total_elements = context_stats['total_elements']
        element_score = max(0, 100 - (total_elements * 2))  # Fewer elements = better
        
        # Aging effectiveness (25% weight)
        aged_elements = context_stats.get('aged_elements', 0)
        aging_score = min(100, aged_elements * 10)  # Some aging is good
        
        # Weighted average
        overall_score = (
            context_score * 0.30 +
            memory_score * 0.20 + 
            element_score * 0.25 +
            aging_score * 0.25
        )
        
        return max(0, min(100, overall_score))
    
    def cleanup_resources(self) -> Dict[str, Any]:
        """Cleanup resources and return cleanup stats"""
        
        print("🧹 Cleaning up optimization resources...")
        
        # Cleanup memory
        memory_cleanup = self.memory_manager.cleanup_old_memories(retention_days=30)
        
        # Reset context manager
        initial_elements = len(self.context_manager.context_elements)
        self.context_manager = AdvancedContextManager(self.token_budget)
        
        cleanup_stats = {
            'memories_cleaned': memory_cleanup,
            'context_elements_reset': initial_elements,
            'cleanup_timestamp': datetime.now().isoformat()
        }
        
        print(f"✅ Cleanup complete: {memory_cleanup} memories, {initial_elements} context elements")
        
        return cleanup_stats