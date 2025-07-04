"""
Cached ROI Analysis Flow with Persistent Result Storage
Checks cache before execution and stores results for future use
"""

from crewai.flow import Flow, start, listen
import pandas as pd
from typing import Dict, Any, Optional
import time

from .base_flow import FlowState
from .optimized_roi_flow import OptimizedROIFlow
from ..persistence.analysis_cache import get_analysis_cache
from ..context.context_manager import ContextStrategy

class CachedROIFlow(Flow[FlowState]):
    """ROI analysis flow with intelligent persistent caching"""
    
    def __init__(self):
        super().__init__()
        self.analysis_cache = get_analysis_cache()
        self.optimized_flow = OptimizedROIFlow()
        self.request_hash = None
        self.cache_hit = False
        
        # Initialize state
        self.state = FlowState()
    
    @start()
    def check_cache_and_execute(self, analysis_parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Check cache for existing results, execute analysis if not found
        
        Args:
            analysis_parameters: Analysis configuration parameters
            
        Returns:
            Analysis results (from cache or fresh execution)
        """
        print("🔍 Checking cache for existing analysis results...")
        
        # Prepare analysis parameters
        if not analysis_parameters:
            analysis_parameters = {
                'analysis_type': 'roi_analysis',
                'context_strategy': 'progressive_pruning',
                'token_budget': 4000,
                'target_audience': 'health-conscious millennials',
                'budget': 100000
            }
        
        # Generate request hash
        self.request_hash = self.analysis_cache.generate_request_hash(
            analysis_type='roi_analysis',
            data_path=self.state.data_file_path,
            parameters=analysis_parameters
        )
        
        print(f"📋 Request hash: {self.request_hash}")
        
        # Check for exact cache hit
        cached_result = self.analysis_cache.retrieve_cached_result(self.request_hash)
        
        if cached_result:
            print("🎯 Cache HIT! Returning cached analysis results")
            self.cache_hit = True
            
            # Add cache metadata to result
            result = cached_result['final_result'].copy()
            result['cache_info'] = {
                'cache_hit': True,
                'cached_at': cached_result['cached_at'],
                'request_hash': self.request_hash,
                'execution_time_saved': 'Estimated 30-60 seconds',
                'cost_saved': 'Estimated $0.0007',
                'cache_source': 'persistent_storage'
            }
            
            return result
        
        # Check for similar analyses
        similar_analyses = self.analysis_cache.find_similar_analyses(
            analysis_type='roi_analysis',
            data_path=self.state.data_file_path,
            parameters=analysis_parameters,
            similarity_threshold=0.9
        )
        
        if similar_analyses:
            print(f"🔍 Found {len(similar_analyses)} similar analyses")
            best_match = similar_analyses[0]
            
            if best_match['similarity_score'] > 0.95:
                print(f"📊 Using highly similar analysis (similarity: {best_match['similarity_score']:.1%})")
                
                # Retrieve the similar result
                similar_result = self.analysis_cache.retrieve_cached_result(best_match['request_hash'])
                if similar_result:
                    result = similar_result['final_result'].copy()
                    result['cache_info'] = {
                        'cache_hit': True,
                        'similarity_match': True,
                        'similarity_score': best_match['similarity_score'],
                        'original_request_hash': best_match['request_hash'],
                        'cache_source': 'similar_analysis'
                    }
                    return result
        
        # No suitable cache found - execute fresh analysis
        print("❌ Cache MISS - Executing fresh analysis...")
        return self._execute_fresh_analysis(analysis_parameters)
    
    def _execute_fresh_analysis(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute fresh analysis and cache results"""
        print("🚀 Executing fresh ROI analysis...")
        
        start_time = time.time()
        
        # Configure optimized flow
        self.optimized_flow.state = self.state
        
        # Execute the optimized flow steps
        try:
            # Step 1: Load and cache data
            data_reference = self.optimized_flow.load_and_cache_data()
            
            # Cache intermediate result
            self.analysis_cache.cache_intermediate_result(
                request_hash=self.request_hash,
                step_name='data_loading',
                result={'data_reference': data_reference},
                step_metadata={'step': 1, 'description': 'Data loading and caching'}
            )
            
            # Step 2: Analyze profitability
            profitability_reference = self.optimized_flow.analyze_profitability(data_reference)
            
            # Cache intermediate result
            self.analysis_cache.cache_intermediate_result(
                request_hash=self.request_hash,
                step_name='profitability_analysis',
                result={'profitability_reference': profitability_reference},
                step_metadata={'step': 2, 'description': 'Profitability analysis across dimensions'}
            )
            
            # Step 3: Optimize budget
            optimization_reference = self.optimized_flow.optimize_budget(profitability_reference)
            
            # Cache intermediate result
            self.analysis_cache.cache_intermediate_result(
                request_hash=self.request_hash,
                step_name='budget_optimization',
                result={'optimization_reference': optimization_reference},
                step_metadata={'step': 3, 'description': 'Budget optimization and ROI projections'}
            )
            
            # Step 4: Generate final report
            final_result = self.optimized_flow.generate_report(optimization_reference)
            
            execution_time = time.time() - start_time
            
            # Add execution metadata
            final_result['execution_info'] = {
                'execution_time_seconds': execution_time,
                'cache_hit': False,
                'request_hash': self.request_hash,
                'fresh_execution': True
            }
            
            # Cache the complete analysis result
            intermediate_results = {
                'data_loading': {'data_reference': data_reference},
                'profitability_analysis': {'profitability_reference': profitability_reference},
                'budget_optimization': {'optimization_reference': optimization_reference}
            }
            
            cache_file = self.analysis_cache.cache_analysis_result(
                request_hash=self.request_hash,
                analysis_type='roi_analysis',
                data_path=self.state.data_file_path,
                parameters=parameters,
                result=final_result,
                intermediate_results=intermediate_results,
                ttl=7 * 24 * 3600  # 7 days
            )
            
            final_result['cache_info'] = {
                'cache_hit': False,
                'cached_for_future': True,
                'cache_file': cache_file,
                'request_hash': self.request_hash
            }
            
            print(f"✅ Fresh analysis completed and cached in {execution_time:.2f}s")
            
            return final_result
            
        except Exception as e:
            print(f"❌ Analysis execution failed: {e}")
            raise
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        return self.analysis_cache.get_cache_statistics()
    
    def cleanup_cache(self) -> Dict[str, Any]:
        """Clean up expired cache entries"""
        return self.analysis_cache.cleanup_expired_cache()

class CachedFlowRunner:
    """
    Enhanced flow runner with persistent caching capabilities
    """
    
    def __init__(self, use_mem0: bool = True):
        self.analysis_cache = get_analysis_cache()
        self.execution_stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'similar_matches': 0,
            'total_time_saved': 0,
            'total_cost_saved': 0
        }
    
    def run_roi_analysis(self, data_file_path: str,
                        context_strategy: ContextStrategy = ContextStrategy.PROGRESSIVE_PRUNING,
                        force_refresh: bool = False,
                        **kwargs) -> Dict[str, Any]:
        """
        Run ROI analysis with intelligent caching
        
        Args:
            data_file_path: Path to source data
            context_strategy: Context optimization strategy
            force_refresh: Force fresh execution ignoring cache
            **kwargs: Additional parameters
            
        Returns:
            Analysis results with cache information
        """
        self.execution_stats['total_requests'] += 1
        
        # Prepare parameters
        analysis_parameters = {
            'context_strategy': context_strategy.value,
            'force_refresh': force_refresh,
            **kwargs
        }
        
        if force_refresh:
            print("🔄 Force refresh requested - bypassing cache")
            
            # Generate hash for forced execution
            flow = CachedROIFlow()
            flow.state.data_file_path = data_file_path
            
            request_hash = self.analysis_cache.generate_request_hash(
                analysis_type='roi_analysis',
                data_path=data_file_path,
                parameters=analysis_parameters
            )
            
            # Remove existing cache entry if it exists
            try:
                self.analysis_cache._remove_cache_entry(request_hash)
            except:
                pass
            
            # Execute fresh analysis
            result = flow._execute_fresh_analysis(analysis_parameters)
            self.execution_stats['cache_misses'] += 1
            
        else:
            # Normal cached execution
            flow = CachedROIFlow()
            flow.state.data_file_path = data_file_path
            
            result = flow.check_cache_and_execute(analysis_parameters)
            
            # Update statistics
            cache_info = result.get('cache_info', {})
            if cache_info.get('cache_hit'):
                if cache_info.get('similarity_match'):
                    self.execution_stats['similar_matches'] += 1
                else:
                    self.execution_stats['cache_hits'] += 1
                
                # Estimate time and cost saved
                self.execution_stats['total_time_saved'] += 45  # Estimated seconds saved
                self.execution_stats['total_cost_saved'] += 0.0007  # Estimated cost saved
            else:
                self.execution_stats['cache_misses'] += 1
        
        # Add execution statistics to result
        result['execution_stats'] = self.execution_stats.copy()
        result['cache_performance'] = self._calculate_cache_performance()
        
        return result
    
    def _calculate_cache_performance(self) -> Dict[str, Any]:
        """Calculate cache performance metrics"""
        total = self.execution_stats['total_requests']
        if total == 0:
            return {'cache_hit_rate': 0, 'message': 'No requests processed yet'}
        
        cache_hits = self.execution_stats['cache_hits'] + self.execution_stats['similar_matches']
        hit_rate = (cache_hits / total) * 100
        
        return {
            'cache_hit_rate': hit_rate,
            'exact_hits': self.execution_stats['cache_hits'],
            'similar_matches': self.execution_stats['similar_matches'],
            'cache_misses': self.execution_stats['cache_misses'],
            'total_time_saved_minutes': self.execution_stats['total_time_saved'] / 60,
            'total_cost_saved_usd': self.execution_stats['total_cost_saved'],
            'efficiency_rating': 'Excellent' if hit_rate > 80 else 'Good' if hit_rate > 60 else 'Fair' if hit_rate > 40 else 'Poor'
        }
    
    def get_cache_insights(self) -> Dict[str, Any]:
        """Get insights about cache usage and recommendations"""
        cache_stats = self.analysis_cache.get_cache_statistics()
        performance = self._calculate_cache_performance()
        
        insights = {
            'cache_statistics': cache_stats,
            'performance_metrics': performance,
            'recommendations': []
        }
        
        # Generate recommendations
        if performance['cache_hit_rate'] < 50:
            insights['recommendations'].append({
                'type': 'performance',
                'message': 'Low cache hit rate detected',
                'action': 'Consider running similar analyses to build up cache'
            })
        
        if cache_stats['total_size_mb'] > 1000:  # > 1GB
            insights['recommendations'].append({
                'type': 'storage',
                'message': 'Large cache size detected',
                'action': 'Consider running cache cleanup to free space'
            })
        
        if cache_stats['total_entries'] > 100:
            insights['recommendations'].append({
                'type': 'maintenance',
                'message': 'Many cached analyses available',
                'action': 'Cache is working well - consider increasing TTL for frequently used analyses'
            })
        
        return insights
    
    def cleanup_cache(self) -> Dict[str, Any]:
        """Clean up cache and return statistics"""
        return self.analysis_cache.cleanup_expired_cache()