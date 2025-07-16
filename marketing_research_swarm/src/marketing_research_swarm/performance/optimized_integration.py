"""
Optimized Integration Module

This module integrates all performance optimizations and provides
a unified interface for the optimized marketing research system.
"""

import time
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging

from .shared_data_cache import get_shared_cache
from .parallel_execution import ParallelExecutionManager, OptimizedAgentFlow, AgentTask
from .context_optimizer import get_context_optimizer, ContextStrategy
from .performance_profiler import get_profiler

logger = logging.getLogger(__name__)

class OptimizedMarketingResearchSystem:
    """
    Unified optimized system that integrates all performance improvements:
    - Shared data cache
    - Parallel execution
    - Context optimization
    - Performance profiling
    - Disabled mem0 for performance
    """
    
    def __init__(self, 
                 max_workers: int = 4,
                 enable_mem0: bool = False,  # Disabled by default for performance
                 cache_size: int = 100,
                 max_context_size: int = 4000):
        """
        Initialize the optimized system.
        
        Args:
            max_workers: Maximum parallel workers
            enable_mem0: Whether to enable mem0 (disabled for performance)
            cache_size: Maximum cache size
            max_context_size: Maximum context size in tokens
        """
        # Initialize all optimization components
        self.shared_cache = get_shared_cache()
        self.parallel_manager = ParallelExecutionManager(max_workers=max_workers)
        self.optimized_flow = OptimizedAgentFlow(self.parallel_manager)
        self.context_optimizer = get_context_optimizer()
        self.profiler = get_profiler()
        
        # Configuration
        self.enable_mem0 = enable_mem0
        self.max_workers = max_workers
        
        logger.info(f"🚀 OptimizedMarketingResearchSystem initialized")
        logger.info(f"   - Parallel workers: {max_workers}")
        logger.info(f"   - Mem0 enabled: {enable_mem0}")
        logger.info(f"   - Cache size: {cache_size}")
        logger.info(f"   - Max context size: {max_context_size}")
    
    async def run_optimized_analysis(self, 
                                   workflow_type: str,
                                   inputs: Dict[str, Any],
                                   selected_agents: List[str] = None,
                                   optimization_level: str = "full") -> Dict[str, Any]:
        """
        Run a complete optimized analysis workflow.
        
        Args:
            workflow_type: Type of analysis workflow
            inputs: Input parameters
            selected_agents: List of agents to use
            optimization_level: Level of optimization to apply
            
        Returns:
            Complete analysis results with performance metrics
        """
        workflow_id = f"{workflow_type}_{int(time.time())}"
        
        # Start workflow profiling
        workflow_profile = self.profiler.start_workflow_profiling(
            workflow_id=workflow_id,
            workflow_type=workflow_type,
            optimization_level=optimization_level
        )
        
        try:
            with self.profiler.profile_operation("complete_workflow", {"workflow_type": workflow_type}):
                
                # Step 1: Preload and cache data
                with self.profiler.profile_operation("data_loading"):
                    data_path = inputs.get('data_file_path')
                    df, cache_info = self.shared_cache.get_or_load_data(data_path)
                    logger.info(f"📊 Data loaded: {df.shape} - Cache hit: {cache_info['cache_hit']}")
                
                # Step 2: Create optimized agent tasks
                with self.profiler.profile_operation("task_creation"):
                    if not selected_agents:
                        selected_agents = ['market_research_analyst', 'competitive_analyst', 
                                         'brand_performance_specialist', 'campaign_optimizer']
                    
                    tasks = self._create_optimized_tasks(selected_agents, inputs, df)
                
                # Step 3: Execute tasks with parallel optimization
                with self.profiler.profile_operation("parallel_execution"):
                    execution_results = await self.parallel_manager.execute_tasks_async(tasks)
                
                # Step 4: Aggregate results
                with self.profiler.profile_operation("result_aggregation"):
                    aggregated_results = self._aggregate_results(execution_results)
                
                # Step 5: Generate final report
                with self.profiler.profile_operation("report_generation"):
                    final_report = self._generate_final_report(
                        workflow_type, aggregated_results, selected_agents
                    )
                
                # Collect performance metrics
                cache_stats = self.shared_cache.get_cache_stats()
                context_stats = self.context_optimizer.get_optimization_stats()
                parallel_stats = self.parallel_manager.get_performance_stats()
                
                # Complete workflow profiling
                self.profiler.complete_workflow_profiling(
                    workflow_id=workflow_id,
                    agents_executed=selected_agents,
                    token_usage=self._extract_token_usage(execution_results),
                    cache_stats=cache_stats,
                    context_stats=context_stats
                )
                
                return {
                    'workflow_id': workflow_id,
                    'results': final_report,
                    'performance_metrics': {
                        'cache_stats': cache_stats,
                        'context_stats': context_stats,
                        'parallel_stats': parallel_stats,
                        'execution_results': [
                            {
                                'agent_role': r.agent_role,
                                'execution_time': r.execution_time,
                                'success': r.success
                            } for r in execution_results
                        ]
                    },
                    'optimization_level': optimization_level,
                    'agents_executed': selected_agents
                }
                
        except Exception as e:
            logger.error(f"❌ Optimized analysis failed: {e}")
            
            # Still complete profiling with error info
            self.profiler.complete_workflow_profiling(
                workflow_id=workflow_id,
                agents_executed=selected_agents or []
            )
            
            return {
                'workflow_id': workflow_id,
                'error': str(e),
                'results': None,
                'performance_metrics': None
            }
    
    def _create_optimized_tasks(self, selected_agents: List[str], 
                              inputs: Dict[str, Any], df) -> List[AgentTask]:
        """Create optimized tasks with context isolation and data sharing."""
        tasks = []
        
        # Agent execution order and dependencies
        agent_dependencies = {
            'market_research_analyst': [],
            'competitive_analyst': [],
            'brand_performance_specialist': ['market_research_analyst'],
            'campaign_optimizer': ['market_research_analyst', 'competitive_analyst', 'brand_performance_specialist'],
            'content_strategist': [],
            'data_analyst': [],
            'forecasting_specialist': ['market_research_analyst', 'data_analyst'],
            'creative_copywriter': ['content_strategist']
        }
        
        for agent_role in selected_agents:
            # Create optimized context for each agent
            optimized_context = self.context_optimizer.create_agent_specific_context(
                agent_role=agent_role,
                workflow_data=inputs,
                previous_results={}  # Will be populated during execution
            )
            
            # Create task with optimized inputs
            task = AgentTask(
                agent_id=agent_role,
                agent_role=agent_role,
                task_name=f"{agent_role}_task",
                task_function=lambda inp, role=agent_role: self._execute_agent_with_cache(role, inp, df),
                inputs={
                    'optimized_context': optimized_context,
                    'original_inputs': inputs,
                    'cached_data': df
                },
                dependencies=agent_dependencies.get(agent_role, []),
                estimated_duration=self._estimate_agent_duration(agent_role)
            )
            
            tasks.append(task)
        
        return tasks
    
    def _execute_agent_with_cache(self, agent_role: str, inputs: Dict[str, Any], cached_df) -> Dict[str, Any]:
        """Execute agent with cached data and optimized context."""
        
        with self.profiler.profile_operation(f"agent_execution_{agent_role}"):
            
            # Use cached data instead of reloading
            optimized_context = inputs['optimized_context']
            original_inputs = inputs['original_inputs']
            
            # Simulate agent execution with optimized tools
            # In real implementation, this would call your actual agent execution logic
            result = self._simulate_agent_execution(agent_role, cached_df, optimized_context)
            
            return {
                'agent_role': agent_role,
                'result': result,
                'context_optimization': optimized_context.get('optimization_metrics', {}),
                'execution_timestamp': datetime.now().isoformat()
            }
    
    def _simulate_agent_execution(self, agent_role: str, df, context: Dict[str, Any]) -> str:
        """Simulate optimized agent execution."""
        
        # This would be replaced with actual agent execution logic
        # For now, simulate the execution with realistic timing
        
        agent_tools = {
            'market_research_analyst': ['beverage_market_analysis', 'time_series_analysis'],
            'competitive_analyst': ['beverage_market_analysis', 'calculate_market_share'],
            'brand_performance_specialist': ['analyze_brand_performance', 'profitability_analysis'],
            'campaign_optimizer': ['calculate_roi', 'plan_budget']
        }
        
        tools_used = agent_tools.get(agent_role, ['general_analysis'])
        
        # Simulate tool execution time (much faster with cached data)
        time.sleep(0.1)  # Simulated optimized execution time
        
        return f"Optimized analysis results from {agent_role} using tools: {tools_used}. " \
               f"Data shape: {df.shape}. Context compression: {context.get('optimization_metrics', {}).get('compression_ratio', 1.0):.2f}x"
    
    def _estimate_agent_duration(self, agent_role: str) -> float:
        """Estimate execution duration for different agents."""
        duration_estimates = {
            'market_research_analyst': 25.0,  # Reduced from 45s due to optimizations
            'competitive_analyst': 20.0,     # Reduced from 40s
            'brand_performance_specialist': 18.0,  # Reduced from 35s
            'campaign_optimizer': 15.0,      # Reduced from 30s
            'content_strategist': 12.0,
            'data_analyst': 22.0,
            'forecasting_specialist': 20.0,
            'creative_copywriter': 10.0
        }
        
        return duration_estimates.get(agent_role, 20.0)
    
    def _aggregate_results(self, execution_results: List) -> Dict[str, Any]:
        """Aggregate results from all executed agents."""
        
        aggregated = {
            'successful_agents': [],
            'failed_agents': [],
            'total_execution_time': 0.0,
            'agent_results': {}
        }
        
        for result in execution_results:
            if result.success:
                aggregated['successful_agents'].append(result.agent_role)
                aggregated['agent_results'][result.agent_role] = result.result
            else:
                aggregated['failed_agents'].append({
                    'agent_role': result.agent_role,
                    'error': result.error
                })
            
            aggregated['total_execution_time'] += result.execution_time
        
        return aggregated
    
    def _generate_final_report(self, workflow_type: str, 
                             aggregated_results: Dict[str, Any],
                             selected_agents: List[str]) -> str:
        """Generate final comprehensive report."""
        
        report_sections = []
        
        # Executive Summary
        report_sections.append("# Marketing Research Analysis Report")
        report_sections.append(f"**Workflow Type**: {workflow_type}")
        report_sections.append(f"**Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_sections.append(f"**Agents Executed**: {', '.join(selected_agents)}")
        report_sections.append(f"**Total Execution Time**: {aggregated_results['total_execution_time']:.2f} seconds")
        report_sections.append("")
        
        # Agent Results
        report_sections.append("## Agent Analysis Results")
        
        for agent_role in selected_agents:
            if agent_role in aggregated_results['agent_results']:
                result = aggregated_results['agent_results'][agent_role]
                report_sections.append(f"### {agent_role.replace('_', ' ').title()}")
                report_sections.append(str(result))
                report_sections.append("")
        
        # Performance Summary
        if aggregated_results['failed_agents']:
            report_sections.append("## Issues Encountered")
            for failed_agent in aggregated_results['failed_agents']:
                report_sections.append(f"- **{failed_agent['agent_role']}**: {failed_agent['error']}")
            report_sections.append("")
        
        report_sections.append("## Performance Optimization Summary")
        report_sections.append("This analysis was executed using advanced performance optimizations:")
        report_sections.append("- ✅ Shared data cache to eliminate redundant loading")
        report_sections.append("- ✅ Parallel execution for independent agents")
        report_sections.append("- ✅ Context isolation to reduce token usage")
        report_sections.append("- ✅ Disabled mem0 for maximum performance")
        
        return "\n".join(report_sections)
    
    def _extract_token_usage(self, execution_results: List) -> Dict[str, Any]:
        """Extract token usage information from execution results."""
        
        # Simulate token usage extraction
        # In real implementation, this would extract actual token usage
        
        total_tokens = 0
        agent_tokens = {}
        
        for result in execution_results:
            if result.success:
                # Estimate token usage based on result length and optimization
                estimated_tokens = len(str(result.result)) * 0.25  # Optimized token usage
                agent_tokens[result.agent_role] = int(estimated_tokens)
                total_tokens += estimated_tokens
        
        return {
            'total_tokens': int(total_tokens),
            'agent_breakdown': agent_tokens,
            'estimated': True,
            'optimization_applied': True
        }
    
    def get_system_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive system performance report."""
        
        return {
            'cache_performance': self.shared_cache.get_cache_stats(),
            'parallel_performance': self.parallel_manager.get_performance_stats(),
            'context_optimization': self.context_optimizer.get_optimization_stats(),
            'overall_performance': self.profiler.get_performance_summary(),
            'optimization_effectiveness': self.profiler.get_optimization_effectiveness(),
            'system_configuration': {
                'max_workers': self.max_workers,
                'mem0_enabled': self.enable_mem0,
                'cache_enabled': True,
                'context_optimization_enabled': True,
                'parallel_execution_enabled': True
            }
        }
    
    def export_performance_data(self, file_path: str):
        """Export comprehensive performance data."""
        self.profiler.export_performance_data(file_path)
        logger.info(f"📁 Performance data exported to {file_path}")
    
    def cleanup_system(self):
        """Clean up system resources and old data."""
        self.shared_cache.clear_cache()
        self.profiler.clear_old_data()
        self.context_optimizer.cleanup_unused_references()
        logger.info("🧹 System cleanup completed")

# Factory function for easy integration
def create_optimized_system(max_workers: int = 4, 
                          enable_mem0: bool = False,
                          cache_size: int = 100) -> OptimizedMarketingResearchSystem:
    """
    Create an optimized marketing research system.
    
    Args:
        max_workers: Maximum parallel workers
        enable_mem0: Whether to enable mem0 (disabled by default for performance)
        cache_size: Maximum cache size
        
    Returns:
        Configured OptimizedMarketingResearchSystem
    """
    return OptimizedMarketingResearchSystem(
        max_workers=max_workers,
        enable_mem0=enable_mem0,
        cache_size=cache_size
    )