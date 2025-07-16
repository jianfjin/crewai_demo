"""
Parallel Execution Manager for Agent Workflows

This module provides parallel execution capabilities for independent agents,
significantly reducing total execution time by running agents concurrently
when they don't have dependencies on each other.
"""

import asyncio
import time
import threading
from typing import Dict, Any, List, Optional, Tuple, Callable
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ExecutionMode(Enum):
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HYBRID = "hybrid"

@dataclass
class AgentTask:
    """Represents a task to be executed by an agent."""
    agent_id: str
    agent_role: str
    task_name: str
    task_function: Callable
    inputs: Dict[str, Any]
    dependencies: List[str] = None
    priority: int = 1
    estimated_duration: float = 60.0

@dataclass
class ExecutionResult:
    """Result of an agent task execution."""
    agent_id: str
    agent_role: str
    task_name: str
    result: Any
    execution_time: float
    start_time: datetime
    end_time: datetime
    success: bool
    error: Optional[str] = None

class ParallelExecutionManager:
    """
    Manages parallel execution of agent tasks with dependency resolution.
    """
    
    def __init__(self, max_workers: int = 4, enable_async: bool = True):
        """
        Initialize the parallel execution manager.
        
        Args:
            max_workers: Maximum number of concurrent workers
            enable_async: Whether to use async execution
        """
        self.max_workers = max_workers
        self.enable_async = enable_async
        self.execution_history: List[ExecutionResult] = []
        self._lock = threading.RLock()
        
        # Performance metrics
        self.total_sequential_time = 0.0
        self.total_parallel_time = 0.0
        self.time_saved = 0.0
        
        logger.info(f"🚀 ParallelExecutionManager initialized (workers={max_workers}, async={enable_async})")
    
    def analyze_dependencies(self, tasks: List[AgentTask]) -> Dict[str, List[str]]:
        """
        Analyze task dependencies and create execution groups.
        
        Args:
            tasks: List of agent tasks
            
        Returns:
            Dictionary mapping execution phases to agent IDs
        """
        dependency_graph = {}
        for task in tasks:
            dependency_graph[task.agent_id] = task.dependencies or []
        
        # Topological sort to determine execution order
        execution_phases = {}
        visited = set()
        phase = 0
        
        while len(visited) < len(tasks):
            current_phase = []
            
            for task in tasks:
                if task.agent_id in visited:
                    continue
                
                # Check if all dependencies are satisfied
                dependencies_satisfied = all(
                    dep in visited for dep in (task.dependencies or [])
                )
                
                if dependencies_satisfied:
                    current_phase.append(task.agent_id)
            
            if not current_phase:
                # Circular dependency detected, break it
                remaining_tasks = [t for t in tasks if t.agent_id not in visited]
                current_phase = [remaining_tasks[0].agent_id]
                logger.warning(f"⚠️ Circular dependency detected, breaking with {current_phase[0]}")
            
            execution_phases[f"phase_{phase}"] = current_phase
            visited.update(current_phase)
            phase += 1
        
        logger.info(f"📊 Dependency analysis: {len(execution_phases)} phases identified")
        return execution_phases
    
    def execute_tasks_parallel(self, tasks: List[AgentTask], 
                             execution_mode: ExecutionMode = ExecutionMode.HYBRID) -> List[ExecutionResult]:
        """
        Execute tasks with optimal parallelization strategy.
        
        Args:
            tasks: List of agent tasks to execute
            execution_mode: Execution strategy
            
        Returns:
            List of execution results
        """
        start_time = time.time()
        
        if execution_mode == ExecutionMode.SEQUENTIAL:
            results = self._execute_sequential(tasks)
        elif execution_mode == ExecutionMode.PARALLEL:
            results = self._execute_parallel(tasks)
        else:  # HYBRID
            results = self._execute_hybrid(tasks)
        
        total_time = time.time() - start_time
        
        # Calculate performance metrics
        sequential_estimate = sum(task.estimated_duration for task in tasks)
        time_saved = sequential_estimate - total_time
        
        self.total_parallel_time += total_time
        self.time_saved += max(0, time_saved)
        
        logger.info(f"✅ Executed {len(tasks)} tasks in {total_time:.2f}s (saved {time_saved:.2f}s)")
        
        with self._lock:
            self.execution_history.extend(results)
        
        return results
    
    def _execute_sequential(self, tasks: List[AgentTask]) -> List[ExecutionResult]:
        """Execute tasks sequentially."""
        logger.info("🔄 Executing tasks sequentially")
        results = []
        
        for task in tasks:
            result = self._execute_single_task(task)
            results.append(result)
        
        return results
    
    def _execute_parallel(self, tasks: List[AgentTask]) -> List[ExecutionResult]:
        """Execute all tasks in parallel (ignoring dependencies)."""
        logger.info(f"⚡ Executing {len(tasks)} tasks in parallel")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_task = {
                executor.submit(self._execute_single_task, task): task 
                for task in tasks
            }
            
            results = []
            for future in as_completed(future_to_task):
                result = future.result()
                results.append(result)
        
        # Sort results by original task order
        task_order = {task.agent_id: i for i, task in enumerate(tasks)}
        results.sort(key=lambda r: task_order.get(r.agent_id, 999))
        
        return results
    
    def _execute_hybrid(self, tasks: List[AgentTask]) -> List[ExecutionResult]:
        """Execute tasks with dependency-aware parallelization."""
        logger.info(f"🔀 Executing {len(tasks)} tasks with hybrid strategy")
        
        execution_phases = self.analyze_dependencies(tasks)
        task_map = {task.agent_id: task for task in tasks}
        all_results = []
        
        for phase_name, agent_ids in execution_phases.items():
            phase_tasks = [task_map[agent_id] for agent_id in agent_ids]
            
            if len(phase_tasks) == 1:
                # Single task - execute directly
                logger.info(f"📍 {phase_name}: Executing {phase_tasks[0].agent_role}")
                result = self._execute_single_task(phase_tasks[0])
                all_results.append(result)
            else:
                # Multiple independent tasks - execute in parallel
                logger.info(f"⚡ {phase_name}: Executing {len(phase_tasks)} tasks in parallel")
                
                with ThreadPoolExecutor(max_workers=min(self.max_workers, len(phase_tasks))) as executor:
                    future_to_task = {
                        executor.submit(self._execute_single_task, task): task 
                        for task in phase_tasks
                    }
                    
                    phase_results = []
                    for future in as_completed(future_to_task):
                        result = future.result()
                        phase_results.append(result)
                    
                    all_results.extend(phase_results)
        
        return all_results
    
    def _execute_single_task(self, task: AgentTask) -> ExecutionResult:
        """Execute a single agent task."""
        start_time = datetime.now()
        execution_start = time.time()
        
        try:
            logger.info(f"🤖 Starting {task.agent_role}: {task.task_name}")
            
            # Execute the task function
            result = task.task_function(task.inputs)
            
            execution_time = time.time() - execution_start
            end_time = datetime.now()
            
            logger.info(f"✅ Completed {task.agent_role} in {execution_time:.2f}s")
            
            return ExecutionResult(
                agent_id=task.agent_id,
                agent_role=task.agent_role,
                task_name=task.task_name,
                result=result,
                execution_time=execution_time,
                start_time=start_time,
                end_time=end_time,
                success=True
            )
            
        except Exception as e:
            execution_time = time.time() - execution_start
            end_time = datetime.now()
            
            logger.error(f"❌ Failed {task.agent_role}: {str(e)}")
            
            return ExecutionResult(
                agent_id=task.agent_id,
                agent_role=task.agent_role,
                task_name=task.task_name,
                result=None,
                execution_time=execution_time,
                start_time=start_time,
                end_time=end_time,
                success=False,
                error=str(e)
            )
    
    async def execute_tasks_async(self, tasks: List[AgentTask]) -> List[ExecutionResult]:
        """Execute tasks asynchronously for better performance."""
        if not self.enable_async:
            return self.execute_tasks_parallel(tasks, ExecutionMode.HYBRID)
        
        logger.info(f"🚀 Executing {len(tasks)} tasks asynchronously")
        
        execution_phases = self.analyze_dependencies(tasks)
        task_map = {task.agent_id: task for task in tasks}
        all_results = []
        
        for phase_name, agent_ids in execution_phases.items():
            phase_tasks = [task_map[agent_id] for agent_id in agent_ids]
            
            if len(phase_tasks) == 1:
                # Single task
                result = await self._execute_single_task_async(phase_tasks[0])
                all_results.append(result)
            else:
                # Multiple tasks in parallel
                logger.info(f"⚡ {phase_name}: Executing {len(phase_tasks)} tasks async")
                
                async_tasks = [
                    self._execute_single_task_async(task) 
                    for task in phase_tasks
                ]
                
                phase_results = await asyncio.gather(*async_tasks, return_exceptions=True)
                
                # Handle exceptions
                for i, result in enumerate(phase_results):
                    if isinstance(result, Exception):
                        task = phase_tasks[i]
                        result = ExecutionResult(
                            agent_id=task.agent_id,
                            agent_role=task.agent_role,
                            task_name=task.task_name,
                            result=None,
                            execution_time=0.0,
                            start_time=datetime.now(),
                            end_time=datetime.now(),
                            success=False,
                            error=str(result)
                        )
                    all_results.append(result)
        
        return all_results
    
    async def _execute_single_task_async(self, task: AgentTask) -> ExecutionResult:
        """Execute a single task asynchronously."""
        start_time = datetime.now()
        execution_start = time.time()
        
        try:
            logger.info(f"🤖 Starting async {task.agent_role}: {task.task_name}")
            
            # Run the task in a thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, task.task_function, task.inputs)
            
            execution_time = time.time() - execution_start
            end_time = datetime.now()
            
            logger.info(f"✅ Completed async {task.agent_role} in {execution_time:.2f}s")
            
            return ExecutionResult(
                agent_id=task.agent_id,
                agent_role=task.agent_role,
                task_name=task.task_name,
                result=result,
                execution_time=execution_time,
                start_time=start_time,
                end_time=end_time,
                success=True
            )
            
        except Exception as e:
            execution_time = time.time() - execution_start
            end_time = datetime.now()
            
            logger.error(f"❌ Failed async {task.agent_role}: {str(e)}")
            
            return ExecutionResult(
                agent_id=task.agent_id,
                agent_role=task.agent_role,
                task_name=task.task_name,
                result=None,
                execution_time=execution_time,
                start_time=start_time,
                end_time=end_time,
                success=False,
                error=str(e)
            )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for parallel execution."""
        with self._lock:
            successful_executions = [r for r in self.execution_history if r.success]
            failed_executions = [r for r in self.execution_history if not r.success]
            
            if successful_executions:
                avg_execution_time = sum(r.execution_time for r in successful_executions) / len(successful_executions)
                total_execution_time = sum(r.execution_time for r in successful_executions)
            else:
                avg_execution_time = 0
                total_execution_time = 0
            
            stats = {
                'total_tasks_executed': len(self.execution_history),
                'successful_tasks': len(successful_executions),
                'failed_tasks': len(failed_executions),
                'success_rate_percentage': (len(successful_executions) / len(self.execution_history) * 100) if self.execution_history else 0,
                'average_execution_time_seconds': round(avg_execution_time, 2),
                'total_execution_time_seconds': round(total_execution_time, 2),
                'total_time_saved_seconds': round(self.time_saved, 2),
                'parallel_efficiency': round((self.time_saved / max(self.total_parallel_time, 1)) * 100, 2)
            }
            
            return stats

class OptimizedAgentFlow:
    """
    Optimized flow for your specific 4-agent workflow:
    market_research_analyst, competitive_analyst, brand_performance_specialist, campaign_optimizer
    """
    
    def __init__(self, parallel_manager: ParallelExecutionManager):
        self.parallel_manager = parallel_manager
        
        # Define agent dependencies for your workflow
        self.agent_dependencies = {
            'market_research_analyst': [],  # Independent
            'competitive_analyst': [],  # Independent  
            'brand_performance_specialist': ['market_research_analyst'],  # Depends on market research
            'campaign_optimizer': ['market_research_analyst', 'competitive_analyst', 'brand_performance_specialist']  # Depends on all others
        }
    
    def create_optimized_tasks(self, crew_instance, inputs: Dict[str, Any]) -> List[AgentTask]:
        """Create optimized task list for your 4-agent workflow."""
        tasks = []
        
        # Phase 1: Independent agents (can run in parallel)
        tasks.append(AgentTask(
            agent_id='market_research_analyst',
            agent_role='market_research_analyst',
            task_name='market_research',
            task_function=lambda inp: self._execute_agent_task(crew_instance, 'market_research_analyst', inp),
            inputs=inputs,
            dependencies=[],
            priority=1,
            estimated_duration=45.0
        ))
        
        tasks.append(AgentTask(
            agent_id='competitive_analyst',
            agent_role='competitive_analyst', 
            task_name='competitive_analysis',
            task_function=lambda inp: self._execute_agent_task(crew_instance, 'competitive_analyst', inp),
            inputs=inputs,
            dependencies=[],
            priority=1,
            estimated_duration=40.0
        ))
        
        # Phase 2: Brand performance (depends on market research)
        tasks.append(AgentTask(
            agent_id='brand_performance_specialist',
            agent_role='brand_performance_specialist',
            task_name='brand_performance',
            task_function=lambda inp: self._execute_agent_task(crew_instance, 'brand_performance_specialist', inp),
            inputs=inputs,
            dependencies=['market_research_analyst'],
            priority=2,
            estimated_duration=35.0
        ))
        
        # Phase 3: Campaign optimization (depends on all others)
        tasks.append(AgentTask(
            agent_id='campaign_optimizer',
            agent_role='campaign_optimizer',
            task_name='campaign_optimization',
            task_function=lambda inp: self._execute_agent_task(crew_instance, 'campaign_optimizer', inp),
            inputs=inputs,
            dependencies=['market_research_analyst', 'competitive_analyst', 'brand_performance_specialist'],
            priority=3,
            estimated_duration=30.0
        ))
        
        return tasks
    
    def _execute_agent_task(self, crew_instance, agent_role: str, inputs: Dict[str, Any]) -> Any:
        """Execute a specific agent task."""
        # This would integrate with your existing crew execution logic
        # For now, return a placeholder that integrates with your crew system
        return f"Result from {agent_role} with inputs: {inputs}"
    
    async def execute_optimized_workflow(self, crew_instance, inputs: Dict[str, Any]) -> List[ExecutionResult]:
        """Execute the optimized 4-agent workflow."""
        logger.info("🚀 Starting optimized 4-agent workflow")
        
        tasks = self.create_optimized_tasks(crew_instance, inputs)
        results = await self.parallel_manager.execute_tasks_async(tasks)
        
        # Log performance improvement
        total_time = sum(r.execution_time for r in results if r.success)
        sequential_estimate = sum(t.estimated_duration for t in tasks)
        improvement = ((sequential_estimate - total_time) / sequential_estimate * 100) if sequential_estimate > 0 else 0
        
        logger.info(f"✅ Workflow completed: {improvement:.1f}% faster than sequential execution")
        
        return results

def create_parallel_flow(max_workers: int = 4) -> Tuple[ParallelExecutionManager, OptimizedAgentFlow]:
    """Create a parallel execution flow for the marketing research workflow."""
    parallel_manager = ParallelExecutionManager(max_workers=max_workers)
    optimized_flow = OptimizedAgentFlow(parallel_manager)
    
    return parallel_manager, optimized_flow