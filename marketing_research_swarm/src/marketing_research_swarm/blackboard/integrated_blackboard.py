"""
Integrated Blackboard System for Marketing Research Swarm

This module provides a unified blackboard system that integrates with existing
AdvancedContextManager, MarketingMemoryManager, and AnalysisCacheManager
to provide maximum token efficiency through shared state management.

Key Features:
- Seamless integration with existing managers
- Centralized state with distributed caching
- Event-driven workflow coordination
- Token usage optimization through state sharing
- Memory-aware context management
"""

import json
import time
import threading
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import uuid
from datetime import datetime, timedelta
import logging

# Import existing managers for integration
try:
    from ..context.context_manager import AdvancedContextManager
    from ..memory.mem0_integration import MarketingMemoryManager
    from ..persistence.analysis_cache import AnalysisCacheManager, get_analysis_cache
    from ..utils.token_tracker import TokenTracker
    from .shared_state_manager import SharedStateManager, WorkflowState, TaskState, WorkflowStatus, TaskStatus
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")
    # Fallback for testing
    AdvancedContextManager = None
    MarketingMemoryManager = None
    AnalysisCacheManager = None
    get_analysis_cache = None
    TokenTracker = None
    SharedStateManager = None


@dataclass
class IntegratedWorkflowContext:
    """Enhanced workflow context that integrates all managers"""
    workflow_id: str
    context_data: Dict[str, Any]
    memory_data: Dict[str, Any]
    cached_results: Dict[str, Any]
    shared_state: Dict[str, Any]
    token_usage: Dict[str, int]
    created_at: datetime
    last_updated: datetime


class IntegratedBlackboardSystem:
    """
    Unified blackboard system that integrates with existing managers
    to provide maximum token efficiency through coordinated state management.
    """
    
    def __init__(self, 
                 enable_context_manager: bool = True,
                 enable_memory_manager: bool = True,
                 enable_cache_manager: bool = True,
                 enable_token_tracking: bool = True):
        """
        Initialize the integrated blackboard system.
        
        Args:
            enable_context_manager: Whether to use AdvancedContextManager
            enable_memory_manager: Whether to use MarketingMemoryManager
            enable_cache_manager: Whether to use AnalysisCacheManager
            enable_token_tracking: Whether to use TokenTracker
        """
        self.logger = logging.getLogger(__name__)
        self._lock = threading.RLock()
        
        # Initialize core blackboard
        if SharedStateManager:
            self.shared_state_manager = SharedStateManager()
        else:
            self.shared_state_manager = None
            
        # Initialize integrated managers
        self.context_manager = None
        self.memory_manager = None
        self.cache_manager = None
        self.token_tracker = None
        
        if enable_context_manager and AdvancedContextManager:
            try:
                self.context_manager = AdvancedContextManager()
                self.logger.info("AdvancedContextManager initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize AdvancedContextManager: {e}")
                
        if enable_memory_manager and MarketingMemoryManager:
            try:
                self.memory_manager = MarketingMemoryManager()
                self.logger.info("MarketingMemoryManager initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize MarketingMemoryManager: {e}")
                
        if enable_cache_manager and get_analysis_cache:
            try:
                self.cache_manager = get_analysis_cache()
                self.logger.info("AnalysisCacheManager initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize AnalysisCacheManager: {e}")
                
        if enable_token_tracking and TokenTracker:
            try:
                self.token_tracker = TokenTracker()
                self.logger.info("TokenTracker initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize TokenTracker: {e}")
        
        # Integrated workflow contexts
        self.workflow_contexts: Dict[str, IntegratedWorkflowContext] = {}
        
        # Event handlers for cross-manager coordination
        self.event_handlers: Dict[str, List[Callable]] = {}
        
    def create_integrated_workflow(self, 
                                 workflow_type: str,
                                 initial_data: Dict[str, Any],
                                 workflow_config: Optional[Dict[str, Any]] = None) -> str:
        """
        Create a new integrated workflow that coordinates all managers.
        
        Args:
            workflow_type: Type of workflow (e.g., 'roi_analysis', 'brand_performance')
            initial_data: Initial data for the workflow
            workflow_config: Optional configuration for the workflow
            
        Returns:
            workflow_id: Unique identifier for the workflow
        """
        with self._lock:
            workflow_id = str(uuid.uuid4())
            
            # Initialize context data
            context_data = {}
            if self.context_manager:
                try:
                    # Create optimized context for the workflow
                    context_data = self.context_manager.create_context(
                        context_type=workflow_type,
                        initial_data=initial_data
                    )
                except Exception as e:
                    self.logger.warning(f"Context manager error: {e}")
            
            # Initialize memory data
            memory_data = {}
            if self.memory_manager:
                try:
                    # Store workflow context in memory
                    memory_key = f"workflow_{workflow_id}"
                    self.memory_manager.store_context(memory_key, {
                        'workflow_type': workflow_type,
                        'initial_data': initial_data,
                        'created_at': datetime.now().isoformat()
                    })
                    memory_data = {'memory_key': memory_key}
                except Exception as e:
                    self.logger.warning(f"Memory manager error: {e}")
            
            # Check for cached results
            cached_results = {}
            if self.cache_manager:
                try:
                    # Look for relevant cached analysis
                    cache_key = self._generate_cache_key(workflow_type, initial_data)
                    cached_result = self.cache_manager.get_cached_analysis(cache_key)
                    if cached_result:
                        cached_results = {'cache_key': cache_key, 'data': cached_result}
                except Exception as e:
                    self.logger.warning(f"Cache manager error: {e}")
            
            # Initialize shared state
            shared_state = {}
            if self.shared_state_manager:
                try:
                    # Create workflow in shared state manager
                    shared_workflow_id = self.shared_state_manager.create_workflow(
                        workflow_type=workflow_type,
                        initial_data=initial_data
                    )
                    shared_state = {'shared_workflow_id': shared_workflow_id}
                except Exception as e:
                    self.logger.warning(f"Shared state manager error: {e}")
            
            # Initialize token tracking
            token_usage = {}
            if self.token_tracker:
                try:
                    self.token_tracker.start_tracking(workflow_id)
                    token_usage = {'tracking_started': True}
                except Exception as e:
                    self.logger.warning(f"Token tracker error: {e}")
            
            # Create integrated workflow context
            workflow_context = IntegratedWorkflowContext(
                workflow_id=workflow_id,
                context_data=context_data,
                memory_data=memory_data,
                cached_results=cached_results,
                shared_state=shared_state,
                token_usage=token_usage,
                created_at=datetime.now(),
                last_updated=datetime.now()
            )
            
            self.workflow_contexts[workflow_id] = workflow_context
            
            # Emit workflow created event
            self._emit_event('workflow_created', {
                'workflow_id': workflow_id,
                'workflow_type': workflow_type,
                'has_cached_results': bool(cached_results)
            })
            
            self.logger.info(f"Created integrated workflow {workflow_id} of type {workflow_type}")
            return workflow_id
    
    def get_optimized_context(self, workflow_id: str, agent_role: str) -> Dict[str, Any]:
        """
        Get optimized context for an agent, leveraging all available managers.
        
        Args:
            workflow_id: Workflow identifier
            agent_role: Role of the requesting agent
            
        Returns:
            Optimized context data for the agent
        """
        with self._lock:
            if workflow_id not in self.workflow_contexts:
                return {}
            
            workflow_context = self.workflow_contexts[workflow_id]
            optimized_context = {}
            
            # Get context-specific data
            if self.context_manager and workflow_context.context_data:
                try:
                    agent_context = self.context_manager.get_agent_context(
                        agent_role=agent_role,
                        context_data=workflow_context.context_data
                    )
                    optimized_context.update(agent_context)
                except Exception as e:
                    self.logger.warning(f"Context manager error for agent {agent_role}: {e}")
            
            # Get relevant memories
            if self.memory_manager and workflow_context.memory_data:
                try:
                    memory_key = workflow_context.memory_data.get('memory_key')
                    if memory_key:
                        relevant_memories = self.memory_manager.get_relevant_memories(
                            query=f"agent_role:{agent_role}",
                            limit=5
                        )
                        if relevant_memories:
                            optimized_context['relevant_memories'] = relevant_memories
                except Exception as e:
                    self.logger.warning(f"Memory manager error for agent {agent_role}: {e}")
            
            # Include cached results if available
            if workflow_context.cached_results:
                optimized_context['cached_results'] = workflow_context.cached_results
            
            # Get shared state data
            if self.shared_state_manager and workflow_context.shared_state:
                try:
                    shared_workflow_id = workflow_context.shared_state.get('shared_workflow_id')
                    if shared_workflow_id:
                        shared_data = self.shared_state_manager.get_workflow_state(shared_workflow_id)
                        if shared_data:
                            optimized_context['shared_workflow_state'] = shared_data
                except Exception as e:
                    self.logger.warning(f"Shared state manager error for agent {agent_role}: {e}")
            
            return optimized_context
    
    def update_agent_results(self, 
                           workflow_id: str, 
                           agent_role: str, 
                           results: Dict[str, Any],
                           token_usage: Optional[Dict[str, int]] = None) -> bool:
        """
        Update workflow with agent results, coordinating across all managers.
        
        Args:
            workflow_id: Workflow identifier
            agent_role: Role of the agent providing results
            results: Results from the agent
            token_usage: Optional token usage information
            
        Returns:
            Success status
        """
        with self._lock:
            if workflow_id not in self.workflow_contexts:
                return False
            
            workflow_context = self.workflow_contexts[workflow_id]
            
            # Update context manager
            if self.context_manager:
                try:
                    self.context_manager.update_context(
                        context_id=workflow_id,
                        agent_role=agent_role,
                        new_data=results
                    )
                except Exception as e:
                    self.logger.warning(f"Context manager update error: {e}")
            
            # Store in memory
            if self.memory_manager:
                try:
                    memory_key = f"agent_result_{workflow_id}_{agent_role}"
                    self.memory_manager.store_context(memory_key, {
                        'agent_role': agent_role,
                        'results': results,
                        'timestamp': datetime.now().isoformat()
                    })
                except Exception as e:
                    self.logger.warning(f"Memory manager store error: {e}")
            
            # Cache results if appropriate
            if self.cache_manager and self._should_cache_results(agent_role, results):
                try:
                    cache_key = f"{workflow_id}_{agent_role}_results"
                    self.cache_manager.cache_analysis(cache_key, results)
                except Exception as e:
                    self.logger.warning(f"Cache manager store error: {e}")
            
            # Update shared state
            if self.shared_state_manager and workflow_context.shared_state:
                try:
                    shared_workflow_id = workflow_context.shared_state.get('shared_workflow_id')
                    if shared_workflow_id:
                        self.shared_state_manager.update_task_result(
                            workflow_id=shared_workflow_id,
                            task_id=f"{agent_role}_task",
                            result=results
                        )
                except Exception as e:
                    self.logger.warning(f"Shared state update error: {e}")
            
            # Track token usage
            if self.token_tracker and token_usage:
                try:
                    self.token_tracker.track_usage(workflow_id, token_usage)
                except Exception as e:
                    self.logger.warning(f"Token tracker error: {e}")
            
            # Update workflow context
            workflow_context.last_updated = datetime.now()
            
            # Emit results updated event
            self._emit_event('agent_results_updated', {
                'workflow_id': workflow_id,
                'agent_role': agent_role,
                'results_size': len(str(results))
            })
            
            return True
    
    def get_workflow_summary(self, workflow_id: str) -> Dict[str, Any]:
        """
        Get a comprehensive summary of the workflow state across all managers.
        
        Args:
            workflow_id: Workflow identifier
            
        Returns:
            Comprehensive workflow summary
        """
        with self._lock:
            if workflow_id not in self.workflow_contexts:
                return {}
            
            workflow_context = self.workflow_contexts[workflow_id]
            summary = {
                'workflow_id': workflow_id,
                'created_at': workflow_context.created_at.isoformat(),
                'last_updated': workflow_context.last_updated.isoformat(),
                'managers_status': {}
            }
            
            # Context manager status
            if self.context_manager:
                try:
                    context_stats = self.context_manager.get_context_stats()
                    summary['managers_status']['context_manager'] = {
                        'active': True,
                        'stats': context_stats
                    }
                except Exception as e:
                    summary['managers_status']['context_manager'] = {
                        'active': False,
                        'error': str(e)
                    }
            
            # Memory manager status
            if self.memory_manager:
                try:
                    memory_stats = self.memory_manager.get_memory_stats()
                    summary['managers_status']['memory_manager'] = {
                        'active': True,
                        'stats': memory_stats
                    }
                except Exception as e:
                    summary['managers_status']['memory_manager'] = {
                        'active': False,
                        'error': str(e)
                    }
            
            # Cache manager status
            if self.cache_manager:
                try:
                    cache_stats = self.cache_manager.get_cache_stats()
                    summary['managers_status']['cache_manager'] = {
                        'active': True,
                        'stats': cache_stats
                    }
                except Exception as e:
                    summary['managers_status']['cache_manager'] = {
                        'active': False,
                        'error': str(e)
                    }
            
            # Token tracker status
            if self.token_tracker:
                try:
                    token_stats = self.token_tracker.get_usage_summary(workflow_id)
                    summary['managers_status']['token_tracker'] = {
                        'active': True,
                        'stats': token_stats
                    }
                except Exception as e:
                    summary['managers_status']['token_tracker'] = {
                        'active': False,
                        'error': str(e)
                    }
            
            # Shared state status
            if self.shared_state_manager and workflow_context.shared_state:
                try:
                    shared_workflow_id = workflow_context.shared_state.get('shared_workflow_id')
                    if shared_workflow_id:
                        shared_stats = self.shared_state_manager.get_workflow_state(shared_workflow_id)
                        summary['managers_status']['shared_state'] = {
                            'active': True,
                            'stats': shared_stats
                        }
                except Exception as e:
                    summary['managers_status']['shared_state'] = {
                        'active': False,
                        'error': str(e)
                    }
            
            return summary
    
    def cleanup_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """
        Clean up workflow resources across all managers.
        
        Args:
            workflow_id: Workflow identifier
            
        Returns:
            Cleanup statistics
        """
        with self._lock:
            cleanup_stats = {
                'workflow_id': workflow_id,
                'cleanup_actions': [],
                'errors': []
            }
            
            if workflow_id not in self.workflow_contexts:
                cleanup_stats['errors'].append('Workflow not found')
                return cleanup_stats
            
            workflow_context = self.workflow_contexts[workflow_id]
            
            # Cleanup context manager
            if self.context_manager:
                try:
                    self.context_manager.cleanup_context(workflow_id)
                    cleanup_stats['cleanup_actions'].append('context_manager_cleaned')
                except Exception as e:
                    cleanup_stats['errors'].append(f'context_manager_error: {e}')
            
            # Cleanup memory manager (optional - memories might be kept for learning)
            if self.memory_manager:
                try:
                    # Optionally clean up temporary memories
                    memory_key = workflow_context.memory_data.get('memory_key')
                    if memory_key:
                        # Note: We might want to keep memories for learning
                        cleanup_stats['cleanup_actions'].append('memory_manager_noted')
                except Exception as e:
                    cleanup_stats['errors'].append(f'memory_manager_error: {e}')
            
            # Cleanup shared state
            if self.shared_state_manager and workflow_context.shared_state:
                try:
                    shared_workflow_id = workflow_context.shared_state.get('shared_workflow_id')
                    if shared_workflow_id:
                        self.shared_state_manager.cleanup_workflow(shared_workflow_id)
                        cleanup_stats['cleanup_actions'].append('shared_state_cleaned')
                except Exception as e:
                    cleanup_stats['errors'].append(f'shared_state_error: {e}')
            
            # Finalize token tracking
            if self.token_tracker:
                try:
                    final_stats = self.token_tracker.stop_tracking(workflow_id)
                    cleanup_stats['final_token_stats'] = final_stats
                    cleanup_stats['cleanup_actions'].append('token_tracking_finalized')
                except Exception as e:
                    cleanup_stats['errors'].append(f'token_tracker_error: {e}')
            
            # Remove workflow context
            del self.workflow_contexts[workflow_id]
            cleanup_stats['cleanup_actions'].append('workflow_context_removed')
            
            # Emit cleanup event
            self._emit_event('workflow_cleaned', {
                'workflow_id': workflow_id,
                'cleanup_stats': cleanup_stats
            })
            
            return cleanup_stats
    
    def _generate_cache_key(self, workflow_type: str, data: Dict[str, Any]) -> str:
        """Generate a cache key for the workflow data."""
        key_data = {
            'workflow_type': workflow_type,
            'data_hash': hash(str(sorted(data.items())))
        }
        return f"integrated_workflow_{hash(str(key_data))}"
    
    def _should_cache_results(self, agent_role: str, results: Dict[str, Any]) -> bool:
        """Determine if results should be cached."""
        # Cache results from analytical agents
        analytical_agents = [
            'data_analyst', 'market_research_analyst', 
            'forecasting_specialist', 'brand_performance_specialist'
        ]
        return agent_role in analytical_agents and len(str(results)) > 100
    
    def _emit_event(self, event_type: str, event_data: Dict[str, Any]):
        """Emit an event to registered handlers."""
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    handler(event_data)
                except Exception as e:
                    self.logger.warning(f"Event handler error for {event_type}: {e}")
    
    def register_event_handler(self, event_type: str, handler: Callable):
        """Register an event handler."""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        stats = {
            'active_workflows': len(self.workflow_contexts),
            'managers_initialized': {
                'context_manager': self.context_manager is not None,
                'memory_manager': self.memory_manager is not None,
                'cache_manager': self.cache_manager is not None,
                'token_tracker': self.token_tracker is not None,
                'shared_state_manager': self.shared_state_manager is not None
            },
            'event_handlers': {
                event_type: len(handlers) 
                for event_type, handlers in self.event_handlers.items()
            }
        }
        
        return stats


# Global instance for easy access
_global_integrated_blackboard = None

def get_integrated_blackboard() -> IntegratedBlackboardSystem:
    """Get the global integrated blackboard system instance."""
    global _global_integrated_blackboard
    if _global_integrated_blackboard is None:
        _global_integrated_blackboard = IntegratedBlackboardSystem()
    return _global_integrated_blackboard