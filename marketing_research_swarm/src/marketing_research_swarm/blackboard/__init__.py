"""
Integrated Blackboard System for Marketing Research Swarm

This module implements a unified blackboard system that integrates with existing
managers for maximum token efficiency through coordinated state management.
"""

from .shared_state_manager import SharedStateManager, WorkflowState, TaskState
from .integrated_blackboard import IntegratedBlackboardSystem, get_integrated_blackboard
from .blackboard_crew import BlackboardMarketingResearchCrew, create_blackboard_crew
from .state_aware_agents import StateAwareAgent

# Optional import for backward compatibility
try:
    from .blackboard_orchestrator import BlackboardOrchestrator
except ImportError:
    BlackboardOrchestrator = None

__all__ = [
    'SharedStateManager',
    'WorkflowState', 
    'TaskState',
    'IntegratedBlackboardSystem',
    'get_integrated_blackboard',
    'BlackboardMarketingResearchCrew',
    'create_blackboard_crew',
    'StateAwareAgent'
]

# Add BlackboardOrchestrator if available
if BlackboardOrchestrator:
    __all__.append('BlackboardOrchestrator')