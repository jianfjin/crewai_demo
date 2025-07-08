"""
Result Reference System for Agent Interdependency

This module implements a reference-based system where:
1. Each agent's output is stored with a unique reference key
2. Subsequent agents receive only relevant reference keys
3. Agents retrieve previous results using reference keys
4. Context is isolated per agent execution
"""

import uuid
import json
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ResultReference:
    """Reference to a stored agent result."""
    reference_key: str
    agent_role: str
    task_type: str
    created_at: datetime
    summary: str
    data_size: int
    dependencies: List[str] = None

class ResultReferenceManager:
    """Manages storage and retrieval of agent results via reference keys."""
    
    def __init__(self):
        self.stored_results: Dict[str, Dict[str, Any]] = {}
        self.result_references: Dict[str, ResultReference] = {}
        self.agent_dependencies: Dict[str, List[str]] = {}
    
    def store_agent_result(self, agent_role: str, task_type: str, result_data: Any, 
                          dependencies: List[str] = None) -> str:
        """
        Store agent result and return reference key.
        
        Args:
            agent_role: Role of the agent that produced the result
            task_type: Type of task performed
            result_data: The actual result data
            dependencies: List of reference keys this result depends on
            
        Returns:
            Reference key for retrieving the result
        """
        reference_key = f"{agent_role}_{task_type}_{uuid.uuid4().hex[:8]}"
        
        # Store the actual result data
        self.stored_results[reference_key] = {
            'data': result_data,
            'metadata': {
                'agent_role': agent_role,
                'task_type': task_type,
                'created_at': datetime.now().isoformat(),
                'dependencies': dependencies or []
            }
        }
        
        # Create reference metadata
        summary = self._create_result_summary(result_data)
        data_size = len(str(result_data))
        
        self.result_references[reference_key] = ResultReference(
            reference_key=reference_key,
            agent_role=agent_role,
            task_type=task_type,
            created_at=datetime.now(),
            summary=summary,
            data_size=data_size,
            dependencies=dependencies or []
        )
        
        # Track dependencies
        if dependencies:
            self.agent_dependencies[reference_key] = dependencies
        
        print(f"📦 Stored result: {reference_key} ({data_size} bytes)")
        return reference_key
    
    def retrieve_result(self, reference_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve result by reference key."""
        if reference_key in self.stored_results:
            print(f"📥 Retrieved result: {reference_key}")
            return self.stored_results[reference_key]
        return None
    
    def get_relevant_references(self, agent_role: str, task_type: str) -> List[ResultReference]:
        """
        Get relevant result references for an agent based on task dependencies.
        
        Args:
            agent_role: Current agent role
            task_type: Current task type
            
        Returns:
            List of relevant result references
        """
        relevant_refs = []
        
        # Define task dependencies
        task_dependencies = {
            'data_analysis': [],  # First task, no dependencies
            'campaign_optimization': ['data_analysis'],  # Depends on data analysis
            'content_strategy': ['data_analysis'],  # Depends on data analysis
            'market_research': [],  # Independent analysis
            'brand_performance': ['market_research', 'data_analysis'],  # Depends on both
            'sales_forecast': ['data_analysis', 'market_research']  # Depends on both
        }
        
        required_task_types = task_dependencies.get(task_type, [])
        
        for ref_key, ref in self.result_references.items():
            if ref.task_type in required_task_types:
                relevant_refs.append(ref)
        
        print(f"🔗 Found {len(relevant_refs)} relevant references for {agent_role}:{task_type}")
        return relevant_refs
    
    def create_isolated_context(self, agent_role: str, task_type: str, 
                               base_inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create isolated context for an agent with only relevant information.
        
        Args:
            agent_role: Current agent role
            task_type: Current task type
            base_inputs: Base input parameters
            
        Returns:
            Isolated context with relevant references
        """
        context = {
            'agent_role': agent_role,
            'task_type': task_type,
            'base_inputs': base_inputs,
            'available_references': [],
            'context_window_size': 0
        }
        
        # Get relevant references
        relevant_refs = self.get_relevant_references(agent_role, task_type)
        
        # Add reference summaries (not full data) to context
        for ref in relevant_refs:
            context['available_references'].append({
                'reference_key': ref.reference_key,
                'agent_role': ref.agent_role,
                'task_type': ref.task_type,
                'summary': ref.summary,
                'created_at': ref.created_at.isoformat()
            })
        
        context['context_window_size'] = len(str(context))
        
        print(f"🎯 Created isolated context for {agent_role}: {context['context_window_size']} bytes")
        return context
    
    def _create_result_summary(self, result_data: Any) -> str:
        """Create a concise summary of result data."""
        if isinstance(result_data, dict):
            keys = list(result_data.keys())[:5]  # First 5 keys
            return f"Dict with keys: {keys}"
        elif isinstance(result_data, str):
            return result_data[:200] + "..." if len(result_data) > 200 else result_data
        elif isinstance(result_data, list):
            return f"List with {len(result_data)} items"
        else:
            return str(result_data)[:200]
    
    def get_dependency_chain(self, reference_key: str) -> List[str]:
        """Get the full dependency chain for a reference."""
        chain = []
        
        def _build_chain(ref_key: str):
            if ref_key in self.agent_dependencies:
                for dep in self.agent_dependencies[ref_key]:
                    _build_chain(dep)
                    if dep not in chain:
                        chain.append(dep)
        
        _build_chain(reference_key)
        return chain
    
    def cleanup_old_results(self, max_age_hours: int = 24):
        """Clean up old results to manage memory."""
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
        
        to_remove = []
        for ref_key, ref in self.result_references.items():
            if ref.created_at.timestamp() < cutoff_time:
                to_remove.append(ref_key)
        
        for ref_key in to_remove:
            del self.stored_results[ref_key]
            del self.result_references[ref_key]
            if ref_key in self.agent_dependencies:
                del self.agent_dependencies[ref_key]
        
        print(f"🧹 Cleaned up {len(to_remove)} old results")
        return len(to_remove)

# Global instance
_global_reference_manager = None

def get_reference_manager() -> ResultReferenceManager:
    """Get the global result reference manager."""
    global _global_reference_manager
    if _global_reference_manager is None:
        _global_reference_manager = ResultReferenceManager()
    return _global_reference_manager