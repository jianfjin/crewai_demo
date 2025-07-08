"""
Enhanced Token Tracker for Blackboard System

This module provides enhanced token tracking that properly captures
actual LLM usage from CrewAI agents and integrates with the blackboard system.
"""

import time
from typing import Dict, Any, Optional
from datetime import datetime
from ..utils.token_tracker import TokenTracker, get_token_tracker

class BlackboardTokenTracker:
    """Enhanced token tracker that integrates with the blackboard system."""
    
    def __init__(self):
        self.base_tracker = get_token_tracker()
        self.workflow_tokens: Dict[str, Dict[str, Any]] = {}
        self.agent_tokens: Dict[str, Dict[str, Any]] = {}
    
    def start_workflow_tracking(self, workflow_id: str) -> bool:
        """Start tracking tokens for a workflow."""
        try:
            # Start crew tracking in base tracker
            crew_usage = self.base_tracker.start_crew_tracking(workflow_id)
            
            # Initialize workflow tracking
            self.workflow_tokens[workflow_id] = {
                'start_time': datetime.now(),
                'crew_usage': crew_usage,
                'agents': {},
                'total_tokens': 0
            }
            
            print(f"🔍 Started token tracking for workflow: {workflow_id}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start workflow tracking: {e}")
            return False
    
    def start_agent_tracking(self, workflow_id: str, agent_role: str, task_name: str) -> bool:
        """Start tracking tokens for a specific agent."""
        try:
            # Start task tracking in base tracker
            task_usage = self.base_tracker.start_task_tracking(task_name, agent_role)
            
            # Initialize agent tracking
            agent_key = f"{workflow_id}_{agent_role}"
            self.agent_tokens[agent_key] = {
                'workflow_id': workflow_id,
                'agent_role': agent_role,
                'task_name': task_name,
                'start_time': datetime.now(),
                'task_usage': task_usage,
                'llm_calls': 0,
                'total_tokens': 0
            }
            
            print(f"🤖 Started agent tracking: {agent_role} in {workflow_id}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start agent tracking: {e}")
            return False
    
    def record_llm_call(self, workflow_id: str, agent_role: str, 
                       prompt: str, response: str, actual_usage: Optional[Dict] = None) -> Dict[str, int]:
        """Record an LLM call and return token usage."""
        try:
            # Record in base tracker
            token_usage = self.base_tracker.record_llm_usage(prompt, response, actual_usage)
            
            # Update agent tracking
            agent_key = f"{workflow_id}_{agent_role}"
            if agent_key in self.agent_tokens:
                self.agent_tokens[agent_key]['llm_calls'] += 1
                self.agent_tokens[agent_key]['total_tokens'] += token_usage.total_tokens
            
            # Update workflow tracking
            if workflow_id in self.workflow_tokens:
                self.workflow_tokens[workflow_id]['total_tokens'] += token_usage.total_tokens
            
            print(f"📊 Recorded LLM call: {agent_role} used {token_usage.total_tokens} tokens")
            
            return {
                'prompt_tokens': token_usage.prompt_tokens,
                'completion_tokens': token_usage.completion_tokens,
                'total_tokens': token_usage.total_tokens
            }
            
        except Exception as e:
            print(f"❌ Failed to record LLM call: {e}")
            return {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
    
    def complete_agent_tracking(self, workflow_id: str, agent_role: str) -> Dict[str, Any]:
        """Complete tracking for an agent and return usage stats."""
        try:
            agent_key = f"{workflow_id}_{agent_role}"
            
            if agent_key not in self.agent_tokens:
                return {}
            
            # Complete task in base tracker
            self.base_tracker.complete_current_task()
            
            # Get agent stats
            agent_data = self.agent_tokens[agent_key]
            end_time = datetime.now()
            duration = (end_time - agent_data['start_time']).total_seconds()
            
            stats = {
                'agent_role': agent_role,
                'task_name': agent_data['task_name'],
                'duration_seconds': duration,
                'llm_calls': agent_data['llm_calls'],
                'total_tokens': agent_data['total_tokens'],
                'completed_at': end_time.isoformat()
            }
            
            # Store in workflow tracking
            if workflow_id in self.workflow_tokens:
                self.workflow_tokens[workflow_id]['agents'][agent_role] = stats
            
            print(f"✅ Completed agent tracking: {agent_role} used {stats['total_tokens']} tokens")
            return stats
            
        except Exception as e:
            print(f"❌ Failed to complete agent tracking: {e}")
            return {}
    
    def complete_workflow_tracking(self, workflow_id: str) -> Dict[str, Any]:
        """Complete tracking for a workflow and return final stats."""
        try:
            if workflow_id not in self.workflow_tokens:
                return {}
            
            # Complete crew tracking in base tracker
            workflow_data = self.workflow_tokens[workflow_id]
            if 'crew_usage' in workflow_data:
                workflow_data['crew_usage'].complete()
            
            # Calculate final stats
            end_time = datetime.now()
            duration = (end_time - workflow_data['start_time']).total_seconds()
            
            final_stats = {
                'workflow_id': workflow_id,
                'total_duration_seconds': duration,
                'total_tokens': workflow_data['total_tokens'],
                'agents': workflow_data['agents'],
                'completed_at': end_time.isoformat()
            }
            
            # Get actual usage from base tracker if available
            if hasattr(self.base_tracker, 'crew_usage') and self.base_tracker.crew_usage:
                actual_usage = self.base_tracker.crew_usage.total_token_usage
                final_stats.update({
                    'actual_total_tokens': actual_usage.total_tokens,
                    'actual_prompt_tokens': actual_usage.prompt_tokens,
                    'actual_completion_tokens': actual_usage.completion_tokens
                })
            
            print(f"🎯 Completed workflow tracking: {workflow_id} used {final_stats['total_tokens']} tokens")
            return final_stats
            
        except Exception as e:
            print(f"❌ Failed to complete workflow tracking: {e}")
            return {}
    
    def get_workflow_stats(self, workflow_id: str) -> Dict[str, Any]:
        """Get current stats for a workflow."""
        if workflow_id in self.workflow_tokens:
            return self.workflow_tokens[workflow_id]
        return {}
    
    def get_agent_stats(self, workflow_id: str, agent_role: str) -> Dict[str, Any]:
        """Get current stats for an agent."""
        agent_key = f"{workflow_id}_{agent_role}"
        if agent_key in self.agent_tokens:
            return self.agent_tokens[agent_key]
        return {}

# Global instance
_global_blackboard_tracker = None

def get_blackboard_tracker() -> BlackboardTokenTracker:
    """Get the global blackboard token tracker."""
    global _global_blackboard_tracker
    if _global_blackboard_tracker is None:
        _global_blackboard_tracker = BlackboardTokenTracker()
    return _global_blackboard_tracker