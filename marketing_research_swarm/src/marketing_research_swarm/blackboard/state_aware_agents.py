


# Additional imports for StateAwareAgent
import time
from typing import Dict, Any, Optional
from crewai import Agent
from .integrated_blackboard import IntegratedBlackboardSystem


class StateAwareAgent(Agent):
    """
    Enhanced CrewAI Agent that integrates with the blackboard system
    for token-efficient execution through shared state management.
    """
    
    def __init__(self, 
                 role: str,
                 goal: str,
                 backstory: str,
                 blackboard_system: IntegratedBlackboardSystem,
                 workflow_id: str = None,
                 tools=None,
                 llm=None,
                 allow_delegation: bool = False,
                 **kwargs):
        """Initialize state-aware agent."""
        super().__init__(
            role=role,
            goal=goal,
            backstory=backstory,
            tools=tools or [],
            llm=llm,
            allow_delegation=allow_delegation,
            **kwargs
        )
        
        self.workflow_id = workflow_id or f"workflow_{int(time.time())}"
        self.blackboard_system = blackboard_system
        self.execution_context = {}
        self.token_usage = {}
        
    def set_workflow_context(self, blackboard_system: IntegratedBlackboardSystem, workflow_id: str = None):
        """Set or update workflow context."""
        self.workflow_id = workflow_id or f"workflow_{int(time.time())}"
        self.blackboard_system = blackboard_system
    
    def execute(self, task) -> str:
        """Execute task using blackboard system for context and coordination."""
        start_time = time.time()
        
        try:
            # Get optimized context from blackboard
            optimized_context = self.blackboard_system.get_optimized_context(
                workflow_id=self.workflow_id,
                agent_role=self.role
            )
            
            # Store context for use during execution
            self.execution_context = optimized_context
            
            # Check for cached results
            if "cached_results" in optimized_context:
                cached_data = optimized_context["cached_results"].get("data")
                if cached_data and self._is_cache_relevant(task, cached_data):
                    print(f"[{self.role}] Using cached results for task")
                    result = self._adapt_cached_result(task, cached_data)
                    
                    # Update blackboard with cached result usage
                    self.blackboard_system.update_agent_results(
                        workflow_id=self.workflow_id,
                        agent_role=self.role,
                        results={"result": result, "source": "cache"},
                        token_usage={"input_tokens": 0, "output_tokens": len(str(result)) // 4}
                    )
                    
                    return result
            
            # Execute task with optimized context
            enhanced_task_description = self._enhance_task_with_context(task, optimized_context)
            
            # Execute using parent class method
            result = super().execute(enhanced_task_description)
            
            # Calculate token usage
            execution_time = time.time() - start_time
            estimated_tokens = self._estimate_token_usage(task, result, execution_time)
            
            # Update blackboard with results
            self.blackboard_system.update_agent_results(
                workflow_id=self.workflow_id,
                agent_role=self.role,
                results={"result": result, "source": "execution"},
                token_usage=estimated_tokens
            )
            
            return result
            
        except Exception as e:
            print(f"[{self.role}] Execution error: {e}")
            
            # Update blackboard with error
            self.blackboard_system.update_agent_results(
                workflow_id=self.workflow_id,
                agent_role=self.role,
                results={"error": str(e), "source": "error"},
                token_usage={"input_tokens": 0, "output_tokens": 0}
            )
            
            raise
    
    def _enhance_task_with_context(self, task, context):
        """Enhance task description with relevant context from blackboard."""
        base_description = task.description if hasattr(task, "description") else str(task)
        
        # Add relevant context elements
        context_additions = []
        
        # Add relevant memories if available
        if "relevant_memories" in context:
            memories = context["relevant_memories"][:3]  # Limit to top 3
            if memories:
                memory_text = "Previous insights: " + "; ".join([
                    mem.get("content", "")[:100] for mem in memories
                ])
                context_additions.append(memory_text)
        
        # Add shared workflow state if available
        if "shared_workflow_state" in context:
            workflow_state = context["shared_workflow_state"]
            if isinstance(workflow_state, dict) and "status" in workflow_state:
                context_additions.append(f"Workflow status: {workflow_state['status']}")
        
        # Combine with original description
        if context_additions:
            enhanced_description = f"{base_description}\n\nContext: {' | '.join(context_additions)}"
        else:
            enhanced_description = base_description
        
        return enhanced_description
    
    def _is_cache_relevant(self, task, cached_data):
        """Determine if cached data is relevant for the current task."""
        if not cached_data:
            return False
        
        task_str = str(task).lower()
        cache_str = str(cached_data).lower()
        
        # Check for keyword overlap
        task_keywords = set(task_str.split())
        cache_keywords = set(cache_str.split())
        
        overlap = len(task_keywords.intersection(cache_keywords))
        overlap_ratio = overlap / len(task_keywords) if task_keywords else 0
        
        return overlap_ratio > 0.3  # 30% keyword overlap threshold
    
    def _adapt_cached_result(self, task, cached_data):
        """Adapt cached result for the current task."""
        if isinstance(cached_data, dict):
            # Extract relevant parts of cached analysis
            result_parts = []
            
            for key, value in cached_data.items():
                if any(keyword in str(task).lower() for keyword in [key.lower(), str(value).lower()[:50]]):
                    result_parts.append(f"{key}: {value}")
            
            if result_parts:
                return f"Based on cached analysis:\n" + "\n".join(result_parts[:5])
        
        return f"Leveraging previous analysis: {str(cached_data)[:500]}..."
    
    def _estimate_token_usage(self, task, result: str, execution_time: float):
        """Estimate token usage for the execution."""
        # Simple estimation based on text length
        input_text = str(task) + str(self.execution_context)
        output_text = str(result)
        
        # Rough estimation: 1 token ~= 4 characters
        input_tokens = len(input_text) // 4
        output_tokens = len(output_text) // 4
        
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "execution_time": execution_time
        }
    
    def get_context_summary(self):
        """Get summary of current execution context."""
        return {
            "workflow_id": self.workflow_id,
            "agent_role": self.role,
            "context_keys": list(self.execution_context.keys()),
            "has_cached_data": "cached_results" in self.execution_context,
            "has_memories": "relevant_memories" in self.execution_context,
            "has_shared_state": "shared_workflow_state" in self.execution_context
        }

