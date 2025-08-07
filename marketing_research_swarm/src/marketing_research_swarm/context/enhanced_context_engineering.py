"""
Enhanced Context Engineering Implementation
Based on the article "Optimizing LangChain AI Agents with Contextual Engineering"

This module implements advanced context engineering techniques including:
1. Scratchpads for short-term memory
2. Enhanced checkpointing
3. InMemoryStore for long-term memory
4. Context compression and isolation
5. Dynamic context management
"""

import json
import uuid
import hashlib
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging

from langchain_core.stores import InMemoryStore
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)

class ContextType(Enum):
    """Types of context data."""
    SCRATCHPAD = "scratchpad"
    CHECKPOINT = "checkpoint"
    LONG_TERM = "long_term"
    COMPRESSED = "compressed"
    ISOLATED = "isolated"

@dataclass
class ScratchpadEntry:
    """Entry in an agent's scratchpad for short-term memory."""
    id: str
    agent_role: str
    step: int
    content: Any
    reasoning: str
    timestamp: datetime
    context_size: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "agent_role": self.agent_role,
            "step": self.step,
            "content": self.content,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.isoformat(),
            "context_size": self.context_size
        }

@dataclass
class ContextCheckpoint:
    """Checkpoint for saving agent state at each step."""
    checkpoint_id: str
    thread_id: str
    agent_role: str
    step: int
    state: Dict[str, Any]
    scratchpad: List[ScratchpadEntry]
    timestamp: datetime
    token_usage: Dict[str, int]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "thread_id": self.thread_id,
            "agent_role": self.agent_role,
            "step": self.step,
            "state": self.state,
            "scratchpad": [entry.to_dict() for entry in self.scratchpad],
            "timestamp": self.timestamp.isoformat(),
            "token_usage": self.token_usage
        }

class EnhancedContextEngineering:
    """
    Enhanced context engineering system implementing best practices from the article.
    
    Features:
    - Scratchpads for short-term memory
    - Enhanced checkpointing with state persistence
    - InMemoryStore for long-term memory across threads
    - Context compression and isolation
    - Dynamic context management
    """
    
    def __init__(self, max_scratchpad_size: int = 10, max_checkpoint_history: int = 50):
        """
        Initialize the enhanced context engineering system.
        
        Args:
            max_scratchpad_size: Maximum entries per agent scratchpad
            max_checkpoint_history: Maximum checkpoints to keep per thread
        """
        self.max_scratchpad_size = max_scratchpad_size
        self.max_checkpoint_history = max_checkpoint_history
        
        # Scratchpads for short-term memory (per agent)
        self.scratchpads: Dict[str, List[ScratchpadEntry]] = {}
        
        # Checkpointing system
        self.checkpointer = MemorySaver()
        self.checkpoint_history: Dict[str, List[ContextCheckpoint]] = {}
        
        # InMemoryStore for long-term memory across threads
        self.long_term_store = InMemoryStore()
        
        # Context compression cache
        self.compression_cache: Dict[str, Any] = {}
        
        # Context isolation mappings
        self.isolation_mappings: Dict[str, Dict[str, Any]] = {}
        
        logger.info("🧠 Enhanced Context Engineering initialized")
    
    def create_scratchpad_entry(self, agent_role: str, step: int, content: Any, 
                              reasoning: str) -> ScratchpadEntry:
        """
        Create a new scratchpad entry for an agent's short-term memory.
        
        Args:
            agent_role: Role of the agent
            step: Current step number
            content: Content to store
            reasoning: Reasoning behind this step
            
        Returns:
            ScratchpadEntry object
        """
        entry_id = f"scratch_{agent_role}_{step}_{uuid.uuid4().hex[:8]}"
        content_size = len(json.dumps(content, default=str))
        
        entry = ScratchpadEntry(
            id=entry_id,
            agent_role=agent_role,
            step=step,
            content=content,
            reasoning=reasoning,
            timestamp=datetime.now(),
            context_size=content_size
        )
        
        # Add to agent's scratchpad
        if agent_role not in self.scratchpads:
            self.scratchpads[agent_role] = []
        
        self.scratchpads[agent_role].append(entry)
        
        # Maintain scratchpad size limit
        if len(self.scratchpads[agent_role]) > self.max_scratchpad_size:
            self.scratchpads[agent_role] = self.scratchpads[agent_role][-self.max_scratchpad_size:]
        
        logger.debug(f"📝 Created scratchpad entry for {agent_role} step {step}")
        return entry
    
    def get_scratchpad_context(self, agent_role: str, max_entries: int = 5) -> Dict[str, Any]:
        """
        Get recent scratchpad entries for context.
        
        Args:
            agent_role: Role of the agent
            max_entries: Maximum entries to return
            
        Returns:
            Dictionary with scratchpad context
        """
        if agent_role not in self.scratchpads:
            return {"scratchpad_entries": [], "total_entries": 0}
        
        recent_entries = self.scratchpads[agent_role][-max_entries:]
        
        return {
            "scratchpad_entries": [entry.to_dict() for entry in recent_entries],
            "total_entries": len(self.scratchpads[agent_role]),
            "agent_role": agent_role,
            "last_updated": recent_entries[-1].timestamp.isoformat() if recent_entries else None
        }
    
    def create_checkpoint(self, thread_id: str, agent_role: str, step: int, 
                         state: Dict[str, Any], token_usage: Dict[str, int]) -> ContextCheckpoint:
        """
        Create a checkpoint to save agent state at each step.
        
        Args:
            thread_id: Thread identifier
            agent_role: Role of the agent
            step: Current step number
            state: Current state to checkpoint
            token_usage: Token usage statistics
            
        Returns:
            ContextCheckpoint object
        """
        checkpoint_id = f"checkpoint_{thread_id}_{agent_role}_{step}_{uuid.uuid4().hex[:8]}"
        
        # Get current scratchpad for this agent
        scratchpad = self.scratchpads.get(agent_role, [])
        
        checkpoint = ContextCheckpoint(
            checkpoint_id=checkpoint_id,
            thread_id=thread_id,
            agent_role=agent_role,
            step=step,
            state=state.copy(),
            scratchpad=scratchpad.copy(),
            timestamp=datetime.now(),
            token_usage=token_usage.copy()
        )
        
        # Store checkpoint
        if thread_id not in self.checkpoint_history:
            self.checkpoint_history[thread_id] = []
        
        self.checkpoint_history[thread_id].append(checkpoint)
        
        # Maintain checkpoint history limit
        if len(self.checkpoint_history[thread_id]) > self.max_checkpoint_history:
            self.checkpoint_history[thread_id] = self.checkpoint_history[thread_id][-self.max_checkpoint_history:]
        
        # Also save to LangGraph checkpointer
        config = {"configurable": {"thread_id": thread_id}}
        self.checkpointer.put(config, checkpoint.to_dict())
        
        logger.debug(f"💾 Created checkpoint {checkpoint_id} for {agent_role}")
        return checkpoint
    
    def restore_from_checkpoint(self, thread_id: str, checkpoint_id: Optional[str] = None) -> Optional[ContextCheckpoint]:
        """
        Restore state from a checkpoint.
        
        Args:
            thread_id: Thread identifier
            checkpoint_id: Specific checkpoint ID (if None, gets latest)
            
        Returns:
            ContextCheckpoint object or None if not found
        """
        if thread_id not in self.checkpoint_history:
            return None
        
        checkpoints = self.checkpoint_history[thread_id]
        
        if checkpoint_id:
            # Find specific checkpoint
            for checkpoint in reversed(checkpoints):
                if checkpoint.checkpoint_id == checkpoint_id:
                    logger.info(f"🔄 Restored from checkpoint {checkpoint_id}")
                    return checkpoint
        else:
            # Return latest checkpoint
            if checkpoints:
                latest = checkpoints[-1]
                logger.info(f"🔄 Restored from latest checkpoint {latest.checkpoint_id}")
                return latest
        
        return None
    
    def store_long_term_memory(self, key: str, value: Any, namespace: str = "default") -> None:
        """
        Store data in long-term memory across threads.
        
        Args:
            key: Storage key
            value: Value to store
            namespace: Namespace for organization
        """
        namespaced_key = f"{namespace}:{key}"
        self.long_term_store.mset([(namespaced_key, value)])
        logger.debug(f"🧠 Stored long-term memory: {namespaced_key}")
    
    def retrieve_long_term_memory(self, key: str, namespace: str = "default") -> Any:
        """
        Retrieve data from long-term memory.
        
        Args:
            key: Storage key
            namespace: Namespace for organization
            
        Returns:
            Stored value or None if not found
        """
        namespaced_key = f"{namespace}:{key}"
        results = self.long_term_store.mget([namespaced_key])
        value = results[0] if results and results[0] is not None else None
        
        if value:
            logger.debug(f"🧠 Retrieved long-term memory: {namespaced_key}")
        
        return value
    
    def search_long_term_memory(self, pattern: str, namespace: str = "default") -> List[Tuple[str, Any]]:
        """
        Search long-term memory by pattern.
        
        Args:
            pattern: Search pattern
            namespace: Namespace to search in
            
        Returns:
            List of (key, value) tuples
        """
        # Simple pattern matching (could be enhanced with regex)
        results = []
        prefix = f"{namespace}:"
        
        # Note: InMemoryStore doesn't have a direct search method,
        # so we'd need to implement this differently in a real scenario
        # This is a simplified implementation
        
        logger.debug(f"🔍 Searched long-term memory for pattern: {pattern}")
        return results
    
    def compress_context(self, context: Dict[str, Any], compression_ratio: float = 0.5) -> Dict[str, Any]:
        """
        Compress context to reduce token usage.
        
        Args:
            context: Context to compress
            compression_ratio: Target compression ratio (0.5 = 50% reduction)
            
        Returns:
            Compressed context
        """
        context_hash = hashlib.md5(json.dumps(context, sort_keys=True, default=str).encode()).hexdigest()
        
        # Check cache first
        if context_hash in self.compression_cache:
            logger.debug(f"📦 Using cached compression for context")
            return self.compression_cache[context_hash]
        
        compressed = self._apply_compression(context, compression_ratio)
        
        # Cache the result
        self.compression_cache[context_hash] = compressed
        
        logger.debug(f"📦 Compressed context with ratio {compression_ratio}")
        return compressed
    
    def _apply_compression(self, context: Dict[str, Any], ratio: float) -> Dict[str, Any]:
        """Apply compression algorithms to context."""
        compressed = {}
        
        for key, value in context.items():
            if isinstance(value, str) and len(value) > 200:
                # Compress long strings
                target_length = int(len(value) * ratio)
                compressed[key] = value[:target_length] + "... [compressed]"
            elif isinstance(value, list) and len(value) > 10:
                # Compress long lists
                target_length = int(len(value) * ratio)
                compressed[key] = value[:target_length] + ["... [compressed]"]
            elif isinstance(value, dict):
                # Recursively compress dictionaries
                compressed[key] = self._apply_compression(value, ratio)
            else:
                compressed[key] = value
        
        return compressed
    
    def create_isolated_context(self, agent_role: str, full_context: Dict[str, Any], 
                              relevant_keys: List[str] = None) -> Dict[str, Any]:
        """
        Create isolated context for specific agent.
        
        Args:
            agent_role: Role of the agent
            full_context: Complete context
            relevant_keys: Keys relevant to this agent
            
        Returns:
            Isolated context
        """
        # Define agent-specific relevance if not provided
        if relevant_keys is None:
            relevance_mapping = {
                'market_research_analyst': ['market_data', 'industry_trends', 'consumer_behavior'],
                'competitive_analyst': ['competitive_landscape', 'market_share', 'competitor_analysis'],
                'data_analyst': ['raw_data', 'analytics', 'statistical_models'],
                'content_strategist': ['content_performance', 'audience_data', 'brand_voice'],
                'brand_performance_specialist': ['brand_metrics', 'performance_data', 'consumer_sentiment'],
                'forecasting_specialist': ['historical_data', 'trend_analysis', 'prediction_models']
            }
            relevant_keys = relevance_mapping.get(agent_role, [])
        
        isolated = {
            'agent_role': agent_role,
            'isolation_timestamp': datetime.now().isoformat(),
            'context_type': ContextType.ISOLATED.value
        }
        
        # Add relevant data
        for key in relevant_keys:
            if key in full_context:
                isolated[key] = full_context[key]
        
        # Add scratchpad context
        scratchpad_context = self.get_scratchpad_context(agent_role)
        isolated['scratchpad'] = scratchpad_context
        
        # Store isolation mapping
        isolation_id = f"isolation_{agent_role}_{uuid.uuid4().hex[:8]}"
        self.isolation_mappings[isolation_id] = {
            'agent_role': agent_role,
            'relevant_keys': relevant_keys,
            'created_at': datetime.now(),
            'context_size': len(json.dumps(isolated, default=str))
        }
        
        isolated['isolation_id'] = isolation_id
        
        logger.debug(f"🔒 Created isolated context for {agent_role}")
        return isolated
    
    def get_context_for_agent(self, agent_role: str, thread_id: str, step: int,
                             full_context: Dict[str, Any], strategy: str = "smart") -> Dict[str, Any]:
        """
        Get optimized context for an agent using the specified strategy.
        
        Args:
            agent_role: Role of the agent
            thread_id: Thread identifier
            step: Current step
            full_context: Complete context
            strategy: Optimization strategy ('smart', 'isolated', 'compressed', 'minimal')
            
        Returns:
            Optimized context for the agent
        """
        if strategy == "isolated":
            return self.create_isolated_context(agent_role, full_context)
        elif strategy == "compressed":
            isolated = self.create_isolated_context(agent_role, full_context)
            return self.compress_context(isolated, compression_ratio=0.6)
        elif strategy == "minimal":
            # Only essential data
            return {
                'agent_role': agent_role,
                'thread_id': thread_id,
                'step': step,
                'scratchpad': self.get_scratchpad_context(agent_role, max_entries=3)
            }
        else:  # smart strategy
            # Combine isolation and compression intelligently
            isolated = self.create_isolated_context(agent_role, full_context)
            
            # Add long-term memory context
            memory_key = f"agent_context_{agent_role}"
            long_term_context = self.retrieve_long_term_memory(memory_key, "agent_contexts")
            if long_term_context:
                isolated['long_term_memory'] = long_term_context
            
            # Compress if context is too large
            context_size = len(json.dumps(isolated, default=str))
            if context_size > 8000:  # 8KB threshold
                isolated = self.compress_context(isolated, compression_ratio=0.7)
            
            return isolated
    
    def update_agent_memory(self, agent_role: str, new_insights: Dict[str, Any]) -> None:
        """
        Update agent's long-term memory with new insights.
        
        Args:
            agent_role: Role of the agent
            new_insights: New insights to store
        """
        memory_key = f"agent_context_{agent_role}"
        existing_memory = self.retrieve_long_term_memory(memory_key, "agent_contexts") or {}
        
        # Merge new insights
        updated_memory = {**existing_memory, **new_insights}
        updated_memory['last_updated'] = datetime.now().isoformat()
        
        self.store_long_term_memory(memory_key, updated_memory, "agent_contexts")
        logger.debug(f"🧠 Updated long-term memory for {agent_role}")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics."""
        total_scratchpad_entries = sum(len(entries) for entries in self.scratchpads.values())
        total_checkpoints = sum(len(checkpoints) for checkpoints in self.checkpoint_history.values())
        
        return {
            'scratchpads': {
                'total_agents': len(self.scratchpads),
                'total_entries': total_scratchpad_entries,
                'agents': list(self.scratchpads.keys())
            },
            'checkpoints': {
                'total_threads': len(self.checkpoint_history),
                'total_checkpoints': total_checkpoints,
                'threads': list(self.checkpoint_history.keys())
            },
            'long_term_memory': {
                'store_type': 'InMemoryStore',
                'status': 'active'
            },
            'compression_cache': {
                'cached_contexts': len(self.compression_cache)
            },
            'isolation_mappings': {
                'total_isolations': len(self.isolation_mappings)
            }
        }
    
    def cleanup_old_data(self, max_age_hours: int = 24) -> Dict[str, int]:
        """
        Clean up old data to free memory.
        
        Args:
            max_age_hours: Maximum age in hours for data retention
            
        Returns:
            Dictionary with cleanup statistics
        """
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        cleanup_stats = {
            'scratchpad_entries_removed': 0,
            'checkpoints_removed': 0,
            'isolation_mappings_removed': 0
        }
        
        # Clean up old scratchpad entries
        for agent_role, entries in self.scratchpads.items():
            old_count = len(entries)
            self.scratchpads[agent_role] = [
                entry for entry in entries 
                if entry.timestamp > cutoff_time
            ]
            cleanup_stats['scratchpad_entries_removed'] += old_count - len(self.scratchpads[agent_role])
        
        # Clean up old checkpoints
        for thread_id, checkpoints in self.checkpoint_history.items():
            old_count = len(checkpoints)
            self.checkpoint_history[thread_id] = [
                checkpoint for checkpoint in checkpoints
                if checkpoint.timestamp > cutoff_time
            ]
            cleanup_stats['checkpoints_removed'] += old_count - len(self.checkpoint_history[thread_id])
        
        # Clean up old isolation mappings
        old_mappings = list(self.isolation_mappings.keys())
        for isolation_id, mapping in list(self.isolation_mappings.items()):
            if mapping['created_at'] < cutoff_time:
                del self.isolation_mappings[isolation_id]
                cleanup_stats['isolation_mappings_removed'] += 1
        
        logger.info(f"🧹 Cleanup completed: {cleanup_stats}")
        return cleanup_stats

# Global instance
_global_enhanced_context_engineering = None

def get_enhanced_context_engineering() -> EnhancedContextEngineering:
    """Get the global enhanced context engineering instance."""
    global _global_enhanced_context_engineering
    if _global_enhanced_context_engineering is None:
        _global_enhanced_context_engineering = EnhancedContextEngineering()
    return _global_enhanced_context_engineering