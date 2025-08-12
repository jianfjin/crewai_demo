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
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ContextStrategy(Enum):
    """Context optimization strategies"""
    FULL_CONTEXT = "full_context"
    COMPRESSED = "compressed"
    REFERENCE_BASED = "reference_based"
    SMART_SUMMARY = "smart_summary"


@dataclass
class ContextMetrics:
    """Metrics for context optimization"""
    original_size: int = 0
    optimized_size: int = 0
    compression_ratio: float = 0.0
    processing_time: float = 0.0
    strategy_used: str = ""


class EnhancedContextEngineering:
    """Enhanced context engineering for optimized agent performance"""
    
    def __init__(self):
        self.scratchpads = {}
        self.long_term_memory = {}
        self.checkpoints = {}
        self.context_cache = {}
        
    def create_scratchpad(self, agent_id: str, initial_context: Dict[str, Any] = None) -> str:
        """Create a scratchpad for an agent"""
        scratchpad_id = f"scratchpad_{agent_id}_{uuid.uuid4().hex[:8]}"
        self.scratchpads[scratchpad_id] = {
            "agent_id": agent_id,
            "context": initial_context or {},
            "created_at": datetime.now(),
            "last_updated": datetime.now()
        }
        return scratchpad_id
    
    def update_scratchpad(self, scratchpad_id: str, updates: Dict[str, Any]):
        """Update scratchpad with new information"""
        if scratchpad_id in self.scratchpads:
            self.scratchpads[scratchpad_id]["context"].update(updates)
            self.scratchpads[scratchpad_id]["last_updated"] = datetime.now()
    
    def get_scratchpad(self, scratchpad_id: str) -> Dict[str, Any]:
        """Get scratchpad content"""
        return self.scratchpads.get(scratchpad_id, {})
    
    def store_long_term_memory(self, key: str, value: Any, namespace: str = "default"):
        """Store information in long-term memory"""
        if namespace not in self.long_term_memory:
            self.long_term_memory[namespace] = {}
        self.long_term_memory[namespace][key] = {
            "value": value,
            "stored_at": datetime.now(),
            "access_count": 0
        }
    
    def retrieve_long_term_memory(self, key: str, namespace: str = "default") -> Any:
        """Retrieve information from long-term memory"""
        if namespace in self.long_term_memory and key in self.long_term_memory[namespace]:
            self.long_term_memory[namespace][key]["access_count"] += 1
            return self.long_term_memory[namespace][key]["value"]
        return None
    
    def search_long_term_memory(self, pattern: str, namespace: str = "default") -> List[Tuple[str, Any]]:
        """Search long-term memory for patterns"""
        results = []
        if namespace in self.long_term_memory:
            for key, data in self.long_term_memory[namespace].items():
                if pattern.lower() in key.lower() or pattern.lower() in str(data["value"]).lower():
                    results.append((key, data["value"]))
        return results
    
    def create_checkpoint(self, checkpoint_id: str, state: Dict[str, Any]):
        """Create a checkpoint of the current state"""
        self.checkpoints[checkpoint_id] = {
            "state": state,
            "created_at": datetime.now(),
            "checksum": hashlib.md5(json.dumps(state, sort_keys=True).encode()).hexdigest()
        }
    
    def restore_checkpoint(self, checkpoint_id: str) -> Dict[str, Any]:
        """Restore state from a checkpoint"""
        if checkpoint_id in self.checkpoints:
            return self.checkpoints[checkpoint_id]["state"]
        return {}
    
    def optimize_context(self, context: Dict[str, Any], strategy: ContextStrategy = ContextStrategy.SMART_SUMMARY) -> Tuple[Dict[str, Any], ContextMetrics]:
        """Optimize context using the specified strategy"""
        start_time = datetime.now()
        original_size = len(json.dumps(context))
        
        if strategy == ContextStrategy.FULL_CONTEXT:
            optimized_context = context
        elif strategy == ContextStrategy.COMPRESSED:
            optimized_context = self._compress_context(context)
        elif strategy == ContextStrategy.REFERENCE_BASED:
            optimized_context = self._create_reference_context(context)
        else:  # SMART_SUMMARY
            optimized_context = self._create_smart_summary(context)
        
        optimized_size = len(json.dumps(optimized_context))
        processing_time = (datetime.now() - start_time).total_seconds()
        
        metrics = ContextMetrics(
            original_size=original_size,
            optimized_size=optimized_size,
            compression_ratio=1 - (optimized_size / original_size) if original_size > 0 else 0,
            processing_time=processing_time,
            strategy_used=strategy.value
        )
        
        return optimized_context, metrics
    
    def _compress_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Compress context by removing redundant information"""
        compressed = {}
        for key, value in context.items():
            if isinstance(value, str) and len(value) > 500:
                # Truncate long strings
                compressed[key] = value[:500] + "..."
            elif isinstance(value, list) and len(value) > 10:
                # Limit list size
                compressed[key] = value[:10]
            else:
                compressed[key] = value
        return compressed
    
    def _create_reference_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create reference-based context"""
        reference_id = f"ref_{uuid.uuid4().hex[:8]}"
        self.context_cache[reference_id] = context
        return {
            "reference_id": reference_id,
            "summary": f"Context with {len(context)} keys stored as reference",
            "keys": list(context.keys())
        }
    
    def _create_smart_summary(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create intelligent summary of context"""
        summary = {
            "context_summary": f"Context contains {len(context)} items",
            "key_items": {}
        }
        
        # Keep important keys and summarize others
        important_keys = ["target_audience", "campaign_type", "budget", "analysis_focus"]
        for key in important_keys:
            if key in context:
                summary["key_items"][key] = context[key]
        
        # Add summary of other items
        other_keys = [k for k in context.keys() if k not in important_keys]
        if other_keys:
            summary["other_items"] = f"{len(other_keys)} additional items: {', '.join(other_keys[:5])}"
        
        return summary


# Global instance
_global_enhanced_context_engineering = None


def get_enhanced_context_engineering() -> EnhancedContextEngineering:
    """Get the global enhanced context engineering instance"""
    global _global_enhanced_context_engineering
    if _global_enhanced_context_engineering is None:
        _global_enhanced_context_engineering = EnhancedContextEngineering()
        logger.info("✅ Enhanced Context Engineering initialized")
    return _global_enhanced_context_engineering