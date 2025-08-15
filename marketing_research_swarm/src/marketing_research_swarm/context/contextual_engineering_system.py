"""
Advanced Contextual Engineering System
Implements the four core strategies: Write, Select, Compress, Isolate

Based on the contextual engineering guide and LangGraph best practices.
"""

import os
import json
import logging
import hashlib
import tiktoken
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import uuid

from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

class ContextType(Enum):
    """Types of context for isolation strategy."""
    INSTRUCTIONS = "instructions"
    KNOWLEDGE = "knowledge" 
    TOOLS = "tools"
    FEEDBACK = "feedback"
    SCRATCHPAD = "scratchpad"

class CompressionStrategy(Enum):
    """Context compression strategies."""
    SUMMARIZATION = "summarization"
    TRIMMING = "trimming"
    CHUNKING = "chunking"
    HEURISTIC = "heuristic"

@dataclass
class ScratchpadEntry:
    """Individual scratchpad entry with metadata."""
    id: str
    agent_id: str
    timestamp: datetime
    content: Dict[str, Any]
    context_type: ContextType
    token_cost: int
    relevance_score: float = 0.0
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'id': self.id,
            'agent_id': self.agent_id,
            'timestamp': self.timestamp.isoformat(),
            'content': self.content,
            'context_type': self.context_type.value,
            'token_cost': self.token_cost,
            'relevance_score': self.relevance_score,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed.isoformat() if self.last_accessed else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ScratchpadEntry':
        """Create from dictionary."""
        return cls(
            id=data['id'],
            agent_id=data['agent_id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            content=data['content'],
            context_type=ContextType(data['context_type']),
            token_cost=data['token_cost'],
            relevance_score=data.get('relevance_score', 0.0),
            access_count=data.get('access_count', 0),
            last_accessed=datetime.fromisoformat(data['last_accessed']) if data.get('last_accessed') else None
        )

@dataclass
class ContextBudget:
    """Token budget allocation for different context types."""
    total_budget: int
    instructions: int = 0
    knowledge: int = 0
    tools: int = 0
    feedback: int = 0
    scratchpad: int = 0
    
    def __post_init__(self):
        """Auto-allocate budget if not specified."""
        if all(getattr(self, field.name) == 0 for field in self.__dataclass_fields__.values() if field.name != 'total_budget'):
            # Default allocation percentages
            self.instructions = int(self.total_budget * 0.3)  # 30%
            self.knowledge = int(self.total_budget * 0.4)     # 40%
            self.tools = int(self.total_budget * 0.15)        # 15%
            self.feedback = int(self.total_budget * 0.1)      # 10%
            self.scratchpad = int(self.total_budget * 0.05)   # 5%
    
    def get_budget_for_type(self, context_type: ContextType) -> int:
        """Get budget allocation for specific context type."""
        return getattr(self, context_type.value, 0)

class ContextualEngineeringSystem:
    """
    Advanced Contextual Engineering System implementing all four strategies:
    1. Write (Scratchpad) - Persistent context storage
    2. Select - Intelligent relevance-based filtering
    3. Compress - Summarization and trimming strategies
    4. Isolate - Agent-specific context separation
    """
    
    def __init__(self, model_name: str = "gpt-4o-mini", storage_path: str = "cache/contextual_engineering"):
        self.model_name = model_name
        self.storage_path = storage_path
        self.tokenizer = tiktoken.encoding_for_model(model_name)
        
        # Initialize LLM for compression
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        
        # Storage for different strategies
        self.scratchpads: Dict[str, List[ScratchpadEntry]] = {}  # agent_id -> entries
        self.isolated_contexts: Dict[str, Dict[ContextType, List[ScratchpadEntry]]] = {}  # agent_id -> type -> entries
        self.compressed_contexts: Dict[str, Dict] = {}  # compression cache
        self.context_budgets: Dict[str, ContextBudget] = {}  # workflow_id -> budget
        
        # Metrics tracking
        self.strategy_metrics = {
            'scratchpad_writes': 0,
            'context_selections': 0,
            'compressions_performed': 0,
            'isolations_created': 0,
            'tokens_saved_compression': 0,
            'tokens_saved_selection': 0
        }
        
        # Ensure storage directory exists
        os.makedirs(storage_path, exist_ok=True)
        
        logger.info(f"🧠 Contextual Engineering System initialized with {model_name}")
    
    # ==========================================
    # STRATEGY 1: WRITE (Scratchpad)
    # ==========================================
    
    def create_scratchpad_entry(self, agent_id: str, content: Dict[str, Any], 
                              context_type: ContextType = ContextType.SCRATCHPAD) -> ScratchpadEntry:
        """
        Strategy 1: Write - Create persistent context storage entry.
        
        Args:
            agent_id: Unique identifier for the agent
            content: Content to store in scratchpad
            context_type: Type of context for organization
            
        Returns:
            ScratchpadEntry: Created entry with metadata
        """
        
        # Calculate token cost
        content_str = json.dumps(content, default=str)
        token_cost = len(self.tokenizer.encode(content_str))
        
        # Create entry
        entry = ScratchpadEntry(
            id=str(uuid.uuid4()),
            agent_id=agent_id,
            timestamp=datetime.now(),
            content=content,
            context_type=context_type,
            token_cost=token_cost
        )
        
        # Store in scratchpad
        if agent_id not in self.scratchpads:
            self.scratchpads[agent_id] = []
        
        self.scratchpads[agent_id].append(entry)
        
        # Update metrics
        self.strategy_metrics['scratchpad_writes'] += 1
        
        # Persist to storage
        self._persist_scratchpad(agent_id)
        
        logger.info(f"📝 Created scratchpad entry for {agent_id}: {token_cost} tokens")
        return entry
    
    def get_scratchpad_context(self, agent_id: str, max_entries: Optional[int] = None) -> List[ScratchpadEntry]:
        """Get scratchpad context for an agent."""
        
        entries = self.scratchpads.get(agent_id, [])
        
        if max_entries:
            # Return most recent entries
            entries = sorted(entries, key=lambda x: x.timestamp, reverse=True)[:max_entries]
        
        # Update access tracking
        for entry in entries:
            entry.access_count += 1
            entry.last_accessed = datetime.now()
        
        return entries
    
    def update_scratchpad_entry(self, entry_id: str, agent_id: str, new_content: Dict[str, Any]) -> bool:
        """Update existing scratchpad entry."""
        
        entries = self.scratchpads.get(agent_id, [])
        
        for entry in entries:
            if entry.id == entry_id:
                # Update content and recalculate token cost
                entry.content.update(new_content)
                content_str = json.dumps(entry.content, default=str)
                entry.token_cost = len(self.tokenizer.encode(content_str))
                entry.timestamp = datetime.now()  # Update timestamp
                
                # Persist changes
                self._persist_scratchpad(agent_id)
                
                logger.info(f"📝 Updated scratchpad entry {entry_id} for {agent_id}")
                return True
        
        return False
    
    # ==========================================
    # STRATEGY 2: SELECT (Context Selection)
    # ==========================================
    
    def select_relevant_context(self, agent_id: str, current_task: str, 
                              context_budget: ContextBudget,
                              context_types: Optional[List[ContextType]] = None) -> Dict[str, Any]:
        """
        Strategy 2: Select - Intelligent relevance-based filtering.
        
        Args:
            agent_id: Agent requesting context
            current_task: Current task description for relevance scoring
            context_budget: Token budget constraints
            context_types: Specific context types to include
            
        Returns:
            Dict containing selected context and metadata
        """
        
        if context_types is None:
            context_types = list(ContextType)
        
        selected_context = {}
        total_tokens_used = 0
        selection_metadata = {
            'total_available_entries': 0,
            'selected_entries': 0,
            'tokens_saved': 0,
            'selection_efficiency': 0.0
        }
        
        for context_type in context_types:
            budget_for_type = context_budget.get_budget_for_type(context_type)
            
            if budget_for_type <= 0:
                continue
            
            # Get available context for this type
            available_entries = self._get_context_by_type(agent_id, context_type)
            selection_metadata['total_available_entries'] += len(available_entries)
            
            if not available_entries:
                continue
            
            # Calculate relevance scores
            scored_entries = self._calculate_relevance_scores(available_entries, current_task)
            
            # Select entries within budget
            selected_entries = []
            type_tokens_used = 0
            
            for entry in scored_entries:
                if type_tokens_used + entry.token_cost <= budget_for_type:
                    selected_entries.append(entry)
                    type_tokens_used += entry.token_cost
                    
                    # Update entry access tracking
                    entry.access_count += 1
                    entry.last_accessed = datetime.now()
                else:
                    break
            
            if selected_entries:
                selected_context[context_type.value] = [entry.to_dict() for entry in selected_entries]
                total_tokens_used += type_tokens_used
                selection_metadata['selected_entries'] += len(selected_entries)
        
        # Calculate selection efficiency
        total_available_tokens = sum(
            sum(entry.token_cost for entry in self._get_context_by_type(agent_id, ct))
            for ct in context_types
        )
        
        if total_available_tokens > 0:
            selection_metadata['tokens_saved'] = total_available_tokens - total_tokens_used
            selection_metadata['selection_efficiency'] = total_tokens_used / total_available_tokens
        
        # Update metrics
        self.strategy_metrics['context_selections'] += 1
        self.strategy_metrics['tokens_saved_selection'] += selection_metadata['tokens_saved']
        
        logger.info(f"🎯 Selected context for {agent_id}: {total_tokens_used}/{context_budget.total_budget} tokens")
        
        return {
            'selected_context': selected_context,
            'tokens_used': total_tokens_used,
            'metadata': selection_metadata
        }
    
    def _calculate_relevance_scores(self, entries: List[ScratchpadEntry], current_task: str) -> List[ScratchpadEntry]:
        """Calculate relevance scores for context entries."""
        
        task_keywords = set(current_task.lower().split())
        
        for entry in entries:
            # Simple relevance scoring based on keyword overlap and recency
            content_str = json.dumps(entry.content, default=str).lower()
            content_keywords = set(content_str.split())
            
            # Keyword overlap score
            overlap_score = len(task_keywords.intersection(content_keywords)) / max(len(task_keywords), 1)
            
            # Recency score (more recent = higher score)
            hours_old = (datetime.now() - entry.timestamp).total_seconds() / 3600
            recency_score = max(0, 1 - (hours_old / 24))  # Decay over 24 hours
            
            # Access frequency score
            frequency_score = min(1.0, entry.access_count / 10)  # Cap at 10 accesses
            
            # Combined relevance score
            entry.relevance_score = (overlap_score * 0.5) + (recency_score * 0.3) + (frequency_score * 0.2)
        
        # Sort by relevance score (highest first)
        return sorted(entries, key=lambda x: x.relevance_score, reverse=True)
    
    # ==========================================
    # STRATEGY 3: COMPRESS (Context Compression)
    # ==========================================
    
    def compress_context(self, context_data: Union[List[ScratchpadEntry], str], 
                        target_tokens: int,
                        strategy: CompressionStrategy = CompressionStrategy.SUMMARIZATION) -> Dict[str, Any]:
        """
        Strategy 3: Compress - Summarization and trimming strategies.
        
        Args:
            context_data: Context to compress (entries or raw text)
            target_tokens: Target token count after compression
            strategy: Compression strategy to use
            
        Returns:
            Dict containing compressed context and metadata
        """
        
        # Convert context to string if needed
        if isinstance(context_data, list):
            context_str = self._entries_to_string(context_data)
            original_entries = context_data
        else:
            context_str = context_data
            original_entries = []
        
        original_tokens = len(self.tokenizer.encode(context_str))
        
        if original_tokens <= target_tokens:
            # No compression needed
            return {
                'compressed_content': context_str,
                'original_tokens': original_tokens,
                'compressed_tokens': original_tokens,
                'compression_ratio': 1.0,
                'strategy_used': 'none',
                'tokens_saved': 0
            }
        
        # Apply compression strategy
        if strategy == CompressionStrategy.SUMMARIZATION:
            compressed_content = self._compress_by_summarization(context_str, target_tokens)
        elif strategy == CompressionStrategy.TRIMMING:
            compressed_content = self._compress_by_trimming(context_str, target_tokens)
        elif strategy == CompressionStrategy.CHUNKING:
            compressed_content = self._compress_by_chunking(context_str, target_tokens)
        elif strategy == CompressionStrategy.HEURISTIC:
            compressed_content = self._compress_by_heuristic(context_str, target_tokens)
        else:
            compressed_content = context_str
        
        compressed_tokens = len(self.tokenizer.encode(compressed_content))
        compression_ratio = compressed_tokens / original_tokens
        tokens_saved = original_tokens - compressed_tokens
        
        # Update metrics
        self.strategy_metrics['compressions_performed'] += 1
        self.strategy_metrics['tokens_saved_compression'] += tokens_saved
        
        # Cache compression result
        cache_key = hashlib.md5(context_str.encode()).hexdigest()
        self.compressed_contexts[cache_key] = {
            'compressed_content': compressed_content,
            'compression_ratio': compression_ratio,
            'strategy': strategy.value,
            'timestamp': datetime.now()
        }
        
        logger.info(f"🗜️ Compressed context: {original_tokens} → {compressed_tokens} tokens ({compression_ratio:.2%})")
        
        return {
            'compressed_content': compressed_content,
            'original_tokens': original_tokens,
            'compressed_tokens': compressed_tokens,
            'compression_ratio': compression_ratio,
            'strategy_used': strategy.value,
            'tokens_saved': tokens_saved
        }
    
    def _compress_by_summarization(self, content: str, target_tokens: int) -> str:
        """Compress content using LLM summarization."""
        
        try:
            # Calculate target length (rough approximation: 1 token ≈ 4 characters)
            target_chars = target_tokens * 4
            
            prompt = f"""
            Please summarize the following content to approximately {target_chars} characters while preserving the most important information:

            {content}

            Summary:
            """
            
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return response.content.strip()
            
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            # Fallback to trimming
            return self._compress_by_trimming(content, target_tokens)
    
    def _compress_by_trimming(self, content: str, target_tokens: int) -> str:
        """Compress content by trimming to token limit."""
        
        tokens = self.tokenizer.encode(content)
        if len(tokens) <= target_tokens:
            return content
        
        # Trim to target tokens
        trimmed_tokens = tokens[:target_tokens]
        return self.tokenizer.decode(trimmed_tokens)
    
    def _compress_by_chunking(self, content: str, target_tokens: int) -> str:
        """Compress content by selecting most relevant chunks."""
        
        # Split into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=target_tokens * 2,  # Larger chunks for selection
            chunk_overlap=50
        )
        
        chunks = splitter.split_text(content)
        
        if len(chunks) <= 1:
            return self._compress_by_trimming(content, target_tokens)
        
        # Select chunks that fit within target
        selected_chunks = []
        current_tokens = 0
        
        for chunk in chunks:
            chunk_tokens = len(self.tokenizer.encode(chunk))
            if current_tokens + chunk_tokens <= target_tokens:
                selected_chunks.append(chunk)
                current_tokens += chunk_tokens
            else:
                break
        
        return " ".join(selected_chunks)
    
    def _compress_by_heuristic(self, content: str, target_tokens: int) -> str:
        """Compress content using heuristic rules."""
        
        lines = content.split('\n')
        
        # Remove empty lines and very short lines
        filtered_lines = [line for line in lines if len(line.strip()) > 10]
        
        # Join and check if within target
        filtered_content = '\n'.join(filtered_lines)
        
        if len(self.tokenizer.encode(filtered_content)) <= target_tokens:
            return filtered_content
        
        # Fallback to trimming
        return self._compress_by_trimming(filtered_content, target_tokens)
    
    # ==========================================
    # STRATEGY 4: ISOLATE (Context Isolation)
    # ==========================================
    
    def create_isolated_context(self, agent_id: str, context_type: ContextType) -> str:
        """
        Strategy 4: Isolate - Create isolated context space for agent and type.
        
        Args:
            agent_id: Agent identifier
            context_type: Type of context to isolate
            
        Returns:
            Isolation key for accessing the isolated context
        """
        
        isolation_key = f"{agent_id}_{context_type.value}"
        
        if agent_id not in self.isolated_contexts:
            self.isolated_contexts[agent_id] = {}
        
        if context_type not in self.isolated_contexts[agent_id]:
            self.isolated_contexts[agent_id][context_type] = []
            
            # Update metrics
            self.strategy_metrics['isolations_created'] += 1
            
            logger.info(f"🔒 Created isolated context: {isolation_key}")
        
        return isolation_key
    
    def add_to_isolated_context(self, agent_id: str, context_type: ContextType, 
                              content: Dict[str, Any]) -> ScratchpadEntry:
        """Add content to isolated context space."""
        
        # Ensure isolation exists
        self.create_isolated_context(agent_id, context_type)
        
        # Create entry
        entry = ScratchpadEntry(
            id=str(uuid.uuid4()),
            agent_id=agent_id,
            timestamp=datetime.now(),
            content=content,
            context_type=context_type,
            token_cost=len(self.tokenizer.encode(json.dumps(content, default=str)))
        )
        
        # Add to isolated context
        self.isolated_contexts[agent_id][context_type].append(entry)
        
        logger.info(f"🔒 Added to isolated context {agent_id}_{context_type.value}: {entry.token_cost} tokens")
        return entry
    
    def get_isolated_context(self, agent_id: str, context_type: ContextType, 
                           max_tokens: Optional[int] = None) -> List[ScratchpadEntry]:
        """Get content from isolated context space."""
        
        if agent_id not in self.isolated_contexts:
            return []
        
        if context_type not in self.isolated_contexts[agent_id]:
            return []
        
        entries = self.isolated_contexts[agent_id][context_type]
        
        if max_tokens:
            # Select entries within token budget
            selected_entries = []
            current_tokens = 0
            
            # Sort by recency (most recent first)
            sorted_entries = sorted(entries, key=lambda x: x.timestamp, reverse=True)
            
            for entry in sorted_entries:
                if current_tokens + entry.token_cost <= max_tokens:
                    selected_entries.append(entry)
                    current_tokens += entry.token_cost
                else:
                    break
            
            return selected_entries
        
        return entries
    
    def prevent_context_contamination(self, agent_id: str, context_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prevent context contamination between agents and types."""
        
        cleaned_context = {}
        
        for key, value in context_data.items():
            # Remove agent-specific identifiers from other agents
            if isinstance(value, str) and agent_id not in value:
                # Check if this contains references to other agents
                other_agent_refs = [aid for aid in self.isolated_contexts.keys() if aid != agent_id and aid in value]
                
                if other_agent_refs:
                    # Remove or sanitize references to other agents
                    sanitized_value = value
                    for other_agent in other_agent_refs:
                        sanitized_value = sanitized_value.replace(other_agent, "[AGENT_REF]")
                    cleaned_context[key] = sanitized_value
                else:
                    cleaned_context[key] = value
            else:
                cleaned_context[key] = value
        
        return cleaned_context
    
    # ==========================================
    # UTILITY METHODS
    # ==========================================
    
    def _get_context_by_type(self, agent_id: str, context_type: ContextType) -> List[ScratchpadEntry]:
        """Get all context entries of a specific type for an agent."""
        
        entries = []
        
        # From scratchpad
        scratchpad_entries = self.scratchpads.get(agent_id, [])
        entries.extend([e for e in scratchpad_entries if e.context_type == context_type])
        
        # From isolated contexts
        if agent_id in self.isolated_contexts and context_type in self.isolated_contexts[agent_id]:
            entries.extend(self.isolated_contexts[agent_id][context_type])
        
        return entries
    
    def _entries_to_string(self, entries: List[ScratchpadEntry]) -> str:
        """Convert scratchpad entries to string format."""
        
        content_parts = []
        for entry in entries:
            content_str = json.dumps(entry.content, default=str, indent=2)
            content_parts.append(f"[{entry.timestamp.isoformat()}] {entry.agent_id}:\n{content_str}")
        
        return "\n\n".join(content_parts)
    
    def _persist_scratchpad(self, agent_id: str):
        """Persist scratchpad to storage."""
        
        try:
            file_path = os.path.join(self.storage_path, f"scratchpad_{agent_id}.json")
            
            entries_data = [entry.to_dict() for entry in self.scratchpads.get(agent_id, [])]
            
            with open(file_path, 'w') as f:
                json.dump(entries_data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to persist scratchpad for {agent_id}: {e}")
    
    def _load_scratchpad(self, agent_id: str):
        """Load scratchpad from storage."""
        
        try:
            file_path = os.path.join(self.storage_path, f"scratchpad_{agent_id}.json")
            
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    entries_data = json.load(f)
                
                self.scratchpads[agent_id] = [ScratchpadEntry.from_dict(data) for data in entries_data]
                logger.info(f"📁 Loaded {len(entries_data)} scratchpad entries for {agent_id}")
                
        except Exception as e:
            logger.error(f"Failed to load scratchpad for {agent_id}: {e}")
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics."""
        
        total_entries = sum(len(entries) for entries in self.scratchpads.values())
        total_isolated_entries = sum(
            sum(len(type_entries) for type_entries in agent_contexts.values())
            for agent_contexts in self.isolated_contexts.values()
        )
        
        return {
            'strategy_metrics': self.strategy_metrics.copy(),
            'storage_metrics': {
                'total_scratchpad_entries': total_entries,
                'total_isolated_entries': total_isolated_entries,
                'active_agents': len(self.scratchpads),
                'isolated_agents': len(self.isolated_contexts),
                'compressed_contexts_cached': len(self.compressed_contexts)
            },
            'efficiency_metrics': {
                'avg_tokens_saved_per_compression': (
                    self.strategy_metrics['tokens_saved_compression'] / 
                    max(self.strategy_metrics['compressions_performed'], 1)
                ),
                'avg_tokens_saved_per_selection': (
                    self.strategy_metrics['tokens_saved_selection'] / 
                    max(self.strategy_metrics['context_selections'], 1)
                )
            }
        }
    
    def cleanup_old_context(self, max_age_hours: int = 24):
        """Clean up old context entries to prevent memory bloat."""
        
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        cleaned_count = 0
        
        # Clean scratchpads
        for agent_id in list(self.scratchpads.keys()):
            original_count = len(self.scratchpads[agent_id])
            self.scratchpads[agent_id] = [
                entry for entry in self.scratchpads[agent_id]
                if entry.timestamp > cutoff_time
            ]
            cleaned_count += original_count - len(self.scratchpads[agent_id])
        
        # Clean isolated contexts
        for agent_id in self.isolated_contexts:
            for context_type in self.isolated_contexts[agent_id]:
                original_count = len(self.isolated_contexts[agent_id][context_type])
                self.isolated_contexts[agent_id][context_type] = [
                    entry for entry in self.isolated_contexts[agent_id][context_type]
                    if entry.timestamp > cutoff_time
                ]
                cleaned_count += original_count - len(self.isolated_contexts[agent_id][context_type])
        
        # Clean compressed contexts cache
        original_cache_size = len(self.compressed_contexts)
        self.compressed_contexts = {
            key: value for key, value in self.compressed_contexts.items()
            if value['timestamp'] > cutoff_time
        }
        cleaned_count += original_cache_size - len(self.compressed_contexts)
        
        logger.info(f"🧹 Cleaned up {cleaned_count} old context entries")
        return cleaned_count

# Global instance
_global_contextual_system = None

def get_contextual_engineering_system() -> ContextualEngineeringSystem:
    """Get the global contextual engineering system instance."""
    global _global_contextual_system
    if _global_contextual_system is None:
        _global_contextual_system = ContextualEngineeringSystem()
    return _global_contextual_system