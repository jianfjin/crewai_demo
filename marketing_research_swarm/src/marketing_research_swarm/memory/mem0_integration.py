"""
Memory Integration with Mem0

This module provides integration with Mem0 for persistent memory management
across marketing research analysis sessions. It enables the system to remember
insights, patterns, and context from previous analyses.

Key Features:
- Persistent memory across sessions
- Context-aware memory retrieval
- Automatic memory organization
- Integration with CrewAI agents
- Local memory storage
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import os
from mem0 import Memory

logger = logging.getLogger(__name__)

class Mem0Integration:
    """
    Integration class for Mem0 memory management.
    
    This class provides methods to store, retrieve, and manage memories
    for marketing research analysis sessions using local Mem0 storage.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Mem0 integration with local storage.
        
        Args:
            config: Optional configuration for Mem0 client
        """
        try:
            # Default configuration for local Mem0
            default_config = {
                "vector_store": {
                    "provider": "chroma",
                    "config": {
                        "collection_name": "marketing_research_memory",
                        "path": "./db"
                    }
                },
                "embedder": {
                    "provider": "openai",
                    "config": {
                        "model": "text-embedding-ada-002"
                    }
                }
            }
            
            # Use provided config or default
            self.config = config or default_config
            
            # Initialize local Memory client
            self.memory = Memory.from_config(self.config)
            
            logger.info("✅ Mem0 integration initialized successfully with local storage")
            
        except Exception as e:
            logger.error(f"❌ Error initializing Mem0 integration: {str(e)}")
            # Fallback to basic memory without Mem0
            self.memory = None
            self._fallback_memory = {}
    
    def add_memory(self, content: str, user_id: str = "marketing_analyst", 
                   metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add a memory to the Mem0 storage.
        
        Args:
            content: The content to store in memory
            user_id: User identifier for the memory
            metadata: Additional metadata for the memory
            
        Returns:
            bool: True if memory was added successfully
        """
        try:
            if self.memory is None:
                # Fallback storage
                memory_id = f"{user_id}_{datetime.now().timestamp()}"
                self._fallback_memory[memory_id] = {
                    "content": content,
                    "user_id": user_id,
                    "metadata": metadata or {},
                    "timestamp": datetime.now().isoformat()
                }
                logger.info(f"📝 Memory stored in fallback storage: {memory_id}")
                return True
            
            # Add memory using Mem0
            result = self.memory.add(
                messages=[{"role": "user", "content": content}],
                user_id=user_id,
                metadata=metadata or {}
            )
            
            logger.info(f"📝 Memory added successfully for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error adding memory: {str(e)}")
            return False
    
    def search_memories(self, query: str, user_id: str = "marketing_analyst", 
                       limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for relevant memories based on a query.
        
        Args:
            query: Search query
            user_id: User identifier
            limit: Maximum number of memories to return
            
        Returns:
            List of relevant memories
        """
        try:
            if self.memory is None:
                # Fallback search
                results = []
                for memory_id, memory_data in self._fallback_memory.items():
                    if (memory_data["user_id"] == user_id and 
                        query.lower() in memory_data["content"].lower()):
                        results.append({
                            "id": memory_id,
                            "memory": memory_data["content"],
                            "metadata": memory_data["metadata"],
                            "score": 0.8  # Default score for fallback
                        })
                        if len(results) >= limit:
                            break
                
                logger.info(f"🔍 Found {len(results)} memories in fallback storage")
                return results
            
            # Search using Mem0
            results = self.memory.search(
                query=query,
                user_id=user_id,
                limit=limit
            )
            
            logger.info(f"🔍 Found {len(results)} relevant memories for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error searching memories: {str(e)}")
            return []
    
    def get_all_memories(self, user_id: str = "marketing_analyst") -> List[Dict[str, Any]]:
        """
        Get all memories for a specific user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of all memories for the user
        """
        try:
            if self.memory is None:
                # Fallback retrieval
                results = []
                for memory_id, memory_data in self._fallback_memory.items():
                    if memory_data["user_id"] == user_id:
                        results.append({
                            "id": memory_id,
                            "memory": memory_data["content"],
                            "metadata": memory_data["metadata"]
                        })
                
                logger.info(f"📚 Retrieved {len(results)} memories from fallback storage")
                return results
            
            # Get all memories using Mem0
            results = self.memory.get_all(user_id=user_id)
            
            logger.info(f"📚 Retrieved {len(results)} memories for user {user_id}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error retrieving memories: {str(e)}")
            return []
    
    def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a specific memory.
        
        Args:
            memory_id: ID of the memory to delete
            
        Returns:
            bool: True if memory was deleted successfully
        """
        try:
            if self.memory is None:
                # Fallback deletion
                if memory_id in self._fallback_memory:
                    del self._fallback_memory[memory_id]
                    logger.info(f"🗑️ Memory deleted from fallback storage: {memory_id}")
                    return True
                return False
            
            # Delete using Mem0
            self.memory.delete(memory_id=memory_id)
            
            logger.info(f"🗑️ Memory deleted successfully: {memory_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error deleting memory: {str(e)}")
            return False
    
    def add_analysis_insight(self, analysis_type: str, insight: str, 
                           data_characteristics: Dict[str, Any],
                           user_id: str = "marketing_analyst") -> bool:
        """
        Add an analysis insight to memory with structured metadata.
        
        Args:
            analysis_type: Type of analysis (roi, sales_forecast, brand_performance)
            insight: The insight content
            data_characteristics: Characteristics of the data analyzed
            user_id: User identifier
            
        Returns:
            bool: True if insight was added successfully
        """
        try:
            metadata = {
                "type": "analysis_insight",
                "analysis_type": analysis_type,
                "timestamp": datetime.now().isoformat(),
                "data_characteristics": data_characteristics
            }
            
            content = f"Analysis Insight ({analysis_type}): {insight}"
            
            return self.add_memory(content, user_id, metadata)
            
        except Exception as e:
            logger.error(f"❌ Error adding analysis insight: {str(e)}")
            return False
    
    def get_relevant_insights(self, analysis_type: str, 
                            data_characteristics: Dict[str, Any],
                            user_id: str = "marketing_analyst",
                            limit: int = 3) -> List[Dict[str, Any]]:
        """
        Get relevant insights for a specific analysis type and data characteristics.
        
        Args:
            analysis_type: Type of analysis
            data_characteristics: Characteristics of the current data
            user_id: User identifier
            limit: Maximum number of insights to return
            
        Returns:
            List of relevant insights
        """
        try:
            # Create search query based on analysis type and data characteristics
            query_parts = [analysis_type]
            
            if "industry" in data_characteristics:
                query_parts.append(data_characteristics["industry"])
            if "region" in data_characteristics:
                query_parts.append(data_characteristics["region"])
            if "time_period" in data_characteristics:
                query_parts.append(data_characteristics["time_period"])
            
            query = " ".join(query_parts)
            
            # Search for relevant memories
            memories = self.search_memories(query, user_id, limit * 2)  # Get more to filter
            
            # Filter for analysis insights
            insights = []
            for memory in memories:
                metadata = memory.get("metadata", {})
                if (metadata.get("type") == "analysis_insight" and 
                    metadata.get("analysis_type") == analysis_type):
                    insights.append(memory)
                    if len(insights) >= limit:
                        break
            
            logger.info(f"🧠 Found {len(insights)} relevant insights for {analysis_type}")
            return insights
            
        except Exception as e:
            logger.error(f"❌ Error getting relevant insights: {str(e)}")
            return []
    
    def add_pattern_discovery(self, pattern: str, context: Dict[str, Any],
                            user_id: str = "marketing_analyst") -> bool:
        """
        Add a discovered pattern to memory.
        
        Args:
            pattern: Description of the discovered pattern
            context: Context in which the pattern was discovered
            user_id: User identifier
            
        Returns:
            bool: True if pattern was added successfully
        """
        try:
            metadata = {
                "type": "pattern_discovery",
                "timestamp": datetime.now().isoformat(),
                "context": context
            }
            
            content = f"Pattern Discovery: {pattern}"
            
            return self.add_memory(content, user_id, metadata)
            
        except Exception as e:
            logger.error(f"❌ Error adding pattern discovery: {str(e)}")
            return False
    
    def get_memory_stats(self, user_id: str = "marketing_analyst") -> Dict[str, Any]:
        """
        Get statistics about stored memories.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary with memory statistics
        """
        try:
            memories = self.get_all_memories(user_id)
            
            stats = {
                "total_memories": len(memories),
                "analysis_insights": 0,
                "pattern_discoveries": 0,
                "other_memories": 0,
                "memory_types": {}
            }
            
            for memory in memories:
                metadata = memory.get("metadata", {})
                memory_type = metadata.get("type", "other")
                
                if memory_type == "analysis_insight":
                    stats["analysis_insights"] += 1
                elif memory_type == "pattern_discovery":
                    stats["pattern_discoveries"] += 1
                else:
                    stats["other_memories"] += 1
                
                stats["memory_types"][memory_type] = stats["memory_types"].get(memory_type, 0) + 1
            
            logger.info(f"📊 Memory stats: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error getting memory stats: {str(e)}")
            return {
                "total_memories": 0,
                "error": str(e)
            }

# Backward compatibility alias
MarketingMemoryManager = Mem0Integration
