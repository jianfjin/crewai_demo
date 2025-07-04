"""
Mem0 Integration for Long-term Memory Management
Reduces token usage through intelligent memory storage and retrieval
"""

from typing import Dict, Any, List, Optional
import time
import json
import hashlib
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

load_dotenv()

try:
    from mem0 import MemoryClient
    MEM0_AVAILABLE = True
except ImportError:
    MEM0_AVAILABLE = False
    print("⚠️  Mem0 not available. Install with: pip install mem0ai")

class MockMemoryClient:
    """Mock client for when Mem0 is not available"""
    
    def __init__(self):
        self.memories = {}
    
    def add(self, messages: List[Dict], user_id: str, metadata: Dict = None):
        """Mock add method"""
        key = f"{user_id}_{int(time.time())}"
        self.memories[key] = {
            'messages': messages,
            'metadata': metadata or {},
            'created': time.time()
        }
        return {'id': key}
    
    def search(self, query: str, user_id: str, limit: int = 5):
        """Mock search method"""
        # Simple keyword matching
        results = []
        for key, memory in self.memories.items():
            if user_id in key:
                for message in memory['messages']:
                    if any(word.lower() in message.get('content', '').lower() 
                          for word in query.split()):
                        results.append({
                            'id': key,
                            'memory': message['content'],
                            'score': 0.8,
                            'metadata': memory['metadata']
                        })
        
        return results[:limit]
    
    def get_all(self, user_id: str):
        """Mock get all method"""
        return [memory for key, memory in self.memories.items() if user_id in key]

class MarketingMemoryManager:
    """
    Intelligent memory management using Mem0 for long-term context storage
    """
    
    def __init__(self, use_mock: bool = False):
        if MEM0_AVAILABLE and not use_mock:
            self.memory_client = MemoryClient(api_key=os.getenv("MEM0_API_KEY"))
        else:
            self.memory_client = MockMemoryClient()
            if not use_mock:
                print("🔄 Using mock memory client (Mem0 not available)")
        
        self.session_id = f"marketing_research_{int(time.time())}"
        self.insight_cache = {}
        
    def store_analysis_insights(self, analysis_type: str, insights: Dict[str, Any], 
                              metadata: Dict[str, Any] = None) -> str:
        """
        Store key insights from analysis in long-term memory
        
        Args:
            analysis_type: Type of analysis (roi, forecast, brand)
            insights: Analysis insights to store
            metadata: Additional metadata
            
        Returns:
            Memory ID for reference
        """
        # Extract and compress key insights
        compressed_insights = self._compress_insights(insights)
        
        # Create memory entry
        memory_content = self._format_insights_for_storage(analysis_type, compressed_insights)
        
        # Prepare metadata
        storage_metadata = {
            'analysis_type': analysis_type,
            'timestamp': time.time(),
            'session_id': self.session_id,
            'insight_count': len(compressed_insights),
            'original_size': len(str(insights)),
            'compressed_size': len(str(compressed_insights))
        }
        
        if metadata:
            storage_metadata.update(metadata)
        
        # Store in Mem0
        try:
            result = self.memory_client.add(
                messages=[{
                    'role': 'assistant',
                    'content': memory_content
                }],
                user_id=f"marketing_{analysis_type}",
                metadata=storage_metadata
            )
            
            memory_id = result.get('id', f"mem_{int(time.time())}")
            
            # Cache locally for quick access
            self.insight_cache[memory_id] = {
                'insights': compressed_insights,
                'metadata': storage_metadata,
                'created': time.time()
            }
            
            print(f"💾 Stored insights in memory: {memory_id}")
            return memory_id
            
        except Exception as e:
            print(f"⚠️  Error storing insights: {e}")
            return self._store_locally(analysis_type, compressed_insights, storage_metadata)
    
    def get_relevant_context(self, query: str, analysis_type: str, 
                           max_tokens: int = 500) -> Dict[str, Any]:
        """
        Retrieve relevant context for current analysis
        
        Args:
            query: Current analysis query/context
            analysis_type: Type of analysis
            max_tokens: Maximum tokens to return
            
        Returns:
            Relevant context dictionary
        """
        try:
            # Search for relevant memories
            memories = self.memory_client.search(
                query=query,
                user_id=f"marketing_{analysis_type}",
                limit=5
            )
            
            # Process and compress results
            relevant_context = self._process_search_results(memories, max_tokens)
            
            print(f"🔍 Retrieved {len(relevant_context.get('insights', []))} relevant insights")
            return relevant_context
            
        except Exception as e:
            print(f"⚠️  Error retrieving context: {e}")
            return self._get_local_context(query, analysis_type, max_tokens)
    
    def get_historical_patterns(self, analysis_type: str, 
                              lookback_days: int = 30) -> Dict[str, Any]:
        """
        Get historical patterns for the analysis type
        
        Args:
            analysis_type: Type of analysis
            lookback_days: How many days to look back
            
        Returns:
            Historical patterns and trends
        """
        cutoff_time = time.time() - (lookback_days * 24 * 3600)
        
        try:
            # Get all memories for analysis type
            all_memories = self.memory_client.get_all(f"marketing_{analysis_type}")
            
            # Filter by time and extract patterns
            recent_memories = [
                memory for memory in all_memories
                if memory.get('metadata', {}).get('timestamp', 0) > cutoff_time
            ]
            
            patterns = self._extract_patterns(recent_memories)
            
            print(f"📊 Extracted patterns from {len(recent_memories)} historical analyses")
            return patterns
            
        except Exception as e:
            print(f"⚠️  Error getting historical patterns: {e}")
            return {'patterns': [], 'error': str(e)}
    
    def cleanup_old_memories(self, retention_days: int = 90) -> int:
        """
        Clean up old memories to prevent storage bloat
        
        Args:
            retention_days: Days to retain memories
            
        Returns:
            Number of memories cleaned up
        """
        cutoff_time = time.time() - (retention_days * 24 * 3600)
        cleaned_count = 0
        
        # Clean local cache
        expired_keys = [
            key for key, value in self.insight_cache.items()
            if value['created'] < cutoff_time
        ]
        
        for key in expired_keys:
            del self.insight_cache[key]
            cleaned_count += 1
        
        print(f"🧹 Cleaned up {cleaned_count} old memories")
        return cleaned_count
    
    def _compress_insights(self, insights: Dict[str, Any]) -> Dict[str, Any]:
        """Compress insights to essential information"""
        compressed = {}
        
        # Extract key metrics
        for key, value in insights.items():
            if self._is_key_insight(key, value):
                compressed[key] = self._compress_value(value)
        
        # Limit size
        if len(str(compressed)) > 1000:  # 1KB limit
            compressed = self._further_compress(compressed)
        
        return compressed
    
    def _is_key_insight(self, key: str, value: Any) -> bool:
        """Determine if insight is worth storing"""
        key_lower = key.lower()
        
        # Important keywords
        important_keywords = [
            'top', 'best', 'highest', 'lowest', 'key', 'main', 'primary',
            'recommendation', 'insight', 'finding', 'result', 'metric',
            'performance', 'roi', 'margin', 'revenue', 'profit'
        ]
        
        # Check if key contains important keywords
        if any(keyword in key_lower for keyword in important_keywords):
            return True
        
        # Check if value is a meaningful metric
        if isinstance(value, (int, float)) and value != 0:
            return True
        
        # Check if value is a short, meaningful string
        if isinstance(value, str) and 10 <= len(value) <= 200:
            return True
        
        return False
    
    def _compress_value(self, value: Any) -> Any:
        """Compress individual values"""
        if isinstance(value, str):
            # Keep only first 100 characters for strings
            return value[:100] + "..." if len(value) > 100 else value
        elif isinstance(value, dict):
            # Keep only first 3 items for dictionaries
            return {k: v for k, v in list(value.items())[:3]}
        elif isinstance(value, list):
            # Keep only first 3 items for lists
            return value[:3]
        else:
            return value
    
    def _further_compress(self, compressed: Dict[str, Any]) -> Dict[str, Any]:
        """Further compress if still too large"""
        # Keep only the most important items
        priority_keys = []
        
        for key in compressed.keys():
            key_lower = key.lower()
            if any(priority in key_lower for priority in ['top', 'best', 'key', 'main']):
                priority_keys.append(key)
        
        # If we have priority keys, keep only those
        if priority_keys:
            return {k: compressed[k] for k in priority_keys[:5]}
        
        # Otherwise, keep first 5 items
        return {k: v for k, v in list(compressed.items())[:5]}
    
    def _format_insights_for_storage(self, analysis_type: str, insights: Dict[str, Any]) -> str:
        """Format insights for memory storage"""
        formatted_parts = [f"Analysis Type: {analysis_type}"]
        
        for key, value in insights.items():
            formatted_parts.append(f"{key}: {value}")
        
        return "; ".join(formatted_parts)
    
    def _process_search_results(self, memories: List[Dict], max_tokens: int) -> Dict[str, Any]:
        """Process search results and compress to token limit"""
        relevant_insights = []
        current_tokens = 0
        
        for memory in memories:
            memory_content = memory.get('memory', '')
            memory_tokens = len(memory_content.split()) * 1.3  # Rough estimation
            
            if current_tokens + memory_tokens <= max_tokens:
                relevant_insights.append({
                    'content': memory_content,
                    'score': memory.get('score', 0),
                    'metadata': memory.get('metadata', {})
                })
                current_tokens += memory_tokens
            else:
                # Add truncated version if space allows
                if current_tokens + 50 <= max_tokens:  # Space for summary
                    truncated = memory_content[:100] + "..."
                    relevant_insights.append({
                        'content': truncated,
                        'score': memory.get('score', 0),
                        'metadata': memory.get('metadata', {}),
                        'truncated': True
                    })
                break
        
        return {
            'insights': relevant_insights,
            'total_memories': len(memories),
            'included_memories': len(relevant_insights),
            'estimated_tokens': int(current_tokens)
        }
    
    def _store_locally(self, analysis_type: str, insights: Dict[str, Any], 
                      metadata: Dict[str, Any]) -> str:
        """Store insights locally as fallback"""
        memory_id = f"local_{analysis_type}_{int(time.time())}"
        
        self.insight_cache[memory_id] = {
            'insights': insights,
            'metadata': metadata,
            'created': time.time()
        }
        
        return memory_id
    
    def _get_local_context(self, query: str, analysis_type: str, max_tokens: int) -> Dict[str, Any]:
        """Get context from local cache as fallback"""
        relevant_insights = []
        current_tokens = 0
        
        query_words = query.lower().split()
        
        for memory_id, memory_data in self.insight_cache.items():
            if analysis_type in memory_id:
                insights = memory_data['insights']
                insight_text = str(insights)
                
                # Simple relevance scoring
                score = sum(1 for word in query_words if word in insight_text.lower())
                
                if score > 0:
                    insight_tokens = len(insight_text.split()) * 1.3
                    
                    if current_tokens + insight_tokens <= max_tokens:
                        relevant_insights.append({
                            'content': insight_text,
                            'score': score,
                            'metadata': memory_data['metadata']
                        })
                        current_tokens += insight_tokens
        
        return {
            'insights': relevant_insights,
            'total_memories': len(self.insight_cache),
            'included_memories': len(relevant_insights),
            'estimated_tokens': int(current_tokens),
            'source': 'local_cache'
        }
    
    def _extract_patterns(self, memories: List[Dict]) -> Dict[str, Any]:
        """Extract patterns from historical memories"""
        patterns = {
            'common_insights': [],
            'trending_metrics': [],
            'frequent_recommendations': [],
            'analysis_frequency': {}
        }
        
        # Analyze memories for patterns
        all_content = []
        for memory in memories:
            content = memory.get('memory', '')
            all_content.append(content)
            
            # Track analysis frequency
            analysis_type = memory.get('metadata', {}).get('analysis_type', 'unknown')
            patterns['analysis_frequency'][analysis_type] = patterns['analysis_frequency'].get(analysis_type, 0) + 1
        
        # Find common phrases/insights
        if all_content:
            combined_text = ' '.join(all_content).lower()
            words = combined_text.split()
            
            # Find frequent important terms
            important_terms = ['roi', 'margin', 'revenue', 'profit', 'performance', 'optimization']
            for term in important_terms:
                if words.count(term) >= 2:  # Appears in multiple analyses
                    patterns['trending_metrics'].append(term)
        
        return patterns
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about memory usage"""
        return {
            'local_cache_size': len(self.insight_cache),
            'session_id': self.session_id,
            'mem0_available': MEM0_AVAILABLE,
            'total_stored_insights': sum(
                len(data['insights']) for data in self.insight_cache.values()
            ),
            'oldest_memory': min(
                (data['created'] for data in self.insight_cache.values()),
                default=time.time()
            ),
            'newest_memory': max(
                (data['created'] for data in self.insight_cache.values()),
                default=time.time()
            )
        }