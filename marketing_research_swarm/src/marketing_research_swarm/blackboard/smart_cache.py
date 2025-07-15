"""
Smart Cache System for Blackboard Optimization
Prevents tool re-execution and enables intelligent result sharing between agents
"""

import hashlib
import json
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class CacheEntry:
    """Represents a cached tool result"""
    tool_name: str
    parameters: Dict[str, Any]
    result: Any
    timestamp: datetime
    agent_name: str
    execution_time: float
    cache_key: str
    dependencies: List[str] = None  # Other cache keys this result depends on

class SmartCache:
    """Intelligent cache system for tool results with dependency tracking"""
    
    def __init__(self, ttl_minutes: int = 60):
        self.cache: Dict[str, CacheEntry] = {}
        self.ttl = timedelta(minutes=ttl_minutes)
        self.access_count: Dict[str, int] = {}
        self.dependency_graph: Dict[str, List[str]] = {}  # cache_key -> dependent_keys
        
    def _generate_cache_key(self, tool_name: str, parameters: Dict[str, Any]) -> str:
        """Generate a unique cache key for tool + parameters"""
        # Sort parameters for consistent hashing
        sorted_params = json.dumps(parameters, sort_keys=True, default=str)
        param_hash = hashlib.md5(sorted_params.encode()).hexdigest()[:8]
        return f"{tool_name}_{param_hash}"
    
    def get_cached_result(self, tool_name: str, parameters: Dict[str, Any]) -> Optional[Any]:
        """Get cached result if available and valid"""
        cache_key = self._generate_cache_key(tool_name, parameters)
        
        if cache_key in self.cache:
            entry = self.cache[cache_key]
            
            # Check if cache entry is still valid
            if datetime.now() - entry.timestamp < self.ttl:
                self.access_count[cache_key] = self.access_count.get(cache_key, 0) + 1
                print(f"🎯 Cache HIT: {tool_name} (saved execution)")
                return entry.result
            else:
                # Remove expired entry
                self._remove_cache_entry(cache_key)
        
        print(f"❌ Cache MISS: {tool_name} (will execute)")
        return None
    
    def store_result(self, tool_name: str, parameters: Dict[str, Any], result: Any, 
                    agent_name: str, execution_time: float, dependencies: List[str] = None) -> str:
        """Store tool result in cache"""
        cache_key = self._generate_cache_key(tool_name, parameters)
        
        entry = CacheEntry(
            tool_name=tool_name,
            parameters=parameters,
            result=result,
            timestamp=datetime.now(),
            agent_name=agent_name,
            execution_time=execution_time,
            cache_key=cache_key,
            dependencies=dependencies or []
        )
        
        self.cache[cache_key] = entry
        self.access_count[cache_key] = 1
        
        # Update dependency graph
        if dependencies:
            for dep_key in dependencies:
                if dep_key not in self.dependency_graph:
                    self.dependency_graph[dep_key] = []
                self.dependency_graph[dep_key].append(cache_key)
        
        print(f"💾 Cached: {tool_name} -> {cache_key}")
        return cache_key
    
    def get_related_results(self, agent_name: str, tool_types: List[str] = None) -> Dict[str, Any]:
        """Get results from previous agents that current agent can use"""
        related_results = {}
        
        for cache_key, entry in self.cache.items():
            # Skip results from the same agent
            if entry.agent_name == agent_name:
                continue
                
            # Filter by tool types if specified
            if tool_types and entry.tool_name not in tool_types:
                continue
            
            # Check if result is still valid
            if datetime.now() - entry.timestamp < self.ttl:
                related_results[f"{entry.agent_name}_{entry.tool_name}"] = {
                    'result': entry.result,
                    'agent': entry.agent_name,
                    'tool': entry.tool_name,
                    'timestamp': entry.timestamp.isoformat(),
                    'cache_key': cache_key
                }
        
        return related_results
    
    def invalidate_dependencies(self, cache_key: str):
        """Invalidate all cache entries that depend on the given key"""
        if cache_key in self.dependency_graph:
            dependent_keys = self.dependency_graph[cache_key]
            for dep_key in dependent_keys:
                if dep_key in self.cache:
                    print(f"🗑️ Invalidating dependent cache: {dep_key}")
                    self._remove_cache_entry(dep_key)
    
    def _remove_cache_entry(self, cache_key: str):
        """Remove cache entry and clean up references"""
        if cache_key in self.cache:
            del self.cache[cache_key]
        if cache_key in self.access_count:
            del self.access_count[cache_key]
        if cache_key in self.dependency_graph:
            del self.dependency_graph[cache_key]
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_entries = len(self.cache)
        total_accesses = sum(self.access_count.values())
        
        # Calculate hit ratio
        hit_ratio = 0.0
        if total_accesses > 0:
            hits = sum(count - 1 for count in self.access_count.values() if count > 1)
            hit_ratio = hits / total_accesses
        
        # Get most accessed entries
        top_entries = sorted(
            [(key, count) for key, count in self.access_count.items()],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        return {
            'total_entries': total_entries,
            'total_accesses': total_accesses,
            'hit_ratio': hit_ratio,
            'top_accessed': top_entries,
            'cache_size_mb': self._estimate_cache_size(),
            'dependency_chains': len(self.dependency_graph)
        }
    
    def _estimate_cache_size(self) -> float:
        """Estimate cache size in MB"""
        try:
            import sys
            total_size = 0
            for entry in self.cache.values():
                total_size += sys.getsizeof(entry.result)
                total_size += sys.getsizeof(entry.parameters)
            return total_size / (1024 * 1024)  # Convert to MB
        except:
            return 0.0
    
    def cleanup_expired(self):
        """Remove expired cache entries"""
        current_time = datetime.now()
        expired_keys = []
        
        for cache_key, entry in self.cache.items():
            if current_time - entry.timestamp >= self.ttl:
                expired_keys.append(cache_key)
        
        for key in expired_keys:
            self._remove_cache_entry(key)
        
        if expired_keys:
            print(f"🧹 Cleaned up {len(expired_keys)} expired cache entries")
    
    def clear_cache(self):
        """Clear all cache entries"""
        self.cache.clear()
        self.access_count.clear()
        self.dependency_graph.clear()
        print("🗑️ Cache cleared")

# Global cache instance
_global_smart_cache = None

def get_smart_cache() -> SmartCache:
    """Get global smart cache instance"""
    global _global_smart_cache
    if _global_smart_cache is None:
        _global_smart_cache = SmartCache()
    return _global_smart_cache