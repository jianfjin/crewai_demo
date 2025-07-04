"""
Persistent Analysis Cache System
Stores intermediate and final analysis results for future retrieval
"""

import hashlib
import json
import pickle
import time
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import os
import sqlite3
from pathlib import Path

from marketing_research_swarm.memory.mem0_integration import MarketingMemoryManager

class AnalysisCacheManager:
    """
    Manages persistent caching of analysis results with multiple storage backends
    """
    
    def __init__(self, cache_dir: str = "cache/analysis", use_mem0: bool = True):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize SQLite database for metadata
        self.db_path = self.cache_dir / "analysis_cache.db"
        self._init_database()
        
        # Initialize Mem0 for semantic caching with updated interface
        self.memory_manager = MarketingMemoryManager() if use_mem0 else None
        
        # Cache configuration
        self.default_ttl = 7 * 24 * 3600  # 7 days
        self.max_cache_size_gb = 5  # 5GB limit
        
    def _init_database(self):
        """Initialize SQLite database for cache metadata"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache_entries (
                    request_hash TEXT PRIMARY KEY,
                    analysis_type TEXT NOT NULL,
                    data_path TEXT NOT NULL,
                    parameters TEXT NOT NULL,
                    result_path TEXT NOT NULL,
                    cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    file_size_bytes INTEGER DEFAULT 0,
                    tags TEXT DEFAULT ''
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_analysis_type ON cache_entries(analysis_type)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_expires_at ON cache_entries(expires_at)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cached_at ON cache_entries(cached_at)
            """)
            
            conn.commit()
    
    def generate_request_hash(self, analysis_type: str, data_path: str,
                            parameters: Dict[str, Any]) -> str:
        """
        Generate a unique hash for the analysis request
        
        Args:
            analysis_type: Type of analysis
            data_path: Path to the data file
            parameters: Analysis parameters
            
        Returns:
            Unique hash string for the request
        """
        # Create a deterministic representation of the request
        request_data = {
            'analysis_type': analysis_type,
            'data_path': data_path,
            'parameters': parameters
        }
        
        # Sort parameters to ensure consistent hashing
        sorted_request = json.dumps(request_data, sort_keys=True)
        
        # Generate SHA-256 hash
        return hashlib.sha256(sorted_request.encode()).hexdigest()
    
    def cache_analysis_result(self, request_hash: str, analysis_type: str,
                            result: Any, parameters: Dict[str, Any],
                            ttl_seconds: Optional[int] = None) -> bool:
        """
        Cache analysis result with metadata
        
        Args:
            request_hash: Unique hash for the request
            analysis_type: Type of analysis
            result: Analysis result to cache
            parameters: Analysis parameters
            ttl_seconds: Time to live in seconds
            
        Returns:
            True if cached successfully
        """
        try:
            # Calculate expiration time
            ttl = ttl_seconds or self.default_ttl
            expires_at = datetime.now() + timedelta(seconds=ttl)
            
            # Store result as pickle file
            result_filename = f"{request_hash}.pkl"
            result_path = self.cache_dir / result_filename
            
            with open(result_path, 'wb') as f:
                pickle.dump(result, f)
            
            # Get file size
            file_size = result_path.stat().st_size
            
            # Store metadata in database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO cache_entries 
                    (request_hash, analysis_type, data_path, parameters, result_path,
                     expires_at, file_size_bytes, tags)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    request_hash,
                    analysis_type,
                    parameters.get('data_file_path', ''),
                    json.dumps(parameters),
                    str(result_path),
                    expires_at.isoformat(),
                    file_size,
                    self._generate_tags(analysis_type, parameters)
                ))
                conn.commit()
            
            # Store in Mem0 for semantic search
            if self.memory_manager:
                self._store_in_mem0(request_hash, analysis_type, result, parameters)
            
            return True
            
        except Exception as e:
            print(f"Error caching analysis result: {e}")
            return False
    
    def get_cached_result(self, request_hash: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached analysis result
        
        Args:
            request_hash: Unique hash for the request
            
        Returns:
            Cached result or None if not found/expired
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT result_path, expires_at, cached_at
                    FROM cache_entries 
                    WHERE request_hash = ?
                """, (request_hash,))
                
                row = cursor.fetchone()
                if not row:
                    return None
                
                result_path, expires_at_str, cached_at_str = row
                
                # Check if expired
                expires_at = datetime.fromisoformat(expires_at_str)
                if datetime.now() > expires_at:
                    # Clean up expired entry
                    self._remove_cache_entry(request_hash)
                    return None
                
                # Load result from file
                result_file = Path(result_path)
                if not result_file.exists():
                    # Clean up orphaned database entry
                    self._remove_cache_entry(request_hash)
                    return None
                
                with open(result_file, 'rb') as f:
                    result = pickle.load(f)
                
                # Update access statistics
                conn.execute("""
                    UPDATE cache_entries 
                    SET access_count = access_count + 1,
                        last_accessed = CURRENT_TIMESTAMP
                    WHERE request_hash = ?
                """, (request_hash,))
                conn.commit()
                
                return {
                    'result': result,
                    'cached_at': cached_at_str,
                    'request_hash': request_hash
                }
                
        except Exception as e:
            print(f"Error retrieving cached result: {e}")
            return None
    
    def find_similar_analyses(self, analysis_type: str, data_path: str,
                            parameters: Dict[str, Any], 
                            similarity_threshold: float = 0.8) -> List[Dict[str, Any]]:
        """
        Find similar cached analyses using Mem0 semantic search
        
        Args:
            analysis_type: Type of analysis
            data_path: Path to data file
            parameters: Analysis parameters
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of similar cached analyses
        """
        if not self.memory_manager:
            return []
        
        try:
            # Search for similar analyses in Mem0
            similar_results = self._search_mem0_similar(analysis_type, parameters)
            
            # Filter by similarity threshold and validate cache entries
            valid_results = []
            for result in similar_results:
                if result.get('score', 0) >= similarity_threshold:
                    request_hash = result.get('metadata', {}).get('request_hash')
                    if request_hash:
                        cached_result = self.get_cached_result(request_hash)
                        if cached_result:
                            valid_results.append({
                                'request_hash': request_hash,
                                'similarity_score': result.get('score', 0),
                                'cached_result': cached_result,
                                'metadata': result.get('metadata', {})
                            })
            
            return valid_results
            
        except Exception as e:
            print(f"Error finding similar analyses: {e}")
            return []
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics
        
        Returns:
            Dictionary with cache statistics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Total entries
                cursor = conn.execute("SELECT COUNT(*) FROM cache_entries")
                total_entries = cursor.fetchone()[0]
                
                # Storage usage
                cursor = conn.execute("SELECT SUM(file_size_bytes) FROM cache_entries")
                total_size_bytes = cursor.fetchone()[0] or 0
                
                # Analysis type breakdown
                cursor = conn.execute("""
                    SELECT analysis_type, COUNT(*) 
                    FROM cache_entries 
                    GROUP BY analysis_type
                """)
                type_breakdown = dict(cursor.fetchall())
                
                # Recent activity
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM cache_entries 
                    WHERE last_accessed > datetime('now', '-7 days')
                """)
                recent_access_count = cursor.fetchone()[0]
                
                # Average access count
                cursor = conn.execute("SELECT AVG(access_count) FROM cache_entries")
                avg_access_count = cursor.fetchone()[0] or 0
                
                # Expired entries
                cursor = conn.execute("""
                    SELECT COUNT(*) FROM cache_entries 
                    WHERE expires_at < datetime('now')
                """)
                expired_count = cursor.fetchone()[0]
                
                return {
                    'total_cached_analyses': total_entries,
                    'storage_used_mb': total_size_bytes / (1024 * 1024),
                    'analysis_type_breakdown': type_breakdown,
                    'recent_access_count': recent_access_count,
                    'avg_access_count': round(avg_access_count, 2),
                    'expired_entries': expired_count,
                    'cache_hit_rate': self._calculate_hit_rate(),
                    'avg_cache_response_time': 0.05,  # Estimated
                    'avg_fresh_response_time': 30.0,  # Estimated
                    'last_cleanup': self._get_last_cleanup_time()
                }
                
        except Exception as e:
            print(f"Error getting cache stats: {e}")
            return {'error': str(e)}
    
    def cleanup_old_entries(self, max_age_days: int = 7) -> Dict[str, Any]:
        """
        Clean up old and expired cache entries
        
        Args:
            max_age_days: Maximum age in days for cache entries
            
        Returns:
            Cleanup statistics
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=max_age_days)
            
            with sqlite3.connect(self.db_path) as conn:
                # Find entries to remove
                cursor = conn.execute("""
                    SELECT request_hash, result_path, file_size_bytes
                    FROM cache_entries 
                    WHERE expires_at < ? OR cached_at < ?
                """, (datetime.now().isoformat(), cutoff_date.isoformat()))
                
                entries_to_remove = cursor.fetchall()
                
                removed_count = 0
                freed_space = 0
                
                for request_hash, result_path, file_size in entries_to_remove:
                    # Remove file
                    try:
                        Path(result_path).unlink(missing_ok=True)
                        freed_space += file_size or 0
                        removed_count += 1
                    except Exception as e:
                        print(f"Error removing file {result_path}: {e}")
                
                # Remove database entries
                conn.execute("""
                    DELETE FROM cache_entries 
                    WHERE expires_at < ? OR cached_at < ?
                """, (datetime.now().isoformat(), cutoff_date.isoformat()))
                
                conn.commit()
                
                # Get remaining count
                cursor = conn.execute("SELECT COUNT(*) FROM cache_entries")
                remaining_count = cursor.fetchone()[0]
                
                return {
                    'removed_count': removed_count,
                    'freed_space_mb': freed_space / (1024 * 1024),
                    'remaining_count': remaining_count,
                    'cleanup_timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            print(f"Error during cleanup: {e}")
            return {'error': str(e)}
    
    def _remove_cache_entry(self, request_hash: str):
        """Remove a specific cache entry"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Get file path before deletion
                cursor = conn.execute("""
                    SELECT result_path FROM cache_entries WHERE request_hash = ?
                """, (request_hash,))
                row = cursor.fetchone()
                
                if row:
                    result_path = row[0]
                    # Remove file
                    Path(result_path).unlink(missing_ok=True)
                
                # Remove database entry
                conn.execute("""
                    DELETE FROM cache_entries WHERE request_hash = ?
                """, (request_hash,))
                conn.commit()
                
        except Exception as e:
            print(f"Error removing cache entry: {e}")
    
    def _generate_tags(self, analysis_type: str, parameters: Dict[str, Any]) -> str:
        """Generate searchable tags for the cache entry"""
        tags = [analysis_type]
        
        # Add parameter-based tags
        if 'context_strategy' in parameters:
            tags.append(f"strategy:{parameters['context_strategy']}")
        
        if 'target_audience' in parameters:
            tags.append(f"audience:{parameters['target_audience']}")
        
        if 'budget' in parameters:
            tags.append(f"budget:{parameters['budget']}")
        
        return ','.join(tags)
    
    def _store_in_mem0(self, request_hash: str, analysis_type: str,
                      result: Any, parameters: Dict[str, Any]):
        """Store analysis metadata in Mem0 for semantic search using new interface"""
        if not self.memory_manager:
            return
        
        try:
            # Create a searchable description of the analysis
            description = f"Analysis: {analysis_type}"
            if 'target_audience' in parameters:
                description += f" for {parameters['target_audience']}"
            if 'budget' in parameters:
                description += f" with budget {parameters['budget']}"
            
            # Add result summary if available
            if isinstance(result, dict):
                if 'summary' in result:
                    description += f". Summary: {result['summary']}"
                elif 'analysis' in result:
                    # Truncate long analysis text
                    analysis_text = str(result['analysis'])[:500]
                    description += f". Analysis: {analysis_text}"
            
            # Store in Mem0 using the new interface
            success = self.memory_manager.add_analysis_insight(
                analysis_type=analysis_type,
                insight=description,
                data_characteristics={
                    'request_hash': request_hash,
                    'parameters': parameters,
                    'timestamp': datetime.now().isoformat()
                }
            )
            
            if not success:
                print(f"Failed to store analysis in Mem0: {request_hash}")
                
        except Exception as e:
            print(f"Error storing in Mem0: {e}")
    
    def _search_mem0_similar(self, analysis_type: str, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search for similar analyses in Mem0 using new interface"""
        if not self.memory_manager:
            return []
        
        try:
            # Search using the new interface
            insights = self.memory_manager.get_relevant_insights(
                analysis_type=analysis_type,
                data_characteristics=parameters,
                limit=10
            )
            
            # Convert to expected format
            results = []
            for insight in insights:
                metadata = insight.get('metadata', {})
                results.append({
                    'score': insight.get('score', 0.8),  # Default score
                    'content': insight.get('memory', ''),
                    'metadata': metadata
                })
            
            return results
            
        except Exception as e:
            print(f"Error searching Mem0: {e}")
            return []
    
    def _calculate_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT SUM(access_count) FROM cache_entries")
                total_accesses = cursor.fetchone()[0] or 0
                
                cursor = conn.execute("SELECT COUNT(*) FROM cache_entries")
                total_entries = cursor.fetchone()[0] or 0
                
                if total_entries == 0:
                    return 0.0
                
                # Estimate hit rate based on access patterns
                return min(total_accesses / (total_entries * 2), 1.0)
                
        except Exception:
            return 0.0
    
    def _get_last_cleanup_time(self) -> str:
        """Get the last cleanup time"""
        try:
            cleanup_marker = self.cache_dir / ".last_cleanup"
            if cleanup_marker.exists():
                return datetime.fromtimestamp(cleanup_marker.stat().st_mtime).isoformat()
            return "Never"
        except Exception:
            return "Unknown"

# Global cache instance
_global_analysis_cache: Optional[AnalysisCacheManager] = None

def get_analysis_cache() -> AnalysisCacheManager:
    """Get the global analysis cache instance"""
    global _global_analysis_cache
    if _global_analysis_cache is None:
        _global_analysis_cache = AnalysisCacheManager()
    return _global_analysis_cache