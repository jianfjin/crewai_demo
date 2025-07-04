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

from ..memory.mem0_integration import MarketingMemoryManager

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
        
        # Initialize Mem0 for semantic caching
        self.memory_manager = MarketingMemoryManager(use_mock=not use_mem0) if use_mem0 else None
        
        # Cache configuration
        self.default_ttl = 7 * 24 * 3600  # 7 days
        self.max_cache_size_gb = 5  # 5GB limit
        
    def _init_database(self):
        """Initialize SQLite database for cache metadata"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS analysis_cache (
                    request_hash TEXT PRIMARY KEY,
                    analysis_type TEXT NOT NULL,
                    data_hash TEXT NOT NULL,
                    parameters_hash TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    last_accessed TIMESTAMP NOT NULL,
                    access_count INTEGER DEFAULT 1,
                    file_path TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    ttl INTEGER NOT NULL,
                    tags TEXT,
                    metadata TEXT
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_analysis_type ON analysis_cache(analysis_type)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_data_hash ON analysis_cache(data_hash)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at ON analysis_cache(created_at)
            """)
    
    def generate_request_hash(self, analysis_type: str, data_path: str, 
                            parameters: Dict[str, Any]) -> str:
        """
        Generate unique hash for analysis request
        
        Args:
            analysis_type: Type of analysis
            data_path: Path to source data
            parameters: Analysis parameters
            
        Returns:
            Unique request hash
        """
        # Get data file hash
        data_hash = self._get_file_hash(data_path)
        
        # Create parameters hash (excluding volatile parameters)
        stable_params = self._extract_stable_parameters(parameters)
        params_str = json.dumps(stable_params, sort_keys=True)
        params_hash = hashlib.md5(params_str.encode()).hexdigest()
        
        # Combine into request hash
        request_content = f"{analysis_type}:{data_hash}:{params_hash}"
        return hashlib.sha256(request_content.encode()).hexdigest()
    
    def cache_analysis_result(self, request_hash: str, analysis_type: str,
                            data_path: str, parameters: Dict[str, Any],
                            result: Any, intermediate_results: Dict[str, Any] = None,
                            ttl: Optional[int] = None) -> str:
        """
        Cache complete analysis result with intermediate steps
        
        Args:
            request_hash: Unique request identifier
            analysis_type: Type of analysis
            data_path: Source data path
            parameters: Analysis parameters
            result: Final analysis result
            intermediate_results: Intermediate analysis steps
            ttl: Time to live in seconds
            
        Returns:
            Cache file path
        """
        ttl = ttl or self.default_ttl
        
        # Prepare cache data
        cache_data = {
            'request_hash': request_hash,
            'analysis_type': analysis_type,
            'data_path': data_path,
            'parameters': parameters,
            'final_result': result,
            'intermediate_results': intermediate_results or {},
            'cached_at': datetime.now().isoformat(),
            'cache_version': '1.0'
        }
        
        # Save to file
        cache_file = self.cache_dir / f"{request_hash}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        # Store metadata in database
        data_hash = self._get_file_hash(data_path)
        params_hash = self._get_parameters_hash(parameters)
        file_size = cache_file.stat().st_size
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO analysis_cache 
                (request_hash, analysis_type, data_hash, parameters_hash, 
                 created_at, last_accessed, file_path, file_size, ttl, 
                 tags, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                request_hash, analysis_type, data_hash, params_hash,
                datetime.now(), datetime.now(), str(cache_file), file_size, ttl,
                self._generate_tags(analysis_type, parameters),
                json.dumps(self._extract_metadata(result))
            ))
        
        # Store key insights in Mem0 for semantic search
        if self.memory_manager:
            self._store_in_mem0(request_hash, analysis_type, result, parameters)
        
        print(f"💾 Cached analysis result: {request_hash}")
        return str(cache_file)
    
    def retrieve_cached_result(self, request_hash: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached analysis result
        
        Args:
            request_hash: Unique request identifier
            
        Returns:
            Cached analysis data or None if not found/expired
        """
        # Check database for metadata
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT file_path, created_at, ttl, access_count
                FROM analysis_cache 
                WHERE request_hash = ?
            """, (request_hash,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            file_path, created_at, ttl, access_count = row
            
            # Check if expired
            created_time = datetime.fromisoformat(created_at)
            if datetime.now() - created_time > timedelta(seconds=ttl):
                self._remove_cache_entry(request_hash)
                return None
            
            # Update access metadata
            conn.execute("""
                UPDATE analysis_cache 
                SET last_accessed = ?, access_count = access_count + 1
                WHERE request_hash = ?
            """, (datetime.now(), request_hash))
        
        # Load cached data
        try:
            with open(file_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            print(f"📦 Retrieved cached result: {request_hash} (accessed {access_count + 1} times)")
            return cache_data
            
        except Exception as e:
            print(f"⚠️  Error loading cached result: {e}")
            self._remove_cache_entry(request_hash)
            return None
    
    def find_similar_analyses(self, analysis_type: str, data_path: str,
                            parameters: Dict[str, Any], 
                            similarity_threshold: float = 0.8) -> List[Dict[str, Any]]:
        """
        Find similar cached analyses using semantic search
        
        Args:
            analysis_type: Type of analysis
            data_path: Source data path
            parameters: Analysis parameters
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of similar cached analyses
        """
        similar_analyses = []
        
        # Get data hash for exact data matches
        data_hash = self._get_file_hash(data_path)
        
        # Search database for same data and analysis type
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT request_hash, parameters_hash, metadata, created_at, access_count
                FROM analysis_cache 
                WHERE analysis_type = ? AND data_hash = ?
                ORDER BY access_count DESC, created_at DESC
                LIMIT 10
            """, (analysis_type, data_hash))
            
            for row in cursor.fetchall():
                request_hash, params_hash, metadata, created_at, access_count = row
                
                # Calculate parameter similarity
                similarity = self._calculate_parameter_similarity(
                    parameters, self._get_parameters_hash(parameters), params_hash
                )
                
                if similarity >= similarity_threshold:
                    similar_analyses.append({
                        'request_hash': request_hash,
                        'similarity_score': similarity,
                        'created_at': created_at,
                        'access_count': access_count,
                        'metadata': json.loads(metadata) if metadata else {}
                    })
        
        # Use Mem0 for semantic similarity search
        if self.memory_manager:
            semantic_matches = self._search_mem0_similar(analysis_type, parameters)
            similar_analyses.extend(semantic_matches)
        
        # Sort by similarity and recency
        similar_analyses.sort(key=lambda x: (x['similarity_score'], x['access_count']), reverse=True)
        
        return similar_analyses[:5]  # Return top 5 matches
    
    def cache_intermediate_result(self, request_hash: str, step_name: str, 
                                result: Any, step_metadata: Dict[str, Any] = None) -> str:
        """
        Cache intermediate analysis step result
        
        Args:
            request_hash: Parent analysis request hash
            step_name: Name of the analysis step
            result: Intermediate result
            step_metadata: Additional metadata for the step
            
        Returns:
            Intermediate cache reference
        """
        # Create intermediate cache hash
        intermediate_hash = f"{request_hash}_{step_name}"
        
        # Prepare intermediate data
        intermediate_data = {
            'parent_hash': request_hash,
            'step_name': step_name,
            'result': result,
            'metadata': step_metadata or {},
            'cached_at': datetime.now().isoformat()
        }
        
        # Save intermediate result
        cache_file = self.cache_dir / f"intermediate_{intermediate_hash}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(intermediate_data, f)
        
        print(f"💾 Cached intermediate result: {step_name} for {request_hash}")
        return intermediate_hash
    
    def retrieve_intermediate_result(self, request_hash: str, step_name: str) -> Optional[Any]:
        """
        Retrieve cached intermediate result
        
        Args:
            request_hash: Parent analysis request hash
            step_name: Name of the analysis step
            
        Returns:
            Intermediate result or None if not found
        """
        intermediate_hash = f"{request_hash}_{step_name}"
        cache_file = self.cache_dir / f"intermediate_{intermediate_hash}.pkl"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                intermediate_data = pickle.load(f)
            
            print(f"📦 Retrieved intermediate result: {step_name}")
            return intermediate_data['result']
            
        except Exception as e:
            print(f"⚠️  Error loading intermediate result: {e}")
            return None
    
    def cleanup_expired_cache(self) -> Dict[str, int]:
        """
        Clean up expired cache entries
        
        Returns:
            Cleanup statistics
        """
        cleanup_stats = {
            'expired_entries': 0,
            'freed_space_mb': 0,
            'total_entries_before': 0,
            'total_entries_after': 0
        }
        
        current_time = datetime.now()
        
        with sqlite3.connect(self.db_path) as conn:
            # Count total entries before cleanup
            cursor = conn.execute("SELECT COUNT(*) FROM analysis_cache")
            cleanup_stats['total_entries_before'] = cursor.fetchone()[0]
            
            # Find expired entries
            cursor = conn.execute("""
                SELECT request_hash, file_path, file_size
                FROM analysis_cache 
                WHERE datetime(created_at, '+' || ttl || ' seconds') < ?
            """, (current_time,))
            
            expired_entries = cursor.fetchall()
            
            for request_hash, file_path, file_size in expired_entries:
                # Remove file
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        cleanup_stats['freed_space_mb'] += file_size / (1024 * 1024)
                except Exception as e:
                    print(f"⚠️  Error removing cache file {file_path}: {e}")
                
                # Remove intermediate files
                intermediate_pattern = f"intermediate_{request_hash}_*.pkl"
                for intermediate_file in self.cache_dir.glob(intermediate_pattern):
                    try:
                        intermediate_file.unlink()
                    except Exception:
                        pass
                
                cleanup_stats['expired_entries'] += 1
            
            # Remove expired entries from database
            conn.execute("""
                DELETE FROM analysis_cache 
                WHERE datetime(created_at, '+' || ttl || ' seconds') < ?
            """, (current_time,))
            
            # Count total entries after cleanup
            cursor = conn.execute("SELECT COUNT(*) FROM analysis_cache")
            cleanup_stats['total_entries_after'] = cursor.fetchone()[0]
        
        print(f"🧹 Cache cleanup: {cleanup_stats['expired_entries']} expired entries removed, "
              f"{cleanup_stats['freed_space_mb']:.2f} MB freed")
        
        return cleanup_stats
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        stats = {
            'total_entries': 0,
            'total_size_mb': 0,
            'analysis_types': {},
            'access_patterns': {},
            'cache_hit_rate': 0,
            'oldest_entry': None,
            'newest_entry': None,
            'most_accessed': None
        }
        
        with sqlite3.connect(self.db_path) as conn:
            # Total entries and size
            cursor = conn.execute("""
                SELECT COUNT(*), SUM(file_size), MIN(created_at), MAX(created_at)
                FROM analysis_cache
            """)
            row = cursor.fetchone()
            if row[0]:
                stats['total_entries'] = row[0]
                stats['total_size_mb'] = (row[1] or 0) / (1024 * 1024)
                stats['oldest_entry'] = row[2]
                stats['newest_entry'] = row[3]
            
            # Analysis types breakdown
            cursor = conn.execute("""
                SELECT analysis_type, COUNT(*), SUM(file_size)
                FROM analysis_cache
                GROUP BY analysis_type
            """)
            for analysis_type, count, size in cursor.fetchall():
                stats['analysis_types'][analysis_type] = {
                    'count': count,
                    'size_mb': (size or 0) / (1024 * 1024)
                }
            
            # Access patterns
            cursor = conn.execute("""
                SELECT AVG(access_count), MAX(access_count)
                FROM analysis_cache
            """)
            row = cursor.fetchone()
            if row[0]:
                stats['access_patterns'] = {
                    'average_access_count': row[0],
                    'max_access_count': row[1]
                }
            
            # Most accessed entry
            cursor = conn.execute("""
                SELECT request_hash, analysis_type, access_count
                FROM analysis_cache
                ORDER BY access_count DESC
                LIMIT 1
            """)
            row = cursor.fetchone()
            if row:
                stats['most_accessed'] = {
                    'request_hash': row[0],
                    'analysis_type': row[1],
                    'access_count': row[2]
                }
        
        return stats
    
    def _get_file_hash(self, file_path: str) -> str:
        """Get hash of file content"""
        try:
            with open(file_path, 'rb') as f:
                file_content = f.read()
            return hashlib.md5(file_content).hexdigest()
        except Exception:
            # Fallback to file path and modification time
            stat = os.stat(file_path)
            content = f"{file_path}:{stat.st_mtime}:{stat.st_size}"
            return hashlib.md5(content.encode()).hexdigest()
    
    def _get_parameters_hash(self, parameters: Dict[str, Any]) -> str:
        """Get hash of parameters"""
        stable_params = self._extract_stable_parameters(parameters)
        params_str = json.dumps(stable_params, sort_keys=True)
        return hashlib.md5(params_str.encode()).hexdigest()
    
    def _extract_stable_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Extract stable parameters (excluding volatile ones like timestamps)"""
        volatile_keys = {'timestamp', 'session_id', 'execution_id', 'temp_', 'cache_'}
        
        stable_params = {}
        for key, value in parameters.items():
            if not any(volatile in key.lower() for volatile in volatile_keys):
                if isinstance(value, dict):
                    stable_params[key] = self._extract_stable_parameters(value)
                else:
                    stable_params[key] = value
        
        return stable_params
    
    def _generate_tags(self, analysis_type: str, parameters: Dict[str, Any]) -> str:
        """Generate searchable tags for the analysis"""
        tags = [analysis_type]
        
        # Add parameter-based tags
        for key, value in parameters.items():
            if isinstance(value, str) and len(value) < 50:
                tags.append(f"{key}:{value}")
            elif isinstance(value, (int, float)):
                tags.append(f"{key}:{value}")
        
        return ','.join(tags)
    
    def _extract_metadata(self, result: Any) -> Dict[str, Any]:
        """Extract searchable metadata from analysis result"""
        metadata = {}
        
        if isinstance(result, dict):
            # Extract key metrics and insights
            for key, value in result.items():
                if 'insight' in key.lower() or 'summary' in key.lower():
                    metadata[key] = str(value)[:200]  # Truncate long values
                elif isinstance(value, (int, float)):
                    metadata[key] = value
        
        return metadata
    
    def _store_in_mem0(self, request_hash: str, analysis_type: str, 
                      result: Any, parameters: Dict[str, Any]):
        """Store analysis insights in Mem0 for semantic search"""
        if not self.memory_manager:
            return
        
        # Extract key insights for semantic storage
        insights = self._extract_semantic_insights(result)
        
        # Store with enhanced metadata
        self.memory_manager.store_analysis_insights(
            analysis_type=f"cached_{analysis_type}",
            insights=insights,
            metadata={
                'request_hash': request_hash,
                'parameters': parameters,
                'cached_at': time.time(),
                'cache_type': 'persistent'
            }
        )
    
    def _extract_semantic_insights(self, result: Any) -> Dict[str, Any]:
        """Extract semantic insights for Mem0 storage"""
        insights = {}
        
        if isinstance(result, dict):
            # Look for key insights, recommendations, and metrics
            for key, value in result.items():
                if any(keyword in key.lower() for keyword in 
                      ['insight', 'recommendation', 'finding', 'conclusion', 'summary']):
                    insights[key] = value
                elif isinstance(value, (int, float)) and 'metric' in key.lower():
                    insights[key] = value
        
        return insights
    
    def _search_mem0_similar(self, analysis_type: str, parameters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search for similar analyses in Mem0"""
        if not self.memory_manager:
            return []
        
        # Create search query from parameters
        query_parts = [analysis_type]
        for key, value in parameters.items():
            if isinstance(value, str) and len(value) < 100:
                query_parts.append(f"{key} {value}")
        
        query = " ".join(query_parts)
        
        # Search Mem0
        context = self.memory_manager.get_relevant_context(
            query=query,
            analysis_type=f"cached_{analysis_type}",
            max_tokens=500
        )
        
        # Convert Mem0 results to similar analyses format
        similar = []
        for insight in context.get('insights', []):
            metadata = insight.get('metadata', {})
            if 'request_hash' in metadata:
                similar.append({
                    'request_hash': metadata['request_hash'],
                    'similarity_score': insight.get('score', 0.5),
                    'created_at': metadata.get('cached_at', time.time()),
                    'access_count': 1,
                    'metadata': metadata,
                    'source': 'mem0'
                })
        
        return similar
    
    def _calculate_parameter_similarity(self, params1: Dict[str, Any], 
                                      hash1: str, hash2: str) -> float:
        """Calculate similarity between parameter sets"""
        if hash1 == hash2:
            return 1.0
        
        # Simple similarity based on hash difference
        # In a more sophisticated implementation, you could compare actual parameter values
        return 0.5  # Placeholder similarity score
    
    def _remove_cache_entry(self, request_hash: str):
        """Remove cache entry and associated files"""
        with sqlite3.connect(self.db_path) as conn:
            # Get file path before deletion
            cursor = conn.execute(
                "SELECT file_path FROM analysis_cache WHERE request_hash = ?",
                (request_hash,)
            )
            row = cursor.fetchone()
            
            if row:
                file_path = row[0]
                # Remove main cache file
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                except Exception:
                    pass
                
                # Remove intermediate files
                intermediate_pattern = f"intermediate_{request_hash}_*.pkl"
                for intermediate_file in self.cache_dir.glob(intermediate_pattern):
                    try:
                        intermediate_file.unlink()
                    except Exception:
                        pass
            
            # Remove from database
            conn.execute("DELETE FROM analysis_cache WHERE request_hash = ?", (request_hash,))

# Global cache manager instance
_global_analysis_cache = None

def get_analysis_cache() -> AnalysisCacheManager:
    """Get global analysis cache instance"""
    global _global_analysis_cache
    if _global_analysis_cache is None:
        _global_analysis_cache = AnalysisCacheManager()
    return _global_analysis_cache