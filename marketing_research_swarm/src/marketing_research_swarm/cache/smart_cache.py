"""
Smart caching system for data and analysis results
"""

import hashlib
import pickle
import time
import os
from typing import Any, Dict, Optional, Union
import pandas as pd
from datetime import datetime, timedelta

class SmartCache:
    """
    Intelligent caching system with automatic cleanup and reference management
    """
    
    def __init__(self, cache_dir: str = "cache", max_size_mb: int = 500, default_ttl: int = 3600):
        self.cache_dir = cache_dir
        self.max_size = max_size_mb * 1024 * 1024  # Convert to bytes
        self.default_ttl = default_ttl
        self.memory_cache = {}  # In-memory cache for small items
        self.metadata = {}  # Cache metadata
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load existing metadata
        self._load_metadata()
    
    def store(self, key: str, data: Any, ttl: Optional[int] = None, 
              force_disk: bool = False) -> str:
        """
        Store data in cache and return reference key
        
        Args:
            key: Cache key
            data: Data to cache
            ttl: Time to live in seconds
            force_disk: Force storage to disk even for small items
            
        Returns:
            Cache reference string
        """
        ttl = ttl or self.default_ttl
        data_hash = self._create_hash(data)
        reference = f"cache://{data_hash[:12]}"
        
        # Estimate data size
        data_size = self._estimate_size(data)
        
        # Decide storage location
        if data_size < 1024 * 100 and not force_disk:  # < 100KB, store in memory
            self.memory_cache[reference] = {
                'data': data,
                'created': time.time(),
                'ttl': ttl,
                'size': data_size,
                'access_count': 0,
                'last_access': time.time()
            }
        else:
            # Store on disk
            file_path = os.path.join(self.cache_dir, f"{data_hash}.pkl")
            try:
                with open(file_path, 'wb') as f:
                    pickle.dump(data, f)
                
                # Store metadata
                self.metadata[reference] = {
                    'file_path': file_path,
                    'created': time.time(),
                    'ttl': ttl,
                    'size': data_size,
                    'access_count': 0,
                    'last_access': time.time(),
                    'data_type': type(data).__name__
                }
                
                self._save_metadata()
                
            except Exception as e:
                print(f"Warning: Failed to cache to disk: {e}")
                # Fallback to memory cache
                self.memory_cache[reference] = {
                    'data': data,
                    'created': time.time(),
                    'ttl': ttl,
                    'size': data_size,
                    'access_count': 0,
                    'last_access': time.time()
                }
        
        # Cleanup if needed
        self._cleanup_if_needed()
        
        return reference
    
    def retrieve(self, reference: str) -> Optional[Any]:
        """
        Retrieve data by reference
        
        Args:
            reference: Cache reference string
            
        Returns:
            Cached data or None if not found/expired
        """
        if not reference.startswith('cache://'):
            return None
        
        current_time = time.time()
        
        # Check memory cache first
        if reference in self.memory_cache:
            entry = self.memory_cache[reference]
            
            # Check expiration
            if current_time - entry['created'] > entry['ttl']:
                del self.memory_cache[reference]
                return None
            
            # Update access metadata
            entry['access_count'] += 1
            entry['last_access'] = current_time
            
            return entry['data']
        
        # Check disk cache
        if reference in self.metadata:
            entry = self.metadata[reference]
            
            # Check expiration
            if current_time - entry['created'] > entry['ttl']:
                self._remove_from_disk(reference)
                return None
            
            # Load from disk
            try:
                with open(entry['file_path'], 'rb') as f:
                    data = pickle.load(f)
                
                # Update access metadata
                entry['access_count'] += 1
                entry['last_access'] = current_time
                self._save_metadata()
                
                return data
                
            except Exception as e:
                print(f"Warning: Failed to load from cache: {e}")
                self._remove_from_disk(reference)
                return None
        
        return None
    
    def create_data_reference(self, data: Any, cache_type: str = "data") -> str:
        """
        Create a cache reference for data with automatic storage decision
        
        Args:
            data: Data to cache
            cache_type: Type of data being cached
            
        Returns:
            Cache reference string
        """
        # Create descriptive key
        timestamp = int(time.time())
        data_hash = self._create_hash(data)[:8]
        key = f"{cache_type}_{timestamp}_{data_hash}"
        
        return self.store(key, data)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        memory_size = sum(entry['size'] for entry in self.memory_cache.values())
        disk_size = sum(entry['size'] for entry in self.metadata.values())
        
        return {
            'memory_items': len(self.memory_cache),
            'disk_items': len(self.metadata),
            'total_items': len(self.memory_cache) + len(self.metadata),
            'memory_size_mb': memory_size / (1024 * 1024),
            'disk_size_mb': disk_size / (1024 * 1024),
            'total_size_mb': (memory_size + disk_size) / (1024 * 1024),
            'max_size_mb': self.max_size / (1024 * 1024)
        }
    
    def cleanup_expired(self) -> int:
        """Clean up expired cache entries"""
        current_time = time.time()
        cleaned_count = 0
        
        # Clean memory cache
        expired_memory = [
            ref for ref, entry in self.memory_cache.items()
            if current_time - entry['created'] > entry['ttl']
        ]
        
        for ref in expired_memory:
            del self.memory_cache[ref]
            cleaned_count += 1
        
        # Clean disk cache
        expired_disk = [
            ref for ref, entry in self.metadata.items()
            if current_time - entry['created'] > entry['ttl']
        ]
        
        for ref in expired_disk:
            self._remove_from_disk(ref)
            cleaned_count += 1
        
        return cleaned_count
    
    def _create_hash(self, data: Any) -> str:
        """Create hash for data"""
        if isinstance(data, pd.DataFrame):
            # Hash DataFrame content
            content = data.to_string()
        elif isinstance(data, dict):
            # Hash dictionary content
            content = str(sorted(data.items()))
        else:
            content = str(data)
        
        return hashlib.md5(content.encode()).hexdigest()
    
    def _estimate_size(self, data: Any) -> int:
        """Estimate data size in bytes"""
        try:
            return len(pickle.dumps(data))
        except:
            # Fallback estimation
            if isinstance(data, pd.DataFrame):
                return data.memory_usage(deep=True).sum()
            elif isinstance(data, str):
                return len(data.encode())
            else:
                return len(str(data).encode())
    
    def _cleanup_if_needed(self):
        """Clean up cache if size limit exceeded"""
        total_size = (
            sum(entry['size'] for entry in self.memory_cache.values()) +
            sum(entry['size'] for entry in self.metadata.values())
        )
        
        if total_size > self.max_size:
            # Clean up least recently used items
            all_items = []
            
            # Add memory items
            for ref, entry in self.memory_cache.items():
                all_items.append((ref, entry['last_access'], entry['size'], 'memory'))
            
            # Add disk items
            for ref, entry in self.metadata.items():
                all_items.append((ref, entry['last_access'], entry['size'], 'disk'))
            
            # Sort by last access time (oldest first)
            all_items.sort(key=lambda x: x[1])
            
            # Remove items until under 80% capacity
            target_size = self.max_size * 0.8
            current_size = total_size
            
            for ref, last_access, size, location in all_items:
                if current_size <= target_size:
                    break
                
                if location == 'memory':
                    if ref in self.memory_cache:
                        del self.memory_cache[ref]
                        current_size -= size
                else:  # disk
                    self._remove_from_disk(ref)
                    current_size -= size
    
    def _remove_from_disk(self, reference: str):
        """Remove item from disk cache"""
        if reference in self.metadata:
            entry = self.metadata[reference]
            
            # Remove file
            try:
                if os.path.exists(entry['file_path']):
                    os.remove(entry['file_path'])
            except Exception as e:
                print(f"Warning: Failed to remove cache file: {e}")
            
            # Remove metadata
            del self.metadata[reference]
            self._save_metadata()
    
    def _load_metadata(self):
        """Load cache metadata from disk"""
        metadata_path = os.path.join(self.cache_dir, 'metadata.pkl')
        
        try:
            if os.path.exists(metadata_path):
                with open(metadata_path, 'rb') as f:
                    self.metadata = pickle.load(f)
        except Exception as e:
            print(f"Warning: Failed to load cache metadata: {e}")
            self.metadata = {}
    
    def _save_metadata(self):
        """Save cache metadata to disk"""
        metadata_path = os.path.join(self.cache_dir, 'metadata.pkl')
        
        try:
            with open(metadata_path, 'wb') as f:
                pickle.dump(self.metadata, f)
        except Exception as e:
            print(f"Warning: Failed to save cache metadata: {e}")

# Global cache instance
_global_cache = None

def get_cache() -> SmartCache:
    """Get global cache instance"""
    global _global_cache
    if _global_cache is None:
        _global_cache = SmartCache()
    return _global_cache

def cached_tool_call(func):
    """Decorator for automatic tool output caching"""
    def wrapper(*args, **kwargs):
        cache = get_cache()
        
        # Create cache key from function name and arguments
        cache_key = f"{func.__name__}_{hash(str(args) + str(kwargs))}"
        
        # Try to get from cache
        cached_result = cache.retrieve(f"cache://{cache_key}")
        if cached_result is not None:
            print(f"📦 Using cached result for {func.__name__}")
            return cached_result
        
        # Execute function
        result = func(*args, **kwargs)
        
        # Cache result if it's large enough
        if cache._estimate_size(result) > 1000:  # 1KB threshold
            reference = cache.store(cache_key, result)
            print(f"💾 Cached result for {func.__name__}: {reference}")
        
        return result
    
    return wrapper