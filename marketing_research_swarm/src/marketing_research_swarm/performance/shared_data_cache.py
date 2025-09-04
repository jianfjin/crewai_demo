"""
Shared Data Cache System for Performance Optimization

This module provides a centralized data cache that eliminates redundant data loading
across tools and agents, significantly improving performance.
"""

import pandas as pd
import numpy as np
import hashlib
import time
import threading
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import os
import json

logger = logging.getLogger(__name__)

class SharedDataCache:
    """
    High-performance shared data cache with lifecycle management.
    Eliminates redundant data loading across tools and agents.
    """
    
    def __init__(self, max_cache_size: int = 100, cache_ttl_minutes: int = 60):
        """
        Initialize the shared data cache.
        
        Args:
            max_cache_size: Maximum number of datasets to cache
            cache_ttl_minutes: Time-to-live for cached data in minutes
        """
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, datetime] = {}
        self._load_times: Dict[str, float] = {}
        self._lock = threading.RLock()
        self.max_cache_size = max_cache_size
        self.cache_ttl = timedelta(minutes=cache_ttl_minutes)
        
        # Performance metrics
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_load_time_saved = 0.0
        
        logger.info(f"🚀 SharedDataCache initialized (max_size={max_cache_size}, ttl={cache_ttl_minutes}min)")
    
    def _generate_cache_key(self, data_path: str, **kwargs) -> str:
        """Generate a unique cache key for the data request."""
        key_data = {
            'data_path': data_path or 'sample_data',
            'kwargs': sorted(kwargs.items()) if kwargs else []
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self._access_times:
            return False
        
        age = datetime.now() - self._access_times[cache_key]
        return age < self.cache_ttl
    
    def _cleanup_expired_cache(self):
        """Remove expired cache entries."""
        with self._lock:
            current_time = datetime.now()
            expired_keys = [
                key for key, access_time in self._access_times.items()
                if current_time - access_time > self.cache_ttl
            ]
            
            for key in expired_keys:
                self._remove_cache_entry(key)
            
            if expired_keys:
                logger.info(f"🧹 Cleaned up {len(expired_keys)} expired cache entries")
    
    def _remove_cache_entry(self, cache_key: str):
        """Remove a specific cache entry."""
        if cache_key in self._cache:
            del self._cache[cache_key]
        if cache_key in self._access_times:
            del self._access_times[cache_key]
        if cache_key in self._load_times:
            del self._load_times[cache_key]
    
    def _enforce_cache_size_limit(self):
        """Enforce maximum cache size by removing least recently used entries."""
        with self._lock:
            if len(self._cache) <= self.max_cache_size:
                return
            
            # Sort by access time (least recently used first)
            sorted_keys = sorted(
                self._access_times.items(),
                key=lambda x: x[1]
            )
            
            # Remove oldest entries
            entries_to_remove = len(self._cache) - self.max_cache_size
            for i in range(entries_to_remove):
                cache_key = sorted_keys[i][0]
                self._remove_cache_entry(cache_key)
            
            logger.info(f"📦 Removed {entries_to_remove} LRU cache entries to enforce size limit")
    
    def get_or_load_data(self, data_path: str = None, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Get cached data or load it if not cached.
        
        Args:
            data_path: Path to the data file
            **kwargs: Additional parameters for data loading
            
        Returns:
            Tuple of (DataFrame, cache_info)
        """
        start_time = time.time()
        cache_key = self._generate_cache_key(data_path, **kwargs)
        
        with self._lock:
            # Check if data is cached and valid
            if cache_key in self._cache and self._is_cache_valid(cache_key):
                # Cache hit
                self._access_times[cache_key] = datetime.now()
                self.cache_hits += 1
                
                cached_data = self._cache[cache_key]
                df = cached_data['dataframe'].copy()  # Return copy to prevent modification
                
                # Calculate time saved
                original_load_time = self._load_times.get(cache_key, 0)
                time_saved = original_load_time
                self.total_load_time_saved += time_saved
                
                cache_info = {
                    'cache_hit': True,
                    'cache_key': cache_key,
                    'data_shape': df.shape,
                    'time_saved_seconds': time_saved,
                    'access_time': time.time() - start_time
                }
                
                logger.info(f"⚡ Cache HIT: {cache_key[:8]}... (saved {time_saved:.3f}s)")
                return df, cache_info
            
            # Cache miss - load data
            self.cache_misses += 1
            logger.info(f"💾 Cache MISS: Loading data for {cache_key[:8]}...")
            
            # Load the data
            load_start = time.time()
            df = self._load_data_from_source(data_path, **kwargs)
            load_time = time.time() - load_start
            
            # Cache the data
            self._cache[cache_key] = {
                'dataframe': df.copy(),  # Store copy to prevent external modification
                'metadata': {
                    'data_path': data_path,
                    'shape': df.shape,
                    'columns': list(df.columns),
                    'load_time': load_time,
                    'loaded_at': datetime.now().isoformat()
                }
            }
            
            self._access_times[cache_key] = datetime.now()
            self._load_times[cache_key] = load_time
            
            # Cleanup and enforce limits
            self._cleanup_expired_cache()
            self._enforce_cache_size_limit()
            
            cache_info = {
                'cache_hit': False,
                'cache_key': cache_key,
                'data_shape': df.shape,
                'load_time_seconds': load_time,
                'access_time': time.time() - start_time
            }
            
            logger.info(f"📊 Data loaded and cached: {df.shape} in {load_time:.3f}s")
            return df, cache_info
    
    def _load_data_from_source(self, data_path: str = None, **kwargs) -> pd.DataFrame:
        """Load data from the actual source."""
        # FIXED: Check if data_path exists before falling back to sample data
        if not data_path or not os.path.exists(data_path):
            if data_path:
                logger.warning(f"⚠️ Data file not found: {data_path}")
                logger.info("🔄 Falling back to sample data")
            else:
                logger.info("🔄 No data path provided, using sample data")
            return self._create_sample_beverage_data()
        
        try:
            # FIXED: Always try to load the actual file first
            logger.info(f"📁 Attempting to load data from: {data_path}")
            
            if data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
            elif data_path.endswith('.json'):
                df = pd.read_json(data_path)
            elif data_path.endswith('.parquet'):
                df = pd.read_parquet(data_path)
            else:
                # Try CSV first, then JSON
                try:
                    df = pd.read_csv(data_path)
                except:
                    df = pd.read_json(data_path)
            
            logger.info(f"✅ Successfully loaded data from {data_path}: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading data from {data_path}: {e}")
            logger.info("🔄 Falling back to sample data")
            return self._create_sample_beverage_data()
    
    def _create_sample_beverage_data(self) -> pd.DataFrame:
        """Create optimized sample beverage data."""
        np.random.seed(42)  # For reproducible results
        
        # Pre-generate data arrays for better performance
        n_days = 365
        n_brands = 6
        n_records = n_days * n_brands
        
        dates = pd.date_range('2023-01-01', '2024-12-31', freq='D')[:n_days]
        brands = ['Coca-Cola', 'Pepsi', 'Sprite', 'Fanta', 'Dr Pepper', 'Mountain Dew']
        categories = ['Cola', 'Lemon-Lime', 'Orange', 'Energy', 'Diet']
        regions = ['North America', 'Europe', 'Asia Pacific', 'Latin America', 'Africa']
        
        # Vectorized data generation for performance
        data_arrays = {
            'sale_date': np.repeat(dates, n_brands),
            'brand': np.tile(brands, n_days),
            'category': np.random.choice(categories, n_records),
            'region': np.random.choice(regions, n_records),
        }
        
        # Generate seasonal patterns vectorized
        day_of_year = np.array([d.dayofyear for d in data_arrays['sale_date']])
        seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * day_of_year / 365)
        base_sales = np.random.normal(1000, 200, n_records) * seasonal_factor
        
        # Calculate financial metrics vectorized
        units_sold = np.maximum(10, base_sales.astype(int))
        price_per_unit = np.random.uniform(1.5, 3.5, n_records)
        total_revenue = units_sold * price_per_unit
        cost_per_unit = price_per_unit * np.random.uniform(0.4, 0.7, n_records)
        total_cost = units_sold * cost_per_unit
        profit = total_revenue - total_cost
        profit_margin = np.where(total_revenue > 0, (profit / total_revenue * 100), 0)
        
        # Create DataFrame efficiently
        df = pd.DataFrame({
            'sale_date': data_arrays['sale_date'],
            'year': data_arrays['sale_date'].year,
            'month': data_arrays['sale_date'].month,
            'quarter': ['Q' + str((m-1)//3 + 1) for m in data_arrays['sale_date'].month],
            'region': data_arrays['region'],
            'country': ['Country_' + r.replace(' ', '_') for r in data_arrays['region']],
            'store_id': [f"STORE_{i % 100:03d}" for i in range(n_records)],
            'brand': data_arrays['brand'],
            'category': data_arrays['category'],
            'units_sold': units_sold,
            'price_per_unit': np.round(price_per_unit, 2),
            'total_revenue': np.round(total_revenue, 2),
            'cost_per_unit': np.round(cost_per_unit, 2),
            'total_cost': np.round(total_cost, 2),
            'profit': np.round(profit, 2),
            'profit_margin': np.round(profit_margin, 2)
        })
        
        logger.info(f"🎲 Generated sample beverage data: {df.shape}")
        return df
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache performance statistics."""
        with self._lock:
            total_requests = self.cache_hits + self.cache_misses
            hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
            
            # Calculate cache memory usage estimate
            memory_usage_mb = 0
            for cache_data in self._cache.values():
                df = cache_data['dataframe']
                memory_usage_mb += df.memory_usage(deep=True).sum() / (1024 * 1024)
            
            stats = {
                'cache_hits': self.cache_hits,
                'cache_misses': self.cache_misses,
                'hit_rate_percentage': round(hit_rate, 2),
                'total_time_saved_seconds': round(self.total_load_time_saved, 3),
                'cached_datasets': len(self._cache),
                'max_cache_size': self.max_cache_size,
                'estimated_memory_usage_mb': round(memory_usage_mb, 2),
                'cache_ttl_minutes': self.cache_ttl.total_seconds() / 60
            }
            
            return stats
    
    def clear_cache(self):
        """Clear all cached data."""
        with self._lock:
            cleared_count = len(self._cache)
            self._cache.clear()
            self._access_times.clear()
            self._load_times.clear()
            
            logger.info(f"🗑️ Cleared {cleared_count} cache entries")
    
    def preload_data(self, data_paths: list, **kwargs):
        """Preload data into cache for better performance."""
        logger.info(f"🔄 Preloading {len(data_paths)} datasets...")
        
        for data_path in data_paths:
            try:
                df, cache_info = self.get_or_load_data(data_path, **kwargs)
                logger.info(f"✅ Preloaded: {data_path} ({df.shape})")
            except Exception as e:
                logger.error(f"❌ Failed to preload {data_path}: {e}")

# Global shared cache instance
_global_shared_cache = None

def get_shared_cache() -> SharedDataCache:
    """Get the global shared data cache instance."""
    global _global_shared_cache
    if _global_shared_cache is None:
        _global_shared_cache = SharedDataCache()
    return _global_shared_cache

def clear_global_cache():
    """Clear the global cache."""
    global _global_shared_cache
    if _global_shared_cache:
        _global_shared_cache.clear_cache()