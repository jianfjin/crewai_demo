"""
Base Flow with State Management and Context Engineering
"""

from crewai.flow import Flow
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
import pandas as pd
import hashlib
import pickle
import time
import json
from datetime import datetime, timedelta

class FlowState(BaseModel):
    """Shared state across flow steps with intelligent caching"""
    
    # Data caching
    data_cache: Dict[str, Any] = Field(default_factory=dict)
    analysis_results: Dict[str, Any] = Field(default_factory=dict)
    tool_outputs: Dict[str, Any] = Field(default_factory=dict)
    
    # Context management
    context_budget: int = Field(default=4000)  # Max tokens per step
    current_step: str = Field(default="")
    step_history: List[str] = Field(default_factory=list)
    
    # Cache metadata
    cache_metadata: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    
    # Configuration
    data_file_path: str = Field(default="data/beverage_sales.csv")
    analysis_type: str = Field(default="")
    
    class Config:
        arbitrary_types_allowed = True

class ContextManager:
    """Advanced context management with multiple strategies"""
    
    def __init__(self, token_budget: int = 4000):
        self.token_budget = token_budget
        self.strategies = {
            'critical': 1.0,    # Always keep
            'important': 0.7,   # Keep if space
            'useful': 0.4,      # Summarize
            'optional': 0.1     # Remove first
        }
    
    def optimize_context(self, context: Dict[str, Any], strategy: str = 'progressive') -> Dict[str, Any]:
        """Apply context optimization strategy"""
        current_tokens = self._estimate_tokens(context)
        
        if current_tokens <= self.token_budget:
            return context
        
        if strategy == 'progressive':
            return self._progressive_pruning(context)
        elif strategy == 'summary':
            return self._abstracted_summaries(context)
        elif strategy == 'minimal':
            return self._minimal_context(context)
        else:
            return self._stateless_context(context)
    
    def _progressive_pruning(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Progressive pruning based on priority"""
        optimized = {}
        remaining_budget = self.token_budget
        
        # Sort by priority
        prioritized_items = sorted(
            context.items(),
            key=lambda x: self._get_priority_score(x[0], x[1]),
            reverse=True
        )
        
        for key, value in prioritized_items:
            item_tokens = self._estimate_tokens(value)
            
            if item_tokens <= remaining_budget:
                optimized[key] = value
                remaining_budget -= item_tokens
            elif remaining_budget > 100:  # Minimum space for summary
                optimized[key] = self._create_summary(value)
                remaining_budget -= 100
            
        return optimized
    
    def _abstracted_summaries(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create abstracted summaries of large content"""
        summaries = {}
        
        for key, value in context.items():
            if self._estimate_tokens(value) > 200:
                summaries[key] = self._create_summary(value)
            else:
                summaries[key] = value
                
        return summaries
    
    def _minimal_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Keep only essential context"""
        essential_keys = ['key_insights', 'summary', 'recommendations', 'metrics']
        return {k: v for k, v in context.items() if any(ek in k.lower() for ek in essential_keys)}
    
    def _stateless_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Remove all historical context, keep only current step"""
        current_keys = ['current_analysis', 'immediate_results', 'step_output']
        return {k: v for k, v in context.items() if any(ck in k.lower() for ck in current_keys)}
    
    def _get_priority_score(self, key: str, value: Any) -> float:
        """Calculate priority score for context item"""
        key_lower = key.lower()
        
        if any(critical in key_lower for critical in ['error', 'critical', 'key_insights']):
            return 1.0
        elif any(important in key_lower for important in ['summary', 'metrics', 'results']):
            return 0.7
        elif any(useful in key_lower for useful in ['analysis', 'data', 'trends']):
            return 0.4
        else:
            return 0.1
    
    def _create_summary(self, content: Any) -> Dict[str, Any]:
        """Create intelligent summary of content"""
        if isinstance(content, pd.DataFrame):
            return {
                'type': 'dataframe_summary',
                'shape': content.shape,
                'columns': content.columns.tolist()[:10],  # Limit columns
                'key_stats': {
                    'total_revenue': content.get('total_revenue', pd.Series()).sum() if 'total_revenue' in content.columns else 0,
                    'record_count': len(content)
                }
            }
        elif isinstance(content, dict):
            return {
                'type': 'dict_summary',
                'keys': list(content.keys())[:5],  # Limit keys
                'sample_values': {k: str(v)[:50] for k, v in list(content.items())[:3]}
            }
        elif isinstance(content, list):
            return {
                'type': 'list_summary',
                'length': len(content),
                'sample_items': content[:3] if len(content) > 3 else content
            }
        else:
            content_str = str(content)
            return {
                'type': 'text_summary',
                'length': len(content_str),
                'preview': content_str[:100] + "..." if len(content_str) > 100 else content_str
            }
    
    def _estimate_tokens(self, content: Any) -> int:
        """Estimate token count for content"""
        if isinstance(content, str):
            return len(content.split()) * 1.3  # Rough token estimation
        elif isinstance(content, dict):
            return sum(self._estimate_tokens(str(k) + str(v)) for k, v in content.items())
        elif isinstance(content, list):
            return sum(self._estimate_tokens(item) for item in content)
        elif isinstance(content, pd.DataFrame):
            return content.memory_usage(deep=True).sum() // 100  # Rough estimation
        else:
            return len(str(content).split()) * 1.3

class SmartCache:
    """Intelligent caching system with automatic cleanup"""
    
    def __init__(self, max_size_mb: int = 100, default_ttl: int = 3600):
        self.cache = {}
        self.max_size = max_size_mb * 1024 * 1024
        self.default_ttl = default_ttl
    
    def store(self, key: str, data: Any, ttl: Optional[int] = None) -> str:
        """Store data and return reference key"""
        ttl = ttl or self.default_ttl
        
        # Create hash-based reference
        data_hash = self._create_hash(data)
        reference = f"cache://{data_hash[:12]}"
        
        # Store with metadata
        self.cache[reference] = {
            'data': data,
            'created': time.time(),
            'ttl': ttl,
            'size': self._estimate_size(data),
            'access_count': 0,
            'last_access': time.time()
        }
        
        self._cleanup_if_needed()
        return reference
    
    def retrieve(self, reference: str) -> Optional[Any]:
        """Retrieve data by reference"""
        if reference.startswith('cache://') and reference in self.cache:
            entry = self.cache[reference]
            
            # Check expiration
            if time.time() - entry['created'] > entry['ttl']:
                del self.cache[reference]
                return None
            
            # Update access metadata
            entry['access_count'] += 1
            entry['last_access'] = time.time()
            
            return entry['data']
        
        return None
    
    def _create_hash(self, data: Any) -> str:
        """Create hash for data"""
        if isinstance(data, pd.DataFrame):
            content = data.to_string()
        else:
            content = str(data)
        
        return hashlib.md5(content.encode()).hexdigest()
    
    def _estimate_size(self, data: Any) -> int:
        """Estimate data size in bytes"""
        try:
            return len(pickle.dumps(data))
        except:
            return len(str(data).encode())
    
    def _cleanup_if_needed(self):
        """Clean up cache if size limit exceeded"""
        total_size = sum(entry['size'] for entry in self.cache.values())
        
        if total_size > self.max_size:
            # Sort by last access time (LRU)
            sorted_items = sorted(
                self.cache.items(),
                key=lambda x: x[1]['last_access']
            )
            
            # Remove oldest items until under 80% capacity
            target_size = self.max_size * 0.8
            for ref, entry in sorted_items:
                if total_size <= target_size:
                    break
                del self.cache[ref]
                total_size -= entry['size']

class BaseMarketingFlow(Flow[FlowState]):
    """Base flow class with common functionality"""
    
    def __init__(self):
        super().__init__()
        self.context_manager = ContextManager()
        self.cache = SmartCache()
    
    def _cache_data(self, data: Any, cache_type: str = "data") -> str:
        """Cache data and return reference"""
        cache_key = f"{cache_type}_{int(time.time())}"
        reference = self.cache.store(cache_key, data)
        
        # Update state metadata
        self.state.cache_metadata[reference] = {
            'type': cache_type,
            'created': time.time(),
            'size': self.cache._estimate_size(data)
        }
        
        return reference
    
    def _get_cached_data(self, reference: str) -> Optional[Any]:
        """Retrieve cached data by reference"""
        return self.cache.retrieve(reference)
    
    def _create_data_hash(self, data: Any) -> str:
        """Create hash for data identification"""
        return self.cache._create_hash(data)
    
    def _optimize_context_for_step(self, context: Dict[str, Any], step_name: str) -> Dict[str, Any]:
        """Optimize context for specific step"""
        self.state.current_step = step_name
        self.state.step_history.append(step_name)
        
        # Apply context optimization
        optimized = self.context_manager.optimize_context(
            context, 
            strategy='progressive'
        )
        
        return optimized
    
    def _extract_key_insights(self, analysis_result: Any) -> Dict[str, Any]:
        """Extract key insights from analysis result"""
        if isinstance(analysis_result, dict):
            insights = {}
            
            # Extract key metrics
            for key, value in analysis_result.items():
                if any(keyword in key.lower() for keyword in ['top', 'best', 'highest', 'key', 'main']):
                    insights[key] = value
                elif isinstance(value, (int, float)) and abs(value) > 0:
                    insights[key] = value
            
            return insights
        
        return {'summary': str(analysis_result)[:200]}
    
    def _create_compact_summary(self, data: Any) -> str:
        """Create compact summary for context"""
        if isinstance(data, pd.DataFrame):
            return f"DataFrame: {data.shape[0]} rows, {data.shape[1]} columns"
        elif isinstance(data, dict):
            return f"Analysis results with {len(data)} key findings"
        elif isinstance(data, list):
            return f"List with {len(data)} items"
        else:
            return str(data)[:100] + "..." if len(str(data)) > 100 else str(data)