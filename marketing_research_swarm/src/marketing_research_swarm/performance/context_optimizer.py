"""
Context Isolation Optimizer for Token Usage Reduction

This module provides advanced context optimization techniques to minimize token usage
while maintaining agent effectiveness through intelligent context isolation and reference management.
"""

import json
import uuid
import hashlib
import time
from typing import Dict, Any, List, Optional, Set, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ContextStrategy(Enum):
    FULL_CONTEXT = "full_context"
    ISOLATED_CONTEXT = "isolated_context"
    REFERENCE_BASED = "reference_based"
    SMART_SUMMARY = "smart_summary"

@dataclass
class ContextReference:
    """Reference to stored context data."""
    reference_id: str
    data_type: str
    summary: str
    size_bytes: int
    created_at: datetime
    access_count: int = 0

@dataclass
class ContextMetrics:
    """Metrics for context optimization performance."""
    original_size_bytes: int
    optimized_size_bytes: int
    compression_ratio: float
    references_created: int
    token_savings_estimated: int

class ContextOptimizer:
    """
    Advanced context optimizer that reduces token usage through intelligent
    context isolation, reference management, and smart summarization.
    """
    
    def __init__(self, max_context_size: int = 4000, reference_threshold: int = 500):
        """
        Initialize the context optimizer.
        
        Args:
            max_context_size: Maximum context size in tokens (approximate)
            reference_threshold: Minimum size in characters to create a reference
        """
        self.max_context_size = max_context_size
        self.reference_threshold = reference_threshold
        
        # Storage for references and summaries
        self.reference_store: Dict[str, Any] = {}
        self.context_summaries: Dict[str, str] = {}
        self.access_patterns: Dict[str, List[datetime]] = {}
        
        # Performance metrics
        self.total_token_savings = 0
        self.total_references_created = 0
        self.optimization_history: List[ContextMetrics] = []
        
        logger.info(f"🎯 ContextOptimizer initialized (max_size={max_context_size}, threshold={reference_threshold})")
    
    def optimize_context_for_agent(self, 
                                 agent_role: str,
                                 full_context: Dict[str, Any],
                                 strategy: ContextStrategy = ContextStrategy.SMART_SUMMARY) -> Tuple[Dict[str, Any], ContextMetrics]:
        """
        Optimize context for a specific agent using the specified strategy.
        
        Args:
            agent_role: Role of the agent requesting context
            full_context: Complete context data
            strategy: Optimization strategy to use
            
        Returns:
            Tuple of (optimized_context, metrics)
        """
        start_time = time.time()
        original_size = len(json.dumps(full_context, default=str))
        
        if strategy == ContextStrategy.FULL_CONTEXT:
            optimized_context = full_context
            metrics = ContextMetrics(
                original_size_bytes=original_size,
                optimized_size_bytes=original_size,
                compression_ratio=1.0,
                references_created=0,
                token_savings_estimated=0
            )
        elif strategy == ContextStrategy.ISOLATED_CONTEXT:
            optimized_context = self._create_isolated_context(agent_role, full_context)
            optimized_size = len(json.dumps(optimized_context, default=str))
            metrics = self._calculate_metrics(original_size, optimized_size, 0)
        elif strategy == ContextStrategy.REFERENCE_BASED:
            optimized_context, references_created = self._create_reference_based_context(agent_role, full_context)
            optimized_size = len(json.dumps(optimized_context, default=str))
            metrics = self._calculate_metrics(original_size, optimized_size, references_created)
        else:  # SMART_SUMMARY
            optimized_context, references_created = self._create_smart_summary_context(agent_role, full_context)
            optimized_size = len(json.dumps(optimized_context, default=str))
            metrics = self._calculate_metrics(original_size, optimized_size, references_created)
        
        # Update performance tracking
        self.optimization_history.append(metrics)
        self.total_token_savings += metrics.token_savings_estimated
        self.total_references_created += metrics.references_created
        
        optimization_time = time.time() - start_time
        logger.info(f"🎯 Optimized context for {agent_role}: {metrics.compression_ratio:.2f}x compression in {optimization_time:.3f}s")
        
        return optimized_context, metrics
    
    def _create_isolated_context(self, agent_role: str, full_context: Dict[str, Any]) -> Dict[str, Any]:
        """Create isolated context with only relevant data for the agent."""
        
        # Define relevance mapping for different agent roles
        relevance_mapping = {
            'market_research_analyst': {
                'high_priority': ['market_data', 'industry_trends', 'consumer_behavior', 'market_size'],
                'medium_priority': ['competitive_landscape', 'economic_indicators'],
                'low_priority': ['technical_specs', 'internal_metrics']
            },
            'competitive_analyst': {
                'high_priority': ['competitive_landscape', 'market_share', 'competitor_analysis', 'pricing_data'],
                'medium_priority': ['market_data', 'industry_trends'],
                'low_priority': ['consumer_behavior', 'technical_specs']
            },
            'brand_performance_specialist': {
                'high_priority': ['brand_metrics', 'performance_data', 'market_share', 'consumer_sentiment'],
                'medium_priority': ['competitive_landscape', 'market_data'],
                'low_priority': ['technical_specs', 'economic_indicators']
            },
            'campaign_optimizer': {
                'high_priority': ['campaign_data', 'performance_metrics', 'roi_data', 'budget_allocation'],
                'medium_priority': ['market_data', 'competitive_landscape'],
                'low_priority': ['technical_specs', 'detailed_analytics']
            },
            'content_strategist': {
                'high_priority': ['content_performance', 'audience_data', 'engagement_metrics', 'brand_voice'],
                'medium_priority': ['market_trends', 'competitive_content'],
                'low_priority': ['technical_specs', 'financial_data']
            },
            'data_analyst': {
                'high_priority': ['raw_data', 'analytics', 'statistical_models', 'data_quality'],
                'medium_priority': ['performance_metrics', 'trends'],
                'low_priority': ['brand_voice', 'creative_assets']
            }
        }
        
        agent_relevance = relevance_mapping.get(agent_role, {
            'high_priority': [],
            'medium_priority': [],
            'low_priority': []
        })
        
        isolated_context = {
            'agent_role': agent_role,
            'timestamp': datetime.now().isoformat(),
            'context_strategy': 'isolated'
        }
        
        # Add high priority data first
        for key, value in full_context.items():
            if any(priority_key in key.lower() for priority_key in agent_relevance['high_priority']):
                isolated_context[key] = value
        
        # Add medium priority data if we have space
        current_size = len(json.dumps(isolated_context, default=str))
        if current_size < self.max_context_size * 0.7:  # 70% threshold
            for key, value in full_context.items():
                if key not in isolated_context and any(priority_key in key.lower() for priority_key in agent_relevance['medium_priority']):
                    isolated_context[key] = value
                    current_size = len(json.dumps(isolated_context, default=str))
                    if current_size > self.max_context_size * 0.9:  # 90% threshold
                        break
        
        return isolated_context
    
    def _create_reference_based_context(self, agent_role: str, full_context: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
        """Create context using references for large data objects."""
        
        reference_context = {
            'agent_role': agent_role,
            'timestamp': datetime.now().isoformat(),
            'context_strategy': 'reference_based',
            'references': {}
        }
        
        references_created = 0
        
        for key, value in full_context.items():
            value_str = json.dumps(value, default=str)
            value_size = len(value_str)
            
            if value_size > self.reference_threshold:
                # Create reference for large data
                reference_id = self._create_reference(key, value, value_size)
                reference_context['references'][key] = {
                    'reference_id': reference_id,
                    'data_type': type(value).__name__,
                    'size_bytes': value_size,
                    'summary': self._create_data_summary(key, value)
                }
                references_created += 1
            else:
                # Include small data directly
                reference_context[key] = value
        
        return reference_context, references_created
    
    def _create_smart_summary_context(self, agent_role: str, full_context: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
        """Create context with intelligent summarization and selective references."""
        
        summary_context = {
            'agent_role': agent_role,
            'timestamp': datetime.now().isoformat(),
            'context_strategy': 'smart_summary',
            'data_summaries': {},
            'references': {}
        }
        
        references_created = 0
        
        # Get agent-specific relevance
        isolated_context = self._create_isolated_context(agent_role, full_context)
        
        for key, value in full_context.items():
            value_str = json.dumps(value, default=str)
            value_size = len(value_str)
            
            if key in isolated_context:
                # High relevance - include directly or as reference
                if value_size > self.reference_threshold:
                    reference_id = self._create_reference(key, value, value_size)
                    summary_context['references'][key] = {
                        'reference_id': reference_id,
                        'data_type': type(value).__name__,
                        'size_bytes': value_size,
                        'summary': self._create_data_summary(key, value)
                    }
                    references_created += 1
                else:
                    summary_context[key] = value
            else:
                # Lower relevance - create summary only
                summary_context['data_summaries'][key] = {
                    'summary': self._create_data_summary(key, value),
                    'size_bytes': value_size,
                    'data_type': type(value).__name__
                }
        
        return summary_context, references_created
    
    def _create_reference(self, key: str, value: Any, size_bytes: int) -> str:
        """Create a reference for storing large data objects."""
        reference_id = f"ref_{hashlib.md5(f'{key}_{time.time()}'.encode()).hexdigest()[:8]}"
        
        reference = ContextReference(
            reference_id=reference_id,
            data_type=type(value).__name__,
            summary=self._create_data_summary(key, value),
            size_bytes=size_bytes,
            created_at=datetime.now()
        )
        
        # Store the actual data and reference
        self.reference_store[reference_id] = {
            'data': value,
            'reference': reference
        }
        
        logger.debug(f"📎 Created reference {reference_id} for {key} ({size_bytes} bytes)")
        return reference_id
    
    def _create_data_summary(self, key: str, value: Any) -> str:
        """Create an intelligent summary of data based on its type and content."""
        
        if isinstance(value, dict):
            if len(value) == 0:
                return "Empty dictionary"
            
            # Summarize dictionary structure
            keys = list(value.keys())[:5]  # First 5 keys
            summary = f"Dictionary with {len(value)} keys: {', '.join(keys)}"
            if len(value) > 5:
                summary += f" and {len(value) - 5} more"
            
            # Add sample values for insight
            if keys:
                sample_key = keys[0]
                sample_value = value[sample_key]
                if isinstance(sample_value, (int, float, str)) and len(str(sample_value)) < 50:
                    summary += f". Sample: {sample_key}={sample_value}"
            
            return summary
            
        elif isinstance(value, list):
            if len(value) == 0:
                return "Empty list"
            
            summary = f"List with {len(value)} items"
            if value:
                first_item = value[0]
                summary += f" of type {type(first_item).__name__}"
                if isinstance(first_item, (int, float, str)) and len(str(first_item)) < 50:
                    summary += f". First item: {first_item}"
            
            return summary
            
        elif isinstance(value, str):
            if len(value) < 100:
                return f"String: {value}"
            else:
                return f"Long string ({len(value)} chars): {value[:50]}..."
                
        elif isinstance(value, (int, float)):
            return f"Number: {value}"
            
        else:
            return f"{type(value).__name__} object"
    
    def retrieve_reference(self, reference_id: str) -> Optional[Any]:
        """Retrieve data by reference ID."""
        if reference_id in self.reference_store:
            # Update access tracking
            if reference_id not in self.access_patterns:
                self.access_patterns[reference_id] = []
            self.access_patterns[reference_id].append(datetime.now())
            
            # Update access count
            self.reference_store[reference_id]['reference'].access_count += 1
            
            return self.reference_store[reference_id]['data']
        
        return None
    
    def _calculate_metrics(self, original_size: int, optimized_size: int, references_created: int) -> ContextMetrics:
        """Calculate optimization metrics."""
        compression_ratio = original_size / max(optimized_size, 1)
        size_reduction = original_size - optimized_size
        
        # Estimate token savings (rough approximation: 1 token ≈ 4 characters)
        token_savings = max(0, size_reduction // 4)
        
        return ContextMetrics(
            original_size_bytes=original_size,
            optimized_size_bytes=optimized_size,
            compression_ratio=compression_ratio,
            references_created=references_created,
            token_savings_estimated=token_savings
        )
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        if not self.optimization_history:
            return {
                'total_optimizations': 0,
                'average_compression_ratio': 0,
                'total_token_savings': 0,
                'total_references_created': 0
            }
        
        avg_compression = sum(m.compression_ratio for m in self.optimization_history) / len(self.optimization_history)
        total_original_size = sum(m.original_size_bytes for m in self.optimization_history)
        total_optimized_size = sum(m.optimized_size_bytes for m in self.optimization_history)
        
        stats = {
            'total_optimizations': len(self.optimization_history),
            'average_compression_ratio': round(avg_compression, 2),
            'total_token_savings': self.total_token_savings,
            'total_references_created': self.total_references_created,
            'total_size_reduction_bytes': total_original_size - total_optimized_size,
            'active_references': len(self.reference_store),
            'most_accessed_references': self._get_most_accessed_references(5)
        }
        
        return stats
    
    def _get_most_accessed_references(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get the most frequently accessed references."""
        reference_access = []
        
        for ref_id, ref_data in self.reference_store.items():
            reference = ref_data['reference']
            reference_access.append({
                'reference_id': ref_id,
                'access_count': reference.access_count,
                'data_type': reference.data_type,
                'size_bytes': reference.size_bytes,
                'summary': reference.summary
            })
        
        # Sort by access count
        reference_access.sort(key=lambda x: x['access_count'], reverse=True)
        
        return reference_access[:limit]
    
    def cleanup_unused_references(self, max_age_hours: int = 24):
        """Clean up old, unused references to free memory."""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        references_to_remove = []
        for ref_id, ref_data in self.reference_store.items():
            reference = ref_data['reference']
            
            # Remove if old and never accessed, or very old
            if (reference.created_at < cutoff_time and reference.access_count == 0) or \
               (reference.created_at < cutoff_time - timedelta(hours=48)):
                references_to_remove.append(ref_id)
        
        for ref_id in references_to_remove:
            del self.reference_store[ref_id]
            if ref_id in self.access_patterns:
                del self.access_patterns[ref_id]
        
        if references_to_remove:
            logger.info(f"🧹 Cleaned up {len(references_to_remove)} unused references")
    
    def create_agent_specific_context(self, agent_role: str, workflow_data: Dict[str, Any], 
                                    previous_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create optimized context specifically for your 4-agent workflow.
        
        Args:
            agent_role: One of your 4 agents
            workflow_data: Initial workflow data
            previous_results: Results from previous agents
            
        Returns:
            Optimized context for the agent
        """
        
        # Combine all available data
        full_context = {
            'workflow_data': workflow_data,
            'previous_results': previous_results or {},
            'agent_role': agent_role,
            'timestamp': datetime.now().isoformat()
        }
        
        # Apply smart optimization
        optimized_context, metrics = self.optimize_context_for_agent(
            agent_role=agent_role,
            full_context=full_context,
            strategy=ContextStrategy.SMART_SUMMARY
        )
        
        # Add agent-specific instructions
        agent_instructions = {
            'market_research_analyst': "Focus on market structure, trends, and opportunities. Use beverage_market_analysis and time_series_analysis tools.",
            'competitive_analyst': "Analyze competitive landscape and market positioning. Use competitive analysis tools and market share calculations.",
            'brand_performance_specialist': "Evaluate brand metrics and performance indicators. Focus on brand health and market position.",
            'campaign_optimizer': "Develop optimization strategies based on all previous analysis. Focus on ROI and budget allocation."
        }
        
        optimized_context['agent_instructions'] = agent_instructions.get(agent_role, "Perform your assigned analysis tasks.")
        optimized_context['optimization_metrics'] = {
            'compression_ratio': metrics.compression_ratio,
            'token_savings': metrics.token_savings_estimated,
            'references_created': metrics.references_created
        }
        
        return optimized_context

# Global context optimizer instance
_global_context_optimizer = None

def get_context_optimizer() -> ContextOptimizer:
    """Get the global context optimizer instance."""
    global _global_context_optimizer
    if _global_context_optimizer is None:
        _global_context_optimizer = ContextOptimizer()
    return _global_context_optimizer