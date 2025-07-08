"""
Advanced Context Management System
Implements multiple strategies for token optimization
"""

from typing import Dict, Any, List, Optional, Union
import time
import json
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import pandas as pd

class ContextPriority(Enum):
    CRITICAL = "critical"
    IMPORTANT = "important" 
    USEFUL = "useful"
    OPTIONAL = "optional"

class ContextStrategy(Enum):
    PROGRESSIVE_PRUNING = "progressive_pruning"
    ABSTRACTED_SUMMARIES = "abstracted_summaries"
    MINIMAL_CONTEXT = "minimal_context"
    STATELESS = "stateless"

@dataclass
class ContextElement:
    key: str
    value: Any
    priority: ContextPriority
    created_at: float
    last_accessed: float
    token_estimate: int
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

class AdvancedContextManager:
    # Make ContextPriority accessible as class attribute
    ContextPriority = ContextPriority
    """
    Advanced context management with multiple optimization strategies
    """
    
    def __init__(self, token_budget: int = 4000, aging_threshold: int = 300):
        self.token_budget = token_budget
        self.aging_threshold = aging_threshold  # seconds
        self.context_elements: Dict[str, ContextElement] = {}
        self.dependency_graph: Dict[str, List[str]] = {}
        
        # Strategy weights
        self.priority_weights = {
            ContextPriority.CRITICAL: 1.0,
            ContextPriority.IMPORTANT: 0.7,
            ContextPriority.USEFUL: 0.4,
            ContextPriority.OPTIONAL: 0.1
        }
    
    def add_context(self, key: str, value: Any, priority: ContextPriority = ContextPriority.USEFUL, 
                   dependencies: List[str] = None) -> None:
        """Add context element with metadata"""
        current_time = time.time()
        token_estimate = self._estimate_tokens(value)
        
        element = ContextElement(
            key=key,
            value=value,
            priority=priority,
            created_at=current_time,
            last_accessed=current_time,
            token_estimate=token_estimate,
            dependencies=dependencies or []
        )
        
        self.context_elements[key] = element
        
        # Update dependency graph
        if dependencies:
            for dep in dependencies:
                if dep not in self.dependency_graph:
                    self.dependency_graph[dep] = []
                self.dependency_graph[dep].append(key)
    
    def get_optimized_context(self, strategy: ContextStrategy = ContextStrategy.PROGRESSIVE_PRUNING,
                            required_keys: List[str] = None) -> Dict[str, Any]:
        """Get optimized context using specified strategy"""
        
        # Apply aging first
        self._apply_aging()
        
        # Get current context
        current_context = {k: v.value for k, v in self.context_elements.items()}
        
        # Apply optimization strategy
        if strategy == ContextStrategy.PROGRESSIVE_PRUNING:
            return self._progressive_pruning(current_context, required_keys)
        elif strategy == ContextStrategy.ABSTRACTED_SUMMARIES:
            return self._abstracted_summaries(current_context)
        elif strategy == ContextStrategy.MINIMAL_CONTEXT:
            return self._minimal_context(current_context, required_keys)
        elif strategy == ContextStrategy.STATELESS:
            return self._stateless_context(current_context, required_keys)
        else:
            return current_context
    
    def _progressive_pruning(self, context: Dict[str, Any], required_keys: List[str] = None) -> Dict[str, Any]:
        """Progressive pruning based on priority and dependencies"""
        required_keys = required_keys or []
        optimized = {}
        remaining_budget = self.token_budget
        
        # Always include required keys
        for key in required_keys:
            if key in context:
                optimized[key] = context[key]
                if key in self.context_elements:
                    remaining_budget -= self.context_elements[key].token_estimate
        
        # Sort remaining elements by priority and age
        remaining_elements = [
            (k, v) for k, v in self.context_elements.items() 
            if k not in required_keys
        ]
        
        remaining_elements.sort(
            key=lambda x: (
                self.priority_weights[x[1].priority],
                -x[1].last_accessed,  # More recent = higher priority
                -len(self.dependency_graph.get(x[0], []))  # More dependents = higher priority
            ),
            reverse=True
        )
        
        # Add elements until budget exhausted
        for key, element in remaining_elements:
            if element.token_estimate <= remaining_budget:
                optimized[key] = element.value
                remaining_budget -= element.token_estimate
                element.last_accessed = time.time()
            elif remaining_budget > 100:  # Space for summary
                optimized[key] = self._create_intelligent_summary(element.value)
                remaining_budget -= 100
        
        return optimized
    
    def _abstracted_summaries(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create abstracted summaries of all large content"""
        summaries = {}
        
        for key, value in context.items():
            element = self.context_elements.get(key)
            if element and element.token_estimate > 200:
                summaries[key] = self._create_intelligent_summary(value)
            else:
                summaries[key] = value
        
        return summaries
    
    def _minimal_context(self, context: Dict[str, Any], required_keys: List[str] = None) -> Dict[str, Any]:
        """Keep only critical and required context"""
        required_keys = required_keys or []
        minimal = {}
        
        # Include required keys
        for key in required_keys:
            if key in context:
                minimal[key] = context[key]
        
        # Include only critical elements
        for key, element in self.context_elements.items():
            if element.priority == ContextPriority.CRITICAL and key not in minimal:
                minimal[key] = element.value
        
        return minimal
    
    def _stateless_context(self, context: Dict[str, Any], required_keys: List[str] = None) -> Dict[str, Any]:
        """Remove all historical context, keep only current step"""
        required_keys = required_keys or []
        stateless = {}
        
        # Current step indicators
        current_indicators = ['current_', 'step_', 'immediate_', 'active_']
        
        for key, value in context.items():
            if (key in required_keys or 
                any(indicator in key.lower() for indicator in current_indicators) or
                (key in self.context_elements and 
                 self.context_elements[key].priority == ContextPriority.CRITICAL)):
                stateless[key] = value
        
        return stateless
    
    def _apply_aging(self) -> None:
        """Apply automatic aging to context elements"""
        current_time = time.time()
        aged_keys = []
        
        for key, element in self.context_elements.items():
            age = current_time - element.created_at
            last_access_age = current_time - element.last_accessed
            
            # Age out old, unused elements
            if (age > self.aging_threshold and 
                last_access_age > self.aging_threshold and
                element.priority != ContextPriority.CRITICAL and
                not self._has_dependencies(key)):
                aged_keys.append(key)
        
        # Remove aged elements
        for key in aged_keys:
            del self.context_elements[key]
            if key in self.dependency_graph:
                del self.dependency_graph[key]
    
    def _has_dependencies(self, key: str) -> bool:
        """Check if element has dependencies that prevent removal"""
        return key in self.dependency_graph and len(self.dependency_graph[key]) > 0
    
    def _create_intelligent_summary(self, content: Any) -> Dict[str, Any]:
        """Create intelligent summary based on content type"""
        if isinstance(content, pd.DataFrame):
            return {
                'type': 'dataframe',
                'shape': content.shape,
                'columns': content.columns.tolist()[:10],
                'memory_usage': f"{content.memory_usage(deep=True).sum() / 1024:.1f} KB",
                'key_stats': self._extract_dataframe_stats(content)
            }
        elif isinstance(content, dict):
            return {
                'type': 'analysis_result',
                'keys': list(content.keys())[:10],
                'key_metrics': self._extract_key_metrics(content),
                'summary': self._extract_text_summary(content)
            }
        elif isinstance(content, list):
            return {
                'type': 'list',
                'length': len(content),
                'sample_items': content[:3] if content else [],
                'item_types': list(set(type(item).__name__ for item in content[:10]))
            }
        elif isinstance(content, str):
            return {
                'type': 'text',
                'length': len(content),
                'word_count': len(content.split()),
                'preview': content[:150] + "..." if len(content) > 150 else content,
                'key_phrases': self._extract_key_phrases(content)
            }
        else:
            return {
                'type': type(content).__name__,
                'string_repr': str(content)[:100] + "..." if len(str(content)) > 100 else str(content)
            }
    
    def _extract_dataframe_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract key statistics from DataFrame"""
        stats = {}
        
        # Numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            stats['numeric_summary'] = {
                'total_revenue': df.get('total_revenue', pd.Series()).sum() if 'total_revenue' in df.columns else None,
                'record_count': len(df),
                'numeric_columns': len(numeric_cols)
            }
        
        # Categorical columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            stats['categorical_summary'] = {
                col: df[col].nunique() for col in categorical_cols[:5]
            }
        
        return stats
    
    def _extract_key_metrics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key metrics from analysis results"""
        metrics = {}
        
        # Look for common metric patterns
        metric_patterns = ['total', 'average', 'max', 'min', 'count', 'percentage', 'ratio', 'margin', 'roi']
        
        for key, value in data.items():
            key_lower = key.lower()
            if any(pattern in key_lower for pattern in metric_patterns):
                if isinstance(value, (int, float)):
                    metrics[key] = value
                elif isinstance(value, str) and any(char.isdigit() for char in value):
                    metrics[key] = value
        
        return metrics
    
    def _extract_text_summary(self, data: Dict[str, Any]) -> str:
        """Extract text summary from complex data"""
        summary_parts = []
        
        for key, value in list(data.items())[:5]:  # Limit to first 5 items
            if isinstance(value, str) and len(value) < 100:
                summary_parts.append(f"{key}: {value}")
            elif isinstance(value, (int, float)):
                summary_parts.append(f"{key}: {value}")
        
        return "; ".join(summary_parts)
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases from text"""
        # Simple keyword extraction
        words = text.lower().split()
        
        # Look for important business terms
        important_terms = ['revenue', 'profit', 'margin', 'roi', 'performance', 'analysis', 'optimization']
        key_phrases = []
        
        for term in important_terms:
            if term in words:
                # Find context around the term
                for i, word in enumerate(words):
                    if word == term and i > 0 and i < len(words) - 1:
                        phrase = f"{words[i-1]} {word} {words[i+1]}"
                        key_phrases.append(phrase)
        
        return key_phrases[:5]  # Limit to 5 phrases
    
    def _estimate_tokens(self, content: Any) -> int:
        """Estimate token count for content"""
        if isinstance(content, str):
            return int(len(content.split()) * 1.3)
        elif isinstance(content, dict):
            return sum(self._estimate_tokens(f"{k}: {v}") for k, v in content.items())
        elif isinstance(content, list):
            return sum(self._estimate_tokens(str(item)) for item in content[:10])  # Limit for estimation
        elif isinstance(content, pd.DataFrame):
            return int(content.memory_usage(deep=True).sum() / 100)  # Rough estimation
        else:
            return int(len(str(content).split()) * 1.3)
    
    def get_context_stats(self) -> Dict[str, Any]:
        """Get statistics about current context"""
        total_tokens = sum(element.token_estimate for element in self.context_elements.values())
        
        priority_breakdown = {}
        for priority in ContextPriority:
            count = sum(1 for element in self.context_elements.values() if element.priority == priority)
            tokens = sum(element.token_estimate for element in self.context_elements.values() if element.priority == priority)
            priority_breakdown[priority.value] = {'count': count, 'tokens': tokens}
        
        return {
            'total_elements': len(self.context_elements),
            'total_tokens': total_tokens,
            'budget_utilization': f"{(total_tokens / self.token_budget) * 100:.1f}%",
            'priority_breakdown': priority_breakdown,
            'dependencies': len(self.dependency_graph),
            'aged_elements': self._count_aged_elements()
        }
    
    def _count_aged_elements(self) -> int:
        """Count elements that would be aged out"""
        current_time = time.time()
        aged_count = 0
        
        for element in self.context_elements.values():
            age = current_time - element.created_at
            last_access_age = current_time - element.last_accessed
            
            if (age > self.aging_threshold and 
                last_access_age > self.aging_threshold and
                element.priority != ContextPriority.CRITICAL):
                aged_count += 1
        
        return aged_count