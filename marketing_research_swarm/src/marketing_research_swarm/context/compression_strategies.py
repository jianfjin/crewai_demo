"""
Context Compression Strategies
Implements Feature 3: Compression strategies with summarization and context trimming
"""

import json
import re
import hashlib
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime
import pandas as pd

logger = logging.getLogger(__name__)


class CompressionLevel(Enum):
    """Compression level options"""
    NONE = "none"
    LIGHT = "light"
    MEDIUM = "medium"
    AGGRESSIVE = "aggressive"
    EXTREME = "extreme"


@dataclass
class CompressionResult:
    """Result of a compression operation"""
    original_size: int
    compressed_size: int
    compression_ratio: float
    method_used: str
    quality_score: float
    metadata: Dict[str, Any]


class ContextCompressor:
    """
    Advanced context compression system that reduces token usage while
    preserving essential information through intelligent summarization.
    """
    
    def __init__(self):
        """Initialize the context compressor."""
        self.compression_methods = {
            CompressionLevel.LIGHT: self._light_compression,
            CompressionLevel.MEDIUM: self._medium_compression,
            CompressionLevel.AGGRESSIVE: self._aggressive_compression,
            CompressionLevel.EXTREME: self._extreme_compression
        }
        
    def compress_context(
        self,
        context: Dict[str, Any],
        level: CompressionLevel = CompressionLevel.MEDIUM,
        preserve_keys: List[str] = None,
        target_reduction: float = 0.5
    ) -> Tuple[Dict[str, Any], CompressionResult]:
        """
        Compress context using the specified compression level.
        
        Args:
            context: Context dictionary to compress
            level: Compression level to apply
            preserve_keys: Keys that should not be compressed
            target_reduction: Target compression ratio (0.5 = 50% reduction)
            
        Returns:
            Tuple of (compressed_context, compression_result)
        """
        preserve_keys = preserve_keys or []
        
        # Calculate original size
        original_size = self._calculate_context_size(context)
        
        # Apply compression
        if level == CompressionLevel.NONE:
            compressed_context = context.copy()
            method_used = "none"
        else:
            compression_method = self.compression_methods[level]
            compressed_context = compression_method(context, preserve_keys, target_reduction)
            method_used = level.value
        
        # Calculate compressed size
        compressed_size = self._calculate_context_size(compressed_context)
        
        # Calculate metrics
        compression_ratio = 1 - (compressed_size / max(original_size, 1))
        quality_score = self._assess_compression_quality(context, compressed_context)
        
        result = CompressionResult(
            original_size=original_size,
            compressed_size=compressed_size,
            compression_ratio=compression_ratio,
            method_used=method_used,
            quality_score=quality_score,
            metadata={
                "preserved_keys": preserve_keys,
                "target_reduction": target_reduction,
                "items_compressed": len(context) - len(compressed_context)
            }
        )
        
        logger.info(
            f"Context compressed: {original_size} -> {compressed_size} tokens "
            f"({compression_ratio:.1%} reduction) using {method_used}"
        )
        
        return compressed_context, result
    
    def _light_compression(
        self,
        context: Dict[str, Any],
        preserve_keys: List[str],
        target_reduction: float
    ) -> Dict[str, Any]:
        """Light compression: Remove redundant data and trim long strings."""
        
        compressed = {}
        
        for key, value in context.items():
            if key in preserve_keys:
                compressed[key] = value
                continue
            
            if isinstance(value, str):
                # Trim very long strings
                if len(value) > 1000:
                    compressed[key] = self._create_string_summary(value, max_length=500)
                else:
                    compressed[key] = value
            
            elif isinstance(value, dict):
                # Remove empty or very small dict entries
                if len(value) > 0:
                    compressed[key] = self._compress_dict_light(value)
            
            elif isinstance(value, list):
                # Limit list size
                if len(value) > 20:
                    compressed[key] = value[:15] + [f"... and {len(value) - 15} more items"]
                else:
                    compressed[key] = value
            
            else:
                compressed[key] = value
        
        return compressed
    
    def _medium_compression(
        self,
        context: Dict[str, Any],
        preserve_keys: List[str],
        target_reduction: float
    ) -> Dict[str, Any]:
        """Medium compression: Summarize large objects and remove low-priority items."""
        
        compressed = {}
        
        # Calculate item priorities
        item_priorities = self._calculate_item_priorities(context)
        
        for key, value in context.items():
            if key in preserve_keys:
                compressed[key] = value
                continue
            
            priority = item_priorities.get(key, 0.5)
            
            # Skip low-priority items
            if priority < 0.3:
                continue
            
            if isinstance(value, str):
                if len(value) > 500:
                    compressed[key] = self._create_intelligent_summary(value)
                else:
                    compressed[key] = value
            
            elif isinstance(value, dict):
                compressed[key] = self._compress_dict_medium(value)
            
            elif isinstance(value, list):
                compressed[key] = self._compress_list_medium(value)
            
            elif isinstance(value, pd.DataFrame):
                compressed[key] = self._compress_dataframe(value)
            
            else:
                compressed[key] = value
        
        return compressed
    
    def _aggressive_compression(
        self,
        context: Dict[str, Any],
        preserve_keys: List[str],
        target_reduction: float
    ) -> Dict[str, Any]:
        """Aggressive compression: Keep only essential information."""
        
        compressed = {}
        
        # Calculate item priorities
        item_priorities = self._calculate_item_priorities(context)
        
        # Sort items by priority
        sorted_items = sorted(
            context.items(),
            key=lambda x: item_priorities.get(x[0], 0),
            reverse=True
        )
        
        current_size = 0
        target_size = self._calculate_context_size(context) * (1 - target_reduction)
        
        for key, value in sorted_items:
            if key in preserve_keys:
                compressed[key] = value
                current_size += self._estimate_item_size(value)
                continue
            
            # Compress the value
            compressed_value = self._compress_value_aggressive(value)
            item_size = self._estimate_item_size(compressed_value)
            
            # Add if within budget
            if current_size + item_size <= target_size:
                compressed[key] = compressed_value
                current_size += item_size
            else:
                # Try to add a minimal summary
                summary = self._create_minimal_summary(value)
                summary_size = self._estimate_item_size(summary)
                
                if current_size + summary_size <= target_size:
                    compressed[f"{key}_summary"] = summary
                    current_size += summary_size
        
        return compressed
    
    def _extreme_compression(
        self,
        context: Dict[str, Any],
        preserve_keys: List[str],
        target_reduction: float
    ) -> Dict[str, Any]:
        """Extreme compression: Keep only critical information."""
        
        compressed = {}
        
        # Always preserve required keys
        for key in preserve_keys:
            if key in context:
                compressed[key] = context[key]
        
        # Calculate item priorities
        item_priorities = self._calculate_item_priorities(context)
        
        # Keep only highest priority items
        high_priority_items = {
            k: v for k, v in context.items()
            if item_priorities.get(k, 0) > 0.7 and k not in preserve_keys
        }
        
        # Compress high-priority items
        for key, value in high_priority_items.items():
            compressed[key] = self._create_minimal_summary(value)
        
        # Create overall context summary
        compressed["_context_summary"] = self._create_context_overview(context)
        
        return compressed
    
    def _compress_dict_light(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Light compression for dictionaries."""
        compressed = {}
        
        for key, value in data.items():
            if isinstance(value, str) and len(value) > 200:
                compressed[key] = value[:150] + "..."
            elif isinstance(value, list) and len(value) > 10:
                compressed[key] = value[:8] + ["..."]
            elif value is not None and str(value).strip():  # Skip empty values
                compressed[key] = value
        
        return compressed
    
    def _compress_dict_medium(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Medium compression for dictionaries."""
        if len(data) <= 5:
            return data
        
        # Keep most important keys
        important_keys = ["analysis", "summary", "result", "conclusion", "recommendation"]
        compressed = {}
        
        # Add important keys first
        for key in important_keys:
            if key in data:
                compressed[key] = data[key]
        
        # Add other keys up to limit
        remaining_keys = [k for k in data.keys() if k not in important_keys]
        for key in remaining_keys[:3]:
            compressed[key] = data[key]
        
        if len(remaining_keys) > 3:
            compressed["_additional_fields"] = len(remaining_keys) - 3
        
        return compressed
    
    def _compress_list_medium(self, data: List[Any]) -> Union[List[Any], Dict[str, Any]]:
        """Medium compression for lists."""
        if len(data) <= 10:
            return data
        
        # Sample from beginning, middle, and end
        compressed = data[:3] + data[len(data)//2-1:len(data)//2+2] + data[-3:]
        
        return {
            "sample_items": compressed,
            "total_count": len(data),
            "compression_note": f"Showing 8 of {len(data)} items"
        }
    
    def _compress_dataframe(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compress pandas DataFrame to essential information."""
        return {
            "type": "dataframe",
            "shape": df.shape,
            "columns": df.columns.tolist()[:10],
            "dtypes": df.dtypes.to_dict(),
            "sample_data": df.head(3).to_dict() if len(df) > 0 else {},
            "summary_stats": self._get_dataframe_summary(df)
        }
    
    def _get_dataframe_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get summary statistics for a DataFrame."""
        summary = {"row_count": len(df), "column_count": len(df.columns)}
        
        # Numeric columns summary
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            summary["numeric_summary"] = {
                "columns": len(numeric_cols),
                "total_sum": df[numeric_cols].sum().sum() if len(numeric_cols) > 0 else 0
            }
        
        # Categorical columns summary
        categorical_cols = df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            summary["categorical_summary"] = {
                "columns": len(categorical_cols),
                "unique_values": {col: df[col].nunique() for col in categorical_cols[:3]}
            }
        
        return summary
    
    def _compress_value_aggressive(self, value: Any) -> Any:
        """Aggressively compress a single value."""
        if isinstance(value, str):
            if len(value) > 100:
                return self._create_intelligent_summary(value, max_length=50)
            return value
        
        elif isinstance(value, dict):
            # Keep only 2-3 most important keys
            important_keys = ["result", "analysis", "summary", "value", "score"]
            compressed = {}
            
            for key in important_keys:
                if key in value:
                    compressed[key] = value[key]
                    if len(compressed) >= 2:
                        break
            
            if not compressed and value:
                # Take first key if no important keys found
                first_key = next(iter(value))
                compressed[first_key] = value[first_key]
            
            return compressed
        
        elif isinstance(value, list):
            if len(value) <= 3:
                return value
            return [value[0], "...", value[-1]]
        
        else:
            return value
    
    def _create_string_summary(self, text: str, max_length: int = 200) -> str:
        """Create a summary of a long string."""
        if len(text) <= max_length:
            return text
        
        # Try to find sentence boundaries
        sentences = re.split(r'[.!?]+', text)
        
        summary = ""
        for sentence in sentences:
            if len(summary + sentence) <= max_length - 3:
                summary += sentence + ". "
            else:
                break
        
        if not summary.strip():
            # Fallback to simple truncation
            summary = text[:max_length-3]
        
        return summary.strip() + "..."
    
    def _create_intelligent_summary(self, content: Any, max_length: int = 150) -> str:
        """Create an intelligent summary of content."""
        text = str(content)
        
        if len(text) <= max_length:
            return text
        
        # Extract key phrases and numbers
        key_phrases = re.findall(r'\b(?:increase|decrease|improve|reduce|optimize|analyze|recommend|conclude)\w*\b', text.lower())
        numbers = re.findall(r'\b\d+(?:\.\d+)?%?\b', text)
        
        # Build summary
        summary_parts = []
        
        if key_phrases:
            summary_parts.append(f"Key actions: {', '.join(set(key_phrases[:3]))}")
        
        if numbers:
            summary_parts.append(f"Key metrics: {', '.join(numbers[:3])}")
        
        # Add beginning of text
        text_start = text[:max_length//2].split('.')[0]
        if text_start:
            summary_parts.insert(0, text_start)
        
        summary = ". ".join(summary_parts)
        
        if len(summary) > max_length:
            summary = summary[:max_length-3] + "..."
        
        return summary
    
    def _create_minimal_summary(self, content: Any) -> str:
        """Create a minimal summary of content."""
        if isinstance(content, dict):
            return f"Dict with {len(content)} keys: {list(content.keys())[:3]}"
        elif isinstance(content, list):
            return f"List with {len(content)} items"
        elif isinstance(content, str):
            return content[:50] + "..." if len(content) > 50 else content
        elif isinstance(content, pd.DataFrame):
            return f"DataFrame: {content.shape[0]} rows × {content.shape[1]} cols"
        else:
            return f"{type(content).__name__}: {str(content)[:30]}"
    
    def _create_context_overview(self, context: Dict[str, Any]) -> str:
        """Create an overview of the entire context."""
        overview_parts = []
        
        # Count different types of content
        type_counts = {}
        for value in context.values():
            type_name = type(value).__name__
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        overview_parts.append(f"Context contains {len(context)} items")
        overview_parts.append(f"Types: {dict(type_counts)}")
        
        # Identify key sections
        key_sections = [k for k in context.keys() if any(
            keyword in k.lower() for keyword in ["result", "analysis", "summary", "data"]
        )]
        
        if key_sections:
            overview_parts.append(f"Key sections: {key_sections[:5]}")
        
        return ". ".join(overview_parts)
    
    def _calculate_item_priorities(self, context: Dict[str, Any]) -> Dict[str, float]:
        """Calculate priority scores for context items."""
        priorities = {}
        
        for key, value in context.items():
            priority = 0.5  # Base priority
            
            key_lower = key.lower()
            
            # High priority keywords
            if any(keyword in key_lower for keyword in ["result", "analysis", "summary", "conclusion"]):
                priority += 0.3
            
            # Medium priority keywords
            if any(keyword in key_lower for keyword in ["data", "metric", "performance", "insight"]):
                priority += 0.2
            
            # Low priority keywords
            if any(keyword in key_lower for keyword in ["temp", "cache", "debug", "log"]):
                priority -= 0.2
            
            # Content-based priority
            if isinstance(value, dict) and len(value) > 0:
                priority += 0.1
            elif isinstance(value, str) and len(value) > 100:
                priority += 0.1
            elif isinstance(value, pd.DataFrame):
                priority += 0.2
            
            priorities[key] = max(0.0, min(1.0, priority))
        
        return priorities
    
    def _calculate_context_size(self, context: Dict[str, Any]) -> int:
        """Calculate the total size of context in tokens."""
        total_size = 0
        for value in context.values():
            total_size += self._estimate_item_size(value)
        return total_size
    
    def _estimate_item_size(self, item: Any) -> int:
        """Estimate the size of an item in tokens."""
        if isinstance(item, str):
            return max(1, len(item.split()) * 1.3)
        elif isinstance(item, dict):
            return sum(self._estimate_item_size(f"{k}: {v}") for k, v in item.items())
        elif isinstance(item, list):
            return sum(self._estimate_item_size(str(i)) for i in item[:10])
        elif isinstance(item, pd.DataFrame):
            return item.memory_usage(deep=True).sum() // 100
        else:
            return max(1, len(str(item).split()) * 1.3)
    
    def _assess_compression_quality(self, original: Dict[str, Any], compressed: Dict[str, Any]) -> float:
        """Assess the quality of compression (information preservation)."""
        
        # Simple heuristic based on key preservation and content similarity
        original_keys = set(original.keys())
        compressed_keys = set(compressed.keys())
        
        # Key preservation score
        key_preservation = len(compressed_keys & original_keys) / max(len(original_keys), 1)
        
        # Content preservation score (simplified)
        content_preservation = 0.0
        common_keys = original_keys & compressed_keys
        
        if common_keys:
            for key in common_keys:
                orig_str = str(original[key])
                comp_str = str(compressed[key])
                
                # Simple similarity based on length ratio
                if len(orig_str) > 0:
                    similarity = min(len(comp_str) / len(orig_str), 1.0)
                    content_preservation += similarity
            
            content_preservation /= len(common_keys)
        
        # Overall quality score
        quality = (key_preservation * 0.4) + (content_preservation * 0.6)
        return quality


# Global instance
_global_compressor = None


def get_context_compressor() -> ContextCompressor:
    """Get the global context compressor instance."""
    global _global_compressor
    if _global_compressor is None:
        _global_compressor = ContextCompressor()
    return _global_compressor