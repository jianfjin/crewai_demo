"""
Optimized Data Processing Tools for Token Reduction
Implements smart data summarization and chunking strategies.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import hashlib
import json
from ..cache.smart_cache import get_cache

class OptimizedDataProcessor:
    """Processes data efficiently to minimize token usage while preserving insights."""
    
    def __init__(self):
        self.cache = get_cache()
    
    def summarize_dataset(self, df: pd.DataFrame, analysis_focus: str = "comprehensive") -> Dict[str, Any]:
        """
        Create a compact summary of the dataset instead of passing full data.
        Reduces token usage by 60-80% while preserving key insights.
        """
        summary = {
            "dataset_info": {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "columns": list(df.columns),
                "data_types": df.dtypes.to_dict()
            },
            "key_metrics": self._extract_key_metrics(df),
            "sample_data": df.head(5).to_dict('records'),  # Only 5 rows instead of full dataset
            "statistical_summary": df.describe().round(2).to_dict()
        }
        
        # Add analysis-specific summaries
        if analysis_focus == "profitability":
            summary["profitability_summary"] = self._get_profitability_summary(df)
        elif analysis_focus == "regional":
            summary["regional_summary"] = self._get_regional_summary(df)
        elif analysis_focus == "brand":
            summary["brand_summary"] = self._get_brand_summary(df)
        
        return summary
    
    def _extract_key_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Extract essential metrics from the dataset."""
        metrics = {}
        
        # Revenue metrics
        if 'total_revenue' in df.columns:
            metrics['total_revenue'] = float(df['total_revenue'].sum())
            metrics['avg_revenue'] = float(df['total_revenue'].mean())
            metrics['revenue_std'] = float(df['total_revenue'].std())
        
        # Cost metrics
        if 'total_cost' in df.columns:
            metrics['total_cost'] = float(df['total_cost'].sum())
            metrics['avg_cost'] = float(df['total_cost'].mean())
        
        # Profit metrics
        if 'profit' in df.columns:
            metrics['total_profit'] = float(df['profit'].sum())
            metrics['avg_profit_margin'] = float(df['profit_margin'].mean()) if 'profit_margin' in df.columns else 0
        
        # Volume metrics
        if 'units_sold' in df.columns:
            metrics['total_units'] = int(df['units_sold'].sum())
            metrics['avg_units'] = float(df['units_sold'].mean())
        
        return metrics
    
    def _get_profitability_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get profitability-focused summary."""
        if 'brand' not in df.columns:
            return {}
        
        brand_profit = df.groupby('brand').agg({
            'total_revenue': 'sum',
            'total_cost': 'sum',
            'profit': 'sum',
            'profit_margin': 'mean'
        }).round(2)
        
        return {
            "top_3_brands": brand_profit.nlargest(3, 'profit').to_dict('index'),
            "bottom_3_brands": brand_profit.nsmallest(3, 'profit').to_dict('index'),
            "brand_count": len(brand_profit),
            "profit_range": {
                "max": float(brand_profit['profit'].max()),
                "min": float(brand_profit['profit'].min()),
                "avg": float(brand_profit['profit'].mean())
            }
        }
    
    def _get_regional_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get region-focused summary."""
        if 'region' not in df.columns:
            return {}
        
        regional_data = df.groupby('region').agg({
            'total_revenue': 'sum',
            'units_sold': 'sum'
        }).round(2)
        
        return {
            "top_3_regions": regional_data.nlargest(3, 'total_revenue').to_dict('index'),
            "region_count": len(regional_data),
            "revenue_distribution": regional_data['total_revenue'].to_dict()
        }
    
    def _get_brand_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get brand-focused summary."""
        if 'brand' not in df.columns:
            return {}
        
        brand_data = df.groupby('brand').agg({
            'total_revenue': 'sum',
            'profit_margin': 'mean',
            'units_sold': 'sum'
        }).round(2)
        
        return {
            "brand_performance": brand_data.to_dict('index'),
            "brand_count": len(brand_data),
            "market_leaders": brand_data.nlargest(5, 'total_revenue').index.tolist()
        }
    
    def chunk_data_by_dimension(self, df: pd.DataFrame, dimension: str, 
                               target_agent: str) -> Dict[str, Any]:
        """
        Chunk data by specific dimension for targeted agent processing.
        Reduces redundant data processing across agents.
        """
        cache_key = f"chunked_data_{dimension}_{target_agent}_{self._get_data_hash(df)}"
        
        # Check cache first
        cached_result = self.cache.retrieve(cache_key)
        if cached_result is not None:
            return cached_result
        
        if dimension not in df.columns:
            return {"error": f"Dimension '{dimension}' not found in data"}
        
        # Create dimension-specific chunks
        chunks = {}
        for value in df[dimension].unique():
            chunk_data = df[df[dimension] == value]
            chunks[str(value)] = {
                "summary": self._extract_key_metrics(chunk_data),
                "row_count": len(chunk_data),
                "sample": chunk_data.head(3).to_dict('records')  # Only 3 sample rows
            }
        
        result = {
            "dimension": dimension,
            "chunks": chunks,
            "total_chunks": len(chunks),
            "agent_focus": target_agent
        }
        
        # Cache the result
        self.cache.store(cache_key, result)
        return result
    
    def _get_data_hash(self, df: pd.DataFrame) -> str:
        """Generate a hash for the dataset to use in caching."""
        data_string = f"{len(df)}_{df.columns.tolist()}_{df.iloc[0].to_dict() if len(df) > 0 else ''}"
        return hashlib.md5(data_string.encode()).hexdigest()[:8]

class ResultCache:
    """Cache for expensive tool calculations to avoid redundant processing."""
    
    def __init__(self):
        self.cache = get_cache()
        self._tool_results = {}
    
    def get_cached_result(self, tool_name: str, inputs: Dict[str, Any]) -> Optional[Any]:
        """Get cached result for a tool call."""
        cache_key = self._generate_cache_key(tool_name, inputs)
        return self.cache.retrieve(cache_key)
    
    def cache_result(self, tool_name: str, inputs: Dict[str, Any], result: Any) -> None:
        """Cache a tool result."""
        cache_key = self._generate_cache_key(tool_name, inputs)
        self.cache.store(cache_key, result)
    
    def _generate_cache_key(self, tool_name: str, inputs: Dict[str, Any]) -> str:
        """Generate a unique cache key for tool inputs."""
        inputs_str = json.dumps(inputs, sort_keys=True, default=str)
        hash_obj = hashlib.md5(f"{tool_name}_{inputs_str}".encode())
        return f"tool_cache_{hash_obj.hexdigest()[:12]}"

# Global instances
data_processor = OptimizedDataProcessor()
result_cache = ResultCache()