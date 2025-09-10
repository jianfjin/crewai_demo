"""
Data Context Manager for LangGraph Workflow

This module provides data schema, preview, and context information to all agents
to improve their understanding and analysis capabilities.
"""

import os
import logging
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class DataContextManager:
    """Manages data context, schema, and preview information for workflow agents."""
    
    def __init__(self):
        self.data_cache = {}
        self.schema_cache = {}
        self.preview_cache = {}
        
    def get_data_context(self, data_source: str = "beverage_sales") -> Dict[str, Any]:
        """Get comprehensive data context including schema, preview, and summary."""
        
        try:
            # Load data if not cached
            if data_source not in self.data_cache:
                self._load_data(data_source)
            
            df = self.data_cache.get(data_source)
            if df is None or df.empty:
                return {"error": "No data available"}
            
            # Generate comprehensive context
            context = {
                "data_source": data_source,
                "schema": self._get_schema_info(df),
                "preview": self._get_data_preview(df),
                "summary": self._get_data_summary(df),
                "distinct_values": self._get_distinct_values(df),
                "data_quality": self._assess_data_quality(df),
                "suggested_queries": self._generate_suggested_queries(df),
                "timestamp": datetime.now().isoformat()
            }
            
            # Ensure all data types are serializable
            context = self._ensure_serializable(context)
            
            return context
            
        except Exception as e:
            logger.error(f"Error getting data context: {e}")
            return {"error": str(e)}
    
    def _load_data(self, data_source: str):
        """Load data from various sources."""
        
        try:
            # Try multiple data source paths
            data_paths = [
                f"marketing_research_swarm/data/{data_source}.csv",
                f"data/{data_source}.csv",
                f"src/marketing_research_swarm/data/{data_source}.csv"
            ]
            
            df = None
            for path in data_paths:
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    logger.info(f"✅ Loaded data from {path}: {df.shape}")
                    break
            
            if df is None:
                # Fallback to tool-based data loading
                try:
                    from ..tools.langgraph_tools import get_cached_data
                    df = get_cached_data()
                    logger.info(f"✅ Loaded data from tools cache: {df.shape}")
                except Exception as tool_error:
                    logger.error(f"Failed to load data from tools: {tool_error}")
                    df = pd.DataFrame()  # Empty dataframe as last resort
            
            self.data_cache[data_source] = df
            
        except Exception as e:
            logger.error(f"Error loading data for {data_source}: {e}")
            self.data_cache[data_source] = pd.DataFrame()
    
    def _get_schema_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get detailed schema information."""
        
        schema = {
            "columns": [],
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "data_types": {},
            "nullable_columns": [],
            "numeric_columns": [],
            "categorical_columns": [],
            "date_columns": []
        }
        
        for col in df.columns:
            col_info = {
                "name": col,
                "dtype": str(df[col].dtype),
                "null_count": int(df[col].isnull().sum()),
                "null_percentage": float((df[col].isnull().sum() / len(df)) * 100),
                "unique_count": int(df[col].nunique()),
                "sample_values": [self._ensure_serializable(val) for val in df[col].dropna().head(3).tolist()]
            }
            
            schema["columns"].append(col_info)
            schema["data_types"][col] = str(df[col].dtype)
            
            # Categorize columns
            if df[col].isnull().any():
                schema["nullable_columns"].append(col)
            
            if pd.api.types.is_numeric_dtype(df[col]):
                schema["numeric_columns"].append(col)
                # Convert numpy types to native Python types
                col_info["min"] = float(df[col].min()) if pd.notna(df[col].min()) else None
                col_info["max"] = float(df[col].max()) if pd.notna(df[col].max()) else None
                col_info["mean"] = float(df[col].mean()) if pd.notna(df[col].mean()) else None
            
            if pd.api.types.is_categorical_dtype(df[col]) or df[col].nunique() < 20:
                schema["categorical_columns"].append(col)
            
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                schema["date_columns"].append(col)
        
        return schema
    
    def _get_data_preview(self, df: pd.DataFrame, n_rows: int = 10) -> Dict[str, Any]:
        """Get data preview with sample rows."""
        
        preview = {
            "head": df.head(n_rows).to_dict('records'),
            "tail": df.tail(3).to_dict('records'),
            "random_sample": df.sample(min(5, len(df))).to_dict('records') if len(df) > 0 else []
        }
        
        return preview
    
    def _get_data_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get statistical summary of the data."""
        
        summary = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024 / 1024,
            "numeric_summary": {},
            "categorical_summary": {}
        }
        
        # Numeric columns summary
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            numeric_summary = df[numeric_cols].describe().to_dict()
            # Convert numpy types to native Python types
            summary["numeric_summary"] = self._ensure_serializable(numeric_summary)
        
        # Categorical columns summary
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            summary["categorical_summary"][col] = {
                "unique_count": df[col].nunique(),
                "top_values": df[col].value_counts().head(5).to_dict(),
                "most_common": df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None
            }
        
        return summary
    
    def _get_distinct_values(self, df: pd.DataFrame, max_values: int = 50) -> Dict[str, List]:
        """Get distinct values for categorical columns."""
        
        distinct_values = {}
        
        for col in df.columns:
            if df[col].nunique() <= max_values:
                distinct_values[col] = sorted(df[col].dropna().unique().tolist())
        
        return distinct_values
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data quality metrics."""
        
        quality = {
            "completeness": float((1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100),
            "duplicate_rows": int(df.duplicated().sum()),
            "duplicate_percentage": float((df.duplicated().sum() / len(df)) * 100),
            "columns_with_nulls": df.columns[df.isnull().any()].tolist(),
            "data_freshness": "Current" if len(df) > 0 else "No data"
        }
        
        return quality
    
    def _generate_suggested_queries(self, df: pd.DataFrame) -> List[str]:
        """Generate suggested analysis queries based on data structure."""
        
        suggestions = []
        
        # Basic queries
        suggestions.append("SELECT * FROM beverage_sales LIMIT 10")
        
        # Column-based suggestions
        if 'brand' in df.columns:
            suggestions.extend([
                "SELECT brand, COUNT(*) as transactions FROM beverage_sales GROUP BY brand ORDER BY transactions DESC",
                "SELECT brand, SUM(total_revenue) as revenue FROM beverage_sales GROUP BY brand ORDER BY revenue DESC"
            ])
        
        if 'region' in df.columns:
            suggestions.append("SELECT region, COUNT(*) as transactions FROM beverage_sales GROUP BY region")
        
        if 'category' in df.columns:
            suggestions.append("SELECT category, AVG(profit_margin) as avg_margin FROM beverage_sales GROUP BY category")
        
        # Time-based suggestions
        date_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_cols:
            suggestions.append(f"SELECT {date_cols[0]}, COUNT(*) FROM beverage_sales GROUP BY {date_cols[0]} ORDER BY {date_cols[0]}")
        
        if 'year' in df.columns and 'quarter' in df.columns:
            suggestions.append("SELECT year, quarter, SUM(total_revenue) FROM beverage_sales GROUP BY year, quarter ORDER BY year, quarter")
        
        return suggestions
    
    def generate_context_prompt(self, data_source: str = "beverage_sales") -> str:
        """Generate a formatted context prompt for LLM agents."""
        
        context = self.get_data_context(data_source)
        
        if "error" in context:
            return f"Data context error: {context['error']}"
        
        schema = context["schema"]
        preview = context["preview"]
        summary = context["summary"]
        distinct_values = context["distinct_values"]
        
        prompt = f"""
## 📊 DATA CONTEXT INFORMATION

**Dataset:** {data_source}
**Shape:** {schema['shape']['rows']:,} rows × {schema['shape']['columns']} columns
**Data Quality:** {context['data_quality']['completeness']:.1f}% complete

### 🏗️ SCHEMA INFORMATION
**Columns Available:**
"""
        
        for col_info in schema["columns"][:10]:  # Show first 10 columns
            prompt += f"- **{col_info['name']}** ({col_info['dtype']}): {col_info['unique_count']} unique values"
            if col_info['sample_values']:
                prompt += f" | Examples: {', '.join(map(str, col_info['sample_values']))}"
            prompt += "\n"
        
        if len(schema["columns"]) > 10:
            prompt += f"... and {len(schema['columns']) - 10} more columns\n"
        
        prompt += f"""
### 📋 SAMPLE DATA (First 3 rows)
"""
        for i, row in enumerate(preview["head"][:3], 1):
            prompt += f"**Row {i}:** {row}\n"
        
        prompt += f"""
### 🎯 KEY INSIGHTS
- **Numeric Columns:** {', '.join(schema['numeric_columns'])}
- **Categorical Columns:** {', '.join(schema['categorical_columns'])}
- **Date Columns:** {', '.join(schema['date_columns'])}

### 🏷️ DISTINCT VALUES (for key columns)
"""
        
        for col, values in list(distinct_values.items())[:5]:  # Show first 5 categorical columns
            prompt += f"- **{col}:** {', '.join(map(str, values[:10]))}"
            if len(values) > 10:
                prompt += f" (and {len(values) - 10} more)"
            prompt += "\n"
        
        prompt += f"""
### 💡 ANALYSIS CAPABILITIES
Based on this data structure, you can analyze:
- Brand performance and comparisons
- Regional market dynamics  
- Category trends and profitability
- Time-series patterns (if date columns available)
- Customer behavior and segmentation

**Use this context to inform your analysis and tool selection.**
"""
        
        return prompt
    
    def _ensure_serializable(self, obj):
        """Recursively convert numpy types and other non-serializable types to native Python types."""
        
        if isinstance(obj, dict):
            return {key: self._ensure_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._ensure_serializable(item) for item in obj]
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        elif hasattr(obj, 'tolist'):  # numpy array
            return obj.tolist()
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            # Convert other types to string as fallback
            try:
                # Try to convert to native Python type
                if hasattr(obj, '__float__'):
                    return float(obj)
                elif hasattr(obj, '__int__'):
                    return int(obj)
                else:
                    return str(obj)
            except:
                return str(obj)

# Global instance
_data_context_manager = None

def get_data_context_manager() -> DataContextManager:
    """Get global data context manager instance."""
    global _data_context_manager
    if _data_context_manager is None:
        _data_context_manager = DataContextManager()
    return _data_context_manager