"""
Cached and Optimized Tools for Token Reduction
Implements caching, result sharing, and structured outputs.
"""

from crewai.tools import BaseTool
import pandas as pd
import numpy as np
from typing import Dict, Any, Union, Optional
from pydantic import BaseModel
import hashlib
import json
from .optimized_data_processor import data_processor, result_cache

class StructuredAnalysisResult(BaseModel):
    """Structured output format to reduce verbose text generation."""
    analysis_type: str
    key_metrics: Dict[str, float]
    top_performers: Dict[str, Dict[str, float]]
    insights: list[str]
    recommendations: list[str]
    data_summary: Dict[str, Any]

class CachedProfitabilityTool(BaseTool):
    name: str = "Cached Profitability Analysis"
    description: str = "Analyzes profitability with caching and structured output"

    def _run(self, data_file_path: str, analysis_dimension: str = 'brand') -> StructuredAnalysisResult:
        """Run profitability analysis with caching and optimized data processing."""
        
        # Check cache first
        cache_inputs = {"data_file_path": data_file_path, "dimension": analysis_dimension}
        cached_result = result_cache.get_cached_result("profitability_analysis", cache_inputs)
        if cached_result:
            return cached_result
        
        # Load and process data efficiently
        try:
            df = pd.read_csv(data_file_path)
            
            # Use optimized data processing
            data_summary = data_processor.summarize_dataset(df, "profitability")
            chunked_data = data_processor.chunk_data_by_dimension(df, analysis_dimension, "profitability_analyst")
            
            # Perform analysis on summarized data
            grouped = df.groupby(analysis_dimension).agg({
                'total_revenue': 'sum',
                'total_cost': 'sum', 
                'profit': 'sum',
                'profit_margin': 'mean'
            }).round(2)
            
            # Extract key metrics
            key_metrics = {
                "total_revenue": float(df['total_revenue'].sum()),
                "total_cost": float(df['total_cost'].sum()),
                "total_profit": float(df['profit'].sum()),
                "avg_margin": float(grouped['profit_margin'].mean()),
                "entities_analyzed": len(grouped)
            }
            
            # Get top performers (limit to top 3 to reduce tokens)
            top_3 = grouped.nlargest(3, 'profit')
            top_performers = {}
            for idx, row in top_3.iterrows():
                top_performers[str(idx)] = {
                    "revenue": float(row['total_revenue']),
                    "profit": float(row['profit']),
                    "margin": float(row['profit_margin'])
                }
            
            # Generate concise insights
            best_performer = grouped['profit'].idxmax()
            worst_performer = grouped['profit'].idxmin()
            
            insights = [
                f"Top performer: {best_performer} with {grouped.loc[best_performer, 'profit']:.0f} profit",
                f"Margin range: {grouped['profit_margin'].min():.1f}% - {grouped['profit_margin'].max():.1f}%",
                f"Revenue concentration: Top 3 represent {(top_3['total_revenue'].sum()/key_metrics['total_revenue']*100):.1f}% of total"
            ]
            
            # Generate actionable recommendations
            recommendations = [
                f"Focus marketing on {best_performer} (highest profit)",
                f"Optimize {worst_performer} operations (lowest profit)",
                "Analyze top performer strategies for replication"
            ]
            
            result = StructuredAnalysisResult(
                analysis_type=f"profitability_by_{analysis_dimension}",
                key_metrics=key_metrics,
                top_performers=top_performers,
                insights=insights,
                recommendations=recommendations,
                data_summary={"rows_analyzed": len(df), "dimension": analysis_dimension}
            )
            
            # Cache the result
            result_cache.cache_result("profitability_analysis", cache_inputs, result)
            return result
            
        except Exception as e:
            return StructuredAnalysisResult(
                analysis_type="error",
                key_metrics={},
                top_performers={},
                insights=[f"Analysis failed: {str(e)}"],
                recommendations=["Check data format and try again"],
                data_summary={}
            )

class CachedROITool(BaseTool):
    name: str = "Cached ROI Calculator"
    description: str = "Calculates ROI with caching and structured output"

    def _run(self, revenue: float = None, cost: float = None, data_file_path: str = None) -> Dict[str, Any]:
        """Calculate ROI with caching."""
        
        cache_inputs = {"revenue": revenue, "cost": cost, "data_file_path": data_file_path}
        cached_result = result_cache.get_cached_result("roi_calculation", cache_inputs)
        if cached_result:
            return cached_result
        
        try:
            # If data file provided, extract revenue/cost from summary
            if data_file_path and (revenue is None or cost is None):
                df = pd.read_csv(data_file_path)
                summary = data_processor.summarize_dataset(df)
                revenue = revenue or summary["key_metrics"].get("total_revenue", 0)
                cost = cost or summary["key_metrics"].get("total_cost", 0)
            
            roi_percentage = ((revenue - cost) / cost) * 100 if cost > 0 else 0
            net_profit = revenue - cost
            
            result = {
                "roi_percentage": round(roi_percentage, 2),
                "net_profit": round(net_profit, 2),
                "revenue": round(revenue, 2),
                "cost": round(cost, 2),
                "category": "profitable" if roi_percentage > 0 else "needs_optimization"
            }
            
            result_cache.cache_result("roi_calculation", cache_inputs, result)
            return result
            
        except Exception as e:
            return {"error": str(e), "roi_percentage": 0}

class CachedForecastTool(BaseTool):
    name: str = "Cached Sales Forecast"
    description: str = "Forecasts sales with caching and structured output"

    def _run(self, data_file_path: str, periods: int = 30, target_column: str = "total_revenue") -> Dict[str, Any]:
        """Generate sales forecast with caching."""
        
        cache_inputs = {"data_file_path": data_file_path, "periods": periods, "target_column": target_column}
        cached_result = result_cache.get_cached_result("sales_forecast", cache_inputs)
        if cached_result:
            return cached_result
        
        try:
            df = pd.read_csv(data_file_path)
            
            # Use data summary instead of full dataset for forecasting
            summary = data_processor.summarize_dataset(df)
            
            # Simple trend-based forecast (optimized for token efficiency)
            if target_column in df.columns:
                recent_avg = df[target_column].tail(10).mean()
                growth_rate = 0.02  # Assume 2% growth (simplified)
                
                forecast_values = []
                for i in range(periods):
                    forecast_value = recent_avg * (1 + growth_rate) ** i
                    forecast_values.append(round(forecast_value, 2))
                
                result = {
                    "forecast_periods": periods,
                    "target_column": target_column,
                    "baseline_value": round(recent_avg, 2),
                    "total_forecast": round(sum(forecast_values), 2),
                    "avg_forecast": round(sum(forecast_values) / periods, 2),
                    "growth_assumption": f"{growth_rate*100}% per period",
                    "confidence": "medium"
                }
            else:
                result = {"error": f"Column {target_column} not found"}
            
            result_cache.cache_result("sales_forecast", cache_inputs, result)
            return result
            
        except Exception as e:
            return {"error": str(e), "forecast_periods": 0}

class SharedResultsTool(BaseTool):
    name: str = "Shared Results Manager"
    description: str = "Manages shared results between agents to avoid redundant processing"

    def _run(self, action: str, key: str = None, data: Any = None) -> Any:
        """Manage shared results between agents."""
        
        if action == "store":
            if key and data:
                result_cache.cache_result("shared_results", {"key": key}, data)
                return {"status": "stored", "key": key}
        
        elif action == "retrieve":
            if key:
                result = result_cache.get_cached_result("shared_results", {"key": key})
                return result if result else {"status": "not_found", "key": key}
        
        elif action == "list":
            # Return available shared results (simplified)
            return {"available_keys": ["market_summary", "profitability_data", "regional_analysis"]}
        
        return {"error": "Invalid action or missing parameters"}

# Create optimized tool instances
cached_profitability_analysis = CachedProfitabilityTool()
cached_roi_calculator = CachedROITool()
cached_sales_forecast = CachedForecastTool()
shared_results_manager = SharedResultsTool()