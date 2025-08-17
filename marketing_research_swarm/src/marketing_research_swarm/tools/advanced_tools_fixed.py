from crewai.tools import BaseTool
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import json
import os
import hashlib

# Global cache for data sharing
_GLOBAL_DATA_CACHE = {}

def get_cached_data(data_path: str) -> pd.DataFrame:
    """
    Get cached data using the same strategy as optimized tools
    """
    global _GLOBAL_DATA_CACHE
    
    # Generate cache key
    cache_key = hashlib.md5(data_path.encode()).hexdigest()
    
    # Check if data is already cached
    if cache_key in _GLOBAL_DATA_CACHE:
        return _GLOBAL_DATA_CACHE[cache_key].copy()
    
    # Load and cache data
    try:
        if data_path.endswith('.csv'):
            df = pd.read_csv(data_path)
        elif data_path.endswith('.json'):
            df = pd.read_json(data_path)
        else:
            # Try CSV first, then JSON
            try:
                df = pd.read_csv(data_path)
            except:
                df = pd.read_json(data_path)
        
        # Cache the data
        _GLOBAL_DATA_CACHE[cache_key] = df.copy()
        return df
        
    except Exception as e:
        print(f"Error loading data from {data_path}: {e}")
        # Return empty DataFrame with expected columns
        return pd.DataFrame({
            'sale_date': [],
            'year': [],
            'month': [],
            'quarter': [],
            'region': [],
            'country': [],
            'store_id': [],
            'brand': [],
            'category': [],
            'units_sold': [],
            'price_per_unit': [],
            'total_revenue': [],
            'cost_per_unit': [],
            'total_cost': [],
            'profit': [],
            'profit_margin': []
        })

class BeverageMarketAnalysisTool(BaseTool):
    name: str = "beverage_market_analysis"
    description: str = "Analyze beverage market structure, brand performance, and category dynamics"

    def _run(self, data_path: str) -> str:
        """Analyze beverage market structure and dynamics"""
        try:
            # Load data using cached approach
            df = get_cached_data(data_path)
            
            if df.empty:
                return json.dumps({
                    'error': 'No data available',
                    'total_brands': 0,
                    'total_categories': 0,
                    'total_regions': 0,
                    'market_overview': 'No market data available for analysis'
                })
            
            # Basic market analysis
            total_brands = df['brand'].nunique() if 'brand' in df.columns else 0
            total_categories = df['category'].nunique() if 'category' in df.columns else 0
            total_regions = df['region'].nunique() if 'region' in df.columns else 0
            total_revenue = df['total_revenue'].sum() if 'total_revenue' in df.columns else 0
            
            # Top performing brands
            if 'brand' in df.columns and 'total_revenue' in df.columns:
                brand_performance = df.groupby('brand')['total_revenue'].sum().sort_values(ascending=False)
                top_brands = brand_performance.head(5).to_dict()
            else:
                top_brands = {}
            
            # Category analysis
            if 'category' in df.columns and 'total_revenue' in df.columns:
                category_performance = df.groupby('category')['total_revenue'].sum().sort_values(ascending=False)
                top_categories = category_performance.head(5).to_dict()
            else:
                top_categories = {}
            
            # Regional analysis
            if 'region' in df.columns and 'total_revenue' in df.columns:
                regional_performance = df.groupby('region')['total_revenue'].sum().sort_values(ascending=False)
                top_regions = regional_performance.head(5).to_dict()
            else:
                top_regions = {}
            
            analysis = {
                'total_brands': total_brands,
                'total_categories': total_categories,
                'total_regions': total_regions,
                'total_market_value': total_revenue,
                'top_brands': top_brands,
                'top_categories': top_categories,
                'top_regions': top_regions,
                'regions_covered': list(df['region'].unique()) if 'region' in df.columns else [],
                'market_overview': f"Comprehensive beverage market analysis covering {total_brands} brands across {total_categories} categories in {total_regions} regions with total market value of ${total_revenue:,.2f}"
            }
            
            return json.dumps(analysis, indent=2)
            
        except Exception as e:
            return f"Error in beverage market analysis: {str(e)}"

class AnalyzeBrandPerformanceTool(BaseTool):
    name: str = "analyze_brand_performance"
    description: str = "Analyze individual brand performance metrics and market positioning"

    def _run(self, data_path: str) -> str:
        """Analyze brand performance metrics"""
        try:
            # Load data using cached approach
            df = get_cached_data(data_path)
            
            if df.empty:
                return json.dumps({
                    'error': 'No data available',
                    'top_brands': [],
                    'performance_metrics': {},
                    'brand_insights': 'No brand data available for analysis'
                })
            
            # Check required columns
            required_cols = ['brand', 'total_revenue']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                return json.dumps({
                    'error': f'Missing required columns: {missing_cols}',
                    'available_columns': list(df.columns),
                    'brand_insights': 'Cannot perform brand analysis due to missing data columns'
                })
            
            # Brand performance analysis
            agg_dict = {'total_revenue': ['sum', 'mean']}
            
            # Add optional columns if they exist
            if 'profit' in df.columns:
                agg_dict['profit'] = ['sum', 'mean']
            if 'profit_margin' in df.columns:
                agg_dict['profit_margin'] = 'mean'
            if 'units_sold' in df.columns:
                agg_dict['units_sold'] = 'sum'
            
            brand_metrics = df.groupby('brand').agg(agg_dict).round(2)
            
            # Flatten column names
            brand_metrics.columns = ['_'.join(col).strip() for col in brand_metrics.columns]
            
            # Top performing brands
            top_brands = brand_metrics.sort_values('total_revenue_sum', ascending=False).head(10)
            
            # Performance insights
            total_brands = len(brand_metrics)
            avg_revenue_per_brand = brand_metrics['total_revenue_sum'].mean()
            top_performer = brand_metrics['total_revenue_sum'].idxmax()
            
            # Build performance metrics
            performance_metrics = {
                'highest_revenue': brand_metrics['total_revenue_sum'].max(),
                'lowest_revenue': brand_metrics['total_revenue_sum'].min(),
                'average_revenue': avg_revenue_per_brand
            }
            
            # Add profit margin metrics if available
            if 'profit_margin_mean' in brand_metrics.columns:
                performance_metrics.update({
                    'highest_profit_margin': brand_metrics['profit_margin_mean'].max(),
                    'average_profit_margin': brand_metrics['profit_margin_mean'].mean()
                })
            
            analysis = {
                'total_brands_analyzed': total_brands,
                'average_revenue_per_brand': avg_revenue_per_brand,
                'top_performing_brand': top_performer,
                'top_brands': list(top_brands.index[:5]),  # Just brand names
                'performance_metrics': performance_metrics,
                'growth_analysis': {
                    'total_revenue': brand_metrics['total_revenue_sum'].sum(),
                    'brand_count': total_brands,
                    'market_concentration': 'High' if top_brands.iloc[0]['total_revenue_sum'] > avg_revenue_per_brand * 3 else 'Medium'
                },
                'brand_insights': f"Analysis of {total_brands} brands shows {top_performer} as the top performer with average revenue per brand of ${avg_revenue_per_brand:,.2f}"
            }
            
            return json.dumps(analysis, indent=2)
            
        except Exception as e:
            return f"Error in brand performance analysis: {str(e)}"

class ProfitabilityAnalysisTool(BaseTool):
    name: str = "profitability_analysis"
    description: str = "Analyze profitability metrics across different dimensions"

    def _run(self, data_path: str, analysis_dimension: str = "brand") -> str:
        """Analyze profitability by specified dimension"""
        try:
            # Load data using cached approach
            df = get_cached_data(data_path)
            
            if df.empty:
                return json.dumps({
                    'error': 'No data available',
                    'total_revenue': 0,
                    'total_cost': 0,
                    'profit_margin': 0,
                    'roi': 0,
                    'profitability_insights': 'No data available for profitability analysis'
                })
            
            # Map analysis dimensions to actual column names
            dimension_mapping = {
                'brand': 'brand',
                'category': 'category', 
                'region': 'region',
                'brand_performance': 'brand',  # Handle incorrect dimension name
                'overall': None,  # Overall analysis
                'Cola': 'category'  # Handle specific category
            }
            
            dimension_col = dimension_mapping.get(analysis_dimension, analysis_dimension)
            
            # Check required columns
            required_cols = ['total_revenue']
            if 'total_cost' not in df.columns:
                # Calculate total_cost if not available
                if 'cost_per_unit' in df.columns and 'units_sold' in df.columns:
                    df['total_cost'] = df['cost_per_unit'] * df['units_sold']
                else:
                    # Estimate cost as 60% of revenue
                    df['total_cost'] = df['total_revenue'] * 0.6
            
            if 'profit' not in df.columns:
                df['profit'] = df['total_revenue'] - df['total_cost']
            
            if 'profit_margin' not in df.columns:
                df['profit_margin'] = (df['profit'] / df['total_revenue'] * 100).fillna(0)
            
            # Perform analysis
            if dimension_col and dimension_col in df.columns:
                # Dimension-based analysis
                profitability = df.groupby(dimension_col).agg({
                    'total_revenue': 'sum',
                    'total_cost': 'sum',
                    'profit': 'sum',
                    'profit_margin': 'mean'
                }).round(2)
                
                # Calculate ROI
                profitability['roi'] = (profitability['profit'] / profitability['total_cost'] * 100).fillna(0)
                
                # Get top performers
                top_performers = profitability.sort_values('profit', ascending=False).head(5)
                
                analysis = {
                    'analysis_dimension': analysis_dimension,
                    'total_revenue': profitability['total_revenue'].sum(),
                    'total_cost': profitability['total_cost'].sum(),
                    'total_profit': profitability['profit'].sum(),
                    'average_profit_margin': profitability['profit_margin'].mean(),
                    'average_roi': profitability['roi'].mean(),
                    'top_performers': top_performers.to_dict(),
                    'profitability_insights': f"Profitability analysis by {analysis_dimension} shows average profit margin of {profitability['profit_margin'].mean():.2f}% and ROI of {profitability['roi'].mean():.2f}%"
                }
            else:
                # Overall analysis
                total_revenue = df['total_revenue'].sum()
                total_cost = df['total_cost'].sum()
                total_profit = df['profit'].sum()
                avg_margin = df['profit_margin'].mean()
                roi = (total_profit / total_cost * 100) if total_cost > 0 else 0
                
                analysis = {
                    'analysis_dimension': 'overall',
                    'total_revenue': total_revenue,
                    'total_cost': total_cost,
                    'total_profit': total_profit,
                    'profit_margin': avg_margin,
                    'roi': roi,
                    'profitability_insights': f"Overall profitability analysis shows profit margin of {avg_margin:.2f}% and ROI of {roi:.2f}%"
                }
            
            return json.dumps(analysis, indent=2)
            
        except Exception as e:
            return f"Error in profitability analysis: {str(e)}"

class CrossSectionalAnalysisTool(BaseTool):
    name: str = "cross_sectional_analysis"
    description: str = "Perform cross-sectional analysis across different segments"

    def _run(self, data_path: str, segment_column: str, value_column: str) -> str:
        """Perform cross-sectional analysis"""
        try:
            # Load data using cached approach
            df = get_cached_data(data_path)
            
            if df.empty:
                return json.dumps({
                    'error': 'No data available',
                    'segment_performance': {},
                    'top_performers': [],
                    'performance_gaps': [],
                    'comparative_insights': 'No data available for cross-sectional analysis'
                })
            
            # Map common column name variations
            column_mapping = {
                'Cola_sales': 'total_revenue',
                'sales': 'total_revenue',
                'total_sales': 'total_revenue',
                'revenue': 'total_revenue'
            }
            
            # Use mapped column name if available
            actual_value_column = column_mapping.get(value_column, value_column)
            
            # Check if columns exist
            if segment_column not in df.columns:
                available_segments = [col for col in df.columns if col in ['brand', 'category', 'region', 'country']]
                if available_segments:
                    segment_column = available_segments[0]
                else:
                    return json.dumps({
                        'error': f'Segment column {segment_column} not found',
                        'available_columns': list(df.columns),
                        'comparative_insights': 'Cannot perform cross-sectional analysis due to missing segment column'
                    })
            
            if actual_value_column not in df.columns:
                # Try to find a suitable value column
                value_columns = [col for col in df.columns if col in ['total_revenue', 'profit', 'units_sold']]
                if value_columns:
                    actual_value_column = value_columns[0]
                else:
                    return json.dumps({
                        'error': f'Value column {value_column} not found',
                        'available_columns': list(df.columns),
                        'comparative_insights': 'Cannot perform cross-sectional analysis due to missing value column'
                    })
            
            # Perform cross-sectional analysis
            segment_performance = df.groupby(segment_column)[actual_value_column].agg([
                'sum', 'mean', 'count', 'std'
            ]).round(2)
            
            # Calculate additional metrics
            segment_performance['coefficient_of_variation'] = (
                segment_performance['std'] / segment_performance['mean'] * 100
            ).fillna(0)
            
            # Identify top and bottom performers
            top_performers = segment_performance.sort_values('sum', ascending=False).head(5)
            bottom_performers = segment_performance.sort_values('sum', ascending=True).head(3)
            
            # Calculate performance gaps
            max_performance = segment_performance['sum'].max()
            min_performance = segment_performance['sum'].min()
            performance_gap = max_performance - min_performance
            
            # Generate insights
            total_segments = len(segment_performance)
            avg_performance = segment_performance['sum'].mean()
            
            analysis = {
                'segment_column': segment_column,
                'value_column': actual_value_column,
                'total_segments': total_segments,
                'segment_performance': segment_performance.to_dict(),
                'top_performers': list(top_performers.index),
                'bottom_performers': list(bottom_performers.index),
                'performance_gaps': {
                    'max_performance': max_performance,
                    'min_performance': min_performance,
                    'performance_gap': performance_gap,
                    'gap_percentage': (performance_gap / max_performance * 100) if max_performance > 0 else 0
                },
                'statistical_summary': {
                    'mean': segment_performance['sum'].mean(),
                    'median': segment_performance['sum'].median(),
                    'std_deviation': segment_performance['sum'].std()
                },
                'comparative_insights': f"Cross-sectional analysis of {total_segments} {segment_column} segments shows {top_performers.index[0]} as top performer with significant performance gaps across segments"
            }
            
            return json.dumps(analysis, indent=2)
            
        except Exception as e:
            return f"Error in cross-sectional analysis: {str(e)}"

class TimeSeriesAnalysisTool(BaseTool):
    name: str = "time_series_analysis"
    description: str = "Perform time series analysis on sales data"

    def _run(self, data_path: str, date_column: str, value_column: str) -> str:
        """Perform time series analysis"""
        try:
            # Load data using cached approach
            df = get_cached_data(data_path)
            
            if df.empty:
                return json.dumps({
                    'error': 'No data available',
                    'trend_analysis': {},
                    'seasonal_patterns': {},
                    'time_series_insights': 'No data available for time series analysis'
                })
            
            # Map common column name variations
            date_mapping = {
                'date': 'sale_date',
                'sales_date': 'sale_date'
            }
            
            value_mapping = {
                'sales': 'total_revenue',
                'Cola_sales': 'total_revenue',
                'total_sales': 'total_revenue',
                'revenue': 'total_revenue'
            }
            
            # Use mapped column names if available
            actual_date_column = date_mapping.get(date_column, date_column)
            actual_value_column = value_mapping.get(value_column, value_column)
            
            # Check if columns exist
            if actual_date_column not in df.columns:
                date_columns = [col for col in df.columns if 'date' in col.lower()]
                if date_columns:
                    actual_date_column = date_columns[0]
                else:
                    return json.dumps({
                        'error': f'Date column {date_column} not found',
                        'available_columns': list(df.columns),
                        'time_series_insights': 'Cannot perform time series analysis due to missing date column'
                    })
            
            if actual_value_column not in df.columns:
                value_columns = [col for col in df.columns if col in ['total_revenue', 'profit', 'units_sold']]
                if value_columns:
                    actual_value_column = value_columns[0]
                else:
                    return json.dumps({
                        'error': f'Value column {value_column} not found',
                        'available_columns': list(df.columns),
                        'time_series_insights': 'Cannot perform time series analysis due to missing value column'
                    })
            
            # Convert date column to datetime
            df[actual_date_column] = pd.to_datetime(df[actual_date_column])
            
            # Sort by date
            df = df.sort_values(actual_date_column)
            
            # Aggregate by date
            daily_data = df.groupby(actual_date_column)[actual_value_column].sum().reset_index()
            
            # Calculate basic statistics
            total_periods = len(daily_data)
            avg_value = daily_data[actual_value_column].mean()
            trend = 'increasing' if daily_data[actual_value_column].iloc[-1] > daily_data[actual_value_column].iloc[0] else 'decreasing'
            
            # Monthly aggregation if we have enough data
            if total_periods > 30:
                df['month'] = df[actual_date_column].dt.to_period('M')
                monthly_data = df.groupby('month')[actual_value_column].sum()
                seasonal_pattern = monthly_data.to_dict()
            else:
                seasonal_pattern = {}
            
            analysis = {
                'date_column': actual_date_column,
                'value_column': actual_value_column,
                'total_periods': total_periods,
                'date_range': {
                    'start_date': daily_data[actual_date_column].min().isoformat(),
                    'end_date': daily_data[actual_date_column].max().isoformat()
                },
                'trend_analysis': {
                    'overall_trend': trend,
                    'average_value': avg_value,
                    'total_value': daily_data[actual_value_column].sum(),
                    'volatility': daily_data[actual_value_column].std()
                },
                'seasonal_patterns': seasonal_pattern,
                'statistical_summary': {
                    'mean': daily_data[actual_value_column].mean(),
                    'median': daily_data[actual_value_column].median(),
                    'std_deviation': daily_data[actual_value_column].std(),
                    'min_value': daily_data[actual_value_column].min(),
                    'max_value': daily_data[actual_value_column].max()
                },
                'time_series_insights': f"Time series analysis shows {trend} trend over {total_periods} periods with average value of {avg_value:.2f}"
            }
            
            return json.dumps(analysis, indent=2)
            
        except Exception as e:
            return f"Error in time series analysis: {str(e)}"

class ForecastSalesTool(BaseTool):
    name: str = "forecast_sales"
    description: str = "Forecast future sales based on historical data"

    def _run(self, data_path: str, periods: int = 30, forecast_column: str = "sales") -> str:
        """Forecast sales for specified periods"""
        try:
            # Load data using cached approach
            df = get_cached_data(data_path)
            
            if df.empty:
                return json.dumps({
                    'error': 'No data available',
                    'forecast_periods': periods,
                    'forecast_values': [],
                    'forecast_insights': 'No data available for sales forecasting'
                })
            
            # Map common column name variations
            column_mapping = {
                'sales': 'total_revenue',
                'Cola_sales': 'total_revenue',
                'total_sales': 'total_revenue',
                'revenue': 'total_revenue'
            }
            
            # Use mapped column name if available
            actual_forecast_column = column_mapping.get(forecast_column, forecast_column)
            
            # Check if column exists
            if actual_forecast_column not in df.columns:
                value_columns = [col for col in df.columns if col in ['total_revenue', 'profit', 'units_sold']]
                if value_columns:
                    actual_forecast_column = value_columns[0]
                else:
                    return json.dumps({
                        'error': f'Forecast column {forecast_column} not found',
                        'available_columns': list(df.columns),
                        'forecast_insights': 'Cannot perform forecasting due to missing forecast column'
                    })
            
            # Simple forecasting based on historical average and trend
            historical_values = df[actual_forecast_column].dropna()
            
            if len(historical_values) < 2:
                return json.dumps({
                    'error': 'Insufficient historical data for forecasting',
                    'forecast_insights': 'Need at least 2 data points for forecasting'
                })
            
            # Calculate basic statistics
            mean_value = historical_values.mean()
            trend = (historical_values.iloc[-1] - historical_values.iloc[0]) / len(historical_values)
            
            # Generate simple linear forecast
            forecast_values = []
            for i in range(1, periods + 1):
                forecast_value = mean_value + (trend * i)
                forecast_values.append(max(0, forecast_value))  # Ensure non-negative
            
            # Calculate confidence metrics
            historical_std = historical_values.std()
            confidence_interval = 1.96 * historical_std  # 95% confidence interval
            
            analysis = {
                'forecast_column': actual_forecast_column,
                'forecast_periods': periods,
                'historical_data_points': len(historical_values),
                'forecast_values': forecast_values,
                'forecast_summary': {
                    'average_forecast': np.mean(forecast_values),
                    'total_forecast': sum(forecast_values),
                    'trend_direction': 'increasing' if trend > 0 else 'decreasing',
                    'confidence_interval': confidence_interval
                },
                'historical_baseline': {
                    'historical_mean': mean_value,
                    'historical_trend': trend,
                    'historical_std': historical_std
                },
                'forecast_insights': f"Forecasted {periods} periods with average value of {np.mean(forecast_values):.2f} based on historical trend of {trend:.2f} per period"
            }
            
            return json.dumps(analysis, indent=2)
            
        except Exception as e:
            return f"Error forecasting sales: {str(e)}"

# Create tool instances
beverage_market_analysis = BeverageMarketAnalysisTool()
analyze_brand_performance = AnalyzeBrandPerformanceTool()
profitability_analysis = ProfitabilityAnalysisTool()
cross_sectional_analysis = CrossSectionalAnalysisTool()
time_series_analysis = TimeSeriesAnalysisTool()
forecast_sales = ForecastSalesTool()

def get_tools():
    """
    Get all available tools for the marketing research workflow.
    
    Returns:
        List of tool instances that can be used by agents
    """
    return [
        beverage_market_analysis,
        analyze_brand_performance,
        profitability_analysis,
        cross_sectional_analysis,
        time_series_analysis,
        forecast_sales
    ]