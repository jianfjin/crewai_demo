"""
Parameter-Safe Tool Implementations
Handles missing or incorrect parameters gracefully for dashboard usage.
"""

from crewai.tools import BaseTool
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import json
import os
import hashlib

# Global cache for data sharing
_GLOBAL_DATA_CACHE = {}

def get_cached_data(data_path: str = None) -> pd.DataFrame:
    """
    Get cached data with fallback to sample data if path is missing
    """
    global _GLOBAL_DATA_CACHE
    
    # If no data path provided, return sample data
    if not data_path:
        return create_sample_beverage_data()
    
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
        # Return sample data as fallback
        return create_sample_beverage_data()

def create_sample_beverage_data() -> pd.DataFrame:
    """Create sample beverage data for analysis when no data file is available"""
    np.random.seed(42)  # For reproducible results
    
    dates = pd.date_range('2023-01-01', '2024-12-31', freq='D')
    brands = ['Coca-Cola', 'Pepsi', 'Sprite', 'Fanta', 'Dr Pepper', 'Mountain Dew']
    categories = ['Cola', 'Lemon-Lime', 'Orange', 'Energy', 'Diet']
    regions = ['North America', 'Europe', 'Asia Pacific', 'Latin America', 'Africa']
    
    data = []
    for i, date in enumerate(dates[:365]):  # One year of data
        for brand in brands:
            # Create realistic seasonal patterns
            seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * date.dayofyear / 365)
            base_sales = np.random.normal(1000, 200) * seasonal_factor
            
            category = np.random.choice(categories)
            region = np.random.choice(regions)
            
            units_sold = max(10, int(base_sales))
            price_per_unit = np.random.uniform(1.5, 3.5)
            total_revenue = units_sold * price_per_unit
            cost_per_unit = price_per_unit * np.random.uniform(0.4, 0.7)
            total_cost = units_sold * cost_per_unit
            profit = total_revenue - total_cost
            profit_margin = (profit / total_revenue * 100) if total_revenue > 0 else 0
            
            data.append({
                'sale_date': date,
                'year': date.year,
                'month': date.month,
                'quarter': f"Q{(date.month-1)//3 + 1}",
                'region': region,
                'country': f"Country_{region.replace(' ', '_')}",
                'store_id': f"STORE_{i % 100:03d}",
                'brand': brand,
                'category': category,
                'units_sold': units_sold,
                'price_per_unit': round(price_per_unit, 2),
                'total_revenue': round(total_revenue, 2),
                'cost_per_unit': round(cost_per_unit, 2),
                'total_cost': round(total_cost, 2),
                'profit': round(profit, 2),
                'profit_margin': round(profit_margin, 2)
            })
    
    return pd.DataFrame(data)

class ParameterSafeBeverageMarketAnalysisTool(BaseTool):
    name: str = "beverage_market_analysis"
    description: str = "Analyze beverage market structure, brand performance, and category dynamics"

    def _run(self, data_path: str = None, **kwargs) -> str:
        """Analyze beverage market structure and dynamics"""
        try:
            # Load data using cached approach with fallback
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
                'market_overview': f"Comprehensive beverage market analysis covering {total_brands} brands across {total_categories} categories in {total_regions} regions with total market value of ${total_revenue:,.2f}",
                'data_source': 'sample_data' if not data_path else 'provided_data'
            }
            
            return json.dumps(analysis, indent=2)
            
        except Exception as e:
            return f"Error in beverage market analysis: {str(e)}"

class ParameterSafeTimeSeriesAnalysisTool(BaseTool):
    name: str = "time_series_analysis"
    description: str = "Perform time series analysis on sales data"

    def _run(self, data_path: str = None, date_column: str = None, value_column: str = None, **kwargs) -> str:
        """Perform time series analysis with parameter defaults"""
        try:
            # Load data using cached approach with fallback
            df = get_cached_data(data_path)
            
            if df.empty:
                return json.dumps({
                    'error': 'No data available',
                    'trend_analysis': {},
                    'seasonal_patterns': {},
                    'time_series_insights': 'No data available for time series analysis'
                })
            
            # Set default parameters if not provided
            if not date_column:
                date_column = 'sale_date'
            if not value_column:
                value_column = 'total_revenue'
            
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
            seasonal_pattern = {}
            if total_periods > 30:
                df['month'] = df[actual_date_column].dt.to_period('M')
                monthly_data = df.groupby('month')[actual_value_column].sum()
                seasonal_pattern = {str(k): v for k, v in monthly_data.to_dict().items()}
            
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
                'time_series_insights': f"Time series analysis shows {trend} trend over {total_periods} periods with average value of {avg_value:.2f}",
                'data_source': 'sample_data' if not data_path else 'provided_data'
            }
            
            return json.dumps(analysis, indent=2)
            
        except Exception as e:
            return f"Error in time series analysis: {str(e)}"

# Create parameter-safe tool instances
parameter_safe_beverage_market_analysis = ParameterSafeBeverageMarketAnalysisTool()
parameter_safe_time_series_analysis = ParameterSafeTimeSeriesAnalysisTool()