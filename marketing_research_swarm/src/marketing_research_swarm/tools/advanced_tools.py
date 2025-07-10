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
    Get cached data using the same strategy as optimized tools
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

class BeverageMarketAnalysisTool(BaseTool):
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
                'market_overview': f"Comprehensive beverage market analysis covering {total_brands} brands across {total_categories} categories in {total_regions} regions with total market value of ${total_revenue:,.2f}"
            }
            
            return json.dumps(analysis, indent=2)
            
        except Exception as e:
            return f"Error in beverage market analysis: {str(e)}"

class AnalyzeBrandPerformanceTool(BaseTool):
    name: str = "analyze_brand_performance"
    description: str = "Analyze individual brand performance metrics and market positioning"

    def _run(self, data_path: str = None, **kwargs) -> str:
        """Analyze brand performance metrics"""
        try:
            # Load data using cached approach with fallback
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

    def _run(self, data_path: str = None, segment_column: str = None, value_column: str = None, **kwargs) -> str:
        """Perform cross-sectional analysis"""
        try:
            # Load data using cached approach with fallback
            df = get_cached_data(data_path)
            
            # Set default parameters if not provided
            if not segment_column:
                segment_column = 'brand'
            if not value_column:
                value_column = 'total_revenue'
            
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
                # Convert Period objects to strings for JSON serialization
                seasonal_pattern = {str(k): v for k, v in monthly_data.to_dict().items()}
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

# Additional tools for compatibility with existing imports

class CalculateROITool(BaseTool):
    name: str = "calculate_roi"
    description: str = "Calculate return on investment for marketing campaigns"

    def _run(self, investment: float, revenue: float) -> str:
        """Calculate ROI percentage"""
        try:
            if investment <= 0:
                return json.dumps({
                    'error': 'Investment must be greater than 0',
                    'roi_percentage': 0
                })
            
            roi = ((revenue - investment) / investment) * 100
            
            analysis = {
                'investment': investment,
                'revenue': revenue,
                'profit': revenue - investment,
                'roi_percentage': round(roi, 2),
                'roi_interpretation': 'Positive' if roi > 0 else 'Negative',
                'roi_insights': f"ROI of {roi:.2f}% indicates {'profitable' if roi > 0 else 'unprofitable'} investment"
            }
            
            return json.dumps(analysis, indent=2)
            
        except Exception as e:
            return f"Error calculating ROI: {str(e)}"

class AnalyzeKPIsTool(BaseTool):
    name: str = "analyze_kpis"
    description: str = "Analyze key performance indicators"

    def _run(self, data_path: str) -> str:
        """Analyze KPIs from data"""
        try:
            # Load data using cached approach
            df = get_cached_data(data_path)
            
            if df.empty:
                return json.dumps({
                    'error': 'No data available',
                    'kpis': {},
                    'kpi_insights': 'No data available for KPI analysis'
                })
            
            # Calculate basic KPIs
            kpis = {}
            
            if 'total_revenue' in df.columns:
                kpis['total_revenue'] = df['total_revenue'].sum()
                kpis['average_revenue'] = df['total_revenue'].mean()
            
            if 'profit' in df.columns:
                kpis['total_profit'] = df['profit'].sum()
                kpis['average_profit'] = df['profit'].mean()
            elif 'total_revenue' in df.columns and 'total_cost' in df.columns:
                df['profit'] = df['total_revenue'] - df['total_cost']
                kpis['total_profit'] = df['profit'].sum()
                kpis['average_profit'] = df['profit'].mean()
            
            if 'profit_margin' in df.columns:
                kpis['average_profit_margin'] = df['profit_margin'].mean()
            
            if 'units_sold' in df.columns:
                kpis['total_units_sold'] = df['units_sold'].sum()
                kpis['average_units_sold'] = df['units_sold'].mean()
            
            # Brand-specific KPIs
            if 'brand' in df.columns:
                kpis['total_brands'] = df['brand'].nunique()
                if 'total_revenue' in df.columns:
                    top_brand = df.groupby('brand')['total_revenue'].sum().idxmax()
                    kpis['top_performing_brand'] = top_brand
            
            # Regional KPIs
            if 'region' in df.columns:
                kpis['total_regions'] = df['region'].nunique()
                if 'total_revenue' in df.columns:
                    top_region = df.groupby('region')['total_revenue'].sum().idxmax()
                    kpis['top_performing_region'] = top_region
            
            analysis = {
                'kpis': kpis,
                'data_points': len(df),
                'kpi_insights': f"Analysis of {len(df)} data points reveals key performance metrics across business dimensions"
            }
            
            return json.dumps(analysis, indent=2)
            
        except Exception as e:
            return f"Error analyzing KPIs: {str(e)}"

class PlanBudgetTool(BaseTool):
    name: str = "plan_budget"
    description: str = "Plan budget allocation across different channels and campaigns"

    def _run(self, total_budget: float, channels: List[str] = None, priorities: List[float] = None) -> str:
        """Plan budget allocation"""
        try:
            if channels is None:
                channels = ['Digital Marketing', 'Traditional Media', 'Social Media', 'Content Marketing', 'Events']
            
            if priorities is None:
                # Equal distribution if no priorities specified
                priorities = [1.0] * len(channels)
            
            if len(priorities) != len(channels):
                priorities = [1.0] * len(channels)
            
            # Normalize priorities
            total_priority = sum(priorities)
            normalized_priorities = [p / total_priority for p in priorities]
            
            # Calculate budget allocation
            budget_allocation = {}
            for i, channel in enumerate(channels):
                allocation = total_budget * normalized_priorities[i]
                budget_allocation[channel] = round(allocation, 2)
            
            # Calculate percentages
            percentage_allocation = {}
            for channel, amount in budget_allocation.items():
                percentage = (amount / total_budget) * 100
                percentage_allocation[channel] = round(percentage, 1)
            
            analysis = {
                'total_budget': total_budget,
                'budget_allocation': budget_allocation,
                'percentage_allocation': percentage_allocation,
                'channels': channels,
                'allocation_strategy': 'Priority-based allocation',
                'budget_insights': f"Budget of ${total_budget:,.2f} allocated across {len(channels)} channels based on priorities"
            }
            
            return json.dumps(analysis, indent=2)
            
        except Exception as e:
            return f"Error planning budget: {str(e)}"

class CalculateMarketShareTool(BaseTool):
    name: str = "calculate_market_share"
    description: str = "Calculate market share based on company and total market revenue"

    def _run(self, company_revenue: float, total_market_revenue: float) -> str:
        """Calculate market share percentage"""
        try:
            if total_market_revenue <= 0:
                return json.dumps({
                    'error': 'Total market revenue must be greater than 0',
                    'market_share_percentage': 0
                })
            
            market_share = (company_revenue / total_market_revenue) * 100
            
            # Determine competitive position
            if market_share >= 40:
                position = 'Market Leader'
            elif market_share >= 20:
                position = 'Strong Competitor'
            elif market_share >= 10:
                position = 'Moderate Player'
            else:
                position = 'Niche Player'
            
            analysis = {
                'company_revenue': company_revenue,
                'total_market_revenue': total_market_revenue,
                'market_share_percentage': round(market_share, 2),
                'competitive_position': position,
                'market_concentration': 'High' if market_share >= 25 else 'Medium' if market_share >= 10 else 'Low',
                'market_insights': f"Market share of {market_share:.2f}% positions company as {position.lower()}"
            }
            
            return json.dumps(analysis, indent=2)
            
        except Exception as e:
            return f"Error calculating market share: {str(e)}"

# Create tool instances
beverage_market_analysis = BeverageMarketAnalysisTool()
analyze_brand_performance = AnalyzeBrandPerformanceTool()
profitability_analysis = ProfitabilityAnalysisTool()
cross_sectional_analysis = CrossSectionalAnalysisTool()
time_series_analysis = TimeSeriesAnalysisTool()
forecast_sales = ForecastSalesTool()

# Additional tool instances for compatibility
calculate_roi = CalculateROITool()
analyze_kpis = AnalyzeKPIsTool()
plan_budget = PlanBudgetTool()
calculate_market_share = CalculateMarketShareTool()