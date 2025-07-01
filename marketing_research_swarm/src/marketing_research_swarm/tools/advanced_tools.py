from crewai.tools import BaseTool
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings
warnings.filterwarnings('ignore')

class CalculateROITool(BaseTool):
    name: str = "Calculate ROI"
    description: str = "Calculates Return on Investment (ROI) for marketing campaigns using revenue and cost data."

    def _run(self, revenue: float, cost: float, additional_costs: float = 0) -> str:
        try:
            total_cost = cost + additional_costs
            roi = ((revenue - total_cost) / total_cost) * 100
            
            result = f"""
ROI Analysis Results:
- Revenue: ${revenue:,.2f}
- Total Cost: ${total_cost:,.2f}
- ROI: {roi:.2f}%
- Net Profit: ${revenue - total_cost:,.2f}

ROI Interpretation:
- ROI > 0%: Profitable campaign
- ROI > 100%: Highly successful campaign
- ROI < 0%: Campaign needs optimization
"""
            return result
        except Exception as e:
            return f"Error calculating ROI: {str(e)}"

class AnalyzeKPIsTool(BaseTool):
    name: str = "Analyze KPIs"
    description: str = "Analyzes key performance indicators for marketing campaigns including conversion rates, customer acquisition cost, and lifetime value."

    def _run(self, data_path: str = None, **kwargs) -> str:
        try:
            if data_path:
                df = pd.read_csv(data_path)
            else:
                # Use provided metrics
                metrics = kwargs
            
            # Calculate common KPIs
            results = "KPI Analysis Results:\n\n"
            
            if 'clicks' in kwargs and 'impressions' in kwargs:
                ctr = (kwargs['clicks'] / kwargs['impressions']) * 100
                results += f"Click-Through Rate (CTR): {ctr:.2f}%\n"
            
            if 'conversions' in kwargs and 'clicks' in kwargs:
                conversion_rate = (kwargs['conversions'] / kwargs['clicks']) * 100
                results += f"Conversion Rate: {conversion_rate:.2f}%\n"
            
            if 'cost' in kwargs and 'conversions' in kwargs:
                cac = kwargs['cost'] / kwargs['conversions']
                results += f"Customer Acquisition Cost (CAC): ${cac:.2f}\n"
            
            if 'revenue' in kwargs and 'conversions' in kwargs:
                avg_order_value = kwargs['revenue'] / kwargs['conversions']
                results += f"Average Order Value: ${avg_order_value:.2f}\n"
            
            return results
        except Exception as e:
            return f"Error analyzing KPIs: {str(e)}"

class ForecastSalesTool(BaseTool):
    name: str = "Forecast Sales"
    description: str = "Performs sales forecasting using time series analysis and machine learning techniques."

    def _run(self, data_path: str, periods: int = 30) -> str:
        try:
            df = pd.read_csv(data_path)
            df['Date'] = pd.to_datetime(df['Date'])
            
            # Aggregate sales by date
            daily_sales = df.groupby('Date')['Sales'].sum().reset_index()
            daily_sales = daily_sales.sort_values('Date')
            
            # Simple linear regression forecast
            X = np.arange(len(daily_sales)).reshape(-1, 1)
            y = daily_sales['Sales'].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Forecast future periods
            future_X = np.arange(len(daily_sales), len(daily_sales) + periods).reshape(-1, 1)
            forecast = model.predict(future_X)
            
            # Calculate trend
            trend = "increasing" if model.coef_[0] > 0 else "decreasing"
            
            result = f"""
Sales Forecast Results:
- Historical Average Daily Sales: ${daily_sales['Sales'].mean():,.2f}
- Trend: {trend} (slope: {model.coef_[0]:.2f})
- Forecasted Average for Next {periods} Days: ${forecast.mean():,.2f}
- Total Forecasted Sales: ${forecast.sum():,.2f}
- Confidence: Based on linear trend analysis

Key Insights:
- Current daily sales range: ${daily_sales['Sales'].min():,.2f} - ${daily_sales['Sales'].max():,.2f}
- Sales volatility: {daily_sales['Sales'].std():,.2f}
"""
            return result
        except Exception as e:
            return f"Error forecasting sales: {str(e)}"

class PlanBudgetTool(BaseTool):
    name: str = "Plan Budget"
    description: str = "Creates budget allocation recommendations for marketing campaigns across different channels."

    def _run(self, total_budget: float, channels: list = None, priorities: dict = None) -> str:
        try:
            if not channels:
                channels = ['Social Media', 'Search Ads', 'Email Marketing', 'Content Marketing', 'Influencer Marketing']
            
            if not priorities:
                # Default allocation based on industry best practices
                priorities = {
                    'Social Media': 0.30,
                    'Search Ads': 0.25,
                    'Email Marketing': 0.15,
                    'Content Marketing': 0.20,
                    'Influencer Marketing': 0.10
                }
            
            result = f"Budget Allocation Plan (Total: ${total_budget:,.2f}):\n\n"
            
            for channel in channels:
                if channel in priorities:
                    allocation = total_budget * priorities[channel]
                    percentage = priorities[channel] * 100
                    result += f"{channel}: ${allocation:,.2f} ({percentage:.1f}%)\n"
            
            result += f"""

Budget Planning Recommendations:
- Reserve 10-15% for testing new channels
- Monitor performance weekly and reallocate as needed
- Focus on channels with highest ROI
- Consider seasonal adjustments
"""
            return result
        except Exception as e:
            return f"Error planning budget: {str(e)}"

class AnalyzeBrandPerformanceTool(BaseTool):
    name: str = "Analyze Brand Performance"
    description: str = "Analyzes brand performance metrics including awareness, sentiment, and market positioning."

    def _run(self, brand_metrics: dict = None, **kwargs) -> str:
        try:
            if not brand_metrics:
                brand_metrics = kwargs
            
            result = "Brand Performance Analysis:\n\n"
            
            # Brand awareness metrics
            if 'brand_awareness' in brand_metrics:
                awareness = brand_metrics['brand_awareness']
                result += f"Brand Awareness: {awareness}%\n"
                if awareness > 70:
                    result += "- Excellent brand recognition\n"
                elif awareness > 50:
                    result += "- Good brand recognition\n"
                else:
                    result += "- Needs improvement in brand awareness\n"
            
            # Sentiment analysis
            if 'sentiment_score' in brand_metrics:
                sentiment = brand_metrics['sentiment_score']
                result += f"\nBrand Sentiment Score: {sentiment}/10\n"
                if sentiment > 7:
                    result += "- Positive brand sentiment\n"
                elif sentiment > 5:
                    result += "- Neutral brand sentiment\n"
                else:
                    result += "- Negative brand sentiment - needs attention\n"
            
            # Market positioning
            if 'market_position' in brand_metrics:
                position = brand_metrics['market_position']
                result += f"\nMarket Position: #{position}\n"
            
            result += f"""

Brand Performance Recommendations:
- Monitor social media mentions and sentiment
- Conduct regular brand awareness surveys
- Track competitor positioning
- Invest in brand building activities
"""
            return result
        except Exception as e:
            return f"Error analyzing brand performance: {str(e)}"

class CalculateMarketShareTool(BaseTool):
    name: str = "Calculate Market Share"
    description: str = "Calculates market share and competitive positioning analysis."

    def _run(self, company_revenue: float, total_market_revenue: float, competitors: dict = None) -> str:
        try:
            market_share = (company_revenue / total_market_revenue) * 100
            
            result = f"""
Market Share Analysis:
- Company Revenue: ${company_revenue:,.2f}
- Total Market Revenue: ${total_market_revenue:,.2f}
- Market Share: {market_share:.2f}%

Market Position:
"""
            if market_share > 25:
                result += "- Market Leader\n"
            elif market_share > 15:
                result += "- Strong Market Player\n"
            elif market_share > 5:
                result += "- Established Competitor\n"
            else:
                result += "- Niche Player\n"
            
            if competitors:
                result += "\nCompetitor Analysis:\n"
                for competitor, revenue in competitors.items():
                    comp_share = (revenue / total_market_revenue) * 100
                    result += f"- {competitor}: {comp_share:.2f}% market share\n"
            
            result += f"""

Strategic Recommendations:
- Focus on market share growth opportunities
- Analyze competitor strategies
- Identify underserved market segments
- Consider strategic partnerships
"""
            return result
        except Exception as e:
            return f"Error calculating market share: {str(e)}"

class TimeSeriesAnalysisTool(BaseTool):
    name: str = "Time Series Analysis"
    description: str = "Performs comprehensive time series analysis on marketing and sales data to identify trends, seasonality, and patterns."

    def _run(self, data_path: str, date_column: str = 'Date', value_column: str = 'Sales') -> str:
        try:
            df = pd.read_csv(data_path)
            df[date_column] = pd.to_datetime(df[date_column])
            
            # Aggregate data by date if needed
            if len(df.groupby(date_column)) < len(df):
                daily_data = df.groupby(date_column)[value_column].sum().reset_index()
            else:
                daily_data = df[[date_column, value_column]].copy()
            
            daily_data = daily_data.sort_values(date_column)
            
            # Basic statistics
            mean_value = daily_data[value_column].mean()
            std_value = daily_data[value_column].std()
            trend_slope = np.polyfit(range(len(daily_data)), daily_data[value_column], 1)[0]
            
            result = f"""
Time Series Analysis Results:

Basic Statistics:
- Average {value_column}: {mean_value:,.2f}
- Standard Deviation: {std_value:,.2f}
- Coefficient of Variation: {(std_value/mean_value)*100:.2f}%

Trend Analysis:
- Trend Direction: {'Increasing' if trend_slope > 0 else 'Decreasing'}
- Trend Strength: {abs(trend_slope):.2f} units per day
- Overall Growth Rate: {(trend_slope/mean_value)*100:.2f}% per day

Pattern Insights:
- Data Points: {len(daily_data)}
- Date Range: {daily_data[date_column].min().strftime('%Y-%m-%d')} to {daily_data[date_column].max().strftime('%Y-%m-%d')}
- Peak Value: {daily_data[value_column].max():,.2f}
- Minimum Value: {daily_data[value_column].min():,.2f}

Recommendations:
- Monitor trend continuation
- Investigate peak performance periods
- Plan for seasonal variations
"""
            return result
        except Exception as e:
            return f"Error in time series analysis: {str(e)}"

class CrossSectionalAnalysisTool(BaseTool):
    name: str = "Cross Sectional Analysis"
    description: str = "Performs cross-sectional analysis to compare performance across different segments, regions, or products."

    def _run(self, data_path: str, segment_column: str = 'Region', value_column: str = 'Sales') -> str:
        try:
            df = pd.read_csv(data_path)
            
            # Group by segment and calculate statistics
            segment_stats = df.groupby(segment_column)[value_column].agg([
                'count', 'sum', 'mean', 'std', 'min', 'max'
            ]).round(2)
            
            # Calculate market share by segment
            total_value = df[value_column].sum()
            segment_stats['market_share'] = (segment_stats['sum'] / total_value * 100).round(2)
            
            result = f"""
Cross-Sectional Analysis by {segment_column}:

Performance Summary:
"""
            for segment in segment_stats.index:
                stats = segment_stats.loc[segment]
                result += f"""
{segment}:
- Total {value_column}: {stats['sum']:,.2f}
- Average {value_column}: {stats['mean']:,.2f}
- Market Share: {stats['market_share']:.2f}%
- Data Points: {stats['count']}
- Range: {stats['min']:,.2f} - {stats['max']:,.2f}
"""
            
            # Find best and worst performers
            best_performer = segment_stats['sum'].idxmax()
            worst_performer = segment_stats['sum'].idxmin()
            
            result += f"""
Key Insights:
- Best Performing {segment_column}: {best_performer}
- Lowest Performing {segment_column}: {worst_performer}
- Performance Gap: {segment_stats.loc[best_performer, 'sum'] / segment_stats.loc[worst_performer, 'sum']:.2f}x

Strategic Recommendations:
- Focus resources on high-performing segments
- Investigate success factors in {best_performer}
- Develop improvement plans for {worst_performer}
- Consider segment-specific strategies
"""
            return result
        except Exception as e:
            return f"Error in cross-sectional analysis: {str(e)}"

# Create tool instances
calculate_roi = CalculateROITool()
analyze_kpis = AnalyzeKPIsTool()
forecast_sales = ForecastSalesTool()
plan_budget = PlanBudgetTool()
analyze_brand_performance = AnalyzeBrandPerformanceTool()
calculate_market_share = CalculateMarketShareTool()
time_series_analysis = TimeSeriesAnalysisTool()
cross_sectional_analysis = CrossSectionalAnalysisTool()