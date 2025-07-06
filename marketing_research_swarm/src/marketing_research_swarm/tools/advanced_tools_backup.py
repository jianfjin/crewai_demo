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
            if total_cost > 0:
                roi = ((revenue - total_cost) / total_cost) * 100
            else:
                roi = 0
            
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
            
            if 'clicks' in kwargs and 'impressions' in kwargs and kwargs['impressions'] > 0:
                ctr = (kwargs['clicks'] / kwargs['impressions']) * 100
                results += f"Click-Through Rate (CTR): {ctr:.2f}%\n"
            
            if 'conversions' in kwargs and 'clicks' in kwargs and kwargs['clicks'] > 0:
                conversion_rate = (kwargs['conversions'] / kwargs['clicks']) * 100
                results += f"Conversion Rate: {conversion_rate:.2f}%\n"
            
            if 'cost' in kwargs and 'conversions' in kwargs and kwargs['conversions'] > 0:
                cac = kwargs['cost'] / kwargs['conversions']
                results += f"Customer Acquisition Cost (CAC): ${cac:.2f}\n"
            
            if 'revenue' in kwargs and 'conversions' in kwargs and kwargs['conversions'] > 0:
                avg_order_value = kwargs['revenue'] / kwargs['conversions']
                results += f"Average Order Value: ${avg_order_value:.2f}\n"
            
            return results
        except Exception as e:
            return f"Error analyzing KPIs: {str(e)}"

class ForecastSalesTool(BaseTool):
    name: str = "Forecast Sales"
    description: str = "Performs sales forecasting using time series analysis and machine learning techniques on beverage sales data."

    def _run(self, data_path: str, periods: int = 30, forecast_column: str = 'total_revenue') -> str:
        try:
            df = pd.read_csv(data_path)
            df['sale_date'] = pd.to_datetime(df['sale_date'])
            
            # Aggregate sales by date
            daily_sales = df.groupby('sale_date')[forecast_column].sum().reset_index()
            daily_sales = daily_sales.sort_values('sale_date')
            
            # Simple linear regression forecast
            X = np.arange(len(daily_sales)).reshape(-1, 1)
            y = daily_sales[forecast_column].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            # Forecast future periods
            future_X = np.arange(len(daily_sales), len(daily_sales) + periods).reshape(-1, 1)
            forecast = model.predict(future_X)
            
            # Calculate trend
            trend = "increasing" if model.coef_[0] > 0 else "decreasing"
            
            # Additional analysis for beverage data
            total_units = df['units_sold'].sum()
            avg_price = df['price_per_unit'].mean()
            avg_profit_margin = df['profit_margin'].mean()
            
            result = f"""
Sales Forecast Results:
- Historical Average Daily Revenue: ${daily_sales[forecast_column].mean():,.2f}
- Trend: {trend} (slope: ${model.coef_[0]:.2f}/day)
- Forecasted Average for Next {periods} Days: ${forecast.mean():,.2f}
- Total Forecasted Revenue: ${forecast.sum():,.2f}
- Confidence: Based on linear trend analysis

Beverage Market Insights:
- Total Units Sold (Historical): {total_units:,}
- Average Price per Unit: ${avg_price:.2f}
- Average Profit Margin: {avg_profit_margin:.2f}%
- Current daily revenue range: ${daily_sales[forecast_column].min():,.2f} - ${daily_sales[forecast_column].max():,.2f}
- Revenue volatility: ${daily_sales[forecast_column].std():,.2f}

Forecast Recommendations:
- Monitor seasonal patterns in beverage consumption
- Consider weather and seasonal factors
- Track competitor pricing and promotions
- Plan inventory based on forecasted demand
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
            if total_market_revenue > 0:
                market_share = (company_revenue / total_market_revenue) * 100
            else:
                market_share = 0
            
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
            
            if competitors and total_market_revenue > 0:
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
    description: str = "Performs comprehensive time series analysis on beverage sales data to identify trends, seasonality, and patterns across multiple metrics."

    def _run(self, data_path: str, date_column: str = 'sale_date', value_column: str = 'total_revenue') -> str:
        try:
            df = pd.read_csv(data_path)
            df[date_column] = pd.to_datetime(df[date_column])
            
            # Aggregate data by date if needed
            if len(df.groupby(date_column)) < len(df):
                daily_data = df.groupby(date_column).agg({
                    value_column: 'sum',
                    'units_sold': 'sum',
                    'profit': 'sum',
                    'profit_margin': 'mean'
                }).reset_index()
            else:
                daily_data = df[[date_column, value_column, 'units_sold', 'profit', 'profit_margin']].copy()
            
            daily_data = daily_data.sort_values(date_column)
            
            # Basic statistics for multiple metrics
            mean_revenue = daily_data[value_column].mean()
            std_revenue = daily_data[value_column].std()
            trend_slope = np.polyfit(range(len(daily_data)), daily_data[value_column], 1)[0]
            
            # Protect against division by zero
            if mean_revenue == 0:
                mean_revenue = 0.01
            
            mean_units = daily_data['units_sold'].mean()
            mean_profit = daily_data['profit'].mean()
            mean_margin = daily_data['profit_margin'].mean()
            
            # Seasonal analysis by quarter
            df['quarter'] = df['quarter']
            quarterly_performance = df.groupby('quarter').agg({
                'total_revenue': 'sum',
                'units_sold': 'sum',
                'profit_margin': 'mean'
            }).round(2)
            
            result = f"""
Time Series Analysis Results:

Revenue Analysis:
- Average Daily Revenue: ${mean_revenue:,.2f}
- Revenue Standard Deviation: ${std_revenue:,.2f}
- Revenue Coefficient of Variation: {(std_revenue/mean_revenue)*100:.2f}%

Units & Profitability:
- Average Daily Units Sold: {mean_units:,.0f}
- Average Daily Profit: ${mean_profit:,.2f}
- Average Profit Margin: {mean_margin:.2f}%

Trend Analysis:
- Revenue Trend Direction: {'Increasing' if trend_slope > 0 else 'Decreasing'}
- Revenue Trend Strength: ${abs(trend_slope):.2f} per day
- Overall Growth Rate: {(trend_slope/mean_revenue)*100:.2f}% per day

Quarterly Performance:
"""
            for quarter in quarterly_performance.index:
                qdata = quarterly_performance.loc[quarter]
                result += f"- {quarter}: ${qdata['total_revenue']:,.2f} revenue, {qdata['units_sold']:,} units, {qdata['profit_margin']:.2f}% margin\n"
            
            result += f"""
Pattern Insights:
- Data Points: {len(daily_data)}
- Date Range: {daily_data[date_column].min().strftime('%Y-%m-%d')} to {daily_data[date_column].max().strftime('%Y-%m-%d')}
- Peak Revenue Day: ${daily_data[value_column].max():,.2f}
- Minimum Revenue Day: ${daily_data[value_column].min():,.2f}

Beverage Industry Recommendations:
- Monitor seasonal consumption patterns
- Track quarterly performance trends
- Optimize pricing strategies based on margin trends
- Plan inventory for peak demand periods
- Consider weather impact on beverage sales
"""
            return result
        except Exception as e:
            return f"Error in time series analysis: {str(e)}"

class CrossSectionalAnalysisTool(BaseTool):
    name: str = "Cross Sectional Analysis"
    description: str = "Performs cross-sectional analysis to compare beverage performance across different segments, regions, brands, or categories."

    def _run(self, data_path: str, segment_column: str = 'region', value_column: str = 'total_revenue') -> str:
        try:
            df = pd.read_csv(data_path)
            
            # Group by segment and calculate comprehensive statistics
            segment_stats = df.groupby(segment_column).agg({
                value_column: ['count', 'sum', 'mean', 'std', 'min', 'max'],
                'units_sold': 'sum',
                'profit': 'sum',
                'profit_margin': 'mean',
                'price_per_unit': 'mean'
            }).round(2)
            
            # Flatten column names
            segment_stats.columns = ['_'.join(col).strip() for col in segment_stats.columns]
            
            # Calculate market share by segment
            total_value = df[value_column].sum()
            if total_value > 0:
                segment_stats['market_share'] = (segment_stats[f'{value_column}_sum'] / total_value * 100).round(2)
            else:
                segment_stats['market_share'] = 0
            
            result = f"""
Cross-Sectional Analysis by {segment_column.title()}:

Performance Summary:
"""
            for segment in segment_stats.index:
                stats = segment_stats.loc[segment]
                result += f"""
{segment}:
- Total Revenue: ${stats[f'{value_column}_sum']:,.2f}
- Average Revenue per Transaction: ${stats[f'{value_column}_mean']:,.2f}
- Market Share: {stats['market_share']:.2f}%
- Total Units Sold: {stats['units_sold_sum']:,}
- Total Profit: ${stats['profit_sum']:,.2f}
- Average Profit Margin: {stats['profit_margin_mean']:.2f}%
- Average Price per Unit: ${stats['price_per_unit_mean']:.2f}
- Transactions: {stats[f'{value_column}_count']}
- Revenue Range: ${stats[f'{value_column}_min']:,.2f} - ${stats[f'{value_column}_max']:,.2f}
"""
            
            # Find best and worst performers
            best_performer = segment_stats['market_share'].idxmax()
            worst_performer = segment_stats['market_share'].idxmin()
            
            # Additional analysis for beverage industry
            if segment_column == 'brand':
                result += "\nBrand Performance Insights:\n"
                top_brands = segment_stats.nlargest(3, 'market_share')
                for brand in top_brands.index:
                    margin = segment_stats.loc[brand, 'profit_margin_mean']
                    share = segment_stats.loc[brand, 'market_share']
                    result += f"- {brand}: {share:.1f}% market share, {margin:.1f}% profit margin\n"
            
            elif segment_column == 'category':
                result += "\nCategory Performance Insights:\n"
                for category in segment_stats.index:
                    avg_price = segment_stats.loc[category, 'price_per_unit_mean']
                    units = segment_stats.loc[category, 'units_sold_sum']
                    result += f"- {category}: ${avg_price:.2f} avg price, {units:,} units sold\n"
            
            result += f"""
Key Insights:
- Best Performing {segment_column.title()}: {best_performer} ({segment_stats.loc[best_performer, 'market_share']:.1f}% market share)
- Lowest Performing {segment_column.title()}: {worst_performer} ({segment_stats.loc[worst_performer, 'market_share']:.1f}% market share)
- Performance Gap: {segment_stats.loc[best_performer, 'market_share'] / segment_stats.loc[worst_performer, 'market_share']:.2f}x

Strategic Recommendations:
- Focus marketing resources on high-performing {segment_column}s
- Investigate success factors in {best_performer}
- Develop improvement plans for {worst_performer}
- Consider {segment_column}-specific pricing and promotion strategies
- Analyze profit margin opportunities across {segment_column}s
"""
            return result
        except Exception as e:
            return f"Error in cross-sectional analysis: {str(e)}"

class BeverageMarketAnalysisTool(BaseTool):
    name: str = "Beverage Market Analysis"
    description: str = "Performs comprehensive beverage market analysis including brand performance, category trends, and regional insights."

    def _run(self, data_path: str) -> str:
        try:
            df = pd.read_csv(data_path)
            
            # Overall market statistics
            total_revenue = df['total_revenue'].sum()
            total_units = df['units_sold'].sum()
            avg_price = df['price_per_unit'].mean()
            avg_margin = df['profit_margin'].mean()
            
            # Brand analysis
            brand_performance = df.groupby('brand').agg({
                'total_revenue': 'sum',
                'units_sold': 'sum',
                'profit_margin': 'mean'
            }).round(2)
            if total_revenue > 0:
                brand_performance['market_share'] = (brand_performance['total_revenue'] / total_revenue * 100).round(2)
            else:
                brand_performance['market_share'] = 0
            brand_performance = brand_performance.sort_values('market_share', ascending=False)
            
            # Category analysis
            category_performance = df.groupby('category').agg({
                'total_revenue': 'sum',
                'units_sold': 'sum',
                'price_per_unit': 'mean',
                'profit_margin': 'mean'
            }).round(2)
            if total_revenue > 0:
                category_performance['market_share'] = (category_performance['total_revenue'] / total_revenue * 100).round(2)
            else:
                category_performance['market_share'] = 0
            category_performance = category_performance.sort_values('market_share', ascending=False)
            
            # Regional analysis
            regional_performance = df.groupby('region').agg({
                'total_revenue': 'sum',
                'units_sold': 'sum',
                'profit_margin': 'mean'
            }).round(2)
            if total_revenue > 0:
                regional_performance['market_share'] = (regional_performance['total_revenue'] / total_revenue * 100).round(2)
            else:
                regional_performance['market_share'] = 0
            
            result = f"""
Comprehensive Beverage Market Analysis:

Market Overview:
- Total Market Revenue: ${total_revenue:,.2f}
- Total Units Sold: {total_units:,}
- Average Price per Unit: ${avg_price:.2f}
- Average Profit Margin: {avg_margin:.2f}%

Top Brand Performance:
"""
            for brand in brand_performance.head(5).index:
                data = brand_performance.loc[brand]
                result += f"- {brand}: {data['market_share']:.1f}% share, ${data['total_revenue']:,.0f} revenue, {data['profit_margin']:.1f}% margin\n"
            
            result += f"""
Category Performance:
"""
            for category in category_performance.index:
                data = category_performance.loc[category]
                result += f"- {category}: {data['market_share']:.1f}% share, ${data['price_per_unit']:.2f} avg price, {data['profit_margin']:.1f}% margin\n"
            
            result += f"""
Regional Performance:
"""
            for region in regional_performance.index:
                data = regional_performance.loc[region]
                result += f"- {region}: {data['market_share']:.1f}% share, {data['units_sold']:,} units, {data['profit_margin']:.1f}% margin\n"
            
            # Market insights
            top_brand = brand_performance.index[0]
            top_category = category_performance.index[0]
            top_region = regional_performance['market_share'].idxmax()
            
            result += f"""
Key Market Insights:
- Market Leader: {top_brand} ({brand_performance.loc[top_brand, 'market_share']:.1f}% share)
- Dominant Category: {top_category} ({category_performance.loc[top_category, 'market_share']:.1f}% share)
- Strongest Region: {top_region} ({regional_performance.loc[top_region, 'market_share']:.1f}% share)
- Price Range: ${df['price_per_unit'].min():.2f} - ${df['price_per_unit'].max():.2f}
- Margin Range: {df['profit_margin'].min():.1f}% - {df['profit_margin'].max():.1f}%

Strategic Recommendations:
- Focus on high-margin categories and brands
- Expand successful brands to underperforming regions
- Optimize pricing strategies by category
- Investigate regional preferences and adapt offerings
- Monitor competitive positioning in key categories
"""
            return result
        except Exception as e:
            return f"Error in beverage market analysis: {str(e)}"

class ProfitabilityAnalysisTool(BaseTool):
    name: str = "Profitability Analysis"
    description: str = "Analyzes profitability metrics across different dimensions including profit margins, cost structures, and pricing strategies."

    def _run(self, data_path: str, analysis_dimension: str = 'brand') -> str:
        try:
            df = pd.read_csv(data_path)
            
            # Overall profitability metrics
            total_revenue = df['total_revenue'].sum()
            total_cost = df['total_cost'].sum()
            total_profit = df['profit'].sum()
            overall_margin = (total_profit / total_revenue * 100)
            
            # Profitability by dimension
            profit_analysis = df.groupby(analysis_dimension).agg({
                'total_revenue': 'sum',
                'total_cost': 'sum',
                'profit': 'sum',
                'profit_margin': 'mean',
                'price_per_unit': 'mean',
                'cost_per_unit': 'mean',
                'units_sold': 'sum'
            }).round(2)
            
            profit_analysis['calculated_margin'] = (profit_analysis['profit'] / profit_analysis['total_revenue'] * 100).round(2)
            profit_analysis = profit_analysis.sort_values('calculated_margin', ascending=False)
            
            result = f"""
Profitability Analysis by {analysis_dimension.title()}:

Overall Market Profitability:
- Total Revenue: ${total_revenue:,.2f}
- Total Cost: ${total_cost:,.2f}
- Total Profit: ${total_profit:,.2f}
- Overall Margin: {overall_margin:.2f}%

Profitability Breakdown:
"""
            for item in profit_analysis.index:
                data = profit_analysis.loc[item]
                if data['total_cost'] > 0:
                    roi = ((data['total_revenue'] - data['total_cost']) / data['total_cost'] * 100)
                else:
                    roi = 0
                result += f"""
{item}:
- Revenue: ${data['total_revenue']:,.2f}
- Cost: ${data['total_cost']:,.2f}
- Profit: ${data['profit']:,.2f}
- Profit Margin: {data['calculated_margin']:.2f}%
- ROI: {roi:.2f}%
- Avg Price: ${data['price_per_unit']:.2f}
- Avg Cost: ${data['cost_per_unit']:.2f}
- Units Sold: {data['units_sold']:,}
"""
            
            # Identify best and worst performers
            best_margin = profit_analysis.index[0]
            worst_margin = profit_analysis.index[-1]
            
            result += f"""
Profitability Insights:
- Highest Margin {analysis_dimension.title()}: {best_margin} ({profit_analysis.loc[best_margin, 'calculated_margin']:.2f}%)
- Lowest Margin {analysis_dimension.title()}: {worst_margin} ({profit_analysis.loc[worst_margin, 'calculated_margin']:.2f}%)
- Margin Spread: {profit_analysis.loc[best_margin, 'calculated_margin'] - profit_analysis.loc[worst_margin, 'calculated_margin']:.2f} percentage points

Optimization Recommendations:
- Focus on high-margin {analysis_dimension}s for growth
- Investigate cost reduction opportunities for {worst_margin}
- Consider pricing optimization for low-margin items
- Analyze volume vs. margin trade-offs
- Benchmark against industry standards
"""
            return result
        except Exception as e:
            return f"Error in profitability analysis: {str(e)}"

# Create tool instances
calculate_roi = CalculateROITool()
analyze_kpis = AnalyzeKPIsTool()
forecast_sales = ForecastSalesTool()
plan_budget = PlanBudgetTool()
analyze_brand_performance = AnalyzeBrandPerformanceTool()
calculate_market_share = CalculateMarketShareTool()
time_series_analysis = TimeSeriesAnalysisTool()
cross_sectional_analysis = CrossSectionalAnalysisTool()
beverage_market_analysis = BeverageMarketAnalysisTool()
profitability_analysis = ProfitabilityAnalysisTool()