"""
Optimized tools that work with cached data and return structured results
"""

from crewai.tools import BaseTool
import pandas as pd
import numpy as np
from typing import Dict, Any, Union, Optional
from pydantic import BaseModel, ConfigDict
import hashlib
import pickle

class ProfitabilityResult(BaseModel):
    """Structured result for profitability analysis"""
    model_config = {"arbitrary_types_allowed": True}
    
    analysis_dimension: str
    total_revenue: float
    total_cost: float
    total_profit: float
    overall_margin: float
    top_performer: Dict[str, Any]
    bottom_performer: Dict[str, Any]
    performance_breakdown: Dict[str, Dict[str, float]]
    key_insights: Dict[str, str]

class ROIResult(BaseModel):
    """Structured result for ROI calculation"""
    model_config = {"arbitrary_types_allowed": True}
    
    revenue: float
    total_cost: float
    roi_percentage: float
    net_profit: float
    roi_category: str  # profitable, highly_successful, needs_optimization

class BudgetPlan(BaseModel):
    """Structured result for budget planning"""
    model_config = {"arbitrary_types_allowed": True}
    
    total_budget: float
    channel_allocations: Dict[str, float]
    allocation_percentages: Dict[str, float]
    recommendations: list[str]
    optimization_score: float

class OptimizedProfitabilityTool(BaseTool):
    name: str = "Optimized Profitability Analysis"
    description: str = "Analyzes profitability using cached DataFrame data and returns structured results"
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(self, data: Union[pd.DataFrame, str], analysis_dimension: str = 'brand') -> ProfitabilityResult:
        """
        Run profitability analysis on cached data
        
        Args:
            data: DataFrame or cache reference
            analysis_dimension: Dimension to analyze (brand, category, region)
        """
        try:
            # Handle cached data reference
            if isinstance(data, str) and data.startswith('cache://'):
                # In real implementation, retrieve from cache
                # For now, assume we have the data
                raise ValueError("Cache retrieval not implemented in this demo")
            
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Data must be a pandas DataFrame")
            
            df = data
            
            # Perform profitability analysis
            grouped = df.groupby(analysis_dimension).agg({
                'total_revenue': 'sum',
                'total_cost': 'sum', 
                'profit': 'sum',
                'profit_margin': 'mean',
                'units_sold': 'sum'
            }).round(2)
            
            # Calculate overall metrics
            total_revenue = df['total_revenue'].sum()
            total_cost = df['total_cost'].sum()
            total_profit = df['profit'].sum()
            overall_margin = (total_profit / total_revenue * 100) if total_revenue > 0 else 0
            
            # Find top and bottom performers
            grouped['calculated_margin'] = (grouped['profit'] / grouped['total_revenue'] * 100).fillna(0)
            top_performer_idx = grouped['calculated_margin'].idxmax()
            bottom_performer_idx = grouped['calculated_margin'].idxmin()
            
            top_performer = {
                'name': top_performer_idx,
                'revenue': float(grouped.loc[top_performer_idx, 'total_revenue']),
                'margin': float(grouped.loc[top_performer_idx, 'calculated_margin']),
                'profit': float(grouped.loc[top_performer_idx, 'profit'])
            }
            
            bottom_performer = {
                'name': bottom_performer_idx,
                'revenue': float(grouped.loc[bottom_performer_idx, 'total_revenue']),
                'margin': float(grouped.loc[bottom_performer_idx, 'calculated_margin']),
                'profit': float(grouped.loc[bottom_performer_idx, 'profit'])
            }
            
            # Create performance breakdown (top 5)
            top_5 = grouped.nlargest(5, 'calculated_margin')
            performance_breakdown = {}
            for idx, row in top_5.iterrows():
                performance_breakdown[str(idx)] = {
                    'revenue': float(row['total_revenue']),
                    'margin': float(row['calculated_margin']),
                    'profit': float(row['profit']),
                    'units': float(row['units_sold'])
                }
            
            # Generate key insights
            key_insights = {
                'top_performer_insight': f"{top_performer['name']} leads with {top_performer['margin']:.1f}% margin",
                'market_concentration': f"Top 5 {analysis_dimension}s represent {(top_5['total_revenue'].sum()/total_revenue*100):.1f}% of revenue",
                'margin_spread': f"Margin spread: {grouped['calculated_margin'].max():.1f}% - {grouped['calculated_margin'].min():.1f}%",
                'optimization_opportunity': f"Focus on {top_performer['name']} for highest returns"
            }
            
            return ProfitabilityResult(
                analysis_dimension=analysis_dimension,
                total_revenue=float(total_revenue),
                total_cost=float(total_cost),
                total_profit=float(total_profit),
                overall_margin=float(overall_margin),
                top_performer=top_performer,
                bottom_performer=bottom_performer,
                performance_breakdown=performance_breakdown,
                key_insights=key_insights
            )
            
        except Exception as e:
            # Return error result
            return ProfitabilityResult(
                analysis_dimension=analysis_dimension,
                total_revenue=0.0,
                total_cost=0.0,
                total_profit=0.0,
                overall_margin=0.0,
                top_performer={'name': 'Error', 'revenue': 0, 'margin': 0, 'profit': 0},
                bottom_performer={'name': 'Error', 'revenue': 0, 'margin': 0, 'profit': 0},
                performance_breakdown={},
                key_insights={'error': f"Analysis failed: {str(e)}"}
            )

class OptimizedROITool(BaseTool):
    name: str = "Optimized ROI Calculator"
    description: str = "Calculates ROI with structured output and categorization"

    def _run(self, revenue: float, cost: float, additional_costs: float = 0) -> ROIResult:
        """Calculate ROI with structured output"""
        try:
            total_cost = cost + additional_costs
            
            if total_cost > 0:
                roi_percentage = ((revenue - total_cost) / total_cost) * 100
            else:
                roi_percentage = 0
            
            net_profit = revenue - total_cost
            
            # Categorize ROI
            if roi_percentage > 100:
                roi_category = "highly_successful"
            elif roi_percentage > 0:
                roi_category = "profitable"
            else:
                roi_category = "needs_optimization"
            
            return ROIResult(
                revenue=float(revenue),
                total_cost=float(total_cost),
                roi_percentage=float(roi_percentage),
                net_profit=float(net_profit),
                roi_category=roi_category
            )
            
        except Exception as e:
            return ROIResult(
                revenue=0.0,
                total_cost=0.0,
                roi_percentage=0.0,
                net_profit=0.0,
                roi_category="error"
            )

class OptimizedBudgetTool(BaseTool):
    name: str = "Optimized Budget Planner"
    description: str = "Creates optimized budget allocation with structured output"

    def _run(self, total_budget: float, insights: Dict[str, Any] = None, 
             channels: list = None, priorities: Dict[str, float] = None) -> BudgetPlan:
        """Create optimized budget plan based on insights"""
        try:
            if not channels:
                channels = ['Social Media', 'Search Ads', 'Email Marketing', 'Content Marketing', 'Influencer Marketing']
            
            if not priorities:
                # Default allocation
                priorities = {
                    'Social Media': 0.30,
                    'Search Ads': 0.25,
                    'Email Marketing': 0.15,
                    'Content Marketing': 0.20,
                    'Influencer Marketing': 0.10
                }
            
            # Adjust priorities based on insights
            if insights:
                priorities = self._adjust_priorities_from_insights(priorities, insights)
            
            # Calculate allocations
            channel_allocations = {}
            allocation_percentages = {}
            
            for channel in channels:
                if channel in priorities:
                    allocation = total_budget * priorities[channel]
                    channel_allocations[channel] = float(allocation)
                    allocation_percentages[channel] = float(priorities[channel] * 100)
            
            # Generate recommendations
            recommendations = self._generate_budget_recommendations(insights, channel_allocations)
            
            # Calculate optimization score
            optimization_score = self._calculate_optimization_score(priorities, insights)
            
            return BudgetPlan(
                total_budget=float(total_budget),
                channel_allocations=channel_allocations,
                allocation_percentages=allocation_percentages,
                recommendations=recommendations,
                optimization_score=float(optimization_score)
            )
            
        except Exception as e:
            return BudgetPlan(
                total_budget=float(total_budget),
                channel_allocations={},
                allocation_percentages={},
                recommendations=[f"Error in budget planning: {str(e)}"],
                optimization_score=0.0
            )
    
    def _adjust_priorities_from_insights(self, base_priorities: Dict[str, float], 
                                       insights: Dict[str, Any]) -> Dict[str, float]:
        """Adjust budget priorities based on analysis insights"""
        adjusted = base_priorities.copy()
        
        # Look for premium/high-margin insights
        insight_text = str(insights).lower()
        
        if 'premium' in insight_text or 'high margin' in insight_text:
            # Increase influencer marketing for premium products
            adjusted['Influencer Marketing'] = min(0.20, adjusted.get('Influencer Marketing', 0.10) + 0.05)
            adjusted['Social Media'] = max(0.20, adjusted.get('Social Media', 0.30) - 0.05)
        
        if 'digital' in insight_text or 'online' in insight_text:
            # Increase digital channels
            adjusted['Search Ads'] = min(0.35, adjusted.get('Search Ads', 0.25) + 0.05)
            adjusted['Email Marketing'] = max(0.10, adjusted.get('Email Marketing', 0.15) - 0.05)
        
        # Normalize to ensure sum = 1.0
        total = sum(adjusted.values())
        if total > 0:
            adjusted = {k: v/total for k, v in adjusted.items()}
        
        return adjusted
    
    def _generate_budget_recommendations(self, insights: Dict[str, Any], 
                                       allocations: Dict[str, float]) -> list[str]:
        """Generate budget recommendations based on insights and allocations"""
        recommendations = [
            "Monitor performance weekly and adjust allocations based on ROI",
            "Reserve 10-15% of budget for testing new channels and strategies",
            "Focus spend on channels with highest conversion rates"
        ]
        
        if insights:
            # Add insight-specific recommendations
            top_performer = insights.get('top_performer')
            if top_performer:
                recommendations.append(f"Prioritize marketing for {top_performer} based on high performance")
        
        # Add allocation-specific recommendations
        top_channel = max(allocations.items(), key=lambda x: x[1]) if allocations else None
        if top_channel:
            recommendations.append(f"Monitor {top_channel[0]} closely as it receives largest allocation (${top_channel[1]:,.0f})")
        
        return recommendations
    
    def _calculate_optimization_score(self, priorities: Dict[str, float], 
                                    insights: Dict[str, Any]) -> float:
        """Calculate optimization score based on allocation strategy"""
        base_score = 75.0  # Base score for reasonable allocation
        
        # Bonus for insight-driven adjustments
        if insights:
            base_score += 10.0
        
        # Bonus for balanced allocation (no single channel > 40%)
        max_allocation = max(priorities.values()) if priorities else 0
        if max_allocation <= 0.40:
            base_score += 10.0
        
        # Bonus for diversification (at least 3 channels with >5%)
        significant_channels = sum(1 for v in priorities.values() if v > 0.05)
        if significant_channels >= 3:
            base_score += 5.0
        
        return min(100.0, base_score)

# Create tool instances
optimized_profitability_analysis = OptimizedProfitabilityTool()
optimized_roi_calculator = OptimizedROITool()
optimized_budget_planner = OptimizedBudgetTool()