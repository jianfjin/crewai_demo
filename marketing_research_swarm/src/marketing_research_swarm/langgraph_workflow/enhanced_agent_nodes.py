"""
Enhanced Agent Nodes with Smart Tool Selection

This module provides enhanced agent node functions that integrate smart tool selection
into the LangGraph workflow, replacing the original agent nodes with optimized versions.
"""

import os
import yaml
import logging
from typing import Dict, Any
from datetime import datetime

from .state import MarketingResearchState, store_agent_result, store_agent_error
from .enhanced_agents import EnhancedLangGraphAgent

logger = logging.getLogger(__name__)


def load_agent_configs(config_path: str) -> Dict[str, Dict[str, Any]]:
    """Load agent configurations from YAML file."""
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Failed to load agent configs from {config_path}: {e}")
        return {}


def create_enhanced_agent_from_config(agent_name: str, agent_config: Dict[str, Any]) -> EnhancedLangGraphAgent:
    """Create an enhanced LangGraph agent from configuration."""
    return EnhancedLangGraphAgent(
        role=agent_config.get('role', agent_name),
        goal=agent_config.get('goal', ''),
        backstory=agent_config.get('backstory', ''),
        tools=agent_config.get('tools', []),
        llm_config=agent_config.get('llm', 'openai/gpt-4o-mini')
    )


# Enhanced Agent Node Functions
def enhanced_market_research_analyst_node(state: MarketingResearchState) -> MarketingResearchState:
    """Enhanced Market Research Analyst node with smart tool selection."""
    agent_role = 'market_research_analyst'
    
    try:
        # Load agent config
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'agents.yaml')
        agent_configs = load_agent_configs(config_path)
        agent_config = agent_configs.get(agent_role, {})
        
        # Create enhanced agent
        agent = create_enhanced_agent_from_config(agent_role, agent_config)
        
        # Define enhanced task with smart tool integration
        task_description = f"""
        Conduct comprehensive market research on the beverage industry using enhanced sales data and smart analytical tools.
        
        **Specific Research Focus**:
        - Target audience: {state['target_audience']}
        - Campaign type: {state.get('campaign_type', 'marketing campaign')}
        - Budget: ${state.get('budget', 0):,} over {state.get('duration', 'campaign period')}
        - Brands to research: {', '.join(state.get('brands', ['all brands']))}
        
        **Market Analysis Requirements**:
        - Market structure and dynamics for {', '.join(state.get('market_segments', ['selected']))} segments
        - Seasonal patterns and trends in {', '.join(state.get('product_categories', ['selected']))} categories
        - Pricing strategies and profit margins for {', '.join(state.get('brands', ['selected brands']))}
        - Regional performance analysis across {', '.join(state.get('market_segments', ['target']))} markets
        - Competitive landscape overview for {', '.join(state.get('brands', ['target brands']))}
        - Strategic recommendations for {state.get('analysis_focus', 'beverage market')}
        
        **Expected Deliverables**:
        - Market opportunity assessment for {state['target_audience']}
        - Brand positioning insights for: {', '.join(state.get('brands', ['selected brands']))}
        - Regional market dynamics in: {', '.join(state.get('market_segments', ['target markets']))}
        - Competitive intelligence and strategic recommendations
        
        **Business Context**:
        - Business Objective: {state.get('business_objective', 'Market optimization')}
        - Competitive Landscape: {state.get('competitive_landscape', 'Competitive market environment')}
        - Key Metrics: {', '.join(state.get('key_metrics', ['performance metrics']))}
        """
        
        # Execute task with smart tool selection
        result = agent.execute_task_with_smart_tools(state, task_description)
        
        # Store result in state
        state = store_agent_result(state, agent_role, result)
        
        logger.info(f"Enhanced {agent_role} completed with {result.get('tool_results', {}).get('_execution_metadata', {}).get('total_tools_executed', 0)} tools")
        
    except Exception as e:
        logger.error(f"Error in enhanced {agent_role} node: {e}")
        state = store_agent_error(state, agent_role, str(e))
    
    return state


def enhanced_competitive_analyst_node(state: MarketingResearchState) -> MarketingResearchState:
    """Enhanced Competitive Analyst node with smart tool selection."""
    agent_role = 'competitive_analyst'
    
    try:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'agents.yaml')
        agent_configs = load_agent_configs(config_path)
        agent_config = agent_configs.get(agent_role, {})
        
        agent = create_enhanced_agent_from_config(agent_role, agent_config)
        
        task_description = f"""
        Analyze competitive landscape, market positioning, and brand dynamics using smart analytical tools.
        
        **Specific Competitive Analysis Requirements**:
        - Target brands for analysis: {', '.join(state.get('brands', ['all brands']))}
        - Market segments to examine: {', '.join(state.get('market_segments', ['all segments']))}
        - Product categories focus: {', '.join(state.get('product_categories', ['all categories']))}
        - Target audience context: {state['target_audience']}
        - Budget context: ${state.get('budget', 0):,} over {state.get('duration', 'campaign period')}
        
        **Competitive Intelligence Focus**:
        - Competitive positioning analysis for {', '.join(state.get('brands', ['selected brands']))}
        - Brand market share evaluation in {', '.join(state.get('market_segments', ['target']))} markets
        - Competitive threats and opportunities for {', '.join(state.get('brands', ['target brands']))}
        - Market structure analysis in {', '.join(state.get('product_categories', ['focus']))} categories
        - Strategic competitive insights for {state.get('campaign_type', 'marketing campaign')}
        
        **Competitive Landscape Context**: {state.get('competitive_landscape', 'Dynamic competitive environment')}
        
        **Expected Deliverables**:
        - Brand positioning analysis for: {', '.join(state.get('brands', ['selected brands']))}
        - Market share insights in: {', '.join(state.get('market_segments', ['target markets']))}
        - Competitive threat assessment and strategic recommendations
        - Competitive advantage identification and exploitation strategies
        """
        
        result = agent.execute_task_with_smart_tools(state, task_description)
        state = store_agent_result(state, agent_role, result)
        
        logger.info(f"Enhanced {agent_role} completed with smart tool selection")
        
    except Exception as e:
        logger.error(f"Error in enhanced {agent_role} node: {e}")
        state = store_agent_error(state, agent_role, str(e))
    
    return state


def enhanced_data_analyst_node(state: MarketingResearchState) -> MarketingResearchState:
    """Enhanced Data Analyst node with smart tool selection."""
    agent_role = 'data_analyst'
    
    try:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'agents.yaml')
        agent_configs = load_agent_configs(config_path)
        agent_config = agent_configs.get(agent_role, {})
        
        agent = create_enhanced_agent_from_config(agent_role, agent_config)
        
        task_description = f"""
        Perform comprehensive data analysis on beverage sales to uncover profitability insights, trends, and performance patterns using smart analytical tools.
        
        **Data Analysis Requirements**:
        - Analyze specifically these brands: {', '.join(state.get('brands', ['all brands']))}
        - Focus on these market segments: {', '.join(state.get('market_segments', ['all segments']))}
        - Examine these product categories: {', '.join(state.get('product_categories', ['all categories']))}
        - Track these key metrics: {', '.join(state.get('key_metrics', ['performance metrics']))}
        - Budget context: ${state.get('budget', 0):,} over {state.get('duration', 'campaign period')}
        
        **Analysis Focus Areas**:
        - Profitability analysis across the specified brands, categories, and regions
        - Time series analysis for trend identification in selected markets
        - Cross-sectional analysis for performance comparison between chosen brands
        - KPI analysis and performance metrics for {', '.join(state.get('brands', ['selected brands']))}
        - Statistical insights and data interpretation for {state.get('target_audience', 'target audience')}
        - Forecasting analysis for {state.get('forecast_periods', 30)} periods
        
        **Expected Deliverables**:
        - Brand-specific performance analysis for: {', '.join(state.get('brands', ['selected brands']))}
        - Market segment performance in: {', '.join(state.get('market_segments', ['target markets']))}
        - Actionable insights for {state.get('target_audience', 'target audience')} campaign
        - Data-driven recommendations with specific metrics and projections
        """
        
        result = agent.execute_task_with_smart_tools(state, task_description)
        state = store_agent_result(state, agent_role, result)
        
        logger.info(f"Enhanced {agent_role} completed with comprehensive data analysis")
        
    except Exception as e:
        logger.error(f"Error in enhanced {agent_role} node: {e}")
        state = store_agent_error(state, agent_role, str(e))
    
    return state


def enhanced_content_strategist_node(state: MarketingResearchState) -> MarketingResearchState:
    """Enhanced Content Strategist node with smart tool selection."""
    agent_role = 'content_strategist'
    
    try:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'agents.yaml')
        agent_configs = load_agent_configs(config_path)
        agent_config = agent_configs.get(agent_role, {})
        
        agent = create_enhanced_agent_from_config(agent_role, agent_config)
        
        task_description = f"""
        Develop a comprehensive content strategy for beverage brands based on market research insights and analytical data.
        
        **Specific Content Strategy Requirements**:
        - Target audience: {state['target_audience']}
        - Campaign type: {state.get('campaign_type', 'marketing campaign')}
        - Budget allocation: ${state.get('budget', 0):,} over {state.get('duration', 'campaign period')}
        - Target brands: {', '.join(state.get('brands', ['selected brands']))}
        - Market segments: {', '.join(state.get('market_segments', ['target segments']))}
        - Product categories: {', '.join(state.get('product_categories', ['focus categories']))}
        
        **Campaign Goals**: {', '.join(state.get('campaign_goals', ['brand awareness', 'engagement']))}
        
        **Content Strategy Focus Areas**:
        - Channel-specific content recommendations for {state.get('campaign_type', 'multi-channel campaign')}
        - Seasonal campaign strategies for {', '.join(state.get('product_categories', ['target']))} categories
        - Brand positioning strategies for {', '.join(state.get('brands', ['target brands']))}
        - Social media campaign ideas targeting {state['target_audience']}
        - Influencer collaboration strategies in {', '.join(state.get('market_segments', ['target']))} markets
        - Regional content adaptation for {', '.join(state.get('market_segments', ['target']))} regions
        
        **Content Requirements**:
        - Budget-conscious strategies within ${state.get('budget', 0):,} allocation
        - {state.get('duration', 'Campaign duration')}-specific content calendar
        - Brand-specific messaging for: {', '.join(state.get('brands', ['selected brands']))}
        
        **Expected Deliverables**:
        - Content strategy framework for {', '.join(state.get('brands', ['target brands']))}
        - Channel recommendations for {state.get('campaign_type', 'campaign')}
        - Regional content adaptation strategy for {', '.join(state.get('market_segments', ['target markets']))}
        
        **Integration Requirements**:
        - Build upon market research insights from previous agents
        - Reference competitive analysis findings when developing positioning strategies
        - Integrate data analysis results into content recommendations
        """
        
        result = agent.execute_task_with_smart_tools(state, task_description)
        state = store_agent_result(state, agent_role, result)
        
        logger.info(f"Enhanced {agent_role} completed with data-driven content strategy")
        
    except Exception as e:
        logger.error(f"Error in enhanced {agent_role} node: {e}")
        state = store_agent_error(state, agent_role, str(e))
    
    return state


def enhanced_creative_copywriter_node(state: MarketingResearchState) -> MarketingResearchState:
    """Enhanced Creative Copywriter node with smart tool selection."""
    agent_role = 'creative_copywriter'
    
    try:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'agents.yaml')
        agent_configs = load_agent_configs(config_path)
        agent_config = agent_configs.get(agent_role, {})
        
        agent = create_enhanced_agent_from_config(agent_role, agent_config)
        
        task_description = f"""
        Create compelling marketing copy for beverage campaigns targeting {state['target_audience']} based on analytical insights.
        
        **Specific Copywriting Requirements**:
        - Target audience: {state['target_audience']}
        - Campaign type: {state.get('campaign_type', 'marketing campaign')}
        - Budget context: ${state.get('budget', 0):,} over {state.get('duration', 'campaign period')}
        - Key brands to write for: {', '.join(state.get('brands', ['selected brands']))}
        - Product categories: {', '.join(state.get('product_categories', ['focus categories']))}
        - Market segments: {', '.join(state.get('market_segments', ['target segments']))}
        
        **Campaign Goals**: {', '.join(state.get('campaign_goals', ['brand awareness', 'engagement']))}
        
        **Copy Development Focus**:
        - Social media posts and content for {', '.join(state.get('brands', ['target brands']))}
        - Product descriptions and messaging for {', '.join(state.get('product_categories', ['focus']))} categories
        - Campaign taglines and slogans for {state.get('campaign_type', 'campaign')}
        - Promotional content targeting {state['target_audience']}
        - Brand voice and messaging guidelines for {', '.join(state.get('brands', ['selected brands']))}
        - Regional content variations for {', '.join(state.get('market_segments', ['target']))} markets
        
        **Copy Requirements**:
        - Align with {state.get('analysis_focus', 'campaign focus')} messaging
        - Budget-appropriate content for ${state.get('budget', 0):,} campaign
        - {state.get('duration', 'Campaign duration')}-specific messaging
        
        **Expected Deliverables**:
        - Brand-specific copy for: {', '.join(state.get('brands', ['selected brands']))}
        - Regional copy variations for: {', '.join(state.get('market_segments', ['target markets']))}
        - Campaign messaging aligned with {', '.join(state.get('campaign_goals', ['objectives']))}
        
        **Integration Requirements**:
        - Use content strategy guidance from the Content Strategist
        - Reference market research findings for audience-appropriate messaging
        - Align copy with data analysis insights on brand performance
        """
        
        result = agent.execute_task_with_smart_tools(state, task_description)
        state = store_agent_result(state, agent_role, result)
        
        logger.info(f"Enhanced {agent_role} completed with strategic copy development")
        
    except Exception as e:
        logger.error(f"Error in enhanced {agent_role} node: {e}")
        state = store_agent_error(state, agent_role, str(e))
    
    return state


def enhanced_campaign_optimizer_node(state: MarketingResearchState) -> MarketingResearchState:
    """Enhanced Campaign Optimizer node with smart tool selection."""
    agent_role = 'campaign_optimizer'
    
    try:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'agents.yaml')
        agent_configs = load_agent_configs(config_path)
        agent_config = agent_configs.get(agent_role, {})
        
        agent = create_enhanced_agent_from_config(agent_role, agent_config)
        
        task_description = f"""
        Develop comprehensive optimization strategies for beverage marketing campaigns using analytical insights and smart tools.
        
        **Specific Campaign Optimization Requirements**:
        - Total budget: ${state.get('budget', 0):,}
        - Campaign duration: {state.get('duration', 'campaign period')}
        - Target audience: {state['target_audience']}
        - Campaign type: {state.get('campaign_type', 'marketing campaign')}
        - Target brands: {', '.join(state.get('brands', ['selected brands']))}
        - Market segments: {', '.join(state.get('market_segments', ['target segments']))}
        - Product categories: {', '.join(state.get('product_categories', ['focus categories']))}
        - Expected revenue: ${state.get('expected_revenue', 25000):,}
        
        **Campaign Goals**: {', '.join(state.get('campaign_goals', ['optimization', 'ROI improvement']))}
        
        **Optimization Focus Areas**:
        - Budget allocation recommendations across {', '.join(state.get('market_segments', ['target']))} regions
        - ROI projections for {', '.join(state.get('brands', ['target brands']))} brands
        - Channel strategy optimization for {state.get('campaign_type', 'multi-channel campaign')}
        - Regional focus strategies for {', '.join(state.get('market_segments', ['target']))} markets
        - Performance improvement recommendations for {', '.join(state.get('product_categories', ['focus']))} categories
        - Cost optimization suggestions within ${state.get('budget', 0):,} budget
        
        **Key Metrics to Optimize**: {', '.join(state.get('key_metrics', ['ROI', 'efficiency']))}
        
        **Expected Deliverables**:
        - Budget allocation strategy for ${state.get('budget', 0):,} across {', '.join(state.get('market_segments', ['target markets']))}
        - ROI optimization plan for {', '.join(state.get('brands', ['target brands']))}
        - Channel performance optimization for {state.get('campaign_type', 'campaign')}
        - Regional investment priorities and cost optimization recommendations
        
        **Integration Requirements**:
        - Use data analysis results to inform budget allocation decisions
        - Reference content strategy recommendations for channel optimization
        - Build upon market research insights for regional prioritization
        """
        
        result = agent.execute_task_with_smart_tools(state, task_description)
        state = store_agent_result(state, agent_role, result)
        
        logger.info(f"Enhanced {agent_role} completed with optimization strategies")
        
    except Exception as e:
        logger.error(f"Error in enhanced {agent_role} node: {e}")
        state = store_agent_error(state, agent_role, str(e))
    
    return state


def enhanced_brand_performance_specialist_node(state: MarketingResearchState) -> MarketingResearchState:
    """Enhanced Brand Performance Specialist node with smart tool selection."""
    agent_role = 'brand_performance_specialist'
    
    try:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'agents.yaml')
        agent_configs = load_agent_configs(config_path)
        agent_config = agent_configs.get(agent_role, {})
        
        agent = create_enhanced_agent_from_config(agent_role, agent_config)
        
        task_description = f"""
        Analyze brand performance in the beverage market using comprehensive analytical tools and data-driven insights.
        
        **Specific Brand Performance Analysis Requirements**:
        - Key brands to analyze: {', '.join(state.get('brands', ['selected brands']))}
        - Market segments focus: {', '.join(state.get('market_segments', ['target segments']))}
        - Product categories: {', '.join(state.get('product_categories', ['focus categories']))}
        - Target audience context: {state['target_audience']}
        - Campaign budget context: ${state.get('budget', 0):,} over {state.get('duration', 'period')}
        - Brand metrics to track: {', '.join(state.get('key_metrics', ['brand_performance', 'market_share']))}
        
        **Brand Performance Focus Areas**:
        - Brand positioning and market share analysis for {', '.join(state.get('brands', ['target brands']))}
        - Competitive landscape evaluation in {', '.join(state.get('market_segments', ['target']))} markets
        - Brand health metrics and performance indicators for {', '.join(state.get('brands', ['selected brands']))}
        - Growth opportunities identification in {', '.join(state.get('product_categories', ['focus']))} categories
        - Strategic brand recommendations for {state.get('campaign_type', 'marketing initiatives')}
        - Market penetration analysis across {', '.join(state.get('market_segments', ['target']))} regions
        
        **Brand Context**:
        - Brand awareness: {state.get('brand_metrics', {}).get('brand_awareness', 'N/A')}%
        - Sentiment score: {state.get('brand_metrics', {}).get('sentiment_score', 'N/A')}
        - Market position: {state.get('brand_metrics', {}).get('market_position', 'N/A')}
        
        **Expected Deliverables**:
        - Brand performance dashboard for: {', '.join(state.get('brands', ['selected brands']))}
        - Market share analysis in: {', '.join(state.get('market_segments', ['target markets']))}
        - Brand growth opportunities and strategic recommendations
        - Competitive positioning insights for target brands
        
        **Integration Requirements**:
        - Use competitive analysis results to assess brand positioning
        - Reference data analysis findings for performance metrics validation
        - Build upon market research insights for growth opportunity identification
        """
        
        result = agent.execute_task_with_smart_tools(state, task_description)
        state = store_agent_result(state, agent_role, result)
        
        logger.info(f"Enhanced {agent_role} completed with brand performance insights")
        
    except Exception as e:
        logger.error(f"Error in enhanced {agent_role} node: {e}")
        state = store_agent_error(state, agent_role, str(e))
    
    return state


def enhanced_forecasting_specialist_node(state: MarketingResearchState) -> MarketingResearchState:
    """Enhanced Forecasting Specialist node with smart tool selection."""
    agent_role = 'forecasting_specialist'
    
    try:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'agents.yaml')
        agent_configs = load_agent_configs(config_path)
        agent_config = agent_configs.get(agent_role, {})
        
        agent = create_enhanced_agent_from_config(agent_role, agent_config)
        
        task_description = f"""
        Generate accurate sales forecasts and predictive models using advanced analytical tools and data science techniques.
        
        **Specific Forecasting Requirements**:
        - Forecast periods: {state.get('forecast_periods', 30)} days
        - Target brands for forecasting: {', '.join(state.get('brands', ['selected brands']))}
        - Market segments: {', '.join(state.get('market_segments', ['target segments']))}
        - Product categories: {', '.join(state.get('product_categories', ['focus categories']))}
        - Expected revenue target: ${state.get('expected_revenue', 25000):,}
        - Campaign context: ${state.get('budget', 0):,} over {state.get('duration', 'forecast period')}
        - Target audience: {state['target_audience']}
        
        **Forecasting Focus Areas**:
        - Sales forecasting with confidence intervals for {', '.join(state.get('brands', ['target brands']))}
        - Predictive modeling and trend analysis in {', '.join(state.get('market_segments', ['target']))} markets
        - Scenario planning for {state.get('campaign_type', 'marketing campaign')} impact
        - Market trend predictions for {', '.join(state.get('product_categories', ['focus']))} categories
        - Revenue projections for {', '.join(state.get('brands', ['selected brands']))}
        - Risk assessment and forecasting accuracy for ${state.get('budget', 0):,} investment
        
        **Key Metrics to Forecast**: {', '.join(state.get('key_metrics', ['revenue', 'sales']))}
        
        **Expected Deliverables**:
        - {state.get('forecast_periods', 30)}-day sales forecast for: {', '.join(state.get('brands', ['selected brands']))}
        - Revenue projections across: {', '.join(state.get('market_segments', ['target markets']))}
        - Risk assessment for ${state.get('expected_revenue', 25000):,} revenue target
        - Market trend predictions and scenario analysis
        
        **Integration Requirements**:
        - Use market research findings to inform trend predictions
        - Reference data analysis results for historical pattern validation
        - Consider competitive analysis insights for scenario planning
        """
        
        result = agent.execute_task_with_smart_tools(state, task_description)
        state = store_agent_result(state, agent_role, result)
        
        logger.info(f"Enhanced {agent_role} completed with predictive analytics")
        
    except Exception as e:
        logger.error(f"Error in enhanced {agent_role} node: {e}")
        state = store_agent_error(state, agent_role, str(e))
    
    return state


def enhanced_report_summarizer_node(state: MarketingResearchState) -> MarketingResearchState:
    """Enhanced Report Summarizer node to generate the final report."""
    agent_role = 'report_summarizer'
    
    try:
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'agents.yaml')
        agent_configs = load_agent_configs(config_path)
        agent_config = agent_configs.get(agent_role, {})
        
        agent = create_enhanced_agent_from_config(agent_role, agent_config)
        
        # Consolidate all previous agent results into a comprehensive summary
        all_results = state.get('agent_results', {})
        agent_errors = state.get('agent_errors', {})
        
        summary_of_results = ""
        for agent_name, result in all_results.items():
            # Handle both dictionary and string results
            summary_of_results += f"### {agent_name.replace('_', ' ').title()} Analysis\n"
            
            if isinstance(result, dict):
                # Handle dictionary results
                analysis = result.get('analysis', 'N/A')
                if isinstance(analysis, dict):
                    # If analysis is a dict, convert to string
                    analysis = str(analysis)
                summary_of_results += f"- **Analysis**: {analysis}\n"
                
                # Extract key insights if available
                if 'key_insights' in result:
                    insights = result['key_insights']
                    if isinstance(insights, list):
                        summary_of_results += f"- **Key Insights**: {', '.join(str(i) for i in insights[:5])}\n"
                    else:
                        summary_of_results += f"- **Key Insights**: {insights}\n"
                        
                # Extract recommendations if available
                if 'recommendations' in result:
                    recommendations = result['recommendations']
                    if isinstance(recommendations, list):
                        summary_of_results += f"- **Recommendations**: {', '.join(str(r) for r in recommendations[:5])}\n"
                    else:
                        summary_of_results += f"- **Recommendations**: {recommendations}\n"
            elif isinstance(result, str):
                # Handle string results
                summary_of_results += f"- **Analysis**: {result[:500]}{'...' if len(result) > 500 else ''}\n"
            else:
                # Handle other result types
                summary_of_results += f"- **Result**: {str(result)}\n"
            
            summary_of_results += "\n"
        
        # Add any agent errors to the summary
        if agent_errors:
            summary_of_results += "### Agent Execution Issues\n"
            for agent_name, error in agent_errors.items():
                summary_of_results += f"- **{agent_name.replace('_', ' ').title()}**: {error}\n"
            summary_of_results += "\n"

        task_description = f"""
        Synthesize all the provided analysis results into a single, comprehensive final report.
        
        **Analysis Summary from All Agents**:
        {summary_of_results}
        
        **Final Report Requirements**:
        - **Executive Summary**: Start with a high-level overview of the key findings and strategic recommendations.
        - **Integrated Insights**: Combine insights from all agents to tell a cohesive story about the market, competition, and brand performance.
        - **Strategic Recommendations**: Provide a consolidated list of actionable recommendations based on the entire body of research.
        - **Conclusion**: End with a concluding paragraph that summarizes the overall outlook and next steps.
        
        Your final output should be a polished, professional report ready for executive review.
        """
        
        # Since this agent has no tools, we call the LLM directly with the consolidated context
        from langchain.schema import SystemMessage, HumanMessage
        
        system_message = SystemMessage(content=f"You are a {agent.role} with the goal to {agent.goal}. Your backstory is: {agent.backstory}")
        human_message = HumanMessage(content=task_description)
        
        response = agent.llm.invoke([system_message, human_message])
        
        final_report = {
            'final_summary': response.content,
            'timestamp': datetime.now().isoformat()
        }
        
        state = store_agent_result(state, agent_role, final_report)
        state['final_report'] = final_report
        
        logger.info(f"Enhanced {agent_role} completed and generated the final report.")
        
    except Exception as e:
        logger.error(f"Error in enhanced {agent_role} node: {e}")
        state = store_agent_error(state, agent_role, str(e))
    
    return state


# Enhanced Agent Node Mapping
ENHANCED_AGENT_NODES = {
    # Lowercase with underscores (original format)
    'market_research_analyst': enhanced_market_research_analyst_node,
    'competitive_analyst': enhanced_competitive_analyst_node,
    'data_analyst': enhanced_data_analyst_node,
    'content_strategist': enhanced_content_strategist_node,
    'creative_copywriter': enhanced_creative_copywriter_node,
    'campaign_optimizer': enhanced_campaign_optimizer_node,
    'brand_performance_specialist': enhanced_brand_performance_specialist_node,
    'forecasting_specialist': enhanced_forecasting_specialist_node,
    'report_summarizer': enhanced_report_summarizer_node,
    
    # Title case with spaces (dashboard format) - CRITICAL FIX
    'Market Research Analyst': enhanced_market_research_analyst_node,
    'Competitive Analyst': enhanced_competitive_analyst_node,
    'Data Analyst': enhanced_data_analyst_node,
    'Content Strategist': enhanced_content_strategist_node,
    'Creative Copywriter': enhanced_creative_copywriter_node,
    'Campaign Optimizer': enhanced_campaign_optimizer_node,
    'Brand Performance Analyst': enhanced_brand_performance_specialist_node,
    'Sales Forecast Analyst': enhanced_forecasting_specialist_node,
    'ROI Analysis Expert': enhanced_forecasting_specialist_node,  # Using forecasting for ROI as well
    'Report Summarizer': enhanced_report_summarizer_node,
}
