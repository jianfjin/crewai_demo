"""
Enhanced Agent Dependencies with Smart Result Sharing
Optimizes all agent combinations and prevents redundant tool executions
"""

from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json

class AgentPhase(Enum):
    """Phases of marketing research workflow"""
    FOUNDATION = 1      # Data collection and basic research
    ANALYSIS = 2        # Deep analysis and processing  
    STRATEGY = 3        # Strategic planning and optimization
    CONTENT = 4         # Content creation and execution

@dataclass
class AgentDependency:
    """Represents a dependency relationship between agents"""
    dependent_agent: str
    required_agent: str
    shared_tools: List[str]
    shared_data_types: List[str]
    dependency_strength: float  # 0.0 to 1.0

class EnhancedAgentDependencyManager:
    """Enhanced dependency manager with smart result sharing"""
    
    def __init__(self):
        self.agent_definitions = self._initialize_enhanced_agent_definitions()
        self.tool_sharing_matrix = self._initialize_tool_sharing_matrix()
        self.data_flow_graph = self._initialize_data_flow_graph()
        
    def _initialize_enhanced_agent_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Initialize comprehensive agent definitions with enhanced dependencies"""
        return {
            'market_research_analyst': {
                'phase': AgentPhase.FOUNDATION,
                'dependencies': [],  # Foundation agent - no dependencies
                'provides': ['market_structure', 'industry_trends', 'consumer_insights', 'market_data'],
                'consumes': [],
                'tools': ['beverage_market_analysis', 'time_series_analysis', 'cross_sectional_analysis'],
                'weight': 1.2,
                'execution_priority': 1
            },
            'data_analyst': {
                'phase': AgentPhase.FOUNDATION,
                'dependencies': ['market_research_analyst'],  # Benefits from market context
                'provides': ['data_insights', 'statistical_analysis', 'performance_metrics', 'kpi_analysis'],
                'consumes': ['market_structure', 'industry_trends'],
                'tools': ['profitability_analysis', 'time_series_analysis', 'cross_sectional_analysis', 'analyze_kpis'],
                'weight': 1.1,
                'execution_priority': 2
            },
            'competitive_analyst': {
                'phase': AgentPhase.ANALYSIS,
                'dependencies': ['market_research_analyst'],  # Needs market context
                'provides': ['competitive_landscape', 'market_positioning', 'competitive_threats'],
                'consumes': ['market_structure', 'industry_trends'],
                'tools': ['beverage_market_analysis', 'calculate_market_share', 'cross_sectional_analysis'],
                'weight': 1.0,
                'execution_priority': 3
            },
            'brand_performance_specialist': {
                'phase': AgentPhase.ANALYSIS,
                'dependencies': ['market_research_analyst', 'data_analyst'],  # Needs both market and data context
                'provides': ['brand_metrics', 'performance_analysis', 'brand_health'],
                'consumes': ['market_data', 'data_insights', 'performance_metrics'],
                'tools': ['analyze_brand_performance', 'calculate_market_share', 'beverage_market_analysis'],
                'weight': 0.9,
                'execution_priority': 4
            },
            'forecasting_specialist': {
                'phase': AgentPhase.ANALYSIS,
                'dependencies': ['market_research_analyst', 'data_analyst'],  # Needs historical data and trends
                'provides': ['sales_forecasts', 'trend_predictions', 'scenario_analysis'],
                'consumes': ['market_data', 'data_insights', 'industry_trends'],
                'tools': ['forecast_sales', 'analyze_kpis'],
                'weight': 0.9,
                'execution_priority': 5
            },
            'brand_strategist': {
                'phase': AgentPhase.STRATEGY,
                'dependencies': ['competitive_analyst', 'brand_performance_specialist'],  # Needs competitive and brand context
                'provides': ['brand_strategy', 'positioning_recommendations', 'growth_strategies'],
                'consumes': ['competitive_landscape', 'brand_metrics', 'market_positioning'],
                'tools': ['analyze_brand_performance', 'profitability_analysis'],
                'weight': 0.8,
                'execution_priority': 6
            },
            'campaign_optimizer': {
                'phase': AgentPhase.STRATEGY,
                'dependencies': ['data_analyst', 'competitive_analyst', 'forecasting_specialist'],  # Needs data, competition, and forecasts
                'provides': ['campaign_strategies', 'budget_allocation', 'roi_optimization'],
                'consumes': ['data_insights', 'competitive_landscape', 'sales_forecasts'],
                'tools': ['plan_budget', 'calculate_roi'],
                'weight': 0.7,
                'execution_priority': 7
            },
            'content_strategist': {
                'phase': AgentPhase.CONTENT,
                'dependencies': ['brand_strategist', 'campaign_optimizer'],  # Needs strategy context
                'provides': ['content_strategy', 'messaging_framework', 'channel_recommendations'],
                'consumes': ['brand_strategy', 'campaign_strategies', 'positioning_recommendations'],
                'tools': ['search', 'web_search'],
                'weight': 0.6,
                'execution_priority': 8
            },
            'creative_copywriter': {
                'phase': AgentPhase.CONTENT,
                'dependencies': ['content_strategist'],  # Needs content strategy
                'provides': ['creative_assets', 'copy_content', 'campaign_materials'],
                'consumes': ['content_strategy', 'messaging_framework'],
                'tools': ['search', 'web_search'],
                'weight': 0.5,
                'execution_priority': 9
            }
        }
    
    def _initialize_tool_sharing_matrix(self) -> Dict[str, Dict[str, float]]:
        """Initialize matrix showing which tools can be shared between agents"""
        return {
            'beverage_market_analysis': {
                'market_research_analyst': 1.0,  # Primary user
                'competitive_analyst': 0.8,     # High reuse potential
                'brand_performance_specialist': 0.6,  # Medium reuse potential
                'data_analyst': 0.4             # Low reuse potential
            },
            'time_series_analysis': {
                'market_research_analyst': 1.0,  # Primary user
                'data_analyst': 0.9,            # High reuse potential
                'forecasting_specialist': 0.7   # Medium reuse potential
            },
            'cross_sectional_analysis': {
                'market_research_analyst': 1.0,  # Primary user
                'data_analyst': 0.9,            # High reuse potential
                'competitive_analyst': 0.7      # Medium reuse potential
            },
            'profitability_analysis': {
                'data_analyst': 1.0,            # Primary user
                'brand_strategist': 0.8,        # High reuse potential
                'campaign_optimizer': 0.6       # Medium reuse potential
            },
            'analyze_kpis': {
                'data_analyst': 1.0,            # Primary user
                'forecasting_specialist': 0.7,  # Medium reuse potential
                'brand_performance_specialist': 0.5  # Low reuse potential
            },
            'calculate_market_share': {
                'competitive_analyst': 1.0,     # Primary user
                'brand_performance_specialist': 0.8  # High reuse potential
            },
            'analyze_brand_performance': {
                'brand_performance_specialist': 1.0,  # Primary user
                'brand_strategist': 0.9         # High reuse potential
            },
            'forecast_sales': {
                'forecasting_specialist': 1.0   # Primary user only
            },
            'plan_budget': {
                'campaign_optimizer': 1.0       # Primary user only
            },
            'calculate_roi': {
                'campaign_optimizer': 1.0       # Primary user only
            }
        }
    
    def _initialize_data_flow_graph(self) -> Dict[str, List[str]]:
        """Initialize data flow graph showing what data flows between agents"""
        return {
            'market_research_analyst': ['data_analyst', 'competitive_analyst'],
            'data_analyst': ['brand_performance_specialist', 'forecasting_specialist', 'campaign_optimizer'],
            'competitive_analyst': ['brand_strategist', 'campaign_optimizer'],
            'brand_performance_specialist': ['brand_strategist'],
            'forecasting_specialist': ['campaign_optimizer'],
            'brand_strategist': ['content_strategist'],
            'campaign_optimizer': ['content_strategist'],
            'content_strategist': ['creative_copywriter'],
            'creative_copywriter': []
        }
    
    def get_optimal_execution_order(self, selected_agents: List[str]) -> List[str]:
        """Get optimal execution order based on dependencies and tool sharing"""
        if not selected_agents:
            return []
        
        # Get agent definitions for selected agents
        agent_defs = {agent: self.agent_definitions[agent] for agent in selected_agents if agent in self.agent_definitions}
        
        # Sort by execution priority first
        sorted_by_priority = sorted(agent_defs.items(), key=lambda x: x[1]['execution_priority'])
        
        # Then apply dependency resolution
        ordered_agents = []
        remaining_agents = {agent: info for agent, info in sorted_by_priority}
        
        while remaining_agents:
            # Find agents with no unresolved dependencies
            ready_agents = []
            for agent, info in remaining_agents.items():
                dependencies = info['dependencies']
                unresolved_deps = [dep for dep in dependencies if dep in remaining_agents]
                if not unresolved_deps:
                    ready_agents.append(agent)
            
            if not ready_agents:
                # Circular dependency or missing agent - add remaining by priority
                ready_agents = [list(remaining_agents.keys())[0]]
            
            # Sort ready agents by priority and add to order
            ready_agents.sort(key=lambda x: remaining_agents[x]['execution_priority'])
            for agent in ready_agents:
                ordered_agents.append(agent)
                del remaining_agents[agent]
        
        return ordered_agents
    
    def get_shared_tool_opportunities(self, selected_agents: List[str]) -> Dict[str, Dict[str, Any]]:
        """Identify tools that can be shared between selected agents"""
        shared_opportunities = {}
        
        for tool_name, sharing_matrix in self.tool_sharing_matrix.items():
            # Find which selected agents can use this tool
            applicable_agents = []
            for agent in selected_agents:
                if agent in sharing_matrix:
                    applicable_agents.append({
                        'agent': agent,
                        'reuse_potential': sharing_matrix[agent]
                    })
            
            if len(applicable_agents) > 1:
                # Sort by reuse potential
                applicable_agents.sort(key=lambda x: x['reuse_potential'], reverse=True)
                
                shared_opportunities[tool_name] = {
                    'primary_agent': applicable_agents[0]['agent'],
                    'secondary_agents': [a['agent'] for a in applicable_agents[1:]],
                    'total_reuse_potential': sum(a['reuse_potential'] for a in applicable_agents[1:]),
                    'execution_strategy': 'cache_and_share'
                }
        
        return shared_opportunities
    
    def get_data_sharing_plan(self, selected_agents: List[str]) -> Dict[str, Dict[str, Any]]:
        """Create a plan for sharing data between agents"""
        execution_order = self.get_optimal_execution_order(selected_agents)
        data_sharing_plan = {}
        
        for i, agent in enumerate(execution_order):
            if agent not in self.agent_definitions:
                continue
                
            agent_info = self.agent_definitions[agent]
            
            # Find what data this agent can consume from previous agents
            available_data = []
            for prev_agent in execution_order[:i]:
                if prev_agent in self.agent_definitions:
                    prev_info = self.agent_definitions[prev_agent]
                    provided_data = prev_info['provides']
                    consumed_data = agent_info['consumes']
                    
                    # Find intersection
                    shared_data = list(set(provided_data) & set(consumed_data))
                    if shared_data:
                        available_data.append({
                            'source_agent': prev_agent,
                            'data_types': shared_data
                        })
            
            data_sharing_plan[agent] = {
                'execution_order': i + 1,
                'available_data': available_data,
                'provides': agent_info['provides'],
                'consumes': agent_info['consumes'],
                'optimization_potential': len(available_data) * 0.2  # 20% efficiency gain per shared data source
            }
        
        return data_sharing_plan
    
    def calculate_efficiency_score(self, selected_agents: List[str]) -> Dict[str, Any]:
        """Calculate efficiency score for the selected agent combination"""
        if not selected_agents:
            return {'score': 0.0, 'breakdown': {}}
        
        # Get sharing opportunities
        tool_sharing = self.get_shared_tool_opportunities(selected_agents)
        data_sharing = self.get_data_sharing_plan(selected_agents)
        
        # Calculate tool sharing efficiency
        tool_efficiency = 0.0
        if tool_sharing:
            total_reuse_potential = sum(opp['total_reuse_potential'] for opp in tool_sharing.values())
            tool_efficiency = min(total_reuse_potential / len(selected_agents), 1.0)
        
        # Calculate data sharing efficiency  
        data_efficiency = 0.0
        if data_sharing:
            total_optimization = sum(plan['optimization_potential'] for plan in data_sharing.values())
            data_efficiency = min(total_optimization / len(selected_agents), 1.0)
        
        # Calculate dependency efficiency (fewer dependencies = higher efficiency)
        dependency_efficiency = 1.0
        for agent in selected_agents:
            if agent in self.agent_definitions:
                deps = self.agent_definitions[agent]['dependencies']
                unresolved_deps = [dep for dep in deps if dep not in selected_agents]
                if unresolved_deps:
                    dependency_efficiency -= 0.1 * len(unresolved_deps)
        
        dependency_efficiency = max(dependency_efficiency, 0.0)
        
        # Overall efficiency score
        overall_score = (tool_efficiency * 0.4 + data_efficiency * 0.4 + dependency_efficiency * 0.2)
        
        return {
            'score': overall_score,
            'breakdown': {
                'tool_sharing_efficiency': tool_efficiency,
                'data_sharing_efficiency': data_efficiency,
                'dependency_efficiency': dependency_efficiency,
                'tool_sharing_opportunities': len(tool_sharing),
                'data_sharing_opportunities': sum(len(plan['available_data']) for plan in data_sharing.values())
            },
            'recommendations': self._generate_efficiency_recommendations(selected_agents, tool_sharing, data_sharing)
        }
    
    def _generate_efficiency_recommendations(self, selected_agents: List[str], 
                                           tool_sharing: Dict, data_sharing: Dict) -> List[str]:
        """Generate recommendations to improve efficiency"""
        recommendations = []
        
        # Check for missing foundation agents
        foundation_agents = [agent for agent in selected_agents 
                           if agent in self.agent_definitions and 
                           self.agent_definitions[agent]['phase'] == AgentPhase.FOUNDATION]
        
        if not foundation_agents:
            recommendations.append("Add a foundation agent (market_research_analyst or data_analyst) for better data flow")
        
        # Check for tool sharing opportunities
        if len(tool_sharing) < 2:
            recommendations.append("Consider agents with overlapping tools for better cache utilization")
        
        # Check for data flow gaps
        isolated_agents = []
        for agent in selected_agents:
            if agent in data_sharing:
                if not data_sharing[agent]['available_data']:
                    isolated_agents.append(agent)
        
        if isolated_agents:
            recommendations.append(f"Agents {isolated_agents} have no data dependencies - consider reordering")
        
        # Check for phase balance
        phases = {}
        for agent in selected_agents:
            if agent in self.agent_definitions:
                phase = self.agent_definitions[agent]['phase']
                phases[phase] = phases.get(phase, 0) + 1
        
        if len(phases) == 1:
            recommendations.append("Consider agents from different phases for comprehensive analysis")
        
        return recommendations

# Global instance
_enhanced_dependency_manager = None

def get_enhanced_dependency_manager() -> EnhancedAgentDependencyManager:
    """Get global enhanced dependency manager instance"""
    global _enhanced_dependency_manager
    if _enhanced_dependency_manager is None:
        _enhanced_dependency_manager = EnhancedAgentDependencyManager()
    return _enhanced_dependency_manager