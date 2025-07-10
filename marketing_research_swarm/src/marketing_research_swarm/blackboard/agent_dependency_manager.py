"""
Agent Dependency Manager for Marketing Research Swarm
Handles automatic agent ordering, dependency resolution, and optimal combinations
"""

from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class AgentPhase(Enum):
    """Phases of marketing research workflow"""
    FOUNDATION = 1      # Data collection and basic research
    ANALYSIS = 2        # Deep analysis and processing
    STRATEGY = 3        # Strategic planning and optimization
    CONTENT = 4         # Content creation and execution

@dataclass
class AgentInfo:
    """Information about an agent including dependencies and capabilities"""
    name: str
    phase: AgentPhase
    dependencies: List[str]  # Agents this agent depends on
    provides: List[str]      # Data/insights this agent provides
    weight: float           # Token usage weight (1.0 = average)
    description: str

class AgentDependencyManager:
    """Manages agent dependencies and optimal ordering"""
    
    def __init__(self):
        self.agents = self._initialize_agent_definitions()
        self.analysis_types = self._initialize_analysis_types()
    
    def _initialize_agent_definitions(self) -> Dict[str, AgentInfo]:
        """Initialize comprehensive agent definitions with dependencies"""
        return {
            'market_research_analyst': AgentInfo(
                name='market_research_analyst',
                phase=AgentPhase.FOUNDATION,
                dependencies=[],  # No dependencies - foundation agent
                provides=['market_data', 'industry_trends', 'consumer_insights'],
                weight=1.2,
                description='Conducts foundational market research and trend analysis'
            ),
            'data_analyst': AgentInfo(
                name='data_analyst',
                phase=AgentPhase.FOUNDATION,
                dependencies=[],  # Can work independently with raw data
                provides=['data_analysis', 'statistical_insights', 'data_patterns'],
                weight=1.1,
                description='Analyzes raw data and provides statistical insights'
            ),
            'competitive_analyst': AgentInfo(
                name='competitive_analyst',
                phase=AgentPhase.ANALYSIS,
                dependencies=['market_research_analyst'],  # Needs market context
                provides=['competitive_landscape', 'competitor_analysis', 'market_positioning'],
                weight=1.0,
                description='Analyzes competitive landscape and market positioning'
            ),
            'brand_performance_specialist': AgentInfo(
                name='brand_performance_specialist',
                phase=AgentPhase.ANALYSIS,
                dependencies=['market_research_analyst', 'data_analyst'],  # Needs market and data context
                provides=['brand_metrics', 'performance_analysis', 'kpi_insights'],
                weight=0.9,
                description='Evaluates brand performance and KPI metrics'
            ),
            'brand_strategist': AgentInfo(
                name='brand_strategist',
                phase=AgentPhase.STRATEGY,
                dependencies=['competitive_analyst', 'brand_performance_specialist'],  # Needs competitive and performance insights
                provides=['brand_strategy', 'positioning_strategy', 'brand_recommendations'],
                weight=0.8,
                description='Develops comprehensive brand strategy and positioning'
            ),
            'campaign_optimizer': AgentInfo(
                name='campaign_optimizer',
                phase=AgentPhase.STRATEGY,
                dependencies=['competitive_analyst', 'brand_performance_specialist'],  # Needs market and performance data
                provides=['campaign_optimization', 'budget_allocation', 'channel_strategy'],
                weight=0.7,
                description='Optimizes campaigns and budget allocation'
            ),
            'forecasting_specialist': AgentInfo(
                name='forecasting_specialist',
                phase=AgentPhase.STRATEGY,
                dependencies=['data_analyst', 'competitive_analyst'],  # Needs data and market context
                provides=['sales_forecasts', 'trend_predictions', 'market_projections'],
                weight=0.9,
                description='Creates forecasts and market projections'
            ),
            'content_strategist': AgentInfo(
                name='content_strategist',
                phase=AgentPhase.CONTENT,
                dependencies=['brand_strategist', 'competitive_analyst'],  # Needs strategy and market context
                provides=['content_strategy', 'messaging_framework', 'content_recommendations'],
                weight=0.8,
                description='Develops content strategy and messaging framework'
            ),
            'creative_copywriter': AgentInfo(
                name='creative_copywriter',
                phase=AgentPhase.CONTENT,
                dependencies=['content_strategist', 'brand_strategist'],  # Needs content strategy and brand direction
                provides=['creative_copy', 'marketing_materials', 'campaign_content'],
                weight=0.6,
                description='Creates compelling copy and marketing materials'
            )
        }
    
    def _initialize_analysis_types(self) -> Dict[str, Dict]:
        """Initialize optimal agent combinations for different analysis types"""
        return {
            'market_research': {
                'name': 'Market Research Analysis',
                'description': 'Comprehensive market research and competitive analysis',
                'agents': ['market_research_analyst', 'competitive_analyst', 'content_strategist'],
                'focus': 'Market understanding and positioning',
                'expected_duration': '2-3 hours',
                'token_efficiency': 'High'
            },
            'brand_performance': {
                'name': 'Brand Performance Analysis',
                'description': 'Deep dive into brand metrics and performance optimization',
                'agents': ['market_research_analyst', 'data_analyst', 'brand_performance_specialist', 'brand_strategist'],
                'focus': 'Brand metrics and strategic optimization',
                'expected_duration': '3-4 hours',
                'token_efficiency': 'Medium-High'
            },
            'sales_forecast': {
                'name': 'Sales Forecasting Analysis',
                'description': 'Data-driven sales forecasting and trend analysis',
                'agents': ['data_analyst', 'market_research_analyst', 'forecasting_specialist'],
                'focus': 'Predictive analytics and sales projections',
                'expected_duration': '2-3 hours',
                'token_efficiency': 'High'
            },
            'campaign_optimization': {
                'name': 'Campaign Optimization',
                'description': 'Complete campaign strategy and optimization',
                'agents': ['market_research_analyst', 'competitive_analyst', 'brand_performance_specialist', 'campaign_optimizer'],
                'focus': 'Campaign strategy and budget optimization',
                'expected_duration': '3-4 hours',
                'token_efficiency': 'Medium'
            },
            'content_strategy': {
                'name': 'Content Strategy Development',
                'description': 'Comprehensive content strategy and creative development',
                'agents': ['market_research_analyst', 'competitive_analyst', 'brand_strategist', 'content_strategist', 'creative_copywriter'],
                'focus': 'Content strategy and creative execution',
                'expected_duration': '4-5 hours',
                'token_efficiency': 'Medium'
            },
            'comprehensive': {
                'name': 'Comprehensive Analysis',
                'description': 'Full-spectrum marketing research and strategy',
                'agents': ['market_research_analyst', 'data_analyst', 'competitive_analyst', 'brand_performance_specialist', 'brand_strategist', 'campaign_optimizer', 'forecasting_specialist', 'content_strategist', 'creative_copywriter'],
                'focus': 'Complete marketing analysis and strategy',
                'expected_duration': '5-6 hours',
                'token_efficiency': 'Low-Medium'
            },
            'custom': {
                'name': 'Custom Analysis',
                'description': 'User-selected agent combination with automatic dependency resolution',
                'agents': [],  # Will be populated by user selection
                'focus': 'Customized analysis based on specific needs',
                'expected_duration': 'Variable',
                'token_efficiency': 'Variable'
            }
        }
    
    def resolve_dependencies(self, selected_agents: List[str]) -> List[str]:
        """
        Resolve dependencies and return optimally ordered agent list
        
        Args:
            selected_agents: List of agent names selected by user
            
        Returns:
            List of agents in optimal execution order
        """
        # Validate all agents exist
        invalid_agents = [agent for agent in selected_agents if agent not in self.agents]
        if invalid_agents:
            raise ValueError(f"Invalid agents: {invalid_agents}")
        
        # Get all required agents (including dependencies)
        required_agents = set(selected_agents)
        self._add_dependencies(required_agents, selected_agents)
        
        # Topological sort to determine execution order
        ordered_agents = self._topological_sort(list(required_agents))
        
        # Filter to only include originally selected agents (dependencies are handled via context)
        final_order = [agent for agent in ordered_agents if agent in selected_agents]
        
        return final_order
    
    def _add_dependencies(self, required_agents: Set[str], selected_agents: List[str]) -> None:
        """Add required dependencies to the agent set"""
        agents_to_check = list(required_agents)
        
        while agents_to_check:
            agent = agents_to_check.pop(0)
            agent_info = self.agents[agent]
            
            for dependency in agent_info.dependencies:
                if dependency not in required_agents:
                    # Only add dependency if it's not already selected by user
                    # Dependencies will be handled through context passing
                    pass
    
    def _topological_sort(self, agents: List[str]) -> List[str]:
        """Perform topological sort based on dependencies and phases"""
        # Create adjacency list
        graph = {agent: [] for agent in agents}
        in_degree = {agent: 0 for agent in agents}
        
        # Build dependency graph
        for agent in agents:
            agent_info = self.agents[agent]
            for dependency in agent_info.dependencies:
                if dependency in agents:
                    graph[dependency].append(agent)
                    in_degree[agent] += 1
        
        # Topological sort with phase consideration
        result = []
        queue = []
        
        # Start with agents that have no dependencies, ordered by phase
        no_deps = [agent for agent in agents if in_degree[agent] == 0]
        no_deps.sort(key=lambda x: (self.agents[x].phase.value, x))
        queue.extend(no_deps)
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            # Process dependents
            dependents = graph[current]
            dependents.sort(key=lambda x: (self.agents[x].phase.value, x))
            
            for dependent in dependents:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        
        # Check for circular dependencies
        if len(result) != len(agents):
            remaining = [agent for agent in agents if agent not in result]
            raise ValueError(f"Circular dependency detected involving: {remaining}")
        
        return result
    
    def get_analysis_types(self) -> Dict[str, Dict]:
        """Get all available analysis types"""
        return self.analysis_types
    
    def get_optimal_combination(self, analysis_type: str) -> List[str]:
        """Get optimal agent combination for analysis type"""
        if analysis_type not in self.analysis_types:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
        
        return self.analysis_types[analysis_type]['agents']
    
    def calculate_token_distribution(self, agents: List[str], total_tokens: int) -> Dict[str, Dict]:
        """Calculate realistic token distribution among agents"""
        if not agents:
            return {}
        
        # Calculate total weight
        total_weight = sum(self.agents[agent].weight for agent in agents)
        
        # Distribute tokens based on weights
        distribution = {}
        remaining_tokens = total_tokens
        
        for i, agent in enumerate(agents):
            agent_info = self.agents[agent]
            
            if i == len(agents) - 1:  # Last agent gets remaining tokens
                agent_tokens = remaining_tokens
            else:
                agent_tokens = int((total_tokens * agent_info.weight) / total_weight)
                remaining_tokens -= agent_tokens
            
            distribution[agent] = {
                'total_tokens': agent_tokens,
                'input_tokens': int(agent_tokens * 0.7),
                'output_tokens': int(agent_tokens * 0.3),
                'cost': agent_tokens * 0.0000025,
                'weight': agent_info.weight,
                'phase': agent_info.phase.name,
                'tasks': {
                    self._get_task_name_for_agent(agent): {
                        'tokens': agent_tokens,
                        'duration': self._estimate_duration(agent_info.weight),
                        'status': 'completed'
                    }
                }
            }
        
        return distribution
    
    def _get_task_name_for_agent(self, agent: str) -> str:
        """Get task name for a given agent"""
        task_mapping = {
            'market_research_analyst': 'market_research',
            'data_analyst': 'data_analysis',
            'competitive_analyst': 'competitive_analysis',
            'brand_performance_specialist': 'brand_performance',
            'brand_strategist': 'brand_strategy',
            'campaign_optimizer': 'campaign_optimization',
            'forecasting_specialist': 'sales_forecast',
            'content_strategist': 'content_strategy',
            'creative_copywriter': 'creative_copywriting'
        }
        return task_mapping.get(agent, 'general_task')
    
    def _estimate_duration(self, weight: float) -> float:
        """Estimate task duration based on agent weight"""
        base_duration = 45.0  # Base duration in seconds
        return base_duration * weight
    
    def validate_combination(self, agents: List[str]) -> Dict[str, any]:
        """Validate agent combination and provide recommendations"""
        if not agents:
            return {'valid': False, 'error': 'No agents selected'}
        
        try:
            ordered_agents = self.resolve_dependencies(agents)
            
            # Calculate efficiency metrics
            phases = set(self.agents[agent].phase for agent in agents)
            total_weight = sum(self.agents[agent].weight for agent in agents)
            avg_weight = total_weight / len(agents)
            
            # Determine efficiency rating
            if len(agents) <= 3 and len(phases) <= 2:
                efficiency = 'High'
            elif len(agents) <= 5 and len(phases) <= 3:
                efficiency = 'Medium'
            else:
                efficiency = 'Low'
            
            return {
                'valid': True,
                'ordered_agents': ordered_agents,
                'efficiency': efficiency,
                'phases_covered': len(phases),
                'estimated_tokens': int(8000 * avg_weight),
                'estimated_duration': sum(self._estimate_duration(self.agents[agent].weight) for agent in agents),
                'recommendations': self._get_recommendations(agents, phases)
            }
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
    
    def _get_recommendations(self, agents: List[str], phases: Set[AgentPhase]) -> List[str]:
        """Get recommendations for improving agent combination"""
        recommendations = []
        
        # Check for missing foundation
        if AgentPhase.FOUNDATION not in phases:
            recommendations.append("Consider adding market_research_analyst or data_analyst for foundational insights")
        
        # Check for unbalanced combinations
        if len(agents) > 6:
            recommendations.append("Large agent combinations may be less efficient - consider focusing on specific analysis type")
        
        # Check for missing strategy
        if AgentPhase.ANALYSIS in phases and AgentPhase.STRATEGY not in phases:
            recommendations.append("Consider adding strategic agents like brand_strategist or campaign_optimizer")
        
        return recommendations

# Global instance
_dependency_manager = None

def get_dependency_manager() -> AgentDependencyManager:
    """Get global dependency manager instance"""
    global _dependency_manager
    if _dependency_manager is None:
        _dependency_manager = AgentDependencyManager()
    return _dependency_manager