"""
Optimization Manager for Token Usage Reduction
Manages switching between original and optimized crews with performance tracking.
Includes context isolation and result reference management.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import json
import uuid
from .blackboard.integrated_blackboard import get_integrated_blackboard
from .utils.token_tracker import get_token_tracker

class OptimizationManager:
    """Manages optimization strategies and performance tracking with context isolation."""
    
    def __init__(self):
        self.optimization_history = []
        self.current_mode = "standard"  # or "optimized"
        self.blackboard = get_integrated_blackboard()
        self.token_tracker = get_token_tracker()
        self.context_isolation_enabled = True
        self.result_references: Dict[str, str] = {}  # Maps logical keys to storage keys
    
    def get_crew_instance(self, mode: str = "optimized", **kwargs):
        """Get crew instance based on optimization mode."""
        
        if mode == "blackboard":
            from .blackboard.blackboard_crew import create_blackboard_crew
            # Create blackboard crew with config paths
            agents_config = kwargs.get('agents_config_path', 'src/marketing_research_swarm/config/agents.yaml')
            tasks_config = kwargs.get('tasks_config_path', 'src/marketing_research_swarm/config/tasks.yaml')
            return create_blackboard_crew(agents_config, tasks_config)
        elif mode == "comprehensive":
            from .flows.comprehensive_dynamic_flow import create_comprehensive_flow
            return create_comprehensive_flow()
        elif mode == "optimized":
            from .crew_optimized import OptimizedMarketingResearchCrew
            return OptimizedMarketingResearchCrew(**kwargs)
        elif mode == "simple_optimized":
            from .crew_simple_optimized import SimpleOptimizedCrew
            return SimpleOptimizedCrew(**kwargs)
        else:
            from .crew_with_tracking import MarketingResearchCrewWithTracking
            return MarketingResearchCrewWithTracking(**kwargs)
    
    def store_tool_output(self, tool_name: str, output: Any, context_key: str = None) -> str:
        """Store tool output and return reference key instead of dumping to context."""
        if not self.context_isolation_enabled:
            return str(output)  # Fallback to direct output
        
        # Generate unique storage key
        storage_key = f"{tool_name}_{uuid.uuid4().hex[:8]}"
        
        # Store in blackboard using update_agent_results method
        try:
            self.blackboard.update_agent_results(
                workflow_id="default",
                agent_role="system",
                results={storage_key: output}
            )
            
            # Track reference if context key provided
            if context_key:
                self.result_references[context_key] = storage_key
            
            print(f"[STORED] {tool_name} output: {storage_key} ({len(str(output))} bytes)")
            return f"[RESULT_REF:{storage_key}]"
        except Exception as e:
            print(f"[ERROR] Failed to store tool output: {e}")
            return str(output)  # Fallback to direct output
    
    def retrieve_by_reference(self, reference_key: str) -> Any:
        """Retrieve stored result by reference key."""
        if reference_key.startswith("[RESULT_REF:") and reference_key.endswith("]"):
            storage_key = reference_key[12:-1]  # Extract key from [RESULT_REF:key]
            try:
                # Try to get from workflow summary
                summary = self.blackboard.get_workflow_summary("default")
                if "agent_results" in summary:
                    for agent_data in summary["agent_results"].values():
                        if storage_key in agent_data.get("results", {}):
                            return agent_data["results"][storage_key]
                return reference_key  # Return as-is if not found
            except Exception as e:
                print(f"[ERROR] Failed to retrieve by reference: {e}")
                return reference_key
        return reference_key  # Return as-is if not a reference
    
    def create_isolated_context(self, agent_role: str, relevant_refs: List[str] = None) -> Dict[str, Any]:
        """Create isolated context window for agent with only relevant data."""
        if not self.context_isolation_enabled:
            return {}
        
        isolated_context = {
            'agent_role': agent_role,
            'timestamp': datetime.now().isoformat(),
            'available_references': {}
        }
        
        # Include only relevant references
        if relevant_refs:
            for ref_key in relevant_refs:
                if ref_key in self.result_references:
                    storage_key = self.result_references[ref_key]
                    # Store reference, not actual data
                    isolated_context['available_references'][ref_key] = f"[RESULT_REF:{storage_key}]"
        
        print(f"[CONTEXT] Created isolated context for {agent_role}: {len(isolated_context['available_references'])} references")
        return isolated_context
    
    def extract_metrics_from_output(self, output: str) -> Dict[str, Any]:
        """Extract token usage metrics from crew output."""
        try:
            # Try to get actual metrics from token tracker
            if hasattr(self.token_tracker, 'crew_usage') and self.token_tracker.crew_usage:
                actual_usage = self.token_tracker.crew_usage.total_token_usage
                return {
                    'total_tokens': actual_usage.total_tokens,
                    'input_tokens': actual_usage.prompt_tokens,
                    'output_tokens': actual_usage.completion_tokens,
                    'total_cost': actual_usage.total_tokens * 0.0000025,  # Rough estimate
                    'successful_requests': len(self.token_tracker.crew_usage.task_usages),
                    'estimated': False,
                    'source': 'actual_tracking'
                }
            else:
                # Fallback to estimation
                estimated_tokens = min(len(output.split()) * 1.3, 8000)  # Rough estimation
                return {
                    'total_tokens': int(estimated_tokens),
                    'input_tokens': int(estimated_tokens * 0.7),
                    'output_tokens': int(estimated_tokens * 0.3),
                    'total_cost': estimated_tokens * 0.0000025,
                    'successful_requests': 1,
                    'estimated': True,
                    'source': 'fallback_estimate'
                }
        except Exception as e:
            print(f"Error extracting metrics: {e}")
            return {
                'total_tokens': 8000,
                'input_tokens': 5600,
                'output_tokens': 2400,
                'total_cost': 0.0025,
                'successful_requests': 1,
                'estimated': True,
                'source': 'error_fallback'
            }

    def run_analysis_with_optimization(self, inputs: Dict[str, Any], 
                                     optimization_level: str = "full",
                                     custom_tasks_config_path: str = None) -> Dict[str, Any]:
        """
        Run analysis with specified optimization level and context isolation.
        
        Args:
            inputs: Analysis inputs
            optimization_level: "none", "partial", "full", "blackboard"
        """
        
        start_time = datetime.now()
        workflow_id = str(uuid.uuid4())
        
        # Start workflow tracking
        if optimization_level in ["blackboard", "comprehensive"]:
            try:
                workflow_context = self.blackboard.create_integrated_workflow(workflow_id, inputs)
                print(f"[WORKFLOW] Started {optimization_level} workflow: {workflow_id}")
            except Exception as e:
                print(f"[ERROR] Failed to start {optimization_level} workflow: {e}")
        
        # Configure optimization based on level
        if optimization_level == "full":
            crew_mode = "optimized"
            agents_config_path = "src/marketing_research_swarm/config/agents_optimized.yaml"
            tasks_config_path = custom_tasks_config_path or "src/marketing_research_swarm/config/tasks_optimized.yaml"
        elif optimization_level == "comprehensive":
            crew_mode = "comprehensive"
            # Comprehensive flow doesn't use config files - it has built-in agent configs
            agents_config_path = None
            tasks_config_path = None
        elif optimization_level == "blackboard":
            crew_mode = "blackboard"
            agents_config_path = "src/marketing_research_swarm/config/agents.yaml"
            tasks_config_path = custom_tasks_config_path or "src/marketing_research_swarm/config/tasks_context_aware.yaml"
        elif optimization_level == "partial":
            crew_mode = "simple_optimized"
            agents_config_path = "src/marketing_research_swarm/config/agents.yaml"
            tasks_config_path = custom_tasks_config_path or "src/marketing_research_swarm/config/tasks.yaml"
        else:  # "none"
            crew_mode = "standard"
            agents_config_path = "src/marketing_research_swarm/config/agents.yaml"
            tasks_config_path = custom_tasks_config_path or "src/marketing_research_swarm/config/tasks.yaml"
        
        try:
            # Get crew instance
            if crew_mode == "comprehensive":
                crew = self.get_crew_instance(mode=crew_mode)
                
                # For comprehensive flow, we need to prepare agent selection and task params
                # Default to all 9 agents for comprehensive analysis
                selected_agents = [
                    'market_research_analyst', 'data_analyst', 'competitive_analyst',
                    'brand_performance_specialist', 'brand_strategist', 'campaign_optimizer',
                    'forecasting_specialist', 'content_strategist', 'creative_copywriter'
                ]
                
                # Convert inputs to task_params format expected by comprehensive flow
                task_params = {
                    'data_file_path': inputs.get('data_file_path', 'data/beverage_sales.csv'),
                    'target_audience': inputs.get('target_audience', 'health-conscious millennials'),
                    'budget': inputs.get('budget', '$100,000'),
                    'duration': inputs.get('duration', '3 months'),
                    'campaign_goals': inputs.get('campaign_goals', 'increase brand awareness and market share')
                }
                
                print(f"[ANALYSIS] Running {optimization_level} optimization with {crew_mode} flow")
                print(f"[AGENTS] Selected agents: {selected_agents}")
                result = crew.kickoff(selected_agents=selected_agents, task_params=task_params)
            else:
                crew = self.get_crew_instance(
                    mode=crew_mode,
                    agents_config_path=agents_config_path,
                    tasks_config_path=tasks_config_path
                )
                
                # Run analysis
                print(f"[ANALYSIS] Running {optimization_level} optimization with {crew_mode} crew")
                result = crew.kickoff(inputs=inputs)
            
            # Extract metrics
            metrics = self.extract_metrics_from_output(str(result))
            
            # Record optimization result
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            optimization_record = {
                'workflow_id': workflow_id,
                'optimization_level': optimization_level,
                'crew_mode': crew_mode,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'duration_seconds': duration,
                'metrics': metrics,
                'inputs': inputs,
                'result_length': len(str(result))
            }
            
            self.optimization_history.append(optimization_record)
            
            # Finalize workflow if using blackboard or comprehensive
            if optimization_level in ["blackboard", "comprehensive"]:
                try:
                    cleanup_result = self.blackboard.cleanup_workflow(workflow_id)
                    print(f"[WORKFLOW] Finalized {optimization_level} workflow: {workflow_id}")
                except Exception as e:
                    print(f"[ERROR] Failed to finalize {optimization_level} workflow: {e}")
            
            return {
                'result': result,
                'metrics': metrics,
                'optimization_record': optimization_record
            }
            
        except Exception as e:
            error_record = {
                'workflow_id': workflow_id,
                'optimization_level': optimization_level,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            self.optimization_history.append(error_record)
            
            print(f"[ERROR] Analysis failed: {e}")
            return {
                'result': f"Analysis failed: {e}",
                'metrics': self.extract_metrics_from_output(""),
                'error': str(e)
            }
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization performance."""
        if not self.optimization_history:
            return {'message': 'No optimization history available'}
        
        successful_runs = [r for r in self.optimization_history if 'error' not in r]
        failed_runs = [r for r in self.optimization_history if 'error' in r]
        
        if successful_runs:
            avg_duration = sum(r['duration_seconds'] for r in successful_runs) / len(successful_runs)
            avg_tokens = sum(r['metrics']['total_tokens'] for r in successful_runs) / len(successful_runs)
            
            return {
                'total_runs': len(self.optimization_history),
                'successful_runs': len(successful_runs),
                'failed_runs': len(failed_runs),
                'average_duration_seconds': avg_duration,
                'average_tokens': avg_tokens,
                'optimization_levels_used': list(set(r['optimization_level'] for r in successful_runs)),
                'latest_run': successful_runs[-1] if successful_runs else None
            }
        else:
            return {
                'total_runs': len(self.optimization_history),
                'successful_runs': 0,
                'failed_runs': len(failed_runs),
                'message': 'No successful runs yet'
            }

# Global optimization manager instance
optimization_manager = OptimizationManager()