"""
Enhanced Marketing Research Crew with Token Tracking
Extends the base crew with comprehensive token usage monitoring and analysis.
"""

import yaml
import os
import requests
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import LLMResult

# Load environment variables
load_dotenv()
from .tools import read_file_tool, read_csv_tool
from .tools.advanced_tools import (
    calculate_roi,
    analyze_kpis,
    forecast_sales,
    plan_budget,
    analyze_brand_performance,
    calculate_market_share,
    time_series_analysis,
    cross_sectional_analysis,
    beverage_market_analysis,
    profitability_analysis,
)
from crewai_tools import SerperDevTool, WebsiteSearchTool, CodeInterpreterTool
from .utils.token_tracker import get_token_tracker, TokenAnalyzer, reset_token_tracker

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

class TokenTrackingCallback(BaseCallbackHandler):
    """Callback handler to track token usage during LLM calls."""
    
    def __init__(self, tracker, task_name: str = "unknown"):
        self.tracker = tracker
        self.task_name = task_name
        self.current_prompt = ""
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: list, **kwargs) -> None:
        """Called when LLM starts processing."""
        if prompts:
            self.current_prompt = prompts[0] if isinstance(prompts[0], str) else str(prompts[0])
    
    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """Called when LLM finishes processing."""
        if response.generations and response.generations[0]:
            response_text = response.generations[0][0].text
            
            # Extract token usage from response if available
            usage_data = None
            if hasattr(response, 'llm_output') and response.llm_output:
                usage_data = response.llm_output.get('token_usage')
            
            # Record the usage
            self.tracker.record_llm_usage(
                prompt=self.current_prompt,
                response=response_text,
                actual_usage=usage_data
            )

class MarketingResearchCrewWithTracking:
    """Enhanced Marketing Research Crew with comprehensive token tracking."""
    
    def __init__(self, agents_config_path, tasks_config_path):
        self.agents_config = load_yaml(agents_config_path)
        self.tasks_config = load_yaml(tasks_config_path)
        self.model_name = "gpt-4o-mini"  # Default model
        
        # Reset and get token tracker
        reset_token_tracker()
        self.token_tracker = get_token_tracker(self.model_name)
        
        # Initialize search tools (with fallback if API keys not available)
        try:
            search_tool = SerperDevTool()
        except:
            search_tool = None
        
        try:
            web_search_tool = WebsiteSearchTool()
        except:
            web_search_tool = None
            
        # Disable Code Interpreter to prevent string formatting issues
        python_repl_tool = None
        
        self.tools = {
            "read_file_tool": read_file_tool,
            "read_csv_tool": read_csv_tool,
            "calculate_roi": calculate_roi,
            "analyze_kpis": analyze_kpis,
            "forecast_sales": forecast_sales,
            "plan_budget": plan_budget,
            "analyze_brand_performance": analyze_brand_performance,
            "calculate_market_share": calculate_market_share,
            "time_series_analysis": time_series_analysis,
            "cross_sectional_analysis": cross_sectional_analysis,
            "beverage_market_analysis": beverage_market_analysis,
            "profitability_analysis": profitability_analysis,
        }
        
        # Add optional tools if available
        if search_tool:
            self.tools["search"] = search_tool
        if web_search_tool:
            self.tools["web_search"] = web_search_tool
        if python_repl_tool:
            self.tools["python_repl"] = python_repl_tool

    def _create_agent(self, agent_config):
        agent_tools = [self.tools[tool_name] for tool_name in agent_config.get('tools', []) if tool_name in self.tools]
        
        llm = None
        llm_config = agent_config.get('llm')
        
        # Try to configure LLM based on available API keys and configuration
        if 'OPENAI_API_KEY' in os.environ and os.environ['OPENAI_API_KEY'] != 'your_openai_api_key_here':
            # Create callback for this agent
            callback = TokenTrackingCallback(self.token_tracker, agent_config.get('role', 'unknown'))
            
            llm = ChatOpenAI(
                model=self.model_name,
                temperature=0.7,
                callbacks=[callback]
            )
        else:
            # Try Ollama as fallback
            try:
                response = requests.get("http://localhost:11434/api/health", timeout=5)
                response.raise_for_status()
                llm = ChatOllama(model="gemma", base_url="http://localhost:11434")
                self.model_name = "gemma"  # Update model name for tracking
            except requests.exceptions.RequestException:
                # If no LLM is available, the agent will use default behavior
                print(f"Warning: No LLM configured for agent {agent_config.get('role', 'unknown')}")
                pass

        return Agent(
            role=agent_config['role'],
            goal=agent_config['goal'],
            backstory=agent_config['backstory'],
            llm=llm,
            allow_delegation=agent_config.get('allow_delegation', False),
            tools=agent_tools
        )

    def _create_task(self, task_config, agents):
        agent_role = task_config['agent'].strip()
        agent = next((a for a in agents if a.role.strip() == agent_role), None)
        return Task(
            description=task_config['description'],
            expected_output=task_config['expected_output'],
            agent=agent
        )

    def kickoff(self, inputs):
        """Execute the crew with comprehensive token tracking."""
        # Start crew tracking
        crew_id = str(uuid.uuid4())
        crew_usage = self.token_tracker.start_crew_tracking(crew_id)
        
        print(f"🔍 Starting token tracking for crew: {crew_id}")
        print(f"📊 Model: {self.model_name}")
        print("-" * 60)
        
        try:
            agents = [self._create_agent(agent_config) for agent_config in self.agents_config.values()]
            tasks = [self._create_task(task_config, agents) for task_config in self.tasks_config.values()]

            # Execute tasks with individual tracking
            results = []
            for i, task in enumerate(tasks):
                task_name = list(self.tasks_config.keys())[i]
                agent_name = task.agent.role if task.agent else "unknown"
                
                print(f"📋 Starting task: {task_name} (Agent: {agent_name})")
                
                # Start task tracking
                task_usage = self.token_tracker.start_task_tracking(task_name, agent_name)
                
                try:
                    # Execute the task
                    crew = Crew(
                        agents=[task.agent],
                        tasks=[task],
                        process=Process.sequential,
                        verbose=True
                    )
                    
                    task_result = crew.kickoff(inputs=inputs)
                    results.append(task_result)
                    
                    # Complete task tracking
                    self.token_tracker.complete_task(status="completed")
                    
                    print(f"✅ Completed task: {task_name}")
                    if self.token_tracker.current_task:
                        print(f"   Tokens used: {task_usage.token_usage.total_tokens}")
                        print(f"   Duration: {task_usage.duration_seconds:.2f}s")
                    
                except Exception as e:
                    # Complete task tracking with error
                    self.token_tracker.complete_task(status="failed", error_message=str(e))
                    print(f"❌ Failed task: {task_name} - {str(e)}")
                    results.append(f"Task {task_name} failed: {str(e)}")
            
            # Complete crew tracking
            final_usage = self.token_tracker.complete_crew_tracking()
            
            # Generate comprehensive result with token analysis
            combined_result = self._format_results_with_token_analysis(results, final_usage)
            
            return combined_result
            
        except Exception as e:
            # Complete crew tracking with error
            self.token_tracker.complete_crew_tracking()
            print(f"❌ Crew execution failed: {str(e)}")
            raise

    def _format_results_with_token_analysis(self, task_results, crew_usage):
        """Format results with comprehensive token usage analysis."""
        
        # Analyze token usage
        analysis = TokenAnalyzer.analyze_crew_usage(crew_usage)
        
        # Format the combined result
        formatted_result = f"""# Marketing Research Analysis Results

## Task Execution Results

"""
        
        # Add individual task results
        task_names = list(self.tasks_config.keys())
        for i, result in enumerate(task_results):
            task_name = task_names[i] if i < len(task_names) else f"Task {i+1}"
            formatted_result += f"""### {task_name.replace('_', ' ').title()}

{result}

---

"""
        
        # Add comprehensive token usage analysis
        formatted_result += f"""
## Token Usage Analysis

### Executive Summary
- **Total Duration**: {analysis['summary']['total_duration_minutes']:.2f} minutes
- **Total Tokens**: {analysis['summary']['total_tokens']:,} tokens
- **Model Used**: {analysis['summary']['model_used']}
- **Total Cost**: ${analysis['summary']['total_cost']['total_cost']:.4f} USD
- **Tasks Completed**: {analysis['summary']['task_count']}

### Cost Breakdown
- **Input Tokens**: {analysis['summary']['prompt_tokens']:,} tokens (${analysis['summary']['total_cost']['input_cost']:.4f})
- **Output Tokens**: {analysis['summary']['completion_tokens']:,} tokens (${analysis['summary']['total_cost']['output_cost']:.4f})
- **Cost per Minute**: ${analysis['efficiency_metrics']['cost_per_minute']:.4f} USD/min

### Efficiency Metrics
- **Tokens per Second**: {analysis['efficiency_metrics']['tokens_per_second']:.2f}
- **Average Tokens per Task**: {analysis['efficiency_metrics']['avg_tokens_per_task']:.0f}

### Agent Performance Breakdown

"""
        
        # Add agent-specific analysis
        for agent_name, stats in analysis['agent_breakdown'].items():
            formatted_result += f"""#### {agent_name.replace('_', ' ').title()}
- **Total Tokens**: {stats['total_tokens']:,}
- **Cost**: ${stats['total_cost']:.4f} USD
- **Duration**: {stats['total_duration']:.1f} seconds
- **LLM Calls**: {stats['llm_calls']}
- **Tool Calls**: {stats['tool_calls']}
- **Efficiency**: {stats['total_tokens']/max(stats['total_duration'], 1):.1f} tokens/sec

"""
        
        # Add task-specific analysis
        formatted_result += """### Task Performance Details

| Task | Agent | Duration (s) | Tokens | Cost ($) | Efficiency | Status |
|------|-------|-------------|---------|----------|------------|---------|
"""
        
        for task in analysis['task_breakdown']:
            formatted_result += f"| {task['task_name']} | {task['agent_name']} | {task['duration_seconds']:.1f} | {task['token_usage']['total_tokens']:,} | {task['cost_analysis']['total_cost']:.4f} | {task['efficiency_metrics']['tokens_per_second']:.1f} t/s | {task['status']} |\n"
        
        # Add optimization recommendations
        if analysis['recommendations']:
            formatted_result += f"""
### Optimization Recommendations

"""
            for i, rec in enumerate(analysis['recommendations'], 1):
                formatted_result += f"{i}. {rec}\n"
        
        # Add detailed cost analysis
        formatted_result += f"""
### Detailed Cost Analysis

**Model Pricing** ({analysis['summary']['model_used']}):
- Input tokens: ${analysis['summary']['total_cost']['input_cost']:.6f} per token
- Output tokens: ${analysis['summary']['total_cost']['output_cost']:.6f} per token

**Token Distribution**:
- Prompt tokens: {(analysis['summary']['prompt_tokens']/analysis['summary']['total_tokens']*100):.1f}%
- Completion tokens: {(analysis['summary']['completion_tokens']/analysis['summary']['total_tokens']*100):.1f}%

**Cost Efficiency**:
- Cost per 1K tokens: ${(analysis['summary']['total_cost']['total_cost']/analysis['summary']['total_tokens']*1000):.4f}
- Tokens per dollar: {(analysis['summary']['total_tokens']/analysis['summary']['total_cost']['total_cost']):.0f}

### Performance Insights

**Fastest Task**: {min(analysis['task_breakdown'], key=lambda x: x['duration_seconds'])['task_name']} ({min(analysis['task_breakdown'], key=lambda x: x['duration_seconds'])['duration_seconds']:.1f}s)
**Most Token-Intensive**: {max(analysis['task_breakdown'], key=lambda x: x['token_usage']['total_tokens'])['task_name']} ({max(analysis['task_breakdown'], key=lambda x: x['token_usage']['total_tokens'])['token_usage']['total_tokens']:,} tokens)
**Most Cost-Effective**: {min(analysis['task_breakdown'], key=lambda x: x['cost_analysis']['total_cost'])['task_name']} (${min(analysis['task_breakdown'], key=lambda x: x['cost_analysis']['total_cost'])['cost_analysis']['total_cost']:.4f})

---
*Token analysis generated by Marketing Research Swarm Token Tracker*
"""
        
        return formatted_result

# Convenience function to maintain compatibility
def MarketingResearchCrew(agents_config_path, tasks_config_path):
    """Factory function that returns the enhanced crew with tracking."""
    return MarketingResearchCrewWithTracking(agents_config_path, tasks_config_path)