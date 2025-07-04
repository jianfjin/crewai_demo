"""
Enhanced Marketing Research Crew with Fixed Token Tracking
Uses crew.usage_metrics for accurate token tracking.
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

# Load environment variables
load_dotenv()

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def load_settings():
    """Load configuration settings from settings.yaml"""
    settings_path = 'src/marketing_research_swarm/config/settings.yaml'
    try:
        with open(settings_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        return {
            'data_sources': {'default_data_path': 'data/beverage_sales.csv'},
            'llm': {'model': 'gpt-4o-mini', 'temperature': 0.7}
        }

class MarketingResearchCrewWithTracking:
    """Enhanced Marketing Research Crew with accurate token tracking."""
    
    def __init__(self, agents_config_path, tasks_config_path):
        self.agents_config = load_yaml(agents_config_path)
        self.tasks_config = load_yaml(tasks_config_path)
        self.settings = load_settings()
        self.model_name = self.settings.get('llm', {}).get('model', 'gpt-4o-mini')
        
        # Initialize tools
        from marketing_research_swarm.tools import read_file_tool, read_csv_tool
        from marketing_research_swarm.tools.advanced_tools import (
            calculate_roi, analyze_kpis, forecast_sales, plan_budget,
            analyze_brand_performance, calculate_market_share,
            time_series_analysis, cross_sectional_analysis,
            beverage_market_analysis, profitability_analysis,
        )
        from crewai_tools import SerperDevTool, WebsiteSearchTool
        
        # Initialize search tools (with fallback if API keys not available)
        try:
            search_tool = SerperDevTool()
        except:
            search_tool = None
        
        try:
            web_search_tool = WebsiteSearchTool()
        except:
            web_search_tool = None
        
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

    def _create_agent(self, agent_config):
        agent_tools = [self.tools[tool_name] for tool_name in agent_config.get('tools', []) if tool_name in self.tools]
        
        llm = None
        
        # Try to configure LLM based on available API keys and configuration
        if 'OPENAI_API_KEY' in os.environ and os.environ['OPENAI_API_KEY'] != 'your_openai_api_key_here':
            llm = ChatOpenAI(
                model=self.model_name,
                temperature=self.settings.get('llm', {}).get('temperature', 0.7)
            )
        else:
            # Try Ollama as fallback
            try:
                response = requests.get("http://localhost:11434/api/health", timeout=5)
                response.raise_for_status()
                llm = ChatOllama(model="gemma", base_url="http://localhost:11434")
                self.model_name = "gemma"
            except requests.exceptions.RequestException:
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
        crew_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        print(f"Starting token tracking for crew: {crew_id}")
        print(f"Model: {self.model_name}")
        print("-" * 60)
        
        try:
            agents = [self._create_agent(agent_config) for agent_config in self.agents_config.values()]
            tasks = [self._create_task(task_config, agents) for task_config in self.tasks_config.values()]

            # Create and execute the crew
            crew = Crew(
                agents=agents,
                tasks=tasks,
                process=Process.sequential,
                verbose=True
            )
            
            # Execute the crew
            result = crew.kickoff(inputs=inputs)
            
            # Extract token usage from crew.usage_metrics
            usage_metrics = self._extract_crew_usage_metrics(crew)
            
            # Calculate duration
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Generate comprehensive result with token analysis
            combined_result = self._format_results_with_token_analysis(
                result, usage_metrics, duration, crew_id
            )
            
            return combined_result
            
        except Exception as e:
            print(f"Crew execution failed: {str(e)}")
            raise

    def _extract_crew_usage_metrics(self, crew):
        """Extract token usage metrics from crew.usage_metrics if available."""
        try:
            if hasattr(crew, 'usage_metrics') and crew.usage_metrics:
                usage = crew.usage_metrics
                print(f"Found crew usage metrics: {usage}")
                return {
                    'total_tokens': getattr(usage, 'total_tokens', 0),
                    'prompt_tokens': getattr(usage, 'prompt_tokens', 0),
                    'completion_tokens': getattr(usage, 'completion_tokens', 0),
                    'total_cost': getattr(usage, 'total_cost', 0.0),
                    'successful_requests': getattr(usage, 'successful_requests', 0)
                }
            else:
                print("No usage_metrics found on crew object")
                # Try alternative methods to get usage
                if hasattr(crew, '_usage'):
                    print(f"Found crew._usage: {crew._usage}")
                    return crew._usage
                return None
        except Exception as e:
            print(f"Warning: Could not extract crew usage metrics: {e}")
            return None

    def _calculate_cost(self, prompt_tokens, completion_tokens):
        """Calculate cost based on gpt-4o-mini pricing."""
        input_cost_per_token = 0.00000015  # $0.15 per 1M tokens
        output_cost_per_token = 0.0000006  # $0.60 per 1M tokens
        
        input_cost = prompt_tokens * input_cost_per_token
        output_cost = completion_tokens * output_cost_per_token
        total_cost = input_cost + output_cost
        
        return {
            'total_cost': total_cost,
            'input_cost': input_cost,
            'output_cost': output_cost
        }

    def _format_results_with_token_analysis(self, result, usage_metrics, duration, crew_id):
        """Format results with comprehensive token usage analysis."""
        
        # Default values if no usage metrics
        total_tokens = 0
        prompt_tokens = 0
        completion_tokens = 0
        cost_analysis = {'total_cost': 0.0, 'input_cost': 0.0, 'output_cost': 0.0}
        
        if usage_metrics:
            total_tokens = usage_metrics.get('total_tokens', 0)
            prompt_tokens = usage_metrics.get('prompt_tokens', 0)
            completion_tokens = usage_metrics.get('completion_tokens', 0)
            
            # Calculate cost if not provided
            if 'total_cost' in usage_metrics and usage_metrics['total_cost'] > 0:
                cost_analysis['total_cost'] = usage_metrics['total_cost']
            else:
                cost_analysis = self._calculate_cost(prompt_tokens, completion_tokens)
            
            print(f"Token usage extracted: {total_tokens} total tokens")
            print(f"Cost calculated: ${cost_analysis['total_cost']:.4f}")
        
        # Format the combined result
        formatted_result = f"""# Marketing Research Analysis Results

## Task Execution Results

{result}

---

## Token Usage Analysis

### Executive Summary
- **Total Duration**: {duration/60:.2f} minutes
- **Total Tokens**: {total_tokens:,} tokens
- **Model Used**: {self.model_name}
- **Total Cost**: ${cost_analysis['total_cost']:.4f} USD
- **Crew ID**: {crew_id}

### Cost Breakdown
- **Input Tokens**: {prompt_tokens:,} tokens (${cost_analysis['input_cost']:.4f})
- **Output Tokens**: {completion_tokens:,} tokens (${cost_analysis['output_cost']:.4f})
- **Cost per Minute**: ${cost_analysis['total_cost']/(duration/60) if duration > 0 else 0:.4f} USD/min

### Efficiency Metrics
- **Tokens per Second**: {total_tokens/duration if duration > 0 else 0:.2f}
- **Cost per 1K Tokens**: ${cost_analysis['total_cost']/max(total_tokens, 1)*1000:.4f}

### Token Distribution
- **Prompt Tokens**: {prompt_tokens/max(total_tokens, 1)*100:.1f}%
- **Completion Tokens**: {completion_tokens/max(total_tokens, 1)*100:.1f}%

### Performance Insights
- **Analysis Duration**: {duration:.1f} seconds
- **Average Tokens per Minute**: {total_tokens/(duration/60) if duration > 0 else 0:.0f}
- **Token Efficiency**: {'High' if total_tokens > 1000 else 'Moderate' if total_tokens > 0 else 'Low'}

---
*Token analysis generated by Marketing Research Swarm Enhanced Token Tracker*
*Using crew.usage_metrics for accurate tracking*
"""
        
        return formatted_result

# Convenience function to maintain compatibility
def MarketingResearchCrew(agents_config_path, tasks_config_path):
    """Factory function that returns the enhanced crew with tracking."""
    return MarketingResearchCrewWithTracking(agents_config_path, tasks_config_path)