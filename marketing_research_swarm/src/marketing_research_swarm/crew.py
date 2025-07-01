import yaml
import os
import requests
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
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
)

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

class MarketingResearchCrew:
    def __init__(self, agents_config_path, tasks_config_path):
        self.agents_config = load_yaml(agents_config_path)
        self.tasks_config = load_yaml(tasks_config_path)
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
        }

    def _create_agent(self, agent_config):
        agent_tools = [self.tools[tool_name] for tool_name in agent_config.get('tools', []) if tool_name in self.tools]
        
        llm = None
        llm_config = agent_config.get('llm')
        if llm_config and llm_config.get('model'):
            if 'OPENAI_API_KEY' in os.environ and os.environ['OPENAI_API_KEY'] != 'your_openai_api_key_here':
                llm = ChatOpenAI(
                    model_name=llm_config['model'],
                    temperature=llm_config.get('temperature', 0.7)
                )
            else:
                try:
                    response = requests.get("http://localhost:11434/api/health")
                    response.raise_for_status()
                    llm = ChatOllama(model="gemma", base_url="http://localhost:11434")
                except requests.exceptions.RequestException as e:
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
        agents = [self._create_agent(agent_config) for agent_config in self.agents_config.values()]
        tasks = [self._create_task(task_config, agents) for task_config in self.tasks_config.values()]

        crew = Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )
        return crew.kickoff(inputs=inputs)
