import yaml
from crewai import Agent, Task, Crew, Process
from .tools import read_file_tool, read_csv_tool

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

class MarketingResearchCrew:
    def __init__(self, agents_config_path, tasks_config_path):
        self.agents_config = load_yaml(agents_config_path)
        self.tasks_config = load_yaml(tasks_config_path)
        self.tools = [read_file_tool, read_csv_tool]

    def _create_agent(self, agent_config):
        return Agent(
            role=agent_config['role'],
            goal=agent_config['goal'],
            backstory=agent_config['backstory'],
            llm=agent_config.get('llm'),
            allow_delegation=agent_config.get('allow_delegation', False),
            tools=self.tools if 'tools' in agent_config else []
        )

    def _create_task(self, task_config, agents):
        agent = next((a for a in agents if a.role == task_config['agent']), None)
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
