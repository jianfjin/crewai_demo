from .crew import MarketingResearchCrew

def main():
    inputs = {
        "target_audience": "health-conscious millennials",
        "campaign_type": "social media",
        "budget": 75000,
        "duration": "6 months"
    }
    
    agents_config_path = 'src/marketing_research_swarm/config/agents.yaml'
    tasks_config_path = 'src/marketing_research_swarm/config/tasks.yaml'
    
    crew = MarketingResearchCrew(agents_config_path, tasks_config_path)
    result = crew.kickoff(inputs)
    print(result)

if __name__ == "__main__":
    main()