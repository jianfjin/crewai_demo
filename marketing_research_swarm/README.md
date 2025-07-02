# Marketing Research Swarm - AI-Powered Marketing Analytics

A comprehensive marketing research analysis platform powered by CrewAI's multi-agent swarm technology. This project enables marketing teams to leverage AI agents for personalized content creation, campaign optimization, and advanced analytics.

## Features

### 🎯 AI-Driven Content Creation
- **Personalized Campaign Copy**: Generate tailored content for different target audiences
- **Creative Asset Generation**: Create high-quality marketing materials at scale
- **Multi-Channel Content**: Optimize content for various marketing channels

### 📊 Advanced Analytics & Insights
- **ROI Calculation**: Comprehensive return on investment analysis
- **KPI Tracking**: Monitor key performance indicators across campaigns
- **Sales Forecasting**: Predictive analytics for future sales trends
- **Brand Performance**: Track and analyze brand metrics
- **Market Share Analysis**: Competitive positioning insights
- **Budget Planning**: AI-recommended budget allocation strategies
- **Token Usage Tracking**: Real-time LLM cost monitoring and optimization

### 🔬 Research Capabilities
- **Time-Series Analysis**: Trend analysis over time periods
- **Cross-Sectional Analysis**: Comparative analysis across segments
- **Market Research**: Comprehensive market intelligence gathering
- **Audience Segmentation**: AI-powered customer segmentation
- **Competitive Analysis**: Monitor and analyze competitor strategies

## Project Structure

```
marketing_research_swarm/
├── .gitignore
├── knowledge/                 # Knowledge base for agents
├── pyproject.toml            # Project dependencies
├── README.md                 # This file
├── .env                      # Environment variables
└── src/
    └── marketing_research_swarm/
        ├── __init__.py
        ├── main.py           # Main execution script
        ├── crew.py           # CrewAI crew configuration
        ├── tools/            # Custom tools for agents
        │   ├── custom_tool.py
        │   └── __init__.py
        └── config/           # Agent and task configurations
            ├── agents.yaml   # Agent definitions
            └── tasks.yaml    # Task definitions
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd marketing_research_swarm
   ```

2. **Install dependencies**:
   ```bash
   pip install poetry
   poetry install
   ```

3. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Activate the environment**:
   ```bash
   poetry shell
   ```

## Quick Start

1. **Configure your API keys** in the `.env` file:
   - OpenAI API key for LLM capabilities
   - Optional: Other LLM provider keys
   - Market data API keys for research

2. **Run the marketing research swarm**:
   ```bash
   crewai run
   # or
   python src/marketing_research_swarm/main.py
   ```

3. **View comprehensive results** including:
   - Marketing analysis and insights
   - Token usage and cost analysis
   - Performance optimization recommendations

4. **Customize agents and tasks** by editing:
   - `src/marketing_research_swarm/config/agents.yaml`
   - `src/marketing_research_swarm/config/tasks.yaml`

## Agent Swarm Architecture

### Core Agents

1. **Market Research Analyst**: Gathers and analyzes market data, trends, and competitive intelligence
2. **Content Strategist**: Creates personalized content strategies and campaign frameworks
3. **Creative Copywriter**: Generates high-quality, targeted marketing copy and creative assets
4. **Data Analyst**: Performs advanced analytics, ROI calculations, and statistical analysis
5. **Campaign Optimizer**: Optimizes campaign performance and budget allocation
6. **Brand Performance Specialist**: Monitors brand metrics and market positioning

### Workflow

1. **Research Phase**: Market Research Analyst gathers intelligence
2. **Strategy Phase**: Content Strategist develops targeted approaches
3. **Creation Phase**: Creative Copywriter generates personalized content
4. **Analysis Phase**: Data Analyst performs comprehensive analytics
5. **Optimization Phase**: Campaign Optimizer refines strategies
6. **Monitoring Phase**: Brand Performance Specialist tracks results

## Token Usage Tracking

The platform includes comprehensive token usage tracking and cost analysis:

### Real-Time Monitoring
- **Token Consumption**: Track prompt and completion tokens for each LLM call
- **Cost Analysis**: Real-time cost calculation based on model pricing
- **Performance Metrics**: Monitor efficiency and response times
- **Agent Breakdown**: Individual performance analysis for each agent

### Sample Token Report
```
## Token Usage Analysis

### Executive Summary
- Total Duration: 4.25 minutes
- Total Tokens: 15,847 tokens
- Model Used: gpt-4o-mini
- Total Cost: $0.0238 USD
- Tasks Completed: 6

### Agent Performance Breakdown
Market Research Analyst: 4,523 tokens ($0.0068)
Data Analyst: 3,891 tokens ($0.0058)
Content Strategist: 2,456 tokens ($0.0037)

### Optimization Recommendations
1. Consider using gpt-4o-mini for cost optimization
2. Optimize prompts for market_research_analyst
3. Break down complex tasks taking over 2 minutes
```

## Usage Examples

### Generate Personalized Campaign Content with Token Tracking
```python
from marketing_research_swarm.crew_with_tracking import MarketingResearchCrew

crew = MarketingResearchCrew(
    'src/marketing_research_swarm/config/agents.yaml',
    'src/marketing_research_swarm/config/tasks.yaml'
)
result = crew.kickoff({
    "target_audience": "millennials interested in sustainable fashion",
    "campaign_type": "social media",
    "budget": 50000,
    "duration": "3 months"
})
# Result includes both marketing insights AND comprehensive token analysis
```

### Perform ROI Analysis
```python
result = crew.kickoff({
    "analysis_type": "roi_calculation",
    "campaign_data": "path/to/campaign_data.csv",
    "time_period": "Q1 2024"
})
```

### Market Share Analysis
```python
result = crew.kickoff({
    "analysis_type": "market_share",
    "industry": "e-commerce",
    "competitors": ["competitor1", "competitor2", "competitor3"],
    "metrics": ["revenue", "customer_acquisition", "brand_awareness"]
})
```

## Configuration

### Agents Configuration (`config/agents.yaml`)
Define agent roles, goals, backstories, and capabilities.

### Tasks Configuration (`config/tasks.yaml`)
Specify tasks, expected outputs, and agent assignments.

### Custom Tools (`tools/`)
Extend functionality with custom tools for specific marketing research needs.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For questions and support, please open an issue in the GitHub repository.

---

**Powered by CrewAI** - Enabling AI agent collaboration for marketing excellence.