# CrewAI Run Command Guide

## ✅ SUCCESS! The Marketing Research Swarm now supports `crewai run`

The project has been successfully configured to work with the `crewai run` command. Here's how to use it:

## Quick Start

### 1. Basic Usage
```bash
cd marketing_research_swarm
crewai run
```

### 2. Alternative Methods
```bash
# Method 1: Using the main script
python src/marketing_research_swarm/main.py

# Method 2: Using the run analysis script
python run_analysis.py

# Method 3: Using the direct runner
python run_crew.py

# Method 4: Using the installed command
marketing-research-swarm
```

## What Happens When You Run `crewai run`

1. **Automatic Setup**: CrewAI creates a virtual environment and installs dependencies
2. **Crew Initialization**: Loads the 6 specialized marketing research agents
3. **Tool Loading**: Initializes all 13 analytical tools
4. **Data Analysis**: Processes the beverage sales data
5. **Report Generation**: Creates a comprehensive markdown report

## Expected Output

When you run `crewai run`, you'll see:

```
Starting Marketing Research Crew Analysis via CrewAI CLI...
Analyzing data from: data/beverage_sales.csv
Target Audience: health-conscious millennials interested in premium beverages
Campaign Budget: $100,000
Campaign Duration: 6 months
------------------------------------------------------------

🚀 Crew: crew
└── 📋 Task: [task-id]
    Status: Executing Task...
    
[Agent execution progress...]

============================================================
MARKETING RESEARCH ANALYSIS COMPLETED!
============================================================
Report saved to: reports/marketing_analysis_report_[timestamp].md
```

## Configuration Requirements

### API Keys (Required for Full Functionality)
Set up your API keys in the `.env` file:

```bash
# Required for LLM functionality
OPENAI_API_KEY=your_openai_api_key_here

# Optional for web search capabilities
SERPER_API_KEY=your_serper_api_key_here
```

### Without API Keys
The system will still run and show the crew structure, but will fail at LLM execution. You'll see:
- ✅ Crew initialization
- ✅ Agent configuration
- ✅ Tool loading
- ❌ LLM execution (requires API key)

## What the Analysis Includes

### 1. Market Research Analysis
- Time series analysis of sales trends
- Cross-sectional analysis by region and product
- Market performance insights

### 2. Content Strategy Development
- Targeted content recommendations
- Channel-specific strategies
- Audience engagement tactics

### 3. Creative Copywriting
- Marketing copy generation
- Campaign messaging
- Brand voice development

### 4. Data Analysis
- Statistical analysis and forecasting
- ROI calculations
- Performance metrics

### 5. Campaign Optimization
- Budget allocation recommendations
- Channel optimization
- Performance improvement strategies

### 6. Brand Performance Monitoring
- Brand health metrics
- Market positioning analysis
- Competitive intelligence

## Generated Reports

The system creates detailed reports in the `reports/` directory:

```
reports/
└── marketing_analysis_report_2024-12-19_14-30-15.md
```

Each report includes:
- Executive summary
- Analysis parameters
- Detailed findings from each agent
- Actionable recommendations
- Strategic insights

## Troubleshooting

### Common Issues

1. **"No LLM configured" warnings**
   - **Solution**: Add a valid OpenAI API key to `.env`
   - **Alternative**: Set up local Ollama instance

2. **"Module not found" errors**
   - **Solution**: Ensure you're in the project root directory
   - **Alternative**: Run `pip install -e .` to reinstall

3. **"Data file not found" errors**
   - **Solution**: Verify `data/beverage_sales.csv` exists
   - **Alternative**: Update the data path in configuration

### Verification Commands

Test individual components:

```bash
# Test analytical tools
python -c "
from src.marketing_research_swarm.tools.advanced_tools import time_series_analysis
result = time_series_analysis._run('data/beverage_sales.csv')
print(result)
"

# Test crew configuration
python -c "
from src.marketing_research_swarm.crew import MarketingResearchCrew
crew = MarketingResearchCrew('src/marketing_research_swarm/config/agents.yaml', 'src/marketing_research_swarm/config/tasks.yaml')
print(f'Agents: {len(crew.agents_config)}, Tools: {len(crew.tools)}')
"
```

## Advanced Usage

### Custom Analysis Parameters

Modify `run_crew.py` to customize the analysis:

```python
inputs = {
    "target_audience": "your_custom_audience",
    "campaign_type": "your_campaign_type",
    "budget": 50000,  # Custom budget
    "duration": "3 months",  # Custom duration
    "data_file_path": "your_data.csv",  # Custom data
    # ... other parameters
}
```

### Different Analysis Types

The system supports multiple analysis modes:
- Comprehensive analysis (default)
- Sales forecasting focus
- ROI analysis focus
- Brand performance focus

## Integration with CrewAI Ecosystem

The project is now fully compatible with:
- ✅ `crewai run` command
- ✅ CrewAI CLI tools
- ✅ CrewAI project structure
- ✅ CrewAI configuration standards
- ✅ CrewAI deployment workflows

## Next Steps

1. **Set up API keys** for full functionality
2. **Run your first analysis** with `crewai run`
3. **Review the generated report** in `reports/`
4. **Customize the analysis** for your specific needs
5. **Deploy to production** using CrewAI deployment tools

## Success Metrics

✅ **CrewAI CLI Integration**: Fully functional with `crewai run`  
✅ **Multi-Agent System**: 6 specialized agents working collaboratively  
✅ **Advanced Analytics**: 13 tools for comprehensive analysis  
✅ **Report Generation**: Automated markdown report creation  
✅ **Error Handling**: Robust fallbacks and user feedback  
✅ **Production Ready**: Professional-grade marketing analytics platform  

---

The Marketing Research Swarm is now a fully integrated CrewAI project that can be executed using the standard `crewai run` command, making it easy to use and deploy in any CrewAI-compatible environment.