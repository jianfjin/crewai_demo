# Marketing Research Swarm - Analysis Guide

## Overview
This guide explains how to run the Marketing Research Crew analysis on the beverage sales data and generate comprehensive marketing insights.

## Quick Start

### 1. Run the Complete Analysis
```bash
cd marketing_research_swarm
python run_analysis.py
```

### 2. Test Individual Components
```bash
python test_analysis.py
```

### 3. Run from Source
```bash
cd src
python -m marketing_research_swarm.main
```

## What the Analysis Does

### Data Analysis
- **Time Series Analysis**: Analyzes sales trends over time
- **Cross-Sectional Analysis**: Compares performance across regions and products
- **Sales Forecasting**: Predicts future sales based on historical data
- **Performance Metrics**: Calculates key performance indicators

### Marketing Insights
- **Market Research**: Comprehensive beverage market analysis
- **Content Strategy**: Targeted content recommendations for health-conscious millennials
- **Creative Copy**: Sample marketing copy for campaigns
- **Campaign Optimization**: Budget allocation and channel recommendations
- **Brand Performance**: Brand health and positioning analysis

### Output
The analysis generates:
1. **Console Output**: Real-time analysis results
2. **Report File**: Detailed markdown report saved to `reports/` directory
3. **Recommendations**: Actionable marketing strategies

## Sample Data
The analysis uses `data/beverage_sales.csv` which contains:
- **Date**: Sales date (2024-01-01 to 2024-01-10)
- **Region**: North or South
- **Product**: Green Tea or Latte
- **Sales**: Sales amount in dollars

## Key Findings from Sample Data
Based on the beverage sales data, the analysis typically reveals:

### Regional Performance
- **South Region**: Consistently outperforms North region
- **Latte Sales**: Stronger in South region
- **Green Tea Sales**: More balanced across regions

### Product Performance
- **Latte**: Higher overall sales volume
- **Green Tea**: Steady growth trend
- **Market Opportunity**: Potential to grow Green Tea in South region

### Trends
- **Overall Growth**: Positive sales trend over the analysis period
- **Regional Gaps**: Opportunity to improve North region performance
- **Product Mix**: Latte dominance suggests market preference

## Customization

### Modify Analysis Parameters
Edit `src/marketing_research_swarm/main.py` to change:
- Target audience
- Campaign budget
- Analysis focus
- Campaign goals

### Add New Data
Replace `data/beverage_sales.csv` with your own data (maintain same format)

### Customize Agents
Edit configuration files:
- `src/marketing_research_swarm/config/agents.yaml`
- `src/marketing_research_swarm/config/tasks.yaml`

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure all dependencies are installed
2. **LLM Configuration**: Set up OpenAI API key or run local Ollama
3. **Data Path**: Verify `data/beverage_sales.csv` exists
4. **Permissions**: Ensure write permissions for `reports/` directory

### Dependencies
```bash
pip install crewai pandas numpy matplotlib seaborn plotly scikit-learn statsmodels
```

### Environment Setup
Create `.env` file with:
```
OPENAI_API_KEY=your_openai_api_key_here
SERPER_API_KEY=your_serper_api_key_here  # Optional for web search
```

## Advanced Usage

### Run Specific Analysis Types
```python
from marketing_research_swarm.main import run_specific_analysis

# Sales forecasting focus
run_specific_analysis("sales_forecast")

# ROI analysis focus
run_specific_analysis("roi_analysis")

# Brand performance focus
run_specific_analysis("brand_performance")
```

### Custom Analysis
```python
from marketing_research_swarm.crew import MarketingResearchCrew

inputs = {
    "target_audience": "your_target_audience",
    "budget": 50000,
    "data_file_path": "your_data.csv"
}

crew = MarketingResearchCrew(
    'src/marketing_research_swarm/config/agents.yaml',
    'src/marketing_research_swarm/config/tasks.yaml'
)
result = crew.kickoff(inputs)
```

## Expected Output

The analysis will generate insights such as:
- Sales trend analysis with growth rates
- Regional performance comparison
- Product performance metrics
- Sales forecasts for future periods
- Marketing strategy recommendations
- Budget allocation suggestions
- Content strategy for target audience
- Campaign optimization recommendations

## Next Steps

After running the analysis:
1. Review the generated report in `reports/` directory
2. Implement recommended marketing strategies
3. Monitor campaign performance
4. Re-run analysis with new data to track improvements
5. Adjust strategies based on results