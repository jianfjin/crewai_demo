# Marketing Research Swarm Dashboard Setup Guide

## 🎯 Overview

The Marketing Research Swarm Dashboard is a comprehensive web-based interface that allows users to dynamically create and execute marketing research tasks using the CrewAI framework. Users can select agents, configure parameters, and visualize results through an intuitive interface.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd /workspaces/crewai_demo/marketing_research_swarm
pip install -r requirements_dashboard.txt
```

### 2. Launch Dashboard
```bash
python run_dashboard.py
```

Or manually:
```bash
streamlit run dashboard.py --server.port 8501
```

### 3. Access Dashboard
Open your web browser and navigate to: `http://localhost:8501`

## 📋 Features

### 🤖 **Dynamic Agent Selection**
- Select from available agents in `agents.yaml`
- View agent descriptions and capabilities
- Mix and match agents for custom analysis workflows

### 📝 **Comprehensive Task Configuration**
- **Campaign Basics**: Target audience, campaign type, budget, duration
- **Analysis Focus**: Business objectives, competitive landscape
- **Market Segments**: Geographic and demographic targeting
- **Product Categories**: Beverage categories and brands
- **Key Metrics**: Performance indicators and KPIs
- **Forecasting**: Revenue projections and time periods
- **Brand Metrics**: Awareness, sentiment, market position

### 🚀 **Automated Execution**
- Generates custom YAML task configurations
- Creates unique task files in `config/` directory
- Executes MarketingResearchCrew with selected parameters
- Real-time progress tracking and error handling

### 📊 **Rich Visualizations**
- **Key Metrics Dashboard**: Bar charts and gauges
- **Budget Visualization**: Interactive budget gauges
- **Market Segments**: Pie charts for geographic distribution
- **Product Categories**: Priority-based bar charts
- **Performance Metrics**: Dynamic metric cards

### 💾 **Results Management**
- **Executive Summary**: Parsed key insights
- **Detailed Results**: Full analysis output
- **Download Options**: JSON and text formats
- **Timestamp Tracking**: Execution history

## 🎛️ Configuration Options

### **Basic Parameters**
```python
{
    "target_audience": "health-conscious millennials and premium beverage consumers",
    "campaign_type": "multi-channel global marketing campaign",
    "budget": 250000,
    "duration": "12 months",
    "analysis_focus": "global beverage market performance and brand optimization",
    "business_objective": "Optimize beverage portfolio performance across global markets"
}
```

### **Advanced Parameters**
```python
{
    "key_metrics": ["brand_performance", "category_trends", "profitability_analysis"],
    "market_segments": ["North America", "Europe", "Asia Pacific"],
    "product_categories": ["Cola", "Juice", "Energy", "Sports"],
    "brands": ["Coca-Cola", "Pepsi", "Red Bull"],
    "forecast_periods": 30,
    "expected_revenue": 25000,
    "competitive_analysis": True,
    "market_share_analysis": True
}
```

### **Brand Metrics**
```python
{
    "brand_metrics": {
        "brand_awareness": 75,
        "sentiment_score": 0.6,
        "market_position": "Leader"
    }
}
```

## 🔧 Technical Architecture

### **Dashboard Components**
1. **Agent Selection Sidebar**: Multi-select interface for agent configuration
2. **Task Configuration Tab**: Comprehensive parameter input forms
3. **Execution Tab**: Real-time analysis execution and monitoring
4. **Results Tab**: Visualization and download capabilities

### **Backend Integration**
- **YAML Generation**: Dynamic task configuration creation
- **CrewAI Integration**: Direct integration with MarketingResearchCrew
- **Result Parsing**: Intelligent extraction of metrics and insights
- **Error Handling**: Comprehensive error management and user feedback

### **File Structure**
```
marketing_research_swarm/
├── dashboard.py                 # Main dashboard application
├── run_dashboard.py            # Launch script
├── requirements_dashboard.txt   # Dashboard dependencies
├── src/marketing_research_swarm/
│   ├── config/
│   │   ├── agents.yaml         # Agent configurations
│   │   └── tasks_custom_*.yaml # Generated task files
│   └── crew.py                 # MarketingResearchCrew class
└── reports/                    # Generated analysis reports
```

## 📊 Dashboard Tabs

### **Tab 1: Task Configuration**
- **Campaign Basics**: Core campaign parameters
- **Analysis Focus**: Strategic objectives and focus areas
- **Advanced Parameters**: Detailed market and product configuration
- **Forecasting & Metrics**: Performance targets and measurement
- **Configuration Preview**: JSON preview of all parameters

### **Tab 2: Execute Analysis**
- **Execution Summary**: Overview of selected configuration
- **Real-time Execution**: Progress tracking and status updates
- **Error Handling**: Detailed error reporting and troubleshooting
- **Results Preview**: Initial results display

### **Tab 3: Results & Visualization**
- **Executive Summary**: Key insights and findings
- **Key Metrics**: Performance indicators and KPIs
- **Recommendations**: Strategic recommendations
- **Visualizations**: Interactive charts and graphs
- **Download Options**: Export results in multiple formats

## 🎨 Visualization Types

### **Performance Metrics**
- Bar charts for key performance indicators
- Gauge charts for budget and targets
- Metric cards for quick insights

### **Market Analysis**
- Pie charts for market segment distribution
- Bar charts for product category priorities
- Geographic distribution maps (future enhancement)

### **Trend Analysis**
- Time series charts for forecasting
- Comparative analysis charts
- Performance trend visualizations

## 🔍 Usage Examples

### **Example 1: Global Beverage Campaign**
```python
# Configuration
target_audience = "health-conscious millennials"
campaign_type = "multi-channel global marketing campaign"
budget = 250000
duration = "12 months"
market_segments = ["North America", "Europe", "Asia Pacific"]
brands = ["Coca-Cola", "Pepsi", "Red Bull"]

# Execution
# Select: market_research_analyst, data_analyst, campaign_optimizer
# Execute analysis
# View results with budget allocation and ROI projections
```

### **Example 2: Brand Performance Analysis**
```python
# Configuration
analysis_focus = "brand performance optimization"
key_metrics = ["brand_performance", "market_share", "sentiment_analysis"]
brands = ["Coca-Cola", "Pepsi"]
brand_metrics = {
    "brand_awareness": 80,
    "sentiment_score": 0.7,
    "market_position": "Leader"
}

# Execution
# Select: brand_performance_specialist, market_research_analyst
# Execute analysis
# View brand comparison charts and market position analysis
```

### **Example 3: Sales Forecasting**
```python
# Configuration
forecast_periods = 90
expected_revenue = 500000
product_categories = ["Energy", "Sports", "Enhanced Water"]
competitive_analysis = True

# Execution
# Select: data_analyst, market_research_analyst
# Execute analysis
# View forecasting charts and revenue projections
```

## 🛠️ Troubleshooting

### **Common Issues**

1. **Import Errors**
   ```
   Error: cannot import name 'MarketingResearchCrew'
   ```
   **Solution**: Ensure you're running from the correct directory and all dependencies are installed.

2. **Agent Configuration Not Found**
   ```
   Error: Agents configuration file not found
   ```
   **Solution**: Verify `src/marketing_research_swarm/config/agents.yaml` exists.

3. **Task Execution Fails**
   ```
   Error: Error executing analysis
   ```
   **Solution**: Check agent selection and parameter configuration. Ensure data files exist.

### **Debug Mode**
To enable debug mode, add this to the top of `dashboard.py`:
```python
import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)
```

## 🔮 Future Enhancements

### **Planned Features**
- **Real-time Data Integration**: Live data feeds and APIs
- **Advanced Visualizations**: 3D charts, geographic maps, interactive dashboards
- **Collaboration Features**: Multi-user support, shared workspaces
- **Export Options**: PDF reports, PowerPoint presentations
- **Scheduling**: Automated recurring analyses
- **Templates**: Pre-configured analysis templates
- **Integration**: Slack, Teams, email notifications

### **Technical Improvements**
- **Performance Optimization**: Caching, lazy loading
- **Mobile Responsiveness**: Mobile-friendly interface
- **Authentication**: User management and access control
- **Database Integration**: Persistent storage for configurations and results
- **API Development**: RESTful API for programmatic access

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the generated error messages in the dashboard
3. Verify all configuration files are present and valid
4. Ensure all dependencies are installed correctly

## 🎉 Getting Started

1. **Launch the dashboard**: `python run_dashboard.py`
2. **Select agents**: Choose 2-3 agents from the sidebar
3. **Configure parameters**: Fill in the task configuration form
4. **Execute analysis**: Click the "Execute Analysis" button
5. **View results**: Explore visualizations and download reports

The dashboard provides a powerful, user-friendly interface for creating sophisticated marketing research analyses with minimal technical knowledge required!