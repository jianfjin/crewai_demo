# LangSmith Monitoring Setup Guide

## Overview
This guide helps you set up LangSmith monitoring for the Marketing Research Dashboard to track analysis runs in real-time.

## Prerequisites
- LANGCHAIN_API_KEY from LangSmith (already configured in your .env)
- Python packages for LangSmith integration

## Installation

### 1. Install Required Packages
```bash
pip install langsmith langchain python-dotenv
```

### 2. Verify Environment Variables
Make sure your `.env` file contains:
```
LANGCHAIN_API_KEY=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
```

### 3. Test LangSmith Connection
```python
from langsmith import Client
import os
from dotenv import load_dotenv

load_dotenv()
client = Client(api_key=os.getenv("LANGCHAIN_API_KEY"))
print("✅ LangSmith connection successful!")
```

## Features Enabled

### 🔍 Real-time Monitoring
- **Live Progress Tracking**: See analysis progress with LangSmith status updates
- **Run History**: View recent analysis runs directly in the dashboard
- **Direct Links**: Click to view detailed traces in LangSmith web interface

### 📊 Enhanced Analytics
- **Token Usage Tracking**: Monitor token consumption per agent and task
- **Performance Metrics**: Track execution time and efficiency
- **Error Debugging**: Detailed error traces for failed runs

### 🚀 Workflow Integration
- **Automatic Tracing**: All LangGraph workflows are automatically traced
- **Metadata Capture**: Analysis configuration and optimization settings recorded
- **Project Organization**: Runs organized under "marketing-research-dashboard" project

## Dashboard Features

### Status Indicators
- **🟢 Monitoring**: LangSmith is active and tracking runs
- **🔴 Disabled**: LangSmith not available (missing API key or packages)

### Monitoring Section
- **🔄 Refresh Runs**: Update run data from LangSmith
- **Project Name**: Configure which LangSmith project to monitor
- **Run Details**: View status, duration, tokens, and direct links

### Progress Tracking
During analysis execution, you'll see:
1. **🔧 Initializing workflow** → Creating trace session
2. **⚡ Applying optimization** → Monitoring agent initialization
3. **🤖 Executing agents** → Tracking execution and token usage
4. **📊 Processing results** → Recording performance metrics
5. **✅ Finalizing analysis** → Analysis trace completed

## LangSmith Web Interface

### Accessing Your Runs
1. **Direct Links**: Click "🔗 View in LangSmith" from dashboard
2. **Project URL**: https://smith.langchain.com/o/default/projects/p/marketing-research-dashboard
3. **Manual Navigation**: Go to LangSmith → Projects → marketing-research-dashboard

### What You'll See
- **Run Timeline**: Chronological view of all analysis runs
- **Token Usage**: Detailed breakdown by agent and task
- **Performance Metrics**: Execution time, success rates, error rates
- **Trace Details**: Step-by-step execution flow with inputs/outputs

## Troubleshooting

### Common Issues

#### "LangSmith not available" Warning
```bash
# Install missing packages
pip install langsmith langchain python-dotenv

# Verify installation
python -c "import langsmith; print('✅ LangSmith installed')"
```

#### "LANGCHAIN_API_KEY not found" Warning
```bash
# Check .env file exists and contains API key
cat .env | grep LANGCHAIN_API_KEY

# Verify environment loading
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(os.getenv('LANGCHAIN_API_KEY'))"
```

#### No Runs Appearing in Dashboard
1. **Check Project Name**: Ensure "marketing-research-dashboard" is correct
2. **Run Analysis**: LangSmith only shows runs after you execute analysis
3. **API Permissions**: Verify your API key has read access to projects

### Debug Commands
```bash
# Test full LangSmith integration
cd marketing_research_swarm
python -c "
from langsmith import Client
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('LANGCHAIN_API_KEY')
if api_key:
    client = Client(api_key=api_key)
    runs = list(client.list_runs(project_name='marketing-research-dashboard', limit=5))
    print(f'✅ Found {len(runs)} runs in project')
else:
    print('❌ No API key found')
"
```

## Best Practices

### Project Organization
- Use descriptive project names for different analysis types
- Archive old projects to keep interface clean
- Use tags to categorize different optimization levels

### Performance Monitoring
- Monitor token usage trends over time
- Compare optimization levels effectiveness
- Track analysis duration improvements

### Error Debugging
- Check LangSmith traces for failed runs
- Use error details to improve configurations
- Monitor agent-specific failure patterns

## Advanced Configuration

### Custom Project Names
```python
# In dashboard, change project name to organize runs
project_name = "marketing-analysis-production"
project_name = "marketing-analysis-testing"
```

### Environment Variables
```bash
# Optional: Set default project
export LANGCHAIN_PROJECT="marketing-research-dashboard"

# Optional: Enable detailed tracing
export LANGCHAIN_TRACING_V2="true"
```

## Support

### Resources
- **LangSmith Docs**: https://docs.smith.langchain.com/
- **API Reference**: https://api.smith.langchain.com/
- **Community**: https://github.com/langchain-ai/langsmith-sdk

### Getting Help
1. Check LangSmith status page for service issues
2. Verify API key permissions in LangSmith settings
3. Review dashboard logs for detailed error messages