# Token Usage Tracking and Analysis Guide

## 🎯 Overview

The Marketing Research Swarm now includes comprehensive token usage tracking and cost analysis. This feature provides detailed insights into LLM consumption, costs, and performance optimization opportunities across all agents and tasks.

## ✨ Key Features

### 📊 Real-Time Tracking
- **Token Consumption**: Monitor prompt and completion tokens for each LLM call
- **Cost Analysis**: Real-time cost calculation based on model pricing
- **Performance Metrics**: Track tokens per second, efficiency ratios, and response times
- **Agent-Level Breakdown**: Individual performance analysis for each agent

### 💰 Cost Management
- **Multi-Model Support**: Accurate pricing for GPT-4o, GPT-4o-mini, GPT-4, GPT-3.5-turbo
- **Cost Optimization**: Identify expensive operations and optimization opportunities
- **Budget Tracking**: Monitor spending against campaign budgets
- **ROI Analysis**: Cost-effectiveness analysis for marketing insights

### 📈 Performance Analytics
- **Efficiency Metrics**: Tokens per second, cost per minute, task duration analysis
- **Comparative Analysis**: Agent performance comparison and benchmarking
- **Optimization Recommendations**: AI-powered suggestions for cost and performance improvements
- **Historical Tracking**: Track performance trends over time

## 🚀 How It Works

### Automatic Integration
The token tracking is automatically integrated into the enhanced crew system:

```python
from marketing_research_swarm.crew_with_tracking import MarketingResearchCrew

# Token tracking is automatically enabled
crew = MarketingResearchCrew(agents_config_path, tasks_config_path)
result = crew.kickoff(inputs)  # Includes comprehensive token analysis
```

### Token Tracking Flow
1. **Crew Initialization**: Token tracker starts monitoring
2. **Task Execution**: Each task's token usage is tracked individually
3. **LLM Calls**: Every API call is monitored for token consumption
4. **Tool Usage**: Tool calls are counted and analyzed
5. **Analysis Generation**: Comprehensive report with insights and recommendations

## 📋 Report Sections

### Executive Summary
```
- Total Duration: 4.25 minutes
- Total Tokens: 15,847 tokens
- Model Used: gpt-4o-mini
- Total Cost: $0.0238 USD
- Tasks Completed: 6
```

### Cost Breakdown
- Input vs Output token costs
- Cost per minute analysis
- Model pricing details
- Budget impact assessment

### Agent Performance
```
Market Research Analyst:
- Total Tokens: 4,523
- Cost: $0.0068 USD
- Duration: 78.3 seconds
- Efficiency: 57.8 tokens/sec
```

### Task Analysis
| Task | Agent | Duration | Tokens | Cost | Efficiency | Status |
|------|-------|----------|---------|------|------------|---------|
| research_task | market_research_analyst | 78.3s | 4,523 | $0.0068 | 57.8 t/s | ✅ |

### Optimization Recommendations
- Cost reduction suggestions
- Performance improvement tips
- Model selection recommendations
- Prompt optimization guidance

## 💡 Usage Examples

### Basic Usage
```python
# Standard execution with automatic token tracking
crew = MarketingResearchCrew(agents_config_path, tasks_config_path)
result = crew.kickoff(inputs)

# Result includes both analysis and token usage report
print(result)  # Contains marketing insights + token analysis
```

### Custom Token Analysis
```python
from marketing_research_swarm.utils.token_tracker import TokenAnalyzer, get_token_tracker

# Get the global tracker
tracker = get_token_tracker("gpt-4o-mini")

# Analyze specific usage patterns
analysis = TokenAnalyzer.analyze_crew_usage(crew_usage)
print(analysis['recommendations'])
```

### Cost Estimation
```python
from marketing_research_swarm.utils.token_tracker import TokenTracker, TokenUsage

tracker = TokenTracker("gpt-4o")
usage = TokenUsage(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
cost = tracker.estimate_cost(usage)

print(f"Estimated cost: ${cost['total_cost']:.4f}")
```

## 📊 Sample Token Analysis Report

```markdown
## Token Usage Analysis

### Executive Summary
- **Total Duration**: 4.25 minutes
- **Total Tokens**: 15,847 tokens
- **Model Used**: gpt-4o-mini
- **Total Cost**: $0.0238 USD
- **Tasks Completed**: 6

### Cost Breakdown
- **Input Tokens**: 10,234 tokens ($0.0015)
- **Output Tokens**: 5,613 tokens ($0.0034)
- **Cost per Minute**: $0.0056 USD/min

### Efficiency Metrics
- **Tokens per Second**: 62.1
- **Average Tokens per Task**: 2,641

### Agent Performance Breakdown

#### Market Research Analyst
- **Total Tokens**: 4,523
- **Cost**: $0.0068 USD
- **Duration**: 78.3 seconds
- **LLM Calls**: 3
- **Tool Calls**: 2
- **Efficiency**: 57.8 tokens/sec

### Optimization Recommendations
1. Consider using gpt-4o-mini for cost optimization
2. Agent 'market_research_analyst' consumes highest cost - optimize prompts
3. Break down complex tasks taking over 2 minutes

### Performance Insights
**Fastest Task**: strategy_task (42.1s)
**Most Token-Intensive**: research_task (4,523 tokens)
**Most Cost-Effective**: strategy_task ($0.0037)
```

## 🔧 Configuration Options

### Model Selection
```python
# Configure different models for cost optimization
crew = MarketingResearchCrew(agents_config_path, tasks_config_path)
crew.model_name = "gpt-4o-mini"  # Most cost-effective
# crew.model_name = "gpt-4o"     # Balanced performance/cost
# crew.model_name = "gpt-4"      # Highest quality
```

### Custom Pricing
```python
# Update pricing for new models or custom rates
tracker = TokenTracker("custom-model")
custom_pricing = {"input": 0.001, "output": 0.002}  # per 1K tokens
cost = tracker.estimate_cost(usage, pricing=custom_pricing)
```

## 📈 Cost Optimization Strategies

### Model Selection Guide
| Model | Cost/1K Tokens | Use Case | Recommendation |
|-------|----------------|----------|----------------|
| gpt-4o-mini | $0.0002 | Cost-sensitive operations | ✅ Recommended for most tasks |
| gpt-4o | $0.010 | Balanced performance | Good for complex analysis |
| gpt-4 | $0.045 | Highest quality | Use for critical tasks only |
| gpt-3.5-turbo | $0.001 | Simple tasks | Good for basic operations |

### Optimization Techniques
1. **Prompt Engineering**: Reduce token usage with concise, effective prompts
2. **Task Segmentation**: Break complex tasks into smaller, focused operations
3. **Model Selection**: Use appropriate models for different complexity levels
4. **Caching**: Implement response caching for repeated operations
5. **Batch Processing**: Group similar operations to reduce overhead

## 🎯 Performance Benchmarks

### Typical Performance Ranges
- **Tokens per Second**: 50-100 (good), 100+ (excellent)
- **Cost per Minute**: $0.001-0.01 (efficient), $0.01+ (review needed)
- **Task Duration**: <60s (fast), 60-120s (normal), 120s+ (optimize)

### Efficiency Targets
- **Token Efficiency**: >80% useful content in responses
- **Cost Efficiency**: <$0.05 per marketing insight
- **Time Efficiency**: <5 minutes for complete analysis

## 🔍 Troubleshooting

### Common Issues

#### High Token Usage
```
Problem: Tasks consuming excessive tokens
Solution: 
- Review prompt length and complexity
- Use more specific, focused prompts
- Consider breaking tasks into smaller operations
```

#### Slow Performance
```
Problem: Low tokens per second rate
Solution:
- Check network connectivity
- Optimize prompt structure
- Consider using faster models for simple tasks
```

#### Unexpected Costs
```
Problem: Higher than expected costs
Solution:
- Review model selection (use gpt-4o-mini for cost savings)
- Analyze token distribution (input vs output)
- Implement prompt optimization strategies
```

## 📚 API Reference

### TokenTracker Class
```python
class TokenTracker:
    def __init__(self, model_name: str = "gpt-4o-mini")
    def start_crew_tracking(self, crew_id: str) -> CrewTokenUsage
    def start_task_tracking(self, task_name: str, agent_name: str) -> TaskTokenUsage
    def record_llm_usage(self, prompt: str, response: str, actual_usage: Dict = None) -> TokenUsage
    def estimate_cost(self, token_usage: TokenUsage, model_name: str = None) -> Dict[str, float]
```

### TokenAnalyzer Class
```python
class TokenAnalyzer:
    @staticmethod
    def analyze_crew_usage(crew_usage: CrewTokenUsage) -> Dict[str, Any]
```

### Data Classes
```python
@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

@dataclass
class TaskTokenUsage:
    task_name: str
    agent_name: str
    start_time: datetime
    token_usage: TokenUsage
    # ... other fields

@dataclass
class CrewTokenUsage:
    crew_id: str
    start_time: datetime
    total_token_usage: TokenUsage
    task_usages: List[TaskTokenUsage]
    # ... other fields
```

## 🎉 Benefits

### For Marketing Teams
- **Budget Control**: Real-time cost monitoring and budget tracking
- **ROI Optimization**: Identify most cost-effective analysis approaches
- **Performance Insights**: Understand which agents provide best value
- **Cost Transparency**: Clear breakdown of analysis costs

### For Technical Teams
- **Performance Monitoring**: Track system efficiency and optimization opportunities
- **Cost Management**: Detailed cost analysis and optimization recommendations
- **Debugging**: Identify performance bottlenecks and inefficiencies
- **Scaling Insights**: Understand resource requirements for larger deployments

### For Business Stakeholders
- **Cost Justification**: Clear ROI analysis for AI-powered marketing insights
- **Budget Planning**: Accurate cost forecasting for marketing analysis projects
- **Value Demonstration**: Quantify the value of AI-driven marketing research
- **Optimization Opportunities**: Data-driven recommendations for cost reduction

---

The token tracking system transforms the Marketing Research Swarm into a cost-aware, performance-optimized platform that provides not only valuable marketing insights but also comprehensive analysis of the resources required to generate those insights.