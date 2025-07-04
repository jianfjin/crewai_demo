# Flow-Based Token Optimization Proposal

## 🎯 **Executive Summary**

Transform the current sequential task system into a flow-based architecture with intelligent state management, context engineering, and caching to reduce token usage by **60-80%** while improving performance and data sharing.

## 📊 **Current Token Usage Issues**

### **Root Causes:**
1. **Context Accumulation**: Each agent receives full conversation history
2. **Data Re-reading**: Multiple agents read the same source data
3. **Result Duplication**: Analysis results passed as text context
4. **No Caching**: Tools re-execute identical operations
5. **Growing Context**: Token usage grows exponentially with each step

### **Current Token Flow:**
```
Data Analyst: 15,000 tokens (data + analysis)
    ↓ (passes full context)
Campaign Optimizer: 25,000 tokens (previous context + new analysis)
Total: 40,000+ tokens
```

## 🚀 **Proposed Solution Architecture**

### **1. Flow-Based Execution with State Management**

```python
from crewai.flow import Flow, start, listen
from pydantic import BaseModel
import pandas as pd
from typing import Dict, Any
import hashlib
import pickle

class FlowState(BaseModel):
    data_cache: Dict[str, Any] = {}
    analysis_results: Dict[str, Any] = {}
    context_budget: int = 4000  # Max tokens per step
    current_step: str = ""
    
class MarketingResearchFlow(Flow[FlowState]):
    
    @start()
    def load_and_cache_data(self) -> str:
        """Load source data once and cache with hash key"""
        # Load data efficiently
        df = pd.read_csv(self.state.data_file_path)
        
        # Create hash key for caching
        data_hash = self._create_hash(df.to_string())
        
        # Cache structured data
        self.state.data_cache[data_hash] = {
            'dataframe': df,
            'summary_stats': df.describe().to_dict(),
            'columns': df.columns.tolist(),
            'shape': df.shape
        }
        
        return data_hash  # Return reference, not data
    
    @listen(load_and_cache_data)
    def analyze_profitability(self, data_key: str) -> str:
        """Analyze profitability using cached data"""
        # Get data from cache
        data = self.state.data_cache[data_key]
        
        # Run analysis with minimal context
        analysis_result = self._run_profitability_analysis(data['dataframe'])
        
        # Cache result with hash
        result_hash = self._create_hash(str(analysis_result))
        self.state.analysis_results[result_hash] = {
            'type': 'profitability',
            'summary': self._summarize_analysis(analysis_result),
            'key_insights': self._extract_key_insights(analysis_result),
            'full_result': analysis_result  # Cached separately
        }
        
        return result_hash
    
    @listen(analyze_profitability)
    def optimize_budget(self, profitability_key: str) -> str:
        """Optimize budget using analysis insights"""
        # Get only summary from cache (not full result)
        analysis_summary = self.state.analysis_results[profitability_key]['summary']
        
        # Run optimization with minimal context
        optimization_result = self._run_budget_optimization(analysis_summary)
        
        # Cache and return reference
        result_hash = self._create_hash(str(optimization_result))
        self.state.analysis_results[result_hash] = optimization_result
        
        return result_hash
```

### **2. Context Engineering System**

```python
class ContextManager:
    def __init__(self, token_budget: int = 4000):
        self.token_budget = token_budget
        self.context_strategies = {
            'critical': 1.0,    # Always keep
            'important': 0.7,   # Keep if space
            'useful': 0.4,      # Summarize
            'optional': 0.1     # Remove first
        }
    
    def optimize_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply progressive context pruning"""
        current_tokens = self._estimate_tokens(context)
        
        if current_tokens <= self.token_budget:
            return context
        
        # Progressive pruning strategy
        optimized = {}
        
        # 1. Keep critical elements
        for key, value in context.items():
            priority = self._get_priority(key, value)
            if priority == 'critical':
                optimized[key] = value
        
        # 2. Summarize important elements
        remaining_budget = self.token_budget - self._estimate_tokens(optimized)
        for key, value in context.items():
            priority = self._get_priority(key, value)
            if priority == 'important':
                if self._estimate_tokens(value) <= remaining_budget:
                    optimized[key] = value
                else:
                    optimized[key] = self._summarize(value)
                    
        return optimized
    
    def _summarize(self, content: Any) -> str:
        """Create compact summaries of large content"""
        if isinstance(content, pd.DataFrame):
            return {
                'shape': content.shape,
                'columns': content.columns.tolist(),
                'summary_stats': content.describe().to_dict()
            }
        elif isinstance(content, dict):
            return {k: str(v)[:100] + "..." if len(str(v)) > 100 else v 
                   for k, v in content.items()}
        else:
            return str(content)[:200] + "..." if len(str(content)) > 200 else str(content)
```

### **3. Intelligent Caching System**

```python
class SmartCache:
    def __init__(self, max_size_mb: int = 100):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size_mb * 1024 * 1024  # Convert to bytes
        
    def store(self, key: str, data: Any, ttl: int = 3600) -> str:
        """Store data with automatic expiration"""
        # Create compact reference
        reference = f"cache://{key[:8]}"
        
        # Store with metadata
        self.cache[reference] = {
            'data': data,
            'created': time.time(),
            'ttl': ttl,
            'size': self._estimate_size(data),
            'access_count': 0
        }
        
        self._cleanup_if_needed()
        return reference
    
    def retrieve(self, reference: str) -> Any:
        """Retrieve data by reference"""
        if reference in self.cache:
            entry = self.cache[reference]
            entry['access_count'] += 1
            entry['last_access'] = time.time()
            return entry['data']
        return None
    
    def _cleanup_if_needed(self):
        """Remove expired or least-used items"""
        current_time = time.time()
        total_size = sum(entry['size'] for entry in self.cache.values())
        
        if total_size > self.max_size:
            # Remove least recently used items
            sorted_items = sorted(
                self.cache.items(),
                key=lambda x: x[1].get('last_access', 0)
            )
            
            for ref, entry in sorted_items:
                if total_size <= self.max_size * 0.8:  # Keep 80% capacity
                    break
                del self.cache[ref]
                total_size -= entry['size']
```

### **4. Mem0 Integration for Long-term Memory**

```python
from mem0 import MemoryClient

class Mem0ContextManager:
    def __init__(self):
        self.memory = MemoryClient()
        
    def store_insights(self, analysis_type: str, insights: Dict[str, Any]):
        """Store key insights in Mem0 for future reference"""
        # Extract and store only key insights, not raw data
        key_points = self._extract_key_points(insights)
        
        self.memory.add(
            messages=[{
                "role": "assistant",
                "content": f"Analysis insights for {analysis_type}: {key_points}"
            }],
            user_id=f"marketing_research_{analysis_type}",
            metadata={
                "analysis_type": analysis_type,
                "timestamp": time.time(),
                "key_metrics": self._extract_metrics(insights)
            }
        )
    
    def get_relevant_context(self, query: str, analysis_type: str) -> str:
        """Retrieve only relevant context for current analysis"""
        relevant_memories = self.memory.search(
            query=query,
            user_id=f"marketing_research_{analysis_type}",
            limit=3  # Limit to prevent context bloat
        )
        
        # Return compact, relevant context
        return self._format_compact_context(relevant_memories)
    
    def _extract_key_points(self, insights: Dict[str, Any]) -> str:
        """Extract only the most important insights"""
        key_points = []
        
        if 'top_performers' in insights:
            key_points.append(f"Top performer: {insights['top_performers'][0]}")
        
        if 'key_metrics' in insights:
            key_points.append(f"Key metrics: {insights['key_metrics']}")
            
        if 'recommendations' in insights:
            key_points.append(f"Main recommendation: {insights['recommendations'][0]}")
            
        return "; ".join(key_points)
```

## 🏗️ **Implementation Plan**

### **Phase 1: Flow Architecture (Week 1)**

```python
# File: src/marketing_research_swarm/flows/roi_analysis_flow.py
class ROIAnalysisFlow(Flow[FlowState]):
    
    @start()
    def load_data(self) -> str:
        """Load and cache source data"""
        return self._cache_data()
    
    @listen(load_data)
    def analyze_profitability(self, data_key: str) -> str:
        """Analyze profitability with minimal context"""
        return self._run_analysis_with_cache(data_key, 'profitability')
    
    @listen(analyze_profitability)
    def optimize_budget(self, analysis_key: str) -> str:
        """Optimize budget using analysis insights"""
        return self._run_optimization_with_cache(analysis_key)
    
    @listen(optimize_budget)
    def generate_report(self, optimization_key: str) -> str:
        """Generate final report with all insights"""
        return self._compile_report([analysis_key, optimization_key])
```

### **Phase 2: Context Engineering (Week 2)**

```python
# File: src/marketing_research_swarm/context/manager.py
class AdvancedContextManager:
    
    def __init__(self):
        self.strategies = {
            'progressive_pruning': ProgressivePruning(),
            'abstracted_summaries': AbstractedSummaries(),
            'minimal_context': MinimalContext(),
            'stateless': StatelessContext()
        }
        
    def apply_strategy(self, context: Any, strategy: str, budget: int) -> Any:
        """Apply specific context management strategy"""
        return self.strategies[strategy].optimize(context, budget)
```

### **Phase 3: Caching System (Week 3)**

```python
# File: src/marketing_research_swarm/cache/smart_cache.py
class ToolOutputCache:
    
    def cached_tool_call(self, tool_func, *args, **kwargs):
        """Decorator for automatic tool output caching"""
        cache_key = self._generate_cache_key(tool_func.__name__, args, kwargs)
        
        if cached_result := self.cache.get(cache_key):
            return cached_result
            
        result = tool_func(*args, **kwargs)
        
        # Cache large outputs with references
        if self._estimate_size(result) > 1000:  # 1KB threshold
            reference = self.cache.store(cache_key, result)
            return f"cached://{reference}"
        
        return result
```

### **Phase 4: Mem0 Integration (Week 4)**

```python
# File: src/marketing_research_swarm/memory/mem0_integration.py
class MarketingMemoryManager:
    
    def __init__(self):
        self.mem0 = MemoryClient()
        self.context_manager = ContextManager()
        
    def store_analysis_insights(self, analysis_result: Dict[str, Any]):
        """Store only key insights, not raw data"""
        insights = self._extract_insights(analysis_result)
        self.mem0.add_insight(insights)
        
    def get_relevant_context(self, current_task: str) -> str:
        """Get minimal relevant context for current task"""
        return self.mem0.search_insights(
            query=current_task,
            limit=3,
            max_tokens=500  # Strict limit
        )
```

## 📊 **Expected Token Reduction**

### **Current vs. Optimized:**

| Component | Current Tokens | Optimized Tokens | Reduction |
|-----------|---------------|------------------|-----------|
| **Data Loading** | 5,000 | 100 (reference) | **98%** |
| **Context Passing** | 15,000 | 2,000 (summary) | **87%** |
| **Tool Outputs** | 10,000 | 500 (cached refs) | **95%** |
| **Analysis Results** | 8,000 | 1,500 (insights) | **81%** |
| **Total per Analysis** | **38,000** | **4,100** | **89%** |

### **Cost Savings:**
- **Current**: $0.0060 per analysis
- **Optimized**: $0.0007 per analysis  
- **Savings**: **88% cost reduction**

## 🎯 **Implementation Priority**

### **High Priority (Immediate Impact):**
1. **Flow-based execution** with state management
2. **Data caching** with hash references
3. **Context pruning** with token budgets
4. **Tool output caching** for large results

### **Medium Priority (Performance):**
1. **Mem0 integration** for long-term memory
2. **Advanced context strategies** (progressive pruning)
3. **Automatic cache cleanup** and expiration
4. **Token budget enforcement** across workflows

### **Low Priority (Enhancement):**
1. **Multi-strategy context management**
2. **Dependency tracking** for context elements
3. **Advanced summarization** techniques
4. **Performance monitoring** and optimization

## 🚀 **Benefits Summary**

### **Token Efficiency:**
- **89% reduction** in token usage
- **88% cost savings** per analysis
- **Faster execution** through caching
- **Scalable architecture** for multiple analysis types

### **Performance Improvements:**
- **Shared data** across agents without duplication
- **Intelligent caching** prevents re-computation
- **Context-aware** task execution
- **Memory-efficient** long-term storage

### **Maintainability:**
- **Modular flow design** for easy updates
- **Clear separation** of concerns
- **Configurable** context strategies
- **Monitoring** and debugging capabilities

This proposal transforms the current token-heavy system into an efficient, scalable, and cost-effective marketing research platform while maintaining the quality and depth of analysis.