"""
Intelligent Context Selection and Filtering
Implements Feature 2: Context Selection with relevance and token budget filtering
"""

import re
import math
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class RelevanceScore(Enum):
    """Relevance scoring levels"""
    CRITICAL = 1.0
    HIGH = 0.8
    MEDIUM = 0.6
    LOW = 0.4
    MINIMAL = 0.2


@dataclass
class ContextItem:
    """Represents a context item with relevance metadata"""
    key: str
    value: Any
    token_count: int
    relevance_score: float
    priority: str
    last_accessed: datetime
    dependencies: List[str]
    agent_specific: bool
    content_type: str
    
    @property
    def efficiency_score(self) -> float:
        """Calculate efficiency score (relevance per token)"""
        return self.relevance_score / max(self.token_count, 1)


class IntelligentContextFilter:
    """
    Intelligent context filtering system that selects the most relevant
    context items based on relevance scores and token budget constraints.
    """
    
    def __init__(self, token_budget: int = 4000):
        """Initialize the context filter."""
        self.token_budget = token_budget
        self.relevance_keywords = self._build_relevance_keywords()
        self.agent_specializations = self._build_agent_specializations()
        
    def _build_relevance_keywords(self) -> Dict[str, List[str]]:
        """Build keyword sets for different relevance categories."""
        return {
            "critical": [
                "error", "failure", "critical", "urgent", "required", "mandatory",
                "essential", "key_metric", "primary_objective", "main_goal"
            ],
            "high": [
                "important", "significant", "major", "primary", "core", "central",
                "revenue", "profit", "roi", "performance", "optimization", "strategy"
            ],
            "medium": [
                "relevant", "useful", "beneficial", "supporting", "secondary",
                "analysis", "insight", "trend", "pattern", "recommendation"
            ],
            "low": [
                "additional", "supplementary", "optional", "nice_to_have",
                "background", "context", "reference", "historical"
            ]
        }
    
    def _build_agent_specializations(self) -> Dict[str, List[str]]:
        """Build specialization keywords for different agent types."""
        return {
            "market_research_analyst": [
                "market", "research", "consumer", "behavior", "trends", "demographics",
                "segmentation", "survey", "focus_group", "market_size"
            ],
            "competitive_analyst": [
                "competitor", "competitive", "benchmark", "market_share", "positioning",
                "swot", "competitive_advantage", "threat", "opportunity"
            ],
            "data_analyst": [
                "data", "analytics", "statistics", "metrics", "kpi", "dashboard",
                "correlation", "regression", "forecast", "model", "algorithm"
            ],
            "content_strategist": [
                "content", "strategy", "messaging", "brand", "communication",
                "channel", "platform", "engagement", "storytelling", "narrative"
            ],
            "creative_copywriter": [
                "copy", "creative", "writing", "headline", "tagline", "campaign",
                "creative_brief", "tone", "voice", "style", "persuasion"
            ],
            "campaign_optimizer": [
                "campaign", "optimization", "budget", "allocation", "roi", "conversion",
                "performance", "efficiency", "cost", "spend", "attribution"
            ],
            "brand_performance_specialist": [
                "brand", "performance", "awareness", "perception", "equity", "loyalty",
                "brand_health", "reputation", "positioning", "differentiation"
            ],
            "forecasting_specialist": [
                "forecast", "prediction", "projection", "trend", "seasonal", "model",
                "time_series", "regression", "scenario", "planning", "future"
            ]
        }
    
    def filter_context(
        self,
        context: Dict[str, Any],
        agent_id: str,
        task_description: str = "",
        required_keys: List[str] = None,
        token_budget: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Filter context based on relevance and token budget.
        
        Args:
            context: Full context dictionary
            agent_id: ID of the agent requesting context
            task_description: Description of the current task
            required_keys: Keys that must be included regardless of budget
            token_budget: Override default token budget
            
        Returns:
            Filtered context dictionary
        """
        budget = token_budget or self.token_budget
        required_keys = required_keys or []
        
        # Convert context to ContextItems
        context_items = self._analyze_context_items(context, agent_id, task_description)
        
        # Apply filtering strategy
        filtered_items = self._apply_intelligent_filtering(
            context_items, agent_id, budget, required_keys
        )
        
        # Convert back to dictionary
        filtered_context = {item.key: item.value for item in filtered_items}
        
        # Log filtering results
        total_tokens = sum(item.token_count for item in filtered_items)
        logger.info(
            f"Context filtered for {agent_id}: "
            f"{len(filtered_context)}/{len(context)} items, "
            f"{total_tokens}/{budget} tokens"
        )
        
        return filtered_context
    
    def _analyze_context_items(
        self,
        context: Dict[str, Any],
        agent_id: str,
        task_description: str
    ) -> List[ContextItem]:
        """Analyze context items and assign relevance scores."""
        
        items = []
        task_keywords = self._extract_keywords(task_description.lower())
        agent_keywords = self.agent_specializations.get(agent_id, [])
        
        for key, value in context.items():
            # Calculate token count
            token_count = self._estimate_tokens(value)
            
            # Calculate relevance score
            relevance_score = self._calculate_relevance_score(
                key, value, task_keywords, agent_keywords
            )
            
            # Determine priority
            priority = self._determine_priority(key, value, relevance_score)
            
            # Check if agent-specific
            agent_specific = self._is_agent_specific(key, value, agent_id)
            
            # Determine content type
            content_type = self._determine_content_type(value)
            
            items.append(ContextItem(
                key=key,
                value=value,
                token_count=token_count,
                relevance_score=relevance_score,
                priority=priority,
                last_accessed=datetime.now(),
                dependencies=self._find_dependencies(key, context.keys()),
                agent_specific=agent_specific,
                content_type=content_type
            ))
        
        return items
    
    def _calculate_relevance_score(
        self,
        key: str,
        value: Any,
        task_keywords: List[str],
        agent_keywords: List[str]
    ) -> float:
        """Calculate relevance score for a context item."""
        
        score = 0.0
        text_content = f"{key} {str(value)}".lower()
        
        # Base score from keyword matching
        for category, keywords in self.relevance_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in text_content)
            if matches > 0:
                if category == "critical":
                    score += matches * 0.3
                elif category == "high":
                    score += matches * 0.2
                elif category == "medium":
                    score += matches * 0.1
                elif category == "low":
                    score += matches * 0.05
        
        # Agent specialization bonus
        agent_matches = sum(1 for keyword in agent_keywords if keyword in text_content)
        score += agent_matches * 0.15
        
        # Task relevance bonus
        task_matches = sum(1 for keyword in task_keywords if keyword in text_content)
        score += task_matches * 0.1
        
        # Key name patterns
        if any(pattern in key.lower() for pattern in ["result", "analysis", "summary"]):
            score += 0.2
        if any(pattern in key.lower() for pattern in ["temp", "cache", "debug"]):
            score -= 0.1
        
        # Content type bonuses
        if isinstance(value, dict) and "analysis" in str(value).lower():
            score += 0.15
        if isinstance(value, list) and len(value) > 0:
            score += 0.1
        
        return min(max(score, 0.0), 1.0)  # Clamp to [0, 1]
    
    def _apply_intelligent_filtering(
        self,
        items: List[ContextItem],
        agent_id: str,
        budget: int,
        required_keys: List[str]
    ) -> List[ContextItem]:
        """Apply intelligent filtering strategy."""
        
        filtered_items = []
        remaining_budget = budget
        
        # Phase 1: Always include required keys
        required_items = [item for item in items if item.key in required_keys]
        for item in required_items:
            filtered_items.append(item)
            remaining_budget -= item.token_count
        
        # Phase 2: Sort remaining items by efficiency score
        remaining_items = [item for item in items if item.key not in required_keys]
        remaining_items.sort(key=lambda x: x.efficiency_score, reverse=True)
        
        # Phase 3: Greedy selection with dependency awareness
        selected_keys = set(item.key for item in filtered_items)
        
        for item in remaining_items:
            # Check if we can afford this item
            if item.token_count <= remaining_budget:
                # Check dependencies
                if self._dependencies_satisfied(item, selected_keys):
                    filtered_items.append(item)
                    selected_keys.add(item.key)
                    remaining_budget -= item.token_count
            
            # Early termination if budget is very low
            if remaining_budget < 50:
                break
        
        # Phase 4: Try to include high-relevance items even if over budget
        if remaining_budget < 0:
            # Remove lowest efficiency items until under budget
            filtered_items.sort(key=lambda x: x.efficiency_score)
            while remaining_budget < 0 and filtered_items:
                removed_item = filtered_items.pop(0)
                if removed_item.key not in required_keys:
                    remaining_budget += removed_item.token_count
        
        return filtered_items
    
    def _dependencies_satisfied(self, item: ContextItem, selected_keys: Set[str]) -> bool:
        """Check if item dependencies are satisfied."""
        return all(dep in selected_keys for dep in item.dependencies)
    
    def _find_dependencies(self, key: str, all_keys: List[str]) -> List[str]:
        """Find dependencies for a context item."""
        dependencies = []
        
        # Simple heuristic: look for keys that this key might depend on
        key_lower = key.lower()
        
        for other_key in all_keys:
            if other_key == key:
                continue
                
            other_lower = other_key.lower()
            
            # Check for explicit dependency patterns
            if f"{other_lower}_" in key_lower:
                dependencies.append(other_key)
            elif key_lower.endswith(f"_{other_lower}"):
                dependencies.append(other_key)
            elif "result" in key_lower and other_lower.replace("_", "") in key_lower:
                dependencies.append(other_key)
        
        return dependencies
    
    def _determine_priority(self, key: str, value: Any, relevance_score: float) -> str:
        """Determine priority level for a context item."""
        if relevance_score >= 0.8:
            return "critical"
        elif relevance_score >= 0.6:
            return "high"
        elif relevance_score >= 0.4:
            return "medium"
        else:
            return "low"
    
    def _is_agent_specific(self, key: str, value: Any, agent_id: str) -> bool:
        """Check if context item is specific to the agent."""
        key_lower = key.lower()
        agent_lower = agent_id.lower()
        
        # Check if key contains agent name
        if agent_lower.replace("_", "") in key_lower:
            return True
        
        # Check agent specialization keywords
        agent_keywords = self.agent_specializations.get(agent_id, [])
        text_content = f"{key} {str(value)}".lower()
        
        matches = sum(1 for keyword in agent_keywords if keyword in text_content)
        return matches >= 2  # Threshold for agent specificity
    
    def _determine_content_type(self, value: Any) -> str:
        """Determine the content type of a value."""
        if isinstance(value, dict):
            if "analysis" in str(value).lower():
                return "analysis_result"
            else:
                return "structured_data"
        elif isinstance(value, list):
            return "list_data"
        elif isinstance(value, str):
            if len(value) > 500:
                return "long_text"
            else:
                return "short_text"
        elif isinstance(value, (int, float)):
            return "numeric"
        else:
            return "other"
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        # Simple keyword extraction
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Filter out common stop words
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "is", "are", "was", "were", "be", "been", "have",
            "has", "had", "do", "does", "did", "will", "would", "could", "should"
        }
        
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return keywords
    
    def _estimate_tokens(self, value: Any) -> int:
        """Estimate token count for a value."""
        if isinstance(value, str):
            return max(1, len(value.split()) * 1.3)
        elif isinstance(value, dict):
            return sum(self._estimate_tokens(f"{k}: {v}") for k, v in value.items())
        elif isinstance(value, list):
            return sum(self._estimate_tokens(str(item)) for item in value[:10])
        else:
            return max(1, len(str(value).split()) * 1.3)
    
    def get_filtering_stats(self, original_context: Dict[str, Any], filtered_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get statistics about the filtering operation."""
        
        original_tokens = sum(self._estimate_tokens(v) for v in original_context.values())
        filtered_tokens = sum(self._estimate_tokens(v) for v in filtered_context.values())
        
        return {
            "original_items": len(original_context),
            "filtered_items": len(filtered_context),
            "reduction_ratio": 1 - (len(filtered_context) / max(len(original_context), 1)),
            "original_tokens": original_tokens,
            "filtered_tokens": filtered_tokens,
            "token_reduction": 1 - (filtered_tokens / max(original_tokens, 1)),
            "efficiency_gain": (original_tokens - filtered_tokens) / max(original_tokens, 1)
        }


# Global instance
_global_context_filter = None


def get_intelligent_filter() -> IntelligentContextFilter:
    """Get the global intelligent context filter instance."""
    global _global_context_filter
    if _global_context_filter is None:
        _global_context_filter = IntelligentContextFilter()
    return _global_context_filter