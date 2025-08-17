"""
Context Quality Monitoring and Token Budget Management

Provides:
- ContextQualityMonitor: evaluates context for poisoning, distraction, confusion, clash
- TokenBudgetManager: allocates token budgets across context categories
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class ContextQualityReport:
    timestamp: str
    size_estimate: int
    poisoning_score: float
    distraction_score: float
    confusion_score: float
    clash_score: float
    notes: Dict[str, Any] = field(default_factory=dict)


class ContextQualityMonitor:
    """Heuristic-based context quality analyzer."""

    def __init__(self) -> None:
        pass

    def evaluate_quality(self, context: Dict[str, Any]) -> ContextQualityReport:
        size_estimate = self._estimate_size(context)
        poisoning = self._detect_poisoning(context)
        distraction = self._detect_distraction(context)
        confusion = self._detect_confusion(context)
        clash = self._detect_clash(context)
        notes = {
            "top_keys": list(context.keys())[:10],
            "size_estimate": size_estimate,
        }
        report = ContextQualityReport(
            timestamp=datetime.now().isoformat(),
            size_estimate=size_estimate,
            poisoning_score=poisoning,
            distraction_score=distraction,
            confusion_score=confusion,
            clash_score=clash,
            notes=notes,
        )
        logger.debug(f"Context quality report: {report}")
        return report

    def _estimate_size(self, ctx: Dict[str, Any]) -> int:
        try:
            return sum(len(str(v)) for v in ctx.values())
        except Exception:
            return len(str(ctx))

    def _detect_poisoning(self, ctx: Dict[str, Any]) -> float:
        # Look for typical error/poison indicators
        text = str(ctx).lower()
        indicators = ["error", "fail", "invalid", "poison", "contradict", "halluc", "unknown"]
        count = sum(text.count(ind) for ind in indicators)
        return float(min(1.0, count / 10.0))

    def _detect_distraction(self, ctx: Dict[str, Any]) -> float:
        # Heuristic: many keys with small values -> distraction
        keys = list(ctx.keys())
        small_values = 0
        for v in ctx.values():
            try:
                if isinstance(v, (str, list, dict)):
                    l = len(v)
                else:
                    l = len(str(v))
                if l < 20:
                    small_values += 1
            except Exception:
                continue
        if not keys:
            return 0.0
        ratio = small_values / max(1, len(keys))
        return float(min(1.0, ratio))

    def _detect_confusion(self, ctx: Dict[str, Any]) -> float:
        # Heuristic: repeated overlapping keys or deeply nested dicts
        key_text = " ".join(ctx.keys()).lower()
        duplicates = len(re.findall(r"(result|summary|data)", key_text))
        nesting = self._max_nesting(ctx)
        score = min(1.0, (duplicates / 10.0) + (nesting / 10.0))
        return float(score)

    def _detect_clash(self, ctx: Dict[str, Any]) -> float:
        # Heuristic: numeric fields with conflicting values across sections
        # We check for common fields like budget, duration, target_audience mismatch forms
        text = str(ctx).lower()
        conflicts = 0
        # Very simple: detect multiple currency-like numbers -> potential clash
        numbers = re.findall(r"\$?\b\d{2,}(?:,\d{3})*(?:\.\d+)?\b", text)
        if len(numbers) > 20:
            conflicts += 1
        # Opposing words
        if "increase" in text and "decrease" in text:
            conflicts += 1
        if "true" in text and "false" in text:
            conflicts += 1
        return float(min(1.0, conflicts / 3.0))

    def _max_nesting(self, obj: Any, depth: int = 0) -> int:
        if isinstance(obj, dict) and obj:
            return max(self._max_nesting(v, depth + 1) for v in obj.values())
        if isinstance(obj, list) and obj:
            return max(self._max_nesting(v, depth + 1) for v in obj)
        return depth


@dataclass
class BudgetAllocations:
    total_budget: int
    allocations: Dict[str, int]


class TokenBudgetManager:
    """Allocates token budgets across context categories."""

    def __init__(self, total_budget: int, strategy: Optional[Dict[str, float]] = None) -> None:
        self.total_budget = int(total_budget) if total_budget is not None else 0
        self.strategy = strategy or {
            "instructions": 0.3,
            "knowledge": 0.4,
            "tools": 0.2,
            "scratchpad": 0.1,
        }
        # Normalize
        s = sum(self.strategy.values())
        if s <= 0:
            self.strategy = {"instructions": 0.25, "knowledge": 0.5, "tools": 0.15, "scratchpad": 0.1}
        else:
            self.strategy = {k: v / s for k, v in self.strategy.items()}

    def allocate_budget(self) -> BudgetAllocations:
        allocations = {k: int(self.total_budget * r) for k, r in self.strategy.items()}
        # Adjust rounding
        diff = self.total_budget - sum(allocations.values())
        if diff > 0 and allocations:
            # Give remainder to the largest bucket
            max_k = max(allocations, key=lambda k: allocations[k])
            allocations[max_k] += diff
        return BudgetAllocations(total_budget=self.total_budget, allocations=allocations)

    def categorize_key(self, key: str, value: Any) -> str:
        k = key.lower()
        if "scratchpad" in k:
            return "scratchpad"
        if any(x in k for x in ["instruction", "prompt", "guidance"]):
            return "instructions"
        if any(x in k for x in ["tool", "function", "api_response", "attachment"]):
            return "tools"
        return "knowledge"
