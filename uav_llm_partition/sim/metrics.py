"""Metrics utilities for UAV LLM partition simulation using standard library."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


def jain_fairness(loads: List[float]) -> float:
    numerator = (sum(loads)) ** 2
    denominator = len(loads) * sum(l ** 2 for l in loads) + 1e-9
    return numerator / denominator if denominator > 0 else 0.0


@dataclass
class IntervalMetrics:
    max_load: float
    fairness: float
    delay: float
    delay_comp: float
    delay_comm: float
    delay_mig: float
    migration_count: int
    migration_volume: float
    failure: bool
    mem_loads: List[float]
    comp_loads: List[float]
    mem_used: List[float]
    comp_used: List[float]


@dataclass
class MetricsLogger:
    history: List[IntervalMetrics] = field(default_factory=list)

    def log(self, interval: IntervalMetrics) -> None:
        self.history.append(interval)

    def aggregate(self) -> Dict[str, float]:
        if not self.history:
            return {}
        return {
            "avg_max_load": sum(m.max_load for m in self.history) / len(self.history),
            "avg_fairness": sum(m.fairness for m in self.history) / len(self.history),
            "avg_delay": sum(m.delay for m in self.history) / len(self.history),
            "total_migrations": sum(m.migration_count for m in self.history),
            "avg_migration_volume": sum(m.migration_volume for m in self.history) / len(self.history),
            "failure_rate": sum(1 if m.failure else 0 for m in self.history) / len(self.history),
            "avg_mem_load": sum(sum(m.mem_loads) / len(m.mem_loads) for m in self.history) / len(self.history),
            "avg_comp_load": sum(sum(m.comp_loads) / len(m.comp_loads) for m in self.history) / len(self.history),
        }

