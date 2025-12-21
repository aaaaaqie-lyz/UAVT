"""Metrics utilities for UAV LLM partition simulation using standard library."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


def jain_fairness(loads: List[float]) -> float:
    numerator = (sum(loads)) ** 2
    denominator = len(loads) * sum(l ** 2 for l in loads) + 1e-9
    return numerator / denominator if denominator > 0 else 0.0


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def comprehensive_score(
    avg_delay: float,
    avg_fairness: float,
    total_migrations: float,
    intervals: int,
    failure_rate: float,
    avg_max_load: float,
    avg_mem_load: float,
    avg_comp_load: float,
    weight_delay: float = 0.3,
    weight_fairness: float = 0.2,
    weight_stability: float = 0.2,
    weight_reliability: float = 0.2,
    weight_efficiency: float = 0.1,
    reference_delay: float = 5.0,
    reference_migrations: float = 10.0,
) -> float:
    delay_score = _clamp01(1.0 / (1.0 + avg_delay / reference_delay))
    fairness_score = _clamp01(avg_fairness)
    migrations_per_interval = total_migrations / max(intervals, 1)
    stability_score = _clamp01(1.0 / (1.0 + migrations_per_interval / reference_migrations))
    reliability_score = _clamp01(1.0 - failure_rate)
    load_eff = _clamp01(1.0 - abs(avg_max_load - 0.7) / 0.7)
    utilization_eff = _clamp01((avg_mem_load + avg_comp_load) / 2.0)
    efficiency_score = _clamp01((load_eff + utilization_eff) / 2.0)
    total = weight_delay + weight_fairness + weight_stability + weight_reliability + weight_efficiency
    return (
        weight_delay * delay_score
        + weight_fairness * fairness_score
        + weight_stability * stability_score
        + weight_reliability * reliability_score
        + weight_efficiency * efficiency_score
    ) / max(total, 1e-6)


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
    failure_reason: str
    mem_loads: List[float]
    comp_loads: List[float]
    mem_used: List[float]
    comp_used: List[float]
    comp_delay_by_dev: List[float]
    comm_delay_by_dev: List[float]
    mig_delay_by_dev: List[float]
    total_delay_by_dev: List[float]
    loads: List[float]
    queues: List[float]
    weights: List[float]
    rho_w: float
    rho_q: float


@dataclass
class MetricsLogger:
    history: List[IntervalMetrics] = field(default_factory=list)

    def log(self, interval: IntervalMetrics) -> None:
        self.history.append(interval)

    def aggregate(self) -> Dict[str, float]:
        if not self.history:
            return {}
        failure_reasons: Dict[str, int] = {}
        for m in self.history:
            if m.failure_reason:
                failure_reasons[m.failure_reason] = failure_reasons.get(m.failure_reason, 0) + 1
        avg_max_load = sum(m.max_load for m in self.history) / len(self.history)
        avg_fairness = sum(m.fairness for m in self.history) / len(self.history)
        avg_delay = sum(m.delay for m in self.history) / len(self.history)
        total_migrations = sum(m.migration_count for m in self.history)
        avg_migration_volume = sum(m.migration_volume for m in self.history) / len(self.history)
        failure_rate = sum(1 if m.failure else 0 for m in self.history) / len(self.history)
        avg_mem_load = sum(sum(m.mem_loads) / len(m.mem_loads) for m in self.history) / len(self.history)
        avg_comp_load = sum(sum(m.comp_loads) / len(m.comp_loads) for m in self.history) / len(self.history)
        avg_queue = sum(sum(m.queues) / len(m.queues) for m in self.history) / len(self.history)
        max_queue = max(max(m.queues) for m in self.history)
        avg_rho_w = sum(m.rho_w for m in self.history) / len(self.history)
        avg_rho_q = sum(m.rho_q for m in self.history) / len(self.history)
        avg_comp_delay = sum(sum(m.comp_delay_by_dev) / len(m.comp_delay_by_dev) for m in self.history) / len(
            self.history
        )
        avg_comm_delay = sum(sum(m.comm_delay_by_dev) / len(m.comm_delay_by_dev) for m in self.history) / len(
            self.history
        )
        avg_mig_delay = sum(sum(m.mig_delay_by_dev) / len(m.mig_delay_by_dev) for m in self.history) / len(
            self.history
        )
        avg_total_delay = sum(sum(m.total_delay_by_dev) / len(m.total_delay_by_dev) for m in self.history) / len(
            self.history
        )

        comp_score = comprehensive_score(
            avg_delay=avg_delay,
            avg_fairness=avg_fairness,
            total_migrations=total_migrations,
            intervals=len(self.history),
            failure_rate=failure_rate,
            avg_max_load=avg_max_load,
            avg_mem_load=avg_mem_load,
            avg_comp_load=avg_comp_load,
        )
        perf_balanced = comprehensive_score(
            avg_delay=avg_delay,
            avg_fairness=avg_fairness,
            total_migrations=total_migrations,
            intervals=len(self.history),
            failure_rate=failure_rate,
            avg_max_load=avg_max_load,
            avg_mem_load=avg_mem_load,
            avg_comp_load=avg_comp_load,
            weight_delay=0.25,
            weight_fairness=0.25,
            weight_stability=0.25,
            weight_reliability=0.25,
        )
        perf_latency = comprehensive_score(
            avg_delay=avg_delay,
            avg_fairness=avg_fairness,
            total_migrations=total_migrations,
            intervals=len(self.history),
            failure_rate=failure_rate,
            avg_max_load=avg_max_load,
            avg_mem_load=avg_mem_load,
            avg_comp_load=avg_comp_load,
            weight_delay=0.5,
            weight_fairness=0.15,
            weight_stability=0.15,
            weight_reliability=0.2,
        )
        perf_reliability = comprehensive_score(
            avg_delay=avg_delay,
            avg_fairness=avg_fairness,
            total_migrations=total_migrations,
            intervals=len(self.history),
            failure_rate=failure_rate,
            avg_max_load=avg_max_load,
            avg_mem_load=avg_mem_load,
            avg_comp_load=avg_comp_load,
            weight_delay=0.2,
            weight_fairness=0.2,
            weight_stability=0.2,
            weight_reliability=0.4,
        )
        perf_stability = comprehensive_score(
            avg_delay=avg_delay,
            avg_fairness=avg_fairness,
            total_migrations=total_migrations,
            intervals=len(self.history),
            failure_rate=failure_rate,
            avg_max_load=avg_max_load,
            avg_mem_load=avg_mem_load,
            avg_comp_load=avg_comp_load,
            weight_delay=0.2,
            weight_fairness=0.2,
            weight_stability=0.4,
            weight_reliability=0.2,
        )
        perf_fairness = comprehensive_score(
            avg_delay=avg_delay,
            avg_fairness=avg_fairness,
            total_migrations=total_migrations,
            intervals=len(self.history),
            failure_rate=failure_rate,
            avg_max_load=avg_max_load,
            avg_mem_load=avg_mem_load,
            avg_comp_load=avg_comp_load,
            weight_delay=0.2,
            weight_fairness=0.4,
            weight_stability=0.2,
            weight_reliability=0.2,
        )

        return {
            "avg_max_load": avg_max_load,
            "avg_fairness": avg_fairness,
            "avg_delay": avg_delay,
            "total_migrations": total_migrations,
            "avg_migration_volume": avg_migration_volume,
            "failure_rate": failure_rate,
            "failure_reasons": failure_reasons,
            "avg_mem_load": avg_mem_load,
            "avg_comp_load": avg_comp_load,
            "avg_queue": avg_queue,
            "max_queue": max_queue,
            "avg_rho_w": avg_rho_w,
            "avg_rho_q": avg_rho_q,
            "avg_comp_delay_per_dev": avg_comp_delay,
            "avg_comm_delay_per_dev": avg_comm_delay,
            "avg_mig_delay_per_dev": avg_mig_delay,
            "avg_total_delay_per_dev": avg_total_delay,
            "comprehensive_score": comp_score,
            "perf_index_balanced": perf_balanced,
            "perf_index_latency": perf_latency,
            "perf_index_reliability": perf_reliability,
            "perf_index_stability": perf_stability,
            "perf_index_fairness": perf_fairness,
        }

