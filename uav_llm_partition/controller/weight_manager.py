"""Dynamic weight computation for UAVs using stdlib math."""
from __future__ import annotations

from typing import List

from uav_llm_partition.utils.num import normalize


class WeightManager:
    def __init__(self, alpha: float = 0.3, beta: float = 0.3, gamma: float = 0.3, delta: float = 0.2) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def _adaptive_weights(self, system_state: dict | None) -> tuple[float, float, float, float]:
        """Return (alpha, beta, gamma, delta) based on coarse system state."""

        if not system_state:
            return self.alpha, self.beta, self.gamma, self.delta
        if system_state.get("is_compute_bound"):
            return 0.5, 0.2, 0.2, 0.1
        if system_state.get("is_memory_bound"):
            return 0.2, 0.5, 0.2, 0.1
        return 0.3, 0.3, 0.3, 0.1

    def compute(
        self,
        compute: List[float],
        memory: List[float],
        los_score: List[float],
        mobility_risk: List[float],
        system_state: dict | None = None,
    ) -> List[float]:
        alpha, beta, gamma, delta = self._adaptive_weights(system_state)
        c_norm = normalize(compute)
        m_norm = normalize(memory)
        los_norm = normalize(los_score)
        risk_norm = normalize(mobility_risk)
        raw = [alpha * c + beta * m + gamma * l - delta * r for c, m, l, r in zip(c_norm, m_norm, los_norm, risk_norm)]
        return normalize(raw)
