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

    def compute(self, compute: List[float], memory: List[float], los_score: List[float], mobility_risk: List[float]) -> List[float]:
        c_norm = normalize(compute)
        m_norm = normalize(memory)
        los_norm = normalize(los_score)
        risk_norm = normalize(mobility_risk)
        return [
            self.alpha * c + self.beta * m + self.gamma * l - self.delta * r
            for c, m, l, r in zip(c_norm, m_norm, los_norm, risk_norm)
        ]

