"""Lyapunov virtual queue for load balancing using lists."""
from __future__ import annotations

from typing import List


class LyapunovQueue:
    def __init__(self, num_uav: int, theta: float = 0.3) -> None:
        self.theta = theta
        self.queue = [0.0 for _ in range(num_uav)]

    def update(self, loads: List[float]) -> List[float]:
        self.queue = [max(q + (u - self.theta), 0.0) for q, u in zip(self.queue, loads)]
        return list(self.queue)

    def update_from_arrival_service(self, arrivals: List[float], service: List[float]) -> List[float]:
        """Update queue using Q(t+1)=max(Q(t)-mu,0)+A with explicit arrivals/service."""

        updated: List[float] = []
        for q, a, mu in zip(self.queue, arrivals, service):
            updated.append(max(q - mu, 0.0) + a)
        self.queue = updated
        return list(self.queue)

    def pressure(self) -> List[float]:
        return list(self.queue)
