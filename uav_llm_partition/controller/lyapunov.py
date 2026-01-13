"""Lyapunov virtual queue for load balancing using lists."""
from __future__ import annotations

from typing import List


class LyapunovQueue:
    def __init__(
        self, num_uav: int, theta: float = 0.3, q_max: float = 10.0, arrival_clip: float = 5.0
    ) -> None:
        self.theta = theta
        self.q_max = q_max
        self.arrival_clip = arrival_clip
        self.queue = [0.0 for _ in range(num_uav)]

    def update(self, loads: List[float]) -> List[float]:
        self.queue = [max(q + (u - self.theta), 0.0) for q, u in zip(self.queue, loads)]
        return list(self.queue)

    def update_from_arrival_service(self, arrivals: List[float], service: List[float]) -> List[float]:
        """Update queue using Q(t+1)=max(Q(t)-mu,0)+A with explicit arrivals/service."""

        updated: List[float] = []
        for q, a, mu in zip(self.queue, arrivals, service):
            a_clip = min(max(a, 0.0), self.arrival_clip)
            q_next = max(q - mu, 0.0) + a_clip
            updated.append(min(q_next, self.q_max))
        self.queue = updated
        return list(self.queue)

    def pressure(self) -> List[float]:
        return list(self.queue)
