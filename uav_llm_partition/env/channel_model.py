"""Channel model computing bandwidth and connectivity matrices using stdlib."""
from __future__ import annotations

from typing import List, Tuple
import math

from uav_llm_partition.utils.num import vector_norm, exp, mean


def _distance_matrix(positions: List[List[float]]) -> List[List[float]]:
    n = len(positions)
    dist = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            diff = [positions[i][k] - positions[j][k] for k in range(3)]
            dist[i][j] = vector_norm(diff)
    return dist


class ChannelModel:
    def __init__(self, base_rate: float = 5e3, los_decay: float = 120.0, outage_distance: float = 220.0) -> None:
        self.base_rate = base_rate
        self.los_decay = los_decay
        self.outage_distance = outage_distance

    def compute(self, positions: List[List[float]]) -> Tuple[List[List[float]], List[List[int]], List[float]]:
        dist = _distance_matrix(positions)
        n = len(positions)
        los_score = [[exp(-dist[i][j] / self.los_decay) for j in range(n)] for i in range(n)]
        bandwidth = [[self.base_rate * los_score[i][j] if i != j else 0.0 for j in range(n)] for i in range(n)]
        connectivity = [[1 if dist[i][j] < self.outage_distance and i != j else 0 for j in range(n)] for i in range(n)]
        los_per_node = [mean(row) for row in los_score]
        return bandwidth, connectivity, los_per_node

