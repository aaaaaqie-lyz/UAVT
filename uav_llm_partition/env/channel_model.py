"""Channel model computing bandwidth and connectivity matrices using stdlib."""
from __future__ import annotations

from typing import List, Sequence, Tuple
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
    def __init__(self, base_rate: float = 0.5, los_decay: float = 120.0, outage_distance: float = 220.0) -> None:
        self.base_rate = base_rate
        self.los_decay = los_decay
        self.outage_distance = outage_distance

    def compute(
        self, positions: List[List[float]], device_types: Sequence[str] | None = None
    ) -> Tuple[List[List[float]], List[List[int]], List[float]]:
        device_types = list(device_types) if device_types is not None else ["uav" for _ in positions]
        dist = _distance_matrix(positions)
        n = len(positions)
        bandwidth = [[0.0 for _ in range(n)] for _ in range(n)]
        connectivity = [[0 for _ in range(n)] for _ in range(n)]
        los_score = [[0.0 for _ in range(n)] for _ in range(n)]

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                ti, tj = device_types[i], device_types[j]
                is_backbone = "cloud" in (ti, tj) or "edge" in (ti, tj)
                if ti == "cloud" and tj == "cloud":
                    bw_base = 100.0
                elif ("cloud" in (ti, tj)) and ("edge" in (ti, tj)):
                    bw_base = 10.0
                elif is_backbone:
                    bw_base = 1.0
                else:
                    bw_base = self.base_rate

                decay = self.los_decay * (4.0 if is_backbone else 1.0)
                los = exp(-dist[i][j] / decay)
                los_score[i][j] = los
                bandwidth[i][j] = bw_base * los if i != j else 0.0
                connectivity[i][j] = 1 if (is_backbone or dist[i][j] < self.outage_distance) and i != j else 0

        los_per_node = [mean([exp(-d / self.los_decay) for d in row]) for row in dist]
        return bandwidth, connectivity, los_per_node

