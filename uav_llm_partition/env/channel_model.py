"""Channel model computing bandwidth and connectivity matrices using stdlib."""
from __future__ import annotations

from typing import List, Sequence, Tuple
import math

from uav_llm_partition.utils.num import vector_norm, exp, mean, rand_normal, clip


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
    def __init__(
        self,
        base_rate: float = 80.0,
        los_decay: float = 120.0,
        outage_distance: float = 220.0,
        fading_rho: float = 0.9,
        fading_sigma: float = 0.15,
    ) -> None:
        self.base_rate = base_rate
        self.los_decay = los_decay
        self.outage_distance = outage_distance
        self.fading_rho = fading_rho
        self.fading_sigma = fading_sigma
        self._fading: List[List[float]] = []

    def _update_fading(self, n: int) -> None:
        if len(self._fading) != n:
            self._fading = [[1.0 for _ in range(n)] for _ in range(n)]
        noise = rand_normal(0.0, self.fading_sigma, n * n)
        idx = 0
        for i in range(n):
            for j in range(n):
                if i == j:
                    self._fading[i][j] = 1.0
                else:
                    value = self.fading_rho * self._fading[i][j] + (1.0 - self.fading_rho) * (1.0 + noise[idx])
                    self._fading[i][j] = clip(value, 0.3, 1.7)
                idx += 1

    def compute(
        self, positions: List[List[float]], device_types: Sequence[str] | None = None
    ) -> Tuple[List[List[float]], List[List[int]], List[float], List[List[float]], List[float], List[float]]:
        device_types = list(device_types) if device_types is not None else ["uav" for _ in positions]
        dist = _distance_matrix(positions)
        n = len(positions)
        bandwidth = [[0.0 for _ in range(n)] for _ in range(n)]
        connectivity = [[0 for _ in range(n)] for _ in range(n)]
        los_score = [[0.0 for _ in range(n)] for _ in range(n)]
        latency = [[0.0 for _ in range(n)] for _ in range(n)]
        rssi = [[0.0 for _ in range(n)] for _ in range(n)]
        snr = [[0.0 for _ in range(n)] for _ in range(n)]
        self._update_fading(n)

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                ti, tj = device_types[i], device_types[j]
                is_backbone = "cloud" in (ti, tj) or "edge" in (ti, tj)
                if ti == "cloud" and tj == "cloud":
                    bw_base = 60.0
                    lat_base = 0.04
                elif ("cloud" in (ti, tj)) and ("edge" in (ti, tj)):
                    bw_base = 20.0
                    lat_base = 0.05
                elif "cloud" in (ti, tj):
                    bw_base = 10.0
                    lat_base = 0.05
                elif "edge" in (ti, tj):
                    bw_base = 30.0
                    lat_base = 0.02
                else:
                    bw_base = self.base_rate
                    lat_base = 0.005

                decay = self.los_decay * (4.0 if is_backbone else 1.0)
                los = exp(-dist[i][j] / decay)
                los_score[i][j] = los
                connected = (is_backbone or dist[i][j] < self.outage_distance) and i != j
                connectivity[i][j] = 1 if connected else 0
                if connected:
                    link_bw = bw_base * los * self._fading[i][j]
                    bandwidth[i][j] = link_bw
                    latency[i][j] = lat_base
                    rssi[i][j] = link_bw / max(dist[i][j], 1e-3)
                    snr[i][j] = 10.0 * math.log10(rssi[i][j] + 1e-6)
                else:
                    bandwidth[i][j] = 0.0
                    latency[i][j] = lat_base * 10.0
                    rssi[i][j] = 0.0
                    snr[i][j] = 0.0

        los_per_node = [mean([exp(-d / self.los_decay) for d in row]) for row in dist]
        snr_per_node = [
            mean([snr[i][j] for j in range(n) if i != j and connectivity[i][j]]) for i in range(n)
        ]
        rssi_per_node = [
            mean([rssi[i][j] for j in range(n) if i != j and connectivity[i][j]]) for i in range(n)
        ]
        return bandwidth, connectivity, los_per_node, latency, snr_per_node, rssi_per_node
