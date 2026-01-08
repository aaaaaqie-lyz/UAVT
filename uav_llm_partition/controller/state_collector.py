"""Collects simulation state for downstream controllers or RL."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from uav_llm_partition.model_partition.blocks import Block
from uav_llm_partition.model_partition.demand_model import BlockDemand


@dataclass
class ControllerState:
    compute: List[float]
    memory: List[float]
    weights: List[float]
    lyapunov: List[float]
    positions: List[List[float]]
    los_score: List[float]
    mobility_risk: List[float]
    demands: Dict[Block, BlockDemand]


class StateCollector:
    def build_state(
        self,
        compute: List[float],
        memory: List[float],
        weights: List[float],
        lyapunov: List[float],
        positions: List[List[float]],
        los_score: List[float],
        mobility_risk: List[float],
        demands: Dict[Block, BlockDemand],
    ) -> ControllerState:
        return ControllerState(
            compute=list(compute),
            memory=list(memory),
            weights=list(weights),
            lyapunov=list(lyapunov),
            positions=[list(p) for p in positions],
            los_score=list(los_score),
            mobility_risk=list(mobility_risk),
            demands=demands,
        )

