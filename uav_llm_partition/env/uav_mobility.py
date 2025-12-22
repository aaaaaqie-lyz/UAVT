"""Mobility model for UAVs plus static edge/cloud nodes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from uav_llm_partition.utils.num import rand_normal, rand_uniform, vector_norm, clip


@dataclass
class UAVState:
    position: List[float]  # length 3
    velocity: List[float]

    def step(self, dt: float, noise_std: float = 0.5) -> None:
        noise = rand_normal(0.0, noise_std, 3)
        self.position = [p + v * dt + n for p, v, n in zip(self.position, self.velocity, noise)]


class MobilityModel:
    """Simple mobility model with random drift and risk estimation."""

    def __init__(
        self, num_uav: int, box: Tuple[float, float, float] = (200.0, 200.0, 80.0), device_types: Sequence[str] | None = None
    ) -> None:
        self.num_uav = num_uav
        self.box = box
        device_types = list(device_types) if device_types is not None else ["uav" for _ in range(num_uav)]
        positions = [rand_uniform(0.0, b, num_uav) for b in box]
        velocities = [rand_uniform(-2.0, 2.0, num_uav) for _ in range(3)]
        self.states: List[UAVState] = []
        for i in range(num_uav):
            if device_types[i] in {"edge", "cloud"}:
                static_pos = [box[0] / 2.0, box[1] / 2.0, 1000.0 if device_types[i] == "cloud" else 10.0]
                self.states.append(UAVState(position=static_pos, velocity=[0.0, 0.0, 0.0]))
            else:
                self.states.append(
                    UAVState(
                        position=[positions[0][i], positions[1][i], positions[2][i]],
                        velocity=[velocities[0][i], velocities[1][i], velocities[2][i]],
                    )
                )

    def update(self, dt: float = 1.0) -> Tuple[List[List[float]], List[float]]:
        positions: List[List[float]] = []
        risks: List[float] = []
        for state in self.states:
            if any(abs(v) > 1e-9 for v in state.velocity):
                state.step(dt)
                state.position = [clip(p, 0.0, bound) for p, bound in zip(state.position, self.box)]
            speed = vector_norm(state.velocity)
            risk = min(speed / 5.0, 1.0)
            positions.append(list(state.position))
            risks.append(risk)
        return positions, risks
