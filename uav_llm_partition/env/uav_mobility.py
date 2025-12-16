"""Mobility model for UAV positions and mobility risk without external deps."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import random

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

    def __init__(self, num_uav: int, box: Tuple[float, float, float] = (200.0, 200.0, 80.0)) -> None:
        self.num_uav = num_uav
        self.box = box
        positions = [rand_uniform(0.0, b, num_uav) for b in box]
        velocities = [rand_uniform(-2.0, 2.0, num_uav) for _ in range(3)]
        self.states = [UAVState(position=[positions[0][i], positions[1][i], positions[2][i]], velocity=[velocities[0][i], velocities[1][i], velocities[2][i]]) for i in range(num_uav)]

    def update(self, dt: float = 1.0) -> Tuple[List[List[float]], List[float]]:
        positions: List[List[float]] = []
        risks: List[float] = []
        for state in self.states:
            state.step(dt)
            state.position = [clip(p, 0.0, bound) for p, bound in zip(state.position, self.box)]
            speed = vector_norm(state.velocity)
            risk = min(speed / 5.0, 1.0)
            positions.append(list(state.position))
            risks.append(risk)
        return positions, risks

