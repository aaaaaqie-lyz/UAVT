"""Resource model that provides compute and memory availability without numpy."""
from __future__ import annotations

from typing import List, Tuple
import random


class ResourceModel:
    def __init__(self, base_compute: float, base_memory: float, noise: float = 0.1) -> None:
        self.base_compute = base_compute
        self.base_memory = base_memory
        self.noise = noise

    def sample(self, num_uav: int) -> Tuple[List[float], List[float]]:
        compute = [self.base_compute * (1.0 + random.uniform(-self.noise, self.noise)) for _ in range(num_uav)]
        memory = [self.base_memory * (1.0 + random.uniform(-self.noise, self.noise)) for _ in range(num_uav)]
        return compute, memory

