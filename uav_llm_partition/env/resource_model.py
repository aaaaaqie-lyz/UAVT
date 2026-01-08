"""Resource model that provides compute and memory availability without numpy."""
from __future__ import annotations

from typing import List, Sequence, Tuple
import random


class ResourceModel:
    def __init__(self, base_compute: float, base_memory: float, noise: float = 0.1) -> None:
        self.base_compute = base_compute
        self.base_memory = base_memory
        self.noise = noise

    def sample(self, device_types: Sequence[str] | int) -> Tuple[List[float], List[float]]:
        """Return per-device compute/memory with heterogeneity by type.

        The legacy signature accepted an integer ``num_uav``; to preserve
        compatibility we still allow an ``int`` which is treated as that many
        generic ``"uav"`` entries.
        """

        if isinstance(device_types, int):
            device_types = ["uav" for _ in range(device_types)]

        compute: List[float] = []
        memory: List[float] = []
        for dev_type in device_types:
            if dev_type == "cloud":
                # Cloud nodes are stronger than UAVs but should remain bounded.
                base_c, base_m = self.base_compute * 10.0, self.base_memory * 10.0
            elif dev_type == "edge":
                # Edge nodes are stronger than UAVs but weaker than cloud.
                base_c, base_m = self.base_compute * 3.0, self.base_memory * 3.0
            else:
                base_c, base_m = self.base_compute, self.base_memory
            comp = base_c * (1.0 + random.uniform(-self.noise, self.noise))
            mem = base_m * (1.0 + random.uniform(-self.noise, self.noise))
            compute.append(comp)
            memory.append(mem)
        return compute, memory
