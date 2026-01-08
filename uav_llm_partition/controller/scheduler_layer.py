"""Layer-level partition baseline for coarse-grained comparison."""
from __future__ import annotations

from typing import Dict, List, Tuple

from uav_llm_partition.model_partition.blocks import Block
from uav_llm_partition.model_partition.demand_model import BlockDemand


class LayerPartitionScheduler:
    """Assign whole layers (heads+proj+ffn) to a single device."""

    def __init__(
        self,
        strategy: str = "round_robin",
        comm_budget: float = 0.05,
        cloud_fallback_only: bool = True,
        cloud_bias: float = 1.0,
    ) -> None:
        self.strategy = strategy
        self.comm_budget = comm_budget
        self._rr_index = 0
        self.mig_overhead = 0.01
        self.cloud_fallback_only = cloud_fallback_only
        self.cloud_bias = cloud_bias

    def _aggregate_by_layer(self, blocks: List[Block], demands: Dict[Block, BlockDemand]) -> Dict[int, BlockDemand]:
        grouped: Dict[int, BlockDemand] = {}
        for blk in blocks:
            d = demands[blk]
            if blk.layer not in grouped:
                grouped[blk.layer] = BlockDemand(memory=0.0, compute=0.0, kv_cache=0.0)
            current = grouped[blk.layer]
            grouped[blk.layer] = BlockDemand(
                memory=current.memory + d.memory,
                compute=current.compute + d.compute,
                kv_cache=current.kv_cache + d.kv_cache,
            )
        return grouped

    def _score(
        self,
        demand: BlockDemand,
        dev: int,
        compute: List[float],
        memory: List[float],
        loads: List[float],
        dev_type: str,
    ) -> float:
        comp_ratio = demand.compute / (compute[dev] + 1e-6)
        mem_ratio = demand.memory / (memory[dev] + 1e-6)
        base = max(comp_ratio, mem_ratio)
        bias = self.cloud_bias if dev_type == "cloud" else 0.0
        return base + loads[dev] + bias

    def assign(
        self,
        blocks: List[Block],
        demands: Dict[Block, BlockDemand],
        compute: List[float],
        memory: List[float],
        weights: List[float],
        lyapunov: List[float],
        prev_assignment: Dict[Block, int],
        dependencies: List[Tuple[Block, Block]],
        activation_sizes: Dict[Tuple[Block, Block], float],
        bandwidth: List[List[float]],
        latency: List[List[float]],
        device_types: List[str] | None = None,
    ) -> Tuple[Dict[Block, int], List[Tuple[Block, int, int]], bool, str]:
        layer_demands = self._aggregate_by_layer(blocks, demands)
        comp_used = [0.0 for _ in compute]
        mem_used = [0.0 for _ in memory]
        assignment: Dict[Block, int] = {}
        migrations: List[Tuple[Block, int, int]] = []
        failed = False
        reason = ""

        device_types = device_types or ["uav" for _ in compute]
        for layer in sorted(layer_demands.keys()):
            demand = layer_demands[layer]
            chosen = None
            if self.strategy == "round_robin":
                for attempt in range(len(compute)):
                    dev = (self._rr_index + attempt) % len(compute)
                    dev_type = device_types[dev] if dev < len(device_types) else "uav"
                    score = self._score(demand, dev, compute, memory, [0.0 for _ in compute], dev_type)
                    if self.cloud_fallback_only and dev_type == "cloud":
                        attempts_left = any(
                            self._score(
                                demand,
                                cand,
                                compute,
                                memory,
                                [0.0 for _ in compute],
                                device_types[cand] if cand < len(device_types) else "uav",
                            )
                            <= 1.0
                            for cand in range(len(compute))
                            if (device_types[cand] if cand < len(device_types) else "uav") != "cloud"
                        )
                        if attempts_left:
                            continue
                    if score <= 1.0:
                        chosen = dev
                        self._rr_index = dev + 1
                        break
            elif self.strategy == "min_load":
                best_score = None
                for dev in range(len(compute)):
                    dev_type = device_types[dev] if dev < len(device_types) else "uav"
                    load_term = max(
                        (comp_used[dev] + demand.compute) / (compute[dev] + 1e-6),
                        (mem_used[dev] + demand.memory) / (memory[dev] + 1e-6),
                    )
                    if load_term > 1.0:
                        continue
                    if self.cloud_fallback_only and dev_type == "cloud":
                        continue
                    if best_score is None or load_term < best_score:
                        best_score = load_term
                        chosen = dev
            else:  # resource_aware
                best_score = None
                for dev in range(len(compute)):
                    dev_type = device_types[dev] if dev < len(device_types) else "uav"
                    score = self._score(
                        demand, dev, compute, memory, [0.0 for _ in compute], dev_type
                    ) / (1.0 + max(weights[dev], 0.0))
                    if score > 1.0:
                        continue
                    if self.cloud_fallback_only and dev_type == "cloud":
                        continue
                    if best_score is None or score < best_score:
                        best_score = score
                        chosen = dev
            if chosen is None:
                failed = True
                reason = "layer_no_feasible"
                break
            comp_used[chosen] += demand.compute
            mem_used[chosen] += demand.memory
            for blk in [b for b in blocks if b.layer == layer]:
                if blk in prev_assignment and prev_assignment[blk] != chosen:
                    migrations.append((blk, prev_assignment[blk], chosen))
                assignment[blk] = chosen

        return assignment, migrations, failed, reason
