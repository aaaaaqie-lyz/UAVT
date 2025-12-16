"""Heuristic scheduler implementing head-level placement and constraint repair."""
from __future__ import annotations

from typing import Dict, List, Tuple

from uav_llm_partition.model_partition.blocks import Block
from uav_llm_partition.model_partition.demand_model import BlockDemand
from uav_llm_partition.utils.num import argmin


class SchedulerHeuristic:
    def __init__(self, rho_w: float = 0.5, rho_q: float = 0.5, mig_overhead: float = 1.0) -> None:
        self.rho_w = rho_w
        self.rho_q = rho_q
        self.mig_overhead = mig_overhead

    def score(
        self,
        block: Block,
        demand: BlockDemand,
        device: int,
        compute: List[float],
        memory: List[float],
        weights: List[float],
        lyapunov: List[float],
        prev_assignment: Dict[Block, int],
    ) -> float:
        m_eff = memory[device] * (1 + self.rho_w * weights[device]) * (1 + self.rho_q / (1 + lyapunov[device]))
        c_eff = compute[device] * (1 + self.rho_w * weights[device]) * (1 + self.rho_q / (1 + lyapunov[device]))
        mem_frac = demand.memory / max(m_eff, 1e-6)
        comp_frac = demand.compute / max(c_eff, 1e-6)
        mig_penalty = 0.0
        if block in prev_assignment and prev_assignment[block] != device:
            kv_size = demand.kv_cache
            mig_penalty = kv_size / (len(compute) + 1e-6) + self.mig_overhead
        return max(mem_frac, comp_frac) + mig_penalty

    def assign(
        self,
        blocks: List[Block],
        demands: Dict[Block, BlockDemand],
        compute: List[float],
        memory: List[float],
        weights: List[float],
        lyapunov: List[float],
        prev_assignment: Dict[Block, int],
    ) -> Tuple[Dict[Block, int], List[Tuple[Block, int, int]], bool]:
        assignment: Dict[Block, int] = {}
        migrations: List[Tuple[Block, int, int]] = []
        sorted_blocks = sorted(blocks, key=lambda b: demands[b].memory, reverse=True)
        for blk in sorted_blocks:
            scores = [self.score(blk, demands[blk], dev, compute, memory, weights, lyapunov, prev_assignment) for dev in range(len(compute))]
            best_dev = argmin(scores)
            assignment[blk] = best_dev
            if blk in prev_assignment and prev_assignment[blk] != best_dev:
                migrations.append((blk, prev_assignment[blk], best_dev))
        assignment, migrations, failed = self._repair(assignment, demands, compute, memory, migrations)
        return assignment, migrations, failed

    def _repair(
        self,
        assignment: Dict[Block, int],
        demands: Dict[Block, BlockDemand],
        compute: List[float],
        memory: List[float],
        migrations: List[Tuple[Block, int, int]],
    ) -> Tuple[Dict[Block, int], List[Tuple[Block, int, int]], bool]:
        num_dev = len(compute)
        changed = True
        failed = False
        while changed:
            changed = False
            for dev in range(num_dev):
                mem_used = sum(demands[b].memory for b, d in assignment.items() if d == dev)
                comp_used = sum(demands[b].compute for b, d in assignment.items() if d == dev)
                if mem_used > memory[dev] or comp_used > compute[dev]:
                    candidates = [b for b, d in assignment.items() if d == dev]
                    if not candidates:
                        continue
                    heavy_block = max(candidates, key=lambda b: (demands[b].memory, demands[b].compute))
                    best_alt, best_score = None, float("inf")
                    for alt in range(num_dev):
                        if alt == dev:
                            continue
                        mem_alt = sum(demands[b].memory for b, d in assignment.items() if d == alt)
                        comp_alt = sum(demands[b].compute for b, d in assignment.items() if d == alt)
                        if mem_alt + demands[heavy_block].memory <= memory[alt] and comp_alt + demands[heavy_block].compute <= compute[alt]:
                            score_alt = mem_alt / memory[alt] + comp_alt / compute[alt]
                            if score_alt < best_score:
                                best_score = score_alt
                                best_alt = alt
                    if best_alt is not None:
                        old = assignment[heavy_block]
                        assignment[heavy_block] = best_alt
                        if (heavy_block, old, best_alt) not in migrations:
                            migrations.append((heavy_block, old, best_alt))
                        changed = True
                    else:
                        failed = True
        return assignment, migrations, failed

