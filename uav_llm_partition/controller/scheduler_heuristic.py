"""Heuristic scheduler implementing head-level placement with feasibility checks."""
from __future__ import annotations

from typing import Dict, List, Tuple

from uav_llm_partition.model_partition.blocks import Block
from uav_llm_partition.model_partition.demand_model import BlockDemand


class SchedulerHeuristic:
    """Heuristic scheduler with DTIS-style feasibility checks and migration throttling."""

    def __init__(
        self,
        comm_budget: float = 1.0,
        lyapunov_penalty: float = 0.5,
        mig_overhead: float = 0.5,
        mig_penalty_scale: float = 1.0,
        migration_budget: int = 3,
    ) -> None:
        self.comm_budget = comm_budget
        self.lyapunov_penalty = lyapunov_penalty
        self.mig_overhead = mig_overhead
        self.mig_penalty_scale = mig_penalty_scale
        self.migration_budget = migration_budget

    def _ratios(
        self,
        block: Block,
        demand: BlockDemand,
        device: int,
        compute: List[float],
        memory: List[float],
        assignment: Dict[Block, int],
        prev_assignment: Dict[Block, int],
        dependencies: List[Tuple[Block, Block]],
        activation_sizes: Dict[Tuple[Block, Block], float],
        bandwidth: List[List[float]],
        comp_used: List[float],
        mem_used: List[float],
    ) -> Tuple[float, float, float]:
        comp_ratio = (comp_used[device] + demand.compute) / max(compute[device], 1e-6)
        mem_ratio = (mem_used[device] + demand.memory) / max(memory[device], 1e-6)
        comm_delay = 0.0
        for up, down in dependencies:
            if block not in (up, down):
                continue
            neighbor = down if block == up else up
            neighbor_dev = assignment.get(neighbor)
            if neighbor_dev is None:
                neighbor_dev = prev_assignment.get(neighbor)
            if neighbor_dev is None or neighbor_dev == device:
                continue
            size = activation_sizes.get((up, down), activation_sizes.get((down, up), 0.0))
            bw = bandwidth[device][neighbor_dev] + 1e-6
            comm_delay += size / bw
        comm_ratio = comm_delay / self.comm_budget
        return comp_ratio, mem_ratio, comm_ratio

    def _score(
        self,
        base_score: float,
        lyapunov: float,
        migration_penalty: float,
    ) -> float:
        lyap_factor = 1.0 + max(lyapunov, 0.0) * self.lyapunov_penalty
        return base_score * lyap_factor + migration_penalty

    def _migration_penalty(
        self,
        block: Block,
        demand: BlockDemand,
        device: int,
        prev_assignment: Dict[Block, int],
        bandwidth: List[List[float]],
        migrations_used: int,
    ) -> Tuple[float, bool]:
        if block not in prev_assignment or prev_assignment[block] == device:
            return 0.0, False
        src = prev_assignment[block]
        bw = bandwidth[src][device] + 1e-6
        mig_cost = (demand.kv_cache / bw + self.mig_overhead) * self.mig_penalty_scale
        if migrations_used >= self.migration_budget:
            mig_cost *= 2.0
        return mig_cost, True

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
    ) -> Tuple[Dict[Block, int], List[Tuple[Block, int, int]], bool]:
        del weights  # weights are unused in the DTIS-style scoring
        assignment: Dict[Block, int] = {}
        migrations: List[Tuple[Block, int, int]] = []
        comp_used = [0.0 for _ in compute]
        mem_used = [0.0 for _ in memory]
        failed = False
        priority = {"head": 0, "ffn": 1, "proj": 2}
        sorted_blocks = sorted(blocks, key=lambda b: (priority.get(b.kind, 3), -demands[b].memory, -demands[b].compute))

        for blk in sorted_blocks:
            demand = demands[blk]
            candidate_scores: List[Tuple[int, float, float, bool, bool]] = []
            for dev in range(len(compute)):
                comp_ratio, mem_ratio, comm_ratio = self._ratios(
                    blk,
                    demand,
                    dev,
                    compute,
                    memory,
                    assignment,
                    prev_assignment,
                    dependencies,
                    activation_sizes,
                    bandwidth,
                    comp_used,
                    mem_used,
                )
                mig_penalty, will_migrate = self._migration_penalty(
                    blk, demand, dev, prev_assignment, bandwidth, len(migrations)
                )
                comm_ratio_with_mig = (comm_ratio * self.comm_budget + mig_penalty) / self.comm_budget
                base_score = max(comp_ratio, mem_ratio, comm_ratio_with_mig)
                feasible = base_score <= 1.0 and (not will_migrate or len(migrations) < self.migration_budget)
                final_score = self._score(base_score, lyapunov[dev], 0.0)
                candidate_scores.append((dev, final_score, base_score, feasible, will_migrate))

            feasible_candidates = [c for c in candidate_scores if c[3]]
            if not feasible_candidates:
                failed = True
                dev, best_score, _, _, will_migrate = min(candidate_scores, key=lambda x: x[1])
            else:
                dev, best_score, _, _, will_migrate = min(feasible_candidates, key=lambda x: x[1])
                if blk in prev_assignment:
                    prev_dev = prev_assignment[blk]
                    prev_base = next((c[2] for c in candidate_scores if c[0] == prev_dev), float("inf"))
                    prev_score = next((c[1] for c in candidate_scores if c[0] == prev_dev), float("inf"))
                    if prev_base <= 1.0 and prev_score <= best_score:
                        dev = prev_dev
                        will_migrate = False

            if blk in prev_assignment and prev_assignment[blk] != dev and will_migrate:
                migrations.append((blk, prev_assignment[blk], dev))
            comp_used[dev] += demand.compute
            mem_used[dev] += demand.memory
            assignment[blk] = dev

        return assignment, migrations, failed
