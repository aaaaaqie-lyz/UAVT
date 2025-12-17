"""Heuristic scheduler implementing head-level placement with feasibility checks."""
from __future__ import annotations

from typing import Dict, List, Tuple

from uav_llm_partition.model_partition.blocks import Block
from uav_llm_partition.model_partition.demand_model import BlockDemand


class SchedulerHeuristic:
    """Heuristic scheduler with DTIS-style feasibility checks and migration throttling."""

    def __init__(
        self,
        comm_budget: float = 0.05,
        lyapunov_penalty: float = 0.7,
        lyapunov_add_penalty: float = 0.1,
        weight_scale: float = 0.5,
        mig_overhead: float = 0.01,
        mig_penalty_scale: float = 1.0,
        migration_budget: int = 8,
        migration_volume_budget: float = 0.5,
        migration_improve_margin: float = 0.05,
        load_guard: float = 0.9,
        load_balance_bias: float = 0.15,
        preventive_threshold: float = 0.85,
        global_load_bias: float = 0.25,
        queue_penalty_scale: float = 0.4,
        preventive_queue_threshold: float = 0.6,
    ) -> None:
        self.comm_budget = comm_budget
        self.lyapunov_penalty = lyapunov_penalty
        self.lyapunov_add_penalty = lyapunov_add_penalty
        self.weight_scale = weight_scale
        self.mig_overhead = mig_overhead
        self.mig_penalty_scale = mig_penalty_scale
        self.migration_budget = migration_budget
        self.migration_volume_budget = migration_volume_budget
        self.migration_improve_margin = migration_improve_margin
        self.load_guard = load_guard
        self.load_balance_bias = load_balance_bias
        self.preventive_threshold = preventive_threshold
        self.global_load_bias = global_load_bias
        self.queue_penalty_scale = queue_penalty_scale
        self.preventive_queue_threshold = preventive_queue_threshold

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

    def _score(self, base_score: float, lyapunov: float, load_term: float, global_load: float) -> float:
        queue = max(lyapunov, 0.0)
        return (
            base_score
            * (1.0 + queue * self.lyapunov_penalty)
            * (1.0 + queue * self.queue_penalty_scale)
            + queue * self.lyapunov_add_penalty
            + self.load_balance_bias * load_term
            + self.global_load_bias * global_load
        )

    def _migration_penalty(
        self,
        block: Block,
        demand: BlockDemand,
        device: int,
        prev_assignment: Dict[Block, int],
        bandwidth: List[List[float]],
        migrations_used: int,
    ) -> Tuple[float, float, bool]:
        if block not in prev_assignment or prev_assignment[block] == device:
            return 0.0, 0.0, False
        src = prev_assignment[block]
        bw = bandwidth[src][device] + 1e-6
        mig_time = demand.kv_cache / bw + self.mig_overhead
        mig_cost = mig_time * self.mig_penalty_scale
        if migrations_used >= self.migration_budget:
            mig_cost *= 2.0
        return mig_cost, demand.kv_cache, True

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
    ) -> Tuple[Dict[Block, int], List[Tuple[Block, int, int]], bool, str]:
        assignment: Dict[Block, int] = {}
        migrations: List[Tuple[Block, int, int]] = []
        migration_volume = 0.0
        comp_used = [0.0 for _ in compute]
        mem_used = [0.0 for _ in memory]
        failed = False
        failure_reason = ""
        priority = {"head": 0, "ffn": 1, "proj": 2}
        sorted_blocks = sorted(blocks, key=lambda b: (priority.get(b.kind, 3), -demands[b].memory, -demands[b].compute))

        for blk in sorted_blocks:
            demand = demands[blk]
            candidate_scores: List[Tuple[int, float, float, bool, bool, float, float, float, float, float]] = []
            for dev in range(len(compute)):
                global_load = max(
                    comp_used[dev] / max(compute[dev], 1e-6),
                    mem_used[dev] / max(memory[dev], 1e-6),
                )
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
                mig_penalty, mig_volume, will_migrate = self._migration_penalty(
                    blk, demand, dev, prev_assignment, bandwidth, len(migrations)
                )
                comm_time = comm_ratio * self.comm_budget + mig_penalty
                comm_ratio_with_mig = comm_time / self.comm_budget

                weight_factor = 1.0 + self.weight_scale * weights[dev]
                comp_ratio_adj = comp_ratio / weight_factor
                mem_ratio_adj = mem_ratio / weight_factor
                comm_ratio_adj = comm_ratio_with_mig / weight_factor

                base_score = max(comp_ratio_adj, mem_ratio_adj, comm_ratio_adj)
                load_term = max(
                    (comp_used[dev] + demand.compute) / max(compute[dev], 1e-6),
                    (mem_used[dev] + demand.memory) / max(memory[dev], 1e-6),
                )
                global_ok = global_load <= self.load_guard or will_migrate
                feasible = (
                    base_score <= 1.0
                    and max(comp_ratio_adj, mem_ratio_adj) <= self.load_guard
                    and global_ok
                    and (not will_migrate or len(migrations) < self.migration_budget)
                    and (not will_migrate or migration_volume + mig_volume <= self.migration_volume_budget)
                )
                final_score = self._score(base_score, lyapunov[dev], load_term, global_load)
                candidate_scores.append(
                    (
                        dev,
                        final_score,
                        base_score,
                        feasible,
                        will_migrate,
                        mig_volume,
                        comp_ratio_adj,
                        mem_ratio_adj,
                        comm_ratio_adj,
                        load_term,
                        global_load,
                    )
                )

            feasible_candidates = [c for c in candidate_scores if c[3]]
            relaxed_candidates = [
                c
                for c in candidate_scores
                if c[2] <= 1.0 and c[5] + migration_volume <= self.migration_volume_budget
            ]
            min_base = min(c[2] for c in candidate_scores)
            if not feasible_candidates and not relaxed_candidates:
                failed = min_base > 1.0
                if failed:
                    failure_reason = "no_feasible"
                pool = candidate_scores
                dev, best_score, _, _, will_migrate, mig_volume, comp_r, mem_r, comm_r, load_term, global_load = min(
                    pool, key=lambda x: (x[1], x[9], x[10])
                )
                chosen_load = load_term
            else:
                pool = feasible_candidates if feasible_candidates else relaxed_candidates
                dev, best_score, _, _, will_migrate, mig_volume, comp_r, mem_r, comm_r, load_term, global_load = min(
                    pool, key=lambda x: (x[1], x[9], x[10])
                )
                chosen_load = load_term
                if not feasible_candidates:
                    failure_reason = failure_reason or "guard_relaxed"
                if blk in prev_assignment:
                    prev_dev = prev_assignment[blk]
                    prev_tuple = next((c for c in candidate_scores if c[0] == prev_dev), None)
                    if prev_tuple:
                        prev_base = prev_tuple[2]
                        prev_score = prev_tuple[1]
                        prev_load = prev_tuple[9]
                    else:
                        prev_base = float("inf")
                        prev_score = float("inf")
                        prev_load = float("inf")

                    if prev_base <= 1.0:
                        queue_pressure = lyapunov[prev_dev] if prev_dev < len(lyapunov) else 0.0
                        preemptive = prev_load >= self.preventive_threshold or queue_pressure >= self.preventive_queue_threshold
                        better_score = best_score + self.migration_improve_margin < prev_score
                        lower_load = chosen_load + self.migration_improve_margin < prev_load
                        if not preemptive and (prev_score <= best_score + self.migration_improve_margin or not will_migrate):
                            dev = prev_dev
                            will_migrate = False
                            mig_volume = 0.0
                        elif preemptive and not (better_score or lower_load):
                            dev = prev_dev
                            will_migrate = False
                            mig_volume = 0.0

            if blk in prev_assignment and prev_assignment[blk] != dev and will_migrate:
                migrations.append((blk, prev_assignment[blk], dev))
                migration_volume += mig_volume
            comp_used[dev] += demand.compute
            mem_used[dev] += demand.memory
            assignment[blk] = dev

        if migration_volume > self.migration_volume_budget and not failed:
            failed = True
            failure_reason = "migration_budget"
        if failed and not failure_reason:
            failure_reason = "constraint"
        return assignment, migrations, failed, failure_reason
