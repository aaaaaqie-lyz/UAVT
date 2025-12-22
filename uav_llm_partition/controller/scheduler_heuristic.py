"""Heuristic scheduler implementing head-level placement with feasibility checks.

Aligned to MARL env scoring: uses lyapunov -> lyap_weight mapping,
comm_penalty_scale, migration_penalty_scale, and device type penalties.
Keeps blocks iteration order as provided (no layer reordering) and
does not estimate comm for neighbors that are not yet assigned, matching
the MARL env behavior.
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Optional

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
        mig_penalty_scale: float = 0.05,  # align with MARL env small scale
        migration_budget: int = 8,
        migration_volume_budget: float = 0.5,
        migration_improve_margin: float = 0.05,
        load_guard: float = 0.9,
        load_balance_bias: float = 0.15,
        preventive_threshold: float = 0.85,
        global_load_bias: float = 0.25,
        queue_penalty_scale: float = 0.4,
        preventive_queue_threshold: float = 0.6,
        queue_block_threshold: float | None = None,
        # MARL-like knobs
        comm_penalty_scale: float = 0.05,
        type_penalty: Dict[str, float] | None = None,
        retain_prev: bool = True,
        debug: bool = False,
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
        self.queue_block_threshold = queue_block_threshold

        self.comm_penalty_scale = comm_penalty_scale
        self.type_penalty = type_penalty or {"uav": 0.0, "edge": 0.2, "cloud": 0.35}
        self.retain_prev = retain_prev
        self.debug = debug

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
    ) -> Tuple[float, float, float, float]:
        """Return (comp_ratio, mem_ratio, comm_ratio, raw_comm_delay).

        Note: comm_delay only accounts for already-assigned neighbors or prev_assignment,
        matching MARL env semantics (no lookahead for unassigned neighbors).
        """
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
        comm_ratio = comm_delay / max(self.comm_budget, 1e-9)
        return comp_ratio, mem_ratio, comm_ratio, comm_delay

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

    def _retain_feasible_prev(
        self,
        blocks: List[Block],
        demands: Dict[Block, BlockDemand],
        compute: List[float],
        memory: List[float],
        prev_assignment: Dict[Block, int],
        dependencies: List[Tuple[Block, Block]],
        activation_sizes: Dict[Tuple[Block, Block], float],
        bandwidth: List[List[float]],
    ) -> Tuple[Dict[Block, int], List[float], List[float], List[Block]]:
        """Keep feasible prev_assignment entries (mirrors MARL env.retain_feasible_prev)."""
        kept: Dict[Block, int] = {}
        comp_used = [0.0 for _ in compute]
        mem_used = [0.0 for _ in memory]
        pending: List[Block] = []
        for block in blocks:
            prev_dev = prev_assignment.get(block)
            if prev_dev is None:
                pending.append(block)
                continue
            demand = demands[block]
            comp_future = comp_used[prev_dev] + demand.compute
            mem_future = mem_used[prev_dev] + demand.memory
            compute_ratio = comp_future / (compute[prev_dev] + 1e-6)
            memory_ratio = mem_future / (memory[prev_dev] + 1e-6)
            comm_ratio = 0.0
            for up, down in dependencies:
                if (up == block and down in kept and kept[down] != prev_dev) or (
                    down == block and up in kept and kept[up] != prev_dev
                ):
                    size = activation_sizes.get((up, down), 0.0)
                    src = kept.get(up, prev_dev)
                    dst = kept.get(down, prev_dev)
                    if src != dst:
                        comm_ratio += size / (bandwidth[src][dst] + 1e-6)
            score = max(compute_ratio, memory_ratio, comm_ratio)
            if score <= self.load_guard:
                kept[block] = prev_dev
                comp_used[prev_dev] = comp_future
                mem_used[prev_dev] = mem_future
            else:
                pending.append(block)
        return kept, comp_used, mem_used, pending

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
        device_types: List[str] | None = None,
    ) -> Tuple[Dict[Block, int], List[Tuple[Block, int, int]], bool, str]:
        # Optionally retain feasible previous assignments (to mirror MARL env)
        kept: Dict[Block, int] = {}
        comp_used = [0.0 for _ in compute]
        mem_used = [0.0 for _ in memory]
        pending_blocks = list(blocks)
        if self.retain_prev and prev_assignment:
            kept, comp_used, mem_used, pending_blocks = self._retain_feasible_prev(
                blocks, demands, compute, memory, prev_assignment, dependencies, activation_sizes, bandwidth
            )

        assignment: Dict[Block, int] = dict(kept)
        migrations: List[Tuple[Block, int, int]] = []
        migration_volume = 0.0
        failed = False
        failure_reason = ""
        device_types = list(device_types) if device_types is not None else ["uav" for _ in compute]

        # IMPORTANT: do not reorder blocks here — iterate in the provided order (like MARL)
        sorted_blocks = list(pending_blocks)

        for blk in sorted_blocks:
            demand = demands[blk]
            candidate_scores: List[Tuple[int, float, float, bool, bool, float, float, float, float, float]] = []
            for dev in range(len(compute)):
                global_load = max(
                    comp_used[dev] / max(compute[dev], 1e-6),
                    mem_used[dev] / max(memory[dev], 1e-6),
                )
                comp_ratio, mem_ratio, comm_ratio, comm_delay = self._ratios(
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

                # weight_factor preserved for backwards compatibility with existing weight handling
                weight_factor = 1.0 + self.weight_scale * weights[dev]
                comp_ratio_adj = comp_ratio / weight_factor
                mem_ratio_adj = mem_ratio / weight_factor
                comm_ratio_adj = comm_ratio / weight_factor

                base_score = max(comp_ratio_adj, mem_ratio_adj, comm_ratio_adj)

                # lyapunov -> lyap_weight mapping like MARL
                queue_pressure = lyapunov[dev] if dev < len(lyapunov) else 0.0
                lyap_weight = min(queue_pressure * 0.1, 3.0)

                # device type penalty
                type_pen = self.type_penalty.get(device_types[dev], 0.0)

                # final score composition aligned to MARL env
                final_score = base_score * (1.0 + lyap_weight)
                final_score += mig_penalty  # already scaled by mig_penalty_scale
                final_score += self.comm_penalty_scale * comm_delay
                final_score += type_pen

                load_term = max(
                    (comp_used[dev] + demand.compute) / max(compute[dev], 1e-6),
                    (mem_used[dev] + demand.memory) / max(memory[dev], 1e-6),
                )

                if self.queue_block_threshold is not None and queue_pressure > self.queue_block_threshold:
                    feasible = False
                else:
                    feasible = True
                global_ok = global_load <= self.load_guard or will_migrate
                feasible = (
                    feasible
                    and base_score <= 1.0
                    and max(comp_ratio_adj, mem_ratio_adj) <= self.load_guard
                    and global_ok
                    and (not will_migrate or len(migrations) < self.migration_budget)
                    and (not will_migrate or migration_volume + mig_volume <= self.migration_volume_budget)
                )

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
                # tie-break: minimize (final_score, load_term, global_load)
                dev, best_score, _, _, will_migrate, mig_volume, comp_r, mem_r, comm_r, load_term, global_load = min(
                    pool, key=lambda x: (x[1], x[9], x[10])
                )
                chosen_load = load_term
                if not feasible_candidates:
                    failure_reason = failure_reason or "guard_relaxed"
                # prefer previous assignment when comparable (stability)
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
                        queue_pressure_prev = lyapunov[prev_dev] if prev_dev < len(lyapunov) else 0.0
                        preemptive = prev_load >= self.preventive_threshold or queue_pressure_prev >= self.preventive_queue_threshold
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

            # optionally debug candidate breakdown for this block (single-step)
            if self.debug:
                try:
                    prim = next(c for c in candidate_scores if c[0] == dev)
                    print(f"DEBUG assign blk={blk} chosen_dev={dev} type={device_types[dev]} final_score={prim[1]:.6f} base={prim[2]:.6f} comp_r={prim[6]:.6f} mem_r={prim[7]:.6f} comm_r={prim[8]:.6f} mig_vol={prim[5]:.6f}")
                except StopIteration:
                    pass

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
