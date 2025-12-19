"""Baseline schedulers for comparison with the heuristic policy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from uav_llm_partition.model_partition.blocks import Block
from uav_llm_partition.model_partition.demand_model import BlockDemand


@dataclass
class SchedulerResult:
    assignment: Dict[Block, int]
    migrations: List[Tuple[Block, int, int]]
    failed: bool
    reason: str


class BaseScheduler:
    """Base scheduler with shared ratio utilities."""

    def __init__(self, comm_budget: float = 0.05, mig_overhead: float = 0.01) -> None:
        self.comm_budget = comm_budget
        self.mig_overhead = mig_overhead
        self._rr_index = 0

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
    ) -> SchedulerResult:
        raise NotImplementedError


class GreedyScheduler(BaseScheduler):
    """Greedy placement minimizing DTIS-style score per block."""

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
    ) -> SchedulerResult:
        assignment: Dict[Block, int] = {}
        comp_used = [0.0 for _ in compute]
        mem_used = [0.0 for _ in memory]
        migrations: List[Tuple[Block, int, int]] = []
        failed = False
        reason = ""

        sorted_blocks = sorted(blocks, key=lambda b: (demands[b].memory, demands[b].compute), reverse=True)
        for blk in sorted_blocks:
            demand = demands[blk]
            best: Tuple[float, int] | None = None
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
                base_score = max(comp_ratio, mem_ratio, comm_ratio)
                if base_score > 1.0:
                    continue
                if best is None or base_score < best[0]:
                    best = (base_score, dev)
            if best is None:
                failed = True
                reason = "no_feasible"
                break
            chosen_dev = best[1]
            if blk in prev_assignment and prev_assignment[blk] != chosen_dev:
                migrations.append((blk, prev_assignment[blk], chosen_dev))
            comp_used[chosen_dev] += demand.compute
            mem_used[chosen_dev] += demand.memory
            assignment[blk] = chosen_dev

        return SchedulerResult(assignment, migrations, failed, reason)


class MinLoadScheduler(BaseScheduler):
    """Always pick the currently lowest-load device among feasible options."""

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
    ) -> SchedulerResult:
        assignment: Dict[Block, int] = {}
        comp_used = [0.0 for _ in compute]
        mem_used = [0.0 for _ in memory]
        migrations: List[Tuple[Block, int, int]] = []
        failed = False
        reason = ""

        for blk in blocks:
            demand = demands[blk]
            best_dev = None
            best_load = None
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
                base_score = max(comp_ratio, mem_ratio, comm_ratio)
                if base_score > 1.0:
                    continue
                load_after = max(
                    (comp_used[dev] + demand.compute) / max(compute[dev], 1e-6),
                    (mem_used[dev] + demand.memory) / max(memory[dev], 1e-6),
                )
                if best_load is None or load_after < best_load:
                    best_load = load_after
                    best_dev = dev
            if best_dev is None:
                failed = True
                reason = "no_feasible"
                break
            if blk in prev_assignment and prev_assignment[blk] != best_dev:
                migrations.append((blk, prev_assignment[blk], best_dev))
            comp_used[best_dev] += demand.compute
            mem_used[best_dev] += demand.memory
            assignment[blk] = best_dev

        return SchedulerResult(assignment, migrations, failed, reason)


class RoundRobinScheduler(BaseScheduler):
    """Round-robin assignment with feasibility guard."""

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
    ) -> SchedulerResult:
        assignment: Dict[Block, int] = {}
        comp_used = [0.0 for _ in compute]
        mem_used = [0.0 for _ in memory]
        migrations: List[Tuple[Block, int, int]] = []
        failed = False
        reason = ""

        for blk in blocks:
            demand = demands[blk]
            attempts = 0
            chosen = None
            while attempts < len(compute):
                dev = (self._rr_index + attempts) % len(compute)
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
                if max(comp_ratio, mem_ratio, comm_ratio) <= 1.0:
                    chosen = dev
                    self._rr_index = dev + 1
                    break
                attempts += 1
            if chosen is None:
                failed = True
                reason = "no_feasible"
                break
            if blk in prev_assignment and prev_assignment[blk] != chosen:
                migrations.append((blk, prev_assignment[blk], chosen))
            comp_used[chosen] += demand.compute
            mem_used[chosen] += demand.memory
            assignment[blk] = chosen

        return SchedulerResult(assignment, migrations, failed, reason)


class ResourceAwareGreedyScheduler(BaseScheduler):
    """Greedy scheduler biased to devices with higher normalized weights."""

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
    ) -> SchedulerResult:
        assignment: Dict[Block, int] = {}
        comp_used = [0.0 for _ in compute]
        mem_used = [0.0 for _ in memory]
        migrations: List[Tuple[Block, int, int]] = []
        failed = False
        reason = ""

        for blk in blocks:
            demand = demands[blk]
            best: Tuple[float, int] | None = None
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
                weight_factor = 1.0 + max(weights[dev], 0.0)
                base_score = max(comp_ratio, mem_ratio, comm_ratio) / weight_factor
                if base_score > 1.0:
                    continue
                if best is None or base_score < best[0]:
                    best = (base_score, dev)
            if best is None:
                failed = True
                reason = "no_feasible"
                break
            chosen_dev = best[1]
            if blk in prev_assignment and prev_assignment[blk] != chosen_dev:
                migrations.append((blk, prev_assignment[blk], chosen_dev))
            comp_used[chosen_dev] += demand.compute
            mem_used[chosen_dev] += demand.memory
            assignment[blk] = chosen_dev

        return SchedulerResult(assignment, migrations, failed, reason)


class DPScheduler(BaseScheduler):
    """Simplified dynamic-programming/beam search scheduler minimizing max load."""

    def __init__(self, comm_budget: float = 0.05, mig_overhead: float = 0.01, beam_width: int = 12) -> None:
        super().__init__(comm_budget=comm_budget, mig_overhead=mig_overhead)
        self.beam_width = beam_width

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
    ) -> SchedulerResult:
        State = Tuple[int, Tuple[float, ...], Tuple[float, ...], Dict[Block, int], List[Tuple[Block, int, int]]]
        initial: State = (0, tuple(0.0 for _ in compute), tuple(0.0 for _ in memory), {}, [])
        beam: List[State] = [initial]

        for blk in blocks:
            demand = demands[blk]
            new_beam: List[State] = []
            for idx, comp_used_t, mem_used_t, assignment, migrations in beam:
                comp_used = list(comp_used_t)
                mem_used = list(mem_used_t)
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
                    base_score = max(comp_ratio, mem_ratio, comm_ratio)
                    if base_score > 1.0:
                        continue
                    next_assignment = dict(assignment)
                    next_assignment[blk] = dev
                    next_comp = list(comp_used)
                    next_mem = list(mem_used)
                    next_comp[dev] += demand.compute
                    next_mem[dev] += demand.memory
                    next_migs = list(migrations)
                    if blk in prev_assignment and prev_assignment[blk] != dev:
                        next_migs.append((blk, prev_assignment[blk], dev))
                    next_state: State = (idx + 1, tuple(next_comp), tuple(next_mem), next_assignment, next_migs)
                    new_beam.append(next_state)
            if not new_beam:
                return SchedulerResult({}, [], True, "no_feasible")
            new_beam.sort(key=lambda s: max(
                max(c / (cap + 1e-6) for c, cap in zip(s[1], compute)),
                max(m / (cap + 1e-6) for m, cap in zip(s[2], memory)),
            ))
            beam = new_beam[: self.beam_width]

        best_state = min(
            beam,
            key=lambda s: max(
                max(c / (cap + 1e-6) for c, cap in zip(s[1], compute)),
                max(m / (cap + 1e-6) for m, cap in zip(s[2], memory)),
            ),
        )
        return SchedulerResult(best_state[3], best_state[4], False, "")

