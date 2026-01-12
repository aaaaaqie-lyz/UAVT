"""Baseline schedulers for comparison with the heuristic policy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import random

from uav_llm_partition.controller.constants import (
    comm_penalty_scale,
    lyap_to_weight_factor,
    lyap_weight_clip,
    mig_overhead,
    migration_penalty_scale,
    type_penalty,
)
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

    def __init__(
        self,
        comm_budget: float = 0.05,
        mig_overhead: float = mig_overhead,
        cloud_block_budget: int | None = 1,
        cloud_volume_budget: float | None = 0.5,
        cloud_fallback_only: bool = True,
        cloud_bias: float = 1.0,
    ) -> None:
        self.comm_budget = comm_budget
        self.mig_overhead = mig_overhead
        self._rr_index = 0
        # Bias placements toward UAVs even when edge/cloud are resource-rich.
        self.type_penalty = type_penalty
        self.cloud_block_budget = cloud_block_budget
        self.cloud_volume_budget = cloud_volume_budget
        self.cloud_fallback_only = cloud_fallback_only
        self.cloud_bias = cloud_bias

    def _cloud_usage(
        self,
        assignment: Dict[Block, int],
        demands: Dict[Block, BlockDemand],
        device_types: List[str],
    ) -> Tuple[int, float]:
        cloud_blocks = 0
        cloud_volume = 0.0
        for blk, dev in assignment.items():
            if dev < len(device_types) and device_types[dev] == "cloud":
                cloud_blocks += 1
                cloud_volume += demands[blk].memory
        return cloud_blocks, cloud_volume

    def _cloud_budget_ok(
        self,
        dev: int,
        demand: BlockDemand,
        device_types: List[str],
        cloud_blocks: int,
        cloud_volume: float,
        non_cloud_feasible: bool,
    ) -> bool:
        dev_type = device_types[dev] if dev < len(device_types) else "uav"
        if dev_type != "cloud":
            return True
        if self.cloud_fallback_only and non_cloud_feasible:
            return False
        block_budget_ok = self.cloud_block_budget is None or cloud_blocks < self.cloud_block_budget
        volume_budget_ok = self.cloud_volume_budget is None or cloud_volume + demand.memory <= self.cloud_volume_budget
        return block_budget_ok and volume_budget_ok

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
        latency: List[List[float]] | None,
        comp_used: List[float],
        mem_used: List[float],
        device_types: List[str],
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
            comm_delay += size / bw + (latency[device][neighbor_dev] if latency else 0.0)
        comm_ratio = comm_delay / self.comm_budget
        type_penalty = self.type_penalty.get(device_types[device], 0.0)
        cloud_penalty = self.cloud_bias if device_types[device] == "cloud" else 0.0
        return comp_ratio, mem_ratio, comm_ratio + type_penalty + cloud_penalty

    def _ratios_with_comm_delay(
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
        latency: List[List[float]] | None,
        comp_used: List[float],
        mem_used: List[float],
        device_types: List[str],
    ) -> Tuple[float, float, float, float]:
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
            comm_delay += size / bw + (latency[device][neighbor_dev] if latency else 0.0)
        comm_ratio = comm_delay / self.comm_budget
        type_penalty = self.type_penalty.get(device_types[device], 0.0)
        cloud_penalty = self.cloud_bias if device_types[device] == "cloud" else 0.0
        comm_ratio += type_penalty + cloud_penalty
        return comp_ratio, mem_ratio, comm_ratio, comm_delay

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
        latency: List[List[float]] | None,
        device_types: List[str] | None = None,
    ) -> SchedulerResult:
        raise NotImplementedError


class GreedyScheduler(BaseScheduler):
    """Greedy placement minimizing DTIS-style score per block."""

    def __init__(
        self,
        comm_budget: float = 0.05,
        mig_overhead: float = mig_overhead,
        migration_budget: int = 1,
        migration_hold_steps: int = 5,
    ) -> None:
        super().__init__(comm_budget=comm_budget, mig_overhead=mig_overhead)
        self.migration_budget = migration_budget
        self.migration_hold_steps = migration_hold_steps
        self._cooldowns: Dict[Block, int] = {}

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
        latency: List[List[float]] | None,
        device_types: List[str] | None = None,
    ) -> SchedulerResult:
        assignment: Dict[Block, int] = {}
        comp_used = [0.0 for _ in compute]
        mem_used = [0.0 for _ in memory]
        migrations: List[Tuple[Block, int, int]] = []
        failed = False
        reason = ""
        device_types = device_types or ["uav" for _ in compute]

        for blk in list(self._cooldowns.keys()):
            self._cooldowns[blk] = max(self._cooldowns[blk] - 1, 0)
            if self._cooldowns[blk] == 0:
                self._cooldowns.pop(blk)

        sorted_blocks = sorted(blocks, key=lambda b: (demands[b].memory, demands[b].compute), reverse=True)
        cloud_blocks, cloud_volume = self._cloud_usage(prev_assignment, demands, device_types)
        for blk in sorted_blocks:
            demand = demands[blk]
            best: Tuple[float, int] | None = None
            non_cloud_feasible = False
            candidates: List[Tuple[float, int]] = []
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
                    latency,
                    comp_used,
                    mem_used,
                    device_types,
                )
                if max(comp_ratio, mem_ratio) > 1.0:
                    continue
                base_score = max(comp_ratio, mem_ratio, comm_ratio)
                if device_types[dev] != "cloud":
                    non_cloud_feasible = True
                candidates.append((base_score, dev))
            for base_score, dev in candidates:
                if not self._cloud_budget_ok(
                    dev, demand, device_types, cloud_blocks, cloud_volume, non_cloud_feasible
                ):
                    continue
                if best is None or base_score < best[0]:
                    best = (base_score, dev)
            if best is None:
                failed = True
                reason = "fallback_prev"
                break
            chosen_dev = best[1]
            if blk in prev_assignment and prev_assignment[blk] != chosen_dev:
                if len(migrations) >= self.migration_budget or blk in self._cooldowns:
                    chosen_dev = prev_assignment[blk]
                else:
                    self._cooldowns[blk] = self.migration_hold_steps
                    migrations.append((blk, prev_assignment[blk], chosen_dev))
            if device_types[chosen_dev] == "cloud":
                cloud_blocks += 1
                cloud_volume += demand.memory
            comp_used[chosen_dev] += demand.compute
            mem_used[chosen_dev] += demand.memory
            assignment[blk] = chosen_dev

        if failed:
            fallback = {blk: prev_assignment.get(blk, assignment.get(blk, 0)) for blk in blocks}
            return SchedulerResult(fallback, [], False, reason)
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
        latency: List[List[float]] | None,
        device_types: List[str] | None = None,
    ) -> SchedulerResult:
        assignment: Dict[Block, int] = {}
        comp_used = [0.0 for _ in compute]
        mem_used = [0.0 for _ in memory]
        migrations: List[Tuple[Block, int, int]] = []
        failed = False
        reason = ""
        device_types = device_types or ["uav" for _ in compute]
        cloud_blocks, cloud_volume = self._cloud_usage(prev_assignment, demands, device_types)

        for blk in blocks:
            demand = demands[blk]
            best_dev = None
            best_load = None
            non_cloud_feasible = False
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
                    latency,
                    comp_used,
                    mem_used,
                    device_types,
                )
                if max(comp_ratio, mem_ratio) > 1.0:
                    continue
                base_score = max(comp_ratio, mem_ratio, comm_ratio)
                if device_types[dev] != "cloud":
                    non_cloud_feasible = True
                load_after = max(
                    (comp_used[dev] + demand.compute) / max(compute[dev], 1e-6),
                    (mem_used[dev] + demand.memory) / max(memory[dev], 1e-6),
                )
                if not self._cloud_budget_ok(
                    dev, demand, device_types, cloud_blocks, cloud_volume, non_cloud_feasible
                ):
                    continue
                if best_load is None or load_after < best_load:
                    best_load = load_after
                    best_dev = dev
            if best_dev is None:
                failed = True
                reason = "no_feasible"
                break
            if blk in prev_assignment and prev_assignment[blk] != best_dev:
                migrations.append((blk, prev_assignment[blk], best_dev))
            if device_types[best_dev] == "cloud":
                cloud_blocks += 1
                cloud_volume += demand.memory
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
        latency: List[List[float]] | None,
        device_types: List[str] | None = None,
    ) -> SchedulerResult:
        assignment: Dict[Block, int] = {}
        comp_used = [0.0 for _ in compute]
        mem_used = [0.0 for _ in memory]
        migrations: List[Tuple[Block, int, int]] = []
        failed = False
        reason = ""
        device_types = device_types or ["uav" for _ in compute]
        cloud_blocks, cloud_volume = self._cloud_usage(prev_assignment, demands, device_types)

        for blk in blocks:
            demand = demands[blk]
            attempts = 0
            chosen = None
            non_cloud_feasible = False
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
                    latency,
                    comp_used,
                    mem_used,
                    device_types,
                )
                if max(comp_ratio, mem_ratio) <= 1.0:
                    if device_types[dev] != "cloud":
                        non_cloud_feasible = True
                    if not self._cloud_budget_ok(
                        dev, demand, device_types, cloud_blocks, cloud_volume, non_cloud_feasible
                    ):
                        attempts += 1
                        continue
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
            if device_types[chosen] == "cloud":
                cloud_blocks += 1
                cloud_volume += demand.memory
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
        latency: List[List[float]] | None,
        device_types: List[str] | None = None,
    ) -> SchedulerResult:
        assignment: Dict[Block, int] = {}
        comp_used = [0.0 for _ in compute]
        mem_used = [0.0 for _ in memory]
        migrations: List[Tuple[Block, int, int]] = []
        failed = False
        reason = ""
        device_types = device_types or ["uav" for _ in compute]
        cloud_blocks, cloud_volume = self._cloud_usage(prev_assignment, demands, device_types)

        for blk in blocks:
            demand = demands[blk]
            best: Tuple[float, int] | None = None
            non_cloud_feasible = False
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
                    latency,
                    comp_used,
                    mem_used,
                    device_types,
                )
                if max(comp_ratio, mem_ratio) > 1.0:
                    continue
                weight_factor = 1.0 + max(weights[dev], 0.0)
                base_score = max(comp_ratio, mem_ratio, comm_ratio) / weight_factor
                if device_types[dev] != "cloud":
                    non_cloud_feasible = True
                if not self._cloud_budget_ok(
                    dev, demand, device_types, cloud_blocks, cloud_volume, non_cloud_feasible
                ):
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
            if device_types[chosen_dev] == "cloud":
                cloud_blocks += 1
                cloud_volume += demand.memory
            comp_used[chosen_dev] += demand.compute
            mem_used[chosen_dev] += demand.memory
            assignment[blk] = chosen_dev

        return SchedulerResult(assignment, migrations, failed, reason)


class DPScheduler(BaseScheduler):
    """Simplified dynamic-programming/beam search scheduler minimizing max load."""

    def __init__(self, comm_budget: float = 0.05, mig_overhead: float = mig_overhead, beam_width: int = 12) -> None:
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
        latency: List[List[float]] | None,
        device_types: List[str] | None = None,
    ) -> SchedulerResult:
        State = Tuple[
            int,
            Tuple[float, ...],
            Tuple[float, ...],
            Dict[Block, int],
            List[Tuple[Block, int, int]],
            float,
        ]
        initial: State = (0, tuple(0.0 for _ in compute), tuple(0.0 for _ in memory), {}, [], 0.0)
        beam: List[State] = [initial]
        device_types = device_types or ["uav" for _ in compute]
        fallback_assignment = {blk: prev_assignment.get(blk, 0) for blk in blocks}

        for blk in blocks:
            demand = demands[blk]
            new_beam: List[State] = []
            for idx, comp_used_t, mem_used_t, assignment, migrations, score in beam:
                comp_used = list(comp_used_t)
                mem_used = list(mem_used_t)
                cloud_blocks, cloud_volume = self._cloud_usage(assignment, demands, device_types)
                non_cloud_feasible = False
                for dev in range(len(compute)):
                    comp_ratio, mem_ratio, comm_ratio, _ = self._ratios_with_comm_delay(
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
                        latency,
                        comp_used,
                        mem_used,
                        device_types,
                    )
                    if max(comp_ratio, mem_ratio) <= 1.0 and device_types[dev] != "cloud":
                        non_cloud_feasible = True
                for dev in range(len(compute)):
                    comp_ratio, mem_ratio, comm_ratio, comm_delay = self._ratios_with_comm_delay(
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
                        latency,
                        comp_used,
                        mem_used,
                        device_types,
                    )
                    base_score = max(comp_ratio, mem_ratio, comm_ratio)
                    if max(comp_ratio, mem_ratio) > 1.0:
                        continue
                    if not self._cloud_budget_ok(
                        dev, demand, device_types, cloud_blocks, cloud_volume, non_cloud_feasible
                    ):
                        continue
                    queue = lyapunov[dev] if dev < len(lyapunov) else 0.0
                    lyap_weight = min(queue * lyap_to_weight_factor, lyap_weight_clip)
                    mig_cost = 0.0
                    if blk in prev_assignment and prev_assignment[blk] != dev:
                        src = prev_assignment[blk]
                        rate = bandwidth[src][dev] + 1e-6
                        mig_cost = demand.kv_cache / rate + self.mig_overhead
                    step_score = base_score * (1.0 + lyap_weight)
                    step_score += migration_penalty_scale * mig_cost
                    step_score += comm_penalty_scale * comm_delay
                    step_score += self.type_penalty.get(device_types[dev], 0.0)
                    next_assignment = dict(assignment)
                    next_assignment[blk] = dev
                    next_comp = list(comp_used)
                    next_mem = list(mem_used)
                    next_comp[dev] += demand.compute
                    next_mem[dev] += demand.memory
                    next_migs = list(migrations)
                    if blk in prev_assignment and prev_assignment[blk] != dev:
                        next_migs.append((blk, prev_assignment[blk], dev))
                    next_state: State = (
                        idx + 1,
                        tuple(next_comp),
                        tuple(next_mem),
                        next_assignment,
                        next_migs,
                        score + step_score,
                    )
                    new_beam.append(next_state)
            if not new_beam:
                return SchedulerResult(fallback_assignment, [], False, "fallback_prev")
            new_beam.sort(key=lambda s: (s[5], max(
                max(c / (cap + 1e-6) for c, cap in zip(s[1], compute)),
                max(m / (cap + 1e-6) for m, cap in zip(s[2], memory)),
            )))
            beam = new_beam[: self.beam_width]

        best_state = min(
            beam,
            key=lambda s: (s[5], max(
                max(c / (cap + 1e-6) for c, cap in zip(s[1], compute)),
                max(m / (cap + 1e-6) for m, cap in zip(s[2], memory)),
            )),
        )
        return SchedulerResult(best_state[3], best_state[4], False, "")


class GeneticScheduler(BaseScheduler):
    """Genetic algorithm scheduler for approximate placement."""

    def __init__(self, population_size: int = 50, generations: int = 20, mutation_rate: float = 0.1) -> None:
        super().__init__()
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate

    def _random_assignment(self, blocks: List[Block], num_devices: int) -> List[int]:
        return [random.randrange(num_devices) for _ in blocks]

    def _evaluate_fitness(
        self,
        individual: List[int],
        blocks: List[Block],
        demands: Dict[Block, BlockDemand],
        compute: List[float],
        memory: List[float],
        dependencies: List[Tuple[Block, Block]],
        activation_sizes: Dict[Tuple[Block, Block], float],
        bandwidth: List[List[float]],
        latency: List[List[float]] | None,
        device_types: List[str],
    ) -> float:
        assignment = {blk: dev for blk, dev in zip(blocks, individual)}
        comp_used = [0.0 for _ in compute]
        mem_used = [0.0 for _ in memory]
        penalty = 0.0
        for blk, dev in assignment.items():
            demand = demands[blk]
            comp_used[dev] += demand.compute
            mem_used[dev] += demand.memory
            comp_ratio, mem_ratio, comm_ratio = self._ratios(
                blk,
                demand,
                dev,
                compute,
                memory,
                assignment,
                {},
                dependencies,
                activation_sizes,
                bandwidth,
                latency,
                comp_used,
                mem_used,
                device_types,
            )
            penalty += max(comp_ratio, mem_ratio, comm_ratio)
        cloud_blocks, cloud_volume = self._cloud_usage(assignment, demands, device_types)
        if self.cloud_block_budget is not None and cloud_blocks > self.cloud_block_budget:
            penalty += cloud_blocks
        if self.cloud_volume_budget is not None and cloud_volume > self.cloud_volume_budget:
            penalty += cloud_volume
        max_load = max(
            max(c / max(cap, 1e-6), m / max(mem, 1e-6))
            for c, cap, m, mem in zip(comp_used, compute, mem_used, memory)
        )
        return -(max_load + penalty / max(len(blocks), 1))

    def _tournament_selection(self, population: List[List[int]], fitness: List[float]) -> List[List[int]]:
        selected: List[List[int]] = []
        for _ in range(len(population)):
            i, j = random.sample(range(len(population)), 2)
            selected.append(population[i] if fitness[i] > fitness[j] else population[j])
        return selected

    def _crossover(self, parents: List[List[int]]) -> List[List[int]]:
        offspring: List[List[int]] = []
        for i in range(0, len(parents), 2):
            p1 = parents[i]
            p2 = parents[(i + 1) % len(parents)]
            cut = random.randrange(1, len(p1)) if len(p1) > 1 else 0
            child1 = p1[:cut] + p2[cut:]
            child2 = p2[:cut] + p1[cut:]
            offspring.extend([child1, child2])
        return offspring[: len(parents)]

    def _mutate(self, individuals: List[List[int]], mutation_rate: float, num_devices: int) -> List[List[int]]:
        for individual in individuals:
            for idx in range(len(individual)):
                if random.random() < mutation_rate:
                    individual[idx] = random.randrange(num_devices)
        return individuals

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
        latency: List[List[float]] | None,
        device_types: List[str] | None = None,
    ) -> SchedulerResult:
        device_types = device_types or ["uav" for _ in compute]
        population = [
            self._random_assignment(blocks, len(compute)) for _ in range(self.population_size)
        ]
        for _ in range(self.generations):
            fitness = [
                self._evaluate_fitness(
                    indiv,
                    blocks,
                    demands,
                    compute,
                    memory,
                    dependencies,
                    activation_sizes,
                    bandwidth,
                    latency,
                    device_types,
                )
                for indiv in population
            ]
            selected = self._tournament_selection(population, fitness)
            offspring = self._crossover(selected)
            population = self._mutate(offspring, self.mutation_rate, len(compute))
        best = max(
            population,
            key=lambda indiv: self._evaluate_fitness(
                indiv,
                blocks,
                demands,
                compute,
                memory,
                dependencies,
                activation_sizes,
                bandwidth,
                latency,
                device_types,
            ),
        )
        assignment = {blk: dev for blk, dev in zip(blocks, best)}
        migrations = [
            (blk, prev_assignment[blk], dev)
            for blk, dev in assignment.items()
            if blk in prev_assignment and prev_assignment[blk] != dev
        ]
        return SchedulerResult(assignment, migrations, False, "")


class ACOScheduler(BaseScheduler):
    """Ant Colony Optimization scheduler for approximate placement."""

    def __init__(
        self,
        num_ants: int = 20,
        evaporation_rate: float = 0.1,
        alpha: float = 1.0,
        beta: float = 2.0,
        iterations: int = 15,
    ) -> None:
        super().__init__()
        self.num_ants = num_ants
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations

    def _heuristic_score(
        self,
        blk: Block,
        dev: int,
        demands: Dict[Block, BlockDemand],
        compute: List[float],
        memory: List[float],
        assignment: Dict[Block, int],
        dependencies: List[Tuple[Block, Block]],
        activation_sizes: Dict[Tuple[Block, Block], float],
        bandwidth: List[List[float]],
        latency: List[List[float]] | None,
        comp_used: List[float],
        mem_used: List[float],
        device_types: List[str],
    ) -> float:
        comp_ratio, mem_ratio, comm_ratio = self._ratios(
            blk,
            demands[blk],
            dev,
            compute,
            memory,
            assignment,
            {},
            dependencies,
            activation_sizes,
            bandwidth,
            latency,
            comp_used,
            mem_used,
            device_types,
        )
        score = max(comp_ratio, mem_ratio, comm_ratio)
        return 1.0 / max(score, 1e-6)

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
        latency: List[List[float]] | None,
        device_types: List[str] | None = None,
    ) -> SchedulerResult:
        device_types = device_types or ["uav" for _ in compute]
        pheromone = [[1.0 for _ in compute] for _ in blocks]
        best_assignment: Dict[Block, int] = {}
        best_fitness = float("-inf")
        for _ in range(self.iterations):
            solutions: List[Dict[Block, int]] = []
            fitness_scores: List[float] = []
            for _ in range(self.num_ants):
                assignment: Dict[Block, int] = {}
                comp_used = [0.0 for _ in compute]
                mem_used = [0.0 for _ in memory]
                for idx, blk in enumerate(blocks):
                    scores = []
                    for dev in range(len(compute)):
                        heuristic = self._heuristic_score(
                            blk,
                            dev,
                            demands,
                            compute,
                            memory,
                            assignment,
                            dependencies,
                            activation_sizes,
                            bandwidth,
                            latency,
                            comp_used,
                            mem_used,
                            device_types,
                        )
                        score = (pheromone[idx][dev] ** self.alpha) * (heuristic ** self.beta)
                        scores.append(score)
                    total = sum(scores)
                    if total <= 0:
                        chosen = random.randrange(len(compute))
                    else:
                        r = random.random() * total
                        acc = 0.0
                        chosen = 0
                        for dev, score in enumerate(scores):
                            acc += score
                            if acc >= r:
                                chosen = dev
                                break
                    assignment[blk] = chosen
                    comp_used[chosen] += demands[blk].compute
                    mem_used[chosen] += demands[blk].memory
                fitness = self._evaluate_fitness(
                    [assignment[b] for b in blocks],
                    blocks,
                    demands,
                    compute,
                    memory,
                    dependencies,
                    activation_sizes,
                    bandwidth,
                    latency,
                    device_types,
                )
                solutions.append(assignment)
                fitness_scores.append(fitness)
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_assignment = assignment
            for i in range(len(blocks)):
                for j in range(len(compute)):
                    pheromone[i][j] *= 1.0 - self.evaporation_rate
            for assignment, fitness in zip(solutions, fitness_scores):
                for idx, blk in enumerate(blocks):
                    dev = assignment[blk]
                    pheromone[idx][dev] += max(fitness, 0.0)
        migrations = [
            (blk, prev_assignment[blk], dev)
            for blk, dev in best_assignment.items()
            if blk in prev_assignment and prev_assignment[blk] != dev
        ]
        return SchedulerResult(best_assignment, migrations, False, "")
