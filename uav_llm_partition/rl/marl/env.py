"""Lightweight multi-agent environment for block-to-device assignment.

This environment mirrors the existing single-agent RL env but allows each UAV
agent to emit a bid/acceptance score for the current block. The environment
then selects the best feasible UAV under DTIS-style ratios.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from uav_llm_partition.model_partition.blocks import Block
from uav_llm_partition.model_partition.demand_model import BlockDemand


@dataclass
class EnvStep:
    next_local_states: Optional[List[List[float]]]
    next_global_state: Optional[List[float]]
    rewards: List[float]
    done: bool
    assignment: Optional[Tuple[Block, int]] = None


class MultiAgentResourceAllocationEnv:
    """Stateless-ish wrapper: one episode corresponds to placing all blocks."""

    def __init__(
        self,
        blocks: Sequence[Block],
        demands: Dict[Block, BlockDemand],
        compute: Sequence[float],
        memory: Sequence[float],
        bandwidth: Sequence[Sequence[float]],
        lyapunov: Sequence[float],
        weights: Sequence[float],
        dependencies: Sequence[Tuple[Block, Block]],
        activation_sizes: Dict[Tuple[Block, Block], float],
        load_guard: float = 1.0,
        prev_assignment: Optional[Dict[Block, int]] = None,
        migration_overhead: float = 0.01,
    ) -> None:
        self.blocks = list(blocks)
        self.demands = demands
        self.compute = list(compute)
        self.memory = list(memory)
        self.bandwidth = bandwidth
        self.lyapunov = list(lyapunov)
        self.weights = list(weights)
        self.dependencies = list(dependencies)
        self.activation_sizes = activation_sizes
        self.load_guard = load_guard
        self.prev_assignment = prev_assignment or {}
        self.migration_overhead = migration_overhead

        # Expected dimensions (base features + block features)
        self.local_state_dim = 16

        # Reward/penalty knobs
        self.migration_penalty_scale = 0.1
        self.comm_penalty_scale = 0.1
        self.comp_penalty_scale = 0.05
        self.stability_bonus = 0.2
        self.stability_margin = 0.05

        self.num_agents = len(self.compute)
        self._max_compute = max(self.compute) if self.compute else 1.0
        self._max_memory = max(self.memory) if self.memory else 1.0
        self._max_block_compute = max((d.compute for d in self.demands.values()), default=1.0)
        self._max_block_memory = max((d.memory for d in self.demands.values()), default=1.0)
        self.global_state_dim = 5 * self.num_agents + 2
        self.reset()

    # ------------------------------------------------------------------
    def reset(self) -> Tuple[List[List[float]], List[float]]:
        self.assignment: Dict[Block, int] = {}
        self.comp_used = [0.0 for _ in range(self.num_agents)]
        self.mem_used = [0.0 for _ in range(self.num_agents)]
        self.block_idx = 0
        return self._get_local_states(), self._get_global_state()

    def retain_feasible_prev(self) -> Tuple[Dict[Block, int], List[float], List[float], List[Block]]:
        """Keep previous assignments that remain feasible to avoid unnecessary migration.

        Returns (kept_assignment, comp_used, mem_used, pending_blocks).
        """

        kept: Dict[Block, int] = {}
        comp_used = [0.0 for _ in range(self.num_agents)]
        mem_used = [0.0 for _ in range(self.num_agents)]
        pending: List[Block] = []
        for block in self.blocks:
            prev_dev = self.prev_assignment.get(block)
            if prev_dev is None:
                pending.append(block)
                continue
            demand = self.demands[block]
            comp_future = comp_used[prev_dev] + demand.compute
            mem_future = mem_used[prev_dev] + demand.memory
            compute_ratio = comp_future / (self.compute[prev_dev] + 1e-6)
            memory_ratio = mem_future / (self.memory[prev_dev] + 1e-6)
            comm_ratio = 0.0
            for up, down in self.dependencies:
                if (up == block and down in kept and kept[down] != prev_dev) or (
                    down == block and up in kept and kept[up] != prev_dev
                ):
                    size = self.activation_sizes.get((up, down), 0.0)
                    src = kept.get(up, prev_dev)
                    dst = kept.get(down, prev_dev)
                    if src != dst:
                        comm_ratio += size / (self.bandwidth[src][dst] + 1e-6)
            score = max(compute_ratio, memory_ratio, comm_ratio)
            if score <= self.load_guard:
                kept[block] = prev_dev
                comp_used[prev_dev] = comp_future
                mem_used[prev_dev] = mem_future
            else:
                pending.append(block)
        return kept, comp_used, mem_used, pending

    def apply_partial_state(
        self,
        kept_assignment: Dict[Block, int],
        comp_used: List[float],
        mem_used: List[float],
        pending_blocks: List[Block],
    ) -> None:
        """Seed the environment with kept assignments and restrict to pending blocks."""

        self.assignment = dict(kept_assignment)
        self.comp_used = list(comp_used)
        self.mem_used = list(mem_used)
        self.blocks = list(pending_blocks)
        self.block_idx = 0
        self._max_block_compute = max((d.compute for d in self.demands.values()), default=1.0)
        self._max_block_memory = max((d.memory for d in self.demands.values()), default=1.0)

    def current_states(self) -> Tuple[List[List[float]], List[float]]:
        """Expose current local/global states for learners."""

        return self._get_local_states(), self._get_global_state()

    # ------------------------------------------------------------------
    def _feasible(self, block: Block, dev: int) -> Tuple[bool, float]:
        demand = self.demands[block]
        comp_future = self.comp_used[dev] + demand.compute
        mem_future = self.mem_used[dev] + demand.memory
        compute_ratio = comp_future / (self.compute[dev] + 1e-6)
        memory_ratio = mem_future / (self.memory[dev] + 1e-6)

        comm_ratio = self._comm_delay(block, dev)
        score = max(compute_ratio, memory_ratio, comm_ratio)
        feasible = score <= self.load_guard
        return feasible, score

    def _comm_delay(self, block: Block, dev: int) -> float:
        comm_ratio = 0.0
        for up, down in self.dependencies:
            if up == block and down in self.assignment:
                dst = self.assignment[down]
                if dst != dev:
                    size = self.activation_sizes.get((up, down), 0.0)
                    comm_ratio += size / (self.bandwidth[dev][dst] + 1e-6)
            elif down == block and up in self.assignment:
                src = self.assignment[up]
                if src != dev:
                    size = self.activation_sizes.get((up, down), 0.0)
                    comm_ratio += size / (self.bandwidth[src][dev] + 1e-6)
        return comm_ratio

    def _migration_cost(self, block: Block, dev: int) -> float:
        prev = self.prev_assignment.get(block)
        if prev is None or prev == dev:
            return 0.0
        kv = self.demands[block].kv_cache
        rate = self.bandwidth[prev][dev] + 1e-6
        return kv / rate + self.migration_overhead

    def _choose_device(self, block: Block, bids: Sequence[float]) -> Tuple[Optional[int], Dict[int, float]]:
        scores: Dict[int, float] = {}
        prev_dev = self.prev_assignment.get(block)
        prev_score = None
        for dev in range(self.num_agents):
            feasible, base_score = self._feasible(block, dev)
            if not feasible:
                continue
            mig_cost = self._migration_cost(block, dev)
            comm_delay = self._comm_delay(block, dev)
            # Lyapunov and migration act as penalties; bids reward willingness
            score = base_score * (1.0 + self.lyapunov[dev])
            score += self.migration_penalty_scale * mig_cost
            score += self.comm_penalty_scale * comm_delay
            score -= bids[dev]
            scores[dev] = score
            if prev_dev is not None and dev == prev_dev:
                prev_score = score - self.stability_margin  # make previous slightly more attractive
        if not scores:
            return None, scores
        best_dev = min(scores, key=scores.get)
        # Sticky preference: keep previous assignment if comparable
        if prev_dev is not None and prev_score is not None:
            if prev_dev in scores and scores.get(best_dev, 1e9) >= prev_score:
                best_dev = prev_dev
        return best_dev, scores

    def step(self, bids: Sequence[float]) -> EnvStep:
        if self.block_idx >= len(self.blocks):
            return EnvStep(None, None, [0.0 for _ in range(self.num_agents)], True, None)

        block = self.blocks[self.block_idx]
        device, score_map = self._choose_device(block, bids)
        failed = device is None
        rewards = [0.0 for _ in range(self.num_agents)]
        if not failed:
            self.assignment[block] = device
            demand = self.demands[block]
            self.comp_used[device] += demand.compute
            self.mem_used[device] += demand.memory
            comp_delay = demand.compute / (self.compute[device] + 1e-6)
            comm_delay = self._comm_delay(block, device)
            mig_cost = self._migration_cost(block, device)
            stick_bonus = 0.0
            if block in self.prev_assignment and self.prev_assignment[block] == device:
                stick_bonus = self.stability_bonus
            rewards[device] = (
                1.0 - min(score_map.get(device, 0.0), 1.0)
                - self.comm_penalty_scale * comm_delay
                - self.comp_penalty_scale * comp_delay
                - self.migration_penalty_scale * mig_cost
                + stick_bonus
            )
        else:
            # penalize everyone if no one could take the block
            rewards = [-0.5 for _ in range(self.num_agents)]

        self.block_idx += 1
        done = self.block_idx >= len(self.blocks)
        next_local = None if done else self._get_local_states()
        next_global = None if done else self._get_global_state()
        return EnvStep(next_local, next_global, rewards, done, (block, device) if not failed else None)

    # ------------------------------------------------------------------
    def _get_local_states(self) -> List[List[float]]:
        states: List[List[float]] = []
        current_block = self.blocks[self.block_idx] if self.block_idx < len(self.blocks) else None
        for dev in range(self.num_agents):
            comp_ratio = self.comp_used[dev] / (self.compute[dev] + 1e-6)
            mem_ratio = self.mem_used[dev] / (self.memory[dev] + 1e-6)
            comm_cost = 0.0
            state = [
                self.compute[dev] / (self._max_compute + 1e-6),
                self.memory[dev] / (self._max_memory + 1e-6),
                self.comp_used[dev] / (self._max_compute + 1e-6),
                self.mem_used[dev] / (self._max_memory + 1e-6),
                comp_ratio,
                mem_ratio,
                self.lyapunov[dev],
                self.weights[dev],
            ]
            if current_block:
                demand = self.demands[current_block]
                comm_cost = self._comm_delay(current_block, dev)
                mig_cost = self._migration_cost(current_block, dev)
                state.extend(
                    [
                        demand.compute / (self._max_block_compute + 1e-6),
                        demand.memory / (self._max_block_memory + 1e-6),
                        demand.kv_cache / (self._max_block_memory + 1e-6),
                        current_block.layer,
                        1 if current_block.kind == "head" else 0,
                        comm_cost,
                        mig_cost,
                        1.0 if self.prev_assignment.get(current_block) == dev else 0.0,
                    ]
                )
            else:
                state.extend([0.0 for _ in range(8)])
            assert len(state) == self.local_state_dim, f"local state dim mismatch {len(state)} != {self.local_state_dim}"
            states.append(state)
        return states

    def _get_global_state(self) -> List[float]:
        loads = [max(c / (cap + 1e-6), m / (mem + 1e-6)) for c, cap, m, mem in zip(
            self.comp_used, self.compute, self.mem_used, self.memory
        )]
        state: List[float] = []
        state.extend([c / (self._max_compute + 1e-6) for c in self.compute])
        state.extend([m / (self._max_memory + 1e-6) for m in self.memory])
        state.extend([c / (self._max_compute + 1e-6) for c in self.comp_used])
        state.extend([m / (self._max_memory + 1e-6) for m in self.mem_used])
        state.extend(loads)
        state.append(sum(loads) / len(loads) if loads else 0.0)
        state.append(max(loads) if loads else 0.0)
        expected = 5 * self.num_agents + 2
        assert len(state) == expected, f"global state dim mismatch {len(state)} != {expected}"
        return state

    def action_mask(self) -> List[bool]:
        if self.block_idx >= len(self.blocks):
            return [False for _ in range(self.num_agents)]
        block = self.blocks[self.block_idx]
        mask: List[bool] = []
        for dev in range(self.num_agents):
            feasible, _ = self._feasible(block, dev)
            mask.append(feasible)
        return mask
