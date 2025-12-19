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

        self.num_agents = len(self.compute)
        self.reset()

    # ------------------------------------------------------------------
    def reset(self) -> Tuple[List[List[float]], List[float]]:
        self.assignment: Dict[Block, int] = {}
        self.comp_used = [0.0 for _ in range(self.num_agents)]
        self.mem_used = [0.0 for _ in range(self.num_agents)]
        self.block_idx = 0
        return self._get_local_states(), self._get_global_state()

    # ------------------------------------------------------------------
    def _feasible(self, block: Block, dev: int) -> Tuple[bool, float]:
        demand = self.demands[block]
        comp_future = self.comp_used[dev] + demand.compute
        mem_future = self.mem_used[dev] + demand.memory
        compute_ratio = comp_future / (self.compute[dev] + 1e-6)
        memory_ratio = mem_future / (self.memory[dev] + 1e-6)

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
        score = max(compute_ratio, memory_ratio, comm_ratio)
        feasible = score <= self.load_guard
        return feasible, score

    def _choose_device(self, block: Block, bids: Sequence[float]) -> Tuple[Optional[int], Dict[int, float]]:
        scores: Dict[int, float] = {}
        for dev in range(self.num_agents):
            feasible, score = self._feasible(block, dev)
            if not feasible:
                continue
            # Lyapunov penalty discourages high backlog
            score = score * (1.0 + self.lyapunov[dev]) - bids[dev]
            scores[dev] = score
        if not scores:
            return None, scores
        return min(scores, key=scores.get), scores

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
            rewards[device] += 1.0 - min(score_map.get(device, 0.0), 1.0)
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
            state = [
                self.compute[dev],
                self.memory[dev],
                self.comp_used[dev],
                self.mem_used[dev],
                comp_ratio,
                mem_ratio,
                self.lyapunov[dev],
                self.weights[dev],
            ]
            if current_block:
                demand = self.demands[current_block]
                state.extend(
                    [
                        demand.compute,
                        demand.memory,
                        demand.kv_cache,
                        current_block.layer,
                        1 if current_block.kind == "head" else 0,
                    ]
                )
            states.append(state)
        return states

    def _get_global_state(self) -> List[float]:
        loads = [max(c / (cap + 1e-6), m / (mem + 1e-6)) for c, cap, m, mem in zip(
            self.comp_used, self.compute, self.mem_used, self.memory
        )]
        state: List[float] = []
        state.extend(self.compute)
        state.extend(self.memory)
        state.extend(self.comp_used)
        state.extend(self.mem_used)
        state.extend(loads)
        state.append(sum(loads) / len(loads) if loads else 0.0)
        state.append(max(loads) if loads else 0.0)
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
