"""Reinforcement-learning environment wrapper for UAVT scheduling.

This environment maps the simulator's per-interval scheduling problem into a
sequential decision process so that an RL agent can place blocks one by one.
The design follows the specification outlined in the user request: a rich
state vector with resource, load, queue, communication, and current-block
features plus an action mask that enforces feasibility.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from uav_llm_partition.model_partition.blocks import Block
from uav_llm_partition.model_partition.demand_model import BlockDemand
from uav_llm_partition.sim.metrics import jain_fairness


@dataclass
class RLStep:
    state: List[float]
    action: int
    reward: float
    done: bool


@dataclass
class RLEpisodeResult:
    assignment: Dict[Block, int]
    migrations: List[Tuple[Block, int, int]]
    failed: bool
    reason: str
    reward: float


class RLResourceAllocationEnv:
    """A lightweight, per-interval RL environment for head-level scheduling."""

    def __init__(
        self,
        blocks: List[Block],
        demands: Dict[Block, BlockDemand],
        compute: List[float],
        memory: List[float],
        weights: List[float],
        lyapunov: List[float],
        dependencies: List[Tuple[Block, Block]],
        activation_sizes: Dict[Tuple[Block, Block], float],
        bandwidth: List[List[float]],
        prev_assignment: Dict[Block, int],
        load_guard: float = 1.0,
        queue_block_threshold: Optional[float] = None,
    ) -> None:
        self.blocks = list(blocks)
        self.demands = demands
        self.compute = compute
        self.memory = memory
        self.weights = weights
        self.lyapunov = lyapunov
        self.dependencies = dependencies
        self.activation_sizes = activation_sizes
        self.bandwidth = bandwidth
        self.prev_assignment = prev_assignment
        self.load_guard = load_guard
        self.queue_block_threshold = queue_block_threshold

        self.comp_used = [0.0 for _ in compute]
        self.mem_used = [0.0 for _ in memory]
        self.assignment: Dict[Block, int] = {}
        self.steps: List[RLStep] = []
        self.current_idx = 0
        self.failed = False
        self.failure_reason = ""
        self.reward = 0.0

        self._max_compute = max(max(compute), 1e-6)
        self._max_memory = max(max(memory), 1e-6)
        self._max_block_compute = max((d.compute for d in demands.values()), default=1.0)
        self._max_block_memory = max((d.memory for d in demands.values()), default=1.0)

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------
    def _action_mask(self, block: Block) -> List[bool]:
        demand = self.demands[block]
        mask: List[bool] = []
        for dev in range(len(self.compute)):
            comp_ratio, mem_ratio, comm_ratio = self._ratios(block, demand, dev)
            base_score = max(comp_ratio, mem_ratio, comm_ratio)
            if self.queue_block_threshold is not None and self.lyapunov[dev] > self.queue_block_threshold:
                mask.append(False)
                continue
            mask.append(base_score <= self.load_guard)
        return mask

    def _ratios(self, block: Block, demand: BlockDemand, dev: int) -> Tuple[float, float, float]:
        comp_ratio = (self.comp_used[dev] + demand.compute) / max(self.compute[dev], 1e-6)
        mem_ratio = (self.mem_used[dev] + demand.memory) / max(self.memory[dev], 1e-6)
        comm_delay = 0.0
        for up, down in self.dependencies:
            if block not in (up, down):
                continue
            neighbor = down if block == up else up
            neighbor_dev = self.assignment.get(neighbor, self.prev_assignment.get(neighbor))
            if neighbor_dev is None or neighbor_dev == dev:
                continue
            size = self.activation_sizes.get((up, down), self.activation_sizes.get((down, up), 0.0))
            bw = self.bandwidth[dev][neighbor_dev] + 1e-6
            comm_delay += size / bw
        comm_ratio = comm_delay / 0.05  # align with BaseScheduler.comm_budget default
        return comp_ratio, mem_ratio, comm_ratio

    def _global_stats(self) -> Tuple[float, float, float]:
        loads = [
            max(
                (cu + 1e-6) / max(c, 1e-6),
                (mu + 1e-6) / max(m, 1e-6),
            )
            for cu, mu, c, m in zip(self.comp_used, self.mem_used, self.compute, self.memory)
        ]
        if not loads:
            return 0.0, 0.0, 0.0
        avg_load = sum(loads) / len(loads)
        max_load = max(loads)
        imbalance = max_load - avg_load
        return avg_load, max_load, imbalance

    def _comm_costs(self, block: Block) -> List[float]:
        costs: List[float] = []
        demand = self.demands[block]
        for dev in range(len(self.compute)):
            _, _, comm_ratio = self._ratios(block, demand, dev)
            costs.append(comm_ratio)
        return costs

    def _block_features(self, block: Optional[Block]) -> List[float]:
        if block is None:
            return [0.0, 0.0, 0.0, 0.0]
        demand = self.demands[block]
        return [
            demand.compute / (self._max_block_compute + 1e-6),
            demand.memory / (self._max_block_memory + 1e-6),
            demand.kv_cache / (self._max_block_memory + 1e-6),
            block.layer / max(1, max(b.layer for b in self.blocks)),
        ]

    def _state(self) -> List[float]:
        block = self.blocks[self.current_idx] if self.current_idx < len(self.blocks) else None
        comp_ratio = [(cu) / max(c, 1e-6) for cu, c in zip(self.comp_used, self.compute)]
        mem_ratio = [(mu) / max(m, 1e-6) for mu, m in zip(self.mem_used, self.memory)]
        avg_load, max_load, imbalance = self._global_stats()
        comm_costs = self._comm_costs(block) if block else [0.0 for _ in self.compute]

        state_vec: List[float] = []
        state_vec.extend([c / self._max_compute for c in self.compute])
        state_vec.extend([m / self._max_memory for m in self.memory])
        state_vec.extend(comp_ratio)
        state_vec.extend(mem_ratio)
        state_vec.extend(self.weights)
        state_vec.extend(self.lyapunov)
        state_vec.extend(self._block_features(block))
        state_vec.extend(comm_costs)
        state_vec.extend([avg_load, max_load, imbalance])
        return state_vec

    # ------------------------------------------------------------------
    # Environment API
    # ------------------------------------------------------------------
    def reset(self) -> List[float]:
        self.comp_used = [0.0 for _ in self.compute]
        self.mem_used = [0.0 for _ in self.memory]
        self.assignment = {}
        self.steps = []
        self.current_idx = 0
        self.failed = False
        self.failure_reason = ""
        self.reward = 0.0
        return self._state()

    def step(self, action: int) -> Tuple[List[float], float, bool, Dict[str, object]]:
        if self.current_idx >= len(self.blocks):
            return self._state(), 0.0, True, {}
        block = self.blocks[self.current_idx]
        mask = self._action_mask(block)
        if not mask[action]:
            self.failed = True
            self.failure_reason = "invalid_action"
            self.reward = -5.0
            self.steps.append(RLStep(self._state(), action, self.reward, True))
            return self._state(), self.reward, True, {}

        demand = self.demands[block]
        self.assignment[block] = action
        self.comp_used[action] += demand.compute
        self.mem_used[action] += demand.memory
        self.current_idx += 1

        done = self.current_idx >= len(self.blocks)
        reward = 0.0
        if done:
            reward = self._final_reward()
            self.reward = reward
        self.steps.append(RLStep(self._state(), action, reward, done))
        return self._state(), reward, done, {}

    def _final_reward(self) -> float:
        avg_load, max_load, imbalance = self._global_stats()
        fairness = jain_fairness([
            max((cu) / max(c, 1e-6), (mu) / max(m, 1e-6))
            for cu, mu, c, m in zip(self.comp_used, self.mem_used, self.compute, self.memory)
        ])
        delay = 0.0
        for blk, dev in self.assignment.items():
            demand = self.demands[blk]
            delay += demand.compute / (self.compute[dev] + 1e-6)
        for up, down in self.dependencies:
            dev_u = self.assignment.get(up, self.prev_assignment.get(up))
            dev_d = self.assignment.get(down, self.prev_assignment.get(down))
            if dev_u is None or dev_d is None or dev_u == dev_d:
                continue
            size = self.activation_sizes.get((up, down), self.activation_sizes.get((down, up), 0.0))
            delay += size / (self.bandwidth[dev_u][dev_d] + 1e-6)

        migration_count = 0
        for blk, new_dev in self.assignment.items():
            old_dev = self.prev_assignment.get(blk)
            if old_dev is not None and old_dev != new_dev:
                migration_count += 1

        delay_penalty = -0.5 * delay
        balance_reward = 2.0 * fairness
        failure_penalty = -10.0 if self.failed else 0.0
        migration_penalty = -0.5 * migration_count
        imbalance_penalty = -imbalance
        utilization_reward = 0.5 * min(avg_load, 0.8)
        return delay_penalty + balance_reward + failure_penalty + migration_penalty + imbalance_penalty + utilization_reward

    def action_mask(self) -> List[bool]:
        if self.current_idx >= len(self.blocks):
            return [False for _ in self.compute]
        return self._action_mask(self.blocks[self.current_idx])

    def episode_result(self) -> RLEpisodeResult:
        migrations: List[Tuple[Block, int, int]] = []
        for blk, new_dev in self.assignment.items():
            old_dev = self.prev_assignment.get(blk)
            if old_dev is not None and old_dev != new_dev:
                migrations.append((blk, old_dev, new_dev))
        return RLEpisodeResult(
            assignment=self.assignment,
            migrations=migrations,
            failed=self.failed,
            reason=self.failure_reason,
            reward=self.reward,
        )
