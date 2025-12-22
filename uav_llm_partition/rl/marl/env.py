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
    team_reward: float
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
        lyapunov_theta: float = 0.3,
        device_types: Sequence[str] | None = None,
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
        self.lyapunov_theta = lyapunov_theta
        self.device_types = list(device_types) if device_types is not None else ["uav" for _ in compute]

        # Expected dimensions (base features + block features)
        self.type_ids = [self._encode_type(t) for t in self.device_types]
        self.local_state_dim = 18

        # Reward/penalty knobs
        self.migration_penalty_scale = 0.05
        self.comm_penalty_scale = 0.05
        self.comp_penalty_scale = 0.05
        self.bid_weight = 2.0
        self.stability_bonus = 0.01
        self.stability_margin = 0.005
        self.queue_decay = 0.02
        self.queue_cap = 5.0

        self.num_agents = len(self.compute)
        self._max_compute = max(self.compute) if self.compute else 1.0
        self._max_memory = max(self.memory) if self.memory else 1.0
        self._max_block_compute = max((d.compute for d in self.demands.values()), default=1.0)
        self._max_block_memory = max((d.memory for d in self.demands.values()), default=1.0)
        self._max_kv = max((d.kv_cache for d in self.demands.values()), default=1.0)
        self.max_layer = max((blk.layer for blk in self.blocks), default=0) + 1
        self.queue = [0.0 for _ in range(self.num_agents)]
        self.global_state_dim = 6 * self.num_agents + 6
        self.reset()

    def _encode_type(self, dev_type: str) -> float:
        mapping = {"uav": 0.0, "edge": 1.0, "cloud": 2.0}
        return mapping.get(dev_type, 0.0) / 2.0

    # ------------------------------------------------------------------
    def reset(self) -> Tuple[List[List[float]], List[float]]:
        self.assignment: Dict[Block, int] = {}
        self.comp_used = [0.0 for _ in range(self.num_agents)]
        self.mem_used = [0.0 for _ in range(self.num_agents)]
        self.block_idx = 0
        self.queue = [max(min(q, self.queue_cap), 0.0) for q in self.lyapunov]
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
        self.queue = [max(min(q, self.queue_cap), 0.0) for q in self.lyapunov]
        self._max_block_compute = max((d.compute for d in self.demands.values()), default=1.0)
        self._max_block_memory = max((d.memory for d in self.demands.values()), default=1.0)
        self._max_kv = max((d.kv_cache for d in self.demands.values()), default=1.0)

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

    def _update_queue(self, dev: int, load_ratio: float) -> None:
        drift = load_ratio - self.lyapunov_theta
        new_q = self.queue[dev] + drift - self.queue_decay
        self.queue[dev] = max(min(new_q, self.queue_cap), 0.0)
        self.lyapunov[dev] = self.queue[dev]

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
            lyap_weight = min(self.lyapunov[dev] * 0.1, 3.0)
            score = base_score * (1.0 + lyap_weight)
            score += self.migration_penalty_scale * mig_cost
            score += self.comm_penalty_scale * comm_delay
            score -= self.bid_weight * bids[dev]
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
            return EnvStep(None, None, [0.0 for _ in range(self.num_agents)], 0.0, True, None)

        block = self.blocks[self.block_idx]
        device, score_map = self._choose_device(block, bids)
        failed = device is None
        rewards = [0.0 for _ in range(self.num_agents)]
        team_reward = 0.0
        if not failed:
            self.assignment[block] = device
            demand = self.demands[block]
            prev_comp = self.comp_used[device]
            prev_mem = self.mem_used[device]
            self.comp_used[device] += demand.compute
            self.mem_used[device] += demand.memory
            comp_delay = demand.compute / (self.compute[device] + 1e-6)
            comm_delay = self._comm_delay(block, device)
            mig_cost = self._migration_cost(block, device)
            prev_load = max(prev_comp / (self.compute[device] + 1e-6), prev_mem / (self.memory[device] + 1e-6))
            load_ratio = max(
                self.comp_used[device] / (self.compute[device] + 1e-6),
                self.mem_used[device] / (self.memory[device] + 1e-6),
            )
            improvement = max(prev_load - load_ratio, 0.0)
            self._update_queue(device, load_ratio)
            queue_penalty = min(self.queue[device] / self.queue_cap, 1.0)
            stick_bonus = self.stability_bonus if block in self.prev_assignment and self.prev_assignment[block] == device else 0.0
            base_reward = 1.0 - min(score_map.get(device, 0.0), 1.0)
            device_reward = base_reward
            device_reward += 0.5 * improvement
            device_reward -= self.comm_penalty_scale * comm_delay
            device_reward -= self.comp_penalty_scale * comp_delay
            device_reward -= self.migration_penalty_scale * mig_cost
            device_reward -= queue_penalty
            device_reward += stick_bonus
            device_reward = max(min(device_reward * 2.0, 1.0), -1.0)
            rewards[device] = device_reward
            # encourage low-queue, low-load peers
            for peer in range(self.num_agents):
                if peer == device:
                    continue
                peer_load = max(
                    self.comp_used[peer] / (self.compute[peer] + 1e-6),
                    self.mem_used[peer] / (self.memory[peer] + 1e-6),
                )
                rewards[peer] = max(min(0.1 * (1.0 - peer_load) - 0.1 * min(self.queue[peer] / self.queue_cap, 1.0), 0.2), -0.2)
            loads = [
                max(c / (cap + 1e-6), m / (mem + 1e-6))
                for c, cap, m, mem in zip(self.comp_used, self.compute, self.mem_used, self.memory)
            ]
            fairness_bonus = 0.1 * (1.0 - max(loads)) if loads else 0.0
            team_reward = (sum(rewards) / max(len(rewards), 1)) + fairness_bonus
        else:
            # penalize everyone if no one could take the block
            team_reward = -0.5
            rewards = [team_reward for _ in range(self.num_agents)]
        if failed:
            team_reward = rewards[0] if rewards else -0.5

        self.block_idx += 1
        done = self.block_idx >= len(self.blocks)
        next_local = None if done else self._get_local_states()
        next_global = None if done else self._get_global_state()
        return EnvStep(next_local, next_global, rewards, team_reward, done, (block, device) if not failed else None)

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
                min(self.queue[dev] / self.queue_cap, 1.0),
                self.weights[dev],
                self.type_ids[dev],
            ]
            if current_block:
                demand = self.demands[current_block]
                comm_cost = self._comm_delay(current_block, dev)
                mig_cost = self._migration_cost(current_block, dev)
                feasible, _ = self._feasible(current_block, dev)
                state.extend(
                    [
                        demand.compute / (self._max_block_compute + 1e-6),
                        demand.memory / (self._max_block_memory + 1e-6),
                        demand.kv_cache / (self._max_kv + 1e-6),
                        current_block.layer / max(self.max_layer, 1),
                        1 if current_block.kind == "head" else 0,
                        comm_cost,
                        mig_cost,
                        1.0 if self.prev_assignment.get(current_block) == dev else 0.0,
                        1.0 if feasible else 0.0,
                    ]
                )
            else:
                state.extend([0.0 for _ in range(9)])
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
        queue_norm = [min(q / self.queue_cap, 1.0) for q in self.queue]
        state.extend(queue_norm)
        type_counts = [self.device_types.count(t) / max(self.num_agents, 1) for t in ("uav", "edge", "cloud")]
        state.extend(type_counts)
        state.append(sum(loads) / len(loads) if loads else 0.0)
        state.append(max(loads) if loads else 0.0)
        mean_q = sum(self.queue) / max(len(self.queue), 1)
        state.append(min(mean_q / self.queue_cap, 1.0))
        expected = self.global_state_dim
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
