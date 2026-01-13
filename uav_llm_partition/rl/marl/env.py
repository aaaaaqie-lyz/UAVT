"""Lightweight multi-agent environment for block-to-device assignment.

This environment mirrors the existing single-agent RL env but allows each UAV
agent to emit a bid/acceptance score for the current block. The environment
then selects the best feasible UAV under DTIS-style ratios.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from uav_llm_partition.controller.constants import (
    comm_penalty_scale,
    lyap_to_weight_factor,
    lyap_weight_clip,
    migration_penalty_scale,
    stability_bonus,
    stability_margin,
    type_penalty,
)
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
        latency: Sequence[Sequence[float]] | None,
        lyapunov: Sequence[float],
        weights: Sequence[float],
        dependencies: Sequence[Tuple[Block, Block]],
        activation_sizes: Dict[Tuple[Block, Block], float],
        load_guard: float = 1.0,
        prev_assignment: Optional[Dict[Block, int]] = None,
        migration_overhead: float = 0.01,
        lyapunov_theta: float = 0.3,
        device_types: Sequence[str] | None = None,
        snr: Sequence[float] | None = None,
        rssi: Sequence[float] | None = None,
        rho_w: float = 0.0,
        rho_q: float = 0.0,
        dpp_v: float = 1.0,
        queue_block_threshold: float | None = 0.9,
        cloud_block_budget: int | None = 1,
        cloud_volume_budget: float | None = 0.5,
        cloud_fallback_only: bool = True,
        cloud_bias: float = 1.0,
        migration_hold_steps: int = 3,
        cooldowns: Dict[Block, int] | None = None,
    ) -> None:
        self.blocks = list(blocks)
        self.demands = demands
        self.compute = list(compute)
        self.memory = list(memory)
        self.bandwidth = bandwidth
        self.latency = latency or [[0.0 for _ in range(len(self.compute))] for _ in range(len(self.compute))]
        self.lyapunov = list(lyapunov)
        self.weights = list(weights)
        self.dependencies = list(dependencies)
        self.activation_sizes = activation_sizes
        self.load_guard = load_guard
        self.prev_assignment = prev_assignment or {}
        self.migration_overhead = migration_overhead
        self.lyapunov_theta = lyapunov_theta
        self.device_types = list(device_types) if device_types is not None else ["uav" for _ in compute]
        self.snr = list(snr) if snr is not None else [0.0 for _ in compute]
        self.rssi = list(rssi) if rssi is not None else [0.0 for _ in compute]
        self.rho_w = rho_w
        self.rho_q = rho_q
        self.dpp_v = dpp_v
        self.queue_block_threshold = queue_block_threshold
        self.cloud_block_budget = cloud_block_budget
        self.cloud_volume_budget = cloud_volume_budget
        self.cloud_fallback_only = cloud_fallback_only
        self.cloud_bias = cloud_bias
        self.migration_hold_steps = migration_hold_steps
        self.cooldowns = dict(cooldowns) if cooldowns is not None else {}
        self.max_retries = 2
        self.num_agents = len(self.compute)

        # Expected dimensions (base features + block features)
        self.type_ids = [self._encode_type(t) for t in self.device_types]
        self._max_bw = max((max(row) for row in self.bandwidth), default=1.0)
        self._avg_bw_per_dev = [
            sum(row) / max(len([v for v in row if v > 0.0]), 1) if row else 0.0 for row in self.bandwidth
        ]
        self._reach_ratio = [
            sum(1 for v in row if v > 0.0) / max(self.num_agents - 1, 1) for row in self.bandwidth
        ]
        self._max_snr = max(self.snr) if self.snr else 1.0
        self._max_rssi = max(self.rssi) if self.rssi else 1.0
        self.local_state_dim = 26

        # Reward/penalty knobs
        self.migration_penalty_scale = migration_penalty_scale
        self.comm_penalty_scale = comm_penalty_scale
        self.comp_penalty_scale = 0.1
        self.bid_weight = 2.0
        self.stability_bonus = stability_bonus
        self.stability_margin = stability_margin
        self.balance_weight = 0.4
        self.queue_threshold = 0.6
        self.delay_norm = 1.0
        self.comm_norm = 1.0
        self.comp_norm = 1.0
        self.mig_norm = 1.0
        self.max_load_weight = 0.4
        self.max_load_hinge = 0.8
        self.comm_reward_weight = 0.3
        self.cut_penalty_weight = 0.2
        self.cut_edge_limit = 2
        self.cut_bytes_limit = 0.05
        self.queue_weight = 0.3
        self.queue_drift_weight = 0.2
        self.max_queue_weight = 0.4
        self.comm_queue_weight = 0.2
        self.failure_penalty = 10.0
        self.team_mix = 0.3
        self.team_mix_base = 0.3
        self.max_load_weight_base = 0.4
        self.imbalance_threshold = 0.7
        self.queue_decay = 0.02
        self.queue_cap = 5.0
        # Penalise non-UAV targets more aggressively so bids must offset the
        # stronger edge/cloud capacity and keep a UAV-first bias.
        self.type_penalty = type_penalty

        self._max_compute = max(self.compute) if self.compute else 1.0
        self._max_memory = max(self.memory) if self.memory else 1.0
        self._avg_compute = sum(self.compute) / max(self.num_agents, 1)
        self._avg_memory = sum(self.memory) / max(self.num_agents, 1)
        self._max_block_compute = max((d.compute for d in self.demands.values()), default=1.0)
        self._max_block_memory = max((d.memory for d in self.demands.values()), default=1.0)
        self._max_kv = max((d.kv_cache for d in self.demands.values()), default=1.0)
        self.max_layer = max((blk.layer for blk in self.blocks), default=0) + 1
        self.queue = [0.0 for _ in range(self.num_agents)]
        self.comm_queue = [0.0 for _ in range(self.num_agents)]
        self.failed_blocks: List[Block] = []
        self.retry_counts: Dict[Block, int] = {}
        self.global_state_dim = 6 * self.num_agents + 14
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
        self.comm_queue = [max(min(q, self.queue_cap), 0.0) for q in self.lyapunov]
        self._avg_compute = sum(self.compute) / max(self.num_agents, 1)
        self._avg_memory = sum(self.memory) / max(self.num_agents, 1)
        for blk in list(self.cooldowns.keys()):
            self.cooldowns[blk] -= 1
            if self.cooldowns[blk] <= 0:
                del self.cooldowns[blk]
        self.failed_blocks = []
        self.retry_counts = {}
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
                        comm_ratio += size / (self.bandwidth[src][dst] + 1e-6) + self.latency[src][dst]
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
        self._max_bw = max((max(row) for row in self.bandwidth), default=1.0)
        self._avg_bw_per_dev = [
            sum(row) / max(len([v for v in row if v > 0.0]), 1) if row else 0.0 for row in self.bandwidth
        ]
        self._reach_ratio = [
            sum(1 for v in row if v > 0.0) / max(self.num_agents - 1, 1) for row in self.bandwidth
        ]
        self._max_snr = max(self.snr) if self.snr else 1.0
        self._max_rssi = max(self.rssi) if self.rssi else 1.0
        self._max_block_compute = max((d.compute for d in self.demands.values()), default=1.0)
        self._max_block_memory = max((d.memory for d in self.demands.values()), default=1.0)
        self._max_kv = max((d.kv_cache for d in self.demands.values()), default=1.0)
        self._avg_compute = sum(self.compute) / max(self.num_agents, 1)
        self._avg_memory = sum(self.memory) / max(self.num_agents, 1)

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
                    comm_ratio += size / (self.bandwidth[dev][dst] + 1e-6) + self.latency[dev][dst]
            elif down == block and up in self.assignment:
                src = self.assignment[up]
                if src != dev:
                    size = self.activation_sizes.get((up, down), 0.0)
                    comm_ratio += size / (self.bandwidth[src][dev] + 1e-6) + self.latency[src][dev]
        return comm_ratio

    def _link_reachable(self, block: Block, dev: int) -> bool:
        for up, down in self.dependencies:
            if block not in (up, down):
                continue
            neighbor = down if block == up else up
            neighbor_dev = self.assignment.get(neighbor, self.prev_assignment.get(neighbor))
            if neighbor_dev is None or neighbor_dev == dev:
                continue
            if self.bandwidth[dev][neighbor_dev] <= 0.0:
                return False
        return True

    def _queue_allows(self, dev: int, strict: bool = True) -> bool:
        if self.queue_block_threshold is None:
            return True
        threshold = self.queue_block_threshold if strict else self.queue_block_threshold * 1.5
        return self.queue[dev] <= threshold

    def _cut_metrics(self, block: Block, dev: int) -> Tuple[int, float]:
        cut_cnt = 0
        cut_bytes = 0.0
        for up, down in self.dependencies:
            if block not in (up, down):
                continue
            neighbor = down if block == up else up
            neighbor_dev = self.assignment.get(neighbor, self.prev_assignment.get(neighbor))
            if neighbor_dev is None or neighbor_dev == dev:
                continue
            size = self.activation_sizes.get((up, down), self.activation_sizes.get((down, up), 0.0))
            cut_cnt += 1
            cut_bytes += size
        return cut_cnt, cut_bytes

    def _migration_cost(self, block: Block, dev: int) -> float:
        prev = self.prev_assignment.get(block)
        if prev is None or prev == dev:
            return 0.0
        kv = self.demands[block].kv_cache
        rate = self.bandwidth[prev][dev] + 1e-6
        return kv / rate + self.migration_overhead

    def _update_queue(self, dev: int, arrival: float, service: float) -> None:
        new_q = max(self.queue[dev] - service, 0.0) + arrival
        new_q = new_q - self.queue_decay
        self.queue[dev] = max(min(new_q, self.queue_cap), 0.0)
        self.lyapunov[dev] = self.queue[dev]

    def _update_comm_queue(self, dev: int, arrival: float, service: float) -> None:
        new_q = max(self.comm_queue[dev] - service, 0.0) + arrival
        new_q = new_q - self.queue_decay
        self.comm_queue[dev] = max(min(new_q, self.queue_cap), 0.0)

    def _cooldown_allows(self, block: Block, dev: int, feasible_prev: bool, prev_dev: int | None) -> bool:
        cooldown = self.cooldowns.get(block, 0)
        if cooldown <= 0:
            return True
        if prev_dev is None or not feasible_prev:
            return True
        return dev == prev_dev

    def _norm(self, value: float, scale: float) -> float:
        if scale <= 0:
            return 0.0
        return min(max(value / scale, 0.0), 1.0)

    def _choose_device(self, block: Block, bids: Sequence[float]) -> Tuple[Optional[int], Dict[int, float]]:
        return self._choose_device_with_queue(block, bids, strict_queue=True)

    def _choose_device_with_queue(
        self,
        block: Block,
        bids: Sequence[float],
        strict_queue: bool,
    ) -> Tuple[Optional[int], Dict[int, float]]:
        scores: Dict[int, float] = {}
        prev_dev = self.prev_assignment.get(block)
        prev_score = None
        cloud_blocks = sum(1 for _, dev in self.assignment.items() if self.device_types[dev] == "cloud")
        cloud_volume = sum(
            self.demands[blk].memory
            for blk, dev in self.assignment.items()
            if self.device_types[dev] == "cloud"
        )
        non_cloud_feasible = False
        prev_feasible, _ = self._feasible(block, prev_dev) if prev_dev is not None else (False, 0.0)
        for dev in range(self.num_agents):
            feasible, base_score = self._feasible(block, dev)
            if not feasible:
                continue
            if not self._link_reachable(block, dev):
                continue
            if not self._cooldown_allows(block, dev, prev_feasible, prev_dev):
                continue
            if not self._queue_allows(dev, strict=strict_queue):
                continue
            if self.device_types[dev] != "cloud":
                non_cloud_feasible = True
            mig_cost = self._migration_cost(block, dev)
            comm_delay = self._comm_delay(block, dev)
            # Lyapunov and migration act as penalties; bids reward willingness
            lyap_weight = min(self.lyapunov[dev] * lyap_to_weight_factor, lyap_weight_clip)
            score = base_score * (1.0 + lyap_weight)
            score += self.migration_penalty_scale * mig_cost
            score += self.comm_penalty_scale * comm_delay
            score += self.type_penalty.get(self.device_types[dev], 0.0)
            if self.device_types[dev] == "cloud":
                score += self.cloud_bias
                if self.cloud_fallback_only and non_cloud_feasible:
                    continue
                if self.cloud_block_budget is not None and cloud_blocks >= self.cloud_block_budget:
                    continue
                if self.cloud_volume_budget is not None and cloud_volume + self.demands[block].memory > self.cloud_volume_budget:
                    continue
            score -= self.bid_weight * bids[dev]
            scores[dev] = score
            if prev_dev is not None and dev == prev_dev:
                prev_score = score - self.stability_margin  # make previous slightly more attractive
        if not scores:
            if strict_queue and self.queue_block_threshold is not None:
                return self._choose_device_with_queue(block, bids, strict_queue=False)
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
            prev_max_load = max(
                (
                    max(c / (cap + 1e-6), m / (mem + 1e-6))
                    for c, cap, m, mem in zip(self.comp_used, self.compute, self.mem_used, self.memory)
                ),
                default=0.0,
            )
            prev_max_q = max(self.queue) if self.queue else 0.0
            prev_comp = self.comp_used[device]
            prev_mem = self.mem_used[device]
            prev_queue = self.queue[device]
            self.comp_used[device] += demand.compute
            self.mem_used[device] += demand.memory
            comp_delay = demand.compute / (self.compute[device] + 1e-6)
            comm_delay = self._comm_delay(block, device)
            cut_edges, cut_bytes = self._cut_metrics(block, device)
            mig_cost = self._migration_cost(block, device)
            prev_load = max(prev_comp / (self.compute[device] + 1e-6), prev_mem / (self.memory[device] + 1e-6))
            load_ratio = max(
                self.comp_used[device] / (self.compute[device] + 1e-6),
                self.mem_used[device] / (self.memory[device] + 1e-6),
            )
            improvement = max(prev_load - load_ratio, 0.0)
            self._update_queue(device, demand.compute, self.compute[device])
            comm_service = max(self._avg_bw_per_dev[device], 1e-6)
            self._update_comm_queue(device, comm_delay, comm_service)
            queue_penalty = max(self.queue[device] / max(self.queue_cap, 1.0) - self.queue_threshold, 0.0)
            queue_drift = max(self.queue[device] - prev_queue, 0.0)
            stick_bonus = 0.0
            if block in self.prev_assignment and self.prev_assignment[block] == device:
                if improvement > 0.0 or queue_penalty <= 0.0:
                    stick_bonus = self.stability_bonus
            base_reward = 1.0 - min(score_map.get(device, 0.0), 1.0)
            comm_term = self._norm(comm_delay, self.comm_norm)
            comp_term = self._norm(comp_delay, self.comp_norm)
            mig_term = self._norm(mig_cost, self.mig_norm)
            device_reward = -self.dpp_v * self._norm(comp_delay + comm_delay, self.delay_norm)
            device_reward -= self.dpp_v * self.comp_penalty_scale * comp_term
            device_reward -= self.dpp_v * self.migration_penalty_scale * mig_term
            device_reward -= self.dpp_v * self.comm_reward_weight * comm_term
            device_reward -= self.rho_q * self.queue_weight * min(queue_penalty, 1.0)
            device_reward -= self.rho_q * self.queue_drift_weight * min(queue_drift / max(self.queue_cap, 1.0), 1.0)
            device_reward -= self.rho_q * self.comm_queue_weight * min(
                self.comm_queue[device] / max(self.queue_cap, 1.0), 1.0
            )
            device_reward -= self.type_penalty.get(self.device_types[device], 0.0)
            device_reward += 0.5 * improvement
            device_reward += stick_bonus
            # encourage low-queue, low-load peers
            for peer in range(self.num_agents):
                if peer == device:
                    continue
                peer_load = max(
                    self.comp_used[peer] / (self.compute[peer] + 1e-6),
                    self.mem_used[peer] / (self.memory[peer] + 1e-6),
                )
                peer_queue = min(self.queue[peer] / max(self.queue_cap, 1.0), 1.0)
                rewards[peer] = 0.2 * (1.0 - peer_load) - 0.2 * peer_queue
            loads = [
                max(c / (cap + 1e-6), m / (mem + 1e-6))
                for c, cap, m, mem in zip(self.comp_used, self.compute, self.mem_used, self.memory)
            ]
            max_load = max(loads) if loads else 0.0
            min_load = min(loads) if loads else 0.0
            imbalance = max_load - min_load
            load_mean = sum(loads) / max(len(loads), 1)
            load_variance = sum((load - load_mean) ** 2 for load in loads) / max(len(loads), 1)
            if imbalance > self.imbalance_threshold:
                self.team_mix = 0.5
                self.max_load_weight = 0.8
            else:
                self.team_mix = self.team_mix_base
                self.max_load_weight = self.max_load_weight_base
            fairness_bonus = 0.1 * (1.0 - max_load) if loads else 0.0
            team_reward = sum(rewards) / max(self.num_agents, 1)
            max_load_hinge = max(max_load - self.max_load_hinge, 0.0)
            team_reward -= self.dpp_v * self.rho_w * self.max_load_weight * (max_load_hinge * max_load_hinge)
            team_reward += fairness_bonus
            team_reward -= self.dpp_v * self.balance_weight * load_variance
            if self.dependencies:
                team_reward -= self.dpp_v * self.cut_penalty_weight * (cut_edges / len(self.dependencies))
            team_reward -= self.dpp_v * self.comm_reward_weight * self._norm(comm_delay, self.comm_norm)
            team_reward -= self.dpp_v * self.cut_penalty_weight * self._norm(cut_bytes, self.comm_norm)
            if cut_edges > self.cut_edge_limit:
                team_reward -= self.dpp_v * self.cut_penalty_weight * (cut_edges - self.cut_edge_limit)
            if cut_bytes > self.cut_bytes_limit:
                team_reward -= self.dpp_v * self.cut_penalty_weight * self._norm(
                    cut_bytes - self.cut_bytes_limit, self.comm_norm
                )
            max_queue = max(self.queue) if self.queue else 0.0
            team_reward -= self.rho_q * self.max_queue_weight * min(max_queue / max(self.queue_cap, 1.0), 1.0)
            team_reward -= self.rho_q * self.queue_drift_weight * min(queue_drift / max(self.queue_cap, 1.0), 1.0)
            max_comm_queue = max(self.comm_queue) if self.comm_queue else 0.0
            team_reward -= self.rho_q * self.comm_queue_weight * min(
                max_comm_queue / max(self.queue_cap, 1.0), 1.0
            )
            mig_gain = 0.0
            if block in self.prev_assignment and self.prev_assignment[block] != device:
                mig_gain = max(prev_max_load - max_load, 0.0) + max(prev_max_q - max_queue, 0.0)
                if self.migration_hold_steps > 0:
                    self.cooldowns[block] = self.migration_hold_steps
            device_reward += mig_gain
            rewards[device] = device_reward
            team_reward = max(min(team_reward, 1.0), -1.0)
        else:
            # penalize everyone if no one could take the block
            team_reward = -self.failure_penalty
            rewards = [team_reward for _ in range(self.num_agents)]
            self.failed_blocks.append(block)
            retry_count = self.retry_counts.get(block, 0) + 1
            self.retry_counts[block] = retry_count
            if retry_count <= self.max_retries:
                self.blocks.append(block)
        if failed:
            team_reward = rewards[0] if rewards else -self.failure_penalty
        rewards = [((1.0 - self.team_mix) * r + self.team_mix * team_reward) for r in rewards]
        rewards = [max(min(r, 1.0), -1.0) for r in rewards]

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
                self.compute[dev] / (self._avg_compute + 1e-6),
                self.memory[dev] / (self._avg_memory + 1e-6),
                self.comp_used[dev] / (self._max_compute + 1e-6),
                self.mem_used[dev] / (self._max_memory + 1e-6),
                comp_ratio,
                mem_ratio,
                min(self.queue[dev] / self.queue_cap, 1.0),
                min(self.comm_queue[dev] / self.queue_cap, 1.0),
                min(self._avg_bw_per_dev[dev] / (self._max_bw + 1e-6), 1.0),
                min(self._reach_ratio[dev], 1.0),
                min(self.snr[dev] / (self._max_snr + 1e-6), 1.0),
                min(self.rssi[dev] / (self._max_rssi + 1e-6), 1.0),
                self.rho_q,
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
        comm_queue_norm = [min(q / self.queue_cap, 1.0) for q in self.comm_queue]
        state.extend(queue_norm)
        state.extend(comm_queue_norm)
        type_counts = [self.device_types.count(t) / max(self.num_agents, 1) for t in ("uav", "edge", "cloud")]
        state.extend(type_counts)
        state.append(sum(loads) / len(loads) if loads else 0.0)
        state.append(max(loads) if loads else 0.0)
        state.append(min(loads) if loads else 0.0)
        mean_bw = sum(self._avg_bw_per_dev) / max(len(self._avg_bw_per_dev), 1)
        mean_reach = sum(self._reach_ratio) / max(len(self._reach_ratio), 1)
        state.append(min(mean_bw / (self._max_bw + 1e-6), 1.0))
        state.append(min(mean_reach, 1.0))
        mean_snr = sum(self.snr) / max(len(self.snr), 1)
        mean_rssi = sum(self.rssi) / max(len(self.rssi), 1)
        state.append(min(mean_snr / (self._max_snr + 1e-6), 1.0))
        state.append(min(mean_rssi / (self._max_rssi + 1e-6), 1.0))
        mean_q = sum(self.queue) / max(len(self.queue), 1)
        state.append(min(mean_q / self.queue_cap, 1.0))
        max_q = max(self.queue) if self.queue else 0.0
        state.append(min(max_q / self.queue_cap, 1.0))
        state.append(self.rho_q)
        state.append(self.rho_w)
        expected = self.global_state_dim
        assert len(state) == expected, f"global state dim mismatch {len(state)} != {expected}"
        return state

    def action_mask(self) -> List[bool]:
        if self.block_idx >= len(self.blocks):
            return [False for _ in range(self.num_agents)]
        block = self.blocks[self.block_idx]
        return self._action_mask_with_queue(block, strict_queue=True)

    def _action_mask_with_queue(self, block: Block, strict_queue: bool) -> List[bool]:
        demand = self.demands[block]
        cloud_blocks = sum(1 for _, dev in self.assignment.items() if self.device_types[dev] == "cloud")
        cloud_volume = sum(
            self.demands[blk].memory
            for blk, dev in self.assignment.items()
            if self.device_types[dev] == "cloud"
        )
        mask: List[bool] = []
        non_cloud_feasible = False
        prelim: List[Tuple[float, bool]] = []
        for dev in range(self.num_agents):
            feasible, base_score = self._feasible(block, dev)
            if feasible and self.device_types[dev] != "cloud":
                non_cloud_feasible = True
            prelim.append((base_score, feasible))
        prev_dev = self.prev_assignment.get(block)
        prev_feasible, _ = self._feasible(block, prev_dev) if prev_dev is not None else (False, 0.0)
        for dev, (_, feasible) in enumerate(prelim):
            if not self._queue_allows(dev, strict=strict_queue):
                mask.append(False)
                continue
            if not feasible:
                mask.append(False)
                continue
            if not self._link_reachable(block, dev):
                mask.append(False)
                continue
            if not self._cooldown_allows(block, dev, prev_feasible, prev_dev):
                mask.append(False)
                continue
            if self.device_types[dev] == "cloud":
                if self.cloud_fallback_only and non_cloud_feasible:
                    mask.append(False)
                    continue
                if self.cloud_block_budget is not None and cloud_blocks >= self.cloud_block_budget:
                    mask.append(False)
                    continue
                if self.cloud_volume_budget is not None and cloud_volume + demand.memory > self.cloud_volume_budget:
                    mask.append(False)
                    continue
            mask.append(True)
        if strict_queue and not any(mask) and self.queue_block_threshold is not None:
            return self._action_mask_with_queue(block, strict_queue=False)
        return mask
