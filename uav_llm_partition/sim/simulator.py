"""Simulator orchestrating UAV mobility, resource updates, scheduling, and metrics."""
from __future__ import annotations

from typing import Dict, List, Tuple

from uav_llm_partition.controller.lyapunov import LyapunovQueue
from uav_llm_partition.controller.scheduler_heuristic import SchedulerHeuristic
from uav_llm_partition.controller.scheduler_baselines import (
    BaseScheduler,
    DPScheduler,
    GreedyScheduler,
    MinLoadScheduler,
    ResourceAwareGreedyScheduler,
    RoundRobinScheduler,
)
from uav_llm_partition.controller.scheduler_layer import LayerPartitionScheduler
from uav_llm_partition.controller.scheduler_rl import RLScheduler
from uav_llm_partition.controller.scheduler_marl import MARLScheduler
from uav_llm_partition.controller.scheduler_rl_param import RLSchedulerParam
from uav_llm_partition.controller.state_collector import StateCollector
from uav_llm_partition.controller.weight_manager import WeightManager
from uav_llm_partition.env.channel_model import ChannelModel
from uav_llm_partition.env.resource_model import ResourceModel
from uav_llm_partition.env.uav_mobility import MobilityModel
from uav_llm_partition.model_partition.blocks import Block
from uav_llm_partition.model_partition.demand_model import BlockDemand, DemandModel
from uav_llm_partition.model_partition.graph_builder import build_dependencies
from uav_llm_partition.sim.metrics import IntervalMetrics, MetricsLogger, jain_fairness
from uav_llm_partition.sim.logger import logger


class Simulator:
    def __init__(
        self,
        num_uav: int,
        num_layers: int,
        num_heads: int,
        hidden_size: int,
        head_dim: int | None,
        intervals: int = 50,
        interval_tokens: int = 16,
        initial_seq_len: int = 128,
        use_lyapunov: bool = True,
        lyapunov_mode: str = "adaptive",
        lyapunov_penalty_value: float = 0.5,
        scheduler_type: str = "heuristic",
        layer_strategy: str = "round_robin",
        lyapunov_theta: float = 0.3,
    ) -> None:
        self.num_uav = num_uav
        self.intervals = intervals
        self.mobility = MobilityModel(num_uav=num_uav)
        self.channel = ChannelModel(base_rate=0.02)
        self.resource = ResourceModel(base_compute=4.0e9, base_memory=0.05)
        self.weights = WeightManager()
        self.demand_model = DemandModel(
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_size=hidden_size,
            head_dim=head_dim,
            interval_tokens=interval_tokens,
            initial_seq_len=initial_seq_len,
        )
        self.use_lyapunov = use_lyapunov
        self.lyapunov_mode = lyapunov_mode
        self.lyapunov_penalty_value = lyapunov_penalty_value
        self.scheduler_type = scheduler_type
        self.layer_strategy = layer_strategy
        self.scheduler = self._create_scheduler(scheduler_type, layer_strategy)
        self.lyapunov = LyapunovQueue(num_uav=num_uav, theta=lyapunov_theta)
        self.rl_param = RLSchedulerParam()
        self.state_collector = StateCollector()
        self.metrics = MetricsLogger()
        self.blocks = self.demand_model.blocks()
        self.dependencies = build_dependencies(self.blocks, num_heads=num_heads)
        self.prev_assignment: Dict[Block, int] = {}
        self._reward_buffer: List[float] = []

    def _create_scheduler(self, scheduler_type: str, layer_strategy: str):
        scheduler_type = (scheduler_type or "heuristic").lower()
        if scheduler_type == "heuristic":
            return SchedulerHeuristic()
        if scheduler_type == "layer":
            return LayerPartitionScheduler(strategy=layer_strategy)
        if scheduler_type == "greedy":
            return GreedyScheduler()
        if scheduler_type == "min_load":
            return MinLoadScheduler()
        if scheduler_type == "round_robin":
            return RoundRobinScheduler()
        if scheduler_type == "resource_aware":
            return ResourceAwareGreedyScheduler()
        if scheduler_type == "dp":
            return DPScheduler()
        if scheduler_type == "rl":
            return RLScheduler()
        if scheduler_type == "marl":
            return MARLScheduler()
        logger.warning("Unknown scheduler_type=%s, fallback to heuristic", scheduler_type)
        return SchedulerHeuristic()

    def run(self) -> MetricsLogger:
        positions, mobility_risk = self.mobility.update()
        bandwidth, conn, los_score = self.channel.compute(positions)
        compute, memory = self.resource.sample(self.num_uav)
        weights = self.weights.compute(compute, memory, los_score, mobility_risk)
        lyapunov = self.lyapunov.pressure() if self.use_lyapunov and self.lyapunov_mode != "none" else [0.0] * self.num_uav
        rl_params = self.rl_param.get()
        if isinstance(self.scheduler, SchedulerHeuristic):
            self.scheduler.weight_scale = rl_params.rho_w
            self.scheduler.lyapunov_penalty = rl_params.rho_q
            if self.lyapunov_mode == "fixed":
                self.scheduler.lyapunov_penalty = self.lyapunov_penalty_value
                self.scheduler.lyapunov_add_penalty = self.lyapunov_penalty_value / 2
            elif self.lyapunov_mode == "none":
                self.scheduler.lyapunov_penalty = 0.0
                self.scheduler.lyapunov_add_penalty = 0.0
            elif self.lyapunov_mode == "enhanced":
                self.scheduler.lyapunov_penalty = max(self.lyapunov_penalty_value, self.scheduler.lyapunov_penalty)
                self.scheduler.queue_block_threshold = 0.9

        for t in range(self.intervals):
            positions, mobility_risk = self.mobility.update()
            bandwidth, conn, los_score = self.channel.compute(positions)
            compute, memory = self.resource.sample(self.num_uav)
            demands = self.demand_model.update_interval()
            activation_sizes = {(u, v): self.demand_model.activation_size(u, v) for u, v in self.dependencies}
            weights = self.weights.compute(compute, memory, los_score, mobility_risk)
            controller_state = self.state_collector.build_state(
                compute=compute,
                memory=memory,
                weights=weights,
                lyapunov=lyapunov,
                positions=positions,
                los_score=los_score,
                mobility_risk=mobility_risk,
                demands=demands,
            )
            lyapunov_for_sched = lyapunov if self.use_lyapunov and self.lyapunov_mode != "none" else [0.0] * self.num_uav
            result = self.scheduler.assign(
                self.blocks,
                demands,
                compute,
                memory,
                weights,
                lyapunov=lyapunov_for_sched,
                prev_assignment=self.prev_assignment,
                dependencies=self.dependencies,
                activation_sizes=activation_sizes,
                bandwidth=bandwidth,
            )
            if isinstance(result, tuple):
                assignment, migrations, failed, failure_reason = result
            else:
                assignment, migrations, failed, failure_reason = (
                    result.assignment,
                    result.migrations,
                    result.failed,
                    result.reason,
                )
            delay_comp, comp_load, comp_delay_by_dev = self._compute_delay(assignment, demands, compute)
            delay_comm, comm_delay_by_dev = self._communication_delay(assignment, bandwidth, activation_sizes)
            delay_mig, mig_volume, mig_delay_by_dev = self._migration_delay(migrations, bandwidth, demands)
            total_delay = delay_comp + delay_comm + delay_mig
            total_delay_by_dev = [
                comp + comm + mig for comp, comm, mig in zip(comp_delay_by_dev, comm_delay_by_dev, mig_delay_by_dev)
            ]
            loads, mem_ratio, comp_ratio, mem_used, comp_used = self._load_vector(assignment, demands, compute, memory)
            max_load = max(loads)
            fairness = jain_fairness(loads)
            if self.use_lyapunov and self.lyapunov_mode != "none":
                lyapunov = self.lyapunov.update(loads)
            else:
                lyapunov = [0.0 for _ in range(self.num_uav)]

            rl_reward = -(
                total_delay
                + max_load
                + (8.0 if failed else 0.0)
                + (0.4 * len(migrations))
                + (0.2 * mig_volume)
            )
            if self.lyapunov_mode == "adaptive" and isinstance(self.scheduler, SchedulerHeuristic):
                self._reward_buffer.append(rl_reward)
                if (t + 1) % rl_params.window == 0:
                    window_reward = sum(self._reward_buffer[-rl_params.window :]) / rl_params.window
                    avg_queue = sum(lyapunov) / len(lyapunov) if lyapunov else 0.0
                    rl_params = self.rl_param.update_from_reward(window_reward, avg_queue=avg_queue)
                    self.scheduler.weight_scale = rl_params.rho_w
                    self.scheduler.lyapunov_penalty = rl_params.rho_q

            metrics = IntervalMetrics(
                max_load=max_load,
                fairness=fairness,
                delay=total_delay,
                delay_comp=delay_comp,
                delay_comm=delay_comm,
                delay_mig=delay_mig,
                migration_count=len(migrations),
                migration_volume=mig_volume,
                failure=failed,
                failure_reason=failure_reason,
                mem_loads=mem_ratio,
                comp_loads=comp_ratio,
                mem_used=mem_used,
                comp_used=comp_used,
                comp_delay_by_dev=comp_delay_by_dev,
                comm_delay_by_dev=comm_delay_by_dev,
                mig_delay_by_dev=mig_delay_by_dev,
                total_delay_by_dev=total_delay_by_dev,
                loads=loads,
                queues=lyapunov,
                weights=weights,
                rho_w=getattr(self.scheduler, "weight_scale", 0.0),
                rho_q=getattr(self.scheduler, "lyapunov_penalty", 0.0),
            )
            self.metrics.log(metrics)
            self.prev_assignment = assignment
            block_list = [[] for _ in range(self.num_uav)]
            kind_order = {"head": 0, "proj": 1, "ffn": 2}
            for blk, dev in assignment.items():
                block_list[dev].append(blk)
            device_lines = []
            for idx in range(self.num_uav):
                blocks = sorted(
                    block_list[idx], key=lambda b: (b.layer, kind_order.get(b.kind, 3), b.head_index or -1)
                )
                layer_set = sorted({b.layer for b in blocks})
                block_desc = []
                for b in blocks:
                    if b.kind == "head":
                        block_desc.append(f"L{b.layer}H{b.head_index}")
                    elif b.kind == "proj":
                        block_desc.append(f"L{b.layer}P")
                    else:
                        block_desc.append(f"L{b.layer}F")
                layer_str = "[" + ",".join(str(l) for l in layer_set) + "]"
                block_str = "[" + ",".join(block_desc) + "]"
                device_lines.append(
                    (
                        "d{idx}:delay={delay:.3f} load={load:.2f} comp={comp:.2f} mem={mem:.2f} q={q:.2f} "
                        "layers={layers} blocks={blocks}"
                    ).format(
                        idx=idx,
                        delay=total_delay_by_dev[idx],
                        load=loads[idx],
                        comp=comp_ratio[idx],
                        mem=mem_ratio[idx],
                        q=lyapunov[idx],
                        layers=layer_str,
                        blocks=block_str,
                    )
                )
            logger.info(
                "[t=%d] max_load=%.3f fairness=%.3f delay=%.3f comp=%.3f comm=%.3f mig=%.3f migs=%d failure=%s reason=%s devices=%s rho_w=%.2f rho_q=%.2f",
                t,
                max_load,
                fairness,
                total_delay,
                delay_comp,
                delay_comm,
                delay_mig,
                len(migrations),
                failed,
                failure_reason or "",
                " | ".join(device_lines),
                getattr(self.scheduler, "weight_scale", 0.0),
                getattr(self.scheduler, "lyapunov_penalty", 0.0),
            )
        return self.metrics

    def _compute_delay(
        self, assignment: Dict[Block, int], demands: Dict[Block, BlockDemand], compute: List[float]
    ) -> Tuple[float, List[float], List[float]]:
        comp_delay = 0.0
        comp_load = [0.0 for _ in range(self.num_uav)]
        for blk, dev in assignment.items():
            comp_delay += demands[blk].compute / (compute[dev] + 1e-6)
            comp_load[dev] += demands[blk].compute
        comp_ratio = [cl / (c + 1e-6) for cl, c in zip(comp_load, compute)]
        comp_delay_by_dev = [cl / (c + 1e-6) for cl, c in zip(comp_load, compute)]
        return comp_delay, comp_ratio, comp_delay_by_dev

    def _communication_delay(
        self,
        assignment: Dict[Block, int],
        bandwidth: List[List[float]],
        activation_sizes: Dict[Tuple[Block, Block], float],
    ) -> Tuple[float, List[float]]:
        delay = 0.0
        per_device = [0.0 for _ in range(self.num_uav)]
        for upstream, downstream in self.dependencies:
            dev_u = assignment[upstream]
            dev_d = assignment[downstream]
            if dev_u != dev_d:
                size = activation_sizes.get((upstream, downstream), activation_sizes.get((downstream, upstream), 0.0))
                bw = bandwidth[dev_u][dev_d] + 1e-6
                edge_delay = size / bw
                delay += edge_delay
                per_device[dev_u] += edge_delay / 2
                per_device[dev_d] += edge_delay / 2
        return delay, per_device

    def _migration_delay(
        self,
        migrations: Tuple[Tuple[Block, int, int], ...] | List[Tuple[Block, int, int]],
        bandwidth: List[List[float]],
        demands: Dict[Block, BlockDemand],
    ) -> Tuple[float, float, List[float]]:
        if not migrations:
            return 0.0, 0.0, [0.0 for _ in range(self.num_uav)]
        delay = 0.0
        volume = 0.0
        per_device = [0.0 for _ in range(self.num_uav)]
        for blk, src, dst in migrations:
            kv = demands[blk].kv_cache
            volume += kv
            rate = bandwidth[src][dst] + 1e-6
            mig_time = kv / rate + self.scheduler.mig_overhead
            delay += mig_time
            per_device[src] += mig_time / 2
            per_device[dst] += mig_time / 2
        return delay, volume, per_device

    def _load_vector(
        self, assignment: Dict[Block, int], demands: Dict[Block, BlockDemand], compute: List[float], memory: List[float]
    ) -> Tuple[List[float], List[float], List[float], List[float], List[float]]:
        mem_load = [0.0 for _ in range(self.num_uav)]
        comp_load = [0.0 for _ in range(self.num_uav)]
        for blk, dev in assignment.items():
            mem_load[dev] += demands[blk].memory
            comp_load[dev] += demands[blk].compute
        mem_ratio = [m / (cap + 1e-6) for m, cap in zip(mem_load, memory)]
        comp_ratio = [c / (cap + 1e-6) for c, cap in zip(comp_load, compute)]
        return [max(mr, cr) for mr, cr in zip(mem_ratio, comp_ratio)], mem_ratio, comp_ratio, mem_load, comp_load
