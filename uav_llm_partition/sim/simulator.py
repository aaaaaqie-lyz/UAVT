"""Simulator orchestrating UAV mobility, resource updates, scheduling, and metrics."""
from __future__ import annotations

from typing import Dict, List, Tuple

from uav_llm_partition.controller.lyapunov import LyapunovQueue
from uav_llm_partition.controller.scheduler_heuristic import SchedulerHeuristic
from uav_llm_partition.controller.weight_manager import WeightManager
from uav_llm_partition.env.channel_model import ChannelModel
from uav_llm_partition.env.resource_model import ResourceModel
from uav_llm_partition.env.uav_mobility import MobilityModel
from uav_llm_partition.model_partition.blocks import Block
from uav_llm_partition.model_partition.demand_model import DemandModel, BlockDemand
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
        head_dim: int,
        intervals: int = 50,
    ) -> None:
        self.num_uav = num_uav
        self.intervals = intervals
        self.mobility = MobilityModel(num_uav=num_uav)
        self.channel = ChannelModel()
        self.resource = ResourceModel(base_compute=1e9, base_memory=32.0)
        self.weights = WeightManager()
        self.demand_model = DemandModel(num_layers=num_layers, num_heads=num_heads, hidden_size=hidden_size, head_dim=head_dim)
        self.scheduler = SchedulerHeuristic()
        self.lyapunov = LyapunovQueue(num_uav=num_uav)
        self.metrics = MetricsLogger()
        self.blocks = self.demand_model.blocks()
        self.dependencies = build_dependencies(self.blocks, num_heads=num_heads)
        self.prev_assignment: Dict[Block, int] = {}

    def run(self) -> MetricsLogger:
        positions, mobility_risk = self.mobility.update()
        bandwidth, conn, los_score = self.channel.compute(positions)
        compute, memory = self.resource.sample(self.num_uav)
        weights = self.weights.compute(compute, memory, los_score, mobility_risk)
        lyapunov = self.lyapunov.pressure()

        for t in range(self.intervals):
            positions, mobility_risk = self.mobility.update()
            bandwidth, conn, los_score = self.channel.compute(positions)
            compute, memory = self.resource.sample(self.num_uav)
            demands = self.demand_model.update_interval()
            activation_sizes = {(u, v): self.demand_model.activation_size(u, v) for u, v in self.dependencies}
            weights = self.weights.compute(compute, memory, los_score, mobility_risk)
            assignment, migrations, failed = self.scheduler.assign(
                self.blocks,
                demands,
                compute,
                memory,
                weights,
                lyapunov=self.lyapunov.pressure(),
                prev_assignment=self.prev_assignment,
                dependencies=self.dependencies,
                activation_sizes=activation_sizes,
                bandwidth=bandwidth,
            )
            delay_comp, comp_load = self._compute_delay(assignment, demands, compute)
            delay_comm = self._communication_delay(assignment, bandwidth)
            delay_mig, mig_volume = self._migration_delay(migrations, bandwidth, demands)
            total_delay = delay_comp + delay_comm + delay_mig
            loads = self._load_vector(assignment, demands, compute, memory)
            max_load = max(loads)
            fairness = jain_fairness(loads)
            lyapunov = self.lyapunov.update(loads)
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
            )
            self.metrics.log(metrics)
            self.prev_assignment = assignment
            logger.info(
                "[t=%d] max_load=%.3f fairness=%.3f delay=%.3f comp=%.3f comm=%.3f mig=%.3f migs=%d failure=%s",
                t,
                max_load,
                fairness,
                total_delay,
                delay_comp,
                delay_comm,
                delay_mig,
                len(migrations),
                failed,
            )
        return self.metrics

    def _compute_delay(self, assignment: Dict[Block, int], demands: Dict[Block, BlockDemand], compute: List[float]) -> Tuple[float, List[float]]:
        comp_delay = 0.0
        comp_load = [0.0 for _ in range(self.num_uav)]
        for blk, dev in assignment.items():
            comp_delay += demands[blk].compute / (compute[dev] + 1e-6)
            comp_load[dev] += demands[blk].compute
        comp_load = [cl / (c + 1e-6) for cl, c in zip(comp_load, compute)]
        return comp_delay, comp_load

    def _communication_delay(self, assignment: Dict[Block, int], bandwidth: List[List[float]]) -> float:
        delay = 0.0
        for upstream, downstream in self.dependencies:
            dev_u = assignment[upstream]
            dev_d = assignment[downstream]
            if dev_u != dev_d:
                size = self.demand_model.activation_size(upstream, downstream)
                bw = bandwidth[dev_u][dev_d] + 1e-6
                delay += size / bw
        return delay

    def _migration_delay(
        self,
        migrations: Tuple[Tuple[Block, int, int], ...] | List[Tuple[Block, int, int]],
        bandwidth: List[List[float]],
        demands: Dict[Block, BlockDemand],
    ) -> Tuple[float, float]:
        if not migrations:
            return 0.0, 0.0
        delay = 0.0
        volume = 0.0
        for blk, src, dst in migrations:
            kv = demands[blk].kv_cache
            volume += kv
            rate = bandwidth[src][dst] + 1e-6
            delay += kv / rate + 1.0
        return delay, volume

    def _load_vector(
        self, assignment: Dict[Block, int], demands: Dict[Block, BlockDemand], compute: List[float], memory: List[float]
    ) -> List[float]:
        mem_load = [0.0 for _ in range(self.num_uav)]
        comp_load = [0.0 for _ in range(self.num_uav)]
        for blk, dev in assignment.items():
            mem_load[dev] += demands[blk].memory
            comp_load[dev] += demands[blk].compute
        mem_ratio = [m / (cap + 1e-6) for m, cap in zip(mem_load, memory)]
        comp_ratio = [c / (cap + 1e-6) for c, cap in zip(comp_load, compute)]
        return [max(mr, cr) for mr, cr in zip(mem_ratio, comp_ratio)]

