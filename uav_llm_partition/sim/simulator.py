"""Simulator orchestrating UAV mobility, resource updates, scheduling, and metrics."""
from __future__ import annotations

from typing import Dict, List, Tuple

from uav_llm_partition.controller.lyapunov import LyapunovQueue
from uav_llm_partition.controller.scheduler_heuristic import SchedulerHeuristic
from uav_llm_partition.controller.scheduler_rl_param import RLSchedulerParam
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
    ) -> None:
        self.num_uav = num_uav
        self.intervals = intervals
        self.mobility = MobilityModel(num_uav=num_uav)
        self.channel = ChannelModel()
        self.resource = ResourceModel(base_compute=1e9, base_memory=0.35)
        self.weights = WeightManager()
        self.demand_model = DemandModel(
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_size=hidden_size,
            head_dim=head_dim,
            interval_tokens=interval_tokens,
            initial_seq_len=initial_seq_len,
        )
        self.scheduler = SchedulerHeuristic()
        self.lyapunov = LyapunovQueue(num_uav=num_uav)
        self.rl_param = RLSchedulerParam()
        self.metrics = MetricsLogger()
        self.blocks = self.demand_model.blocks()
        self.dependencies = build_dependencies(self.blocks, num_heads=num_heads)
        self.prev_assignment: Dict[Block, int] = {}
        self._reward_buffer: List[float] = []

    def run(self) -> MetricsLogger:
        positions, mobility_risk = self.mobility.update()
        bandwidth, conn, los_score = self.channel.compute(positions)
        compute, memory = self.resource.sample(self.num_uav)
        weights = self.weights.compute(compute, memory, los_score, mobility_risk)
        lyapunov = self.lyapunov.pressure()
        rl_params = self.rl_param.get()
        self.scheduler.weight_scale = rl_params.rho_w
        self.scheduler.lyapunov_penalty = rl_params.rho_q

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
            lyapunov = self.lyapunov.update(loads)

            rl_reward = -(
                total_delay
                + max_load
                + (10.0 if failed else 0.0)
                + (0.2 * len(migrations))
                + (0.1 * mig_volume)
            )
            self._reward_buffer.append(rl_reward)
            if (t + 1) % rl_params.window == 0:
                window_reward = sum(self._reward_buffer[-rl_params.window :]) / rl_params.window
                rl_params = self.rl_param.update_from_reward(window_reward)
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
                rho_w=self.scheduler.weight_scale,
                rho_q=self.scheduler.lyapunov_penalty,
            )
            self.metrics.log(metrics)
            self.prev_assignment = assignment
            logger.info(
                "[t=%d] max_load=%.3f fairness=%.3f delay=%.3f comp=%.3f comm=%.3f mig=%.3f migs=%d failure=%s mem=%s comp=%s queue=%s comp_delay=%s comm_delay=%s mig_delay=%s total_delay=%s rho_w=%.2f rho_q=%.2f",
                t,
                max_load,
                fairness,
                total_delay,
                delay_comp,
                delay_comm,
                delay_mig,
                len(migrations),
                failed,
                [round(v, 3) for v in mem_ratio],
                [round(v, 3) for v in comp_ratio],
                [round(q, 3) for q in lyapunov],
                [round(v, 3) for v in comp_delay_by_dev],
                [round(v, 3) for v in comm_delay_by_dev],
                [round(v, 3) for v in mig_delay_by_dev],
                [round(v, 3) for v in total_delay_by_dev],
                self.scheduler.weight_scale,
                self.scheduler.lyapunov_penalty,
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
