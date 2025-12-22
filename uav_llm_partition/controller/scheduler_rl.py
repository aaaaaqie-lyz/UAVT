"""RL-based scheduler that delegates placement to a lightweight agent."""
from __future__ import annotations

from typing import Dict, List, Tuple

from uav_llm_partition.model_partition.blocks import Block
from uav_llm_partition.model_partition.demand_model import BlockDemand
from uav_llm_partition.rl import AgentConfig, RLAgent, RLResourceAllocationEnv
from uav_llm_partition.controller.scheduler_baselines import BaseScheduler, SchedulerResult


class RLScheduler(BaseScheduler):
    """Wraps the RL agent to produce block→device assignments."""

    def __init__(self, comm_budget: float = 0.05, mig_overhead: float = 0.01, epsilon: float = 0.1) -> None:
        super().__init__(comm_budget=comm_budget, mig_overhead=mig_overhead)
        self.agent: RLAgent | None = None
        self.epsilon = epsilon

    def _ensure_agent(self, state_dim: int, action_dim: int) -> None:
        if self.agent is None:
            cfg = AgentConfig(state_dim=state_dim, action_dim=action_dim, epsilon=self.epsilon)
            self.agent = RLAgent(cfg)

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
        device_types: List[str] | None = None,
    ) -> SchedulerResult:
        if not blocks:
            return SchedulerResult({}, [], False, "")

        env = RLResourceAllocationEnv(
            blocks=blocks,
            demands=demands,
            compute=compute,
            memory=memory,
            weights=weights,
            lyapunov=lyapunov,
            dependencies=dependencies,
            activation_sizes=activation_sizes,
            bandwidth=bandwidth,
            prev_assignment=prev_assignment,
            load_guard=1.0,
            queue_block_threshold=0.95,
        )
        state = env.reset()
        state_dim = len(state)
        action_dim = len(compute)
        self._ensure_agent(state_dim, action_dim)
        assert self.agent is not None

        done = False
        while not done:
            mask = env.action_mask()
            action = self.agent.select_action(state, mask)
            next_state, reward, done, _ = env.step(action)
            self.agent.store(env.steps[-1])
            state = next_state

        result = env.episode_result()
        self.agent.update(final_reward=result.reward)
        failed = result.failed or any(
            max((self._ratios(blk, demands[blk], dev, compute, memory, result.assignment, prev_assignment, dependencies, activation_sizes, bandwidth, [0.0] * len(compute), [0.0] * len(memory))[:3]))
            > 1.0
            for blk, dev in result.assignment.items()
        )
        reason = result.reason if result.failed else ("infeasible" if failed else "")
        return SchedulerResult(result.assignment, result.migrations, failed, reason)

    # Reuse BaseScheduler ratio helper for final feasibility check
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
        # Mirror BaseScheduler._ratios but without mutating usage.
        comp_ratio = (comp_used[device] + demand.compute) / max(compute[device], 1e-6)
        mem_ratio = (mem_used[device] + demand.memory) / max(memory[device], 1e-6)
        comm_delay = 0.0
        for up, down in dependencies:
            if block not in (up, down):
                continue
            neighbor = down if block == up else up
            neighbor_dev = assignment.get(neighbor, prev_assignment.get(neighbor))
            if neighbor_dev is None or neighbor_dev == device:
                continue
            size = activation_sizes.get((up, down), activation_sizes.get((down, up), 0.0))
            bw = bandwidth[device][neighbor_dev] + 1e-6
            comm_delay += size / bw
        comm_ratio = comm_delay / self.comm_budget
        return comp_ratio, mem_ratio, comm_ratio
