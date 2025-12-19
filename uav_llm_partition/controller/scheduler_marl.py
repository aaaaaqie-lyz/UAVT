"""Multi-agent RL-based scheduler wrapper."""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from uav_llm_partition.model_partition.blocks import Block
from uav_llm_partition.model_partition.demand_model import BlockDemand
from uav_llm_partition.rl.marl import MAPPOAgent, MultiAgentResourceAllocationEnv
from uav_llm_partition.sim.logger import logger

from .scheduler_baselines import SchedulerResult


class MARLScheduler:
    def __init__(self, load_guard: float = 1.0) -> None:
        self.load_guard = load_guard
        self.agent: MAPPOAgent | None = None

    def _ensure_agent(self, num_agents: int, local_state_dim: int) -> None:
        if self.agent is None:
            # bids are scalar per agent; action_dim equals num_agents to mirror device ids
            self.agent = MAPPOAgent(num_agents=num_agents, local_state_dim=local_state_dim, action_dim=num_agents)

    def assign(
        self,
        blocks: Sequence[Block],
        demands: Dict[Block, BlockDemand],
        compute: Sequence[float],
        memory: Sequence[float],
        weights: Sequence[float],
        lyapunov: Sequence[float],
        prev_assignment: Dict[Block, int],
        dependencies: Sequence[Tuple[Block, Block]],
        activation_sizes: Dict[Tuple[Block, Block], float],
        bandwidth: Sequence[Sequence[float]],
    ) -> SchedulerResult:
        env = MultiAgentResourceAllocationEnv(
            blocks=blocks,
            demands=demands,
            compute=compute,
            memory=memory,
            bandwidth=bandwidth,
            lyapunov=lyapunov,
            weights=weights,
            dependencies=dependencies,
            activation_sizes=activation_sizes,
            load_guard=self.load_guard,
        )
        local_states, _ = env.reset()
        self._ensure_agent(num_agents=len(compute), local_state_dim=len(local_states[0]) if local_states else 1)

        while env.block_idx < len(blocks):
            mask = env.action_mask()
            bids = self.agent.select_bids(local_states, mask)
            step = env.step(bids)
            if step.done:
                break
            local_states = step.next_local_states or []

        assignment = env.assignment
        migrations: List[Tuple[Block, int, int]] = []
        for blk, dev in assignment.items():
            if blk in prev_assignment and prev_assignment[blk] != dev:
                migrations.append((blk, prev_assignment[blk], dev))
        failed = len(assignment) != len(blocks)
        reason = "marl_no_feasible_device" if failed else None
        logger.debug("MARL assignment complete failed=%s reason=%s", failed, reason)
        return SchedulerResult(assignment=assignment, migrations=migrations, failed=failed, reason=reason)
