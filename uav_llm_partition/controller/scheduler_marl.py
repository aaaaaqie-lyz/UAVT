"""Multi-agent RL-based scheduler wrapper."""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from uav_llm_partition.model_partition.blocks import Block
from uav_llm_partition.model_partition.demand_model import BlockDemand
from uav_llm_partition.rl.marl import MAPPOAgent, MultiAgentResourceAllocationEnv
from uav_llm_partition.sim.logger import logger

from .scheduler_baselines import SchedulerResult


class MARLScheduler:
    def __init__(self, load_guard: float = 1.0, model_path: str | None = None, mig_overhead: float = 0.01) -> None:
        self.load_guard = load_guard
        self.agent: MAPPOAgent | None = None
        self.model_path = model_path
        self.mig_overhead = mig_overhead

    def _ensure_agent(self, num_agents: int, local_state_dim: int, global_state_dim: int) -> None:
        """Load or create a MAPPO agent with dimension validation and fallbacks."""

        def _create() -> None:
            self.agent = MAPPOAgent(
                MAPPOAgent.default_config(
                    num_agents=num_agents,
                    local_state_dim=local_state_dim,
                    global_state_dim=global_state_dim,
                )
            )

        if self.agent is not None:
            cfg = self.agent.cfg
            if (
                cfg.num_agents == num_agents
                and cfg.local_state_dim == local_state_dim
                and cfg.global_state_dim == global_state_dim
            ):
                return
            logger.warning(
                "Existing MARL agent dims mismatch (agents=%s/%s, local=%s/%s, global=%s/%s), recreating",
                cfg.num_agents,
                num_agents,
                cfg.local_state_dim,
                local_state_dim,
                cfg.global_state_dim,
                global_state_dim,
            )
            self.agent = None

        if self.model_path:
            try:
                candidate = MAPPOAgent.load(self.model_path)
                cfg = candidate.cfg
                if (
                    cfg.num_agents == num_agents
                    and cfg.local_state_dim == local_state_dim
                    and cfg.global_state_dim == global_state_dim
                ):
                    self.agent = candidate
                    logger.info("Loaded MARL agent from %s", self.model_path)
                    return
                logger.warning(
                    "MARL model dims mismatch; expected agents=%s, local=%s, global=%s, got agents=%s, local=%s, global=%s."
                    " Recreating fresh agent.",
                    num_agents,
                    local_state_dim,
                    global_state_dim,
                    cfg.num_agents,
                    cfg.local_state_dim,
                    cfg.global_state_dim,
                )
            except (FileNotFoundError, KeyError, ValueError, IndexError, TypeError) as exc:
                logger.error("Failed to load MARL model from %s: %s", self.model_path, exc)

        _create()

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
        latency: Sequence[Sequence[float]],
        device_types: Sequence[str] | None = None,
    ) -> SchedulerResult:
        resolved_types = list(device_types) if device_types is not None else getattr(self, "device_types", ["uav" for _ in compute])

        env = MultiAgentResourceAllocationEnv(
            blocks=blocks,
            demands=demands,
            compute=compute,
            memory=memory,
            bandwidth=bandwidth,
            latency=latency,
            lyapunov=lyapunov,
            weights=weights,
            dependencies=dependencies,
            activation_sizes=activation_sizes,
            load_guard=self.load_guard,
            prev_assignment=prev_assignment,
            migration_overhead=self.mig_overhead,
            device_types=resolved_types,
        )
        local_states, global_state = env.reset()

        # Try to keep feasible previous assignments and only reassign what is infeasible
        kept_assignment, comp_used, mem_used, pending_blocks = env.retain_feasible_prev()
        env.apply_partial_state(kept_assignment, comp_used, mem_used, pending_blocks)
        local_states, global_state = env.current_states()

        self._ensure_agent(
            num_agents=len(compute),
            local_state_dim=len(local_states[0]) if local_states else 1,
            global_state_dim=len(global_state) if global_state else 1,
        )

        while env.block_idx < len(env.blocks):
            bids, _, _, _ = self.agent.select_bids(local_states, deterministic=True)
            if resolved_types:
                scaled = []
                for bid, dev_type in zip(bids, resolved_types):
                    if dev_type == "cloud":
                        scale = 0.5
                    elif dev_type == "edge":
                        scale = 0.8
                    else:
                        scale = 1.2
                    scaled.append(bid * scale)
                bids = scaled
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
