"""Stub trainer for multi-agent RL (MAPPO-style)."""
from __future__ import annotations

from typing import Iterable

from .env import MultiAgentResourceAllocationEnv
from .mappo import MAPPOAgent


class MARLTrainer:
    def __init__(self, agent: MAPPOAgent) -> None:
        self.agent = agent

    def run_episode(
        self,
        env: MultiAgentResourceAllocationEnv,
        max_steps: int | None = None,
    ) -> float:
        """Run one rollout and return cumulative reward sum across agents."""

        local_states, _ = env.reset()
        total_reward = 0.0
        steps = 0
        while True:
            mask = env.action_mask()
            bids = self.agent.select_bids(local_states, mask)
            step = env.step(bids)
            total_reward += sum(step.rewards)
            if step.done:
                break
            local_states = step.next_local_states or []
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break
        return total_reward

    def train(
        self,
        env_factory: Iterable[MultiAgentResourceAllocationEnv],
        episodes: int = 10,
    ) -> None:
        for _ in range(episodes):
            for env in env_factory:
                self.run_episode(env)
