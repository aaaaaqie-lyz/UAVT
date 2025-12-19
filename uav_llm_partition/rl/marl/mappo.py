"""Minimal MAPPO-style agent producing per-UAV bids for blocks."""
from __future__ import annotations

from typing import List, Sequence

from uav_llm_partition.rl.networks import PolicyNetwork


class MAPPOAgent:
    """Simplified MAPPO actor collection.

    This implementation keeps one lightweight policy network per agent. It is
    deliberately small and CPU-only to keep the simulation dependency-light.
    """

    def __init__(self, num_agents: int, local_state_dim: int, action_dim: int) -> None:
        self.num_agents = num_agents
        self.local_state_dim = local_state_dim
        self.action_dim = action_dim
        self.actors = [PolicyNetwork(local_state_dim, action_dim, hidden_dims=(64, 32)) for _ in range(num_agents)]

    def select_bids(self, local_states: Sequence[Sequence[float]], action_mask: Sequence[bool]) -> List[float]:
        bids: List[float] = []
        for agent_id, state in enumerate(local_states):
            probs = self.actors[agent_id].forward(state)
            # compress masked actions by redistributing probability mass to valid slots
            masked = [p if m else 0.0 for p, m in zip(probs, action_mask)]
            total = sum(masked)
            if total <= 0.0:
                bids.append(0.0)
                continue
            normed = [p / total for p in masked]
            # use expected device index as a soft bid value
            bid = sum(idx * p for idx, p in enumerate(normed)) / max(self.action_dim - 1, 1)
            bids.append(bid)
        return bids

    def learn(self, *_) -> None:  # placeholder for future training loop
        return None
