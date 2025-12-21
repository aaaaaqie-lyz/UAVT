"""Experience buffer for MAPPO-style multi-agent training."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence


@dataclass
class MAPPOTransition:
    local_states: List[List[float]]
    global_state: List[float]
    bids: List[float]
    log_probs: List[float]
    value: float
    reward: float  # team reward
    rewards_by_agent: List[float]
    done: bool


@dataclass
class MAPPOBuffer:
    transitions: List[MAPPOTransition] = field(default_factory=list)

    def add(
        self,
        local_states: List[List[float]],
        global_state: List[float],
        bids: Sequence[float],
        log_probs: Sequence[float],
        value: float,
        reward: float,
        rewards_by_agent: Sequence[float],
        done: bool,
    ) -> None:
        self.transitions.append(
            MAPPOTransition(
                local_states=list(local_states),
                global_state=list(global_state),
                bids=list(bids),
                log_probs=list(log_probs),
                value=float(value),
                reward=float(reward),
                rewards_by_agent=list(rewards_by_agent),
                done=done,
            )
        )

    def clear(self) -> None:
        self.transitions.clear()

    def as_batch(self) -> dict:
        """Return stacked trajectories for learning."""

        local_states = [t.local_states for t in self.transitions]
        global_states = [t.global_state for t in self.transitions]
        bids = [t.bids for t in self.transitions]
        log_probs = [t.log_probs for t in self.transitions]
        values = [t.value for t in self.transitions]
        rewards = [t.reward for t in self.transitions]
        rewards_by_agent = [t.rewards_by_agent for t in self.transitions]
        dones = [t.done for t in self.transitions]
        return {
            "local_states": local_states,
            "global_states": global_states,
            "bids": bids,
            "log_probs": log_probs,
            "values": values,
            "rewards": rewards,
            "rewards_by_agent": rewards_by_agent,
            "dones": dones,
        }
