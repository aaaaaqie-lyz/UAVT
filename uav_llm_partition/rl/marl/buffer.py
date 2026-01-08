"""Experience buffer for MAPPO-style multi-agent training."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence


@dataclass
class MAPPOTransition:
    local_states: List[List[float]]
    global_state: List[float]
    next_local_states: List[List[float]] | None
    next_global_state: List[float] | None
    actions: List[float]
    bids: List[float]
    log_probs: List[float]
    means: List[float]
    stds: List[float]
    value: float
    next_value: float
    reward: float  # team reward
    rewards_by_agent: List[float]
    action_mask: List[bool]
    done: bool


@dataclass
class MAPPOBuffer:
    transitions: List[MAPPOTransition] = field(default_factory=list)

    def add(
        self,
        local_states: List[List[float]],
        global_state: List[float],
        next_local_states: List[List[float]] | None,
        next_global_state: List[float] | None,
        actions: Sequence[float],
        bids: Sequence[float],
        log_probs: Sequence[float],
        means: Sequence[float],
        stds: Sequence[float],
        value: float,
        next_value: float,
        reward: float,
        rewards_by_agent: Sequence[float],
        action_mask: Sequence[bool],
        done: bool,
    ) -> None:
        self.transitions.append(
            MAPPOTransition(
                local_states=list(local_states),
                global_state=list(global_state),
                next_local_states=list(next_local_states) if next_local_states is not None else None,
                next_global_state=list(next_global_state) if next_global_state is not None else None,
                actions=list(actions),
                bids=list(bids),
                log_probs=list(log_probs),
                means=list(means),
                stds=list(stds),
                value=float(value),
                next_value=float(next_value),
                reward=float(reward),
                rewards_by_agent=list(rewards_by_agent),
                action_mask=list(action_mask),
                done=done,
            )
        )

    def clear(self) -> None:
        self.transitions.clear()

    def as_batch(self) -> dict:
        """Return stacked trajectories for learning."""

        local_states = [t.local_states for t in self.transitions]
        global_states = [t.global_state for t in self.transitions]
        next_local_states = [t.next_local_states for t in self.transitions]
        next_global_states = [t.next_global_state for t in self.transitions]
        bids = [t.bids for t in self.transitions]
        actions = [t.actions for t in self.transitions]
        log_probs = [t.log_probs for t in self.transitions]
        means = [t.means for t in self.transitions]
        stds = [t.stds for t in self.transitions]
        values = [t.value for t in self.transitions]
        next_values = [t.next_value for t in self.transitions]
        rewards = [t.reward for t in self.transitions]
        rewards_by_agent = [t.rewards_by_agent for t in self.transitions]
        action_masks = [t.action_mask for t in self.transitions]
        dones = [t.done for t in self.transitions]
        return {
            "local_states": local_states,
            "global_states": global_states,
            "next_local_states": next_local_states,
            "next_global_states": next_global_states,
            "bids": bids,
            "actions": actions,
            "log_probs": log_probs,
            "means": means,
            "stds": stds,
            "values": values,
            "next_values": next_values,
            "rewards": rewards,
            "rewards_by_agent": rewards_by_agent,
            "action_masks": action_masks,
            "dones": dones,
        }
