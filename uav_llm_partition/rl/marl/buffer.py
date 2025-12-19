"""Experience buffer for MAPPO-style multi-agent training."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence


@dataclass
class MAPPOTransition:
    local_states: List[List[float]]
    global_state: List[float]
    actions: List[int]
    log_probs: List[float]
    value: float
    rewards: List[float]
    done: bool


@dataclass
class MAPPOBuffer:
    transitions: List[MAPPOTransition] = field(default_factory=list)

    def add(
        self,
        local_states: List[List[float]],
        global_state: List[float],
        actions: Sequence[int],
        log_probs: Sequence[float],
        value: float,
        rewards: Sequence[float],
        done: bool,
    ) -> None:
        self.transitions.append(
            MAPPOTransition(
                local_states=list(local_states),
                global_state=list(global_state),
                actions=list(actions),
                log_probs=list(log_probs),
                value=value,
                rewards=list(rewards),
                done=done,
            )
        )

    def clear(self) -> None:
        self.transitions.clear()
