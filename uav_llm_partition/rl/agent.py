"""Minimal RL agent for block-to-device assignment without external deps."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from .networks import PolicyNetwork, ValueNetwork
from .env import RLStep


@dataclass
class AgentConfig:
    state_dim: int
    action_dim: int
    gamma: float = 0.99
    lr: float = 1e-3
    epsilon: float = 0.1
    baseline: bool = True


@dataclass
class TransitionBuffer:
    steps: List[RLStep] = field(default_factory=list)

    def clear(self) -> None:
        self.steps.clear()

    def append(self, step: RLStep) -> None:
        self.steps.append(step)

    def to_arrays(self):
        states = [s.state for s in self.steps]
        actions = [s.action for s in self.steps]
        rewards = [s.reward for s in self.steps]
        dones = [s.done for s in self.steps]
        return states, actions, rewards, dones


class RLAgent:
    """A tiny REINFORCE-style agent with epsilon-greedy exploration."""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.policy = PolicyNetwork(config.state_dim, config.action_dim)
        self.value = ValueNetwork(config.state_dim)
        self.buffer = TransitionBuffer()

    def select_action(self, state: Sequence[float], mask: List[bool], deterministic: bool = False) -> int:
        feasible = [i for i, ok in enumerate(mask) if ok]
        if not feasible:
            return 0
        import random

        if not deterministic and random.random() < self.config.epsilon:
            return int(random.choice(feasible))
        probs = self.policy.forward(state)
        masked = [p if mask[i] else 0.0 for i, p in enumerate(probs)]
        total = sum(masked)
        if total <= 0:
            return int(random.choice(feasible))
        masked = [p / total for p in masked]
        if deterministic:
            return int(max(range(len(masked)), key=lambda i: masked[i]))
        r = random.random()
        cdf = 0.0
        for i, p in enumerate(masked):
            cdf += p
            if r <= cdf:
                return i
        return feasible[-1]

    def store(self, step: RLStep) -> None:
        self.buffer.append(step)

    def update(self, final_reward: Optional[float] = None) -> None:
        if not self.buffer.steps:
            return
        states, actions, rewards, dones = self.buffer.to_arrays()
        returns: List[float] = []
        G = final_reward if final_reward is not None else 0.0
        for r, done in zip(reversed(rewards), reversed(dones)):
            if done:
                G = r
            else:
                G = r + self.config.gamma * G
            returns.insert(0, G)

        advantages = returns.copy()
        if self.config.baseline:
            baselines = [self.value.forward(s) for s in states]
            advantages = [ret - base for ret, base in zip(returns, baselines)]
            self.value.update(states, returns, lr=self.config.lr)

        self.policy.update(states, actions, advantages, lr=self.config.lr)
        self.buffer.clear()
