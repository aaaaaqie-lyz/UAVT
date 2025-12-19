"""Simple training loop utilities for the RL scheduler."""
from __future__ import annotations

from typing import Callable

from .agent import RLAgent
from .env import RLResourceAllocationEnv


def train_agent(
    env_factory: Callable[[], RLResourceAllocationEnv],
    agent: RLAgent,
    episodes: int = 50,
    deterministic_eval: bool = True,
) -> None:
    """Run a lightweight training loop over simulated episodes."""

    for _ in range(episodes):
        env = env_factory()
        state = env.reset()
        done = False
        while not done:
            mask = env.action_mask()
            action = agent.select_action(state, mask)
            next_state, reward, done, _ = env.step(action)
            agent.store(env.steps[-1])
            state = next_state
        agent.update(final_reward=env.reward)
        if deterministic_eval:
            _ = evaluate(env_factory, agent)


def evaluate(env_factory: Callable[[], RLResourceAllocationEnv], agent: RLAgent) -> float:
    """Evaluate the current policy deterministically and return episode reward."""

    env = env_factory()
    state = env.reset()
    done = False
    while not done:
        mask = env.action_mask()
        action = agent.select_action(state, mask, deterministic=True)
        state, _, done, _ = env.step(action)
    agent.buffer.clear()
    return env.reward
