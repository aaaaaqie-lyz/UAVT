"""Simple training loop utilities for the RL scheduler."""
from __future__ import annotations

from typing import Callable

from .agent import RLAgent
from .env import RLResourceAllocationEnv


def train_agent(
    env_factory: Callable[[], RLResourceAllocationEnv],
    agent: RLAgent,
    episodes: int = 500,
    deterministic_eval: bool = True,
) -> None:
    """Run a lightweight training loop over simulated episodes."""

    for ep in range(episodes):
        # decay exploration and learning rate over time
        agent.config.epsilon = max(agent.config.epsilon_min, agent.config.epsilon * agent.config.epsilon_decay)
        lr = agent.config.lr * (agent.config.lr_decay ** max(ep // 100, 0))

        env = env_factory()
        state = env.reset()
        done = False
        while not done:
            mask = env.action_mask()
            action = agent.select_action(state, mask)
            next_state, reward, done, _ = env.step(action)
            agent.store(env.steps[-1])
            state = next_state
        agent.config.lr = lr
        agent.update(final_reward=env.reward)
        eval_reward = None
        if deterministic_eval:
            eval_reward = evaluate(env_factory, agent)
        print(
            f"Episode {ep + 1}/{episodes}: train_reward={env.reward:.3f}"
            + (f", eval_reward={eval_reward:.3f}" if eval_reward is not None else "")
        )


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
