"""MAPPO-inspired agent with shared buffer and PPO-style updates."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

from uav_llm_partition.rl.networks import PolicyNetwork, ValueNetwork


@dataclass
class MAPPOConfig:
    num_agents: int
    local_state_dim: int
    global_state_dim: int
    action_dim: int
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.05
    lr: float = 3e-4
    value_lr_scale: float = 0.25
    ppo_epochs: int = 6
    max_grad_norm: float = 0.5


class MAPPOAgent:
    """Simplified MAPPO actor-critic collection with per-agent actors."""

    def __init__(self, config: MAPPOConfig) -> None:
        self.cfg = config
        self.actors = [
            PolicyNetwork(config.local_state_dim, config.action_dim, hidden_dims=(64, 32))
            for _ in range(config.num_agents)
        ]
        self.critic = ValueNetwork(config.global_state_dim, hidden_dims=(64, 32))

    def select_actions(
        self, local_states: Sequence[Sequence[float]], action_mask: Sequence[bool], deterministic: bool = False
    ) -> tuple[List[int], List[float], float]:
        actions: List[int] = []
        log_probs: List[float] = []
        entropies: List[float] = []
        for agent_id, state in enumerate(local_states):
            action, log_prob, entropy = self.actors[agent_id].get_action_and_log_prob(state, action_mask, deterministic)
            # If entropy collapses, inject a small amount of randomness to keep exploration alive
            if entropy < 0.05 and not deterministic:
                import random

                feasible = [i for i, ok in enumerate(action_mask) if ok]
                if feasible:
                    action = int(random.choice(feasible))
                    log_prob = -math.log(len(feasible))
                    entropy = 0.5
            actions.append(action)
            log_probs.append(log_prob)
            entropies.append(entropy)
        avg_entropy = sum(entropies) / max(len(entropies), 1)
        return actions, log_probs, avg_entropy

    def value(self, global_state: Sequence[float]) -> float:
        return self.critic.forward(global_state)

    def _compute_gae(
        self, rewards: List[float], values: List[float], dones: List[bool]
    ) -> tuple[List[float], List[float]]:
        advantages: List[float] = []
        returns: List[float] = []
        gae = 0.0
        next_value = 0.0
        for t in reversed(range(len(rewards))):
            mask = 0.0 if dones[t] else 1.0
            delta = rewards[t] + self.cfg.gamma * next_value * mask - values[t]
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * gae * mask
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
            next_value = values[t]
        return advantages, returns

    def learn(
        self,
        states: List[List[float]],
        global_states: List[List[float]],
        actions: List[List[int]],
        old_log_probs: List[List[float]],
        rewards: List[float],
        dones: List[bool],
    ) -> dict:
        if not rewards:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        # normalize rewards to avoid exploding returns
        mean_r = sum(rewards) / max(len(rewards), 1)
        std_r = math.sqrt(sum((r - mean_r) ** 2 for r in rewards) / max(len(rewards), 1))
        if std_r > 1e-6:
            rewards = [(r - mean_r) / (std_r + 1e-6) for r in rewards]

        values = [self.value(gs) for gs in global_states]
        advantages, returns = self._compute_gae(rewards, values, dones)
        returns = [max(min(ret, 10.0), -10.0) for ret in returns]
        # advantage normalization
        mean_adv = sum(advantages) / max(len(advantages), 1)
        std_adv = math.sqrt(sum((a - mean_adv) ** 2 for a in advantages) / max(len(advantages), 1))
        if std_adv > 1e-8:
            advantages = [(a - mean_adv) / (std_adv + 1e-8) for a in advantages]

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        for _ in range(self.cfg.ppo_epochs):
            policy_loss = 0.0
            value_loss = 0.0
            entropy_acc = 0.0
            for step_idx in range(len(states)):
                adv = advantages[step_idx]
                ret = returns[step_idx]
                # policy update per agent
                for agent_id, actor in enumerate(self.actors):
                    action = actions[step_idx][agent_id]
                    probs = actor.forward(states[step_idx][agent_id])
                    new_log_prob = actor.get_log_prob(probs, action)
                    ratio = math.exp(new_log_prob - old_log_probs[step_idx][agent_id])
                    surr1 = ratio * adv
                    surr2 = max(1.0 - self.cfg.clip_epsilon, min(1.0 + self.cfg.clip_epsilon, ratio)) * adv
                    entropy_term = -sum(p * math.log(max(p, 1e-8)) for p in probs)
                    policy_loss += -min(surr1, surr2) - self.cfg.entropy_coef * entropy_term
                    entropy_acc += entropy_term

                # value update once per timestep using global critic
                v_pred = self.value(global_states[step_idx])
                value_loss += (v_pred - ret) ** 2

            # apply lightweight updates
            flat_states = [s for per_agent in states for s in per_agent]
            flat_actions = [a for per_agent_actions in actions for a in per_agent_actions]
            flat_advs = [advantages[i] for i in range(len(advantages)) for _ in range(len(self.actors))]
            for actor in self.actors:
                actor.update(flat_states[: len(actions) * len(self.actors)], flat_actions, flat_advs, lr=self.cfg.lr)
            self.critic.update(global_states, returns, lr=self.cfg.lr * self.cfg.value_lr_scale)

            steps_count = max(len(states), 1)
            total_policy_loss += policy_loss / steps_count
            total_value_loss += value_loss / steps_count
            total_entropy += entropy_acc / steps_count

        updates = float(self.cfg.ppo_epochs)
        return {
            "policy_loss": total_policy_loss / updates,
            "value_loss": total_value_loss / updates,
            "entropy": total_entropy / updates,
        }

    # -----------------------------
    # Persistence helpers
    # -----------------------------
    def to_dict(self) -> dict:
        return {
            "config": self.cfg.__dict__,
            "actors": [actor.to_dict() for actor in self.actors],
            "critic": self.critic.to_dict(),
        }

    def save(self, path: str) -> None:
        import json

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str) -> "MAPPOAgent":
        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = MAPPOConfig(**data["config"])
        agent = cls(cfg)
        agent.actors = [PolicyNetwork.from_dict(p) for p in data.get("actors", [])]
        agent.critic = ValueNetwork.from_dict(data["critic"])
        return agent
