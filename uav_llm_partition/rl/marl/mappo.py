"""MAPPO-inspired agent with shared buffer and PPO-style updates."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence

from uav_llm_partition.rl.networks import ContinuousPolicyNetwork, ValueNetwork


@dataclass
class MAPPOConfig:
    num_agents: int
    local_state_dim: int
    global_state_dim: int
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.05
    entropy_decay: float = 0.99
    lr: float = 3e-4
    value_lr_scale: float = 0.25
    ppo_epochs: int = 6
    max_grad_norm: float = 0.5
    bid_std: float = 0.1


class MAPPOAgent:
    """Simplified MAPPO actor-critic collection with per-agent actors."""

    def __init__(self, config: MAPPOConfig) -> None:
        self.cfg = config
        self.actors = [
            ContinuousPolicyNetwork(config.local_state_dim, hidden_dims=(64, 32), std=config.bid_std)
            for _ in range(config.num_agents)
        ]
        self.critic = ValueNetwork(config.global_state_dim, hidden_dims=(64, 32))

    @staticmethod
    def default_config(num_agents: int, local_state_dim: int, global_state_dim: int) -> MAPPOConfig:
        return MAPPOConfig(
            num_agents=num_agents,
            local_state_dim=local_state_dim,
            global_state_dim=global_state_dim,
        )

    def select_bids(
        self,
        local_states: Sequence[Sequence[float]],
        deterministic: bool = False,
    ) -> tuple[List[float], List[float], List[float], List[float]]:
        """Return bids/log-probs/means/stds for each agent."""

        bids: List[float] = []
        log_probs: List[float] = []
        means: List[float] = []
        stds: List[float] = []
        for agent_id, state in enumerate(local_states):
            bid, log_prob, _, mean = self.actors[agent_id].get_action_and_log_prob(
                state, deterministic=deterministic
            )
            bids.append(float(bid))
            log_probs.append(float(log_prob))
            means.append(float(mean))
            stds.append(float(self.actors[agent_id].std))
        return bids, log_probs, means, stds

    def value(self, global_state: Sequence[float]) -> float:
        return self.critic.forward(global_state)

    def _compute_gae(
        self, rewards: List[float], values: List[float], next_values: List[float], dones: List[bool]
    ) -> tuple[List[float], List[float]]:
        advantages: List[float] = []
        returns: List[float] = []
        gae = 0.0
        for t in reversed(range(len(rewards))):
            mask = 0.0 if dones[t] else 1.0
            delta = rewards[t] + self.cfg.gamma * next_values[t] * mask - values[t]
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * gae * mask
            advantages.insert(0, gae)
            returns.insert(0, gae + values[t])
        return advantages, returns

    def learn(
        self,
        states: List[List[List[float]]],
        global_states: List[List[float]],
        bids: List[List[float]],
        old_log_probs: List[List[float]],
        rewards: List[float],
        dones: List[bool],
        values: List[float] | None = None,
        next_values: List[float] | None = None,
        means: List[List[float]] | None = None,
        stds: List[List[float]] | None = None,
        action_masks: List[List[bool]] | None = None,
    ) -> dict:
        if not rewards:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        mean_r = sum(rewards) / max(len(rewards), 1)
        std_r = math.sqrt(sum((r - mean_r) ** 2 for r in rewards) / max(len(rewards), 1))
        if std_r > 1e-6:
            rewards = [(r - mean_r) / (std_r + 1e-6) for r in rewards]

        if values is None:
            values = [self.value(gs) for gs in global_states]
        if next_values is None:
            next_values = [0.0 for _ in values]
        advantages, returns = self._compute_gae(rewards, values, next_values, dones)
        returns = [max(min(ret, 10.0), -10.0) for ret in returns]

        mean_adv = sum(advantages) / max(len(advantages), 1)
        std_adv = math.sqrt(sum((a - mean_adv) ** 2 for a in advantages) / max(len(advantages), 1))
        if std_adv > 1e-8:
            advantages = [(a - mean_adv) / (std_adv + 1e-8) for a in advantages]

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_kl = 0.0

        indices = list(range(len(states)))
        import random

        batch_size = max(8, len(states) // 2) if len(states) > 0 else 1
        for _ in range(self.cfg.ppo_epochs):
            random.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                batch_idx = indices[start : start + batch_size]
                policy_loss = 0.0
                value_loss = 0.0
                entropy_acc = 0.0

                per_agent_states = [[] for _ in range(self.cfg.num_agents)]
                per_agent_bids = [[] for _ in range(self.cfg.num_agents)]
                per_agent_advs = [[] for _ in range(self.cfg.num_agents)]

                for t in batch_idx:
                    adv = advantages[t]
                    ret = returns[t]
                    for agent_id, actor in enumerate(self.actors):
                        if action_masks and not action_masks[t][agent_id]:
                            continue
                        state = states[t][agent_id]
                        bid = bids[t][agent_id]
                        new_mean = actor.forward(state)
                        old_mean = means[t][agent_id] if means else new_mean
                        old_std = stds[t][agent_id] if stds else actor.std
                        old_log_prob = (
                            old_log_probs[t][agent_id]
                            if old_log_probs and old_log_probs[t]
                            else ContinuousPolicyNetwork.log_prob_from_params(old_mean, old_std, bid)
                        )
                        new_log_prob = ContinuousPolicyNetwork.log_prob_from_params(new_mean, actor.std, bid)
                        ratio = math.exp(new_log_prob - old_log_prob)
                        clipped_ratio = max(1.0 - self.cfg.clip_epsilon, min(1.0 + self.cfg.clip_epsilon, ratio))
                        surr1 = ratio * adv
                        surr2 = clipped_ratio * adv
                        surrogate = min(surr1, surr2) if adv >= 0 else max(surr1, surr2)
                        entropy_term = ContinuousPolicyNetwork.entropy_from_std(old_std)
                        policy_loss += -surrogate - self.cfg.entropy_coef * entropy_term
                        entropy_acc += entropy_term
                        total_kl += max(old_log_prob - new_log_prob, 0.0)

                        clipped_adv = clipped_ratio * adv
                        per_agent_states[agent_id].append(state)
                        per_agent_bids[agent_id].append(bid)
                        per_agent_advs[agent_id].append(clipped_adv)

                    v_pred = self.value(global_states[t])
                    value_loss += (v_pred - ret) ** 2

                for agent_id, actor in enumerate(self.actors):
                    actor.update(
                        per_agent_states[agent_id],
                        per_agent_bids[agent_id],
                        [adv for adv in per_agent_advs[agent_id]],
                        lr=self.cfg.lr,
                        max_grad_norm=self.cfg.max_grad_norm,
                    )
                batch_global = [global_states[t] for t in batch_idx]
                batch_returns = [returns[t] for t in batch_idx]
                self.critic.update(batch_global, batch_returns, lr=self.cfg.lr * self.cfg.value_lr_scale)

                steps_count = max(len(batch_idx), 1)
                total_policy_loss += policy_loss / steps_count
                total_value_loss += value_loss / steps_count
                total_entropy += entropy_acc / steps_count

        updates = max((len(indices) / batch_size) * self.cfg.ppo_epochs, 1.0)
        # mild entropy annealing to encourage early exploration and late exploitation
        self.cfg.entropy_coef = max(self.cfg.entropy_coef * self.cfg.entropy_decay, 0.005)
        return {
            "policy_loss": total_policy_loss / updates,
            "value_loss": total_value_loss / updates,
            "entropy": total_entropy / updates,
            "kl": total_kl / updates,
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
        actors = [ContinuousPolicyNetwork.from_dict(p) for p in data.get("actors", [])]
        if len(actors) != cfg.num_agents:
            raise ValueError(
                f"Loaded MARL actor count {len(actors)} does not match num_agents {cfg.num_agents}"
            )
        agent.actors = actors
        agent.critic = ValueNetwork.from_dict(data["critic"])
        return agent
