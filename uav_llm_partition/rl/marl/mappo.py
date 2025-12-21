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
        deterministic: bool = True,
    ) -> tuple[List[float], List[float], float]:
        """Return bids in [0,1], their log-probabilities, and average entropy."""

        bids: List[float] = []
        log_probs: List[float] = []
        entropies: List[float] = []
        for agent_id, state in enumerate(local_states):
            bid, log_prob, entropy = self.actors[agent_id].get_action_and_log_prob(
                state, deterministic=deterministic
            )
            bids.append(float(bid))
            log_probs.append(float(log_prob))
            entropies.append(float(entropy))
        avg_entropy = sum(entropies) / max(len(entropies), 1)
        return bids, log_probs, avg_entropy

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
        states: List[List[List[float]]],
        global_states: List[List[float]],
        bids: List[List[float]],
        old_log_probs: List[List[float]],
        rewards: List[float],
        dones: List[bool],
    ) -> dict:
        if not rewards:
            return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        mean_r = sum(rewards) / max(len(rewards), 1)
        std_r = math.sqrt(sum((r - mean_r) ** 2 for r in rewards) / max(len(rewards), 1))
        if std_r > 1e-6:
            rewards = [(r - mean_r) / (std_r + 1e-6) for r in rewards]

        values = [self.value(gs) for gs in global_states]
        advantages, returns = self._compute_gae(rewards, values, dones)
        returns = [max(min(ret, 10.0), -10.0) for ret in returns]

        mean_adv = sum(advantages) / max(len(advantages), 1)
        std_adv = math.sqrt(sum((a - mean_adv) ** 2 for a in advantages) / max(len(advantages), 1))
        if std_adv > 1e-8:
            advantages = [(a - mean_adv) / (std_adv + 1e-8) for a in advantages]

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        per_agent_states: List[List[List[float]]] = [list() for _ in range(self.cfg.num_agents)]
        per_agent_bids: List[List[float]] = [list() for _ in range(self.cfg.num_agents)]
        per_agent_advs: List[List[float]] = [list() for _ in range(self.cfg.num_agents)]
        for step_idx in range(len(states)):
            for agent_id in range(self.cfg.num_agents):
                per_agent_states[agent_id].append(states[step_idx][agent_id])
                per_agent_bids[agent_id].append(bids[step_idx][agent_id])
                per_agent_advs[agent_id].append(advantages[step_idx])

        for _ in range(self.cfg.ppo_epochs):
            policy_loss = 0.0
            value_loss = 0.0
            entropy_acc = 0.0
            for t in range(len(states)):
                adv = advantages[t]
                ret = returns[t]
                for agent_id, actor in enumerate(self.actors):
                    state = states[t][agent_id]
                    bid = bids[t][agent_id]
                    new_mean = actor.forward(state)
                    # reuse gaussian log-prob formulation
                    bid_clamped = max(min(bid, 1.0 - 1e-6), 1e-6)
                    var = actor.std * actor.std
                    new_log_prob = -0.5 * ((bid_clamped - new_mean) ** 2) / var - math.log(
                        actor.std * math.sqrt(2 * math.pi)
                    )
                    ratio = math.exp(new_log_prob - old_log_probs[t][agent_id])
                    clipped_ratio = max(1.0 - self.cfg.clip_epsilon, min(1.0 + self.cfg.clip_epsilon, ratio))
                    surr1 = ratio * adv
                    surr2 = clipped_ratio * adv
                    entropy_term = actor._entropy()
                    policy_loss += -min(surr1, surr2) - self.cfg.entropy_coef * entropy_term
                    entropy_acc += entropy_term
                v_pred = self.value(global_states[t])
                value_loss += (v_pred - ret) ** 2

            for agent_id, actor in enumerate(self.actors):
                actor.update(per_agent_states[agent_id], per_agent_bids[agent_id], per_agent_advs[agent_id], lr=self.cfg.lr)
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
        actors = [ContinuousPolicyNetwork.from_dict(p) for p in data.get("actors", [])]
        if len(actors) != cfg.num_agents:
            raise ValueError(
                f"Loaded MARL actor count {len(actors)} does not match num_agents {cfg.num_agents}"
            )
        agent.actors = actors
        agent.critic = ValueNetwork.from_dict(data["critic"])
        return agent
