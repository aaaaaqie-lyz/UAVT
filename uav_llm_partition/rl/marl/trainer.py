"""Trainer for multi-agent RL (MAPPO-style)."""
from __future__ import annotations

from typing import Callable, Iterable, List

from .env import MultiAgentResourceAllocationEnv
from .mappo import MAPPOAgent
from .buffer import MAPPOBuffer


class MARLTrainer:
    def __init__(self, agent: MAPPOAgent) -> None:
        self.agent = agent
        self.buffer = MAPPOBuffer()

    def collect_experience(
        self,
        env: MultiAgentResourceAllocationEnv,
        max_steps: int | None = None,
    ) -> float:
        """Roll out one episode, populate buffer, and return cumulative reward."""

        local_states, global_state = env.reset()
        if local_states and (
            len(local_states[0]) != self.agent.cfg.local_state_dim
            or len(global_state) != self.agent.cfg.global_state_dim
            or len(local_states) != self.agent.cfg.num_agents
        ):
            raise ValueError(
                "MARL trainer state dimensions mismatch: agent cfg="
                f"(agents={self.agent.cfg.num_agents}, local={self.agent.cfg.local_state_dim}, global={self.agent.cfg.global_state_dim})"
                f" env=(agents={len(local_states)}, local={len(local_states[0]) if local_states else 0}, global={len(global_state)})"
            )

        total_reward = 0.0
        steps = 0
        while True:
            bids, log_probs, _ = self.agent.select_bids(local_states)
            step = env.step(bids)
            total_reward += step.team_reward

            self.buffer.add(
                local_states=local_states,
                global_state=global_state,
                log_probs=log_probs,
                bids=bids,
                reward=step.team_reward,
                rewards_by_agent=step.rewards,
                done=step.done,
            )

            if step.done:
                break
            local_states = step.next_local_states or []
            global_state = step.next_global_state or []
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break
        return total_reward

    def train(
        self,
        env_factory: Iterable[MultiAgentResourceAllocationEnv] | Callable[[], MultiAgentResourceAllocationEnv],
        episodes: int = 500,
        update_every: int = 4,
    ) -> None:
        rollout_counter = 0
        for ep in range(episodes):
            if callable(env_factory):
                envs = [env_factory()]
            else:
                envs = list(env_factory)
            cumulative = 0.0
            policy_losses: List[float] = []
            value_losses: List[float] = []
            entropies: List[float] = []
            for env in envs:
                cumulative += self.collect_experience(env)
                rollout_counter += 1
                if rollout_counter % update_every == 0:
                    batch = self.buffer.as_batch()
                    stats = self.agent.learn(
                        batch["local_states"],
                        batch["global_states"],
                        batch["bids"],
                        batch["log_probs"],
                        batch["rewards"],
                        batch["dones"],
                    )
                    policy_losses.append(stats["policy_loss"])
                    value_losses.append(stats["value_loss"])
                    entropies.append(stats["entropy"])
                    self.buffer.clear()
            avg_pl = sum(policy_losses) / max(len(policy_losses), 1)
            avg_vl = sum(value_losses) / max(len(value_losses), 1)
            avg_ent = sum(entropies) / max(len(entropies), 1)
            print(
                f"Episode {ep + 1}/{episodes}: total_reward={cumulative:.3f} "
                f"policy_loss={avg_pl:.4f} value_loss={avg_vl:.4f} entropy={avg_ent:.4f}"
            )
        # Final flush if buffer still has rollouts
        if self.buffer.transitions:
            batch = self.buffer.as_batch()
            self.agent.learn(
                batch["local_states"],
                batch["global_states"],
                batch["bids"],
                batch["log_probs"],
                batch["rewards"],
                batch["dones"],
            )
            self.buffer.clear()
