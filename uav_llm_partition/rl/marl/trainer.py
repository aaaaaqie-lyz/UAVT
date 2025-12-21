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
        self.buffer.clear()
        while True:
            actions, log_probs, _ = self.agent.select_actions(local_states)
            bids = self.agent.select_bids(local_states)
            step = env.step(bids)
            total_reward += sum(step.rewards)

            self.buffer.add(
                local_states=local_states,
                global_state=global_state,
                actions=actions,
                log_probs=log_probs,
                rewards=step.rewards,
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
    ) -> None:
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
                # Reduce rewards to a scalar by averaging across agents per step
                rewards = [sum(t.rewards) / len(t.rewards) for t in self.buffer.transitions]
                dones = [t.done for t in self.buffer.transitions]
                local_states = [t.local_states for t in self.buffer.transitions]
                global_states = [t.global_state for t in self.buffer.transitions]
                actions = [t.actions for t in self.buffer.transitions]
                log_probs = [t.log_probs for t in self.buffer.transitions]
                stats = self.agent.learn(local_states, global_states, actions, log_probs, rewards, dones)
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
