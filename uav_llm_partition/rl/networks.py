"""Pure-Python lightweight policy/value helpers for RL scheduling."""
from __future__ import annotations

from typing import Iterable, Sequence

import math
import random

import torch
from torch import nn


def _relu(vec: Iterable[float]) -> list[float]:
    return [max(0.0, v) for v in vec]


def _dot(vec: Sequence[float], weights: Sequence[Sequence[float]], bias: Sequence[float]) -> list[float]:
    out: list[float] = []
    for j in range(len(weights[0])):
        s = bias[j]
        for i, v in enumerate(vec):
            s += v * weights[i][j]
        out.append(s)
    return out


def _softmax(vec: Sequence[float]) -> list[float]:
    shifted = [v - max(vec) for v in vec]
    exps = [math.exp(v) for v in shifted]
    total = sum(exps) + 1e-8
    return [e / total for e in exps]


class PolicyNetwork:
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: Sequence[int] = (64, 32)) -> None:
        dims = [state_dim, *hidden_dims, action_dim]
        self.weights = [
            [[random.uniform(-0.01, 0.01) for _ in range(dims[i + 1])] for _ in range(dims[i])]
            for i in range(len(dims) - 1)
        ]
        self.biases = [[0.0 for _ in range(dims[i + 1])] for i in range(len(dims) - 1)]

    def forward(self, state: Sequence[float]) -> list[float]:
        x = list(state)
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            x = _relu(_dot(x, W, b))
        logits = _dot(x, self.weights[-1], self.biases[-1])
        return _softmax(logits)

    # -----------------------------
    # Persistence helpers
    # -----------------------------
    def to_dict(self) -> dict:
        return {"weights": self.weights, "biases": self.biases}

    @classmethod
    def from_dict(cls, payload: dict) -> "PolicyNetwork":
        obj: "PolicyNetwork" = cls.__new__(cls)  # type: ignore[call-arg]
        obj.weights = payload["weights"]
        obj.biases = payload["biases"]
        return obj

    def save(self, path: str) -> None:
        import json

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str) -> "PolicyNetwork":
        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def get_log_prob(self, probs: Sequence[float], action: int) -> float:
        return math.log(max(probs[action], 1e-8))

    def get_action_and_log_prob(
        self, state: Sequence[float], mask: Sequence[bool] | None = None, deterministic: bool = False
    ) -> tuple[int, float, float]:
        probs = self.forward(state)
        if mask is not None:
            probs = [p if m else 0.0 for p, m in zip(probs, mask)]
            total = sum(probs)
            if total <= 0:
                probs = [1.0 / len(probs) for _ in probs]
            else:
                probs = [p / total for p in probs]
        entropy = -sum(p * math.log(max(p, 1e-8)) for p in probs)
        import random

        if deterministic:
            action = int(max(range(len(probs)), key=lambda i: probs[i]))
        else:
            r = random.random()
            cdf = 0.0
            action = len(probs) - 1
            for i, p in enumerate(probs):
                cdf += p
                if r <= cdf:
                    action = i
                    break
        return action, self.get_log_prob(probs, action), entropy

    def update(self, states, actions, advantages, lr: float = 1e-3, max_grad_norm: float = 0.5) -> None:
        """Backpropagate a simple policy-gradient style update through all layers."""

        for state, action, adv in zip(states, actions, advantages):
            if abs(adv) < 1e-9:
                continue
            # forward pass with cached activations
            activations = [list(state)]
            zs = []
            for W, b in zip(self.weights[:-1], self.biases[:-1]):
                z = _dot(activations[-1], W, b)
                zs.append(z)
                activations.append(_relu(z))
            logits = _dot(activations[-1], self.weights[-1], self.biases[-1])
            probs = _softmax(logits)

            # gradient of log-prob wrt logits
            grad_logits = [p for p in probs]
            grad_logits[action] -= 1.0
            grad_logits = [g * adv for g in grad_logits]

            # backprop output layer
            grad_w_last = [[a * g for g in grad_logits] for a in activations[-1]]
            grad_b_last = list(grad_logits)

            # backprop hidden layers (ReLU)
            grad_prev = [sum(self.weights[-1][i][j] * grad_logits[j] for j in range(len(grad_logits))) for i in range(len(self.weights[-1]))]
            for layer in reversed(range(len(self.weights) - 1)):
                z = zs[layer]
                relu_mask = [1.0 if v > 0 else 0.0 for v in z]
                grad_prev = [g * m for g, m in zip(grad_prev, relu_mask)]
                grad_w = [[activations[layer][i] * grad_prev[j] for j in range(len(grad_prev))] for i in range(len(activations[layer]))]
                grad_b = list(grad_prev)
                # apply gradients
                for i in range(len(self.weights[layer])):
                    for j in range(len(self.weights[layer][i])):
                        self.weights[layer][i][j] -= lr * grad_w[i][j]
                for j in range(len(self.biases[layer])):
                    self.biases[layer][j] -= lr * grad_b[j]
                if layer > 0:
                    grad_prev = [sum(self.weights[layer][i][k] * grad_prev[k] for k in range(len(grad_prev))) for i in range(len(self.weights[layer]))]

            # apply output gradients
            for i in range(len(self.weights[-1])):
                for j in range(len(self.weights[-1][i])):
                    self.weights[-1][i][j] -= lr * grad_w_last[i][j]
            for j in range(len(self.biases[-1])):
                self.biases[-1][j] -= lr * grad_b_last[j]


class ValueNetwork(nn.Module):
    def __init__(self, state_dim: int, hidden_dims: Sequence[int] = (64, 32), lr: float = 1e-3) -> None:
        super().__init__()
        dims = [state_dim, *hidden_dims, 1]
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def _forward_tensor(self, state: torch.Tensor) -> torch.Tensor:
        x = state
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        return self.layers[-1](x).squeeze(-1)

    def forward(self, state: Sequence[float] | torch.Tensor) -> float | torch.Tensor:
        if isinstance(state, torch.Tensor):
            return self._forward_tensor(state)
        state_tensor = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            return float(self._forward_tensor(state_tensor).item())

    # -----------------------------
    # Persistence helpers
    # -----------------------------
    def to_dict(self) -> dict:
        weights = [layer.weight.detach().cpu().tolist() for layer in self.layers]
        biases = [layer.bias.detach().cpu().tolist() for layer in self.layers]
        return {"weights": weights, "biases": biases}

    @classmethod
    def from_dict(cls, payload: dict) -> "ValueNetwork":
        weights = payload["weights"]
        biases = payload["biases"]
        dims = [len(weights[0][0])] + [len(w) for w in weights]
        hidden_dims = dims[1:-1]
        obj = cls(dims[0], hidden_dims=hidden_dims)
        for layer, weight, bias in zip(obj.layers, weights, biases):
            layer.weight.data = torch.tensor(weight, dtype=torch.float32)
            layer.bias.data = torch.tensor(bias, dtype=torch.float32)
        return obj

    def save(self, path: str) -> None:
        import json

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str) -> "ValueNetwork":
        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def update(self, states, targets, lr: float = 1e-3, max_grad_norm: float = 0.5) -> None:
        if not states:
            return
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        state_tensor = torch.tensor(states, dtype=torch.float32)
        target_tensor = torch.tensor(targets, dtype=torch.float32)
        preds = self._forward_tensor(state_tensor)
        loss = torch.mean((preds - target_tensor) ** 2)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
        self.optimizer.step()


class ContinuousPolicyNetwork(nn.Module):
    """Single-output policy with sigmoid squashing for continuous bids in [0, 1]."""

    def __init__(self, state_dim: int, hidden_dims: Sequence[int] = (64, 32), std: float = 0.1, lr: float = 1e-3) -> None:
        super().__init__()
        dims = [state_dim, *hidden_dims, 1]
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self.log_std = nn.Parameter(torch.log(torch.tensor(max(std, 0.05), dtype=torch.float32)))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def _std_tensor(self) -> torch.Tensor:
        return torch.exp(self.log_std)

    def _eff_std_tensor(self) -> torch.Tensor:
        return torch.clamp(self._std_tensor(), min=0.05)

    @property
    def std(self) -> float:
        return float(self._std_tensor().item())

    # -----------------------------
    # Forward / sampling
    # -----------------------------
    def _forward_tensor(self, state: torch.Tensor) -> torch.Tensor:
        x = state
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        logits = self.layers[-1](x)
        return torch.sigmoid(logits).squeeze(-1)

    def forward(self, state: Sequence[float] | torch.Tensor) -> float | torch.Tensor:
        if isinstance(state, torch.Tensor):
            return self._forward_tensor(state)
        state_tensor = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            return float(self._forward_tensor(state_tensor).item())

    @staticmethod
    def log_prob_from_params(mean: float, std: float, bid: float) -> float:
        """Compute Gaussian log-prob using explicit mean/std for ratio math."""

        bid_clamped = max(min(bid, 1.0 - 1e-6), 1e-6)
        var = max(std, 0.05) ** 2
        return -0.5 * ((bid_clamped - mean) ** 2) / (var + 1e-8) - 0.5 * math.log(2 * math.pi * (var + 1e-8))

    @staticmethod
    def entropy_from_std(std: float) -> float:
        var = std * std
        return 0.5 * (1.0 + math.log(2 * math.pi * var))

    def get_action_and_log_prob(
        self, state: Sequence[float], deterministic: bool = False
    ) -> tuple[float, float, float, float]:
        state_tensor = torch.tensor(state, dtype=torch.float32)
        with torch.no_grad():
            mean = self._forward_tensor(state_tensor)
            std_tensor = self._eff_std_tensor()
            dist = torch.distributions.Normal(mean, std_tensor)
            if deterministic:
                bid = mean
            else:
                bid = dist.sample()
            bid = torch.clamp(bid, 0.0, 1.0)
            log_prob = dist.log_prob(bid)
            entropy = dist.entropy()
        return float(bid.item()), float(log_prob.item()), float(entropy.item()), float(mean.item())

    def update(self, states, bids, advantages, lr: float = 1e-3, max_grad_norm: float = 0.5) -> None:
        """Policy gradient update using Gaussian log-prob with sigmoid mean."""

        if not states:
            return
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        state_tensor = torch.tensor(states, dtype=torch.float32)
        bid_tensor = torch.tensor(bids, dtype=torch.float32)
        adv_tensor = torch.tensor(advantages, dtype=torch.float32)
        mean = self._forward_tensor(state_tensor)
        std_tensor = self._eff_std_tensor()
        dist = torch.distributions.Normal(mean, std_tensor)
        log_prob = dist.log_prob(bid_tensor)
        loss = -(log_prob * adv_tensor).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
        self.optimizer.step()

    def update_ppo(
        self,
        states,
        bids,
        old_log_probs,
        advantages,
        clip_eps: float,
        entropy_coef: float,
        lr: float = 1e-3,
        max_grad_norm: float = 0.5,
    ) -> tuple[float, float]:
        """PPO-style clipped update for the Gaussian bid policy."""

        if not states:
            return 0.0, 0.0
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        state_tensor = torch.tensor(states, dtype=torch.float32)
        bid_tensor = torch.tensor(bids, dtype=torch.float32)
        old_log_prob_tensor = torch.tensor(old_log_probs, dtype=torch.float32)
        adv_tensor = torch.tensor(advantages, dtype=torch.float32)

        mean = self._forward_tensor(state_tensor)
        std_tensor = self._eff_std_tensor()
        dist = torch.distributions.Normal(mean, std_tensor)
        log_prob = dist.log_prob(bid_tensor)
        ratio = torch.exp(log_prob - old_log_prob_tensor)
        surr1 = ratio * adv_tensor
        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_tensor
        policy_loss = -torch.min(surr1, surr2).mean()
        entropy = dist.entropy().mean()
        loss = policy_loss - entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
        self.optimizer.step()
        return float(policy_loss.item()), float(entropy.item())

    # -----------------------------
    # Persistence helpers
    # -----------------------------
    def to_dict(self) -> dict:
        weights = [layer.weight.detach().cpu().tolist() for layer in self.layers]
        biases = [layer.bias.detach().cpu().tolist() for layer in self.layers]
        return {"weights": weights, "biases": biases, "log_std": float(self.log_std.detach().cpu().item())}

    @classmethod
    def from_dict(cls, payload: dict) -> "ContinuousPolicyNetwork":
        weights = payload["weights"]
        biases = payload["biases"]
        dims = [len(weights[0][0])] + [len(w) for w in weights]
        hidden_dims = dims[1:-1]
        obj = cls(dims[0], hidden_dims=hidden_dims)
        for layer, weight, bias in zip(obj.layers, weights, biases):
            layer.weight.data = torch.tensor(weight, dtype=torch.float32)
            layer.bias.data = torch.tensor(bias, dtype=torch.float32)
        log_std = payload.get("log_std", math.log(0.1))
        obj.log_std.data = torch.tensor(log_std, dtype=torch.float32)
        return obj

    def save(self, path: str) -> None:
        import json

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str) -> "ContinuousPolicyNetwork":
        import json

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
