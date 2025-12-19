"""Pure-Python lightweight policy/value helpers for RL scheduling."""
from __future__ import annotations

from typing import Iterable, Sequence

import math
import random


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
        # Lightweight update: adjust final-layer weights toward actions with positive advantage.
        for state, action, adv in zip(states, actions, advantages):
            if abs(adv) < 1e-9:
                continue
            x = list(state)
            for W, b in zip(self.weights[:-1], self.biases[:-1]):
                x = _relu(_dot(x, W, b))
            grad_scale = -adv * lr
            if abs(grad_scale) > max_grad_norm:
                grad_scale = max_grad_norm if grad_scale > 0 else -max_grad_norm
            for i, v in enumerate(x):
                self.weights[-1][i][action] -= grad_scale * v
            self.biases[-1][action] -= grad_scale


class ValueNetwork:
    def __init__(self, state_dim: int, hidden_dims: Sequence[int] = (64, 32)) -> None:
        dims = [state_dim, *hidden_dims, 1]
        self.weights = [
            [[random.uniform(-0.01, 0.01) for _ in range(dims[i + 1])] for _ in range(dims[i])]
            for i in range(len(dims) - 1)
        ]
        self.biases = [[0.0 for _ in range(dims[i + 1])] for i in range(len(dims) - 1)]

    def forward(self, state: Sequence[float]) -> float:
        x = list(state)
        for W, b in zip(self.weights[:-1], self.biases[:-1]):
            x = _relu(_dot(x, W, b))
        out = _dot(x, self.weights[-1], self.biases[-1])[0]
        return out

    # -----------------------------
    # Persistence helpers
    # -----------------------------
    def to_dict(self) -> dict:
        return {"weights": self.weights, "biases": self.biases}

    @classmethod
    def from_dict(cls, payload: dict) -> "ValueNetwork":
        obj: "ValueNetwork" = cls.__new__(cls)  # type: ignore[call-arg]
        obj.weights = payload["weights"]
        obj.biases = payload["biases"]
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
        for state, target in zip(states, targets):
            pred = self.forward(state)
            error = pred - target
            # Update final layer only
            x = list(state)
            for W, b in zip(self.weights[:-1], self.biases[:-1]):
                x = _relu(_dot(x, W, b))
            grad_scale = lr * error
            if abs(grad_scale) > max_grad_norm:
                grad_scale = max_grad_norm if grad_scale > 0 else -max_grad_norm
            for i, v in enumerate(x):
                self.weights[-1][i][0] -= grad_scale * v
            self.biases[-1][0] -= grad_scale
