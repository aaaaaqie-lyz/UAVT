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

    def update(self, states, actions, advantages, lr: float = 1e-3) -> None:
        # Lightweight update: adjust final-layer weights toward actions with positive advantage.
        for state, action, adv in zip(states, actions, advantages):
            if adv == 0:
                continue
            x = list(state)
            for W, b in zip(self.weights[:-1], self.biases[:-1]):
                x = _relu(_dot(x, W, b))
            for i, v in enumerate(x):
                self.weights[-1][i][action] -= lr * (-adv) * v
            self.biases[-1][action] -= lr * (-adv)


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

    def update(self, states, targets, lr: float = 1e-3) -> None:
        for state, target in zip(states, targets):
            pred = self.forward(state)
            error = pred - target
            # Update final layer only
            x = list(state)
            for W, b in zip(self.weights[:-1], self.biases[:-1]):
                x = _relu(_dot(x, W, b))
            for i, v in enumerate(x):
                self.weights[-1][i][0] -= lr * error * v
            self.biases[-1][0] -= lr * error
