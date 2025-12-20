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
        """Backprop a mean-squared loss update through all layers."""

        for state, target in zip(states, targets):
            # forward with caches
            activations = [list(state)]
            zs = []
            for W, b in zip(self.weights[:-1], self.biases[:-1]):
                z = _dot(activations[-1], W, b)
                zs.append(z)
                activations.append(_relu(z))
            pred = _dot(activations[-1], self.weights[-1], self.biases[-1])[0]
            error = pred - target

            # gradient clipping on scalar error
            error = max(min(error, max_grad_norm), -max_grad_norm)

            # backprop output layer
            grad_out = [error]
            grad_w_last = [[activations[-1][i] * grad_out[0]] for i in range(len(activations[-1]))]
            grad_b_last = list(grad_out)

            grad_prev = [self.weights[-1][i][0] * grad_out[0] for i in range(len(self.weights[-1]))]
            for layer in reversed(range(len(self.weights) - 1)):
                z = zs[layer]
                relu_mask = [1.0 if v > 0 else 0.0 for v in z]
                grad_prev = [g * m for g, m in zip(grad_prev, relu_mask)]
                grad_w = [[activations[layer][i] * grad_prev[j] for j in range(len(grad_prev))] for i in range(len(activations[layer]))]
                grad_b = list(grad_prev)
                for i in range(len(self.weights[layer])):
                    for j in range(len(self.weights[layer][i])):
                        self.weights[layer][i][j] -= lr * grad_w[i][j]
                for j in range(len(self.biases[layer])):
                    self.biases[layer][j] -= lr * grad_b[j]
                if layer > 0:
                    grad_prev = [sum(self.weights[layer][i][k] * grad_prev[k] for k in range(len(grad_prev))) for i in range(len(self.weights[layer]))]

            for i in range(len(self.weights[-1])):
                self.weights[-1][i][0] -= lr * grad_w_last[i][0]
            self.biases[-1][0] -= lr * grad_b_last[0]
