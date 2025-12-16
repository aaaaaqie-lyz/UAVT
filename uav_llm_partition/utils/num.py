"""Lightweight numeric helpers to avoid external dependencies."""
from __future__ import annotations

import math
import random
from typing import Iterable, List


def zeros(length: int) -> List[float]:
    return [0.0 for _ in range(length)]


def rand_uniform(low: float, high: float, size: int) -> List[float]:
    return [random.uniform(low, high) for _ in range(size)]


def rand_normal(mean: float, std: float, size: int) -> List[float]:
    return [random.gauss(mean, std) for _ in range(size)]


def clip(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def vector_norm(vec: Iterable[float]) -> float:
    return math.sqrt(sum(v * v for v in vec))


def exp(value: float) -> float:
    return math.exp(value)


def mean(values: Iterable[float]) -> float:
    values_list = list(values)
    return sum(values_list) / len(values_list) if values_list else 0.0


def normalize(values: List[float]) -> List[float]:
    max_val = max(values) if values else 1.0
    denom = max_val + 1e-6
    return [v / denom for v in values]


def argmin(values: List[float]) -> int:
    return min(range(len(values)), key=lambda i: values[i])

