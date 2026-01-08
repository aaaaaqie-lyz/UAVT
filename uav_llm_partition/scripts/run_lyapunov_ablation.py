"""Lyapunov ablation study runner.

This script sweeps several Lyapunov configurations to compare how queue
penalties affect delay, fairness, and max-load. It mirrors the suggested
design from the user requirements.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Dict, List

from uav_llm_partition.sim.simulator import Simulator


@dataclass
class AblationConfig:
    name: str
    use_lyapunov: bool
    lyapunov_mode: str
    lyapunov_penalty_value: float | None = None
    lyapunov_theta: float | None = None

    def kwargs(self) -> Dict[str, object]:
        payload = asdict(self)
        payload.pop("name", None)
        return {k: v for k, v in payload.items() if v is not None}


def run_ablation_experiment(cfg: AblationConfig) -> Dict[str, float]:
    sim = Simulator(
        num_uav=4,
        num_layers=2,
        num_heads=4,
        hidden_size=1536,
        head_dim=None,
        intervals=20,
        interval_tokens=10,
        scheduler_type="heuristic",
        **cfg.kwargs(),
    )
    metrics = sim.run()
    return metrics.aggregate()


def main() -> None:
    experiments: List[AblationConfig] = [
        AblationConfig("no_lyapunov", use_lyapunov=False, lyapunov_mode="none"),
        AblationConfig("fixed_0.5", use_lyapunov=True, lyapunov_mode="fixed", lyapunov_penalty_value=0.5),
        AblationConfig("fixed_1.0", use_lyapunov=True, lyapunov_mode="fixed", lyapunov_penalty_value=1.0),
        AblationConfig("adaptive", use_lyapunov=True, lyapunov_mode="adaptive"),
        AblationConfig("enhanced", use_lyapunov=True, lyapunov_mode="enhanced", lyapunov_penalty_value=0.5),
        AblationConfig("theta_0.1", use_lyapunov=True, lyapunov_mode="adaptive", lyapunov_theta=0.1),
        AblationConfig("theta_0.3", use_lyapunov=True, lyapunov_mode="adaptive", lyapunov_theta=0.3),
        AblationConfig("theta_0.5", use_lyapunov=True, lyapunov_mode="adaptive", lyapunov_theta=0.5),
        AblationConfig("theta_0.7", use_lyapunov=True, lyapunov_mode="adaptive", lyapunov_theta=0.7),
    ]

    results: Dict[str, Dict[str, float]] = {}
    for exp in experiments:
        print(f"Running experiment: {exp.name}")
        results[exp.name] = run_ablation_experiment(exp)

    with open("lyapunov_ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n=== Ablation Study Results ===")
    for name, result in results.items():
        print(
            f"{name}: delay={result['avg_delay']:.3f}, fairness={result['avg_fairness']:.3f}, "
            f"max_load={result['avg_max_load']:.3f}, failure_rate={result['failure_rate']:.2%}"
        )


if __name__ == "__main__":
    main()
