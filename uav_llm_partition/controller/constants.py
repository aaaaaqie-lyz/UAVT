"""Shared scheduling constants to keep heuristic and MARL scores aligned."""

comm_penalty_scale: float = 0.05
migration_penalty_scale: float = 0.05
mig_overhead: float = 0.01

type_penalty = {"uav": 0.0, "edge": 0.2, "cloud": 0.35}

lyap_to_weight_factor: float = 0.1
lyap_weight_clip: float = 3.0

stability_margin: float = 0.005
stability_bonus: float = 0.01
