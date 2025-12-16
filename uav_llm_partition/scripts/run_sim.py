"""Entry point to run the UAV LLM partition simulator."""
from __future__ import annotations

from uav_llm_partition.sim.simulator import Simulator


def main() -> None:
    sim = Simulator(num_uav=4, num_layers=2, num_heads=4, hidden_size=1024, head_dim=64, intervals=20)
    metrics = sim.run()
    summary = metrics.aggregate()
    print("Simulation summary:", summary)


if __name__ == "__main__":
    main()

