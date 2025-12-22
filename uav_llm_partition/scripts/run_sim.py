"""Entry point to run the UAV LLM partition simulator."""
from __future__ import annotations

import argparse

from uav_llm_partition.sim.simulator import Simulator


def main() -> None:
    parser = argparse.ArgumentParser(description="Run UAV partition simulator")
    parser.add_argument("--scheduler", type=str, default="heuristic", help="scheduler type (heuristic|marl|rl|...)")
    parser.add_argument("--marl-model", type=str, default=None, help="path to a trained MARL model to load")
    parser.add_argument(
        "--topology",
        type=str,
        default=None,
        help="Comma-separated device types (e.g. uav,uav,edge,cloud). Defaults to all UAVs.",
    )
    args = parser.parse_args()

    device_types = None
    if args.topology:
        device_types = [t.strip().lower() for t in args.topology.split(",") if t.strip()]
        num_uav = len(device_types)
    else:
        num_uav = 4
        device_types = ["uav" for _ in range(num_uav)]

    sim = Simulator(
        num_uav=num_uav,
        num_layers=2,
        num_heads=4,
        hidden_size=1536,
        head_dim=None,
        intervals=20,
        interval_tokens=10,
        initial_seq_len=128,
        scheduler_type=args.scheduler,
        scheduler_model_path=args.marl_model,
        device_types=device_types,
    )
    metrics = sim.run()
    summary = metrics.aggregate()
    lines = [
        "Simulation summary: {",
        "'avg_max_load': {}, 'avg_fairness': {},".format(
            summary.get("avg_max_load"), summary.get("avg_fairness")
        ),
        "'avg_delay': {}, 'total_migrations': {},".format(
            summary.get("avg_delay"), summary.get("total_migrations")
        ),
        "'avg_migration_volume': {}, 'failure_rate': {},".format(
            summary.get("avg_migration_volume"), summary.get("failure_rate")
        ),
        f"'failure_reasons': {summary.get('failure_reasons')},",
        " 'avg_mem_load': {}, 'avg_comp_load': {},".format(
            summary.get("avg_mem_load"), summary.get("avg_comp_load")
        ),
        "'avg_queue': {}, 'max_queue': {},".format(
            summary.get("avg_queue"), summary.get("max_queue")
        ),
        "'avg_rho_w': {}, 'avg_rho_q': {},".format(
            summary.get("avg_rho_w"), summary.get("avg_rho_q")
        ),
        "'avg_comp_delay_per_dev': {}, 'avg_comm_delay_per_dev': {},".format(
            summary.get("avg_comp_delay_per_dev"), summary.get("avg_comm_delay_per_dev")
        ),
        "'avg_mig_delay_per_dev': {}, 'avg_total_delay_per_dev': {}".format(
            summary.get("avg_mig_delay_per_dev"), summary.get("avg_total_delay_per_dev")
        ),
        "}",
    ]
    print("\n".join(lines))


if __name__ == "__main__":
    main()

