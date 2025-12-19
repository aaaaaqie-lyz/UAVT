"""Entry point to run the UAV LLM partition simulator."""
from __future__ import annotations

from uav_llm_partition.sim.simulator import Simulator


def main() -> None:
    sim = Simulator(
        num_uav=4,
        num_layers=2,
        num_heads=4,
        hidden_size=1536,
        head_dim=None,
        intervals=20,
        interval_tokens=10,
        initial_seq_len=128,
    )
    metrics = sim.run()
    summary = metrics.aggregate()
    print("Simulation summary: {")
    print(
        "'avg_max_load': {}, 'avg_fairness': {},".format(
            summary.get("avg_max_load"), summary.get("avg_fairness")
        )
    )
    print(
        "'avg_delay': {}, 'total_migrations': {},".format(
            summary.get("avg_delay"), summary.get("total_migrations")
        )
    )
    print(
        "'avg_migration_volume': {}, 'failure_rate': {},".format(
            summary.get("avg_migration_volume"), summary.get("failure_rate")
        )
    )
    print(f"'failure_reasons': {summary.get('failure_reasons')},")
    print(
        " 'avg_mem_load': {}, 'avg_comp_load': {},".format(
            summary.get("avg_mem_load"), summary.get("avg_comp_load")
        )
    )
    print(
        "'avg_queue': {}, 'max_queue': {},".format(
            summary.get("avg_queue"), summary.get("max_queue")
        )
    )
    print(
        "'avg_rho_w': {}, 'avg_rho_q': {},".format(
            summary.get("avg_rho_w"), summary.get("avg_rho_q")
        )
    )
    print(
        "'avg_comp_delay_per_dev': {}, 'avg_comm_delay_per_dev': {},".format(
            summary.get("avg_comp_delay_per_dev"), summary.get("avg_comm_delay_per_dev")
        )
    )
    print(
        "'avg_mig_delay_per_dev': {}, 'avg_total_delay_per_dev': {}".format(
            summary.get("avg_mig_delay_per_dev"), summary.get("avg_total_delay_per_dev")
        )
    )
    print("}")


if __name__ == "__main__":
    main()

