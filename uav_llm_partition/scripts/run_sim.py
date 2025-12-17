"""Entry point to run the UAV LLM partition simulator."""
from __future__ import annotations

from uav_llm_partition.sim.simulator import Simulator


def format_summary(summary: dict) -> str:
    """Format simulation summary for readable output."""
    lines = ["Simulation summary:"]
    
    # Performance metrics
    lines.append("  Performance:")
    lines.append(f"    avg_max_load: {summary.get('avg_max_load', 0):.4f}")
    lines.append(f"    avg_fairness: {summary.get('avg_fairness', 0):.4f}")
    lines.append(f"    avg_delay: {summary.get('avg_delay', 0):.4f}")
    lines.append(f"    failure_rate: {summary.get('failure_rate', 0):.4f}")
    
    # Migration metrics
    lines.append("  Migration:")
    lines.append(f"    total_migrations: {summary.get('total_migrations', 0)}")
    lines.append(f"    avg_migration_volume: {summary.get('avg_migration_volume', 0):.6f}")
    
    # Failure reasons
    failure_reasons = summary.get('failure_reasons', {})
    if failure_reasons:
        lines.append("  Failure Reasons:")
        for reason, count in failure_reasons.items():
            lines.append(f"    {reason}: {count}")
    
    # Resource utilization
    lines.append("  Resource Utilization:")
    lines.append(f"    avg_mem_load: {summary.get('avg_mem_load', 0):.4f}")
    lines.append(f"    avg_comp_load: {summary.get('avg_comp_load', 0):.4f}")
    
    # Queue metrics
    lines.append("  Queue:")
    lines.append(f"    avg_queue: {summary.get('avg_queue', 0):.4f}")
    lines.append(f"    max_queue: {summary.get('max_queue', 0):.4f}")
    
    # RL parameters
    lines.append("  RL Parameters:")
    lines.append(f"    avg_rho_w: {summary.get('avg_rho_w', 0):.4f}")
    lines.append(f"    avg_rho_q: {summary.get('avg_rho_q', 0):.4f}")
    
    # Delay breakdown
    lines.append("  Delay Breakdown (per device):")
    lines.append(f"    avg_comp_delay: {summary.get('avg_comp_delay_per_dev', 0):.4f}")
    lines.append(f"    avg_comm_delay: {summary.get('avg_comm_delay_per_dev', 0):.4f}")
    lines.append(f"    avg_mig_delay: {summary.get('avg_mig_delay_per_dev', 0):.4f}")
    lines.append(f"    avg_total_delay: {summary.get('avg_total_delay_per_dev', 0):.4f}")
    
    return "\n".join(lines)


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
    print(format_summary(summary))


if __name__ == "__main__":
    main()

