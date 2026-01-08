"""Lightweight entrypoint to train the single-agent RL scheduler prototype."""
from __future__ import annotations

import argparse

from uav_llm_partition.rl.agent import AgentConfig, RLAgent
from uav_llm_partition.rl.env import RLResourceAllocationEnv
from uav_llm_partition.rl.trainer import evaluate, train_agent
from uav_llm_partition.sim.simulator import Simulator


def _env_factory() -> RLResourceAllocationEnv:
    """Create a fresh per-interval RL environment from a simulator snapshot."""

    sim = Simulator(
        num_uav=4,
        num_layers=2,
        num_heads=4,
        hidden_size=1536,
        head_dim=None,
        intervals=1,
        interval_tokens=10,
        initial_seq_len=128,
    )

    # Single-interval snapshot mirroring the first scheduling step
    positions, mobility_risk = sim.mobility.update()
    bandwidth, conn, los_score, latency = sim.channel.compute(
        positions, getattr(sim, "device_types", ["uav"] * sim.num_uav)
    )
    compute, memory = sim.resource.sample(getattr(sim, "device_types", ["uav"] * sim.num_uav))
    demands = sim.demand_model.update_interval()
    activation_sizes = {(u, v): sim.demand_model.activation_size(u, v) for u, v in sim.dependencies}
    weights = sim.weights.compute(compute, memory, los_score, mobility_risk)
    lyapunov = sim.lyapunov.pressure() if sim.use_lyapunov and sim.lyapunov_mode != "none" else [0.0] * sim.num_uav

    return RLResourceAllocationEnv(
        blocks=sim.blocks,
        demands=demands,
        compute=compute,
        memory=memory,
        weights=weights,
        lyapunov=lyapunov,
        dependencies=sim.dependencies,
        activation_sizes=activation_sizes,
        bandwidth=bandwidth,
        latency=latency,
        prev_assignment=sim.prev_assignment,
        load_guard=1.0,
        queue_block_threshold=getattr(sim.scheduler, "queue_block_threshold", None),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train single-agent RL scheduler")
    parser.add_argument("--episodes", type=int, default=500, help="number of training episodes")
    args = parser.parse_args()

    # Inspect the state dimensionality from a sample environment
    sample_env = _env_factory()
    sample_state = sample_env.reset()
    config = AgentConfig(state_dim=len(sample_state), action_dim=len(sample_env.compute))
    agent = RLAgent(config)

    # Train for a larger number of episodes to exercise the loop visibly
    train_agent(_env_factory, agent, episodes=args.episodes)

    # Report a deterministic evaluation reward as a quick sanity check
    eval_reward = evaluate(_env_factory, agent)
    print(f"Finished training RL agent; deterministic eval reward={eval_reward:.3f}")


if __name__ == "__main__":
    main()
