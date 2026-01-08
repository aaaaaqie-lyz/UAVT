"""Entry point for running a toy MARL training loop."""
from __future__ import annotations

import argparse

from uav_llm_partition.controller.scheduler_marl import MARLScheduler
from uav_llm_partition.rl.marl.trainer import MARLTrainer
from uav_llm_partition.sim.simulator import Simulator


def main() -> None:
    parser = argparse.ArgumentParser(description="Train multi-agent scheduler")
    parser.add_argument("--episodes", type=int, default=500, help="number of training episodes")
    parser.add_argument(
        "--model-path", type=str, default="marl_agent.json", help="where to save the trained MARL model"
    )
    parser.add_argument(
        "--resume", action="store_true", help="resume from the provided model path if it exists"
    )
    parser.add_argument(
        "--topology",
        type=str,
        default="uav,uav,uav,cloud",
        help="Comma-separated device types (e.g. uav,uav,cloud).",
    )
    args = parser.parse_args()
    device_types = [t.strip().lower() for t in args.topology.split(",") if t.strip()]
    num_uav = len(device_types)

    def _env_factory():
        # Build a simulator with MARL scheduler to expose block lists and demands
        sim = Simulator(
            num_uav=num_uav,
            num_layers=2,
            num_heads=4,
            hidden_size=1024,
            head_dim=None,
            intervals=1,
            interval_tokens=6,
            scheduler_type="marl",
            use_lyapunov=False,
            device_types=device_types,
        )
        sim_positions, sim_mob = sim.mobility.update()
        bandwidth, conn, los_score = sim.channel.compute(sim_positions, sim.device_types)
        compute, memory = sim.resource.sample(sim.device_types)
        demands = sim.demand_model.update_interval()
        activation_sizes = {(u, v): sim.demand_model.activation_size(u, v) for u, v in sim.dependencies}
        weights = sim.weights.compute(compute, memory, los_score, sim_mob)

        return sim, bandwidth, demands, activation_sizes, weights, compute, memory

    # Initialize a single env to infer dimensions
    sim, bandwidth, demands, activation_sizes, weights, compute, memory = _env_factory()
    scheduler: MARLScheduler = sim._create_scheduler("marl", "round_robin")  # type: ignore[attr-defined]
    if scheduler:
        scheduler.device_types = getattr(sim, "device_types", ["uav" for _ in range(sim.num_uav)])
    if scheduler:
        from uav_llm_partition.rl.marl.env import MultiAgentResourceAllocationEnv
        from uav_llm_partition.rl.marl.mappo import MAPPOAgent

        def _make_env() -> MultiAgentResourceAllocationEnv:
            sim, bandwidth, demands, activation_sizes, weights, compute, memory = _env_factory()
            return MultiAgentResourceAllocationEnv(
                sim.blocks,
                demands,
                compute,
                memory,
                bandwidth,
                [0.0 for _ in range(sim.num_uav)],
                weights,
                sim.dependencies,
                activation_sizes,
                prev_assignment={},
                migration_overhead=scheduler.mig_overhead,
                device_types=sim.device_types,
            )

        env_sample = _make_env()
        local_states, global_state = env_sample.reset()
        cfg = MAPPOAgent.default_config(
            num_agents=len(local_states),
            local_state_dim=len(local_states[0]),
            global_state_dim=len(global_state),
        )
        agent = MAPPOAgent(cfg)
        if args.resume:
            try:
                loaded = MAPPOAgent.load(args.model_path)
                if (
                    loaded.cfg.num_agents == cfg.num_agents
                    and loaded.cfg.local_state_dim == cfg.local_state_dim
                    and loaded.cfg.global_state_dim == cfg.global_state_dim
                ):
                    agent = loaded
                    print(f"Loaded MARL agent from {args.model_path}")
                else:
                    print("Model dimensions mismatch; starting fresh agent")
            except FileNotFoundError:
                pass
        trainer = MARLTrainer(agent)
        trainer.train(_make_env, episodes=args.episodes)
        agent.save(args.model_path)
        print(f"Saved MARL agent to {args.model_path}")
        print("MARL training finished")
    else:
        print("MARL scheduler unavailable")


if __name__ == "__main__":
    main()
