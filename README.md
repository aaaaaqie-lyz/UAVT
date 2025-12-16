# UAV-Swarm Transformer Head-Level Partitioning

This repository provides a minimal simulation of head-level Transformer partitioning across UAVs with Lyapunov-guided heuristic scheduling. The simulator follows the provided specification, including KV-cache-aware head placement, dynamic UAV weights, and per-interval load balancing.

## Features
- **Interval loop:** Updates mobility, channel bandwidth, resources, block demands, scheduling, migrations, delay decomposition, and Lyapunov queues each interval.
- **Head + KV cache constraint:** Heads carry their KV cache when migrating; migration cost uses KV size.
- **Heuristic scheduler with repair:** Uses dynamic weights and Lyapunov pressure to assign blocks while repairing overloads.
- **Metrics:** Tracks max load, Jain fairness, delay breakdown (compute/comm/migration), and migration statistics.

## Layout
- `uav_llm_partition/env`: Mobility, channel, and resource models.
- `uav_llm_partition/model_partition`: Blocks, demand model with KV growth, and dependency graph builder.
- `uav_llm_partition/controller`: Weight manager, Lyapunov queue, heuristic scheduler, and state collector.
- `uav_llm_partition/sim`: Simulator loop, metrics helpers, and logging.
- `uav_llm_partition/scripts/run_sim.py`: Example entry point running a small simulation.

## Quick start
```bash
python -m uav_llm_partition.scripts.run_sim
```
The script prints interval logs and a summary of average metrics after the run.

