#!/usr/bin/env python3
"""Run multi-stage training based on a JSON configuration."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Stage:
    name: str
    topology: str
    penalty: str
    episodes: int
    checkpoint: str


def load_stages(config_path: Path) -> list[Stage]:
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    stages = []
    for entry in payload.get("stages", []):
        stages.append(
            Stage(
                name=entry["name"],
                topology=entry["topology"],
                penalty=entry["penalty"],
                episodes=int(entry["episodes"]),
                checkpoint=entry["checkpoint"],
            )
        )
    return stages


def render_command(template: str, stage: Stage) -> str:
    return template.format(
        name=stage.name,
        topology=stage.topology,
        penalty=stage.penalty,
        episodes=stage.episodes,
        checkpoint=stage.checkpoint,
    )


def run_stage(command: str, dry_run: bool) -> None:
    if dry_run:
        print(command)
        return

    subprocess.run(command, shell=True, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run staged training using a JSON configuration.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/staged_training.json"),
        help="Path to the staged training JSON config.",
    )
    parser.add_argument(
        "--train-cmd",
        required=True,
        help=(
            "Command template to run each stage. "
            "Available fields: {name}, {topology}, {penalty}, "
            "{episodes}, {checkpoint}."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stages = load_stages(args.config)
    if not stages:
        raise SystemExit(f"No stages found in {args.config}")

    for stage in stages:
        command = render_command(args.train_cmd, stage)
        run_stage(command, args.dry_run)


if __name__ == "__main__":
    main()
