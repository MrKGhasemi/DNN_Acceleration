"""Run one baseline or accelerated training profile."""

from __future__ import annotations

import argparse

from .cli import add_experiment_arguments, config_from_args
from .engine import run_experiment
from .io_utils import utc_run_stamp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Food-101 ViT-B/16 with one numerical profile."
    )
    add_experiment_arguments(parser, include_profile=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_dir = args.output_dir / f"{utc_run_stamp()}-{args.profile}"
    config = config_from_args(args, profile=args.profile, output_dir=run_dir)
    result = run_experiment(config, run_dir)
    print(f"result: {run_dir / 'result.json'}")
    print(f"best top-1: {result['best_eval_top1'] * 100:.2f}%")


if __name__ == "__main__":
    main()
