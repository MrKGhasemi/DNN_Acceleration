"""Controlled, repeated A/B benchmark runner."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from pathlib import Path
from typing import cast

from .cli import add_experiment_arguments, config_from_args
from .config import ExperimentConfig, Profile
from .engine import run_experiment
from .io_utils import utc_run_stamp, write_json
from .reporting import aggregate_profile_runs, compute_speedups


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare identical Food-101/ViT training under baseline and "
            "accelerated numerical profiles."
        )
    )
    add_experiment_arguments(parser, include_profile=False)
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=("baseline", "accelerated"),
        default=["baseline", "accelerated"],
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Use at least 2 for alternated run order and more credible timings.",
    )
    parser.add_argument(
        "--discard-first-epochs",
        type=int,
        default=1,
        help="Epochs removed from the steady-state statistic, not from cold timing.",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.repeats < 1:
        raise SystemExit("--repeats must be positive")
    if args.discard_first_epochs < 0:
        raise SystemExit("--discard-first-epochs cannot be negative")
    if args.discard_first_epochs >= args.epochs:
        raise SystemExit("--discard-first-epochs must be smaller than --epochs")
    if len(set(args.profiles)) != len(args.profiles):
        raise SystemExit("--profiles must not contain duplicates")


def _run_worker(config: ExperimentConfig, run_dir: Path) -> None:
    run_experiment(config, run_dir)


def _run_isolated(config: ExperimentConfig, run_dir: Path) -> dict:
    """Use a fresh process so CUDA/compiler caches cannot cross trial boundaries."""

    process = multiprocessing.get_context("spawn").Process(
        target=_run_worker, args=(config, run_dir)
    )
    process.start()
    process.join()
    if process.exitcode != 0:
        raise RuntimeError(
            f"{config.profile} trial failed with process exit code {process.exitcode}"
        )
    return json.loads((run_dir / "result.json").read_text(encoding="utf-8"))


def main() -> None:
    args = build_parser().parse_args()
    _validate_args(args)
    benchmark_dir = Path(args.output_dir) / f"{utc_run_stamp()}-benchmark"
    runs: list[dict] = []

    requested = [cast(Profile, profile) for profile in args.profiles]
    for repeat in range(args.repeats):
        # Alternating order reduces systematic bias from OS caches, GPU thermal
        # state, and one profile always running first.
        ordered = requested if repeat % 2 == 0 else list(reversed(requested))
        for profile in ordered:
            run_dir = benchmark_dir / f"repeat-{repeat + 1}" / profile
            config = config_from_args(
                args,
                profile=profile,
                output_dir=run_dir,
                seed=args.seed + repeat,
            )
            print(f"\n=== repeat {repeat + 1}/{args.repeats}: {profile} ===")
            runs.append(_run_isolated(config, run_dir))

    summary = aggregate_profile_runs(
        runs, discard_first_epochs=args.discard_first_epochs
    )
    payload = {
        "schema_version": 1,
        "benchmark_directory": str(benchmark_dir),
        "discard_first_epochs_for_steady_state": args.discard_first_epochs,
        "summary": summary,
    }
    if set(args.profiles) == {"baseline", "accelerated"}:
        payload["speedups"] = compute_speedups(summary)

    write_json(benchmark_dir / "benchmark.json", payload)
    print(f"\nbenchmark: {benchmark_dir / 'benchmark.json'}")
    if "speedups" in payload:
        speedups = payload["speedups"]
        print(f"cold epoch speedup:   {speedups['cold_epoch_speedup']:.3f}x")
        print(f"steady epoch speedup: {speedups['steady_epoch_speedup']:.3f}x")


if __name__ == "__main__":
    main()
