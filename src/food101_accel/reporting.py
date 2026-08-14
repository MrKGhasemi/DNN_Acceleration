"""Pure-Python result aggregation used by the benchmark and tests."""

from __future__ import annotations

from statistics import mean, stdev
from typing import Any, Iterable


def _mean_and_std(values: list[float]) -> dict[str, float | None]:
    if not values:
        raise ValueError("cannot summarize an empty collection")
    return {
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else None,
    }


def aggregate_profile_runs(
    runs: Iterable[dict[str, Any]], *, discard_first_epochs: int = 1
) -> dict[str, dict[str, Any]]:
    """Aggregate run dictionaries by profile.

    Cold timings include every epoch. Steady-state timings discard the first
    N epochs of each run so compilation and allocator warm-up are visible
    rather than silently hidden.
    """

    grouped: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault(run["profile"], []).append(run)

    result: dict[str, dict[str, Any]] = {}
    for profile, profile_runs in grouped.items():
        cold: list[float] = []
        steady: list[float] = []
        train_throughput: list[float] = []
        top1: list[float] = []
        top5: list[float] = []

        for run in profile_runs:
            epochs = run.get("epochs", [])
            if not epochs:
                raise ValueError(f"run for {profile!r} contains no epochs")
            cold.extend(float(epoch["total_seconds"]) for epoch in epochs)
            steady.extend(
                float(epoch["total_seconds"])
                for epoch in epochs[discard_first_epochs:]
            )
            train_throughput.extend(
                float(epoch["train_samples_per_second"]) for epoch in epochs
            )
            top1.append(float(epochs[-1]["eval_top1"]))
            top5.append(float(epochs[-1]["eval_top5"]))

        result[profile] = {
            "runs": len(profile_runs),
            "cold_epoch_seconds": _mean_and_std(cold),
            "steady_epoch_seconds": _mean_and_std(steady or cold),
            "train_samples_per_second": _mean_and_std(train_throughput),
            "final_eval_top1": _mean_and_std(top1),
            "final_eval_top5": _mean_and_std(top5),
        }

    return result


def compute_speedups(summary: dict[str, dict[str, Any]]) -> dict[str, float]:
    """Return baseline/accelerated ratios; values above one are faster."""

    try:
        baseline = summary["baseline"]
        accelerated = summary["accelerated"]
    except KeyError as exc:
        raise ValueError("both baseline and accelerated profiles are required") from exc

    return {
        "cold_epoch_speedup": (
            baseline["cold_epoch_seconds"]["mean"]
            / accelerated["cold_epoch_seconds"]["mean"]
        ),
        "steady_epoch_speedup": (
            baseline["steady_epoch_seconds"]["mean"]
            / accelerated["steady_epoch_seconds"]["mean"]
        ),
        "training_throughput_speedup": (
            accelerated["train_samples_per_second"]["mean"]
            / baseline["train_samples_per_second"]["mean"]
        ),
    }
