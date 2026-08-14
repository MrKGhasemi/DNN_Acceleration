"""Shared command-line parsing."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import ExperimentConfig, Profile


def add_experiment_arguments(
    parser: argparse.ArgumentParser, *, include_profile: bool
) -> None:
    if include_profile:
        parser.add_argument(
            "--profile", choices=("baseline", "accelerated"), default="accelerated"
        )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.2)
    parser.add_argument("--trainable-encoder-blocks", type=int, default=2)
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="default",
    )
    parser.add_argument(
        "--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16"
    )
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--save-best", action="store_true")


def config_from_args(
    args: argparse.Namespace,
    *,
    profile: Profile,
    output_dir: Path | None = None,
    seed: int | None = None,
) -> ExperimentConfig:
    return ExperimentConfig(
        profile=profile,
        data_root=args.data_root,
        output_dir=output_dir or args.output_dir,
        download=args.download,
        device=args.device,
        seed=args.seed if seed is None else seed,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        epochs=args.epochs,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        trainable_encoder_blocks=args.trainable_encoder_blocks,
        compile_mode=args.compile_mode,
        amp_dtype=args.amp_dtype,
        save_best=args.save_best,
    )
