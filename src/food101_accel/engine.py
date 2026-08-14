"""Training/evaluation engine with CUDA-correct timing."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

from .config import ExperimentConfig
from .data import build_loaders
from .io_utils import write_json
from .model import (
    build_model,
    parameter_counts,
    trainable_parameters,
    unwrap_compiled,
    weight_metadata,
)
from .runtime import (
    autocast_context,
    build_grad_scaler,
    configure_precision,
    environment_metadata,
    maybe_compile,
    resolve_device,
    seed_everything,
    synchronize,
    validate_acceleration_support,
)


@dataclass(slots=True)
class PhaseMetrics:
    loss: float
    top1: float
    top5: float
    seconds: float
    samples: int
    batches: int

    @property
    def samples_per_second(self) -> float:
        return self.samples / self.seconds


def _update_accuracy(
    logits: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    predictions = logits.detach().topk(k=5, dim=1).indices
    top1 = predictions[:, 0].eq(targets).sum()
    top5 = predictions.eq(targets[:, None]).any(dim=1).sum()
    return top1, top5


def _finalize_phase(
    *,
    loss_sum: torch.Tensor,
    top1_correct: torch.Tensor,
    top5_correct: torch.Tensor,
    samples: int,
    batches: int,
    seconds: float,
) -> PhaseMetrics:
    if samples == 0 or batches == 0:
        raise RuntimeError("the data loader produced no batches")
    return PhaseMetrics(
        loss=loss_sum.item() / samples,
        top1=top1_correct.item() / samples,
        top5=top5_correct.item() / samples,
        seconds=seconds,
        samples=samples,
        batches=batches,
    )


def train_one_epoch(
    *,
    model: nn.Module,
    loader: Iterable[Any],
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    scaler: torch.amp.GradScaler,
    config: ExperimentConfig,
    device: torch.device,
) -> PhaseMetrics:
    model.train()
    loss_sum = torch.zeros((), device=device, dtype=torch.float32)
    top1_correct = torch.zeros((), device=device, dtype=torch.int64)
    top5_correct = torch.zeros((), device=device, dtype=torch.int64)
    samples = batches = 0

    synchronize(device)
    started = time.perf_counter()
    for batch_index, (images, targets) in enumerate(loader):
        if config.max_train_batches is not None and batch_index >= config.max_train_batches:
            break
        images = images.to(device, non_blocking=device.type == "cuda")
        targets = targets.to(device, non_blocking=device.type == "cuda")
        optimizer.zero_grad(set_to_none=True)

        with autocast_context(config, device):
            logits = model(images)
            loss = loss_function(logits, targets)

        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = targets.shape[0]
        correct1, correct5 = _update_accuracy(logits, targets)
        # Keep reductions on the GPU. Calling .item() per batch would force a
        # synchronization and distort the runtime being measured.
        loss_sum += loss.detach().float() * batch_size
        top1_correct += correct1
        top5_correct += correct5
        samples += batch_size
        batches += 1

    synchronize(device)
    seconds = time.perf_counter() - started
    return _finalize_phase(
        loss_sum=loss_sum,
        top1_correct=top1_correct,
        top5_correct=top5_correct,
        samples=samples,
        batches=batches,
        seconds=seconds,
    )


@torch.inference_mode()
def evaluate(
    *,
    model: nn.Module,
    loader: Iterable[Any],
    loss_function: nn.Module,
    config: ExperimentConfig,
    device: torch.device,
) -> PhaseMetrics:
    model.eval()
    loss_sum = torch.zeros((), device=device, dtype=torch.float32)
    top1_correct = torch.zeros((), device=device, dtype=torch.int64)
    top5_correct = torch.zeros((), device=device, dtype=torch.int64)
    samples = batches = 0

    synchronize(device)
    started = time.perf_counter()
    for batch_index, (images, targets) in enumerate(loader):
        if config.max_eval_batches is not None and batch_index >= config.max_eval_batches:
            break
        images = images.to(device, non_blocking=device.type == "cuda")
        targets = targets.to(device, non_blocking=device.type == "cuda")
        with autocast_context(config, device):
            logits = model(images)
            loss = loss_function(logits, targets)

        batch_size = targets.shape[0]
        correct1, correct5 = _update_accuracy(logits, targets)
        loss_sum += loss.detach().float() * batch_size
        top1_correct += correct1
        top5_correct += correct5
        samples += batch_size
        batches += 1

    synchronize(device)
    seconds = time.perf_counter() - started
    return _finalize_phase(
        loss_sum=loss_sum,
        top1_correct=top1_correct,
        top5_correct=top5_correct,
        samples=samples,
        batches=batches,
        seconds=seconds,
    )


def run_experiment(config: ExperimentConfig, run_dir: Path) -> dict[str, Any]:
    """Run one profile and persist a machine-readable record."""

    config.validate()
    seed_everything(config.seed)
    device = resolve_device(config.device)
    validate_acceleration_support(config, device)
    configure_precision(config, device)

    train_loader, eval_loader, dataset_metadata = build_loaders(
        config, use_cuda=device.type == "cuda"
    )
    base_model = build_model(
        num_classes=config.num_classes,
        trainable_encoder_blocks=config.trainable_encoder_blocks,
    ).to(device)
    model_metadata = {**weight_metadata(), **parameter_counts(base_model)}
    model = maybe_compile(base_model, config)

    optimizer = torch.optim.Adagrad(
        trainable_parameters(base_model),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )
    loss_function = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    scaler = build_grad_scaler(config)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    result: dict[str, Any] = {
        "schema_version": 1,
        "profile": config.profile,
        "config": config.as_serializable_dict(),
        "environment": environment_metadata(device),
        "dataset": dataset_metadata,
        "model": model_metadata,
        "epochs": [],
    }
    best_top1 = -1.0

    for epoch in range(config.epochs):
        learning_rate = optimizer.param_groups[0]["lr"]
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_function=loss_function,
            scaler=scaler,
            config=config,
            device=device,
        )
        eval_metrics = evaluate(
            model=model,
            loader=eval_loader,
            loss_function=loss_function,
            config=config,
            device=device,
        )
        scheduler.step()

        epoch_record = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": train_metrics.loss,
            "train_top1": train_metrics.top1,
            "train_top5": train_metrics.top5,
            "eval_loss": eval_metrics.loss,
            "eval_top1": eval_metrics.top1,
            "eval_top5": eval_metrics.top5,
            "train_seconds": train_metrics.seconds,
            "eval_seconds": eval_metrics.seconds,
            "total_seconds": train_metrics.seconds + eval_metrics.seconds,
            "train_samples": train_metrics.samples,
            "eval_samples": eval_metrics.samples,
            "train_batches": train_metrics.batches,
            "eval_batches": eval_metrics.batches,
            "train_samples_per_second": train_metrics.samples_per_second,
            "eval_samples_per_second": eval_metrics.samples_per_second,
        }
        result["epochs"].append(epoch_record)
        write_json(run_dir / "result.json", result)

        print(
            f"[{config.profile}] epoch={epoch:02d} "
            f"train={train_metrics.seconds:.2f}s eval={eval_metrics.seconds:.2f}s "
            f"top1={eval_metrics.top1 * 100:.2f}% "
            f"top5={eval_metrics.top5 * 100:.2f}%"
        )

        if config.save_best and eval_metrics.top1 > best_top1:
            best_top1 = eval_metrics.top1
            torch.save(
                {
                    "epoch": epoch,
                    "model": unwrap_compiled(model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "config": config.as_serializable_dict(),
                    "best_eval_top1": best_top1,
                },
                run_dir / "best_checkpoint.pth",
            )

    if device.type == "cuda":
        result["peak_cuda_memory_bytes"] = torch.cuda.max_memory_allocated(device)
    result["best_eval_top1"] = max(epoch["eval_top1"] for epoch in result["epochs"])
    result["best_eval_top5"] = max(epoch["eval_top5"] for epoch in result["epochs"])
    write_json(run_dir / "result.json", result)
    return result
