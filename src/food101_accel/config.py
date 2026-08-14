"""Experiment configuration with no heavyweight imports.

Keeping this module independent of PyTorch makes configuration validation and
CI possible on machines that do not have a CUDA-enabled PyTorch installation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


Profile = Literal["baseline", "accelerated"]
AmpDtype = Literal["bfloat16", "float16"]


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """All inputs needed to reproduce one training profile."""

    profile: Profile
    data_root: Path = Path("data")
    output_dir: Path = Path("artifacts")
    download: bool = False
    device: str = "cuda"
    seed: int = 42
    image_size: int = 384
    num_classes: int = 101
    batch_size: int = 128
    num_workers: int = 4
    epochs: int = 3
    max_train_batches: int | None = None
    max_eval_batches: int | None = None
    learning_rate: float = 1e-3
    weight_decay: float = 5e-4
    label_smoothing: float = 0.2
    trainable_encoder_blocks: int = 2
    compile_mode: str = "default"
    amp_dtype: AmpDtype = "bfloat16"
    save_best: bool = False

    @property
    def use_compile(self) -> bool:
        return self.profile == "accelerated"

    @property
    def use_amp(self) -> bool:
        return self.profile == "accelerated"

    @property
    def allow_tf32(self) -> bool:
        return self.profile == "accelerated"

    @property
    def matmul_precision(self) -> str:
        return "high" if self.profile == "accelerated" else "highest"

    def validate(self) -> None:
        if self.profile not in ("baseline", "accelerated"):
            raise ValueError(f"Unknown profile: {self.profile!r}")
        if self.device not in ("cuda", "cpu", "auto"):
            raise ValueError("device must be 'cuda', 'cpu', or 'auto'")
        if self.image_size != 384:
            raise ValueError("SWAG E2E ViT-B/16 uses 384 x 384 inputs")
        if self.num_classes != 101:
            raise ValueError("Food-101 must have exactly 101 output classes")
        if self.batch_size < 1 or self.num_workers < 0 or self.epochs < 1:
            raise ValueError("batch_size/epochs must be positive and workers non-negative")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        if not 1 <= self.trainable_encoder_blocks <= 12:
            raise ValueError("trainable_encoder_blocks must be between 1 and 12")
        for name in ("max_train_batches", "max_eval_batches"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be positive when supplied")
        if self.use_amp and self.device == "cpu":
            raise ValueError("the accelerated profile in this repo requires CUDA")

    def as_serializable_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["data_root"] = str(self.data_root)
        payload["output_dir"] = str(self.output_dir)
        payload["features"] = {
            "torch_compile": self.use_compile,
            "automatic_mixed_precision": self.use_amp,
            "allow_tf32": self.allow_tf32,
            "float32_matmul_precision": self.matmul_precision,
        }
        return payload
