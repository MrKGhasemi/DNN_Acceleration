"""ViT-B/16 construction and partial fine-tuning policy."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torchvision.models import ViT_B_16_Weights, vit_b_16


WEIGHTS = ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1


def build_model(*, num_classes: int, trainable_encoder_blocks: int) -> nn.Module:
    """Load SWAG E2E weights and fine-tune only the final transformer blocks."""

    model = vit_b_16(weights=WEIGHTS)
    for parameter in model.parameters():
        parameter.requires_grad = False

    for block in model.encoder.layers[-trainable_encoder_blocks:]:
        for parameter in block.parameters():
            parameter.requires_grad = True

    for parameter in model.encoder.ln.parameters():
        parameter.requires_grad = True

    model.heads = nn.Linear(model.hidden_dim, num_classes)
    return model


def parameter_counts(model: nn.Module) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def unwrap_compiled(model: nn.Module) -> nn.Module:
    """Return the original module so checkpoints do not acquire `_orig_mod.` keys."""

    return getattr(model, "_orig_mod", model)


def weight_metadata() -> dict[str, Any]:
    return {
        "architecture": "torchvision.models.vit_b_16",
        "weights": "ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1",
        "input_size": [3, 384, 384],
        "patch_size": 16,
        "hidden_dimension": 768,
        "encoder_blocks": 12,
        "tokens_including_class_token": (384 // 16) ** 2 + 1,
    }
