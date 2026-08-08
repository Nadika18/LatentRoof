"""Latency likelihood and masked privileged-label supervision."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .model import Prediction


def gaussian_nll(prediction: Prediction, target_log_latency: Tensor) -> Tensor:
    variance_inverse = torch.exp(-2.0 * prediction.log_std)
    return (
        prediction.log_std
        + 0.5 * (target_log_latency - prediction.log_latency_mean).square() * variance_inverse
        + 0.5 * math.log(2.0 * math.pi)
    ).mean()


def masked_mse(predicted: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    valid = mask & torch.isfinite(target) & torch.isfinite(predicted)
    if not valid.any():
        return predicted.sum() * 0.0
    return (predicted[valid] - target[valid]).square().mean()


@dataclass(frozen=True)
class LossBreakdown:
    total: Tensor
    latency_nll: Tensor
    auxiliary: Tensor


def combined_loss(
    prediction: Prediction,
    target_log_latency: Tensor,
    auxiliary_targets: dict[str, Tensor],
    auxiliary_masks: dict[str, Tensor],
    *,
    auxiliary_weight: float = 0.2,
) -> LossBreakdown:
    latency = gaussian_nll(prediction, target_log_latency)
    auxiliary_terms = []
    for name, target in auxiliary_targets.items():
        if name in prediction.auxiliary:
            auxiliary_terms.append(
                masked_mse(prediction.auxiliary[name], target, auxiliary_masks[name])
            )
    auxiliary = (
        torch.stack(auxiliary_terms).mean()
        if auxiliary_terms
        else prediction.log_latency_mean.sum() * 0.0
    )
    return LossBreakdown(latency + auxiliary_weight * auxiliary, latency, auxiliary)

