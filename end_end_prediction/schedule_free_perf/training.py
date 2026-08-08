"""Training, validation, checkpointing, and resource-range envelopes."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .data import Example, make_batch, minibatches
from .losses import combined_loss
from .model import ScheduleFreeModel, model_config


@dataclass(frozen=True)
class TrainingConfig:
    mode: str = "latent_physics"
    hidden_dim: int = 128
    message_steps: int = 3
    epochs: int = 30
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    auxiliary_weight: float = 0.2
    patience: int = 8
    seed: int = 0
    device: str = "cpu"


@dataclass
class TrainingResult:
    model: ScheduleFreeModel
    history: list[dict[str, float]]
    hardware_min: list[float]
    hardware_max: list[float]


def _hardware_envelope(examples: list[Example]) -> tuple[list[float], list[float]]:
    vectors = [example.hardware.vector() for example in examples]
    return (
        [min(row[index] for row in vectors) for index in range(len(vectors[0]))],
        [max(row[index] for row in vectors) for index in range(len(vectors[0]))],
    )


def _epoch_loss(
    model: ScheduleFreeModel,
    examples: list[Example],
    config: TrainingConfig,
) -> float:
    if not examples:
        return math.nan
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for records in minibatches(examples, config.batch_size, shuffle=False, seed=0):
            batch = make_batch(records).to(config.device)
            prediction = model(batch.graph, batch.hardware, batch.physics)
            loss = combined_loss(
                prediction,
                batch.target_log_latency,
                batch.auxiliary_targets,
                batch.auxiliary_masks,
                auxiliary_weight=config.auxiliary_weight,
            )
            total += float(loss.total) * len(records)
            count += len(records)
    return total / count


def train_model(
    train: list[Example],
    validation: list[Example],
    config: TrainingConfig,
) -> TrainingResult:
    if not train:
        raise ValueError("training set is empty")
    torch.manual_seed(config.seed)
    model = ScheduleFreeModel(
        mode=config.mode,
        hidden_dim=config.hidden_dim,
        message_steps=config.message_steps,
    ).to(config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_validation = math.inf
    stale = 0
    history: list[dict[str, float]] = []

    for epoch in range(config.epochs):
        model.train()
        total = 0.0
        latency_total = 0.0
        auxiliary_total = 0.0
        count = 0
        for records in minibatches(
            train, config.batch_size, shuffle=True, seed=config.seed + epoch
        ):
            batch = make_batch(records).to(config.device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch.graph, batch.hardware, batch.physics)
            loss = combined_loss(
                prediction,
                batch.target_log_latency,
                batch.auxiliary_targets,
                batch.auxiliary_masks,
                auxiliary_weight=config.auxiliary_weight,
            )
            loss.total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            size = len(records)
            total += float(loss.total.detach()) * size
            latency_total += float(loss.latency_nll.detach()) * size
            auxiliary_total += float(loss.auxiliary.detach()) * size
            count += size
        validation_loss = _epoch_loss(model, validation, config)
        selection_loss = validation_loss if math.isfinite(validation_loss) else total / count
        history.append(
            {
                "epoch": float(epoch + 1),
                "train_loss": total / count,
                "latency_nll": latency_total / count,
                "auxiliary_loss": auxiliary_total / count,
                "validation_loss": validation_loss,
            }
        )
        if selection_loss < best_validation:
            best_validation = selection_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    hardware_min, hardware_max = _hardware_envelope(train)
    return TrainingResult(model, history, hardware_min, hardware_max)


def save_checkpoint(
    result: TrainingResult,
    config: TrainingConfig,
    path: str | Path,
) -> None:
    torch.save(
        {
            "format_version": 1,
            "model_config": model_config(result.model),
            "model_state": result.model.state_dict(),
            "training_config": asdict(config),
            "history": result.history,
            "hardware_min": result.hardware_min,
            "hardware_max": result.hardware_max,
        },
        Path(path),
    )


def load_checkpoint(
    path: str | Path, device: str = "cpu"
) -> tuple[ScheduleFreeModel, dict]:
    payload = torch.load(Path(path), map_location=device, weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("unsupported checkpoint")
    model = ScheduleFreeModel(**payload["model_config"]).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload


def write_history(history: list[dict[str, float]], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2, allow_nan=False)
        handle.write("\n")

