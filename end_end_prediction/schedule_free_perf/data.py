"""Manifest loading, graph preparation, batching, and leakage-safe splits."""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import torch
from torch import Tensor

from .contracts import HardwareSpec, MeasurementRecord, load_hardware, load_manifest
from .model import GraphBatch, batch_graphs, hardware_tensor, physics_tensor
from .stablehlo import SemanticGraph, parse_stablehlo_file

SplitKey = Literal["workload", "hardware", "compiler"]


def load_hardware_catalog(directory: str | Path) -> dict[str, HardwareSpec]:
    catalog: dict[str, HardwareSpec] = {}
    for path in sorted(Path(directory).glob("*.json")):
        spec = load_hardware(path)
        if spec.hardware_id in catalog:
            raise ValueError(f"duplicate hardware id {spec.hardware_id}")
        catalog[spec.hardware_id] = spec
    if not catalog:
        raise ValueError(f"no hardware descriptors found in {directory}")
    return catalog


@dataclass(frozen=True)
class Example:
    record: MeasurementRecord
    graph: SemanticGraph
    hardware: HardwareSpec


def load_examples(
    manifest_path: str | Path,
    hardware_directory: str | Path,
    *,
    skip_parse_errors: bool = False,
) -> tuple[list[Example], list[str]]:
    catalog = load_hardware_catalog(hardware_directory)
    examples: list[Example] = []
    errors: list[str] = []
    graph_cache: dict[tuple[str, str], SemanticGraph] = {}
    for record in load_manifest(manifest_path):
        if record.hardware_id not in catalog:
            errors.append(f"{record.record_id}: unknown hardware {record.hardware_id}")
            continue
        key = (record.workload_id, record.stablehlo_sha256)
        try:
            graph = graph_cache.get(key)
            if graph is None:
                graph = parse_stablehlo_file(record.stablehlo_path, record.workload_id)
                graph_cache[key] = graph
            examples.append(Example(record, graph, catalog[record.hardware_id]))
        except (OSError, ValueError) as error:
            message = f"{record.record_id}: {error}"
            if not skip_parse_errors:
                raise ValueError(message) from error
            errors.append(message)
    if not examples:
        raise ValueError("no usable examples")
    return examples, errors


def split_group(example: Example, key: SplitKey) -> str:
    if key == "workload":
        return example.record.workload_id
    if key == "hardware":
        return example.record.hardware_id
    if key == "compiler":
        return "|".join(sorted(example.record.compiler.items()).__repr__())
    raise ValueError(key)


def _fraction(group: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{group}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


@dataclass(frozen=True)
class Split:
    train: list[Example]
    validation: list[Example]
    test: list[Example]
    key: SplitKey

    def assert_no_leakage(self) -> None:
        groups = [
            {split_group(example, self.key) for example in partition}
            for partition in (self.train, self.validation, self.test)
        ]
        if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
            raise AssertionError(f"{self.key} leakage detected")


def grouped_split(
    examples: list[Example],
    key: SplitKey = "workload",
    validation_fraction: float = 0.1,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> Split:
    if validation_fraction < 0 or test_fraction < 0 or validation_fraction + test_fraction >= 1:
        raise ValueError("invalid split fractions")
    partitions: tuple[list[Example], list[Example], list[Example]] = ([], [], [])
    for example in examples:
        value = _fraction(split_group(example, key), seed)
        target = 2 if value < test_fraction else 1 if value < test_fraction + validation_fraction else 0
        partitions[target].append(example)
    split = Split(partitions[0], partitions[1], partitions[2], key)
    split.assert_no_leakage()
    return split


def leave_one_gpu_out(
    examples: list[Example], held_out_hardware: str, validation_seed: int = 0
) -> Split:
    test = [item for item in examples if item.record.hardware_id == held_out_hardware]
    remaining = [item for item in examples if item.record.hardware_id != held_out_hardware]
    inner = grouped_split(
        remaining, key="workload", validation_fraction=0.1, test_fraction=0.0, seed=validation_seed
    )
    if not test:
        raise ValueError(f"no examples for held-out hardware {held_out_hardware}")
    return Split(inner.train, inner.validation, test, "hardware")


@dataclass
class TrainingBatch:
    graph: GraphBatch
    hardware: Tensor
    physics: Tensor
    target_log_latency: Tensor
    auxiliary_targets: dict[str, Tensor]
    auxiliary_masks: dict[str, Tensor]

    def to(self, device: str | torch.device) -> "TrainingBatch":
        return TrainingBatch(
            graph=self.graph.to(device),
            hardware=self.hardware.to(device),
            physics=self.physics.to(device),
            target_log_latency=self.target_log_latency.to(device),
            auxiliary_targets={key: value.to(device) for key, value in self.auxiliary_targets.items()},
            auxiliary_masks={key: value.to(device) for key, value in self.auxiliary_masks.items()},
        )


def _optional_numeric(value: object) -> tuple[float, bool]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0, False
    return (numeric, math.isfinite(numeric) and numeric >= 0)


def make_batch(examples: list[Example]) -> TrainingBatch:
    graphs = [example.graph for example in examples]
    hardware = [example.hardware for example in examples]
    targets: dict[str, list[float]] = {
        "log_dram_bytes": [],
        "fused_op_ratio": [],
        "log_n_kernels": [],
        "compute_utilization": [],
        "memory_utilization": [],
    }
    masks: dict[str, list[bool]] = {key: [] for key in targets}
    for example in examples:
        labels = example.record.privileged_labels
        dram, has_dram = _optional_numeric(labels.get("label_dram_bytes"))
        fused, has_fused = _optional_numeric(labels.get("label_fused_op_ratio"))
        kernels, has_kernels = _optional_numeric(labels.get("label_n_kernels"))
        achieved, has_achieved = _optional_numeric(labels.get("achieved_tflops"))
        peak = example.hardware.effective_peak_tflops(
            dtype=example.graph.dtype,
            contraction_flops=example.graph.contraction_flops(),
            other_flops=example.graph.other_flops(),
        )
        targets["log_dram_bytes"].append(math.log(max(dram, 1.0)))
        masks["log_dram_bytes"].append(has_dram and dram > 0)
        targets["fused_op_ratio"].append(min(1.0, fused))
        masks["fused_op_ratio"].append(has_fused)
        targets["log_n_kernels"].append(math.log(max(kernels, 1.0)))
        masks["log_n_kernels"].append(has_kernels and kernels > 0)
        targets["compute_utilization"].append(min(1.0, achieved / max(peak, 1e-9)))
        masks["compute_utilization"].append(has_achieved and achieved > 0)
        targets["memory_utilization"].append(0.0)
        masks["memory_utilization"].append(False)
    return TrainingBatch(
        graph=batch_graphs(graphs),
        hardware=hardware_tensor(hardware),
        physics=physics_tensor(graphs, hardware),
        target_log_latency=torch.tensor(
            [math.log(example.record.latency_us) for example in examples], dtype=torch.float32
        ),
        auxiliary_targets={
            key: torch.tensor(values, dtype=torch.float32) for key, values in targets.items()
        },
        auxiliary_masks={key: torch.tensor(values, dtype=torch.bool) for key, values in masks.items()},
    )


def minibatches(
    examples: list[Example], batch_size: int, *, shuffle: bool, seed: int
) -> Iterable[list[Example]]:
    indices = list(range(len(examples)))
    if shuffle:
        random.Random(seed).shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield [examples[index] for index in indices[start : start + batch_size]]

