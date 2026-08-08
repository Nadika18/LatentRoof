"""Regression, ranking, calibration, OOD, and hardware-sensitivity evaluation."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from statistics import median

import torch

from .contracts import HardwareSpec
from .data import Example, make_batch
from .model import ScheduleFreeModel, batch_graphs, hardware_tensor, physics_tensor


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        rank = (index + end - 1) / 2
        for position in range(index, end):
            ranks[order[position]] = rank
        index = end
    return ranks


def _correlation(left: list[float], right: list[float]) -> float:
    if len(left) < 2:
        return math.nan
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    return numerator / (left_scale * right_scale) if left_scale and right_scale else math.nan


@dataclass(frozen=True)
class Metrics:
    count: int
    median_ape_percent: float
    p90_ape_percent: float
    log_r2: float
    spearman: float
    coverage_50: float
    coverage_90: float
    coverage_95: float


def metrics(actual: list[float], predicted: list[float], sigma_log: list[float]) -> Metrics:
    if not actual or not (len(actual) == len(predicted) == len(sigma_log)):
        raise ValueError("metric arrays must be non-empty and aligned")
    ape = [abs(estimate - target) / target * 100 for target, estimate in zip(actual, predicted)]
    actual_log = [math.log(value) for value in actual]
    predicted_log = [math.log(max(value, 1e-12)) for value in predicted]
    mean_actual = sum(actual_log) / len(actual_log)
    denominator = sum((value - mean_actual) ** 2 for value in actual_log)
    r2 = (
        1
        - sum((target - estimate) ** 2 for target, estimate in zip(actual_log, predicted_log))
        / denominator
        if denominator
        else math.nan
    )

    def coverage(z: float) -> float:
        return sum(
            abs(target - estimate) <= z * sigma
            for target, estimate, sigma in zip(actual_log, predicted_log, sigma_log)
        ) / len(actual)

    return Metrics(
        count=len(actual),
        median_ape_percent=median(ape),
        p90_ape_percent=_percentile(ape, 0.9),
        log_r2=r2,
        spearman=_correlation(_ranks(actual), _ranks(predicted)),
        coverage_50=coverage(0.67448975),
        coverage_90=coverage(1.64485363),
        coverage_95=coverage(1.95996398),
    )


def naive_roofline(example: Example) -> float:
    graph, hardware = example.graph, example.hardware
    peak = hardware.effective_peak_tflops(
        dtype=graph.dtype,
        contraction_flops=graph.contraction_flops(),
        other_flops=graph.other_flops(),
    )
    compute = graph.total_flops / (peak * 1e6)
    memory = graph.logical_bytes / (hardware.memory_bandwidth_gbps * 1e3)
    return max(compute, memory) + hardware.launch_overhead_us


def hardware_ood(
    hardware: HardwareSpec,
    lower: list[float] | None,
    upper: list[float] | None,
) -> dict[str, object]:
    reasons = []
    outside_training_range = False
    vector = hardware.vector()
    if lower is not None and upper is not None:
        outside = [
            index
            for index, value in enumerate(vector)
            if value < lower[index] - 1e-6 or value > upper[index] + 1e-6
        ]
        if outside:
            outside_training_range = True
            reasons.append(f"{len(outside)} resource features outside training range")
    if hardware.estimated_fields:
        reasons.append("estimated hardware fields: " + ", ".join(hardware.estimated_fields))
    score = (0.75 if outside_training_range else 0.0) + (
        0.25 if hardware.estimated_fields else 0.0
    )
    return {
        "is_ood": outside_training_range,
        "score": min(1.0, score),
        "reasons": reasons,
    }


def predict_examples(
    model: ScheduleFreeModel,
    examples: list[Example],
    *,
    device: str = "cpu",
    hardware_min: list[float] | None = None,
    hardware_max: list[float] | None = None,
) -> list[dict[str, object]]:
    if not examples:
        return []
    batch = make_batch(examples).to(device)
    model.eval()
    with torch.no_grad():
        prediction = model(batch.graph, batch.hardware, batch.physics)
    rows = []
    for index, example in enumerate(examples):
        rows.append(
            {
                "record_id": example.record.record_id,
                "workload_id": example.record.workload_id,
                "workload_family": example.record.workload_family,
                "hardware_id": example.record.hardware_id,
                "actual_us": example.record.latency_us,
                "predicted_us": float(prediction.latency_us[index].cpu()),
                "sigma_log": math.exp(float(prediction.log_std[index].cpu())),
                "physics_us": float(prediction.physics_us[index].cpu()),
                "lower_bound_us": float(prediction.lower_bound_us[index].cpu()),
                "ood": hardware_ood(example.hardware, hardware_min, hardware_max),
            }
        )
    return rows


def evaluate(
    model: ScheduleFreeModel,
    examples: list[Example],
    *,
    device: str = "cpu",
    hardware_min: list[float] | None = None,
    hardware_max: list[float] | None = None,
) -> dict[str, object]:
    rows = predict_examples(
        model,
        examples,
        device=device,
        hardware_min=hardware_min,
        hardware_max=hardware_max,
    )
    overall = metrics(
        [float(row["actual_us"]) for row in rows],
        [float(row["predicted_us"]) for row in rows],
        [float(row["sigma_log"]) for row in rows],
    )
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["workload_family"])].append(row)
    per_family = {}
    for family, family_rows in sorted(grouped.items()):
        per_family[family] = asdict(
            metrics(
                [float(row["actual_us"]) for row in family_rows],
                [float(row["predicted_us"]) for row in family_rows],
                [float(row["sigma_log"]) for row in family_rows],
            )
        )
    roofline = metrics(
        [example.record.latency_us for example in examples],
        [naive_roofline(example) for example in examples],
        [0.0] * len(examples),
    )
    return {
        "overall": asdict(overall),
        "naive_roofline": asdict(roofline),
        "per_workload_family": per_family,
        "ood_count": sum(bool(row["ood"]["is_ood"]) for row in rows),
        "predictions": rows,
    }


def hardware_swap_predictions(
    model: ScheduleFreeModel,
    graph,
    hardware_specs: list[HardwareSpec],
    device: str = "cpu",
) -> list[dict[str, float | str]]:
    graphs = [graph] * len(hardware_specs)
    model.eval()
    with torch.no_grad():
        prediction = model(
            batch_graphs(graphs).to(device),
            hardware_tensor(hardware_specs).to(device),
            physics_tensor(graphs, hardware_specs).to(device),
        )
    return [
        {"hardware_id": spec.hardware_id, "predicted_us": float(prediction.latency_us[index])}
        for index, spec in enumerate(hardware_specs)
    ]

