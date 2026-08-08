"""Versioned contracts shared by auditing, training, and inference."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"
HARDWARE_FEATURE_DIM = 20


class ContractError(ValueError):
    """Raised when a record would make evaluation scientifically invalid."""


@dataclass(frozen=True)
class HardwareSpec:
    hardware_id: str
    family: str
    fp32_tflops: float
    fp16_tflops: float
    bf16_tflops: float
    tf32_tflops: float
    int8_tops: float
    compute_units: int
    memory_bandwidth_gbps: float
    memory_capacity_gb: float
    l2_cache_mb: float
    local_memory_kb: float
    max_threads_per_unit: int
    execution_width: int
    launch_overhead_us: float
    estimated_fields: tuple[str, ...] = ()
    source: str = ""

    def validate(self) -> None:
        values = (
            self.fp32_tflops,
            self.fp16_tflops,
            self.bf16_tflops,
            self.tf32_tflops,
            self.int8_tops,
            self.compute_units,
            self.memory_bandwidth_gbps,
            self.memory_capacity_gb,
            self.l2_cache_mb,
            self.local_memory_kb,
            self.max_threads_per_unit,
            self.execution_width,
        )
        if not self.hardware_id or not self.family:
            raise ContractError("hardware_id and family are required")
        if any(not math.isfinite(float(value)) or value <= 0 for value in values):
            raise ContractError(f"hardware {self.hardware_id} has invalid resources")
        if self.launch_overhead_us < 0 or not math.isfinite(self.launch_overhead_us):
            raise ContractError("launch overhead must be finite and non-negative")

    def cuda_peak_tflops(self) -> float:
        """CUDA-core / non-Tensor-Core peak (FP32 CUDA cores)."""

        return self.fp32_tflops

    def tensor_peak_tflops(self, dtype: str) -> float:
        """Tensor-Core peak for contraction math in the given dtype."""

        dtype = dtype.lower()
        if dtype == "f32":
            return self.tf32_tflops
        if dtype == "f16":
            return self.fp16_tflops
        if dtype == "bf16":
            return self.bf16_tflops
        return self.tf32_tflops

    def peak_tflops(self, dtype: str) -> float:
        """Legacy single-peak API (dtype → one number). Prefer effective_peak_tflops."""

        dtype = dtype.lower()
        if dtype == "f32":
            return self.fp32_tflops
        if dtype == "f16":
            return self.fp16_tflops
        if dtype == "bf16":
            return self.bf16_tflops
        return self.fp32_tflops

    def effective_peak_tflops(
        self,
        *,
        dtype: str,
        contraction_flops: float,
        other_flops: float,
    ) -> float:
        """FLOP-weighted dual-peak: contractions→TC/TF32, other→CUDA.

        Returns P_eff such that total_flops / P_eff equals
        contraction_flops / P_tc + other_flops / P_cuda.
        """

        total = float(contraction_flops) + float(other_flops)
        if total <= 0 or not math.isfinite(total):
            return self.cuda_peak_tflops()
        p_tc = max(self.tensor_peak_tflops(dtype), 1e-12)
        p_cuda = max(self.cuda_peak_tflops(), 1e-12)
        seconds = float(contraction_flops) / p_tc + float(other_flops) / p_cuda
        if seconds <= 0 or not math.isfinite(seconds):
            return self.cuda_peak_tflops()
        return total / seconds

    def vector(self) -> list[float]:
        """Resource-only vector; product identity and family are excluded."""

        self.validate()
        log = lambda value: math.log1p(float(value))
        return [
            log(self.fp32_tflops),
            log(self.fp16_tflops),
            log(self.bf16_tflops),
            log(self.int8_tops),
            log(self.compute_units),
            log(self.memory_bandwidth_gbps),
            log(self.memory_capacity_gb),
            log(self.l2_cache_mb),
            log(self.local_memory_kb),
            log(self.max_threads_per_unit),
            log(self.execution_width),
            log(self.launch_overhead_us),
            log(self.fp32_tflops / self.memory_bandwidth_gbps),
            log(self.fp16_tflops / self.memory_bandwidth_gbps),
            log(self.memory_bandwidth_gbps / self.compute_units),
            log(self.l2_cache_mb / self.compute_units),
            log(self.local_memory_kb / self.compute_units),
            float(bool(self.estimated_fields)),
            len(self.estimated_fields) / 10.0,
            1.0,
        ]


@dataclass
class MeasurementRecord:
    record_id: str
    workload_id: str
    workload_family: str
    hardware_id: str
    source_dataset: str
    stablehlo_path: str
    stablehlo_sha256: str
    latency_us: float
    latency_cv_percent: float
    config: dict[str, Any]
    privileged_labels: dict[str, float | str | None] = field(default_factory=dict)
    compiler: dict[str, str] = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ContractError(f"unsupported schema version {self.schema_version}")
        if not all(
            (
                self.record_id,
                self.workload_id,
                self.workload_family,
                self.hardware_id,
                self.source_dataset,
                self.stablehlo_path,
                self.stablehlo_sha256,
            )
        ):
            raise ContractError("record identity and provenance are required")
        if self.latency_us <= 0 or not math.isfinite(self.latency_us):
            raise ContractError("latency must be finite and positive")
        if self.latency_cv_percent < 0 or not math.isfinite(self.latency_cv_percent):
            raise ContractError("latency CV must be finite and non-negative")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_hardware(path: str | Path) -> HardwareSpec:
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["estimated_fields"] = tuple(payload.get("estimated_fields", ()))
    spec = HardwareSpec(**payload)
    spec.validate()
    return spec


def write_hardware(spec: HardwareSpec, path: str | Path) -> None:
    spec.validate()
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(asdict(spec), handle, indent=2, sort_keys=True)
        handle.write("\n")


def record_from_dict(payload: dict[str, Any]) -> MeasurementRecord:
    record = MeasurementRecord(**payload)
    record.validate()
    return record


def load_manifest(path: str | Path) -> list[MeasurementRecord]:
    records: list[MeasurementRecord] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(record_from_dict(json.loads(line)))
            except (TypeError, KeyError, ValueError, json.JSONDecodeError) as error:
                raise ContractError(f"{path}:{line_number}: {error}") from error
    if not records:
        raise ContractError(f"{path} contains no records")
    return records


def write_manifest(records: list[MeasurementRecord], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        for record in records:
            record.validate()
            handle.write(json.dumps(asdict(record), sort_keys=True))
            handle.write("\n")

