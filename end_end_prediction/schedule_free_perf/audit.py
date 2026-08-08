"""Read-only auditing and conversion of the legacy measurement datasets."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .contracts import MeasurementRecord, file_sha256, write_manifest
from .stablehlo import parse_stablehlo_file

GPU_NAME_MAP = {
    "NVIDIA GB10": "nvidia_gb10",
    "NVIDIA H200 NVL": "nvidia_h200",
    "NVIDIA H200": "nvidia_h200",
    "NVIDIA RTX PRO 6000 Blackwell Server Edition": "nvidia_rtx_pro_6000",
}

PRIVILEGED_LABELS = (
    "label_dram_bytes",
    "label_fused_op_ratio",
    "label_n_fusions",
    "label_n_kernels",
    "label_ops_before",
    "label_ops_after",
    "label_max_ops_per_fusion",
    "label_mean_ops_per_fusion",
    "label_peak_memory_bytes",
    "label_buffer_reuse_ratio",
    "label_n_logical_buffers",
    "label_n_physical_buffers",
    "label_compute_precision",
    "label_precision_f32_fraction",
    "label_precision_f16_fraction",
    "label_precision_bf16_fraction",
    "achieved_tflops",
)


@dataclass
class DatasetAudit:
    source_rows: int = 0
    valid_rows: int = 0
    invalid_latency: int = 0
    excessive_cv: int = 0
    missing_hlo: int = 0
    stablehlo_parse_errors: int = 0
    unknown_hardware: int = 0
    duplicate_device_workloads: int = 0
    cross_hardware_hlo_mismatches: int = 0
    rows_by_hardware: dict[str, int] = field(default_factory=dict)
    rows_by_workload: dict[str, int] = field(default_factory=dict)
    shared_workloads_across_devices: int = 0
    source_datasets: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _load_rows(dataset_path: Path) -> list[dict[str, Any]]:
    with dataset_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"{dataset_path} must contain a JSON list")
    return payload


def audit_and_convert(
    dataset_paths: list[str | Path],
    *,
    max_cv_percent: float = 20.0,
) -> tuple[DatasetAudit, list[MeasurementRecord]]:
    """Audit sources in order; the first duplicate device/workload row wins."""

    audit = DatasetAudit(source_datasets=[str(Path(path).resolve()) for path in dataset_paths])
    records: list[MeasurementRecord] = []
    seen_device_workloads: set[tuple[str, str]] = set()
    hashes_by_workload: dict[str, set[str]] = defaultdict(set)
    devices_by_workload: dict[str, set[str]] = defaultdict(set)

    for raw_path in dataset_paths:
        dataset_path = Path(raw_path).resolve()
        dataset_dir = dataset_path.parent
        for row in _load_rows(dataset_path):
            audit.source_rows += 1
            experiment_id = str(row.get("experiment_id", "")).strip()
            gpu_name = str(row.get("gpu_name", "")).strip()
            hardware_id = GPU_NAME_MAP.get(gpu_name)
            if hardware_id is None:
                audit.unknown_hardware += 1
                continue

            try:
                latency = float(row.get("latency_us"))
                cv = float(row.get("latency_cv_percent", 0.0))
            except (TypeError, ValueError):
                audit.invalid_latency += 1
                continue
            if latency <= 0 or not math.isfinite(latency) or cv < 0 or not math.isfinite(cv):
                audit.invalid_latency += 1
                continue
            if cv > max_cv_percent:
                audit.excessive_cv += 1
                continue

            hlo_path = dataset_dir / "graphs" / f"{experiment_id}.stablehlo.txt"
            if not experiment_id or not hlo_path.is_file():
                audit.missing_hlo += 1
                continue
            try:
                parse_stablehlo_file(hlo_path, experiment_id)
            except (OSError, ValueError):
                audit.stablehlo_parse_errors += 1
                continue
            duplicate_key = (hardware_id, experiment_id)
            if duplicate_key in seen_device_workloads:
                audit.duplicate_device_workloads += 1
                continue
            seen_device_workloads.add(duplicate_key)

            hlo_hash = file_sha256(hlo_path)
            hashes_by_workload[experiment_id].add(hlo_hash)
            devices_by_workload[experiment_id].add(hardware_id)
            privileged = {key: row.get(key) for key in PRIVILEGED_LABELS}
            workload_family = str(row.get("workload", "unknown"))
            records.append(
                MeasurementRecord(
                    record_id=f"{hardware_id}:{experiment_id}",
                    workload_id=experiment_id,
                    workload_family=workload_family,
                    hardware_id=hardware_id,
                    source_dataset=str(dataset_path),
                    stablehlo_path=str(hlo_path),
                    stablehlo_sha256=hlo_hash,
                    latency_us=latency,
                    latency_cv_percent=cv,
                    config=dict(row.get("config", {})),
                    privileged_labels=privileged,
                    compiler={"stack": "jax_xla", "version": "unknown"},
                )
            )

    audit.valid_rows = len(records)
    audit.cross_hardware_hlo_mismatches = sum(len(hashes) > 1 for hashes in hashes_by_workload.values())
    audit.shared_workloads_across_devices = sum(
        len(devices) > 1 for devices in devices_by_workload.values()
    )
    audit.rows_by_hardware = dict(Counter(record.hardware_id for record in records))
    audit.rows_by_workload = dict(Counter(record.workload_family for record in records))
    if audit.cross_hardware_hlo_mismatches:
        audit.warnings.append(
            "Some workload IDs have different StableHLO across devices; group by ID and inspect "
            "before interpreting paired-device errors."
        )
    if len(audit.rows_by_hardware) < 3:
        audit.warnings.append("Fewer than three GPUs remain after filtering.")
    if "nvidia_gb10" in audit.rows_by_hardware:
        audit.warnings.append(
            "GB10 measurements are distribution-shifted relative to datacenter GPUs; report "
            "results separately and retain uncertainty."
        )
    return audit, records


def write_audit(audit: DatasetAudit, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(asdict(audit), handle, indent=2, sort_keys=True)
        handle.write("\n")


def convert_to_files(
    dataset_paths: list[str | Path],
    manifest_path: str | Path,
    audit_path: str | Path,
    *,
    max_cv_percent: float = 20.0,
) -> DatasetAudit:
    audit, records = audit_and_convert(dataset_paths, max_cv_percent=max_cv_percent)
    write_manifest(records, manifest_path)
    write_audit(audit, audit_path)
    return audit

