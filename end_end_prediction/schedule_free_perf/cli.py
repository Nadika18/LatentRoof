"""Command-line workflows; prediction has no compiler, profiler, or hardware calls."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .audit import audit_and_convert, convert_to_files
from .contracts import load_hardware
from .data import (
    Example,
    grouped_split,
    leave_one_gpu_out,
    load_examples,
    load_hardware_catalog,
)
from .evaluation import evaluate, hardware_ood, hardware_swap_predictions
from .model import batch_graphs, hardware_tensor, physics_tensor
from .stablehlo import parse_stablehlo_file
from .training import TrainingConfig, load_checkpoint, save_checkpoint, train_model, write_history


def _safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    return value


def _print(value: Any) -> None:
    print(json.dumps(_safe(value), indent=2, sort_keys=True, allow_nan=False))


def command_audit(args: argparse.Namespace) -> None:
    audit, _ = audit_and_convert(args.datasets, max_cv_percent=args.max_cv)
    _print(asdict(audit))


def command_convert(args: argparse.Namespace) -> None:
    audit = convert_to_files(
        args.datasets,
        args.manifest,
        args.audit_output,
        max_cv_percent=args.max_cv,
    )
    _print(
        {
            "manifest": str(args.manifest),
            "audit": str(args.audit_output),
            "valid_rows": audit.valid_rows,
            "filtered_rows": audit.source_rows - audit.valid_rows,
        }
    )


def _split(args: argparse.Namespace) -> tuple[list[Example], list[Example], list[Example], list[str]]:
    examples, parse_errors = load_examples(
        args.manifest, args.hardware_dir, skip_parse_errors=args.skip_parse_errors
    )
    if args.held_out_hardware:
        split = leave_one_gpu_out(examples, args.held_out_hardware, args.seed)
    else:
        split = grouped_split(examples, key="workload", seed=args.seed)
    return split.train, split.validation, split.test, parse_errors


def command_train(args: argparse.Namespace) -> None:
    train, validation, test, parse_errors = _split(args)
    config = TrainingConfig(
        mode=args.mode,
        hidden_dim=args.hidden_dim,
        message_steps=args.message_steps,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        auxiliary_weight=args.auxiliary_weight,
        patience=args.patience,
        seed=args.seed,
        device=args.device,
    )
    result = train_model(train, validation, config)
    save_checkpoint(result, config, args.output)
    if args.history:
        write_history(result.history, args.history)
    _print(
        {
            "checkpoint": str(args.output),
            "mode": args.mode,
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
            "parse_errors": len(parse_errors),
            "final_epoch": result.history[-1],
        }
    )


def command_evaluate(args: argparse.Namespace) -> None:
    _, _, test, parse_errors = _split(args)
    model, checkpoint = load_checkpoint(args.checkpoint, args.device)
    report = evaluate(
        model,
        test,
        device=args.device,
        hardware_min=checkpoint.get("hardware_min"),
        hardware_max=checkpoint.get("hardware_max"),
    )
    report["parse_errors"] = parse_errors
    if args.output:
        Path(args.output).write_text(json.dumps(_safe(report), indent=2, sort_keys=True))
        _print({"output": str(args.output), "overall": report["overall"]})
    else:
        _print(report)


def command_compare(args: argparse.Namespace) -> None:
    _, _, test, parse_errors = _split(args)
    checkpoints = {
        "graph_only": args.graph_only,
        "hardware_gnn": args.hardware_gnn,
        "latent_physics": args.latent_physics,
    }
    comparison: dict[str, Any] = {
        "test_rows": len(test),
        "parse_errors": parse_errors,
        "models": {},
    }
    for name, path in checkpoints.items():
        model, checkpoint = load_checkpoint(path, args.device)
        report = evaluate(
            model,
            test,
            device=args.device,
            hardware_min=checkpoint.get("hardware_min"),
            hardware_max=checkpoint.get("hardware_max"),
        )
        comparison["models"][name] = report["overall"]
        if "naive_roofline" not in comparison:
            comparison["naive_roofline"] = report["naive_roofline"]
    if args.output:
        Path(args.output).write_text(
            json.dumps(_safe(comparison), indent=2, sort_keys=True), encoding="utf-8"
        )
    _print(comparison)


def command_predict(args: argparse.Namespace) -> None:
    graph = parse_stablehlo_file(args.stablehlo, args.graph_id)
    hardware = load_hardware(args.hardware)
    model, checkpoint = load_checkpoint(args.checkpoint, args.device)
    model.eval()
    with torch.no_grad():
        prediction = model(
            batch_graphs([graph]).to(args.device),
            hardware_tensor([hardware]).to(args.device),
            physics_tensor([graph], [hardware]).to(args.device),
        )
    sigma = math.exp(float(prediction.log_std[0].cpu()))
    mean = float(prediction.latency_us[0].cpu())
    _print(
        {
            "graph_id": graph.graph_id,
            "hardware_id": hardware.hardware_id,
            "predicted_us": mean,
            "interval_90_us": [
                math.exp(math.log(mean) - 1.64485363 * sigma),
                math.exp(math.log(mean) + 1.64485363 * sigma),
            ],
            "sigma_log": sigma,
            "physics_us": float(prediction.physics_us[0].cpu()),
            "lower_bound_us": float(prediction.lower_bound_us[0].cpu()),
            "ood": hardware_ood(
                hardware, checkpoint.get("hardware_min"), checkpoint.get("hardware_max")
            ),
            "inference_inputs": ["stablehlo", "hardware_specs"],
        }
    )


def command_hardware_swap(args: argparse.Namespace) -> None:
    graph = parse_stablehlo_file(args.stablehlo, args.graph_id)
    catalog = load_hardware_catalog(args.hardware_dir)
    model, _ = load_checkpoint(args.checkpoint, args.device)
    _print(
        hardware_swap_predictions(model, graph, list(catalog.values()), device=args.device)
    )


def _add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("manifest")
    parser.add_argument("--hardware-dir", required=True)
    parser.add_argument("--held-out-hardware")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-parse-errors", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="schedule-free-perf")
    commands = parser.add_subparsers(dest="command", required=True)

    audit = commands.add_parser("audit-data")
    audit.add_argument("datasets", nargs="+")
    audit.add_argument("--max-cv", type=float, default=20.0)
    audit.set_defaults(handler=command_audit)

    convert = commands.add_parser("convert-data")
    convert.add_argument("datasets", nargs="+")
    convert.add_argument("--manifest", required=True)
    convert.add_argument("--audit-output", required=True)
    convert.add_argument("--max-cv", type=float, default=20.0)
    convert.set_defaults(handler=command_convert)

    train = commands.add_parser("train")
    _add_dataset_arguments(train)
    train.add_argument("--output", required=True)
    train.add_argument("--history")
    train.add_argument(
        "--mode", choices=("graph_only", "hardware_gnn", "latent_physics"), default="latent_physics"
    )
    train.add_argument("--hidden-dim", type=int, default=128)
    train.add_argument("--message-steps", type=int, default=3)
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--auxiliary-weight", type=float, default=0.2)
    train.add_argument("--patience", type=int, default=8)
    train.add_argument("--device", default="cpu")
    train.set_defaults(handler=command_train)

    evaluate_parser = commands.add_parser("evaluate")
    _add_dataset_arguments(evaluate_parser)
    evaluate_parser.add_argument("checkpoint")
    evaluate_parser.add_argument("--output")
    evaluate_parser.add_argument("--device", default="cpu")
    evaluate_parser.set_defaults(handler=command_evaluate)

    compare = commands.add_parser("compare")
    _add_dataset_arguments(compare)
    compare.add_argument("--graph-only", required=True)
    compare.add_argument("--hardware-gnn", required=True)
    compare.add_argument("--latent-physics", required=True)
    compare.add_argument("--output")
    compare.add_argument("--device", default="cpu")
    compare.set_defaults(handler=command_compare)

    predict = commands.add_parser("predict")
    predict.add_argument("stablehlo")
    predict.add_argument("hardware")
    predict.add_argument("checkpoint")
    predict.add_argument("--graph-id")
    predict.add_argument("--device", default="cpu")
    predict.set_defaults(handler=command_predict)

    swap = commands.add_parser("hardware-swap")
    swap.add_argument("stablehlo")
    swap.add_argument("hardware_dir")
    swap.add_argument("checkpoint")
    swap.add_argument("--graph-id")
    swap.add_argument("--device", default="cpu")
    swap.set_defaults(handler=command_hardware_swap)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()

