from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from schedule_free_perf.cli import main
from schedule_free_perf.contracts import MeasurementRecord, file_sha256
from schedule_free_perf.data import Example, grouped_split, load_hardware_catalog, make_batch
from schedule_free_perf.evaluation import evaluate, hardware_swap_predictions
from schedule_free_perf.losses import combined_loss
from schedule_free_perf.model import NODE_FEATURE_DIM, ScheduleFreeModel
from schedule_free_perf.stablehlo import parse_stablehlo, parse_stablehlo_file
from schedule_free_perf.training import (
    TrainingConfig,
    save_checkpoint,
    train_model,
)

CODE_ROOT = Path(__file__).parents[1]
PROJECT_ROOT = CODE_ROOT.parent
SAMPLE_HLO = (
    PROJECT_ROOT
    / "data/h200_balanced_20260723_174421/graphs/"
    "softmax_batch16_f32_hidden2048_seq2048.stablehlo.txt"
)


def examples() -> list[Example]:
    graph = parse_stablehlo_file(SAMPLE_HLO, "softmax-smoke")
    catalog = load_hardware_catalog(CODE_ROOT / "hardware")
    rows = []
    for index, hardware in enumerate(catalog.values()):
        record = MeasurementRecord(
            record_id=f"smoke-{index}",
            workload_id="softmax-smoke",
            workload_family="softmax",
            hardware_id=hardware.hardware_id,
            source_dataset="test",
            stablehlo_path=str(SAMPLE_HLO),
            stablehlo_sha256=file_sha256(SAMPLE_HLO),
            latency_us=(300.0, 180.0, 700.0)[index],
            latency_cv_percent=1.0,
            config={},
            privileged_labels={
                "label_dram_bytes": graph.minimum_io_bytes * 1.2,
                "label_fused_op_ratio": 0.5,
                "label_n_kernels": 3,
                "achieved_tflops": 1.0,
            },
        )
        rows.append(Example(record, graph, hardware))
    return rows


class FrontendTests(unittest.TestCase):
    def test_arguments_are_real_nodes_and_edges(self) -> None:
        graph = parse_stablehlo_file(SAMPLE_HLO)
        features, _ = graph.feature_tensors()
        self.assertEqual(len(features[0]), NODE_FEATURE_DIM)
        self.assertTrue(any(node.is_argument for node in graph.nodes))
        self.assertTrue(
            any(graph.nodes[source].is_argument for source, _ in graph.edges),
            "function arguments must participate in graph edges",
        )

    def test_private_functions_and_calls_are_supported(self) -> None:
        source = """
        module {
          func.func public @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
            %0 = call @helper(%arg0) : (tensor<4xf32>) -> tensor<4xf32>
            return %0 : tensor<4xf32>
          }
          func.func private @helper(%arg0: tensor<4xf32>) -> tensor<4xf32> {
            %0 = chlo.square %arg0 : tensor<4xf32> -> tensor<4xf32>
            return %0 : tensor<4xf32>
          }
        }
        """
        graph = parse_stablehlo(source)
        self.assertTrue(any(node.opcode == "call" for node in graph.nodes))
        self.assertTrue(any(node.opcode == "chlo.square" for node in graph.nodes))


class DataAndModelTests(unittest.TestCase):
    def test_grouped_split_has_no_workload_leakage(self) -> None:
        base = examples()
        expanded = []
        for index in range(20):
            item = base[index % len(base)]
            record = MeasurementRecord(**item.record.__dict__)
            record.record_id = f"row-{index}"
            record.workload_id = f"workload-{index // 2}"
            expanded.append(Example(record, item.graph, item.hardware))
        split = grouped_split(expanded, seed=3)
        split.assert_no_leakage()

    def test_model_bounds_auxiliary_loss_and_hardware_sensitivity(self) -> None:
        rows = examples()
        batch = make_batch(rows)
        model = ScheduleFreeModel(hidden_dim=16, message_steps=1)
        prediction = model(batch.graph, batch.hardware, batch.physics)
        self.assertTrue(torch.all(prediction.latency_us >= prediction.lower_bound_us))
        loss = combined_loss(
            prediction,
            batch.target_log_latency,
            batch.auxiliary_targets,
            batch.auxiliary_masks,
        )
        loss.total.backward()
        self.assertTrue(torch.isfinite(loss.total))
        swapped = hardware_swap_predictions(model, rows[0].graph, [row.hardware for row in rows])
        self.assertGreater(len({round(float(row["predicted_us"]), 4) for row in swapped}), 1)


class EndToEndTests(unittest.TestCase):
    checkpoint = CODE_ROOT / "artifacts/test_checkpoint.pt"

    @classmethod
    def setUpClass(cls) -> None:
        rows = examples()
        config = TrainingConfig(
            hidden_dim=16,
            message_steps=1,
            epochs=2,
            batch_size=3,
            patience=2,
        )
        result = train_model(rows, rows, config)
        save_checkpoint(result, config, cls.checkpoint)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.checkpoint.unlink(missing_ok=True)

    def test_training_and_evaluation(self) -> None:
        rows = examples()
        config = TrainingConfig(hidden_dim=16, message_steps=1, epochs=1, batch_size=3)
        result = train_model(rows, rows, config)
        report = evaluate(
            result.model,
            rows,
            hardware_min=result.hardware_min,
            hardware_max=result.hardware_max,
        )
        self.assertEqual(report["overall"]["count"], 3)
        self.assertIn("naive_roofline", report)

    def test_predict_cli_never_starts_external_process(self) -> None:
        hardware = CODE_ROOT / "hardware/h200.json"

        def forbidden(*args, **kwargs):
            raise AssertionError(f"external process attempted: {args!r} {kwargs!r}")

        output = io.StringIO()
        with (
            patch.object(subprocess, "Popen", forbidden),
            patch.object(os, "system", forbidden),
            contextlib.redirect_stdout(output),
        ):
            main(
                [
                    "predict",
                    str(SAMPLE_HLO),
                    str(hardware),
                    str(self.checkpoint),
                ]
            )
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["inference_inputs"], ["stablehlo", "hardware_specs"])
        self.assertGreater(payload["predicted_us"], 0)


if __name__ == "__main__":
    unittest.main()

