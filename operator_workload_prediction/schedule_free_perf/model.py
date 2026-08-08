"""GNN predictors with training-only latent execution supervision."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .contracts import HARDWARE_FEATURE_DIM, HardwareSpec
from .stablehlo import NODE_FEATURE_DIM, SemanticGraph

ModelMode = Literal["graph_only", "hardware_gnn", "latent_physics"]
LATENT_LABELS = (
    "log_dram_bytes",
    "fused_op_ratio",
    "log_n_kernels",
    "compute_utilization",
    "memory_utilization",
)


@dataclass
class GraphBatch:
    nodes: Tensor
    adjacency: Tensor
    mask: Tensor
    graph_scalars: Tensor

    def to(self, device: str | torch.device) -> "GraphBatch":
        return GraphBatch(
            self.nodes.to(device),
            self.adjacency.to(device),
            self.mask.to(device),
            self.graph_scalars.to(device),
        )


def graph_scalars(graph: SemanticGraph) -> list[float]:
    counts = {
        "contraction": sum(node.opcode in {"dot", "dot_general", "convolution"} for node in graph.nodes),
        "reduction": sum(node.opcode in {"reduce", "reduce_window"} for node in graph.nodes),
        "movement": sum(
            node.opcode
            in {
                "broadcast_in_dim",
                "concatenate",
                "gather",
                "pad",
                "reshape",
                "scatter",
                "slice",
                "transpose",
            }
            for node in graph.nodes
        ),
    }
    return [
        math.log1p(graph.total_flops),
        math.log1p(graph.logical_bytes),
        math.log1p(graph.minimum_io_bytes),
        math.log1p(len(graph.nodes)),
        math.log1p(len(graph.edges)),
        math.log1p(counts["contraction"]),
        math.log1p(counts["reduction"]),
        math.log1p(counts["movement"]),
    ]


def batch_graphs(graphs: list[SemanticGraph]) -> GraphBatch:
    if not graphs:
        raise ValueError("at least one graph is required")
    encoded = [graph.feature_tensors() for graph in graphs]
    max_nodes = max(len(features) for features, _ in encoded)
    nodes = torch.zeros((len(graphs), max_nodes, NODE_FEATURE_DIM), dtype=torch.float32)
    adjacency = torch.zeros((len(graphs), max_nodes, max_nodes), dtype=torch.float32)
    mask = torch.zeros((len(graphs), max_nodes), dtype=torch.bool)
    for index, (features, edges) in enumerate(encoded):
        count = len(features)
        nodes[index, :count] = torch.tensor(features)
        adjacency[index, :count, :count] = torch.tensor(edges)
        mask[index, :count] = True
    return GraphBatch(
        nodes=nodes,
        adjacency=adjacency,
        mask=mask,
        graph_scalars=torch.tensor([graph_scalars(graph) for graph in graphs]),
    )


class GraphEncoder(nn.Module):
    def __init__(self, hidden_dim: int, steps: int = 3) -> None:
        super().__init__()
        self.input = nn.Linear(NODE_FEATURE_DIM, hidden_dim)
        self.updates = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * 3, hidden_dim * 2),
                    nn.SiLU(),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                )
                for _ in range(steps)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(steps)])
        self.output = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 8, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

    def forward(self, batch: GraphBatch) -> Tensor:
        mask = batch.mask.to(batch.nodes.dtype).unsqueeze(-1)
        edge_mask = mask * mask.transpose(1, 2)
        edges = batch.adjacency * edge_mask
        hidden = self.input(batch.nodes) * mask
        for update, norm in zip(self.updates, self.norms):
            incoming = torch.bmm(edges, hidden) / edges.sum(2, keepdim=True).clamp_min(1.0)
            reverse = edges.transpose(1, 2)
            outgoing = torch.bmm(reverse, hidden) / reverse.sum(2, keepdim=True).clamp_min(1.0)
            hidden = norm(hidden + update(torch.cat((hidden, incoming, outgoing), -1))) * mask
        mean = hidden.sum(1) / mask.sum(1).clamp_min(1.0)
        maximum = hidden.masked_fill(~batch.mask.unsqueeze(-1), -torch.inf).amax(1)
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        return self.output(torch.cat((mean, maximum, batch.graph_scalars), -1))


@dataclass
class Prediction:
    log_latency_mean: Tensor
    log_std: Tensor
    latency_us: Tensor
    physics_us: Tensor
    lower_bound_us: Tensor
    auxiliary: dict[str, Tensor]


class ScheduleFreeModel(nn.Module):
    """Predict expected stack latency from only semantic graph and hardware resources."""

    def __init__(
        self,
        mode: ModelMode = "latent_physics",
        hidden_dim: int = 128,
        message_steps: int = 3,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.hidden_dim = hidden_dim
        self.message_steps = message_steps
        self.graph_encoder = GraphEncoder(hidden_dim, message_steps)
        self.hardware_encoder = nn.Sequential(
            nn.Linear(HARDWARE_FEATURE_DIM, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        fusion_inputs = hidden_dim if mode == "graph_only" else hidden_dim * 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_inputs, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.raw_latency = nn.Linear(hidden_dim, 1)
        self.log_std_head = nn.Linear(hidden_dim, 1)
        self.latent_head = nn.Linear(hidden_dim, 6)
        self.residual_head = nn.Linear(hidden_dim, 1)
        with torch.no_grad():
            self.latent_head.bias.copy_(torch.tensor([-4.0, 0.0, 0.0, 2.0, 2.0, -3.0]))

    def forward(
        self,
        graph: GraphBatch,
        hardware_features: Tensor,
        physics_inputs: Tensor,
    ) -> Prediction:
        graph_embedding = self.graph_encoder(graph)
        if self.mode == "graph_only":
            fused = self.fusion(graph_embedding)
        else:
            hardware_embedding = self.hardware_encoder(hardware_features)
            fused = self.fusion(torch.cat((graph_embedding, hardware_embedding), -1))

        log_std = -5.0 + 7.0 * torch.sigmoid(self.log_std_head(fused).squeeze(-1))
        if self.mode != "latent_physics":
            log_latency = self.raw_latency(fused).squeeze(-1)
            latency = torch.exp(log_latency)
            return Prediction(
                log_latency_mean=log_latency,
                log_std=log_std,
                latency_us=latency,
                physics_us=torch.zeros_like(latency),
                lower_bound_us=torch.zeros_like(latency),
                auxiliary={},
            )

        # physics_inputs: FLOPs, minimum bytes, logical bytes, peak TFLOP/s,
        # bandwidth GB/s, launch overhead us.
        flops, min_bytes, logical_bytes, peak, bandwidth, launch = physics_inputs.unbind(1)
        compute_floor = flops / (peak * 1e6).clamp_min(1e-12)
        memory_floor = min_bytes / (bandwidth * 1e3).clamp_min(1e-12)
        lower_bound = torch.maximum(compute_floor, memory_floor) + launch

        latent = self.latent_head(fused)
        dram_fraction = torch.sigmoid(latent[:, 0])
        predicted_dram = min_bytes + dram_fraction * torch.relu(logical_bytes - min_bytes)
        fused_ratio = torch.sigmoid(latent[:, 1])
        kernel_count = 1.0 + F.softplus(latent[:, 2])
        compute_util = 0.01 + 0.98 * torch.sigmoid(latent[:, 3])
        memory_util = 0.01 + 0.98 * torch.sigmoid(latent[:, 4])
        wave_factor = 1.0 + F.softplus(latent[:, 5])
        compute_time = compute_floor / compute_util
        memory_time = predicted_dram / (bandwidth * 1e3).clamp_min(1e-12) / memory_util
        execution = torch.maximum(compute_time, memory_time) * wave_factor
        launch_time = kernel_count * launch
        physics = torch.maximum(execution + launch_time, lower_bound)
        latency = physics + F.softplus(self.residual_head(fused).squeeze(-1))
        return Prediction(
            log_latency_mean=torch.log(latency.clamp_min(1e-9)),
            log_std=log_std,
            latency_us=latency,
            physics_us=physics,
            lower_bound_us=lower_bound,
            auxiliary={
                "log_dram_bytes": torch.log(predicted_dram.clamp_min(1.0)),
                "fused_op_ratio": fused_ratio,
                "log_n_kernels": torch.log(kernel_count),
                "compute_utilization": compute_util,
                "memory_utilization": memory_util,
                "wave_factor": wave_factor,
            },
        )


def hardware_tensor(specs: list[HardwareSpec]) -> Tensor:
    return torch.tensor([spec.vector() for spec in specs], dtype=torch.float32)


def physics_tensor(graphs: list[SemanticGraph], specs: list[HardwareSpec]) -> Tensor:
    rows = []
    for graph, spec in zip(graphs, specs):
        peak = spec.effective_peak_tflops(
            dtype=graph.dtype,
            contraction_flops=graph.contraction_flops(),
            other_flops=graph.other_flops(),
        )
        rows.append(
            [
                graph.total_flops,
                graph.minimum_io_bytes,
                graph.logical_bytes,
                peak,
                spec.memory_bandwidth_gbps,
                spec.launch_overhead_us,
            ]
        )
    return torch.tensor(rows, dtype=torch.float32)


def model_config(model: ScheduleFreeModel) -> dict[str, Any]:
    return {
        "mode": model.mode,
        "hidden_dim": model.hidden_dim,
        "message_steps": model.message_steps,
    }

