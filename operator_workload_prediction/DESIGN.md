# Design

## Scientific question

Can a model trained on measured workloads from available GPUs estimate the
latency produced by the historical JAX/XLA software stack on a GPU that was
completely absent from training, using only StableHLO and public hardware
resources?

The model estimates an expected software-stack outcome. It does not claim to
recover an unknowable exact target schedule.

## Inputs

The StableHLO frontend creates real nodes for function arguments, preserves
their edges, supports private helper functions and calls, and extracts:

- operation semantics and graph structure;
- static shapes and dtypes;
- approximate operation counts;
- logical tensor traffic;
- a conservative input/output byte floor.

The hardware encoder receives numerical resources only:

- typed peak throughput;
- compute-unit count;
- memory bandwidth and capacity;
- L2 and local-memory capacity;
- execution width and thread capacity;
- launch overhead.

Product names and architecture-family strings are not neural-model features.

## Latent-physics model

The GNN and hardware encoder jointly estimate latent behavior:

- effective DRAM traffic;
- fusion ratio;
- number of kernels;
- compute utilization;
- memory utilization;
- execution-wave factor.

Where compiler labels exist, they supervise these heads during training.
Missing labels are masked. At inference, all latent values are predicted.

The analytical layer computes:

```text
compute_floor = FLOPs / typed_peak_throughput
memory_floor  = minimum_IO_bytes / memory_bandwidth
lower_bound   = max(compute_floor, memory_floor) + launch_overhead
```

Predicted utilization, DRAM traffic, kernel count, and waves produce a
physics-informed estimate. A non-negative learned residual cannot move the
prediction below the conservative lower bound. A separate head predicts
log-space uncertainty.

## Baselines

1. Naive roofline using StableHLO logical bytes.
2. Graph-only GNN.
3. Hardware-conditioned GNN without physics.
4. Latent-physics model.

All models use the same leakage-safe workload groups and leave-one-GPU-out
test sets.

## Evaluation

Primary tests hold out every measurement from one GPU. Workload groups are
also separated between training and validation. Reports include median/p90
absolute percentage error, log-space R², Spearman correlation, interval
coverage, workload-family breakdowns, OOD status, and hardware-swap
sensitivity.

