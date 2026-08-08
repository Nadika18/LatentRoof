# Limitations and claim boundary

## What can be claimed

After leave-one-GPU-out evaluation, the system may claim cross-GPU forecasting
for the measured workload and hardware distributions. It can report expected
latency, uncertainty, workload ranking, and whether hardware resources are
outside the training range.

## What cannot be claimed

- StableHLO and public specifications do not uniquely determine an exact
  compiler schedule.
- GPU-only results do not establish transfer to TPU, Trainium, Inferentia, or
  unrelated accelerators.
- A hardware descriptor with estimated throughput values is not ground truth.
- Good random-split accuracy is not evidence of unseen-device prediction.
- The naive StableHLO logical-byte sum is not real DRAM traffic.
- An OOD warning does not make an extrapolated prediction accurate.

The historical datasets do not record a reliable compiler version, so the
current target is the aggregate behavior of the stack that produced those
measurements. New collection must record JAX, XLA, CUDA/driver, library,
firmware, clocks, power state, and concurrency.

## Expansion beyond GPUs

TPU and Trainium should be added only after collecting family-specific
measurements. A future system may share the StableHLO semantic encoder and
hardware resource ontology, but each architecture family requires an
execution expert validated with complete family holdouts. No universal
schedule-free transfer claim should be made without that evidence.

