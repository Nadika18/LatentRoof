# Data protocol

## Canonical sources

The converted manifest uses only:

- balanced RTX PRO 6000 measurements;
- balanced H200 measurements;
- the larger cleaned GB10 dataset.

The older GB10 subset and unbalanced RTX dataset are excluded to avoid
duplicate device/workload measurements.

`artifacts/audit.json` records exact source paths, row counts, filtering,
hardware coverage, workload-family coverage, StableHLO consistency, and
warnings. Original files are read-only.

## Conversion rules

A row is retained only when:

- latency and coefficient of variation are finite and non-negative;
- latency is positive;
- CV is at most the configured threshold;
- the matching StableHLO file exists and parses;
- hardware identity is known;
- the device/workload pair has not already appeared.

Every manifest record stores a StableHLO SHA-256 digest and full provenance.
The current audit found one duplicate device/workload row and no cross-device
StableHLO hash mismatches.

## Privileged training labels

XLA-derived values are never inference inputs. Available values are masked
training targets for latent heads, including:

- fusion ratio and fusion count;
- kernel count;
- post-optimization DRAM bytes;
- buffer and peak-memory statistics;
- precision fractions;
- achieved throughput.

Missing labels contribute zero auxiliary loss.

## Valid splits

- Leave one complete GPU out for the cross-GPU claim.
- Group workload IDs across training/validation partitions.
- Optionally hold out compiler versions when reliable version provenance is
  collected.

Random row splits are invalid because the same workload configuration is
measured on multiple GPUs and neighboring shape configurations are highly
related.

