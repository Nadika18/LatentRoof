# Literature basis

- [Habitat, USENIX ATC 2021](https://www.usenix.org/conference/atc21/presentation/yu)
  conditions operation predictors on GPU resources, but later evaluations show
  weak extrapolation to substantially newer GPUs and shapes.
- [A Learned Performance Model for TPUs, MLSys
  2021](https://proceedings.mlsys.org/paper/2021/hash/6bcfac823d40046dca25ef6d6d59cc3f-Abstract.html)
  demonstrates graph neural cost models and training-only compiler labels, but
  predicts optimized TPU kernels rather than schedule-free cross-family
  StableHLO.
- [NeuSight, ASPLOS
  2025](https://doi.org/10.1145/3669940.3707265) provides the closest precedent:
  it forecasts held-out GPUs using operator metadata, hardware specifications,
  tile behavior learned from older GPUs, and physical performance bounds.
  NeuSight estimates unknown tile sizes from a database of similar profiled
  kernels, so it models likely library behavior rather than recovering an exact
  unseen schedule.
- [CDMPP, EuroSys
  2024](https://doi.org/10.1145/3627703.3629572) addresses cross-device tensor
  program latency with compact program representations and domain adaptation.
  Its input is a scheduled tensor program, which is a different inference
  contract from raw StableHLO.
- [PipeWeave, ISCA
  2026](https://arxiv.org/abs/2601.14910) combines per-pipeline analytical
  demand with a learned interaction model across NVIDIA GPUs. It supports the
  hybrid analytical/learned direction but remains schedule-aware and
  GPU-specific.
- [StableHLO cross-architecture methodology, ISPASS
  2026](https://arxiv.org/abs/2604.12090) supports StableHLO as a unified
  workload representation across modeling fidelities. It also states the key
  limitation: analytical paths without target compilation omit
  target-specific compiler optimizations.

The resulting design predicts latent execution behavior from StableHLO and
hardware resources, then applies physical bounds and uncertainty. It does not
use an LLM and does not claim that hidden schedules are identifiable.

