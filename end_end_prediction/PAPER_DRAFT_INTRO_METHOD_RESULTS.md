# Draft: Introduction, Methodology, Results

Paste-ready text for poster / paper. Numbers match operator-level (block-level LOGO) and Ablation E (e2e GPT/BERT compose with amortized launch).

---

\section{Introduction}

Modern ML workloads execute across an increasingly diverse accelerator landscape: multiple GPU generations, domain-specific ASICs such as TPUs, Graphcore IPUs, Amazon Trainium, and Meta MTIA, as well as emerging near-memory, dataflow, and spatial architectures. Each class of device is optimized for a different performance regime and exposes a radically different microarchitecture. Decisions about where to place a model, how to size a cluster, or whether a new chip is worth acquiring ultimately reduce to one question: \emph{how fast will this workload run on that accelerator?}

The standard answer today is to compile the model and profile it on the target device. That procedure is accurate but expensive: it requires physical access to every candidate accelerator, a working software stack, and non-trivial compile-and-measure time. Pure analytical roofline models are cheap, but they systematically mispredict because they ignore fusion, utilization, and launch effects that dominate real stacks. Recent learned predictors close some of this gap, yet remain tightly coupled to GPU-centric execution contracts. NeuSight forecasts latency from operator metadata extracted from a PyTorch graph and public GPU resources. PipeWeave profiles kernel demand on SM instruction pipelines and trains per-kernel MLPs; its kernel decomposer and scheduler are GPU-specific. TPUGraphs conditions on compiler configurations rather than schedule-free IR. As hardware diversifies, compiling and profiling on every target becomes increasingly impractical, and GPU-only modeling assumptions do not transfer.

We therefore develop a \textbf{hardware-aware hybrid performance model} that estimates expected latency \emph{without executing on the target device at inference}. The model takes only a portable StableHLO computation graph and public hardware resource specifications. A graph neural network encodes the workload; an MLP encodes hardware; fused latents drive a dual-peak analytical roofline with a non-negative residual. Privileged compiler labels, when available, supervise a subset of latents during training and are never required at inference. We evaluate leave-one-GPU-out transfer across NVIDIA H200 NVL, RTX PRO 6000 Blackwell, and GB10, and further compose per-block predictions into end-to-end GPT-2 / BERT backbone latencies.

---

\section{Methodology}

\subsection{Input}

\textbf{Workload.} We represent each program as a StableHLO graph. StableHLO is a portable intermediate representation that can be lowered to GPUs, TPUs, and other accelerators, which makes it a natural contract for cross-device modeling. From each graph we extract node features (operation semantics, shapes, dtypes), graph structure (dataflow edges), and graph-level scalars (approximate FLOPs and logical bytes), including a conservative input/output byte floor. Product names and architecture-family strings are \emph{not} model features.

\textbf{Hardware.} The hardware encoder receives a numerical resource vector derived from public specifications only: typed peak throughputs (FP32 CUDA, TF32 / FP16 / BF16 Tensor), INT8 peak, compute-unit (SM) count, memory bandwidth and capacity, L2 and local memory, max threads per SM, execution width, launch overhead, and a small set of derived ratios. After $\log(1+x)$ transforms and ratios, this yields a 20-dimensional hardware feature vector.

At inference the predictor uses \textbf{exactly} StableHLO $+$ these public resources. It does not consume a target compiler schedule, profiler counters, or target-device execution.

\subsection{Data}

We measure StableHLO programs under the JAX/XLA stack on three NVIDIA platforms: H200 NVL (Hopper), RTX PRO 6000 (Blackwell), and GB10 (Grace--Blackwell). The operator / block suite comprises 11 families spanning linear algebra (GEMM, BatchMatMul), element-wise and normalization ops (GELU, Softmax, LayerNorm, Residual), and Transformer components (FFN, Attention, MHA, MLP-3, full Transformer block), in FP32 / FP16 / BF16, with varied batch, sequence, hidden, FFN, and matrix dimensions. The mainline corpus has 6{,}778 latency measurements from 2{,}593 unique configurations (2{,}292 unique graphs). end-to-end composition additionally adds GPT-2 / BERT-dimension coverage blocks for end-to-end composition.

Timing uses a fixed protocol: 4-buffer rotation, 10 warm-up iterations, 2 trials $\times$ 100 runs, IQR outlier filtering, and trial-median latency with coefficient of variation. Privileged XLA-derived quantities (DRAM bytes, fusion ratio, kernel count, achieved throughput) are retained as \emph{training-only} labels and are masked when missing.

\subsection{Model}

A message-passing GNN (embedding width $d{=}128$, two message steps) encodes the StableHLO graph. A two-layer MLP encodes the 20-D hardware vector. A fusion MLP produces a joint embedding that feeds:

\begin{itemize}
  \item a \textbf{latent head} predicting six continuous execution factors: effective DRAM traffic, fusion ratio, kernel count, compute utilization, memory utilization, and wave factor;
  \item a \textbf{residual head} predicting a non-negative additive correction (Softplus);
  \item an \textbf{uncertainty head} predicting log-space observation noise.
\end{itemize}

Latents are inserted into a dual-peak analytical roofline (contractions use Tensor/TF32 peaks; other FLOPs use CUDA FP32 peak) with a hard physics floor. Training is supervised on measured $\log$-latency via heteroscedastic Gaussian NLL. When privileged labels exist, masked MSE supervises DRAM bytes, fusion ratio, kernel count, and compute utilization ($\lambda{=}0.2$). Memory utilization and wave factor are used in the physics layer but have no direct labels (unsupervised for those heads). Optimization uses AdamW ($10^{-3}$ lr, $10^{-4}$ weight decay), batch size 64, up to 20 epochs with early stopping (patience 5), gradient clipping at 5.0, seed 29.

\subsection{Equations}

Dual-peak effective throughput $P_{\mathrm{eff}}$ weights contraction FLOPs on the Tensor-core peak $P_{\mathrm{tc}}$ and remaining FLOPs on the CUDA peak $P_{\mathrm{cuda}}$:
\[
\frac{F_{\mathrm{tot}}}{P_{\mathrm{eff}}}
=
\frac{F_{\mathrm{contr}}}{P_{\mathrm{tc}}}
+
\frac{F_{\mathrm{other}}}{P_{\mathrm{cuda}}}.
\]
Conservative floors and the hard lower bound are
\[
t_{\mathrm{comp}}^{\mathrm{floor}} = \frac{F_{\mathrm{tot}}}{P_{\mathrm{eff}}},\qquad
t_{\mathrm{mem}}^{\mathrm{floor}} = \frac{B_{\min}}{\mathrm{BW}},\qquad
t_{\mathrm{lb}} = \max\!\bigl(t_{\mathrm{comp}}^{\mathrm{floor}},\, t_{\mathrm{mem}}^{\mathrm{floor}}\bigr) + t_{\mathrm{launch}}.
\]
Predicted utilization, DRAM, kernels, and waves give the physics estimate
\[
t_{\mathrm{comp}} = \frac{t_{\mathrm{comp}}^{\mathrm{floor}}}{u_{\mathrm{comp}}},\qquad
t_{\mathrm{mem}} = \frac{B_{\mathrm{DRAM}}}{\mathrm{BW}\, u_{\mathrm{mem}}},\qquad
t_{\mathrm{exec}} = \max(t_{\mathrm{comp}}, t_{\mathrm{mem}})\, w,
\]
\[
t_{\mathrm{physics}} = \max\!\bigl(t_{\mathrm{exec}} + n_{\mathrm{kern}}\, t_{\mathrm{launch}},\, t_{\mathrm{lb}}\bigr),
\qquad
\hat{t} = t_{\mathrm{physics}} + r,\quad r \ge 0.
\]
The training objective is
\[
\mathcal{L}
=
\mathrm{NLL}\!\bigl(\log t_{\mathrm{meas}};\, \log\hat{t},\, \sigma\bigr)
+
\lambda\, \mathrm{MaskedMSE}(z, z^\*),
\quad \lambda = 0.2.
\]
For end-to-end GPT/BERT backbones we compose $N$ transformer-block predictions with amortized launch,
\[
\hat{T} = N\bigl(\hat{t}_{\mathrm{block}} - o_{\mathrm{launch}}\bigr) + o_{\mathrm{launch}} + \hat{t}_{\mathrm{LN}}.
\]

---

\section{Results}

\textbf{Cross-GPU block prediction (operator-level, $d{=}128$).}
Leave-one-GPU-out evaluation yields mean OOD MAPE of approximately 12--18\% across held-out GB10, H200, and RTX, with in-distribution validation typically near 5\% when the held-out device is GB10 or H200. Metrics include mean / median / P90 APE, log-space $R^2$, Spearman correlation, and predictive-interval coverage; we also break errors down by workload family.

\textbf{End-to-end GPT-2 / BERT composition (Ablation E).}
Composing amortized block predictions against measured full backbones (40 configs per GPU: five models $\times$ \{B1, B8\} $\times$ \{S512, S1024\} $\times$ \{bf16, f16\}) achieves overall MAPE of \textbf{17.1\%} on RTX PRO 6000 and \textbf{9.5\%} on H200 NVL. Larger batch-8 jobs are substantially more accurate than batch-1 (overhead-dominated) jobs; e.g.\ GPT-2 Large on H200 reaches low double-digit or single-digit error at B8. Naive $N\times$block composition without launch amortization is much worse (42\% RTX / 30\% H200), confirming that composition---not only the GNN---dominates e2e error.

\textbf{Takeaway.}
A schedule-free StableHLO $+$ public-spec hybrid model transfers across recent NVIDIA GPUs at usable accuracy, and the same block predictor composes into end-to-end Transformer backbones when launch overhead is not multiplied by depth. Extending beyond GPUs remains future work and requires family-specific measurements.

---

\section*{Compact poster bullets (if space is tight)}

\textbf{Intro}
- Diverse accelerators; profiling every target is expensive.
- Roofline is cheap but inaccurate; NeuSight / PipeWeave are strong but GPU-centric.
- We predict latency from StableHLO $+$ public specs only (no target execution).

\textbf{Method}
- GNN (graph) $+$ MLP (hardware) $\rightarrow$ latents $\rightarrow$ dual-peak roofline $+$ residual.
- Supervised log-latency NLL; masked MSE on privileged XLA labels; $\lambda{=}0.2$.
- E2E: $\hat{T}=N(\hat{t}_{\mathrm{block}}-o)+o+\hat{t}_{\mathrm{LN}}$.

\textbf{Results}
- LOGO block OOD $\sim$12--18\% MAPE (D, $d{=}128$).
- E2E GPT/BERT: 17.1\% RTX, 9.5\% H200 (amortized launch).
