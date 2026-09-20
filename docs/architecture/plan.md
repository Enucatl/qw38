# Qwen3.8 Clean-Sheet Architecture Study Plan

**Study status:** Phase 1 planned. No runtime implementation has been inspected or changed.

## Scope and guardrails

This is a first-principles study of a specialized inference architecture for Qwen3.8-27B. Its central design path is:

```text
Qwen3.8 mathematics
        ↓
logical dataflow
        ↓
tensor lifetime and state
        ↓
numerical sensitivity
        ↓
precision strategy
        ↓
custom quantization
        ↓
custom physical layouts
        ↓
offline model compiler
        ↓
Qwen-specific semantic graph
        ↓
decode/prefill execution plans
        ↓
CUDA mappings
        ↓
future experimental validation
```

The BF16 Transformers checkpoint at
`.cache/authorities/qwen3.8-27b-transformers` is the authoritative high-precision
source for model structure, weights, tensor statistics, and all quantization and
packing research. The file `models/Qwen3.8-27B-Q4_K_M.gguf` is not an
architectural constraint; it is reserved as a future black-box practical baseline.

The study may define a new tensor-specific quantization system, bit widths,
grouping, outlier treatment, physical layouts, runtime model format, offline
compiler, semantic graph, and separate decode/prefill schedules. Quantization,
layout, semantic graph, and eventual kernels must be co-designed; none is treated
as a fixed consequence of an existing generic format.

Until the clean-sheet study is frozen, its authors must not inspect Quartz
execution code or CUDA kernels, or llama.cpp/GGML Qwen graph construction or Qwen
CUDA kernels. Allowed evidence is limited to the BF16 model/configuration/tokenizer
assets, checkpoint metadata and statistics, architectural documentation and
papers, CUDA hardware documentation, and general numerical-analysis material.
No production implementation code may be modified during Phase 1.

## Evidence and notation policy

Each substantive result will be labelled as appropriate:

- **DERIVED** — follows from equations, dimensions, or arithmetic.
- **OBSERVED** — directly read from model files, configuration, metadata, or
  tensor statistics.
- **MEASURED** — produced by an explicit study measurement.
- **HYPOTHESIS** — plausible conclusion awaiting experiment.
- **UNKNOWN** — not established.

Mathematical documents will use GitHub Markdown math. Operations will give tensor
dimensions where practical; for example:

$$
W \in \mathbb{R}^{d_\text{out} \times d_\text{in}},\qquad
x \in \mathbb{R}^{d_\text{in}},\qquad
y = Wx \in \mathbb{R}^{d_\text{out}}.
$$

Hypotheses must remain hypotheses even when they are attractive design ideas.

## Stages and dependency order

1. **Establish the source of truth.** TASK-01 inventories the checkpoint and
   derives dimensions and parameter/byte totals. In parallel, TASK-16 establishes
   a hardware vocabulary without looking at model-specific kernels.
2. **Specify the model before designing an engine.** TASK-02 writes the full
   forward mathematical specification. TASK-03 translates it into a logical DAG,
   and TASK-04 determines lifetimes and token-persistent state. TASK-05 measures
   source-weight distributions independently of runtime behavior.
3. **Derive constraints and research spaces.** TASK-06 sets mathematical work and
   irreducible traffic lower bounds; TASK-07 identifies numerical-risk hypotheses.
   TASK-08 develops tensor-specific quantization experiments, followed by TASK-09
   (compiled runtime representation) and TASK-10 (offline compiler).
4. **Define model-native execution contracts.** TASK-11 derives semantic regions
   from dependencies, state, and lifetimes. TASK-12 identifies materialization and
   fusion hypotheses. TASK-13 and TASK-14 independently schedule decode and
   prefill, leading to consumer-driven candidate layouts in TASK-15.
5. **Explore hardware realization and validation.** TASK-17 maps semantic nodes
   and layouts into multiple CUDA experiment spaces. TASK-18 defines quality and
   quantization validation; TASK-19 defines end-to-end and kernel performance
   methodology.
6. **Synthesize and freeze.** TASK-20 connects all findings into a candidate
   architecture and dependency-ordered experiment backlog. TASK-21 consistency
   checks the study and marks it `FROZEN_FOR_COMPARATIVE_REVIEW`. Only after that
   state may comparative study of existing runtimes begin.

The detailed task status, outputs, open questions, and completion criteria are
maintained in [task_ledger.md](task_ledger.md). Dependencies, rather than apparent
implementation convenience, control when a task can be concluded.

## Study deliverables

Phase 1 produces a mathematical and evidence-labelled architecture dossier, not a
runtime. It will include the authoritative inventory and semantics, dataflow and
lifetime analysis, distribution and numerical studies, quantization/layout/format
research spaces, compiler and semantic-graph designs, independent decode and
prefill plans, CUDA mapping alternatives, validation methods, synthesis, review,
and an experiment backlog. The final frozen dossier is the baseline for a later
comparison against Quartz and llama.cpp; those systems must not retroactively
become its design authority without new evidence.
