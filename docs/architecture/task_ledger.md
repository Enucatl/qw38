# Qwen3.8 Clean-Sheet Study Task Ledger

This ledger is the authoritative status record for Phase 1. Unresolved questions
remain recorded after a task is complete.

## TASK-00 — Create the study master plan

**Status:** DONE

**Depends on:** None

**Produces:**
- `docs/architecture/plan.md`
- `docs/architecture/task_ledger.md`

**Purpose:**
Define the clean-sheet scope, evidence rules, study ordering, and task tracking.

**Established results:**
- The BF16 Transformers checkpoint is the primary high-precision authority.
- The Q4_K_M GGUF is reserved for future black-box comparison only.
- Existing Quartz and llama.cpp/GGML Qwen execution implementations are excluded
  until Phase 1 is frozen.
- No production implementation code is in scope.

**Open questions:**
- None for planning; architectural questions are assigned to downstream tasks.

**Downstream impact:**
- Defines the evidence vocabulary, anti-anchoring boundary, and dependency order
  for all remaining tasks.

**Completion criteria:**
- [x] Create the master study plan.
- [x] Initialize the task ledger and all task statuses.

## TASK-01 — Establish authoritative model facts

**Status:** DONE

**Depends on:** TASK-00

**Produces:**
- `docs/architecture/model-inventory.md`
- `scripts/inventory_bf16_checkpoint.py`

**Purpose:**
Establish model configuration, layer ordering, state structures, and complete
BF16 tensor inventory from the authoritative checkpoint.

**Established results:**
- BF16 checkpoint at `.cache/authorities/qwen3.8-27b-transformers`: 1199 tensors,
  18 shards, all `BF16`, 27,781,427,952 parameters, 55,562,855,904 payload bytes
  (matches `metadata.total_size`).
- `Qwen3_5ForConditionalGeneration` with 64 language layers (`48× linear_attention`,
  `16× full_attention` at indices 3,7,…,63); one MTP full-attention block after
  the stack; vision present but deferred in prose (333 tensors in totals).
- Untied embeddings and `lm_head`; structural state implications documented
  (KV candidates, linear-attn conv/SSM parameters, output gate via doubled
  `q_proj`); forward math deferred to TASK-02.
- Level-1/level-2 semantic-family inventory with machine-checkable JSON fence;
  verified by `scripts/inventory_bf16_checkpoint.py`.

**Open questions:**
- Exact forward equations, gate application, linear-attention recurrence, and
  `mrope_section` vs rotary-dim relationship (TASK-02).

**Downstream impact:**
- Provides dimensions, tensor names, and source totals to semantic and statistics
  work.

**Completion criteria:**
- [x] Document observed structure and state implications.
- [x] Inventory tensors by semantic family with parameter and BF16 byte totals.
- [x] Verify totals against the checkpoint.

## TASK-02 — Derive the complete mathematical model

**Status:** DONE

**Depends on:** TASK-01

**Produces:**
- `docs/architecture/model-semantics.md`
- `scripts/check_model_semantics.py`

**Purpose:**
Write the dimensioned forward specification from token and state to logits and
next state.

**Established results:**
- `docs/architecture/model-semantics.md` — dimensioned language + MTP forward map
  with locked equations for embedding, RMSNorm, Gated Attention (sigmoid output
  gate), partial interleaved mRoPE, GDN recurrence, SwiGLU MLP, persistent
  state, primary logits, and MTP mix/block/logits; algebraic equivalents recorded.
- `scripts/check_model_semantics.py` — stdlib checker for config arithmetic,
  `--json` output, and semantics headings, JSON fence, and forbidden-token rules.
- TASK-01 UNKNOWN closures: output gate on extra `q_proj` half; GDN Eq. (10)
  recurrence with stored shape `(48,128,128)`; `mrope_section` sums to
  `d_rot/2`, not `d_rot`.

**Open questions:**
- None for language forward math; vision encoder internals remain deferred.

**Downstream impact:**
- Is the mathematical authority for dataflow, work, precision, and state study.

**Completion criteria:**
- [x] Specify every required operation and recurrent transition with equations.
- [x] State dimensions for inputs, outputs, weights, and state.
- [x] Exclude kernel, graph-layout, and implementation detail.

## TASK-03 — Build the logical dataflow graph

**Status:** DONE

**Depends on:** TASK-02

**Produces:**
- `docs/architecture/dataflow.md`
- `scripts/check_dataflow.py`

**Purpose:**
Represent the mathematical model as logical producers, consumers, fan-out, and
state transitions.

**Established results:**
- `docs/architecture/dataflow.md` — Phase 1 logical DAG for language + MTP with
  eight Mermaid region diagrams, 52-row intermediate catalog, canonical
  logical-≠-physical sentence, and DERIVED sharing/reuse ranking (items 1–11).
- `scripts/check_dataflow.py` — stdlib checker for config arithmetic, `--json`
  output, and dataflow headings, JSON fence, Mermaid node IDs, and
  forbidden-token rules.
- Ledger open question closed: consequential sharing/reuse ranked as graph
  structure (`h`/`h_mid` residual stream, `h_tilde` projection fan-out, KV
  writes, `qkv`/`C_state`, `S`, shared `E`/`W_lm`, live-across `g`/`z`, and
  intra-equation `k_hat`).

**Open questions:**
- None for language logical DAG; vision encoder internals remain deferred.

**Downstream impact:**
- Constrains lifetime analysis, semantic nodes, and materialization study.

**Completion criteria:**
- [x] Diagram all required model regions and token-to-token state.
- [x] Tabulate significant logical intermediates and their consumers.
- [x] State that logical values do not imply physical allocation.

## TASK-04 — Analyze lifetime and persistent state

**Status:** TODO

**Depends on:** TASK-03

**Produces:**
- `docs/architecture/lifetime-and-state.md`

**Purpose:**
Classify value lifetimes and quantify persistent state traffic and storage.

**Established results:**
- Not started.

**Open questions:**
- Exact state sizes and which values require materialization across boundaries.

**Downstream impact:**
- Informs traffic bounds, numerical risk, semantic contracts, and layouts.

**Completion criteria:**
- [ ] Classify every significant value by lifetime and recomputability.
- [ ] Calculate persistent-state storage and per-token read/write volumes.
- [ ] Identify semantic storage candidates without CUDA decisions.

## TASK-05 — Analyze BF16 tensor distributions

**Status:** TODO

**Depends on:** TASK-01

**Produces:**
- `docs/architecture/bf16-tensor-analysis.md`

**Purpose:**
Measure family and layer-level BF16 weight distributions relevant to later
quantization research.

**Established results:**
- Not started.

**Open questions:**
- Distribution variation, scale variation, and outlier structure by tensor family.

**Downstream impact:**
- Supplies measured evidence for quantization and compiler-profile design.

**Completion criteria:**
- [ ] Measure requested global and directional distribution statistics.
- [ ] Compare layers within each major family.
- [ ] Label measurements and avoid quality conclusions without experiments.

## TASK-06 — Derive theoretical computation and traffic lower bounds

**Status:** TODO

**Depends on:** TASK-02, TASK-03, TASK-04

**Produces:**
- `docs/architecture/work-and-traffic.md`

**Purpose:**
Derive decode/prefill mathematical work and irreducible data movement separately.

**Established results:**
- Not started.

**Open questions:**
- Region-level arithmetic intensity and resulting bottleneck hypotheses.

**Downstream impact:**
- Grounds quantization, scheduling, semantic graph, and validation methodology.

**Completion criteria:**
- [ ] Derive symbolic and instantiated work counts.
- [ ] Calculate unavoidable weight, state, and activation traffic.
- [ ] Label bottleneck classifications as hypotheses.

## TASK-07 — Study numerical sensitivity from the mathematics

**Status:** TODO

**Depends on:** TASK-02, TASK-04

**Produces:**
- `docs/architecture/numerical-sensitivity.md`

**Purpose:**
Identify mathematical precision risks across weights, activations, reductions, and
persistent state.

**Established results:**
- Not started.

**Open questions:**
- Which qualitative risk hypotheses survive model-level validation.

**Downstream impact:**
- Frames quantization options, state precision experiments, and semantic contracts.

**Completion criteria:**
- [ ] Analyze requested sensitive operations and accumulation paths.
- [ ] Distinguish all relevant precision roles.
- [ ] Classify risk without claiming experimental proof.

## TASK-08 — Design the custom quantization research space

**Status:** TODO

**Depends on:** TASK-05, TASK-06, TASK-07

**Produces:**
- `docs/architecture/quantization-design-space.md`

**Purpose:**
Define tensor-specific custom quantization experiments from BF16 source evidence.

**Established results:**
- Not started.

**Open questions:**
- Which bit widths, grouping, scales, and outlier policies form the Pareto frontier.

**Downstream impact:**
- Defines possible compiled tensor representations and experiments.

**Completion criteria:**
- [ ] Cover requested design dimensions and metadata/compute implications.
- [ ] Propose family-specific candidate policies without selecting winners.
- [ ] Record quality and decode-complexity risks.

## TASK-09 — Design the custom runtime model format

**Status:** TODO

**Depends on:** TASK-03, TASK-06, TASK-08

**Produces:**
- `docs/architecture/runtime-format-design.md`

**Purpose:**
Specify requirements for a compiler-produced, consumer-oriented model artifact.

**Established results:**
- Not started.

**Open questions:**
- Portable versus backend-specialized artifact boundaries and ideal consumer byte
  sequences.

**Downstream impact:**
- Guides compiler planning, schedules, and physical layouts.

**Completion criteria:**
- [ ] Analyze all requested representation and packing capabilities.
- [ ] Compare portable and backend-specific artifact approaches.
- [ ] Keep decisions open absent compelling evidence.

## TASK-10 — Design the offline model compiler

**Status:** TODO

**Depends on:** TASK-05, TASK-08, TASK-09

**Produces:**
- `docs/architecture/model-compiler-plan.md`

**Purpose:**
Define conceptual compiler stages from BF16 checkpoint to specialized runtime model.

**Established results:**
- Not started.

**Open questions:**
- Which profile decisions are architecture-, calibration-, model-, or backend-driven.

**Downstream impact:**
- Establishes the future build pipeline and profile model.

**Completion criteria:**
- [ ] Describe validation, analysis, quantization, packing, metadata, and integrity stages.
- [ ] Classify decision ownership.
- [ ] Support future quality, balanced, and compression profiles conceptually.

## TASK-11 — Derive the specialized semantic graph

**Status:** TODO

**Depends on:** TASK-03, TASK-04, TASK-06, TASK-07

**Produces:**
- `docs/architecture/semantic-graph.md`

**Purpose:**
Define hardware-independent Qwen-specific execution regions and contracts.

**Established results:**
- Not started.

**Open questions:**
- Natural boundaries, internal values, and synchronization/materialization edges.

**Downstream impact:**
- Is the common contract for fusion, schedules, layouts, and CUDA alternatives.

**Completion criteria:**
- [ ] Derive node boundaries from semantic evidence rather than framework primitives.
- [ ] Specify operations, I/O, state, internal values, and flexibilities per node.
- [ ] Keep contracts hardware-independent.

## TASK-12 — Determine materialization and fusion opportunities

**Status:** TODO

**Depends on:** TASK-03, TASK-04, TASK-11

**Produces:**
- `docs/architecture/materialization-and-fusion.md`

**Purpose:**
Classify physical-materialization need and fusion experiment opportunities.

**Established results:**
- Not started.

**Open questions:**
- Which apparent fusions improve total behavior after working-set and synchronization costs.

**Downstream impact:**
- Constrains decode/prefill boundaries and CUDA experiments.

**Completion criteria:**
- [ ] Classify important intermediates with required justification.
- [ ] List reuse, recomputation, local-working-set, and synchronization tradeoffs.
- [ ] Label every fusion proposal as a hypothesis.

## TASK-13 — Derive a clean-sheet decode execution plan

**Status:** TODO

**Depends on:** TASK-06, TASK-09, TASK-11, TASK-12

**Produces:**
- `docs/architecture/decode-plan.md`

**Purpose:**
Design the ideal semantic schedule for one-token inference independent of current engines.

**Established results:**
- Not started.

**Open questions:**
- Boundary-added traffic relative to mathematical minimum traffic.

**Downstream impact:**
- Establishes decode consumers for physical layout and CUDA mapping research.

**Completion criteria:**
- [ ] Describe loads, state reads/writes, visibility boundaries, and reuse per stage.
- [ ] Separate unavoidable traffic from proposed-boundary traffic.
- [ ] Identify packing and fusion hypotheses without thread geometry.

## TASK-14 — Derive a clean-sheet prefill execution plan

**Status:** TODO

**Depends on:** TASK-06, TASK-09, TASK-11, TASK-12

**Produces:**
- `docs/architecture/prefill-plan.md`

**Purpose:**
Design an independent many-token semantic schedule.

**Established results:**
- Not started.

**Open questions:**
- Decode/prefill representation tradeoffs and potential need for multiple views.

**Downstream impact:**
- Establishes prefill consumers for physical layout and CUDA mapping research.

**Completion criteria:**
- [ ] Explain fundamental matrix-matrix, reuse, tiling, state, and temporary-storage differences.
- [ ] Compare every semantic node with decode.
- [ ] Keep representation tradeoffs unresolved without evidence.

## TASK-15 — Design physical tensor layouts from consumers

**Status:** TODO

**Depends on:** TASK-09, TASK-13, TASK-14

**Produces:**
- `docs/architecture/layout-strategy.md`

**Purpose:**
Develop candidate consumer-driven layouts for weights and persistent state.

**Established results:**
- Not started.

**Open questions:**
- Which planned parallel decompositions justify each candidate ordering and tile.

**Downstream impact:**
- Supplies layout alternatives to CUDA design-space work and compiler planning.

**Completion criteria:**
- [ ] Analyze logical dimensions, consumers, access, tiling, alignment, and conversion.
- [ ] Include GDN, convolution, and KV persistent state.
- [ ] Avoid claims of optimality before CUDA analysis.

## TASK-16 — Build the CUDA hardware model

**Status:** TODO

**Depends on:** TASK-00

**Produces:**
- `docs/architecture/cuda-hardware-model.md`

**Purpose:**
Provide a model-independent CUDA resource and tradeoff reference.

**Established results:**
- Not started.

**Open questions:**
- Target-GPU-specific limits and instruction availability for future experiments.

**Downstream impact:**
- Supplies vocabulary and evaluation criteria for CUDA mapping alternatives.

**Completion criteria:**
- [ ] Cover requested execution, memory, resource, synchronization, and instruction concepts.
- [ ] Explain fusion versus occupancy tradeoffs mathematically.
- [ ] Exclude current Qwen kernel inspection.

## TASK-17 — Map semantic nodes into CUDA design spaces

**Status:** TODO

**Depends on:** TASK-11, TASK-13, TASK-14, TASK-15, TASK-16

**Produces:**
- `docs/architecture/cuda-design-space.md`

**Purpose:**
Describe multiple plausible CUDA ownership and reduction mappings per semantic node.

**Established results:**
- Not started.

**Open questions:**
- Which mappings win on target hardware and profiles after benchmarks.

**Downstream impact:**
- Seeds implementation experiments and performance validation.

**Completion criteria:**
- [ ] Analyze multiple mapping alternatives per semantic node.
- [ ] Estimate work, storage, access, synchronization, occupancy, and mode suitability.
- [ ] Select no winner without measurements.

## TASK-18 — Design quantization evaluation methodology

**Status:** TODO

**Depends on:** TASK-07, TASK-08, TASK-10

**Produces:**
- `docs/architecture/quantization-validation.md`

**Purpose:**
Define local diagnostics and model-level quality comparisons for quantization profiles.

**Established results:**
- Not started.

**Open questions:**
- Final calibration corpora, prompt suite, capability benchmarks, and acceptance frontier.

**Downstream impact:**
- Makes future quantizer selection evidence-driven rather than reconstruction-only.

**Completion criteria:**
- [ ] Define reconstruction diagnostics and their limits.
- [ ] Define teacher-forced and behavioral comparisons.
- [ ] Position Q4_K_M as a future black-box Pareto reference, not a requirement.

## TASK-19 — Design the future performance methodology

**Status:** TODO

**Depends on:** TASK-06, TASK-13, TASK-14, TASK-17

**Produces:**
- `docs/architecture/performance-validation.md`

**Purpose:**
Define decode, prefill, kernel, memory, and end-to-end performance measurements.

**Established results:**
- Not started.

**Open questions:**
- Exact hardware, prompt matrix, and reproducibility protocol for implementation phase.

**Downstream impact:**
- Defines how architectural experiments will be evaluated fairly.

**Completion criteria:**
- [ ] Specify requested decode, prefill, and kernel metrics.
- [ ] Require end-to-end measurement alongside microbenchmarks.
- [ ] Do not run implementation benchmarks in this study task.

## TASK-20 — Synthesize the clean-sheet architecture

**Status:** TODO

**Depends on:** TASK-01 through TASK-19

**Produces:**
- `docs/architecture/clean-sheet-architecture.md`
- `docs/architecture/experiment-backlog.md`

**Purpose:**
Connect completed evidence into a candidate architecture and ordered validation backlog.

**Established results:**
- Not started.

**Open questions:**
- Candidate alternatives and unknowns remaining after all Phase 1 analyses.

**Downstream impact:**
- Forms the documented baseline for future comparative runtime review.

**Completion criteria:**
- [ ] Synthesize the required architecture chain and evidence classes.
- [ ] Include an architecture diagram and alternatives.
- [ ] Create dependency-ordered experiment entries with all requested fields.

## TASK-21 — Freeze the clean-sheet baseline

**Status:** TODO

**Depends on:** TASK-20

**Produces:**
- `docs/architecture/clean-sheet-review.md`

**Purpose:**
Consistency-review and freeze the self-contained Phase 1 architecture dossier.

**Established results:**
- Not started.

**Open questions:**
- Genuine unresolved architectural questions retained after consistency review.

**Downstream impact:**
- Authorizes the subsequent comparative review of existing runtimes without making
  them silent design authority.

**Completion criteria:**
- [ ] Check cross-document equations, dimensions, totals, state, contracts, and proposals.
- [ ] Correct documentation inconsistencies and preserve true unknowns.
- [ ] Mark the reviewed design `FROZEN_FOR_COMPARATIVE_REVIEW`.
