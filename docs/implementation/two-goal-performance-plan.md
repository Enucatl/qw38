# Draft: complete attribution and efficient arithmetic

Status: Task 1 implemented and verified, 2026-09-27. Task 2 remains a draft.
Commands, accounting semantics and verification evidence are recorded in
[Candidate telemetry](telemetry.md).
The user instructed implementation of instrumentation/telemetry on the current
candidate. This does not accept TASK-047 or activate TASK-048. Preserve the
existing candidate and its evidence.

The direction is direct implementation: make the production execution path
observable, then replace expensive scalar contraction and unnecessary precision
with established llama.cpp consumers. No preliminary parameter sweep, broad
counter campaign, or PDL experiment is a prerequisite.

One correction to the premise: llama.cpp does not avoid FP32 everywhere.
It retains FP32 normalization/reductions, softmax statistics, recurrent state
and arithmetic, scaled quantized-dot accumulation, residuals and logits.
The objective is to remove avoidable per-element scalar contraction and match
the reference's precision at each boundary, including its FP32 boundaries.

Use the requested `../llama.cpp` checkout, frozen at
`1945e092030f8668ff93382799502d01490e564d`, as the implementation reference.
Record any local changes before reuse. Historical measurements used
`e6ab7c1a41054a888ada952eab4c886444c2f5ad`; do not attach their timings to this
newer revision. Keep the existing measured comparator identified separately.

## Task 1 — Complete production-path instrumentation

**Deliverable:** one existing-driver command produces a candidate report that
identifies every inference kernel, transfer and first-use operation and exposes
the stage responsible for an observed change. Instrumentation supports both
prefill and actual decode graph replay. It must not change numerical results.

### Implementation

1. Extend existing profiling and request reporting rather than introducing
   another profiler or benchmark framework. Relevant files are
   `src/runtime/profiling.hpp`, `language_model.cpp`, internal submission
   callers, `benchmarks/request_bench.cpp`, and `scripts/task027_profiles.py`.
   Cover the real internal enqueue route: existing standalone layer NVTX
   ranges do not cover `detail::Submission::layer` automatically.

2. Record the full lifecycle with explicit nested boundaries:
   - Initialization; artifact open/map, metadata validation, allocation and
     upload; session creation; plan binding; lazy library/kernel preparation.
   - Prefill call and each chunk, including input length, logical prompt
     length, absolute prefix, layer, selected schedule and matrix shape.
   - Decode token, populated length, graph bucket, capture/instantiate/first
     launch/replay, input submission, readback, completion, host argmax and
     state commit.
   - Within each layer: normalization and operand packing; each projection;
     attention preparation, scan, merge/gating and output projection;
     GDN preparation, convolution, recurrence and output; MLP gate/up,
     SwiGLU/down pack and down/residual; final RMS and vocabulary head.
   Fused operations receive a combined label rather than fabricated separate
   costs. Allocation, transfer and library APIs retain their real phase.

3. Make attribution work during graph replay. Host NVTX ranges recorded at
   capture do not repeat when the graph executes. Associate captured graph
   nodes with their layer/operator/shape through actual trace graph-node and
   launch correlation, preserving that mapping across replay and rebuilding
   it on recapture. Prove repeated names and graph buckets cannot mix layers.
   If the trace cannot expose a needed mapping, add narrowly scoped diagnostic
   graph-compatible markers; do not infer identity from duration or kernel
   name alone. No per-operator synchronization in production.

4. Fix timing semantics and component boundaries:
   - Report host wall time, CUDA event interval, kernel-duration sum, GPU
     activity union, transfers and uncovered intervals as distinct fields.
     An event interval must not be labeled pure GPU execution time.
   - Distinguish first-use work from recurring replay within the same request;
     retain first-use costs in total request latency.
   - Report overlap explicitly. CPU waits and kernel sums are not additional
     elapsed costs to add to wall time.
   - Measure resident component chains through their production producers,
     consumers and epilogues. Include input packing and required output
     projection. Include D2H only where the production API requires it;
     diagnostic output copies are separately named.
   - Repair the incomplete M1 attention boundary through output projection
     and residual addition. Preserve separately useful attention-core timing.

5. Produce one compact machine-readable report and human summary per candidate:
   source/binary/artifact/precision identity, actual dispatch, phase totals,
   first versus subsequent tokens, per-layer/family/shape/prefix costs,
   call counts, resources, memory and baseline deltas. Include both absolute
   milliseconds and share of the enclosing phase. For every changed family,
   reconcile component cost in the request with the observed request delta.
   Report host gaps as measured intervals within named phases; do not invent
   disk, driver, scheduling or cache causes for them.

### Verification and completion

- Reuse saved traces to verify the parser's accounting, overlap, clipping and
  API correlation before collecting new traces. Extend existing parser tests
  for graph-node attribution, repeated kernel names and missing metadata.
- Use existing 256/4096/32768 prompt plus eight-input development cases for
  integrated coverage, with graph transition/replay checks from integration.
  These captures replace additional standalone attribution runs where they
  cover the same question. Compare profiling on/off outputs and quantify
  instrumentation overhead on one representative request; do not subtract
  an assumed correction from performance numbers.
- A report fails validation if inference kernels or copies are unmapped,
  phases are missing, fused work is double counted, graph replay is attributed
  to capture-only host ranges, or timing boundaries disagree with the driver.
  All GPU inference work must have an owner. Name CPU intervals outside APIs
  separately; a trace is not proof of the internal cause of every driver wait.
- Task 1 is complete only when current-candidate results can distinguish a
  kernel regression, added packing/copying, first-use preparation, and exposed
  host/submission gaps without changing the benchmark afterward. If a missing
  boundary prevents that decision, repair the instrumentation within this task.

Instrumentation remains opt-in, with bounded storage and no new inference
synchronization. Existing measured requests and saved evidence are reused;
no repeated full quality run or counter sweep belongs in Task 1.

## Task 2 — Replace inefficient scalar consumers and match reference precision

**Deliverable:** a versioned candidate whose production projection and
attention paths use established packed/vector/tensor arithmetic, with every
remaining scalar/FP32 path explicitly accounted for against llama.cpp.
Task 1 supplies its measurements. Quality, state, capacity and final delivery
validation are included in this task, not a third task.

### Implement the replacements in this order

| Family | Concrete target |
| --- | --- |
| Large mixer projections, currently FP8 | Replace scalar E4M3-to-FP32 weight reconstruction/FMA with llama.cpp's established Q8_0×Q8_1 integer-dot decode consumers. Use a common compact Q8 resident view and its integer-MMA prefill consumer. Quantize directly from canonical source weights; do not requantize the existing FP8 artifact or retain a duplicate full weight view. Reuse compiler plumbing and compact layout where useful; implement/version upstream Q8_0 quantization and its runtime interpretation explicitly. This deliberately replaces the FP8 prefill consumer too, so prefill is a required verification target. |
| Vocabulary head | Consume compact Q8 weights with packed Q8 activation dots, retaining FP32 scaled accumulation and final logits as upstream does. Remove per-weight BF16 reconstruction and scalar FP32 multiplication. Pack the final hidden vector once and include packing in the reported head cost. |
| Q4 MLP, both phases | Retain compact Q4 weights and existing shared gate/up producer. Adopt the reference's activation-scale/sum representation, packed metadata and affine correction where our wider precision adds work. Keep integer DP4A for decode and integer MMA for prefill, with FP32 accumulation where upstream keeps it. A wider load alone is not completion. |
| Attention, both phases | Implement the selected NVIDIA FP16 Q/K/V and single-component probability/PV path, including its accumulator and rescaling behavior. Replace the high+low BF16 double-PV calculation. Match QK accumulation and softmax statistics to upstream, including retained FP32. Update cache writers/readers and replay metadata consistently; do not merely drop the BF16 low component, which is a different arithmetic policy. |
| RMS, nonlinearities, residuals, GDN, small projections and tails | Audit all remaining production kernels. Retain FP32 reductions, recurrence and outputs where the actual reference uses them. Remove redundant conversions/materializations; use existing packed/vector loads and library/warp consumers for avoidable scalar contraction. Retain scalar operations that implement the reference equations or handle genuinely small/tail work, with an explicit reason and measured contribution. |

The Q8 mixer choice is a direct implementation target, not a format contest
or a claim that every comparator tensor is Q8. Existing `Q8G32CandidateV1`
has different scale fitting and code rounding from upstream Q8_0. Encode the
changed rules in the new policy rather than reinterpreting old bytes.
If the faithful replacement fails required quality or integrated performance
checks, report that failed outcome and measured cause. Repair concrete defects,
but do not start indefinite tuning or declare the old scalar FP8 path to have
met this goal. FP32 is not itself proof of an
inefficient kernel: integer dots also need scale application and floating
accumulation, and scalar instructions are appropriate for many nonlinearities.

### Precision and integration contract

- Start with a short operator table recording reference source/dispatch,
  weight and activation types, dot instructions, accumulators, scale/sum
  precision, outputs and persistent state. This table defines the replacement
  precisely and records where FP32 remains. It is a source audit, not a new
  benchmark campaign.
- Introduce a new precision/artifact policy for the changed operands,
  rounding and attention cache. Update compiler, manifest validation, runtime
  dispatch, workspace sizing and snapshot compatibility together. Do not
  relabel these changes as policy 1030 tuning or require old-policy bitwise
  outputs. Preserve independent numerical references, nonfinite handling and
  same-policy replay guarantees.
- Copy/adapt the relevant reference arithmetic and proven consumers, retaining
  required attribution. Do not replace scalar float loops with scalar half
  loops, promise tensor cores for every M1 shape, or expand quantized weights
  globally just to reach a library call.
- Verify the intended integer/MMA instructions in the compiled consumers and
  their actual selection in Task 1's production trace. Generic/reference
  fallbacks cannot silently serve the regular model shapes.
- Keep one resident weight view per tensor, bounded reusable scratch, existing
  session ownership, graph validity, safe state commits and the 2 GiB reserve.
  Keep the established context scheduling unless a replacement consumer
  requires a concrete compatibility change. PDL and broad scheduling research
  are outside these two goals.

### Verification and completion

After each family replacement, run its affected independent numerical tests,
production-path check and existing short NLL screen. Use Task 1's same-shape
resident timings and matched request to identify the cost change. Cover M1,
M256/M512 and actual tails where applicable; do not run the entire quality
suite after every kernel edit.

Once the combined candidate is frozen, run the existing consolidated quality,
replay, context-boundary, populated-32K capacity and matched performance gates
once. Reuse the core-54 and long-context protocol, including existing quality
thresholds; no historical output exception automatically transfers. New
formats require new quality evidence. Compare first-use and recurring work
separately, and measure total prefill/decode/request latency against the frozen
parent and the correctly identified llama.cpp comparator. Run timing without
diagnostic overhead and pair it with a separate attributed capture; keep both
results identified rather than merging their numbers.

Completion requires all of the following:

- No unreviewed production scalar contraction or excess-precision path remains.
  Each is replaced or documented as matching the reference/required tail work.
  The large mixer, head and double-PV replacements above are mandatory.
- Actual dispatch and generated instructions confirm the intended consumers.
- Affected correctness, quality, replay and capacity checks pass.
- Integrated prefill and decode improve against the frozen parent without
  hiding a loss in another required workload. Report llama.cpp parity separately.
- Every performance decision includes the complete relevant operation and its
  production contribution. Missing attribution is repaired, not deferred to
  a later optimization round. Small single-run differences remain observations,
  not statistical claims; no automatic experiment expansion is prescribed.

## Relationship to the existing implementation plan

If adopted, these two tasks form a successor round to FAST-04. Preserve
TASK-045/046 history and TASK-047's blocked observations; freeze and identify
which source/schedule starts the new round. Prefer committed TASK-046 as the
arithmetic parent, retaining the TASK-047 candidate separately until its
schedule is explicitly selected. Instrumentation can cover both candidates.

Task 2 explicitly replaces FAST-04's restrictions requiring FP8 mixer weights,
BF16 decode operands/cache, FP32-only metadata and two-component attention P.
Those old restrictions must not prevent the requested precision replacements.
Fold the applicable TASK-048 consolidated validation obligations into Task 2
when updating the ledger, rather than requiring another delivery task.
TASK-043 remains production until the new candidate passes its final gate.

Planning verification: inspected current QW38 callers and the requested local
llama.cpp sources. In particular, `mmvq.cu` retains float accumulators,
`norm.cu` retains float sum-of-squares, `fattn-mma-f16.cuh` retains float QK
statistics, and `gated_delta_net.cu` retains float recurrent state/arithmetic.
No implementation, measurements, policy activation, commits or pushes were
performed while drafting this plan.
