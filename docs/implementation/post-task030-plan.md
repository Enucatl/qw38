# Fast inference after TASK-030 — FAST-01

Historical plan for completed TASK-033–036. The
[FAST-02 amendment](task_ledger.md#fast-engine-amendment--fast-02-2026-09-27)
governs TASK-037–040 and explicitly supersedes the retained Q4_K per-weight
BF16 rounding, fixed attention schedule and graph deferral for that new batch.
The original plan and measured completion records remain historical evidence.

Planning amendment, 2026-09-26. This applies the fast delivery principles in
[`astraprompt.md`](../../astraprompt.md) to the completed TASK-030 milestone.
It authorizes TASK-033–036 as future work; no implementation, build, artifact
generation, benchmark or evaluation was performed for this replan.
TASK-001–030 results and retired TASK-031/032 identities are preserved.

## Starting point and recommendation

Keep the accepted TASK-026/027 runtime (`d2f02e2`) as the production control.
TASK-030's attention candidate is useful development code, not an accepted
replacement: C92 remains INCONCLUSIVE. Use it as the engineering starting
point, carrying that limitation through final validation.

| Saved observation | Engineering implication |
| --- | --- |
| 32K prefill: 56.647 s versus 12.373 s comparator | About 4.58× latency gap remains after TASK-028's improvement from 208.46 s. |
| 32K prefill GPU categories: attention 27.532 s, projections 14.977 s, unpack/packing 11.138 s | Both attention and projection/conversion need work. Eliminating either category alone cannot achieve parity. |
| Decode-4096: projections 3.196 s out of 4.142 s elapsed for 128 tokens | First optimize compact weight consumers; a large-M GEMM result cannot select decode dispatch. |
| Decode-32768: attention 1.459 s, projections 3.249 s, elapsed 5.292 s | Reuse in attention also matters for populated decode. |
| TASK-029 paired gate/up decode: 15.670 s versus 1.220 s control | The scalar same-weight NVFP4 GEMV, not activation packing, caused the major regression. |
| TASK-029 request prefill: 69,632 native GEMMs, 34,816 SwiGLU launches | Native operands still run through an outer 512-output-row loop; remove that scheduling constraint. |
| TASK-030 model 21.463 GB, state 2.310 GB, scratch 0.257 GB; sampled free memory 8.317 GB at 32K | Capacity permits bounded useful workspace. It does not justify full floating-point weight copies. |

Sources: [TASK-030 report](tasks/TASK-030.md), compact
`.cache/evaluation/qw38-language-v2/task030-support/profiles.json`, and
[TASK-029 report](tasks/TASK-029.md). GPU category sums and host elapsed have
different boundaries. These are instrumented single observations with first
use, not repeat statistics or measured DRAM byte counts. GB here is decimal;
the required reserve remains 2 GiB. Resident bytes are not bytes read per token:
embeddings and retained inactive payloads are not streamed in full each step.

Implement **033 attention → 034 compact projections → 035 FP8 reuse →
036 final decision**. Each development task leaves a usable integrated path
or a measured retained fallback. A rejection must identify the remaining
cost; completing an experiment does not mean the fast-engine goal is met.

## Intended calculation chain

Choose the representation with its actual consumer and scale layout. No
blanket FP8 conversion and no claim that Q4_K is a native FP4 encoding.

| Producer / weights | Stored operand and consumers | Lifetime / precision boundary |
| --- | --- | --- |
| Mixer RMS over FP32 residual | One FP8 E4M3 input plus FP32 scales per row/K128, reused by attention QG/K/V or GDN QKV/Z | Produce once per layer and token range; keep until all siblings finish. Emit the BF16 companion in the same producer only for retained consumers such as tiny GDN A/B. |
| Attention/GDN large weights currently Q8 | Compile directly from pinned BF16 to E4M3 with FP32 scales per N128/K128 block; one TN-compatible resident view | Native groupwise FP8 GEMM for prefill; same-view GEMV initially for M=1. New quantization policy, not a reinterpretation of Q8 bytes. |
| Attention QK, online softmax and PV | BF16 Q/K/V; FP32 scores, maxima, denominator and output accumulation in local tiles | Calculate each probability once. Prefill rounds the local P tile to BF16 for tensor-core PV; this is an explicit new precision experiment. No global score/probability matrix. |
| Attention gate / GDN gated norm | FP32 local arithmetic, then direct FP8 codes/scales for the output projection when selected | This is a new semantic value, so it needs its own pack. Preserve the existing BF16 rounding boundary initially; do not materialize that round trip in global memory. |
| MLP RMS | One shared NVFP4 pack for native gate/up, or shared BF16 input for Q4_K fallback | TASK-029 already does sibling reuse. Keep per-row/K16 E4M3 scales and its exact recipe; repair the consumers. |
| MLP gate/up → SwiGLU → down | FP32 accumulation/nonlinearity; BF16 down input; Q4_K down | Fuse the narrow-output epilogue where practical. Retain a bounded FP32 pair of output slabs only if needed; never a model-wide FP32 intermediate. |
| Retained Q4_K weights | Compact resident codes/scales; decode packed GEMV, prefill on-chip BF16 unpack feeding tensor cores | Preserve Q4_K reconstruction and BF16 operand rounding. BF16 tiles live in registers/shared memory within a kernel, not in a global unpack buffer on the selected optimized path. |
| Final norm / head | Existing Q8 head and one normalized BF16 row initially | Reuse existing decode GEMV for one-row prefill readout; head FP8 is deferred until a head-specific result warrants it. Generation computes the last row once, including when evaluation requested that row. |

Default retained precision is FP32 residuals, reductions, nonlinear evaluation,
accumulators and GDN recurrent state; BF16 embeddings, small parameters,
convolution history and KV. FP32 *arithmetic* does not imply a large FP32
tensor in global memory. Carry an accumulator locally into its epilogue when
possible. Norms and nonlinearities are not evaluated in FP8 merely because
their outputs are stored in FP8.

The existing pinned CUTLASS revision
`098de2a652cf8f00fd70b2df54051c7eccbb855a` contains
`examples/87_blackwell_geforce_gemm_blockwise/87b_blackwell_geforce_fp8_bf16_gemm_groupwise.cu`:
E4M3 operands, FP32 accumulation/scales, M1/N128/K128 scale granularity, and
SM120 schedules. Reuse this C++ implementation pattern; its integration and
quality are still hypotheses. Start with its cooperative 128×128×128 tile.
The scale layouts come from its `ScaleConfig`, not a guessed linear array.

[NVIDIA's SM120 documentation](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html#blackwell-sm120-gemms)
describes TN layouts and distinct ordinary, MX and NVFP4 operand families.
NVFP4's K16 scale scheme is not interchangeable with the proposed FP8 one.
Do not convert an FP8 operand to FP4 just to feed an incompatible kernel;
choose sibling consumers with the same contract. The
[cuBLAS scaling support table](https://docs.nvidia.com/cuda/cublas/#outer-vector-scaling-for-fp8-data-types)
also does not establish its outer-vector mode for SM120; use the inspected
CUTLASS groupwise route. Latest documentation is context, while the pinned
source and one affected operation check determine actual support.

## Prepare once, reuse until the last consumer

- Compile/reorder static weights and scale swizzles once, offline. A lossless
  physical-layout change may repack existing quantized codes; a new quantizer
  starts from pinned BF16 and gets a new policy/layout identity. Upload one
  view per tensor. Never expand/requantize the whole weights per request or
  keep Q8 and FP8 copies resident as an implicit fallback.
- Prepare stable descriptors, dispatch and workspace sizing when binding a
  plan. Patch only dynamic pointers/row counts at execution; do not repeat
  heuristic selection for every output tile. Fallback uses the same view
  where supported, or reloads a separate artifact between processes/runs.
- An activation pack belongs to a semantic producer, layer, residual version,
  token range, shape and precision/scale recipe. Ordinary plan ownership and
  stream ordering suffice; do not build a global memoization cache. Pointer
  equality alone is not identity because scratch is reused.
- Already compatible FP8 codes **and scales** pass directly to all consumers.
  Packing happens outside output-row tile loops. RMS and its amax reduction
  are not repeated per projection or per tile. Values after SwiGLU, gating,
  residual addition, another norm/gamma, RoPE or a different K grouping are
  different operands; sharing them would be incorrect.
- A row's scales depend only on that row's K block. Padding must be zero and
  not influence scales. Freeze RNE/saturation, zero-block and nonfinite rules
  before integration; invalid inputs must not silently become finite zeros.
- Cross-kernel fan-out uses one bounded device workspace span until the last
  consumer completes on the session stream. Registers/shared memory hold
  only intra-kernel tiles. No assumptions about cache pinning or values
  surviving arbitrary kernel launches. Snapshot/reset contracts cover state,
  not reusable dead scratch.

## Expected bottlenecks and decisions we can make now

**Attention:** decode recomputes exponentials inside the loop over 256 value
coordinates (`cuda/attention.cu`, `attention_scan_kernel`). Store
the probability once and reuse it for both normalization and PV. Prefill
already does this, but only QK uses tensor cores; its PV is scalar. Add
tensor-core PV first, retaining FP32 online softmax and causal masking.
Keep Q32/K64 and existing segmentation initially. GQA groups share KV, but
multi-head staging comes only if the new path is still traffic bound: the
existing prefill CTA already uses 102 registers/thread and 57,728 shared bytes,
with one CTA per SM in TASK-028 evidence. More fusion can reduce occupancy.

**Projections:** `PrefillEngine::gemm_tile` writes a BF16 weight tile to global
memory, cuBLAS reads it back, and the output tile is written as FP32 before an
epilogue. Remove these intermediate transactions where the consumer permits.
For retained Q4_K, a fused unpack/tensor-core mainloop is real implementation
work, not a trivial library flag. Reuse the existing Q4_K decoder and CUDA
tile patterns; start with down, then reuse for gate/up if NVFP4 is rejected.
For NVFP4, first try native M=1 with one activation pack and full-N invocation;
keep it only if total pair cost beats the Q4_K control. Avoid polishing the
known slow scalar NVFP4 consumer without evidence that this is the shortest fix.

**FP8:** prepare large Q8-family weights once and keep narrow activations
through fan-out. This primarily removes repeated prefill unpack and can
reduce operand traffic; it is not a promised 2× decode weight-byte reduction
over INT8. Same-view FP8 decode must avoid scalar conversion bottlenecks.
GDN A/B and head retain their current policies initially. Sensitive-family
quality has not been established for the proposed FP8 recipe.

**Scheduling:** retain 256-token chunks initially. A 32K prompt traverses
weights in 128 chunks; larger chunks could amortize weight reads and launches,
but changing all workspaces and recurrence schedules first would lengthen the
loop. After the consumers work, allow one 256-versus-512 prompt-only comparison
if remaining weight streaming/launch cost justifies it and a memory calculation
fits. No chunk search. Reuse requested last-row logits; keep evaluator logits
and generation behavior consistent. GPU argmax/CUDA graphs are later work
only if the remaining trace shows host/readback/launch overhead is material.

## Short development loops, one final gate

For one change: inspect the real producer and all consumers; state one cost
prediction; implement; run one discriminating correctness check; measure one
affected operation including its preparation/epilogue; keep or revert; then
run one short integrated check after a coherent path is ready. Do not add
another measurement unless it could change the next implementation decision.

Use existing fixtures/reference tools and one real shape at M=1 and M=256
for a changed projection, or a bounded attention input with a populated prefix.
Default integrated development workload is the frozen TASK-027 256-token
prompt followed by 8 tokens, with one execution, zero warmups and no repeats.
Use a predeclared small TASK-020 development subset for new precision; never
select or fit it using EVAL-01 answers. Aim for operation checks in seconds
and integrated checks under a minute including load when feasible; report
actual elapsed time, not an unmeasured deadline. No 128-token decode or 32K
full-model request is required after every kernel edit. Different shape,
scale scope or memory lifetime needs a relevant boundary check, not the
whole suite. Development timings with different continuations are not matched
TASK-030 speedup ratios.

Inspect the saved C92 lost case `6082513c8dba4ec68aa68f1bf5854d09` and available
answer margins early. If evidence cannot locate the discrepancy, one controlled
old/new attention diagnostic is justified. It may identify a numerical issue;
it must not produce an answer-specific path, calibration or tolerance. Restoring
one answer is not final acceptance. Repeating deterministic cases or resampling
the same saved loss does not fix the unchanged C92 point-loss gate.

TASK-036 owns core-54 and all 15 output-bound P100 reviews, the single fixed
32K quality case, complete replay/session coverage, capacity and all nine
PERF-01 rows (six executions using request prompt phases). Reuse matching
comparator evidence. Preserve NLL +0.03 aggregate/+0.06 slice, capability
loss 0.02, retrieval/language criteria, uncertainty and provenance rules.
Full-216 remains optional and human-initiated only. No whole-model BF16 run,
repeated medians, exhaustive ablations, tile searches or new evidence framework.

Success means an EVAL-01 accepted, capacity-feasible engine with all nine
matched latency ratios at least 1 versus the retained comparator; also report
memory ratios, TTFT, load cost and residual gaps. A decision task can complete
with RETAIN_CONTROL or unmet speed targets, but must say the overall fast-engine
goal remains unmet. Do not relax quality to obtain a speed milestone.

Track codes/scales/padding, state, activation/output slabs, library workspace,
load transients and allocation peaks; require the 2 GiB reserve at 32,768 + 128
slots. Recalculate from the actual selected layouts, not TASK-019 estimates.
Use existing command/result records and identify instrumentation/first use.
Follow repository `wake-run` rules for future main-thread long commands.
This planning turn stops after document consistency checks.
