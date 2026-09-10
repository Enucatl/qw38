# Performance recovery design after OPT-042

Status: **Proposed**, source analysis and task design only. No kernel, build flag,
runtime, numerical contract, or benchmark result changed in this design pass.
The user explicitly accepts documented accuracy compromises comparable to
llama.cpp and ds4. OPT-044 translates that authorization into testable production
contracts before changing arithmetic. It does not authorize silently changing
historical evidence or reducing model functionality.

## Evidence identity and limitations

Inspected Quartz `74dc79b52bfb7c6f61656428c11c4248702b2125`, pinned llama.cpp
`cc83d7b4824f73cfdda4dfbb47ee39804f71b328`, sibling llama.cpp HEAD
`1945e092030f8668ff93382799502d01490e564d`, and ds4
`c238077a87186381bf626cc531bccffe1fef79e7`. Use `git -C ../llama.cpp show
cc83d7b:<path>` to reproduce pinned-source observations; sibling HEAD is newer
than the benchmark authority. The pinned tree is also under
`.cache/authorities/llama.cpp`. Do not silently benchmark HEAD against pinned
results. ds4's handbook files are locally deleted; this analysis used executable
source, not those missing documents, and did not restore them.

This is an inventory of material differences in the admitted single-session
Qwen path, not a claim to compare every backend or feature in either repository.
No new GPU timing or accuracy experiment was run. **Measured** below means
retained repository evidence; **Source** means directly inspected code;
**Estimated** means arithmetic on measurements; **Proposed** means unvalidated
optimization. Actual llama fusion/dispatch, per-kernel counters, real-input
numeric budgets, and post-OPT-041 exclusive category times still need OPT-043/044.

## How much time must disappear

Latest accepted Quartz numbers come from
[`opt041_fattn_warp_qk.json`](../fixtures/opt041_fattn_warp_qk.json). Same-sitting
llama results are in
[`llama-bench-4k.json`](../evidence/optimization/opt041-fattn-warp-qk/llama-bench-4k.json)
and [`llama-decode-d2048.json`](../evidence/optimization/opt041-fattn-warp-qk/llama-decode-d2048.json).

| Workload | Quartz tok/s | llama tok/s | Estimated equivalent time | Gap |
|---|---:|---:|---|---:|
| 4096 prefill | 2076.98315 | 3187.314548 | 1972 vs 1285 ms per prompt | 1.535x |
| D2048 | 25.2924843 | 67.0912767 | 39.54 vs 14.91 ms/token | 2.653x |
| D128 | 26.1599541 | See same-sitting oracle | 38.23 ms/token Quartz | — |

Equivalent times use reciprocals of mean throughput, not the mean of measured
latencies. Closing the measured gap requires approximately **687 ms per 4K
prompt and 24.63 ms per decode token**. A 9% FFN improvement alone cannot do it.

The most recent *exclusive attribution* predates OPT-039/040/041:
[`prefill`](../evidence/optimization/opt038-post-ladder-gap/quartz-prefill-attribution-4k.json),
[`decode`](../evidence/optimization/opt038-post-ladder-gap/quartz-decode-attribution-d2048.json).

| OPT-038 category | 4K ms | D2048 ms/token | Interpretation |
|---|---:|---:|---|
| FFN MMQ/MMV | 763.71 | 20.25 | Includes surrounding pointwise work; not pure GEMM time |
| Mixer projections | 294.96 | 11.55 | Decode Q8_0 is a separate FP32 kernel |
| GDN core | 576.82 | 2.07 | Prompt inverse hoisting subsequently improved it |
| Attention core | 579.52 | 13.21 | Decode subsequently improved substantially |
| Logits | 1.61 | 1.70 | Full 248320-row Q6_K output, not optional |
| Raw total wall | 2218.06 | 49.15 | Instrumented, not the throughput denominator |

Do not add OPT-040/041 microbench times to these category totals or present
them as fresh attribution. Their isolated complete components suggest roughly
48 × 7.665 = 368 ms GDN and 16 × 34.591 = 553 ms attention, but these are
**estimates on different inputs**, not a reconciled current wall breakdown.

OPT-039 cut the isolated D2048 attention component from 0.8024 to 0.0774 ms,
yet full decode reached only 25.34 tok/s. This is strong evidence that further
attention-only decode work cannot close the remaining gap. OPT-040 supplied
most of the recent prefill improvement. OPT-041 improved full prefill by about
0.8%. OPT-038 was diagnostic; OPT-042 installed nothing.

## Source differences and their consequences

| Area | Quartz today | Pinned llama.cpp / ds4 | Next task |
|---|---|---|---|
| Numeric/build policy | Global `-O2 --fmad=false`; explicit `__fmul_rn` then `__fadd_rn`; strict scalar-oriented decode gates | llama CUDA CMake sets `-use_fast_math`; ds4 NVCC defaults `-O3 --use_fast_math`; packed dots and reordered reductions | OPT-044 |
| Residual RMSNorm | `rms_norm_fp32_to_bf16`, row version, and fused residual versions sum all 5120 squares on thread 0 | llama `norm.cu::rms_norm_f32` uses cooperative reductions and fused norm/multiply | OPT-045 |
| Q4_K decode | `quant_mmv` assigns one warp per output row, serially visits K blocks, unpacks each value into FP32 arithmetic | `mmvq.cu::mul_mat_vec_q` distributes block dots across warps; `vec_dot_q4_K_q8_1` uses packed integer dot products and scale/min correction | OPT-046 |
| Q8_0 decode | `full_scheduler.cu::q8_mmv_bf16` reads direct BF16 activation, dequantizes scalar values, one warp per row | Q8_0 × Q8_1 integer dots in llama; ds4 prequantizes activation once for `matmul_q8_0_preq_kernel` | OPT-047 |
| Q6_K logits | Same packed-load FP32 MMV family; 248320 × 5120 | Packed Q6_K/Q8_1 dots in llama, independent of its FFN fusion | OPT-048 |
| Decode FFN data flow | Gate and up separately quantize identical input; separate output arrays and SwiGLU | llama contains optional gate/up/GLU fusion; ds4 has shared-input/fused FFN paths | OPT-049 |
| Prompt query preparation | Each attention CTA serially prepares query rows using block barriers, RMS, `powf/cosf/sinf`, dual-F16 split; KV partitions repeat preparation | llama prepares norm/RoPE upstream of flash attention; kernel accepts prepared Q and tiled K/V | OPT-050 |
| Prompt attention inner loop | 32 KV rows per step; shared scores; one thread scans a query's scores; dual-F16 QK and P×V work; three serial pairs of GQA heads reload KV | llama holds KQ/VKQ fragments in registers, warp reductions, architecture-dependent tiles and asynchronous stages; conventional F16 arithmetic | OPT-051 |
| Prompt GDN preprocessing | OPT-040 shares inverse norms, but scaled Q/K and `expf(log_decay)` remain inside every column's token loop | llama feeds pre-normalized Q/K to its register-state loop; also computes decay in the loop | OPT-052 |
| GDN state layout | Warp-column prompt loop loads `S[key_lane * value_width + col]`, stride 512 bytes across lanes | llama stores column-major/transposed state, so warp lanes load contiguous floats | OPT-052, as measured secondary variant |
| Prompt MMQ | Already integer MMA, ldmatrix A, shared-Y FFN, 128×128 winner; synchronous staging and explicit unfused scaling arithmetic | llama has mature layout/load/vecdot configuration; ds4 wraps much of this MMQ family directly | OPT-053 |
| Physical batch | 4096 rows through all 64 layers; graphs only for exact-4096 FFN | benchmark llama `n_batch=2048,n_ubatch=512`; different working sets and graph shapes | OPT-054 |
| Graph coverage | 64 FFN subgraphs; mixers/GDN/attention submitted outside; production cancellation can synchronize per layer | llama graph executor and ds4 `g_decode_graph_stream` capture larger stable work | OPT-055 |
| Completion gate | Many tasks can finish with a measured rejection while the engine remains slower | Parity requires actual end-to-end outcomes | OPT-056 and existing OPT-016 |

Sources are `cuda/{full_scheduler.cu,quant_mmv.cu,quant_mmq_mma.cuh,
fattn_mma_f16.cuh,gdn_fused_quality.cuh,scheduler_primitives.cu,mma.cuh}`;
pinned llama `ggml/src/ggml-cuda/{mmvq.cu,vecdotq.cuh,norm.cu,
fattn-mma-f16.cuh,gated_delta_net.cu,mmq.cuh,mmq-config-blackwell.cuh,
CMakeLists.txt}` and `src/models/{qwen35.cpp,delta-net-base.cpp}`;
ds4 `ds4_cuda.cu`, `Makefile`, and
`cuda/mmq/{VENDOR.md,ds4_mmq.cu,ds4_mmq_d2r.cuh,ds4_repack.cu}`.

Important qualifications:

- Pinned llama's `ggml_cuda_should_use_mmvq` has thresholds explicitly marked
  **tuned on RTX 5090**. Specialization is an opportunity, not evidence that
  every Quartz kernel is better. Q4_K/Q6_K fall back to llama's Ampere MMQ
  configuration on Blackwell; FP4-specific hardware does not automatically
  accelerate this GGUF. Do not confuse SM120 with SM100 capabilities.
- Quartz prompt projections already use tensor cores. Repeating “add MMA,”
  “add stream-K,” or the same tile sweep does not address the remaining issue.
- Quartz attention already clips KV traversal to the last live query position;
  it does **not** compute the whole future triangle. Its value is already in
  registers and P×V already uses MMA. The remaining issue is data movement,
  preprocessing, reductions, and extra arithmetic.
- Both selected fused GDN designs can have a serial token recurrence. llama's
  `fused_gdn_ch` name is not proof of a parallel chunk-scan kernel.
  `delta-net-base.cpp::build_delta_net` selects the fused operator for chunks;
  `gated_delta_net_cuda` loops over tokens. A new tensor-core/WY chunked GDN
  algorithm is a possible later research task, not an established missing
  feature or an unconditional next rewrite.
- Q8_0 decode in Quartz currently has **no activation quantization**. Giving it
  Q8_1 is a real new approximation, unlike merely rearranging a Q4_K staged dot.
- BF16 Quartz KV versus F16 llama KV is a numeric and conversion difference,
  not a 2× storage difference. Preserve BF16 KV initially; F16 conversion inside
  optimized attention is covered by the accuracy policy.
- Kernel availability does not prove actual fusion/dispatch. OPT-043 must
  establish what the benchmark ran before assigning savings to llama fusion.
- OPT-038's component protocol mixes prompt association terminology into MMV
  descriptions; `plan.md` and CUD-001/CUD-003 still own current decode gates.
  New production contracts must be explicit, not inferred from that document.

## Why OPT-042 is not a verdict against integer MMV

[`OPT-042`](OPT-042.md) measured real weighted complete times 0.067588 ms
packed and 0.061906 ms integer (about 8.4% time reduction). Its candidate retains
one warp per row, byte assembly, per-256-element subgroup reductions and eight
broadcasts followed by a lane-0 serial fold. That is not llama's full packed-dot
and cooperative-K execution design. It also deliberately excludes Q8_0 and Q6_K.

Both candidate and existing production failed the newly added large synthetic
CUD-001 cases. All tested real cases passed, but layers 3/63 use a normalized
post-prefix hidden vector, **not their own mid-layer activation**. Neither fact
is a model-quality verdict. Audit against a high-precision dequantized dot and
actual layer inputs; preserve the historical rejection. Simply increasing
`3e-4` until this one study passes would be poor validation.

## Accuracy direction authorized by the user

Permit production-only FMA, cooperative FP32 accumulation, block Q8_1 activation
quantization, F16 tensor-core operands with FP32 accumulators, and documented
approximate transcendental functions. Initially keep canonical GGUF quantization,
FP32 residual/state/logits, BF16 persistent activation/KV storage, exact GQA,
full causal attention, and the public transaction/checkpoint model.

OPT-044 must define separate **strict reference** and **optimized production**
contracts. Strict fixtures remain true claims about retained references.
Graph/eager equality is between the *same arithmetic path*. Cross-path output
and recurrent-state comparisons use admitted numeric budgets. Transaction
isolation, layout conversion round trips, frontier/token identity, and checkpoint
round trips within one path remain exact.

Quality admission combines same-input primitive comparisons against an
independent high-precision reference, pinned llama error measurements, held-out
teacher-forced logit/NLL checks, and real greedy/task fixtures. Random benchmark
tokens and permissive ds4 primitive tolerances cannot establish Qwen quality.
See OPT-044 for the concrete default policy and its calibration/freeze sequence.

## Priority and bounded experiments

Execute OPT-043, then OPT-044, then OPT-045–054 in ledger order. Start with
normalization and Q4/Q8 decode because they cover the largest unaddressed sinks.
OPT-050/051 are the principal attention-prefill work. Re-rank remaining *pending*
tasks using new measured milliseconds if attribution changes; record the reason.
OPT-055 subsumes old pending OPT-031 and is intentionally late. OPT-056 is the
end-to-end outcome gate, not another candidate benchmark.

For each component record a target derived from the matched llama measurement:
`gap_ms = Quartz_complete_ms - llama_complete_ms`. Aim for component parity,
then lower overhead or better shape specialization. Do not promise fixed gains
from source inspection. Track remaining full-engine gap after each accepted
change. Do not sum independently measured component speedup percentages.

Previously rejected D2R, persistent stream-K, MMQ stream-K, GDN conv/gate fusion,
PDL, and tile variants are retained evidence, not mandatory reruns. A changed
arithmetic/layout/batch premise and measured remaining bottleneck are required
to reopen one. Do not use a global `--use_fast_math` flag change as a substitute
for removing explicit `__fadd_rn/__fmul_rn` dependencies.

Non-transferable ds4 techniques: MoE routing/sparsity, compressed or sparse
attention, model-specific clamping, expert streaming, multi-GPU pipeline gains,
and speculative decoding. Transfer its direct launcher design, packed loads,
prequantization reuse, graph lifetimes, test structure, and complete-cost
accounting. A complete BF16 weight expansion would be roughly 53.8 GB from the
26.896B parameter inventory, before KV/state: not a viable 32 GB resident path.
Any selective repack must replace or account for storage and satisfy the existing
128K allocation reserve. No new weight quantization/model conversion is planned.

## Shared implementation and evidence protocol

Every linked task is a future implementation instruction. Use this section with
its individual dossier. No subagents or commits unless the user requests them.

1. Read the current code, this design, the task dependencies' final outcomes,
   applicable numerical pins, and the task's named reference functions. Preserve
   strict reference implementations. Use focused CUDA files and simple internal
   launchers; do not introduce a generic backend or vendor a project wholesale.
2. Use C++17/CUDA 13.0.2 and SM120. New optimized functions may use explicit
   `fmaf`/intrinsics or a separate production translation unit with documented
   flags after OPT-044. Strict objects retain their flags. Inspect compiled
   instructions/register usage for claimed DP4A/MMA/FMA/vectorization changes.
3. Add one task contract, one measured fixture, raw samples, and a report under
   `evidence/optimization/optNNN-<short-name>/`. Retain actual input checksums,
   role/layer/position, selected variant, scratch bytes, launch dimensions,
   source revision/license, quality metrics, and keep/reject reason. Use the
   existing OPT-034/039/041 test/harness patterns; contracts must validate
   numerical results and timings, not mostly source-substring assertions.
4. Measure on an exclusive RTX 5090: 3 warmups, 30 alternating component A/B
   samples. Include staging, preparation, reduction/fixup, epilogues, and layout
   conversion in **complete** cost; report kernel-only cost separately. Capture
   *actual* inputs from representative early/middle/late layers in OPT-043.
   Restore identical candidate state before each stateful A/B. Count each real
   projection once per layer: Q4 gate/up/down occur 64 times each; weight a
   combined gate/up shape 128 times and down 64 times, not 128 for each leg.
5. Eligible candidates run unchanged OPT-021 P and OPT-032 D128/D2048 protocols,
   attribution null, graphs created, same-sitting pinned llama. Historical
   OPT-041 denominators remain recorded; add contemporaneous current-production
   control runs, alternated by whole run, to distinguish drift from improvement.
   Require lower complete component time and an end-to-end target improvement
   supported by paired run uncertainty (95% confidence interval on target
   latency difference excludes zero). If three P runs are inconclusive, repeat
   one independent three-run block; if still inconclusive, do not promote.
   Preserve existing 95% cross-workload throughput floors and 105% ceilings on
   both decode p95 measures. Final parity is stricter and belongs to OPT-056.
6. Run OPT-044 quality checks on every changed numerical family and the full
   admitted combination. A performance win cannot waive quality. A rejected
   candidate stays diagnostic or is removed; retain a concise rejection report.
   Update runtime defaults only after acceptance, recapture affected graphs,
   and verify diagnostic selectors match the production selection.
7. Run task-specific pytest and native CUDA checks listed in the dossier.
   For the shared quantization/scheduler gates,
   `QW38_RUN_CUDA_TESTS=1 QW38_CUDA_TEST_TIER=smoke` is the recommended
   implementation-loop command, `correctness` is the reduced numerical gate,
   and `acceptance` is required before recording timing or performance claims.
   Omitting `QW38_CUDA_TEST_TIER` fails closed with an actionable error.
   The two gates share their Docker build and binary executions per pytest
   process. Rebuild the relevant objects in `qw38-cuda:13.0.2` with
   `-w /workspace` before native testing. Run `tests/test_documentation.py` for
   documentation. If Python code is touched use the Python skill, annotations,
   uv, and pytest. Broader checks follow the changed interfaces; no hardware
   throughput claims from skipped or smoke-only tests. Update existing
   chapters/sources and the task's audit row at delivery.

Profilers are allowed by `plan.md`. Prior tasks' `nsight:not_used` records are
historical facts, not a ban on future profiling. Use Nsight Systems for launch
gaps and actual dispatch, targeted Nsight Compute for dominant kernels, and
CUDA events for paired timing. Do not time instrumented runs as throughput.
If counters/tools are unavailable, retain the error and use event component
measurements plus compiler reports; label unavailable stall/traffic claims.

The task sequence is successful only when speed **and** documented model quality
pass. Finishing candidate tasks with rejections does not finish the performance
objective. OPT-016 retains its frozen 2K gate; no 8K/32K/128K speed claim may be
substituted. Numerical stability and memory tests at long context are allowed
and remain distinct from throughput gates.
