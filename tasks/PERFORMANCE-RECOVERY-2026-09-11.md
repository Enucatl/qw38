# Performance recovery after OPT-056

Status: Proposed implementation batch, OPT-057 through OPT-069. This document
records source analysis and retained measurements, not a new performance sitting.
The individual dossiers define bounded changes and tests for implementation.
Start with [OPT-057](OPT-057.md); use the [testing strategy](TASK-TESTING-STRATEGY.md).
No kernel, production selector, numerical pin, hardware setting, or historical
result was changed by this planning pass.

## Evidence and findings

Inspected Quartz `cdd60ebd72a618f34c0055c8f7a8f8eae933e50f`, pinned llama.cpp
`cc83d7b4824f73cfdda4dfbb47ee39804f71b328`, sibling llama.cpp
`1945e092030f8668ff93382799502d01490e564d`, and ds4
`c238077a87186381bf626cc531bccffe1fef79e7`. The sibling is `../llama.cpp`,
not `../llama-cpp`. Read pinned code with `git -C ../llama.cpp show
cc83d7b:<path>` or `.cache/authorities/llama.cpp`; do not treat sibling HEAD as
the performance authority. Reference-derived code needs revision and MIT
attribution, with no wholesale backend import.

Measured history: [OPT-056 report](../evidence/optimization/opt056-performance-gate/REPORT.md)
gives P 2808.50 versus 3263.52 tok/s and D2048 35.72 versus 67.34 tok/s.
Its instrumented FFN intervals are 666.40 ms/prompt and 15.43 ms/decode token.
Mixer intervals are 310.44 ms and 7.10 ms. These are enclosing intervals;
they cannot be subtracted from isolated llama ggml operation measurements.

| Finding | Evidence in current source or retained report | Consequence |
|---|---|---|
| A large Q4 candidate already exists | OPT-046 complete weighted dot 0.01919 ms versus packed 0.06720 ms; rejected 0.000305175781 versus 0.000300 absolute gate | Revalidate arithmetic against independent FP64 and actual llama GPU outputs before another kernel rewrite |
| Current gate/up bypass the Q4 selector | `full_scheduler.cu::execute_ffn` takes `paired_staged`; `launch_q4k_gate_up_swiglu_prequant` is a packed FP32 kernel; `launch_quant_mmv_prequant` also bypasses integer dispatch | A Q4 pin change alone now affects down but leaves gate/up unchanged; test executed paths |
| Arithmetic admission is not implemented consistently | `quant_mmv.cu::production_numerics_optimized_admitted()` always returns false; selected family is strict despite FMA/MMQ/Q8/Q6 keeps | Implement per-family admission and separate retained-reference tests from production tests |
| Numerical calibration is incomplete | OPT-044 records a host replica, no actual llama GPU reduction measurement, and missing held-out llama NLL | Complete calibration; finite-only held-out evidence cannot establish relative quality |
| Decode Q8_1 staging is mislabeled as llama-equivalent | `q4k_decode_dots.cuh::quantize_bf16_q8_1` stores half(sum(integer quants)); pinned `quantize.cu` stores half(sum(original values)); Quartz Q4 reads the field, pinned Q4 MMV recomputes integer sums | Separate layouts/semantics and test large integer sums before interpreting precision failures |
| Functional probes have an unverified scoring assumption | `freeze_quality_inputs.py` appends a bare instruction, without the chat template or assistant answer prefix; all eight tests emit 271; `run_llama_quality_reference.py::CASES` omits these eight | Compare both engines on the exact original prompts, then freeze a valid functional suite independently of candidates |
| Traced scheduler failures predate the Q4 rejection | OPT-046 records NaNs on packed control too; `full_scheduler_test.cu` reruns each token once per trace filter | Reproduce with fresh objects and bounded taps; a NaN is a bug, not a precision concession |
| CUDA build dependency tracking is incomplete | CUDA object rules omit `-MMD -MP`; the final include list contains host depfiles only; several A/B targets are compiled inside pytest | Tight loops need reliable incremental rebuilds and a single explicit target per diagnostic |
| Tiers do not bound every workload | OPT-053 correctness allocates 4096×5120/17408 matrices; decode oracle hard-codes 3+30 runs, 256 output tokens, new session/workspace/graphs per run | Add a production screening tier and a reusable short runner; retain the long protocol only for release claims |
| MMQ resources changed after tile selection | OPT-053 added a second Y tile and retained I=128/J=128, reported occupancy 1; X loading/unpacking remains synchronous | Retune a small tile set, then attempt a separate packed-X pipeline |
| Decode Q8 uses one row per CTA | `q8_decode_dots.cuh::q8_coop_mmv`; pinned `mmvq.cu` supports multiple rows/block and ds4 has `matmul_q8_0_preq_warp8_kernel` | Test row grouping and activation reuse while keeping existing Q8_1 arithmetic |

The 1.7% Q4 envelope miss is evidence to audit the gate, not proof that the
candidate is safe. Likewise, plain completion prompts may explain token 271;
that is a hypothesis until both engines and the tokenizer are checked.

## Hardware and measurement limits

Read-only `nvidia-smi` on 2026-09-11 reported RTX 5090, driver 590.48.01,
32607 MiB visible memory, no compute processes, **400 W enforced power limit**
and 600 W reported default/max. This is an observation of this board, not a
generic 5090 specification. Do not raise the cap as a code optimization. Log
power, clocks, temperature and throttling while both engines run. No fresh GPU
benchmark was performed. `nsys` and `ncu` were absent from the host PATH; this
does not establish whether a profiling container can provide them.

Query `cudaDeviceProp` for SM count, L2 size, registers, shared-memory limits
and opt-in limits. Query kernel attributes and occupancy for each candidate.
Target `sm_120`; do not import SM100-only `tcgen05`/TMEM designs from server
Blackwell examples. NVIDIA's [CUTLASS Blackwell functionality description](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/blackwell_functionality.md)
identifies those examples as SM100. Use runtime attributes for resource limits,
not the name “Blackwell” or a server tuning table.

Estimated scale, not measured DRAM traffic: the 64 layers' three Q4 FFN matrices
contain `64 * 3 * 17408 * 5120 / 256 * 144 = 9,625,927,680` bytes. Dividing by
15.43 ms gives about 624 GB/s of useful weight bytes per enclosing FFN interval.
This is not a bandwidth saturation measurement. Repeated 50.1 MB matrix probes
can have very different cache behavior from cycling all layers. OPT-061 must
measure that difference and relate useful bytes to actual counters when available.

The prefill FFN alone performs approximately 140.2 trillion conventional
multiply/add operations per 4096-token prompt. A 15% reduction in its 666 ms
interval would save about 100 ms; a 30% reduction about 200 ms. Decode needs
roughly 13.1 ms/token removed. Do not add overlapping projected improvements
from separate tasks or equate an isolated 3.5× dot win with an engine speedup.

## Recommended batch and ordering

| Task | Deliverable | Priority / decision |
|---|---|---|
| [OPT-057](OPT-057.md) | Reliable incremental builds and a five-minute screening runner | First; makes every later iteration cheaper |
| [OPT-058](OPT-058.md) | Reproduce scheduler NaNs and repair functional/held-out quality evaluation | Must establish a usable quality baseline |
| [OPT-059](OPT-059.md) | Real GPU numerical calibration and per-family production admission | Prerequisite to arithmetic promotion |
| [OPT-060](OPT-060.md) | Matched full-engine Quartz/llama family attribution | Instrumentation; no speedup gate |
| [OPT-061](OPT-061.md) | Real-input streaming component replay and hardware diagnosis | Removes cache and dispatch ambiguity |
| [OPT-062](OPT-062.md) | Admit existing cooperative Q4 on all three FFN projections | Highest-confidence decode opportunity; no fusion redesign yet |
| [OPT-063](OPT-063.md) | Cooperative integer gate/up/SwiGLU fusion | Reuse OPT-062 arithmetic and shared input |
| [OPT-064](OPT-064.md) | Multiple Q8 output rows per CTA | Next decode projection opportunity |
| [OPT-065](OPT-065.md) | Retune pipelined MMQ tile resources | First prefill experiment; four variants only |
| [OPT-066](OPT-066.md) | Pipeline packed Q4 weight loads | Only after OPT-065; retain Y pipeline |
| [OPT-067](OPT-067.md) | Pair prompt gate/up tiles and fuse SwiGLU output | Conditional on FFN replay savings; two candidates only |
| [OPT-068](OPT-068.md) | Scoped optimized CUDA compilation | Three build variants on one measured family |
| [OPT-069](OPT-069.md) | Combined regression, quality and original llama outcome gates | One batch-level long sitting |

The dependencies are a DAG, not authorization to run multiple GPU benchmarks
concurrently or to spawn agents. OPT-060 can proceed after OPT-057 while quality
work is unresolved. Once OPT-061 is available, OPT-064/065/068 do not wait on
Q4 fusion. A rejected dependency is usable only when it leaves a measured,
working control; a broken runner or unresolved nonfinite baseline is not.

Attention, GDN, microbatching, broad graph capture and Q6 logits are not another
unconditional tuning ladder. Their latest keeps/rejections stand. If OPT-060
finds a new dominant attributable gap, append a concrete task with that evidence.
Weight requantization, reduced vocabulary, sparse attention and speculative
decoding remain outside this batch.

## Numerical policy recommendation

Relax comparisons to the scalar reduction order for optimized production;
keep strict references callable with their original tolerances. OPT-059 defines
a versioned GPU-calibrated policy, before candidate tuning, with 25% headroom
over actual finite llama GPU error and a small absolute floor. This replaces
the unmeasured assumption that a host Q8_1 replica has llama GPU's error.
The 25% is an explicit engineering allowance on primitive error, not on model
quality, and must be reported alongside the old 5% envelope.

Keep aggregate quality at PPL ratio <=1.01 to pinned llama on both 1024-target
spans and recurrence incremental NLL <=0.02. Current retained NLL results do
not justify moving all the way to the legacy 1.05 allowance. Relax exact
teacher-forced top-1 identity when the competing logits are within a frozen,
reference-calibrated numerical margin; report all mismatches. Functional answer
correctness, full vocabulary, zero nonfinites, token/template identity and
transactional state remain hard requirements. Different arithmetic algorithms
may differ numerically; same-path graph replay, checkpoint restoration, token
frontiers and layout mutations still require exact behavior.

Historical pins/fixtures stay historical. Add v2 admission and quality records;
update runtime tests that mistake old selector strings for eternal invariants.
OPT-056's historical failed quality result must not silently become a pass.

## Performance admission recommendation

Separate `screened_out`, `screened_in`, `component_accepted`, `production_kept`,
and `release_passed` in evidence; ledger status alone does not convey these.
Screening measurements guide the next edit, never establish a published speedup.

For new tasks, use the following versioned incremental acceptance after screening:

1. Select at most one survivor on screening inputs. Validate held-out inputs.
2. Collect 3 warmups and 10 paired complete component rounds on rotating weights,
   with AB/BA order alternated by round. Confirm the 95% paired latency interval
   is positive for `control - candidate` (paired Student-t, df=9, two-sided
   critical value 2.262). Use independent rounds as samples,
   not lanes, rows, individual tokens, or inner timing-loop launches.
3. The expected saving using OPT-060 call counts must be >=0.10 ms/decode token
   or >=5 ms/4096 prompt. These are effort thresholds, not speed claims. Smaller
   cleanup-only wins do not justify adding another kernel family in this batch.
4. Run five uninstrumented whole-run pairs on the target: P4096 or D2048 with
   32 predetermined decode tokens. Use fresh/reset equivalent state and separately
   captured graphs for each configuration. Require a one-sided 95% upper bound
   on the mean paired latency regression <=2% of mean control latency (Student-t,
   df=4, critical value 2.132). This permits a statistically clear component win
   whose small end-to-end gain cannot yet be resolved. Report the E2E estimate
   and uncertainty without claiming significant throughput improvement.
5. Run one short control/candidate pair on each other P/D workload as a gross
   guard, retaining 95% throughput floors and 105% observed p95 ceilings. Its
   small-sample p95 is diagnostic, not a tail-latency certification. Run the
   affected structural checks and OPT-059 quality requirements before defaulting.
6. If inconclusive, retain the diagnostic candidate for OPT-069 or reject; no
   repeated sampling until significance. Do not spend a full llama P/D sitting
   on each candidate. Existing frozen protocols keep their 30-run rules.

The old requirement to show statistically significant E2E improvement for
every small component change is relaxed; quality and E2E non-regression remain.
The +5% llama throughput margin is a batch outcome target, not a per-task keep
condition. OPT-069 separately reports parity, robust improvement, and whether
the unchanged OPT-056 +5% gate actually passed. No lowering the historical bar
to mark the blocked task done.

## Shared delivery requirements

Each dossier is implementation guidance, not measured evidence. New paths and
commands named there are deliverables until created. Every implementation adds
one small contract, one result JSON, raw samples/telemetry and a concise report
under its own `evidence/optimization/optNNN-<slug>/`; avoid duplicating raw arrays
across five files. Read-only pytest checks evidence; live runs write to a new
run directory and promote only the current task's evidence deliberately.
Do not regenerate another task's historical fixture or freeze its current pin
as a permanent source-substring assertion.

Use native C++17/CUDA for reference dots and captures, and the existing uv/pytest
tools for orchestration. Any Python implementation follows the Python skill.
Record task-specific reference counts, actual selected paths, workspace bytes,
graph capture identity, toolchain, input hashes, result status and proof limits.
Run changed-interface tests, not a copied list of every previous optimization.
Update the affected existing handbook sections and audit row at delivery.
