# Synchronized runtime timing and NVTX attribution

[Index](README.md) · Implementation tasks: OPT-001, OPT-014, OPT-015, OPT-020, OPT-032, OPT-038, and EDU-037 in
[`implementation_ledger.md`](../implementation_ledger.md) · Contracts:
[`pins/cuda_timing_contract.json`](../pins/cuda_timing_contract.json),
[`pins/cuda_prefill_attribution_contract.json`](../pins/cuda_prefill_attribution_contract.json),
[`pins/opt020_prefill_split_contract.json`](../pins/opt020_prefill_split_contract.json),
[`pins/opt032_decode_oracle_contract.json`](../pins/opt032_decode_oracle_contract.json),
[`pins/opt038_post_ladder_gap_contract.json`](../pins/opt038_post_ladder_gap_contract.json)
· Evidence:
[`cuda/timing_test.cu`](../cuda/timing_test.cu),
[`cuda/prefill_attribution_test.cu`](../cuda/prefill_attribution_test.cu),
[`tests/test_cuda_timing.py`](../tests/test_cuda_timing.py),
[`tests/test_cuda_prefill_attribution.py`](../tests/test_cuda_prefill_attribution.py),
[`tests/test_opt020_prefill_split.py`](../tests/test_opt020_prefill_split.py),
[`tests/test_opt032_decode_oracle.py`](../tests/test_opt032_decode_oracle.py),
[`fixtures/cuda_timing.json`](../fixtures/cuda_timing.json),
[`fixtures/cuda_prefill_attribution.json`](../fixtures/cuda_prefill_attribution.json),
[`fixtures/opt020_prefill_split.json`](../fixtures/opt020_prefill_split.json),
[`fixtures/opt032_decode_oracle.json`](../fixtures/opt032_decode_oracle.json),
[`fixtures/opt038_post_ladder_gap.json`](../fixtures/opt038_post_ladder_gap.json), and
[`evidence/optimization/opt032-decode-oracle/REPORT.md`](../evidence/optimization/opt032-decode-oracle/REPORT.md),
[`evidence/optimization/opt038-post-ladder-gap/REPORT.md`](../evidence/optimization/opt038-post-ladder-gap/REPORT.md)

## Why a normal stopwatch is misleading

A CUDA launch is **asynchronous**: the CPU asks the GPU to run a kernel and can
continue before that kernel finishes. A CPU stopwatch around only the launch
therefore measures how quickly the request was queued, not how long the GPU did
the work. **Synchronization** means waiting until a known GPU marker completes.
Only then is the measured interval finished.

Quartz uses two clocks for two different jobs:

- A **CUDA event** is a timestamp placed into the GPU stream. Two synchronized
  events measure elapsed GPU-stream time around embedding, mixers, FFNs, logits,
  and device-state commit.
- A monotonic **CPU clock** measures host-only operations such as scanning logits
  for greedy sampling and saving a checkpoint. “Monotonic” means the clock only
  advances, so a wall-clock correction cannot produce a negative interval.

These numbers must not be casually added across different requests. BEN-001 now
supplies warm-ups, repeated samples, percentile reporting, and controlled
machine identity as explained in
[`61-benchmark-harness.md`](61-benchmark-harness.md). That harness keeps
throughput samples as ordinary wall times and uses a separate decode-token
`component_probe`; it is not the 2K prefill attribution report. Controlled
cross-runtime comparison remains CMP-002/CMP-003 work.

## What NVTX contributes

NVTX, the NVIDIA Tools Extension API, adds named ranges such as `qw38.gdn` and
`qw38.ffn` to a profiler timeline. A range is a label, not a stopwatch and not a
kernel. It lets a human connect CUDA launches and gaps back to an engine stage.
The implementation places ranges around loading, the complete token, embedding,
each GDN or attention mixer, each FFN, logits, state commit, sampling, and
checkpoint save/restore.

CUDA events answer “how long was this stream interval?” NVTX answers “which
logical engine stage scheduled this work?” Prompt work records those events on
the fused prompt compute stream, not the default stream used by decode. Nsight
Systems is the later timeline tool, while Nsight Compute inspects selected
kernel hardware counters. Live prefill attribution is produced from the timed
`sync_tokens` run itself without requiring a separate Nsight capture.

## The attribution record

`RuntimeTimings` in [`cuda/full_scheduler.h`](../cuda/full_scheduler.h) gives each
value both milliseconds and a `measured` flag. The flag matters: `0 ms` means a
real measurement rounded to zero, while **unavailable** means no such runtime
boundary exists yet.

| Category | V1 ownership at OPT-001 |
|---|---|
| loading | Copy the admitted GGUF bytes into resident GPU storage |
| embedding | Decode one token row and convert it to the FP32 residual |
| GDN | All 48 GDN mixer intervals, accumulated |
| attention | All 16 attention mixer intervals, accumulated |
| FFN | All 64 feed-forward intervals, accumulated |
| logits | Final norm, vocabulary projection, trace copy, and candidate host copies |
| sampling | Host greedy scan of committed logits |
| persistence | Atomic checkpoint save or restore wall time |
| state commit | Copy 16 candidate K/V rows into committed cache storage |
| idle gaps | Token stream span not covered by the named GPU categories |
| graph launch | Unavailable until OPT-003 implements graphs |
| queueing | Unavailable in this scheduler-local record; SRV-001 measures the HTTP gate separately |

The token total starts before embedding and ends after device commit. `idle gaps`
is the non-negative remainder after subtracting embedding, GDN, attention, FFN,
logits, and commit. It includes gaps between recorded stream work; it is not a
claim that the whole GPU was idle.

## Live 2K prefill attribution

`PrefillAttribution` sits beside `RuntimeTimings` and does not rewrite decode
field ownership. It is opt-in: `execute_prompt_chunk` and `sync_tokens` take an
optional trailing pointer that defaults to null, and a null pointer creates no
extra CUDA events on the prompt path. When the pointer is set, the same
production 2048-token `sync_tokens` call is the timed run. Categories come from
synchronized CUDA events recorded on the prompt compute stream, plus a host
monotonic clock around graph launch and the whole evaluated envelope.

This is the mixer versus core split of the former composite GDN and attention
buckets. Live JSON keys are the nine exclusive names `embedding`, `mixer_mmq`,
`gdn_core`, `attention_core`, `ffn_mmq`, `logits`, `commit_sync`, `graph`, and
`other_idle`. Composite `gdn` and `attention` are not summing categories.

| Category | 2K prefill ownership |
|---|---|
| embedding | Host-to-device token copy plus batched embedding decode/widen into the FP32 residual |
| mixer-projection MMQ | Every mixer-block `matrix_prompt`: GDN packed_qkv, value_gate, alpha, beta, output; attention query_gate, key, value, output. FFN gate/up/down stay in FFN/MMQ. Keys: `mixer_mmq` |
| gdn_core | GDN-layer mixer RMSNorm when it runs in the mixer block, `prepare_gdn_gate_rows`, convolution/scan/recurrence, and `gdn_gated_output_rows` |
| attention_core | Attention-layer mixer RMSNorm when it runs in the mixer block, `split_attention_rows`, `launch_attention_prepare_chunk`, and the FP32-to-BF16 convert of attention output rows |
| FFN/MMQ | Fused residual-add-norm, SwiGLU, the three FFN MMQ projections, last-layer residual add, or the ordinary fused prompt-FFN helper that replaces them |
| logits | Final RMSNorm of the last residual row and vocabulary projection on the compute stream |
| commit/sync | Compute-done event, KV scatter, and compute-stream join through `cudaStreamSynchronize` on the fused path |
| graph | Host monotonic time around each `cudaGraphLaunch` of a prompt FFN executable. At 2048 tokens this is measured `0` with zero launches, not unavailable: a 2048-row chunk does not replay the 4096-row prompt FFN graphs |
| other/idle | Non-negative remainder `max(0, wall − exclusive named work)`. Overlapped copy-stream D2H that extends wall time lands here; it is not a tenth GPU interval |

Mixer-projection MMQ is not contiguous with core: each GDN or attention layer
records input-projection `mixer_mmq`, then `gdn_core` or `attention_core`, then
output-projection `mixer_mmq`. The two mixer intervals accumulate into one
TimingValue. Mixer RMSNorm that still launches in the mixer block is core, not
MMQ. Attention-output Q6_K `matrix_prompt` is `mixer_mmq`, not `attention_core`.

`wall` is the host steady clock around the evaluated `sync_tokens` work, not
model upload or graph create. The nine named categories reconstruct that wall
by construction of `other/idle`. Contract and tests use `rel_tol = 1e-4` and
`abs_tol_ms = 0.05`. If exclusive GPU-plus-graph time exceeds wall beyond that
tolerance, the diagnostic fails closed rather than clamping a negative remainder.
`0 ms` with a successful measured flag means empty work, which is the expected
graph result at 2048 tokens. The historical OPT-014 fixture remains the
eight-category snapshot.

Prompt NVTX ranges are labels, not stopwatches. The path always emits
`qw38.prefill_chunk`, `qw38.embedding`, `qw38.gdn` or `qw38.attention`,
`qw38.ffn`, `qw38.logits`, and `qw38.state_commit`, plus nested
`qw38.mixer_mmq` and `qw38.gdn_core` / `qw38.attention_core` around the
corresponding brackets, and nested `qw38.graph_launch` only when a prompt graph
actually launches.

## Measured example and profiler limits

**Measured, RTX 5090:** one position-1 diagnostic token took 60.594017 ms. Its
largest category was FFN at 39.8940468 ms, followed by GDN at 13.7372789 ms,
attention at 4.34175968 ms, logits at 2.43088007 ms, and smaller embedding,
commit, and gap intervals. The attributed sum equals the token span. This single
sample identifies where to investigate; it is not a throughput benchmark.

**Measured, RTX 5090, 2048-token cold prefill:** the retained live report in
[`fixtures/cuda_prefill_attribution.json`](../fixtures/cuda_prefill_attribution.json)
was produced by the timed diagnostic at 2026-09-08T10:46:11Z. One empty-session
2048-token production `sync_tokens` (capacity 131072, one 2048-row chunk) took
41963.8828 ms host wall (~48.80 tok/s). Exclusive compute-stream categories were
FFN/MMQ 32294.9512 ms, GDN 5300.55371 ms, attention 4364.3335 ms, logits
2.86684799 ms, commit/sync 0.576767981 ms, and embedding 0.076063998 ms. Graph
was measured 0 ms with `prompt_graph_launches == 0`. The other/idle remainder was
0.5234375 ms. The eight categories reconstruct wall within the documented
tolerance. Nsight Systems and Nsight Compute were `not_used`. This is
instrumentation of that timed run, not a throughput gate and not llama.cpp
parity.

**Measured, RTX 5090, 2048-token mixer versus core split:** the retained live
report in
[`fixtures/opt020_prefill_split.json`](../fixtures/opt020_prefill_split.json)
was produced by the same timed diagnostic at 2026-09-08T21:12:38Z after the
mixer-projection MMQ split. One empty-session 2048-token production
`sync_tokens` (capacity 131072, one 2048-row chunk) took 2086.2561 ms host wall
(~981.66 tok/s). Exclusive compute-stream categories were mixer-projection MMQ
1199.25122 ms, FFN/MMQ 397.254333 ms, attention_core 273.986725 ms, gdn_core
212.156006 ms, logits 2.33337593 ms, commit/sync 0.590431988 ms, and embedding
0.0736320019 ms. Graph was measured 0 ms with `prompt_graph_launches == 0`.
The other/idle remainder was 0.610351562 ms. The nine categories reconstruct wall
within the documented tolerance. Nsight Systems and Nsight Compute were
`not_used`. This is instrumentation of that timed run, not a throughput gate
and not llama.cpp parity.

OPT-015 consumes those eight OPT-014 percentages as the exclusive 2K category
map. It does not re-time the GPU and does not require Nsight. Rank 1 is
`ffn_mmq` (77.0%), then `gdn` (12.6%), then `attention` (10.4%); `graph` is
measured 0 ms. The ranked sequence is **Proposed** and is not a throughput
gate:
[`evidence/optimization/opt015-2k-recovery/REPORT.md`](../evidence/optimization/opt015-2k-recovery/REPORT.md).

## Exclusive decode attribution

`DecodeAttribution` sits beside `PrefillAttribution` and `RuntimeTimings`. It
does **not** rewrite public decode field ownership. Production
`RuntimeTimings.gdn` / `.attention` / `.ffn` stay composite, and BEN-001
`component_probe` keeps that same composite bracketing. `execute_token` takes an
optional trailing `DecodeAttribution*` that defaults to null. A null pointer
creates no extra CUDA events on the decode path.

When the pointer is set, one production token records nine exclusive summing
categories from CUDA events on the default stream (plus a host monotonic clock
around each decode `cudaGraphLaunch`):

| Category | Decode ownership |
|---|---|
| embedding | Quant row decode plus BF16→FP32 widen |
| mixer_mmv | Every mixer-block `matrix_vector`: GDN packed_qkv, value_gate, alpha, beta, output; attention query_gate, key, value, output. FFN gate/up/down stay in `ffn_mmv` |
| gdn_core | GDN-layer mixer RMSNorm when it actually launches here, GDN prepare/gated-output, and the mixer residual add on GDN layers |
| attention_core | Attention-layer mixer RMSNorm when it actually launches here, attention prepare/query-gate split, FP32-to-BF16 convert, and the mixer residual add on attention layers |
| ffn_mmv | `execute_ffn` or decode graph replay of that FFN |
| logits | Final RMSNorm, vocabulary `matrix_vector`, and D2H of logits/hidden that currently sit in the logits phase |
| state_commit | Attention scatter plus the `cudaDeviceSynchronize` that publishes committed KV |
| graph | Host monotonic time around each decode `cudaGraphLaunch` (same ownership as `RuntimeTimings.graph_launch`). Zero is legal if graphs did not launch |
| other/idle | Non-negative remainder `max(0, wall − exclusive named work)` |

Per GDN or attention layer the path records input-projection `mixer_mmv`, then
`gdn_core` or `attention_core`, then output-projection `mixer_mmv`. Skip a phase
that has no launches. Collect once at the end of the attributed token. If
decode-graph host time overlaps GPU CUDA events, attributed wall is raised so
the nine categories still reconstruct. Contract and tests copy the prefill
remainder rule: `rel_tol = 1e-4` and `abs_tol_ms = 0.05`. If exclusive sum
exceeds wall beyond that, the diagnostic fails closed. Nested NVTX labels
`qw38.mixer_mmv` and `qw38.gdn_core` / `qw38.attention_core` may always emit;
they are labels, not stopwatches.

When both `RuntimeTimings*` and `DecodeAttribution*` are non-null, exclusive
events go only to `DecodeAttribution`; composite `gdn` / `attention` / `ffn` are
derived sums after collect. Attribution diagnostics pass `DecodeAttribution*`
only. Do not drive these exclusive decode categories through `qw38-bench`.

**Measured, RTX 5090:** one exclusive sitting records one attributed production
decode token after a D128 prefix and after a D2048 prefix, plus a refreshed 4K
prefill exclusive report. Those runs perturb execution and are not tok/s
oracles. Live exclusive milliseconds stay in
[`fixtures/opt032_decode_oracle.json`](../fixtures/opt032_decode_oracle.json)
and
[`evidence/optimization/opt032-decode-oracle/REPORT.md`](../evidence/optimization/opt032-decode-oracle/REPORT.md);
this chapter does not replace them. Nsight Systems and Nsight Compute were
`not_used`. This is instrumentation, not a throughput gate and not llama.cpp
parity.

**Measured negative result:** the pinned CUDA 13.0.2 image contains Nsight
Compute 2025.3.1, but the host denied performance-counter access with
`ERR_NVGPUCTRPERM`. Nsight Systems (`nsys`) is not installed in that image.
OPT-002 must retain these facts, obtain an admitted profiling environment, and
use profiler evidence before accepting a fusion. CUDA-event attribution still
works because it does not require privileged hardware counters. OPT-014's 2K
report is admitted without requiring a separate Nsight capture. The later
mixer versus core split is likewise admitted from CUDA events without Nsight.

## Measurement overhead and failure rules

Detailed attribution creates and records many event pairs. Those markers
**perturb** the execution being measured, so the detailed record is opt-in. The
ordinary scheduler retains only its existing whole-compute timing. Optimization
benchmarks must quantify instrumentation overhead and use an appropriate mode.

If creating, recording, synchronizing, or reading an event fails, the diagnostic
returns an explicit status. A missing decode graph measurement or a queue value
outside this scheduler boundary is represented as unavailable, never fabricated
as zero. Prefill graph-at-2048 is measured empty work, not unavailable. Failed
prefill event operations return an error rather than a partial success report.
NVTX ranges are balanced on successful and error paths so a profiler
timeline does not accidentally swallow later work.

## Proof boundary

OPT-001 proves decode category exposure, synchronized ordering, NVTX range
placement, sampling/persistence timing, and explicit unavailable values on this
runtime. OPT-014 records live 2K prefill attribution from stream-aware CUDA
events plus a host-wall other/idle remainder, with graph-at-2048 measured zero,
without requiring a separate Nsight capture. That historical fixture keeps the
eight composite categories. OPT-020 splits mixer-projection MMQ out of those
composite GDN and attention buckets into exclusive `mixer_mmq`, `gdn_core`,
and `attention_core`; it is retained for mixer-versus-core steering and is not
a throughput gate. OPT-015 cites the historical eight-category percentages as
the recovery map; it does not emit a new timed run. OPT-032 adds opt-in
exclusive decode categories on `DecodeAttribution` without splitting public
`RuntimeTimings`; it is instrumentation for D128/D2048 steering, not a
throughput gate and not llama.cpp parity. OPT-038 refreshes those exclusive
categories on current production objects after the OPT-033–OPT-037 ladder and
adds separate OPT-038 attribution diagnostics that store **independent raw
host-wall** milliseconds alongside the unchanged `finish_*_attribution`
adjusted reconstruction. On decode, `finish_decode_attribution` may still raise
`attribution.wall` to `gpu_event_sum + graph` when exclusive CUDA events plus
the host graph interval exceed the host wall; `other_idle` becomes zero after
that raise. `raw_host_wall_ms` is a separate host `steady_clock` around the
attributed `sync_tokens` / `execute_token` (including D2H and commit) and is
never overloaded onto `wall_ms`. `gpu_event_sum_ms`, `graph_host_interval_ms`,
`adjusted_reconstruction_ms`, `wall_raised`, and the nine exclusive
`categories_ms` are stored as different fields. Throughput oracles keep
`attribution: null`; raised or attributed walls are not tok/s denominators.
`finish_decode_attribution` and `finish_prefill_attribution` were not modified.
OPT-038 claims no performance improvement and is not llama.cpp parity. Live
raw-wall identities stay in the report; this chapter does not replace them:
[`evidence/optimization/opt038-post-ladder-gap/REPORT.md`](../evidence/optimization/opt038-post-ladder-gap/REPORT.md).
None of those increments
proves a fusion is beneficial, provides an Nsight report, establishes p50/p95
request latency, or passes the comparative speed gate. SRV-002 now exposes
separately measured queue depth/delay on Chat Completions responses. BEN-001
retains repeated engine samples and keeps its unavailable queue field explicit;
server queue experiments and cross-runtime statistical comparison remain CMP
work.
