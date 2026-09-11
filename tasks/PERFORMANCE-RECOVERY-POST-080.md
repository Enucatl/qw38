# Production numerics v3 after OPT-080

Status: proposed implementation batch OPT-081 through OPT-085. Source review
at Quartz `f577542c0108c8764da40aa692ec44026dab755a`, 2026-09-11. No new GPU
sitting, kernel keep, or throughput claim is made by this plan. Authority
remains llama.cpp `cc83d7b4824f73cfdda4dfbb47ee39804f71b328`, GGUF SHA-256
`31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`.
Authenticate the artifact rather than infer identity from its historical
filename. Hardware remains RTX 5090 with the existing power/clock policy.

Read [the shared testing strategy](TASK-TESTING-STRATEGY.md),
[the post-069 protocol](PERFORMANCE-RECOVERY-POST-069.md), and
[the 2026-09-11 batch protocol](PERFORMANCE-RECOVERY-2026-09-11.md).
This batch supersedes **active** OPT-059 v2 / OPT-074 candidate-admission
semantics. It does not rewrite those reports, change the v2 contract, or
lower OPT-056 / OPT-016.

## Diagnosis

OPT-074 compared **pinned llama GPU** to independent FP64 on production K/M
and froze v2 ceilings from calibration llama-vs-FP64 error. Held-out llama
GPU then exceeded those ceilings on Q4 and Q8 mixer families:

| Family | llama calib max abs | frozen v2 ceiling | llama held-out max abs | v2_admitted |
|---|---:|---:|---:|---|
| q4_down_k17408 | 0.0031 | ~0.0039 | 0.1035 | false |
| q4_gate_up_k5120 | 0.0089 | ~0.011 | 0.0432 | false |
| q8_mixer_k5120 | 0.0278 | ~0.035 | 0.0394 | false |
| q8_mixer_k6144 | 0.0021 | ~0.0027 | 0.0265 | false |
| q6_vocab_k5120 | 0.0256 | ~0.032 | 0.0313 | true |

That is **reference characterization**: llama's own layer/context-dependent
deviation from FP64. It is not evidence that Quartz candidates are worse than
llama. OPT-075/076 nevertheless recorded `retain_packed` / `missing_evidence`.
OPT-070 recorded `inconclusive` / `opt074_coverage_unadmitted` for installed
Q8 `r2_w2` and MMQ `fma_async_x`. OPT-080 froze that combination under v2.

The user authorizes judging optimized production-K kernels **directly against
pinned llama GPU** on identical real captures. Quartz may be as inaccurate as
llama. llama-vs-FP64 drift must not prohibit testing or keeping a kernel that
matches llama.

## Three independent questions

1. **Production numerical admission.** Does Quartz match pinned llama GPU
   closely enough on the same production inputs?
2. **Model quality.** Do PPL, recurrence incremental NLL, OPT-073 engine
   non-regression, and other applicable quality checks pass?
3. **Performance.** Is the candidate measurably faster and safe under
   component/E2E gates?

A candidate may be numerically admitted but quality blocked; numerically and
quality admitted but performance rejected; or `production_kept=true` while
`release_blocked=true` for a known broader quality/release condition
(absolute task accuracy, OPT-056, OPT-016). Do not collapse these into one
correctness verdict. Do not relabel a known absolute quality failure as a
kernel numerical rejection.

## Hard correctness (unchanged)

Fail closed on NaN/Inf; wrong dimensions, dtype, layout, signedness, or
bounds; stale/aliased staging; incorrect Q8/Q4 staging semantics; integer
overflow; incorrect integer-sum semantics; wrong production dispatch;
candidate accidentally testing a fallback/control path; invalid graph-capture
selector state; causal-state corruption; reset/replay mismatch;
checkpoint/cancellation failure; missing required vocabulary/logits;
candidate/reference provenance mismatch.

Continue distinguishing Quartz `sum_q`, llama GPU `sum_x`, FP32 Q8 staging,
and llama Q4 MMV exact/recomputed integer sums. Never loosen a numerical
tolerance to conceal one of these defects.

## Production numerics v3

Pinned llama GPU is the direct numerical authority for optimized production
projection kernels. For every frozen production case `c`:

```
candidate_error[c] = compare(Quartz_candidate[c], llama_gpu[c])
reference_error[c] = compare(llama_gpu[c], fp64[c])
```

Candidate admission uses **only** `candidate_error` against frozen v3 limits.
FP64 remains required characterization and debugging. llama-vs-FP64 error
does **not** define eligibility or candidate tolerance.

**Deleted from active v3 admission:**
`held_out_llama_vs_fp64 <= calibration_derived_llama_vs_fp64_ceiling`.

A family must not become unadmitted merely because pinned llama's FP64 error
changes between calibration and held-out. OPT-074 v2 remains reproducible.

### Case split (reuse OPT-059/074 IDs)

Calibration: layers 0, 3, 31, 32 at tokens 128, 512.
Held-out: layers 62, 63 at tokens 2048, 4095.
Reuse existing family shapes and real BF16 captures where identities remain
valid. Held-out stays untouched until candidate-budget freeze.

Under v3, calibration validates the candidate-vs-llama rule; held-out tests
generalization under the **same frozen rule**; held-out may not enlarge
tolerance; candidate outputs may never influence their own tolerance;
llama-vs-FP64 held-out drift is reported and does not invalidate complete
authority coverage. Do not implement “held-out defines whatever ceiling
makes the candidate pass.”

### Frozen candidate-vs-llama metrics (before any candidate output)

Metrics: maximum absolute error, RMS, `1-cosine`, nonfinite count. Nonfinite
fails closed; never filter finite elements and recompute.

```
scale(llama) = max(1e-6, max_i |llama_i|)
max_abs(candidate, llama) <= abs_floor + rel_abs * scale(llama)
rms(candidate, llama)     <= rms_floor + rel_rms * rms(llama)
1-cos(candidate, llama)   <= cosine_ceiling   (skip cosine on explicit zero vectors)
```

Frozen constants (OPT-081 contract; candidate-independent):

| Symbol | Value | Rationale |
|---|---:|---|
| `abs_floor` | `1e-4` | Absolute slack for quantization-scale association, not FP64 match |
| `rel_abs` | `2e-3` | Direct match to llama (~0.2% of llama scale), not 1.25× llama-vs-FP64 |
| `rms_floor` | `1e-5` | RMS floor independent of llama-vs-FP64 |
| `rel_rms` | `2e-3` | Same relative intent as abs |
| `cosine_ceiling` | `1e-5` | Directional agreement with llama |

These are **not** `1.25 * llama_gpu_vs_fp64 + 1e-6`. Changing them requires a
new versioned policy, not a silent retune after a candidate fails.

Zero vectors: abs/RMS only. Authority identity (llama revision, GGUF SHA,
tensor, activation hash, K/M/N, layer, token, production dispatch) fails
closed. Wrong layout/path cannot pass because arrays happen to be close.

### Distinct v3 fields

Replace overloaded `v2_admitted` as the **active** gate with:

| Field | Meaning |
|---|---|
| `authority_coverage` | llama GPU exports present for required cases |
| `authority_complete` | full-M N=1 production dispatch; no `--row-limit` evidence |
| `reference_fp64_consistency` | llama-vs-FP64 diagnostics (calib/held-out) |
| `reference_drift` | `low` / `high` classification; does not gate candidates |
| `candidate_testable` | complete authority + identity; independent of FP64 drift |
| `candidate_v3_admitted` | candidate-vs-llama metrics pass frozen v3 limits |

Preserve `v2_admitted`, `v2_reason`, and `held_out_validates` as **v2
diagnostics only**. Example: Q4 down may report complete coverage, calib
llama-vs-FP64 ~0.0031, held-out ~0.1035, `reference_drift=high`,
`candidate_testable=true`.

`reference_drift` is `high` when held-out llama-vs-FP64 max abs exceeds
twice the calibration max abs (or the v2 frozen ceiling); otherwise `low`.
That classifier is diagnostic.

## Historical OPT-070–080 disposition

Do not alter old reports.

| ID | Successor |
|---|---|
| OPT-070 | Reopen through OPT-084 (blocked by OPT-074 coverage) |
| OPT-071 | Historical; reuse repaired instrumentation/replay |
| OPT-072 | Historical; annotate that v3 may make previously blocked arithmetic candidates eligible for reconsideration |
| OPT-073 | Retain as quality-policy basis; keep engine regression separate from absolute accuracy |
| OPT-074 | Historical v2; OPT-082 supersedes active admission semantics. `held_out_exceeds_frozen_calibration_ceiling` means v2 reference inconsistency, not candidate rejection |
| OPT-075 | Reopen under OPT-083 (`retain_packed` remains historically correct) |
| OPT-076 | Reopen under OPT-083 |
| OPT-077 | Do **not** reopen: blocker was inconclusive performance, not v2 coverage |
| OPT-078 | Do **not** reopen: blocker was `performance_rejected` / negative complete saving |
| OPT-079 | Retain `kv_once` unless OPT-085 exposes an interaction |
| OPT-080 | Historical v2 batch gate; OPT-085 is the post-v3 combination |

## Dependency order

```
OPT-081 policy
  → OPT-082 authority reclassification
    → OPT-083 Q4 decision  (independent of OPT-084)
    → OPT-084 Q8/MMQ decision  (independent of OPT-083)
      → OPT-085 combined gate
```

First eligible pending task: **OPT-081**. OPT-083 and OPT-084 may proceed in
parallel after OPT-081 and OPT-082.

## Common implementation acceptance

Each task creates `pins/optNNN_iteration_contract.json`, a focused contract,
host evidence tests, and one owned fixture/report with raw sidecars. Commands
in dossiers are interfaces to implement. Reuse OPT-057. Ordinary pytest stays
read-only and GPU-free.

- Feedback includes incremental build and children within 300 seconds.
- Screen at most control plus two named candidates, 1 warmup + 3 rounds,
  unless a dossier narrows it. One survivor gets acceptance.
- Component acceptance: 3 warmups + 10 independently restored AB/BA rounds
  with the existing positive paired CI and ≥0.10 ms/token or ≥5 ms/P4096
  saving rule from the post-069 protocol.
- Target E2E: five uninstrumented AB/BA pairs (D2048+32 or P4096) with the
  existing 2% one-sided regression bound and cross-workload guards.
- Prove actual linked production dispatch; recapture graphs per selector.
- Identity-cache only evidence whose source/flags/input/state/selectors are
  unchanged. Do not rerun GPU work merely to rewrite reports.
- Record separate `candidate_v3_admitted`, `quality_blocked`,
  `performance_rejected`, `production_kept`, `release_blocked`.

Engine-level llama differential (after local projection admission, before
expensive final P/D): identical prompt/token histories; compare hidden taps,
logits, top-1/top-k where meaningful, continuation, and state reset/replay.
Freeze those limits in the OPT-081/085 contracts **before** candidate
results. Purpose: catch accumulated divergence isolated projections miss.

## Core invariant

Quartz production kernels are judged against pinned llama GPU on identical
production inputs. FP64 remains an independent diagnostic oracle and
structural sanity check, but llama's own layer/context-dependent deviation
from FP64 is not a reason to prohibit testing or keeping a Quartz kernel
that matches llama.
