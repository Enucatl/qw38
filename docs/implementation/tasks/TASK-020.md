# TASK-020 — Calibrated precision policy and candidate selection

## Status

DONE

## Milestone

M7 — Strategy and feasibility

## Purpose

Select a provisional format, calibration recipe and family precision policy using measured kernel feasibility and independent quality screening.

## Depends on

- [TASK-019](TASK-019.md)

## Normative references

- [Implementation ledger](../task_ledger.md) — OVERALL-01, revised task contract,
  overall decision rules, measurement envelope and migration of prior obligations.
- [Architecture V0](../../architecture/architecture-v0.md) — retained model semantics
  and controls; reopened decisions follow OVERALL-01.
- [EVAL-01 / PERF-01](../../architecture/evaluation-policy-v0.md) — unchanged
  quality criteria and measurement definitions, with task ownership remapped by the ledger.
- [Technology baseline](../technology-baseline.md).
- [Code standards](../code-standards.md).

## Architecture decisions consumed

| Decision | Contract for this task | Authority |
| -------- | ---------------------- | --------- |
| Q-01/Q-02, projection part of P-02 | Calibrated weights, activations and family exceptions | Reopened by OVERALL-01 |
| A-01/A-02/L-01 | Select layout/views while separating logical quantization from physical packing | OVERALL-01 and retained separation |
| Q-03, P-01, S-01/S-02 | BF16 small/sensitive controls, FP32 arithmetic and state | Retained control |

OVERALL-01 supersedes conflicting restrictions in the former task sequence.
Historical EXP-A–H ordering does not constrain this task. Decisions outside
the reopened scope remain binding.

## Starting point

TASK-019 provides real-shape support, correctness, timings and memory budgets. TASK-018 defines separate calibration and screening inputs; no candidate has a full quality pass.

The user-directed method amendment in the ledger replaces TASK-018's
corpus-wide BF16 activation run for this task. Use GPU-resident quantized
llama.cpp models for calibration statistics and DS4-style paired
teacher-forced continuation scores. Build FP4 GGUF variants directly from the
BF16 source where llama.cpp supports the selected tensors. Retain BF16 source
tensor checks for component attribution; QW38 artifact/runtime acceptance
remains TASK-021/022 work.

## Scope

Quantize from BF16 using the frozen calibration inputs. Start with deterministic
block scaling, then test clipping or calibration improvements only where error
requires them. Specify E2M1 encoding/rounding/saturation, block scale encoding,
second-level scale convention, zero/nonfinite behavior, group axis, activation
scale granularity/lifetime, and packed padding. Test independent weight-only,
activation-only and combined perturbations on representative full layers and
representative inputs, including outliers and prefill/decode inputs.

Choose precision by family: MLP gate/up/down, GDN projections, attention
projections and vocabulary head. Keep embeddings, norms, convolution and small
gate/time parameters as BF16 controls. Treat rotations/smoothing as additional
experiments only if needed, with explicit semantic transformations and runtime
cost; do not make them mandatory or train on evaluation outputs. Use component
BF16 references diagnostically without replacing EVAL-01 with a new BF16 suite.
**Exit:** provisional quantizer/activation/family/layout/dispatch policy backed
by quality screening, kernel timings and memory accounting. Choose MXFP4,
selective higher precision or Q4 fallback if NVFP4 is unsuitable. Full-model
acceptance remains pending TASK-022/026.

## Out of scope

Final evaluation-driven scale fitting, blanket activation/state quantization, mandatory rotations or smoothing, production ABI implementation, and replacing EVAL-01 with reconstruction error or a new BF16 evaluation arm.

## Required interfaces and data representation

Version a candidate-policy record with source/tokenizer/calibration identities; selected tensors/families; quantizer and rounding/clipping parameters; scale axis, encoding, convention and lifetime; packing/padding; proposed dispatch and fallbacks; quality diagnostics and memory cost.

## Required semantics and constraints

Generate candidates directly from BF16. Record weight-only, activation-only and combined error separately, including prompt and populated-decode inputs and early/late layers. Use deterministic rounding, saturation, zero and nonfinite behavior. Test activation scales against chunk composition; any rotation/smoothing must preserve equations before quantization and include runtime cost.

Follow the code standards' identity and manifest-only digest policy. Keep
benchmarks separate from correctness checks and record source, binary,
artifact/policy, inputs, toolchain and hardware identities appropriate to each
result. Partial or invalid evidence cannot establish quality acceptance.

## Tuning defaults

Start with deterministic block scaling. Apply clipping or family-specific precision only when development evidence justifies it; retain frozen evaluation thresholds. NVFP4 leads the investigation, with MXFP4, selective higher precision or Q4 as eligible alternatives.

## Expected files/modules

Offline calibration/quantization diagnostics, candidate policy records, independent reference checks and provisional selection report. Larger calibration frameworks remain offline dependencies.

## Tests required

### Unit and contract checks

Encoding/rounding, block/tensor scales, zero/nonfinite behavior, padding, deterministic calibration and calibration/evaluation separation.

### Reference and numerical checks

Weight-only, activation-only and joint perturbations on representative full layers; reconstruction and output errors for outliers, MLP down, GDN/attention inputs, and head. Independent component BF16 controls diagnose errors.

### Integration checks

DS4-style paired target-NLL checks on frozen development continuations for
BF16-derived FP4 GGUF variants and Q4_K_M, prefill/decode activation
distributions and chunk-sensitive scale checks. Reconcile family choices with
measured kernel support and total memory feasibility.

## Benchmark required

Use TASK-019 matched kernel measurements and measure any changed scale/transform/exception cost. Update memory and conversion costs for the proposed policy; do not claim full-request results before integration.

## Acceptance criteria

- [x] Calibration and development screening identities are frozen and disjoint from final evaluation.
- [x] The quantization and scale contracts are explicit and independently tested.
- [x] Per-family choices and weight/activation ablations have representative
      component evidence, with paired development-continuation evidence from
      actual BF16-derived FP4 GGUF variants; QW38 artifact/runtime quality
      acceptance remains pending TASK-021/022.
- [x] The provisional layout/view/dispatch policy has supported kernels, a full memory budget and justified fallbacks.
- [x] Selection/rejection reasons are recorded; full quality acceptance remains assigned to TASK-022/026.

## Architecture blocker rule

A rejected candidate is a recorded result; use the eligible fallback within OVERALL-01 without relaxing acceptance criteria. Missing required exit evidence prevents completion. A conflict outside the reopened decisions requires the full architecture-blocker report defined in the ledger; obsolete Q4-only or experiment-order restrictions are not blockers.

## Completion report

### Result

Provisional policy selected: Q4G64V0 for MLP gate/up/down, Q8G32V0 for
attention, GDN and vocabulary head, and BF16 for embedding, norms,
convolution and small gate/time controls. Activations remain BF16; residual,
reduction, recurrent and dot-product accumulation remain FP32. One resident
weight view is budgeted. NVFP4 and MXFP4 were screened and rejected for this
selection based on component error, paired continuation results, conversion
cost and feasibility. This is development screening and policy selection,
not QW38 artifact/runtime acceptance or an EVAL-01 pass; those remain assigned
to TASK-021/022/026.

The source checkpoint is `.cache/authorities/qwen3.8-27b-transformers`
(config SHA-256 `191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab`,
safetensors index SHA-256 `77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df`);
tokenizer JSON SHA-256 is
`0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`.
The pinned `ghcr.io/ggml-org/llama.cpp:full-cuda13` image digest was
`sha256:93e004aeaddfcab61c219c65159e28557142054f63d27d4427daf178ec663cc6`.
The source conversion is logged in `.codex-wake-run/429b1d9281b0.log`; it
converted the BF16 source to GGUF and did not run BF16 inference. The pinned
llama.cpp build was `e6ab7c1a4`, scorer SHA-256
`21128e522c6c4d91dd8ffe4a47af6ac0dfff1db3f715f93e140e33c55e32e990`,
binary SHA-256
`3c5577019867678a02e9f3bcd09314f7d64c0fa66effd1be2b658284d1e5d789`.
Scoring and benchmarks ran on an RTX 5090 (SM120), driver 590.48.01, with all
model tensors including the embedding placed on CUDA. The frozen data is
Salesforce/wikitext `wikitext-103-raw-v1`, revision
`b08601e04326c79dfdd32d625aee71d232d685c3`: 128 disjoint 512-token training
windows for calibration and 32 validation windows, each with 384 prompt and
128 teacher-forced target tokens. Manifests are
`.cache/task020/tokens/{train,validation}.manifest.json`; no final evaluation
data or answers were used.

### Changes made

The versioned policy and detailed measurements are in
[`task020-policy.json`](../task020-policy.json) and
[`task020-selection.md`](../task020-selection.md). Calibration used the
GPU-resident quantized Q4_K_M model: 496 tensors, 992 sum/count records, and
65,536 observations per record, summarized in
`.cache/task020/calibration-summary.json`; the matrix is
`.cache/task020/calibration-imatrix.gguf`. Calibration and screening identities
are disjoint from final evaluation. The paired full-GPU comparison is
`.cache/task020/full-gpu-comparisons.json` (32 cases × 128 target IDs per
model). The selected sensitive-Q8 proxy averaged 1.947319 nats/token, versus
1.949951 for the controlled BF16-derived Q4_K_M and 1.945632 for the external
Q4_K_M artifact. It won 10/32 cases against the external artifact; this small
screen is a selection signal only.

Component diagnostics cover independent weight, activation and joint
perturbations on representative early/middle/late full layers, prompt and
populated-decode inputs, outliers and the full vocabulary head. Original
FP4 probe results are `.cache/task020/component-ablation.json` (24
matrix/phase results); selected-policy results are
`.cache/task020/selected-ablation.json` (26); head results are
`.cache/task020/head-ablation.json`. After the repair, replay reproduced all
24 projection/phase results byte for byte with head traces present; output is
`.cache/task020/component-ablation-replay.json` and the command record is
`.codex-wake-run/5689952f8f60.log`.

### Tests run

All commands passed:

- `uv run --python 3.12 --with ruff ruff check scripts/task020_*.py tests/test_task020_*.py` — lint passed.
- `uv run --python 3.12 --with ruff ruff format --check scripts/task020_*.py tests/test_task020_*.py` — formatting passed.
- `uv run --python 3.12 --with pytest --with tokenizers python -m pytest -q tests/test_task020_materialize.py tests/test_task020_compare_continuations.py` — 5 passed.
- `python3 -m py_compile scripts/task020_*.py && bash -n scripts/task020_*.sh && git diff --check` — passed.
- Component replay command recorded in `.codex-wake-run/5689952f8f60.log` — 7 focused tests passed, 24 replayed results byte-identical to the original, including the separate head trace.
- JSON, policy and replay consistency checks — passed.

The unit checks cover E2M1 tie rounding and saturation, UE4M3 scale
boundaries, zero blocks, outliers, nonfinite rejection, scale orientation and
chunk sensitivity. This evidence validates the offline diagnostics and replay;
it does not establish integrated QW38 runtime correctness.

### Benchmark results

On the pinned llama.cpp CUDA proxy, three warmed repetitions each used a
384-token prompt and 128-token generation. Raw samples are
`.cache/task020/bench-q4-k-m.json` and
`.cache/task020/bench-sensitive-q8.json`; command log:
`.codex-wake-run/7e604ed30f3d.log`. Mean prompt throughput was 3,214.87
tokens/s (controlled Q4) versus 3,407.17 (selected sensitive-Q8); generation
throughput was 78.16 versus 70.57 tokens/s. This proxy used generated inputs
and does not establish QW38 latency. TASK-019's SM120 real-shape measurements
remain the kernel feasibility evidence; the FP4 CPU-reference activation
packing path reached 507.611 ms at M=1024 while the native GEMM kernel was
0.166 ms, so no fast GPU activation converter is claimed here.

The inventory-derived budget in
[`task020-memory-budget.json`](../task020-memory-budget.json) estimates
19,958.53 MiB resident payload and 24,073.35 MiB peak, with 5,444.65 MiB
headroom after the 2 GiB reserve. This is an estimate, not a measured
integrated QW38 runtime peak. One-view selection avoids a second complete Q4
projection view estimated at 12,923,699,200 bytes.

### Architecture blocker

None. Review history: Astra returned `CHANGES_REQUIRED` on the initial review
and second review; after the user-authorized targeted repair and evidence
refresh, the final review returned `PASS` on the third review pass. The final
review found no remaining code findings, acceptance gaps or evidence requests.

### Follow-up observations

Full-model QW38 artifact conversion, production runtime integration, and
EVAL-01 quality acceptance remain TASK-021/022/026 work. End-to-end QW38
latency and measured integrated peak memory are not established by these
llama.cpp proxy and inventory-budget results.
