# TASK-018 — Replan contracts and preserve evaluation controls

## Status

DONE

## Milestone

M7 — Strategy and feasibility

## Purpose

Make the new sequence executable with consistent decision authority, preserved controls and a frozen comparison protocol.

## Depends on

- [TASK-017](TASK-017.md)

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
| OVERALL-01 | Reconcile the revised task sequence and reopened decisions | User-directed amendment |
| EVAL-01 / PERF-01 | Preserve quality thresholds, coverage and measurement definitions; remap owners | Policy |
| A-02, Q-03, P-01, S-01/S-02 | Preserve semantics, quantizer/layout separation and initial precision/state controls | Retained control |

OVERALL-01 supersedes conflicting restrictions in the former task sequence.
Historical EXP-A–H ordering does not constrain this task. Decisions outside
the reopened scope remain binding.

## Starting point

TASK-017 is complete. Existing Q4/Q8 decode, BF16 semantic evidence and evaluation tooling exist; the former V0 core gate is incomplete and production prefill does not exist. The task specifications have now been rewritten; the remaining reconciliation and protocol work still needs evidence.

## Scope

Preserve the current artifact, source/build identities, partial evaluations,
profiling results and existing local changes as development evidence. The task
files are synchronized with this replan; reconcile the remaining architecture
decision register, quantization/layout/prefill
plans, technology dependency rationale, and EVAL-01/PERF-01 task references with
OVERALL-01. Distinguish historical V0 controls from proposed candidates.
Inventory which frozen fixtures/references have authenticated provenance and
which require regeneration; retain the 216-case core and six 32768 retrieval
cases, scoring, budgets, and P100 review requirements.

Freeze calibration/development/evaluation separation, the real-shape benchmark
matrix, memory reserve, and comparison protocol. Retain the latest bounded
decode smoke and continuation evidence as controls; do not launch another
exhaustive old-V0 run merely to unlock feasibility work. **Exit:** consistent
task specifications and decision authority, an explicit missing-evidence list,
and a reproducible protocol. The old V0 quality gate remains unpassed; its
candidate acceptance obligation moves to TASK-022/026, not to a waiver.

## Out of scope

Implementing new quantizers or kernels, exhaustive reruns of the old V0 gate to unlock feasibility, threshold relaxation, and marking historical partial evaluations accepted.

## Required interfaces and data representation

Produce an evidence inventory with artifact/build/input identities, provenance, valid coverage, missing evidence and regeneration owners. Freeze a protocol describing disjoint calibration, development screening and final evaluation sets; workload/shape coverage; memory reserve; any additional regression budgets; and measurement/report identities. Existing evaluation harness names may retain TASK-018 provenance without changing their semantics.

## Required semantics and constraints

Keep the 216-case core, six 32768 retrieval cases and P100 adjudication obligations. Preserve immutable evaluation inputs, keys, masks and scoring; inventory reference validity separately from generated-output completeness. Candidate core acceptance is owned by TASK-022, full prefill and 32768 acceptance by TASK-026, and matched performance by TASK-027. Do not fit calibration to final evaluation answers or discard failed/interrupted attempts.

Follow the code standards' identity and manifest-only digest policy. Keep
benchmarks separate from correctness checks and record source, binary,
artifact/policy, inputs, toolchain and hardware identities appropriate to each
result. Partial or invalid evidence cannot establish quality acceptance.

## Tuning defaults

Use the ledger's fixed measurement envelope and EVAL-01/PERF-01 definitions. Freeze proposed comparison budgets before candidate results; no new numerical quality threshold is introduced here.

## Expected files/modules

TASK-018–031 specifications, ledger, affected architecture/quantization/layout/prefill documents, technology dependency rationale, evaluation-policy task references, and evidence/protocol records.

## Tests required

### Unit and contract checks

Check task IDs, titles, milestones, statuses, dependencies, local links and acceptance ownership against the ledger. Check for conflicting normative statements in affected documents.

### Reference and numerical checks

Audit preserved TASK-017 semantic controls and evaluation reference provenance. Review target/mask/count identities and identify invalid reference evidence requiring regeneration; this is not a new full-model evaluation.

### Integration checks

Confirm every old mandatory quality/performance obligation has a revised owner. Confirm the frozen protocol distinguishes calibration, development screening and final evaluation and covers both execution phases. Retain existing smoke and continuation results with their limits.

## Benchmark required

No new throughput benchmark. Preserve the recorded populated-decode profile and distinguish it from slow prompt ingestion and future production-prefill results.

## Acceptance criteria

- [x] TASK-018–031 titles, dependencies, scopes and acceptance criteria match OVERALL-01; prior TASK-018 evidence is retained.
- [x] Affected architecture and policy documents have consistent decision authority and remapped task references.
- [x] The evidence inventory identifies valid controls, incomplete/invalid attempts and regeneration owners without claiming an old-V0 quality pass.
- [x] Calibration/development/evaluation separation, shape/workload matrix, memory reserve and comparison protocol are frozen; TASK-020 must materialize the frozen non-evaluation token manifests before fitting/screening.
- [x] The 216-case core, P100 review and six 32768 cases retain unchanged gates and explicit downstream owners.

## Architecture blocker rule

A rejected candidate is a recorded result; use the eligible fallback within OVERALL-01 without relaxing acceptance criteria. Missing required exit evidence prevents completion. A conflict outside the reopened decisions requires the full architecture-blocker report defined in the ledger; obsolete Q4-only or experiment-order restrictions are not blockers.

## Completion report

### Result

DONE — architecture and policy ownership, evidence inventory, and the frozen comparison protocol pass fresh independent review. The selected candidate is the QW38 arm against unchanged Q4_K_M EVAL-01 references; V0 evidence remains a historical control. TASK-020 must materialize the frozen calibration/development token manifests before fitting or screening; TASK-022/026 must rebind preserved evaluation inputs to the reconciled policy identity before acceptance.

### Changes made

Replaced TASK-018–031 specifications and synchronized the ledger's reconciliation notice. Reconciled the architecture decision register, evaluation ownership, quantization/layout/prefill authority and dependency rationale. Added the evidence inventory, downstream owners, protocol rules and current evidence gaps. Follow-up repairs align Architecture V0's operative validation instructions with EVAL-01/PERF-01 and specify the calibration sampler's repository, byte encoding and ordering. Prior TASK-018 evidence is preserved below.

### Tests run

Documentation/contract checks run after reconciliation (separate from runtime
and timing evidence):

- `python3 scripts/check_quantization_design_space.py --config .cache/authorities/qwen3.8-27b-transformers/config.json` — PASS, exit 0; research-space identities/counts and local authority inputs consistent.
- `python3 scripts/check_quantization_validation.py --config .cache/authorities/qwen3.8-27b-transformers/config.json` — PASS, exit 0; historical methodology/JSON identities consistent. Its unselected flags remain historical and are explicitly subordinated by the OVERALL-01 note.
- `python3 scripts/check_layout_strategy.py --config .cache/authorities/qwen3.8-27b-transformers/config.json` — PASS, exit 0; layout dimensions and machine summary consistent.
- `python3 scripts/check_prefill_plan.py --config .cache/authorities/qwen3.8-27b-transformers/config.json` — PASS, exit 0; semantic schedule and machine summary consistent.
- `git diff --check` — PASS, exit 0.

Follow-up repair validation: manually compared the operative Architecture V0
instructions with EVAL-01/PERF-01 and checked the sampler's byte specification
and report/ledger consistency. `git diff --check` passed again. The four
document checkers above were not rerun for these final prose edits; the fresh
independent reviewer requested no additional evidence and independently
confirmed the final candidate's authority, sampler, identities, and report.

These checks establish document/model-config consistency only. They do not
establish model runtime correctness, evaluation acceptance, corpus provenance,
or a benchmark result. Existing runtime/reference checks and exact identities
are preserved in the historical evidence section and inventory below.

### Benchmark results

No new throughput benchmark is required or was run for TASK-018. Retained
benchmark context is development-only: the profile at populated lengths
507–514 measured 33.17 ms traced, 28.61 ms after the documented Q4 projection
changes, and 27.17 ms in the final untraced run; see
[`immediate-decode-profile.md`](../immediate-decode-profile.md) and its SQLite/
Nsight files. Prompt ingestion for that system remained repeated decode.
Selected P10 and one long-C92 timing records in the preserved historical
report used different timing boundaries and are diagnostic, not parity
evidence. No uncertainty interval or paired performance report exists. The
matched whole-request benchmark owner is TASK-027 after TASK-026 quality
acceptance.

### Architecture blocker

None. The reported conflicts were resolved by aligning Architecture V0 with
OVERALL-01 and EVAL-01/PERF-01; no new architecture decision or threshold
change was required.

### Follow-up observations

The remaining downstream evidence is owned by later tasks: TASK-020 must
materialize the exact calibration/development token manifests before
fitting/screening; TASK-022/026 must rebind preserved evaluation inputs to the
reconciled policy identity before acceptance.

### Independent review

Fresh independent review by `gpt-6-sol` at high reasoning: **PASS**, no
findings and no targeted evidence requested. Reviewed candidate:
`bc3e33823b2a638004a00d30e15260990861a76`.

### OVERALL-01 reconciliation, evidence inventory, and protocol freeze — 2026-09-24

#### Reconciliation result

OVERALL-01 now governs reopened candidate decisions and task ownership. The
architecture register labels the old Q4/Q8, packed-layout and geometry choices
as V0 controls; architecture V0, the quantization/layout/prefill/compiler plans, the
technology baseline, code standards and EVAL-01/PERF-01 identify which
historical statements remain controls and which are superseded for
TASK-018–031. Quality criteria and PERF-01 measurement semantics are
unchanged. The old task sequence and requirements to complete old-V0 quality
before feasibility are historical. No architecture blocker is identified.

#### Evidence inventory

| Evidence | Identity and provenance | Valid coverage and limit | Status / regeneration owner |
| --- | --- | --- | --- |
| Source model metadata | HF authority `.cache/authorities/qwen3.8-27b-transformers`; config SHA-256 `191e0af232104ed8b65258cf3fb2b842e288008baca7633c11b82a1ac7203aab`; safetensors index SHA-256 `77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df`; tokenizer SHA-256 `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3` | Metadata/tokenizer identities are authenticated; source tensor payload bytes are not hashed under code standards | Reuse metadata identities; candidate compiler inputs are owned by TASK-021 |
| Historical V0 artifact | `.qw38` manifest identity `ef59127793383c40e0887315c084180eef12907627ebf6bd1a0956cc7b8a0044`, compiler `qw38-v0:0.1.1`, 17,095,659,090 bytes; 408 Q4G64 tensors, 2 Q8G32, 458 BF16, 1 FP32; no payload digest | Valid V0 control identity, not a selected candidate or a quality pass | Preserve; new candidate artifact identity belongs to TASK-021 |
| Evaluation fixture inventory | `.cache/evaluation/qw38-language-v1/manifest.json`; fixture manifest SHA-256 `f08823bc1b7f50646b0daaf6eea801452c03b0a42d32d60ddc43d69365764b1e`; DS4 revision `c238077a87186381bf626cc531bccffe1fef79e7`; prompts SHA-256 `479616b2aefae98b88ec944843a64e1f44a2659c58d2fda241256a53fac56569` | 222/222 unique prompts: P100 100, C92 92, L12 12, R 18; six each at 512, 4096, 32768. Core is 216 cases, with R512/R4096 12. Inputs/keys/masks remain frozen | Bytes and original manifests are preserved. Manifest policy SHA-256 `a6e405af57cb732ea3061ff9ac7d747fec439ba0435f19c57eab2d7ad0cf019b` refers to the pre-reconciliation policy text; rebind the unchanged input identities to the reconciled policy in TASK-022 before acceptance |
| Q4_K_M reference targets | Capture attempt `llama-20260923T153025Z-1465265`; teacher manifest SHA-256 `e4a56d51a77597cffbd27fbbb4851c169afe3b384abb3245c6eb7c46b6259575`; source refs SHA-256 `57faf60a5fb77a5d91b048e1f8becae0b548c5030bc76cafdc13cc798e4c1d76`; container `ghcr.io/ggml-org/llama.cpp@sha256:93e004aeaddfcab61c219c65159e28557142054f63d27d4427daf178ec663cc6`; server binary SHA-256 `ca7289d8434a76b514699eb39258779d0c0e556b1d270d356690e9a7dd16d501`; GGUF 18,973,870,432 bytes, no payload hash; Q4_K_M, CUDA, 65/65 layers offloaded | 192/192 P100/C92 teacher references (24-token cap); exact IDs/masks validate. Six R32768 cases are immutable retrieval inputs, not teacher-generated answer refs | Capture lineage is preserved; rerun/revalidate for acceptance under TASK-022/026 if reconciled-policy binding or comparator identity validation fails |
| Q4_K_M top-20 probabilities | Attempt `probabilities-20260923T155850Z-1474314`; sidecar SHA-256 `1781535511d82d66642cacd0d4386d75d5d6d8039c44827b7f094ec705a0627c`; input teacher manifest above | 192/192 cases and 4,467 aligned teacher targets; top-20 rows only. Full-vocabulary KL is unavailable | Diagnostic/control only; regenerate only if input identity changes. Never claim full-vocabulary KL |
| Interrupted Q4_K_M generated arm | `.cache/evaluation/qw38-language-v1/runs/llama-20260923T160709Z-1476101/`; 216 outputs logged; exit 1 in fresh-request replay | Output completion is not authenticated complete-arm evidence; per-case provenance gap remains | Retain failed attempt; TASK-022/026 regenerate any required comparator run and do not promote the outputs |
| Interrupted V0 full arm | `.cache/evaluation/qw38-language-v1/runs/v0-20260923T170423Z-1492403/`; source revision `3d6601b9ffb48c50af700194ab53d853393b82ee`, dirty tree recorded; artifact above; evaluator SHA-256 `dea24d9609d644f5309334823ab5112a0c6e7fe8f29dcb96728a2746059a426c`; dev image `sha256:3844dc9c37087cecd40f96e62bd4f305ad405408f0d63312bda8aff8651f2b49`; GPU RTX 5090, driver 590.48.01, 32607 MiB; case TSV SHA-256 `a618c0d9b02653a60c2746393437a4fb5de2fa301630956fe408f99d2fd5b028` | Interrupted at 124/216 after 2,796 s, exit 137; 124 output/logit files but only 66 intact case JSONL rows. Not a pass and not a candidate failure | Retain as diagnostic. No old-V0 rerun required. Candidate core evidence belongs to TASK-022 and full-prefill evidence to TASK-026 |
| State continuation control | Reset/snapshot report SHA-256 `ff1033ca1899f9b7cb6df5b040c9940a642e457f84baaf64881892c0b49dd22f`; replay binary SHA-256 `e8041669eea015e83efe3ee47dd16530dc292c063a8f1a25bea3aae9b64e8381` | PASS at lengths 1, 3, 4, 63, 64, 65, 255, 256, 257 and A/reset/B/reset/A interleave | Retained semantic control; rerun on schedule/state changes under affected downstream task |
| Bounded decode and timing controls | `scripts/run_task018_dev_smoke.sh` (two prompts, eight tokens, 25.9 s including load); [`immediate-decode-profile.md`](../immediate-decode-profile.md) and `.cache/decode-profile-20260923.sqlite` / `.nsys-rep` | Populated decode: traced 33.17 to 28.61 ms; untraced 27.17 ms at populated 507–514. P10 and one C92 comparison are speed diagnostics with non-matched timing boundaries. Prompt ingestion used repeated decode | Preserve as development controls only. No new throughput benchmark is required here; production-prefill and matched PERF-01 rows belong to TASK-027 after TASK-026 |
| Human review and paired quality report | No complete pair report; no resolved P100 review artifact | 0/100 P100 adjudications; no candidate core result; old V0 gate is unpassed | TASK-022/026 own candidate reviews and report; do not infer a pass from generated outputs |

The immutable fixture, capture, probability and run files remain unmodified.
The reconciled `evaluation-policy-v0.md` SHA-256 is
`c6e38ae56b7ad180381ddb44d45e4d804fbba007a535a6fb49e9900b11a0b716`.
Because report identity includes the policy hash, the policy-hash change caused
by this reconciliation is a known manifest identity gap. TASK-022/026 must
rebind or regenerate manifests before using the preserved inputs as acceptance
evidence. No source tensor payload digest is introduced.

#### Frozen comparison protocol

- **Final evaluation:** `qw38-language-v1`, 216 core cases: P100 100, C92 92,
  L12 12, and R512/R4096 12. Keep all 100 P100 qualitative adjudications,
  existing answer/NLL gates, uncertainty method, masks and scoring unchanged.
  Freeze all six R32768 cases now; the later TASK-026 policy selects only
  `R-32768-s0-d0.1` for production-prefill evaluation. Candidate core decode
  acceptance is TASK-022; complete prefill and
  long-context acceptance is TASK-026.
- **Calibration:** use Hub repository `Salesforce/wikitext`, configuration
  `wikitext-103-raw-v1`, split `train`, at
  revision `b08601e04326c79dfdd32d625aee71d232d685c3`. **Development
  screening:** use that same dataset revision's `validation` split. The DS4
  `qw38-language-v1` inputs remain final evaluation and are disjoint from both.
  Join each split's raw `text` rows in source order with one LF (U+000A),
  retaining empty rows and adding no trailing separator or text normalization.
  Tokenize the joined text without special tokens or a chat template, using
  the frozen Qwen3.8 tokenizer (`tokenizer.json`
  SHA-256 `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`).
  TASK-020 records the tokenizer implementation and exact version in each
  materialization manifest.
  Form non-overlapping 512-token windows over each split's concatenated token
  stream, dropping a final incomplete window. For each window, hash the UTF-8
  bytes (without a BOM) of
  `qw38-task020-<split>-v1\n<dataset-revision>\n<window-start-token>`.
  Substitute `train` or `validation` for `<split>`, the exact lowercase
  revision `b08601e04326c79dfdd32d625aee71d232d685c3` for
  `<dataset-revision>`, and the zero-based token offset (0, 512, 1024, ...) in
  unsigned base-10 ASCII with no leading zeros except `0` for
  `<window-start-token>`. Each `\n` denotes one LF byte (`0x0a`), not literal
  backslash/`n` characters. Include no angle brackets, quotes, spaces or final
  newline. Rank windows by ascending lexicographic SHA-256 digest bytes,
  breaking any digest tie by ascending numeric token offset. Take the first
  128 `train` windows (65,536 tokens) for calibration and
  first 32 `validation` windows (16,384 tokens) for development. TASK-020 must
  emit and bind the exact raw-row/window/token manifests, sampler source hash,
  tokenizer identity and calibration outputs before fitting or screening.
- **Calibration coverage:** run the BF16 source control on all sampled tokens
  and record input activations for every one of the 64 language layers: all 48
  GDN and 16 attention layers, MLP down projections, GDN projection inputs,
  and attention Q/G/K/V/O projection inputs. For populated-decode samples,
  prefill tokens 0–383 from each 512-token window, then teacher-force tokens
  384–511 one at a time and record the same inputs at populated lengths 384–511.
  Calibration fitting may use only the `train` samples. The `validation`
  samples are reserved for development screening, never calibration fitting.
  EVAL-01 prompts, targets, answers, masks, keys and teacher outputs are
  forbidden to both.
- **Development screening:** use only the 32 frozen validation windows for
  reconstruction, activation-scale sensitivity and early candidate rejection.
  This screen cannot establish any EVAL-01 pass. TASK-020 must materialize and
  authenticate the exact selected rows/windows/token IDs before candidate
  outputs are inspected; no P100/R evaluation case may be substituted.
- **Feasibility/performance workload:** use actual contraction shapes named by
  TASK-019: hidden 5120, MLP 17408, vocabulary 248320, GDN and attention
  projection families; M values 1, 2, 8, 32, 64, 128, 256, 512 and 1024 where
  workspace allows, plus representative tails. Report logical/padded work and
  kernel-only plus complete quantize/pack/GEMM/epilogue costs. This is distinct
  from full-model PERF-01, which keeps prefill and request T=256/4096/32768 and
  populated decode T=512/4096/32768 with the fixed 128-token continuation.
- **Memory:** on RTX 5090, leave at least 2 GiB of allocatable VRAM uncommitted
  at peak after all resident weights, scales/padding, active state/KV,
  workspace, graphs, conversions and second views. Record free VRAM before
  launch and peak used/free memory for each candidate and comparator. If the
  device exposes less than required, reduce workload or candidate residency
  and report it; do not consume the reserve silently.
- **Comparison and budgets:** keep Q4_K_M llama.cpp and historical V0 as
  explicitly identified controls; compare candidate vs frozen EVAL-01
  references under unchanged +0.03 aggregate / +0.06 slice NLL limits, 2
  percentage-point capability regression limit, P100 adjudication, and paired
  uncertainty. PERF-01's ratio target remains >=1.00 for every row's median,
  with paired 95% interval classification as defined in policy. No additional
  numerical quality, speed, tail-latency, or memory threshold is introduced;
  report all rows and individual losses without weighted averaging.
- **Identity/reporting:** correctness/contract checks and benchmarks have
  separate reports. Every result records exact command, source revision and
  dirty state, compiler/build flags, binary digest, artifact manifest and
  quantization identity, input/policy/grader/tokenizer/mask identities, tool
  versions, image digest, GPU/driver/clocks, memory, timing boundaries,
  repetitions, raw output paths, coverage and invalid/interrupted cases.
  Preserve raw failed attempts. Performance report manifests bind the accepted
  TASK-026 quality identity; TASK-027 owns the matched report.

This protocol freezes dataset revision, split ownership, tokenizer, deterministic
sampling, sequence lengths, coverage, required shapes, reserve, unchanged gates
and report identities. Exact calibration/development token manifests are a
TASK-020 precondition; the evaluation-manifest policy-hash rebind is required
before TASK-022/026 acceptance. Neither gap permits a candidate result to be
called a pass.

## Historical evidence — former TASK-018

The following report is preserved verbatim from the former behavioral-correctness
task. Its old task numbers, statuses, next actions and workflow restrictions are
historical, not instructions for the revised sequence. In particular, the former
TASK-022 long-context obligation now belongs to TASK-026; full candidate core
acceptance belongs to TASK-022. No old partial result is promoted to a pass.
The ledger's [preserved evidence](../task_ledger.md#preserved-evidence-from-the-former-task-018)
provides the current summary, including later decode profiling.

### Result
IN_PROGRESS — P10 speed diagnostic complete; full EVAL-01 acceptance remains pending.
### Changes made
Updated EVAL-01 to make Q4_K_M llama.cpp versus V0 the full comparison pair,
with BF16 restricted to optional elementary sanity. Added fixture
materialization, Q4_K_M teacher capture/validation, a top-20 probability-only
replay bound to the frozen teacher IDs, fixed-key graders, paired metric and
report drivers, a full-core V0 evaluator, and a V0 reset/snapshot replay driver.
The paired report keeps teacher full-vocabulary KL unavailable and reports
V0-only fixed-answer NLL for L12/R.

### Independent review and repair record
The independent review returned `CHANGES_REQUIRED`. The single authorized
repair round addressed all seven findings:

1. **HIGH — prohibited whole-artifact digest.** The V0 wrapper hashed and
   propagated a SHA-256 of the complete `.qw38` payload. Removed that digest;
   the run records artifact byte size and embedded schema/quantization identity
   instead.
2. **HIGH — weak lineage and duplicate coverage checks.** The paired reporter
   accepted rows without proving they came from the same frozen inputs and
   collapsed duplicate llama rows. It now binds fixture, policy, grader,
   metrics, teacher, core TSV, target and mask identities; rejects duplicate or
   incomplete cases; requires all 216 unique cases in each arm; and validates
   output, stop/count, and replay identities against the manifests.
3. **HIGH — unreachable PASS and stale manifest status.** The reporter always
   left qualitative review pending and hardcoded the report manifest as
   `INCONCLUSIVE`. Added validated, output-hash-bound `--reviews` adjudications
   for all 100 P100 cases; absent reviews remain `PENDING`, and the manifest
   now records the computed summary status.
4. **MEDIUM — missing C92 paired outcomes.** Added per-slice llama-pass/V0-fail
   and llama-fail/V0-pass counts and case IDs, plus C92 micro-accuracy.
5. **MEDIUM — incomplete teacher probability disclosure.** Added requested
   and effective `n_probs`, selected target logprobs, and explicit top-20-only
   scope to case and summary output. No full-vocabulary KL is claimed.
6. **MEDIUM — missing capability and retrieval coverage detail.** Added L12/R
   per-arm fixed-key grade summaries, R512/R4096 horizon coverage, and an
   explicit deferred R32768 status/reason; teacher NLL remains unavailable for
   L12/R.
7. **LOW — combined reset/snapshot state.** Split replay reset and snapshot
   outcomes into independent fields so one result cannot mask the other.

Repair files: `scripts/run_task018_v0_eval.sh`,
`scripts/run_task018_pair_report.sh`, `scripts/task018_pair_report.py`, and
`src/runtime/state_replay.cpp`.

Post-repair refresh: Ruff passed across TASK-018 scripts/tests; Python
byte-compilation and `bash -n` passed for the harnesses; fixture validation
passed at 222/222 fixtures, 192/192 teacher references, and 192/192 probability
rows; scoring tests passed (9/9); `git diff --check` passed; and the pinned
CUDA build reported no work for `qw38-evaluate` and `qw38-state-replay`.
Post-repair state replay binary SHA-256 is
`e8041669eea015e83efe3ee47dd16530dc292c063a8f1a25bea3aae9b64e8381`; evaluator
binary SHA-256 remains `dea24d9609d644f5309334823ab5112a0c6e7fe8f29dcb96728a2746059a426c`.
The second review returned `CHANGES_REQUIRED`. The authoritative cache-default
llama attempt reached 216/216 generated outputs but exited 1 during
fresh-request replay; it is diagnostic only and is not accepted as completed
arm coverage. No V0 full arm was started.
### Tests run
Exact commands and outcomes:

- `bash scripts/run_task018_llama_capture.sh` — exit 0; attempt
  `llama-20260923T153025Z-1465265`, 116 s wrapper duration, 192/192 P100/C92
  references captured. The complete capture log, server startup log, server
  log, teacher identity and result are under
  `.cache/evaluation/qw38-language-v1/source_capture_job/attempts/llama-20260923T153025Z-1465265/`.
  The container image was
  `ghcr.io/ggml-org/llama.cpp:full-cuda13@sha256:93e004aeaddfcab61c219c65159e28557142054f63d27d4427daf178ec663cc6`;
  server binary SHA-256 was
  `ca7289d8434a76b514699eb39258779d0c0e556b1d270d356690e9a7dd16d501`.
  Startup recorded architecture `qwen35`, GGUF name `Qwen3.8-27B`, `Q4_K_M`,
  tokenizer vocabulary 248320, matching chat-template SHA-256
  `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`, and
  CUDA offload of 65/65 layers. The local GGUF is 18,973,870,432 bytes; its
  payload was not hashed. Teacher manifest SHA-256:
  `e4a56d51a77597cffbd27fbbb4851c169afe3b384abb3245c6eb7c46b6259575`.
- `uv run --script scripts/task018_validate_fixtures.py --fixtures .cache/evaluation/qw38-language-v1 --require-source-refs`
  — PASS; 222 unique cases (P100 100, C92 92, L12 12, R 18), all declared
  retrieval horizons (512/4096/32768, six each), and exactly 192/192 unique
  reference targets and masks validated. The validator accepts llama.cpp's
  `limit` stop only at the exact 24-token cap; all captured cases stopped at
  that cap.
- `bash scripts/run_task018_llama_probabilities.sh` — PASS; cache-default
  attempt `probabilities-20260923T155850Z-1474314`, 115 s, 192/192 cases and
  4,467 aligned teacher targets. All 4,467 targets appeared in returned top-20
  rows; the replay reproduced every frozen teacher ID and stop. Sidecar SHA-256:
  `1781535511d82d66642cacd0d4386d75d5d6d8039c44827b7f094ec705a0627c`.
  It contains selected target log probabilities and top-20 rows, not a full
  teacher vocabulary distribution. A cache-disabled diagnostic attempt diverged
  at compsec-089 and is excluded; exact-default retry matched the frozen IDs.
- `uv run --python 3.12 --with pytest --with numpy==2.5.3 python -m pytest -q tests/test_task018_scoring.py`
  — PASS, 9 tests.
- `uv run --python 3.12 --with ruff ruff check scripts/task018_materialize.py scripts/task018_capture_llama.py scripts/task018_validate_fixtures.py scripts/task018_scoring.py scripts/task018_metrics.py tests/test_task018_scoring.py`
  — PASS.
- `docker run --rm -v /home/user/qw38:/repo -w /repo qw38-dev:cuda13.4.1-pinned cmake -S . -B .cache/task018-build -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF`
  — PASS; GNU 14.2.0, CUDA 13.4.59, native `120-real`.
- `docker run --rm -v /home/user/qw38:/repo -w /repo qw38-dev:cuda13.4.1-pinned cmake --build .cache/task018-build --target qw38-evaluate qw38-state-replay -j 4`
  — PASS. Latest evaluator binary SHA-256:
  `dea24d9609d644f5309334823ab5112a0c6e7fe8f29dcb96728a2746059a426c`;
  state replay binary SHA-256:
  `e8041669eea015e83efe3ee47dd16530dc292c063a8f1a25bea3aae9b64e8381`;
  development image ID `sha256:3844dc9c37087cecd40f96e62bd4f305ad405408f0d63312bda8aff8651f2b49`.
- V0 infrastructure smoke:
  `docker run --rm --gpus all -v /home/user/qw38:/repo -w /repo qw38-dev:cuda13.4.1-pinned .cache/task018-build/src/qw38-evaluate --artifact build/pinned-debug/qwen-v0.qw38 --cases .cache/task018-smoke.tsv --output .cache/task018-v0-smoke --eos-ids 248044,248046,248063,248064,248065`
  — PASS for diagnostic case `case_000`: 26 prompt tokens, 24 teacher target
  tokens, 24 generated tokens, cap stop; wrote FP32 candidate logits and token
  IDs under `.cache/task018-v0-smoke/`. The input artifact was 17,095,659,090
  bytes. The full evaluator now extracts the artifact's embedded manifest and
  quantization identity into each run; the older one-case smoke predates that
  identity record and remains diagnostic only.
- `uv run --script scripts/task018_prepare_core.py --fixtures .cache/evaluation/qw38-language-v1 --output .cache/task018-prepare-check/core.tsv`
  — PASS; generated 216 exact core cases (P100 100, C92 92, L12 12, R512/R4096
  12) and hashed tokenizer-encoded fixed-answer targets for L12.
- `bash scripts/run_task018_llama_eval.sh` — RUNNING at report update; default
  cache settings match teacher capture. Attempt directory
  `.cache/evaluation/qw38-language-v1/runs/llama-20260923T160709Z-1476101/`,
  container `qw38-task018-eval-1476111`, durable `run.log` and server logs. It
  had produced 155/216 case outputs at 16:35 UTC (P100 100/100, C92 55/92); the cache-disabled
  run in `llama-20260923T160557Z-1475336/` produced partial diagnostic output
  for case_000 through case_006 and is excluded from authoritative coverage.
- Host `cmake` invocation was unavailable (`cmake: command not found`); the
  pinned-container configure/build commands above succeeded.
### Benchmark results
No performance benchmark is required by TASK-018. The cache-default llama.cpp
attempt has 216/216 generated cases, but its coordinator-directed stop
interrupted fresh-request replay, so it is not a complete arm. The V0 arm has
not started and no paired quality gate has a result. The teacher top-20 capture is complete at 4,467/4,467
target IDs. Full-vocabulary KL from the llama.cpp REST endpoint is unavailable
and remains `null`; no threshold is claimed as passed or failed.
### Architecture blocker
None observed. The one-case V0 smoke is diagnostic only and does not establish
an architecture blocker or candidate quality failure.
### Follow-up observations
Fixture coverage is 222/222; Q4_K_M teacher-reference and top-20 coverage is
192/192 P100/C92 cases, with 4,467 aligned target positions. The authoritative
cache-default llama attempt wrote 216/216 generated outputs, then was
coordinator-stopped during fresh-request replay and exited 1. These partial
outputs do not count as complete arm coverage. The V0 infrastructure smoke
covers one diagnostic case and does not count toward suite coverage.
R32768 inputs remain frozen at 6/6 for TASK-022. Remaining evidence is the
authoritative llama.cpp run/replays, full V0 core plus state replay, paired
report and human P100 review; full-vocabulary KL remains unavailable.

### Second review and blocker
The same Sol reviewer returned `CHANGES_REQUIRED` on the revised candidate
(HEAD `3d6601b9`). The single authorized repair round resolved the seven
findings above, but the second review found remaining report-contract findings:

1. Invalid fixture or identity validation throws an exception and leaves only
   an exit code. Emit a durable machine-readable `INVALID` result and reason,
   without scoring partial data.
2. The pair summary/manifest/report omit `mode: language-only`,
   `mtp_enabled: false`, comparator/candidate roles, schedule, precision, and
   Markdown links to the raw generated text.
3. Required focused target-alignment, mask, hash-binding, and replay
   failure-path evidence is not recorded.

Full evidence is also incomplete: the coordinator-directed stop ended the
llama attempt with exit 1 when its server disconnected during fresh-request
replay, after it logged 216/216 generated cases;
its outputs and failure log are preserved in
`.cache/evaluation/qw38-language-v1/runs/llama-20260923T160709Z-1476101/`, but
the attempt is not accepted as complete coverage. No full V0 run, V0 state
replay, paired report, or resolved 100-case P100 adjudication exists.

Per `run-implementation-task-codex`, findings remaining after the single repair
round require `BLOCKED`. Do not treat the generated outputs as a complete
llama arm. The coordinator resumed TASK-018 to complete the V0 core/state
replay and top-20 versus top-20 distribution comparison, with llama.cpp
full-vocabulary KL remaining unavailable. See the resumption report below.

### Coordinator-directed resumed work — 2026-09-23

Addressed the remaining report-contract findings in the current candidate:

- Pair-report validation failures atomically produce `result.json` with
  `status: INVALID`, a reason and `scoring_performed: false`; successful
  summary/manifest/report output includes language-only mode, MTP disabled,
  comparator/candidate roles, schedule, precision and relative links to each
  raw generated text.
- Added focused negative checks for teacher-target alignment, loss-mask
  contents, hash binding and fresh-request replay mismatch, plus durable
  invalid-report output. The focused scoring/report tests passed 10/10;
  Ruff, Python compilation, shell syntax and `git diff --check` passed.
- Added an efficient llama replay-completion path. It reused the preserved
  216 generated llama outputs from the prior attempt and repeated the four
  required fresh-request probes; all four matched. Run manifest SHA-256:
  `318c44a32e623b8817a5f0a30a993a518f3c32012e1f08b280a1616c52f8c1f7`.
  The original full attempt and its exit-1 replay failure remain preserved.
- The full V0 run was coordinator-interrupted at 124/216 after 2,796 seconds
  (17:04:23–17:50:59 UTC), exit 137. Its partial result reports 124 generation
  outputs and 124 candidate-logit files. `interruption.json`, original
  `result.json`, logs and outputs are retained under
  `.cache/evaluation/qw38-language-v1/runs/v0-20260923T170423Z-1492403/`.
  This is incomplete diagnostic evidence, not a V0 quality failure.
- Independently ran the V0 reset/snapshot replay: PASS for lengths
  1, 3, 4, 63, 64, 65, 255, 256 and 257; all reset and snapshot checks were
  true, and the `A,reset,B,reset,A` interleave matched. Result SHA-256:
  `ff1033ca1899f9b7cb6df5b040c9940a642e457f84baaf64881892c0b49dd22f`.
- Ran a prompt-content-selected P10 speed-only diagnostic on case_080,
  case_082–084, case_087 and case_094–098, preserving each frozen 256-token
  cap. V0 completed 10/10 in 47 seconds including artifact load, generation and
  teacher-target logit scoring (755 generated tokens; 16.06 tokens/s end to
  end). A fresh Q4_K_M llama.cpp run completed the same 10 prompts in 12.307
  seconds of requests (0.813 requests/s; 54.85 generated tokens/s), with 24.222
  seconds startup-to-ready and 36.529 seconds launch-to-final-request, excluding
  server shutdown. The V0 and
  llama wall-time scopes differ because V0 also scored the frozen targets.
  Combined diagnostic report:
  `.cache/task018-p10-selected/report.md` and
  `.cache/task018-p10-selected/paired-speed-summary.json`.
- Reran the same P10 on V0 without target/mask inputs to isolate generation
  from teacher-logit scoring. It completed 10/10 with the same 755 generated
  tokens in 44.565 seconds launch-to-exit (16.94 tokens/s), including
  container/model load. The input identity is bound to the P10 selection and
  frozen prompt hashes; no teacher logits were written. Evidence:
  `.cache/task018-p10-generation-only/run-summary.json`, `run.log`,
  `cases.jsonl`, and `input-identity.json`. This remains a diagnostic, not a
  performance-parity or quality result.
- Extended the diagnostic with one long C92 example, `aime2025-16`, at its
  frozen 2048-token cap. Both arms reached the cap. The V0 observed completion
  interval was 71.64 seconds; the Q4_K_M completion request took 32.22 seconds.
  The V0 interval includes session/prompt work and log observation, while the
  llama request timing includes tokenization verification and generation; this
  is indicative, not a calibrated speed-parity measurement. The 11-case
  totals were 2803 V0 tokens in 107.25 seconds launch-to-exit and 2723 Q4_K_M
  tokens in 44.26 seconds of requests (16.02 seconds startup). Report and raw
  evidence are under `.cache/task018-p11-long/`; no `.qw38` payload digest was
  recorded. This remains diagnostic-only evidence.

The subsequent review found that the replay-completion path could not prove
per-case provenance for the failed source attempt. That reuse path was removed;
the preserved replay result is diagnostic only and does not establish a complete
llama arm. Report repairs now gate L12/R fixed answers, bind the pair-report
driver hash, and distinguish interrupted V0 log progress from durable JSONL
rows. The interrupted V0 run logged 124 completed cases, but only 66 case
JSONL rows survived intact; 124 generated and candidate-logit files remain.

The full V0 arm, paired top-20/NLL report, capability grading and 100 P100
qualitative adjudications remain incomplete. The P10 timing is not a quality
gate or substitute for the required 216-case core. Full-vocabulary KL remains
unavailable. Fresh independent Sol review passed for the repair round and
confirmed the code fixes and focused evidence; it also confirmed that full
TASK-018 acceptance remains incomplete because the full arms and P100
adjudications were stopped.
