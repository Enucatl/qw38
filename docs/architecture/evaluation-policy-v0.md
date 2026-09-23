# EVAL-01 / PERF-01 — Evaluation and benchmark policy for Architecture V0

**Decision date:** 2026-09-23. **Status:** selected implementation policy.
**Suites:** `qw38-language-v1`, `qw38-performance-v1`.
**Owners:** implementation TASK-018 (quality), TASK-023 (performance).
This document selects the evaluation contract; it contains no model-quality
measurements and does not mark TASK-018 complete.

## Decision and authority

Adopt DS4's combination of component correctness, reference-continuation
scoring, generated-answer evaluation, and state-continuation regression checks.
Useful generated answers and preservation of reference-continuation likelihood
are both required. Weight reconstruction error, intermediate tensor error, raw
logit MAE, and token-for-token agreement across different implementations cannot
substitute for these model-level checks.

This is the normative selection requested to resolve
`TASK_CONTRACT_INCOMPLETE` for TASK-018. For Architecture V0 it supersedes the
following historical provisions of [quantization-validation.md](quantization-validation.md):
the prohibition on naming a suite or acceptance threshold, the unselected
evaluation/prompt/capability flags, and treating behavioral evidence as merely
secondary. Its language+MTP priority was already superseded by Architecture V0.
The historical document and its generated JSON describe the earlier methodology;
their unselected flags are not blockers for this decision. Calibration corpus,
MTP validation, and a general quality/compression Pareto frontier remain open.

EVAL-01 makes the initial Architecture V0 NLL budgets operational and defines
the capability rule precisely. It does not amend weight formats, precision,
state layout, semantic graph, or experiment ordering. PERF-01 below selects
the additional llama.cpp performance comparison requested by the user.
Evaluation changes require a new suite/policy version and an explicit decision;
results cannot be used to loosen the current suite after inspecting V0 failures.

## What DS4 actually does

Inspected local DS4 revision:
`c238077a87186381bf626cc531bccffe1fef79e7`.
Paths below are relative to `../ds4`; they identify inspected repository files,
not an assertion about a future DS4 release.

| Evidence | Observed strategy | QW38 consequence |
| --- | --- | --- |
| [CONTRIBUTING.md](../../../ds4/CONTRIBUTING.md), correctness and quantization sections | Separate kernel checks, actual long-context recall, and official-continuation NLL | Retain small numerical tests and evaluate the real model |
| [QA_BEFORE_RELEASES.md](../../../ds4/QA_BEFORE_RELEASES.md), section 3 | Teacher-forced continuation and probability checks are release blocking; one sampled answer is insufficient | NLL remains an acceptance gate |
| [Quality-testing README](../../../ds4/gguf-tools/quality-testing/README.md), `score_official.c`, `compare_scores.py` | Fixed prompts and reference continuations; token-weighted NLL, first-token/prefix agreement, top-probability diagnostics | Freeze paired inputs and retain per-case results |
| [Test-vector README](../../../ds4/tests/test-vectors/README.md) | Checkpoint-specific official fixtures and local golden probability slices | References must match the QW38 checkpoint and tokenizer |
| [ds4_eval.c](../../../ds4/ds4_eval.c), `eval_cases`, `build_question_prompt`, answer extractors | Real generation graded against fixed answer keys; 25 GPQA, 25 SuperGPQA, 25 AIME and 17 COMPSEC cases | Reuse the questions and keys with QW38 rendering and explicit grading |
| `tests/ds4_test.c`, long-story recall and session tests | Parse recalled facts; verify persistent-state behavior | Include retrieval and reset/save/restore checks |

The user's interpretation is therefore **partly correct**: DS4 uses generated
answers to test capabilities and does not make internal numerical resemblance
the sole quality criterion. It does **not** de-emphasize all model-level
numerical evidence: continuation NLL and probability agreement are central
release checks. DS4's fixed capability subset is a regression suite, not a
complete run of the named public benchmarks.

Reuse prompts, question data, answer keys, reporting ideas, and extractor test
cases. Do not reuse DeepSeek/GLM completions, token IDs, chat syntax, numerical
thresholds, model-specific caches, or hosted-API quality bands as QW38 truth.
QW38's comparator must reject missing or duplicate case IDs; do not copy DS4's
intersection-only pairing, which could hide missing cases.

## Selected inputs

There is no additional downloadable prose corpus in v1. The NLL corpus is a
fixed set of **QW38 source-generated continuations**, following DS4's approach,
plus the known retrieval answers below. This measures agreement with the source
model on these prompts, not general held-out perplexity. Public questions may
have appeared in model training; make no contamination-free benchmark claim.
All v1 inputs and references are held out from QW38 quantizer calibration and
parameter fitting.

### P100: language, instruction following, and code prompts

Select every row `case_000` through `case_099`, in order, from
`../ds4/gguf-tools/quality-testing/prompts.jsonl` at the inspected revision.
Preserve exact prompt UTF-8 bytes and IDs. File SHA-256:
`007757e6c4b340c209aba8a7e159024ce43b0edb237547cdbc3ca51a4853999a`.
Render one user message without an added system message.

Declare these slices before running the candidate:

- Italian: IDs 010–019, 030–039, 050–059, 070–079, 082, 083, 097–099.
- English: all remaining IDs (including 084, which requests English output).
- Code production: 005, 015, 021, 031, 043, 053, 060, 063, 070, 073, 081, 092.
  This slice overlaps the language slices; never count it twice in the total.

Collect a source reference of at most **24 generated tokens per prompt** for
NLL. Independently generate up to **256 tokens** with the BF16 engine and V0 for
text comparison. Short source references intentionally follow DS4's cheap
continuation screen; they cannot validate long-form reasoning by themselves.

### C92: fixed capability questions

Select **all 92 entries**, in array order, in `eval_cases` in
`../ds4/ds4_eval.c` at the inspected revision. File SHA-256:
`19545bf6c0a55cb91b7e3120344ec69ad4cfb5c87cf91e82ec4191a590013f23`.
Copy `source`, `id`, `domain`, exact `question`, ordered `choice` strings, and
`answer`. Include the explicitly modified GPQA entry under its original label.
The four scored slices are GPQA including that modified entry (25), SuperGPQA
(25), AIME2025 (25), and COMPSEC `compsec-076` through `compsec-092` (17).
These counts and IDs are validation requirements, not sampling targets.

Use DS4's `eval_system_prompt()` and `build_question_prompt()` text at that
revision, then apply the QW38 chat template. Preserve option ordering. Do not
include titles, keys, provenance annotations, or private rationales in prompts.
No few-shot examples, tools, retrieval assistance, retries, or answer repair.
Generate at most **2048 tokens** per case with BF16 and V0, once each.

Also collect a separate source reference of at most **24 tokens per C92
prompt** for NLL. The answer key grades capabilities; a source-generated
continuation is never assumed to be a correct answer key. Report the four C92
NLL slices separately, including arithmetic/reasoning and code comprehension.
COMPSEC checks code understanding; it is not executable code-generation pass@1.

### L12: basic language and answer-format anchors

Use the following exact user strings with no system message. Generate at most
32 tokens. Score by exact UTF-8 equality after removing leading/trailing ASCII
whitespace. No case folding, Unicode normalization, or removal of explanations.

| ID | User prompt | Expected answer |
| --- | --- | --- |
| L01 | `Reply with exactly the word blue.` | `blue` |
| L02 | `Rispondi soltanto con la parola rosso.` | `rosso` |
| L03 | `Text: Mira put the key in the green box. What color is the box? Reply with one word.` | `green` |
| L04 | `Testo: Luca ha tre libri. Quanti libri ha Luca? Rispondi con una cifra.` | `3` |
| L05 | `Translate cat into Italian. Reply with one word.` | `gatto` |
| L06 | `Translate cane from Italian into English. Reply with one word.` | `dog` |
| L07 | `Write the numbers 3, 1, 2 in ascending order, separated by commas with no spaces.` | `1,2,3` |
| L08 | `Reply with a JSON object containing only the key ok with boolean value true. Use no spaces or markdown.` | `{"ok":true}` |
| L09 | `What is 17 multiplied by 23? Reply with digits only.` | `391` |
| L10 | `What is the value of 8 + 7 * 2? Reply with digits only.` | `22` |
| L11 | `In Python, what does len([4, 5, 6]) return? Reply with digits only.` | `3` |
| L12 | `In C, what is the integer result of 7 / 2? Reply with digits only.` | `3` |

These are correctness anchors, not evidence of broad model intelligence.
They are excluded from aggregate NLL to avoid diluting the continuation corpus.

### R: context retrieval

Construct six deterministic prompts at each target prompt horizon
**512, 4096, 32768** tokens: `s` in `{0,1}`, depth `d` in `{0.1,0.5,0.9}`.
For each case, render a single user message with this construction:

1. Start with `Read the archive and remember the locker access code.\n`.
2. Create `n` filler lines, indexed from zero, with exact text
   `Record {j:06d}: archive status is unchanged.\n`.
3. Insert, before filler line `floor(d*n)`, the line
   `The access code for locker L{s} is {731204 + 100*s}.\n`.
4. Append `What is the access code for locker L{s}? Reply with the six digits only.`
5. Choose the largest positive `n` for which the fully rendered/tokenized prompt
   fits the horizon. Freeze actual length, insertion token position, text, and
   token IDs. Require actual length within 64 tokens of the target; a failure
   to construct this is an invalid fixture, not permission to truncate a fact.

Braces describe decimal substitution, not literal prompt text; `\n` means LF.
Depth labels refer to filler-line placement; record actual token-relative depth.
Generate at most **32 tokens**, grade trimmed text against `731204` or `731304`.
NLL teacher targets are the tokenizer encoding of those six digits without
special tokens, appended to the frozen generation prefix. Score every answer
token; do not add EOS to this retrieval target. This directly measures the known
answer, separately from source-continuation agreement.

The **TASK-018 core** requires horizons 512 and 4096 (12 cases). Freeze the
32768 cases at the same time; their execution belongs to TASK-022, when
production prefill exists. This is a selected staged requirement, not an
implementer-selected feasible subset. An optional capacity probe uses the same
six cases at `262144 - 32` prompt tokens if the complete session fits; otherwise
record the memory/capacity limit. Never describe this synthetic retrieval suite
as general long-document understanding.

## Rendering, reference production, and freeze

Use the local source checkpoint and source-oracle software identity recorded in
[TASK-017](../implementation/tasks/TASK-017.md). Its model class is
`Qwen3_5ForConditionalGeneration`; the directory's Qwen3.8 label is not permission
to substitute another checkpoint or model class. BF16 weights and the existing
source attention/precision conventions remain the reference setup.

Apply the checkpoint's own chat template with `add_generation_prompt=true` and
`enable_thinking=false`. Tokenize exactly once, with no extra BOS/EOS beyond the
template. Record the exact rendered text, token IDs, template arguments, special
token IDs, and tokenizer asset hashes. All model runs consume those frozen IDs.
Thinking-mode capability evaluation is outside v1; a non-thinking score is not
comparable to DS4's thinking-mode score.

Greedy means argmax over unmodified FP32 logits, lowest token ID on an exact tie;
no sampling, logit processors, penalties, or forced thinking closure. Stop on
the checkpoint `generation_config.json` EOS ID list or the declared output cap.
Save EOS IDs and cap in the manifest. Include a naturally generated EOS in the
24-token source target and its loss mask, but omit it from rendered answer text.
Record every stop reason; never silently extend a cap after seeing an answer.
Capability cap exhaustion is a scored failure even if an answer appears earlier.
Decode answer token IDs with tokenizer cleanup disabled, removing only the
terminal EOS token. Keep any other generated special-token text visible to the
grader/reviewer; do not hide malformed output with `skip_special_tokens=true`.
Allocate evaluation capacity for the complete prompt plus its output cap, up to
the supported model limit. Never truncate prompts to fit a convenience default;
an allocation/limit failure is missing evidence with its required bytes recorded.

The independent source implementation generates P100/C92 reference token IDs.
Then both the BF16 `.qw38` control and V0 teacher-force **those same IDs**.
This avoids using V0 output to select its own evaluation targets. The source
need not generate full 2048-token capability answers; BF16 and V0 do.

Freeze in this order:

1. Vendor prompt/question data with DS4 revision, hashes, license and dataset
   attribution from its file header. Preserve modified-question provenance.
   Runtime evaluation must not depend on a mutable sibling checkout or network.
2. Materialize L12, every R horizon, template configuration, question keys,
   caps, slices, graders, and this policy in a versioned manifest.
3. Generate source continuations once, without viewing candidate results.
   A reference stopped by EOS is valid; a missing/failed reference is not an
   empty passing case. Preserve even poor source answers.
4. Freeze SHA-256 for prompt bytes, teacher token IDs, loss masks, keys, grader
   implementation, policy, tokenizer/template assets, and the manifest itself.
   Use UTF-8/LF JSONL plus little-endian uint32 token arrays and byte masks;
   hash the stored file bytes. Record manifest-relative paths and lengths.
5. Validate the fixture inventory, then run BF16 and V0. Any later change to
   prompts, keys, masks, rendering, references, or grading invalidates comparison
   under this suite identity and requires a versioned decision and paired rerun.

Source checkpoint identity uses index/config metadata only. Artifact identity
uses the `.qw38` manifest digest and compiler/policy metadata. **Do not hash
checkpoint tensor shards, `.qw38` payloads/scales, or entire model files.**
The [payload digest policy](../implementation/code-standards.md#checkpoint-and-qw38-payload-digest-policy)
continues to apply; these metadata identities do not prove tensor integrity.
Binary, evaluation-fixture, and generated-output hashes are allowed.

## Scoring and acceptance

### Basic correctness and control validity

Existing component mathematical tests, identity reconstruction, token/position
validation, and TASK-017 source/BF16 semantic evidence remain prerequisites.
Require finite logits, complete state commits, valid token IDs, and deterministic
same-schedule replay. Preserve the TASK-017 source comparison; do not introduce
a new arbitrary full-model logit MAE tolerance for this suite.

All L12 and required R cases must pass for **both** BF16 and V0. A BF16 failure
is a control/suite investigation and blocks acceptance; a V0-only failure is a
candidate regression. A zero-correct BF16 C92 slice is insufficient capability
coverage and makes that slice INCONCLUSIVE. Do not declare a powerless control
to be evidence of retained capability.

### Teacher-forced NLL

For prompt `p` and teacher continuation `y`, logits after the last prompt token
score `y[0]`; feed `y[j]` only to obtain logits for `y[j+1]`. Prompt tokens,
padding, and context-only tokens have mask zero. Every target has one owner,
one boolean mask entry, and one score. Reset all state at each case boundary;
never concatenate documents or carry state between prompts. No sliding windows
or overlap are needed for v1.

Compute stable FP32 log-softmax over the **full 248320-token vocabulary**:
`logp[k] = z[k] - max(z) - log(sum(exp(z - max(z))))`.
Accumulate negative log probabilities and counts in FP64. For each model,
`NLL = sum(masked -logp[target]) / sum(mask)`; delta is V0 minus BF16.
Never average unweighted case means. Store per-case numerators and counts.

For the core, P100 + C92 + required R form the aggregate corpus. Require
**aggregate delta <= +0.03 nats/token** and **each declared slice delta <= +0.06**.
Slices are P100 English, Italian and code production; each of the four C92
families; and R separately at each horizon. Also report context bins by actual
prompt length: 1–512, 513–4096, 4097–32768, and above 32768, wherever populated;
each populated bin has the +0.06 gate. Empty bins are untested, never zero.
The 32768 extension has its own +0.06 gate and cannot rescue a failing core
aggregate by dilution. These are retained engineering budgets, not empirical
claims about acceptable degradation for every application.

### Output distributions

At every scored teacher position compute full-vocabulary
`KL(p_BF16 || p_V0) = sum(p_BF16 * (logp_BF16 - logp_V0))`.
Zero reference probabilities contribute zero. Stream rows; retaining all
logits in RAM/VRAM is unnecessary. Save count, mean, p95, maximum and case IDs
of the largest divergences, plus top-1 agreement and top-20 overlap divided by
20. Top-k tie breaking is by token ID. Percentiles use nearest rank.

KL, top-k agreement, target-logprob differences, first-token agreement, and
greedy longest common prefix are diagnostics with **no additional numeric
acceptance threshold** in v1. Non-finite results are invalid and block the run.
Top-20 probabilities must never be presented as full-vocabulary KL. Probability
diagnostics help attribute NLL/behavior failures; raw logit MAE is not a quality
verdict. Save the largest 20 KL positions, breaking ties by case ID and position.

### Generated capability answers

Save complete token IDs and decoded BF16/V0 text before grading. Grade the
generated answer, not the likelihood of multiple-choice options. A grader uses
the last nonempty line after trimming ASCII whitespace; it must begin exactly
`Answer: `. Earlier prose is allowed, later nonempty prose makes the answer
invalid. Do not search for a convenient number/letter elsewhere.

| Slice | Required final value | Score |
| --- | --- | --- |
| GPQA / SuperGPQA | One uppercase option letter present in that question | Exact equality to copied key |
| AIME2025 | One decimal integer in 0–999; leading zeros allowed | Integer equality to copied key |
| COMPSEC | One or more decimal line numbers separated by commas; ASCII spaces around commas allowed; `0` alone means safe | Nonempty subset of the key's accepted lines, with no outside line; duplicate numbers are invalid |

Expand inclusive ranges in COMPSEC **keys**, e.g. `3,13-15` means
`{3,13,14,15}`. These identify alternative acceptable locations, not a demand
to emit every line. Preserve DS4's subset semantics: key `17-20` accepts
`Answer: 20`. Generated ranges are not accepted because the prompt requests
exact line numbers. Accept `0` only for a key whose accepted set is `{0}`;
none of the selected 17 keys is safe. This rule avoids converting valid bug
localization into a false failure. The stricter final-line parsing in QW38 is
intentional and its scores must be labeled with the QW38 grader version.

Every case scores 0 or 1, once. Missing/malformed answers and output-cap stops
score 0; an infrastructure failure is missing evidence, not a model score.
Report per-slice accuracy, paired BF16-pass/V0-fail and BF16-fail/V0-pass counts,
and their IDs. Report the 92-case micro-average, but it cannot cancel a slice
regression. No LLM judge, substring grading, subjective answer repair, or
checkpoint-specific extractor tuning is part of C92.

For every C92 slice and its aggregate, let `loss = accuracy_BF16 - accuracy_V0`:

- PASS the finite-suite regression screen if `loss <= 0.02`.
- If `loss > 0.02` and the lower end of the paired 95% interval is above 0.02,
  FAIL with a demonstrated regression beyond the budget.
- Otherwise INCONCLUSIVE: preserve the regression and block task acceptance.
  Uncertainty is not a waiver for an observed regression above budget.

With 17/25 cases per slice, one net lost answer already exceeds two percentage
points. This intentionally conservative screen does **not** establish population
non-inferiority within two points; report that limitation even when it passes.

### Paired uncertainty

Use 10000 paired case-bootstrap replicates. Resample case IDs with replacement
within each disjoint family, keeping both model results together; recompute
token-weighted NLL and case-weighted accuracy for each replicate. P100 is one
family for the aggregate; C92 has four; R has one per horizon. For an individual
slice resample only its members. Never bootstrap tokens as independent samples.
Report the 2.5th and 97.5th percentiles by nearest rank, without rounding before
gate decisions. NLL gates use point estimates; intervals describe uncertainty.

To avoid an unspecified random generator, derive each draw from SHA-256 of
UTF-8 `qw38-language-v1|{metric_group}|{replicate}|{draw}|{retry}`, with counters
starting at zero and case IDs sorted lexically. Interpret the first eight digest
bytes as an unsigned little-endian integer; rejection-sample below
`floor(2^64/n)*n`, then take modulo `n`. Increment retry only on rejection.
For stratified aggregates, include the family ID in `metric_group` and draw
the original family's number of cases independently for each family.
Record this resampler version. Correlated/translated prompts and tiny slices
limit the inferential value; raw paired counts remain required.

### Open-ended generated text

Produce a side-by-side report for all P100 outputs, with source's short
continuation and BF16/V0 full outputs. Report first-token match and common-prefix
length as descriptive values only. Different valid wording is acceptable.

A reviewer records, for each BF16 and V0 output: requested language followed,
requested format followed, question addressed, a concrete factual/code error
with explanation if present, and degeneration. Each item is `yes`, `no`, or
`not_applicable`, with a note for every negative assessment. Degeneration means
empty output without a requested empty answer, unrelated text throughout, or a
loop of an identical contiguous 8-token block repeated at least four times.
List cap-truncated answers separately. Do not treat a code snippet's visual
plausibility as an executed test or a reviewer impression as benchmark accuracy.

An unadjudicated BF16/V0 difference with an alleged material error leaves this
part INCONCLUSIVE. A confirmed V0-only factual/code error, instruction failure,
or degeneration blocks acceptance; stylistic preferences do not. Record reviewer
identity, decision, and supporting text. This qualitative gate is deliberately
reviewed, not represented as an automatically reproducible scalar score.

## State continuation and execution coverage

For BF16 and V0 independently, run deterministic token streams made by cycling
the P100 `case_000` prompt IDs, truncated at the required lengths. This stream
tests state behavior and is excluded from language-quality averages.

Exercise checkpoints after populated lengths **1, 3, 4, 63, 64, 65, 255, 256,
257**, and continue for eight tokens. For every checkpoint compare an
uninterrupted run, reset plus prefix replay, and snapshot/restore plus suffix.
Require identical logits, greedy tokens, persistent state bytes, position and
population metadata for the same binary/artifact/schedule. Include interleaved
case A, reset, case B, reset, case A; replay A must match a fresh session.
Retain TASK-017's late-failure/poison/reset/restore regression.
Record in-memory versus serialized snapshot coverage; disk persistence is not a
new runtime requirement if only the existing in-memory snapshot API is provided.

TASK-018 uses repeated decode for all prompt ingestion, including BF16's bounded
weight-residency diagnostic. It does not require BF16 model-wide GPU residency.
Reference/candidate runs can be sequential, and cached reference evidence may
be reused only with all identities matching. Slow execution does not permit
silently reducing cases or caps.

TASK-022 reruns the core suite through production prefill followed by decode
and runs R at 32768. Compare the same artifact under repeated decode and
prefill at the checkpoints above with partitions of 1, 63, 64, 65, 255, 256,
and an alternating 63/65 pattern, with tail lengths preserved. Across different
schedules, require correct state positions, the same NLL budgets, capability
and retrieval acceptance, and reviewed generated text; bitwise logits/state
equality is required only for replay of one fixed schedule. Reuse the relevant
component numerical tolerances for component-level boundary checks.

The 32768 reference can be precomputed slowly; absence of BF16 evidence is not
a passing paired comparison. TASK-023 and subsequent experiments must reference
the accepted core plus 32768 extension. If resources prevent a mandatory case,
report missing coverage and BLOCKED. The maximum-capacity probe alone is optional.
No claim covers MTP, vision, sampling, batching, other backends, unrestricted
multilingual behavior, executed code synthesis, or maximum context without
separate evidence.

## Report contract and result states

Emit a machine-readable manifest, per-case JSONL, summary JSON, and readable
Markdown with links to raw generated text. The aggregate result is PASS only
when every mandatory gate passes. Use FAIL for observed correctness/quality
failure, INCONCLUSIVE for insufficient or unadjudicated evidence, and INVALID
for identity/fixture/scorer failures. All three prevent task completion; map
them to TASK-018 BLOCKED with their distinct reasons. A partial smoke run is
explicitly `coverage=partial` and cannot produce an accepted baseline.

Required report fields:

- Suite/policy/grader/template/tokenizer/input/mask hashes; checkpoint metadata
  identity; `.qw38` manifest/compiler/quantization identity; binary digest,
  source revision and dirty state; container/toolchain/GPU/driver identities.
- Mode (`language-only`, MTP disabled), control/source roles, schedule,
  reference-capture identity, precision, generation settings, prompt/target
  counts, context capacity and actual populated length, memory limits.
- Each expected case ID, slice membership, status/stop reason, output IDs/text,
  expected/extracted answer and correctness where applicable, NLL numerator and
  denominator, paired delta, distribution summaries, review annotations.
- Per-slice/aggregate metrics, bootstrap definition/intervals, every gate and
  threshold, state-replay outcomes, covered/untested horizons and reasons.
- Exact commands, run duration and reference setup cost (diagnostic, not a
  throughput benchmark), failures and paths to replay them. Missing values are
  `null` with a reason, never zero. Retain failed cases and both model outputs.

Fix implementation bugs under the current architecture and rerun affected
checks plus the unchanged acceptance suite. A selected-hypothesis failure
requires TASK-018's architecture amendment procedure; it cannot authorize early
EXP-A/B/C/E or an opportunistic precision change.

## PERF-01: matching the local llama.cpp benchmark

Select `models/Qwen3.8-27B-Q4_K_M.gguf` as the required **black-box performance
comparator**. The local file exists and is 18,973,870,432 bytes; availability
does not establish that its tokenizer/checkpoint matches or that a usable CUDA
binary is installed. The inspected local llama.cpp checkout is revision
`1945e092030f8668ff93382799502d01490e564d`. Pin the actual supported revision,
build options, binary digest and effective runtime settings when TASK-023
freezes its measurement manifest; report any revision change from this inspection.
Record GGUF metadata/provenance, architecture, layer dimensions, tokenizer
metadata, quantization labels, file size and path. Confirm the claimed checkpoint
relationship from available conversion/source metadata, not the filename alone;
label missing provenance explicitly. A different architecture is not an eligible
same-model comparator. No new checkpoint or `.qw38` tensor-payload hash is needed.

"Match the benchmark" has two concrete requirements: run comparable workloads,
and report whether QW38 meets a **performance parity target**. This supersedes
the earlier optional/future status of the llama.cpp performance point. It does
not make GGUF the compiler input or its logits a numerical authority. Source
BF16 remains the quality control; different quantization recipes and model
sizes must be visible next to speed results.

### Workloads and measurement boundaries

Run both engines on the same otherwise idle RTX 5090, one active sequence and
one process using the GPU at a time. Require complete language-weight GPU
residency for the comparison; report OOM rather than silently enabling CPU
offload. Disable speculative/MTP decoding, serving concurrency and persistent
cross-request prompt caching. Fix capacity at prompt length plus 128 tokens
on both arms. Record effective KV/state precision, flash-attention mode, graph
mode, host threads, batch/microbatch sizes, clocks/power, memory and driver.
Use each engine's supported production defaults for its internal scheduling,
including llama.cpp's optimized kernels/graphs; these are part of the system
being compared. Freeze the resolved settings before timing, with no post-result
selection of the fastest configuration.

Select one performance token stream: tokenize the exact P100 `case_000` user
prompt **without** chat formatting or special tokens, repeat that ID array,
and take the first `T` tokens. Freeze arrays for each T below and a continuation
of 128 more tokens. This intentionally synthetic workload supplies identical
token/position work; it is not scored for model quality. Validate the GGUF
tokenizer's vocabulary, ID-to-token bytes, special IDs and encode/decode mapping
against the source for all consumed IDs. A mismatch makes token-throughput
ratios INVALID; same-text measurements may be reported separately with both
token counts and no claim of identical token work.

| Row | Inputs and outputs | Timed boundary |
| --- | --- | --- |
| Prefill | T = 256, 4096, 32768; empty state; final-position FP32 logits plus argmax only | Ready empty session to synchronized final logits/argmax; report T/time |
| Populated decode | T = 512, 4096, 32768; state already contains T tokens; feed the same frozen 128-token continuation; one token per call | 128 synchronized decode calls including logits/argmax readout; report 128/time and per-step latency; prefill/restore excluded |
| Complete request | T = 256, 4096, 32768; empty state; generate 128 tokens greedily | Prompt ingestion through production of token 128, including token selection; report TTFT and total latency |

For complete requests, EOS is ignored **only for this fixed-work performance
profile**, with no logit suppression: select the actual argmax, even if it is
EOS, feed it back and continue to exactly 128 outputs. Record divergent output
IDs; identical strings are not required across quantizers. This is distinct
from EVAL-01, which stops on EOS. Token one comes from final prompt logits;
there are exactly **127 subsequent decode calls**, reported separately as
`127 / decode_time`. A populated-decode row has 128 explicit input steps and
must not be mistaken for the complete request's 127-step decode tail.

Tokenization, loading, upload, graph construction, warmup, and reset/restore are
outside steady-state windows and reported separately. Synchronize before and
after timed GPU work; primary latency is host monotonic elapsed time including
launch/readout overhead. GPU-event durations are separately labeled diagnostics.
Prefill excludes intermediate vocabulary projections when they are not requested.
Do not use an HTTP endpoint for one engine and direct inference for the other.

Use at least five untimed warmup runs and twenty measured repetitions per row
per engine, with identical incoming state per repetition. Pair trials by index,
alternate which engine's block runs first, and record run order and thermal
conditions. Reuse engine-specific snapshots or replay outside the timer; do not
transfer state bytes between engines. Report every raw sample, median and p99
(nearest rank; with 20 samples p99 is the maximum and has limited precision).
Measure peak resident GPU memory and total model/state/scratch use separately.

For paired 95% intervals on ratios of medians, bootstrap 10000 paired trial
indices using the EVAL-01 resampler with group `perf/<row-id>`. Throughput is
computed from the same work count and measured time, not averages over unequal
work. Missing runs, errors, offload, identity mismatches or unavailable model
support cannot become zero-speed baselines or infinite speedups.

The stock `llama-bench` is useful supplemental evidence, but its inspected
`test_prompt`/`test_gen` feed synthetic/random token IDs and its combined `-pg`
row is not QW38's greedy request boundary. A default `tg128` at depth zero is
also not decode at a populated length of 4096. Therefore use a small external
adapter around llama.cpp's public model/context/token/logit/state API to consume
the frozen IDs and apply the exact table above. Keep the adapter in QW38's
benchmark tooling; importing llama.cpp kernel implementation is unnecessary.
Retain native `llama-bench` JSON as a sanity comparison if run, label its own
semantics, and never divide its combined pp+tg throughput by QW38 decode speed.

### Target, quality context, and task completion

For each matched row publish QW38/llama throughput or llama/QW38 latency so
values above one always favor QW38. The parity target is **ratio >= 1.00 for
every required row's median**, while EVAL-01 quality passes. Mark the target
demonstrated only when the lower paired 95% bound is also at least 1.00;
mark it unmet when the upper bound is below 1.00; otherwise mark it uncertain.
Also publish p99 and memory ratios; they are required evidence, with no new
tail-latency or memory parity threshold. Do not hide individual losses behind
one geometric mean.

TASK-023 must deliver the complete matched baseline and gap report even if V0
is slower. **Missing comparison evidence blocks TASK-023; missing speed parity
does not block the subsequent authorized optimization experiments.** Otherwise
requiring a faster baseline would prevent the work intended to improve it.
Carry the measured gaps through TASK-024–031 and reassess after the final
experiment. Experiment completion and achievement of the performance target
are separate statuses; an unmet target stays explicitly unmet. A speed gap
never authorizes a quality relaxation or an experiment out of order.

Run the frozen L12, C92, P100 generation and R quality inputs through the GGUF
adapter as contextual evidence before claiming a quality/speed Pareto comparison.
Use the same grading and rendering; record unsupported lengths or identity
mismatches. These scores are informative about the chosen llama quant and do
not replace BF16/V0 gates or require V0 to copy llama's errors. Teacher-forced
GGUF NLL/KL is optional and comparable only if the full token identity matches.
If only timing evidence exists, call it a speed comparison, not equal-quality
performance or a Pareto win.

## Required implementation steps

**Yes: implement EVAL-01 in TASK-018 and PERF-01 in TASK-023. No new predecessor
or architecture experiment is needed.** TASK-018 stays TODO until execution begins. The decision
selects inputs/scoring now; materializing source continuations and their hashes
is an implementation output, not a further normative decision.

Deliver, in this order:

1. Versioned local fixture importer/materializer for P100, C92, L12, R; attribution,
   validators, QW38 rendering, and source-reference capture/freeze.
2. Evaluation driver using the actual V0 decode plan and bounded BF16 control:
   teacher-forced FP32 readout, greedy text generation, reusable reset/snapshot,
   bounded logits processing, and identity-bound reports. Extend the existing
   BF16 diagnostic for generation/evaluation rather than loading the full BF16
   model into VRAM. The current decode CLI emits token IDs and does not implement
   this evaluation contract.
3. Graders and comparison/reporting: paired NLL, full-vocabulary KL, C92/L12/R,
   qualitative review records, complete-inventory checks, uncertainty and gate
   statuses. Python orchestration, if used, follows repository uv conventions;
   inference remains in the C++/CUDA runtime.
4. Focused tests for target alignment/masks, stable NLL/KL, counting, tie-breaking,
   EOS/caps, identity mismatches, missing/duplicate IDs, score extraction and
   malformed answers, statistical fixtures, and same-schedule state replay.
   Then run the required core on the authoritative model and preserve evidence.
5. TASK-022 supplies production-prefill coverage and the frozen 32768 extension.
6. TASK-023 adds the llama.cpp public-API benchmark adapter, frozen performance
   manifest, matching-workload checks, raw timings and parity-gap report, using
   the accepted quality identity. Test timing-window counts, tokenizer mismatch
   rejection, statistics and state restore with small fixtures before model runs.

The existing two-token TASK-017 evidence is useful prerequisite evidence; it
does not supply the frozen corpus, generation graders, long-context coverage,
or the TASK-018 behavioral acceptance result.
