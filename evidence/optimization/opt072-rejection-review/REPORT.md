# OPT-072 — Reevaluate every correctness-based optimization rejection

Status: **review complete, no production pin change**. `claims_throughput:
false`. No speedup is claimed. Authority remains llama.cpp
`cc83d7b4824f73cfdda4dfbb47ee39804f71b328` and GGUF SHA-256
`31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`.
Historical fixtures and rejection reports were referenced, not modified.

## Why this sitting exists

OPT-001 through OPT-069 contain keeps, performance-only losers, admission-blocked
faster kernels, and a few genuine hard errors. A looser assertion does not
accelerate code; it can only reopen an existing faster implementation after
reference-derived v2 budgets exist. This task inventories every rejection,
audits Q4 Q8_1 sum consumers, and records owners. It does not install a kernel.

## Register coverage

The machine-readable register is `fixtures/opt072_rejection_review.json`. Every
OPT-001–OPT-069 task has at least one entry. Variants inside otherwise kept
tasks are separate rows. Each row records original test/reference, metric,
envelope, source, latest equivalent path, disposition, and next owner.

Dispositions used:

| Disposition | Meaning |
|---|---|
| `kept` | installed or protocol keep; not a rejection to reopen |
| `no_repeat` | performance-only loser; tolerance cannot rescue it |
| `already_superseded` | current equivalent exists (do not rebuild the old variant) |
| `numerically_eligible` | staged/current typed checks pass; missing production admission |
| `still_valid` | original failure still stands as a hard or historical rule |
| `blocked` | unresolved dependency with a named owner |

Classification of remaining correctness candidates uses OPT-059's frozen
reference-derived v2 rule `max(strict, 1.25 * llama_gpu + 1e-6)`. No
candidate-fitted epsilon. Synthetic M17/K256 llama GPU vs FP64 original
max_abs **19.0954106** is Q4 quantization error and is **not** copied onto
production K. Production-K llama GPU error remains **unadmitted** (OPT-074).

## Q8_1 consumer audit (behavior, not policy)

Quartz `quantize_bf16_q8_1` stores `half(sum(q))`. Distinct
`quantize_bf16_q8_1_sum_x` stores `half(sum(x))`. Exact int32 sums of the 32
quantized bytes are a third quantity. IEEE half is exact through 2048: **2047**
survives, **2049** loses a unit (this host replica stores 2050; CUDA
`__float2half_rn` may store 2048), and **4064** remains exact. Either neighbor
fails to match exact int32 2049. Negative counterparts match.

The Q4 `UseQ81` consumer treats the stored half field as integer `sum(q)`.
A `sum(x)` producer must not feed that consumer. Production Q8 DP4A (OPT-047/064)
and Q6 integer (OPT-048) read scale and values only; their staging is unchanged.
Pinned llama Q4 MMV recomputes integer sums. The focused typed repair is the
pairing constants in `cuda/q4k_decode_path.cuh`
(`kQ81ConsumerExpects`, `kQ81IllegalPairing`).

## Original Q4 failure

Smallest historical failing input: `q4_k_17x256`. Cooperative integer
`integer_q8_w4` measured max_abs **3.05175781e-4** against CUD-001 **3e-4**;
packed measured **0.000244140625**. OPT-062 later reports integer vs staged
FP64 abs **0** after host Q8 reconstruction (not an expanded 3e-4). That
reduction-order miss is not the production gate. v2 cannot change eligibility
until production-K GPU authority exists. Route: **OPT-074 then OPT-075**.
No 30-sample historical oracle was used.

Current complete-cost evidence already on disk (not a new sitting): OPT-062
packed **15.7643099** ms vs integer_q8 **12.4760742** ms; OPT-063 paired_integer
**11.2725649** ms. Those are screened-in follow-ups, not production keeps.

## Independent legacy candidate not owned by OPT-075–079

`half-scale-q8-1`: control packed Q8Block (recomputes int32 `sum(q)`) versus
candidate integer `UseQ81` consuming Quartz `half(sum(q))`. Two shapes, two
implementations, 16×4 sampled references, optional 1+3 whole-family screen.
OPT-075 forbids a half-scale grid. The case is enumerated in the iteration
contract; the screen was **not** executed and is not a production keep.

## Owners and no-repeat

| Owner | Entries |
|---|---|
| OPT-070 | installed OPT-064 `r2_w2` and OPT-066 `fma_async_x` acceptance evidence |
| OPT-073 | OPT-058 functional prompts/scorers and quality verdicts |
| OPT-074 | missing production-K llama GPU authority |
| OPT-075 | OPT-042/046/062/063 Q4 integer unfused and paired |
| OPT-080 | OPT-016 / OPT-056 / OPT-069 outcome gates |
| none | performance-only no-repeat (OPT-024, 027–030, 037, 054, 055, 065, 067, 068) |

OPT-068 passed primitive numerics and lost on complete FFN cost. Relaxing
correctness cannot rescue that measurement. OPT-064/066 are installed keeps
with incomplete incremental acceptance; they are not numerical rejections.
Historical OPT-046 integer-pin all-NaN logits remain a hard error: numerical
headroom cannot admit nonfinite or layout failures.

## Proof limit

No throughput claim. No production selector change. Register audit has no GPU
work. Acceptance is review completeness: every old rejection is kept, still
valid, evidence-invalid, numerically eligible, already superseded, or blocked
with an owner.
