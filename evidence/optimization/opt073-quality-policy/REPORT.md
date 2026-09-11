# OPT-073 — Separate engine quality from the authority's arithmetic miss

Status: **quality policy frozen**. `claims_throughput: false`. No arithmetic
kernel change and no speedup are claimed. Authority remains llama.cpp
`cc83d7b4824f73cfdda4dfbb47ee39804f71b328` and GGUF SHA-256 `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`.

## Why this sitting exists

OPT-058 established valid no-thinking functional prompts. Both engines still
miss `task_arithmetic`. OPT-069 preflight counted functional tokens and then
release continued into timed oracles after quality-v2 failed. This increment
authenticates the prompts/scorer, freezes a versioned dual-verdict policy, and
fail-closes release on required quality.

## Arithmetic audit (17 + 25)

Human-verifiable truth written first: **17 + 25 = 42**. On the original v2
options, **B = 42 is correct** and **A = 41 is wrong**. C=43 and D=44 are also
wrong. The parser still accepts exactly one of A/B/C/D after stripping
whitespace and rejects empty output, extra text, a second answer, and
teacher-forced greedy tokens.

Authenticated original v2 rendering matches
`src/template.cpp` `render_user_turn(..., enable_thinking=false)` and ends at
`<|im_start|>assistant\n<think>\n\n</think>\n\n`. Tokenizer IDs are the frozen
v2 context. Max new tokens remain 16. Logits are not masked.

Retained two-engine original v2 result:

| Engine | Parsed | First token | Independently correct |
|---|---|---:|---|
| Quartz | A (wrong_choice) | 32 | B / 42 |
| llama.cpp | A (wrong_choice) | 32 | B / 42 |

Both engines emit A. That **does not** prove a Quartz kernel or scorer bug.
No objective rendering/parser defect was found, so the v2 prompt is not
rewritten. Four arithmetic diagnostics were frozen before any candidate
measurement: original v2, one fixed option permutation (42 moves to C), direct
numeric completion under the same template, and one independent paraphrase.
Every diagnostic is retained. Passing paraphrases were not selected.

Frozen llama first-token snapshot on original v2: token 32 (A) logit
26.5466595, runner-up 33 (B) logit
24.3310528, margin
2.2156067.

## Quality-v3 dual verdict

Rationale: Pinned llama and Quartz both emit A (41) on the authenticated no-thinking v2 17+25 item whose independently correct answer is B (42). Rendering, tokenizer IDs, assistant boundary, termination, and the A/B/C/D parser match src/template.cpp enable_thinking=false; this is an authority item miss, not a kernel or scorer defect. Quality-v3 keeps absolute arithmetic accuracy failed while allowing internal candidate admission when every reference-correct functional item stays correct and the known miss is either independently corrected or preserved as authority A within the frozen first-token identity and logit/margin snapshot. Other wrong choices are not waived. Near-ties cannot pass absolute accuracy. Quality-v2, QLT-001, OPT-056, and OPT-016 remain unchanged historical conditions. Missing evidence is incomplete, never pass.

| Suite | Quartz status | Absolute accuracy | Engine non-regression |
|---|---|---|---|
| quality-v2 | fail all=False | n/a (v2 requires B on 17+25) | n/a |
| quality-v3 | fail all=False | fail | pass |

Absolute arithmetic accuracy stays **failed**. Internal candidate admission may
use quality-v3 engine non-regression plus both 1024-target PPL ratios <= 1.01
and recurrence drift <= 0.02. This does **not** replace OPT-056 or OPT-016.
QLT-001 and the original eight-task OPT-056 functional fail remain historical.

Same-input near-ties cannot waive a known wrong semantic answer. Other wrong
choices (C, D, empty, extra text) on the missed item are not waived. Missing
NLL, authority, or generated answers yield **incomplete**, not pass.

## Release preflight

OPT-069 now parses all eight functional answers and records each required
selected quality verdict. Token count alone is not sufficient. A release that
would claim OPT-056 stops while quality-v2 `all` is false, before P/D/2K
oracles. `--diagnostic-performance` is an explicit non-release mode: timing may
run, the quality failure is retained, and keep/release claims are prohibited.

Current retained preflight: status `quality_blocked`;
OPT-056 quality requirement met =
False.
Default release oracles: run=False
eligible=False.

## Proof limit

- no throughput claim
- no arithmetic kernel change
- B=42 remains the correct 17+25 answer
- A=41 remains wrong
- both engines failing is not kernel blame
- no prompt rewrite without an objective defect
- no token mask
- no teacher-forced functional answer
- no selecting only paraphrases that pass
- quality-v3 does not replace OPT-056 or OPT-016
- missing evidence is incomplete
- failed required quality stops release before long timing
- diagnostic performance is not a release or keep
