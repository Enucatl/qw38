# EVAL-01 core coverage amendment — 54 cases

**Decision date:** 2026-09-25. **Status:** selected for routine evaluation.
This amendment supersedes the 216-case *routine coverage* requirements in
[EVAL-01 / PERF-01](evaluation-policy-v0.md) and the implementation task
contracts. The original policy remains the historical definition of the full
suite. Its scoring methods, thresholds, reference identities, grading rules,
and PERF-01 measurements remain in force for the cases actually evaluated.

## Routine core

Use the frozen [core selection](../implementation/eval-core-54.json) from the
existing `qw38-language-v2` 216-case corpus. It selects 15 P100 prompts and
15 C92 questions by uniform sampling without replacement within each family,
using Python `random.Random(38)`, then retains all 12 L12 anchors and all 12
R-512/R-4096 retrieval cases. The 54 IDs in that file are the authoritative
selection; do not resample between candidates, arms, tasks, or runs. Preserve
the existing prompts, reference continuations, answer keys, token masks, and
grader. Report the selection-file digest and `core-54` scope alongside the
suite, policy, fixture, model, and runtime identities.

Score exactly these 54 IDs in both comparison arms. A completed, authenticated
216-case llama.cpp comparator run may supply its matching 54 rows; this does
not launch another full run or turn the routine result into full coverage.
Apply EVAL-01's
NLL, capability, retrieval, generated-text review, uncertainty, and replay
rules to the selected cases; report denominators and selected slice membership.
Review all 15 selected P100 outputs. A core result supports a claim about this
fixed diagnostic sample, not about untested P100/C92 cases. Missing selected
cases, incomplete references, unresolved reviews, or failed gates still block
routine acceptance. `R-32768-s0-d0.1` is the sole required and permitted
32768 quality case for TASK-026 and later applicable gates; the other five
frozen 32768 fixtures remain inventory only.

## Full suite execution

The 216-case P100/C92/L12/R-512/R-4096 evaluation is **manual-only**. It may
run only when a human explicitly starts that full command interactively. An
agent, task runner, CI job, scheduled job, or automatic retry must never launch
or require it, including for TASK-022, TASK-026, promoted changes in
TASK-028/029 development, or final TASK-030 validation. A full run is optional additional
evidence; completion or promotion of those tasks does not depend on it. Label
any such result `full-216`, keep it separate from `core-54`, and never merge
partial coverage across scopes into an accepted report.

The full evaluation already in flight when this amendment was requested is
grandfathered. Let it finish without interruption and preserve its result under
its original captured policy and scope; do not relabel it as core-54 evidence.

## DELIVERY-01 timing amendment — 2026-09-26

Completed TASK-022/026 evidence is unchanged. For the remaining delivery
sequence, TASK-028/029 use their specified development checks; TASK-030 owns
all combined core-54 scoring, output-bound selected-P100 review, the fixed
32768 case, and applicable replay/session gates. Development completion is
not quality acceptance. This supersedes any inherited demand for the whole
routine suite after every intermediate numerical change. Thresholds, graders,
provenance, selected cases and quality uncertainty rules remain binding;
case resampling of saved outputs is distinct from prohibited repeated
performance runs. Missing, failed or inconclusive quality evidence and
unresolved reviews still prevent promotion. Reuse authenticated matching
comparator evidence with explicit identities; never relabel historical
candidate outputs as the new candidate's validation.
