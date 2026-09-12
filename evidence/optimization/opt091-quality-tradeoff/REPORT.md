# OPT-091 — Successor quality gate (`opt091_late_w4_v1`)

Status: **policy_evaluated**.
Successor contract `opt091_late_w4_v1` with strict anchor `opt089_strict`.
`concession_used=False`.
`claims_throughput=False`.

## Policy verdicts

| Config | kernel_parity_pass | model_quality_pass | performance_pass | production_kept |
|---|---|---|---|---|
| packed_paired_staged | True | True | False | False |
| late_w4 | True | True | False | False |

absolute_quality_status=visible_and_separate.
strict_model_quality_pass=True.
successor_model_quality_pass=True.
regression_release_quality_pass=True.
timed_phases_status=not_applicable.
shipping Q4 `integer_q8_late` / FFN `paired_integer`.

PPL ratios: None.
Recurrence incremental NLL: None.
Concession reason: None.

Accepted risk when concession is exercised: at most 1.5% PPL drift on frozen spans
does not bound every task or long context. Roll back to the exact OPT-089 selection
if any later combined Q/PPL/recurrence/state/task test fails.
