# OPT-091 — Successor quality gate (`opt091_late_w4_v1`)

Status: **strict_quality_already_passed**.
Successor contract `opt091_late_w4_v1` with strict anchor `opt089_strict`.
`concession_used=False`.
`claims_throughput=False`.
OPT-056 task_arithmetic and OPT-016 remain the original owners of absolute
accuracy and 2K parity; those failures stay visible and are not relabeled.

## Policy verdicts

| Config | kernel_parity_pass | model_quality_pass | performance_pass | production_kept |
|---|---|---|---|---|
| packed_paired_staged | True | True | False | False |
| late_w4 | True | True | False | False |

absolute_quality_status=fail.
strict_model_quality_pass=True.
successor_model_quality_pass=True.
regression_release_quality_pass=True.
timed_phases_status=not_applicable.
shipping Q4 `integer_q8_late` / FFN `paired_integer`.

PPL ratios: {'held_out_vs_opt088': 0.9993525856672696, 'held_out_vs_opt084': 0.9993525856672696, 'wikitext_vs_opt088': 1.0012655655222131, 'wikitext_vs_opt084': 1.0012655655222131, 'aggregate_vs_opt088': 1.000308618299485, 'aggregate_vs_opt084': 1.000308618299485}.
Recurrence incremental NLL: 0.0.
Concession reason: strict_quality_already_passed.

Accepted risk when concession is exercised: at most 1.5% PPL drift on frozen spans
does not bound every task or long context. Roll back to the exact OPT-089 selection
if any later combined Q/PPL/recurrence/state/task test fails.
