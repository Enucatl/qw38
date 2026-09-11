# OPT-087 — Historical numerical-policy leftover reevaluation

Status: **no_additional_reopen**. Authority llama.cpp `cc83d7b4824f73cfdda4dfbb47ee39804f71b328`, GGUF
SHA-256 `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`. Inventory of remaining OPT-072 leftovers plus
OPT-075–079 outcomes. At most one extra already-implemented candidate may be
reopened; none qualified. Sequence remains kernel parity → model quality →
performance. No large sweep. No production pin change.

`opt074_coverage_unadmitted` is **not** a blocker.
`opt074_family_admission_required=False`.
`claims_throughput=False`.

## Disposition table

| Candidate | Disposition | Reason | Evidence |
|---|---|---|---|
| OPT-075:q4_integer | owned_elsewhere | owned by OPT-085; not reopened here | `fixtures/opt085_q4_reevaluation.json` |
| OPT-076:q4_late | owned_elsewhere | owned by OPT-085; not reopened here | `fixtures/opt085_q4_reevaluation.json` |
| OPT-064:q8_layout | owned_elsewhere | owned by OPT-086; not reopened here | `fixtures/opt086_q8_mmq_reevaluation.json` |
| OPT-066:mmq_x_pipeline | owned_elsewhere | owned by OPT-086; not reopened here | `fixtures/opt086_q8_mmq_reevaluation.json` |
| OPT-077:tiled_decode_gdn | do_not_reopen | component_interval_not_positive; performance, not numerical-policy | `fixtures/opt077_gdn_decode.json` |
| OPT-078:prepared_q_veckv | do_not_reopen | performance_rejected / negative complete saving | `fixtures/opt078_decode_attention.json` |
| OPT-079:kv_once | do_not_reopen | keep; OPT-088 interaction check only | `fixtures/opt079_attention_kv_operands.json` |
| OPT-067:prompt_pair | do_not_reopen | OPT-072 no_repeat performance loser | `fixtures/opt072_rejection_review.json` |
| OPT-065:mmq_tiles | do_not_reopen | OPT-072 no_repeat performance loser | `fixtures/opt072_rejection_review.json` |
| OPT-068:scoped_codegen | do_not_reopen | OPT-072 no_repeat performance loser | `fixtures/opt072_rejection_review.json` |
| OPT-024:mixer_q8_d2r | do_not_reopen | OPT-072 no_repeat performance loser | `fixtures/opt072_rejection_review.json` |
| OPT-027:persistent_fattn | do_not_reopen | OPT-072 no_repeat performance loser | `fixtures/opt072_rejection_review.json` |
| OPT-028:mmq_streamk | do_not_reopen | OPT-072 no_repeat performance loser | `fixtures/opt072_rejection_review.json` |
| OPT-029:gdn_fuse | do_not_reopen | OPT-072 no_repeat performance loser | `fixtures/opt072_rejection_review.json` |
| OPT-030:pdl_launches | do_not_reopen | OPT-072 no_repeat performance loser | `fixtures/opt072_rejection_review.json` |
| OPT-037:ffn_tiles | do_not_reopen | OPT-072 no_repeat performance loser | `fixtures/opt072_rejection_review.json` |
| OPT-054:prefill_microbatch | do_not_reopen | OPT-072 no_repeat performance loser | `fixtures/opt072_rejection_review.json` |
| OPT-055:execution_graphs | do_not_reopen | OPT-072 no_repeat performance loser | `fixtures/opt072_rejection_review.json` |
| half-scale-q8-1 | do_not_reopen | never screened; OPT-075 forbade a half-scale grid; no measured upside | `fixtures/opt072_rejection_review.json` |
| OPT-042:cud001_one_warp | do_not_reopen | do not rebuild; superseded by OPT-085 kernel parity | `fixtures/opt072_rejection_review.json` |

`no_additional_reopen=True`.
`reopened_candidate=None`.
Qualifying OPT-072 leftovers not already tabled: none.

## Confirmed non-reopens

- OPT-075/076 Q4 integer and late-reduction remain owned by OPT-085.
- OPT-064/066 Q8/MMQ remain owned by OPT-086.
- OPT-077 tiled decode GDN stays sequential (`component_interval_not_positive`).
- OPT-078 prepared_q / veckv stays `performance_rejected` with negative
  complete saving. Do not rerun hoping the interval flips.
- OPT-079 `kv_once` remains the keep; OPT-088 checks interactions.
- OPT-067/065/068/024/027–030/037/054/055 stay OPT-072 `no_repeat`
  performance losers. Missing OPT-074 coverage does not make them eligible.
- `half-scale-q8-1` was never screened; OPT-075/085 forbid a half-scale grid.
- Historical CUD-001 3e-4 one-warp Q4 diagnostic is not rebuilt.

Historical OPT-072 report
`evidence/optimization/opt072-rejection-review/REPORT.md` is unmodified.

## tok/s

Official tok/s delta versus the then-current shipping baseline is **0**.
`claims_throughput` is true only when a reopened candidate is kept.
