# OPT-075 — Admit or reject the existing cooperative Q4 FFN

Status: **measured**. Authority llama.cpp
`cc83d7b4824f73cfdda4dfbb47ee39804f71b328`, GGUF SHA-256 `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`. Exactly three configurations:
packed/paired_staged control, integer_q8 separate gate/up, integer_q8
paired_integer. Four warps/row and FP32-scale Q8Block. No new fusion, tile,
compiler, or half-scale grid. OPT-042/046 are reviewed through the current
OPT-062/063 equivalents.

`claims_throughput: false`. Historical **1.26x** / **1.40x** screens are
hypotheses, not expected or newly delivered tok/s.

## Numeric policy (OPT-074 GPU budgets)

Calibration then untouched held-out. Strict legacy 0.0003 is recorded
separately from the versioned production ceiling.
Legacy pass=True; production pass=True.
Near-miss explanation: legacy and production numeric rules agree.
OPT-074 Q4 families admitted=False; reason rows:
[{'id': 'q4_down_k17408', 'v2_admitted': False, 'coverage': 'unadmitted', 'reason': 'held_out_exceeds_frozen_calibration_ceiling'}, {'id': 'q4_gate_up_k5120', 'v2_admitted': False, 'coverage': 'unadmitted', 'reason': 'held_out_exceeds_frozen_calibration_ceiling'}]. An unfinished reference freeze is **not** a numerical
rejection of an integer candidate.

## Dispatch

Every gate/up/down projection must use the intended launch variant in eager
and captured execution, one pair per layer, two staging operations per FFN.
Trace fallback, Q8/Q6 selectors, and staging invalidation stay in place.
Graphs are recaptured after selecting the configuration so the old packed
graph cannot remain in the timed candidate.

Launches: [{'config': 'packed_paired_staged', 'dispatch': {'gate_variant': 'q4k_gate_up_swiglu_prequant', 'up_variant': 'q4k_gate_up_swiglu_prequant', 'down_variant': 'quant_mmv_packed', 'gate_up_stage_count': 1, 'down_stage_count': 1, 'captured_in_graph': False, 'q4_path': 'packed', 'ffn_path': 'paired_staged', 'staging': 'q8_fp32', 'warps_per_row': 4, 'effective_q4_decode': 'packed', 'effective_ffn_decode': 'paired_staged', 'q8_decode': 'dp4a_q8_1', 'q6_decode': 'integer_q8_1', 'capture_key': '5fdf3368e3a1898547bf5ecbe61333b4eaecdc9e4b70d97cdffce2edf7d20920', 'gate_up_calls': 64, 'down_calls': 64, 'invalidate_q8_decode_staging': True, 'staging_ops_per_ffn': 2}, 'mean_ms': 16.4332861}, {'config': 'integer_q8_paired', 'dispatch': {'gate_variant': 'q4k_coop_gate_up_swiglu_prequant_q8', 'up_variant': 'q4k_coop_gate_up_swiglu_prequant_q8', 'down_variant': 'q4k_coop_mmv_q8', 'gate_up_stage_count': 1, 'down_stage_count': 1, 'captured_in_graph': False, 'q4_path': 'integer_q8', 'ffn_path': 'paired_integer', 'staging': 'q8_fp32', 'warps_per_row': 4, 'effective_q4_decode': 'integer_q8', 'effective_ffn_decode': 'paired_integer', 'q8_decode': 'dp4a_q8_1', 'q6_decode': 'integer_q8_1', 'capture_key': '5fdf3368e3a1898547bf5ecbe61333b4eaecdc9e4b70d97cdffce2edf7d20920', 'gate_up_calls': 64, 'down_calls': 64, 'invalidate_q8_decode_staging': True, 'staging_ops_per_ffn': 2}, 'mean_ms': 11.7726463}]

## Complete FFN screen

64-layer rotating complete FFN (norm, one shared gate/up stage, SwiGLU,
separate down, residual) on repaired OPT-071 captures. Control packed vs
integer survivor `integer_q8_paired`.
Component n=10 means: control 16.433 ms,
candidate 11.773 ms.
Paired CI (control-candidate): 4.6181 ..
4.7032 ms; mean diff 4.6606 ms.
Engine D2048+32 pairs: 875.710 vs
717.444 ms.

## Quality (OPT-073)

Internal admission uses quality-v3 engine non-regression plus NLL.
Absolute task accuracy stays visible (legacy/OPT-056).
quality-v3 absolute=fail;
engine non-regression=pass;
quality-v2 all=False.
A functional baseline defect cannot excuse new output/state errors.
Quality is identity-cached once per survivor; no long P/D protocol.

## Decision

Verdict: **retain_packed** (['opt074_coverage_unadmitted_missing_evidence']).
production_kept=False; retain_reason=missing_evidence.
Shipping Q4 stays `packed` /
`paired_staged`.
shipping_unchanged=True.

## tok/s

Speedup versus the then-current packed baseline is **0** while packed is
retained. Historical 1.26x/1.40x remain hypotheses.
