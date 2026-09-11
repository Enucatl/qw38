# OPT-076 — Remove Q4 inner-loop shuffles and scalar unpacking

Status: **measured**. Authority llama.cpp
`cc83d7b4824f73cfdda4dfbb47ee39804f71b328`, GGUF SHA-256 `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`. Control is OPT-075 packed/
paired_staged. Candidates are Q8Block packed-load / late-reduction kernels
at two and four warps/row with the selected paired_integer FFN composition.
FP32-scale Q8Block staging is unchanged. Half-scale Q8_1 is forbidden.
Q8/Q6 and prompt MMQ are untouched.

`claims_throughput: false`. Require >=0.10 ms/token complete-FFN saving and a
positive paired interval, or measured rejection.

## Numeric policy (OPT-074 GPU budgets)

Calibration then untouched held-out. Strict legacy 0.0003 is recorded
separately from the versioned production ceiling.
Legacy pass=True; production pass=True.
OPT-074 Q4 families admitted=False. An unfinished
reference freeze is **not** a numerical rejection.

## Mechanism

`vec_dot_q4k_q8block_late` uses 32-bit qs/q8 loads, nibble mask/shift, exact
integer dot and min-correction sums, and keeps each lane's FP32 contribution
through K. Scale boundaries stay per group/block. Row-end reduction is
unchanged. Unaligned pointers fall back to byte unpack. SASS/resource
evidence is in the native log.

## Dispatch

Launches: [{'config': 'packed_paired_staged', 'dispatch': {'gate_variant': 'q4k_gate_up_swiglu_prequant', 'up_variant': 'q4k_gate_up_swiglu_prequant', 'down_variant': 'quant_mmv_packed', 'gate_up_stage_count': 1, 'down_stage_count': 1, 'captured_in_graph': False, 'q4_path': 'packed', 'ffn_path': 'paired_staged', 'staging': 'q8_fp32', 'warps_per_row': 4, 'effective_q4_decode': 'packed', 'effective_ffn_decode': 'paired_staged', 'q8_decode': 'dp4a_q8_1', 'q6_decode': 'integer_q8_1', 'capture_key': '5fdf3368e3a1898547bf5ecbe61333b4eaecdc9e4b70d97cdffce2edf7d20920', 'gate_up_calls': 64, 'down_calls': 64, 'invalidate_q8_decode_staging': True, 'staging_ops_per_ffn': 2}, 'mean_ms': 16.4213763, 'warps_per_row': 4}, {'config': 'late_w4', 'dispatch': {'gate_variant': 'q4k_coop_gate_up_swiglu_late_prequant_q8', 'up_variant': 'q4k_coop_gate_up_swiglu_late_prequant_q8', 'down_variant': 'q4k_coop_mmv_late_q8', 'gate_up_stage_count': 1, 'down_stage_count': 1, 'captured_in_graph': False, 'q4_path': 'integer_q8_late', 'ffn_path': 'paired_integer', 'staging': 'q8_fp32', 'warps_per_row': 4, 'effective_q4_decode': 'integer_q8_late', 'effective_ffn_decode': 'paired_integer', 'q8_decode': 'dp4a_q8_1', 'q6_decode': 'integer_q8_1', 'capture_key': '5fdf3368e3a1898547bf5ecbe61333b4eaecdc9e4b70d97cdffce2edf7d20920', 'gate_up_calls': 64, 'down_calls': 64, 'invalidate_q8_decode_staging': True, 'staging_ops_per_ffn': 2}, 'mean_ms': 9.8184765, 'warps_per_row': 4}]

## Complete FFN screen

64-layer rotating complete FFN on repaired OPT-071 captures. Control packed vs
late survivor `late_w4`.
Component n=10 means: control 16.421 ms,
candidate 9.818 ms.
Paired CI (control-candidate): 6.5505 ..
6.6553 ms; mean diff 6.6029 ms.
Engine D2048+32 pairs: 876.790 vs
643.448 ms.

## Quality (OPT-073)

quality-v3 absolute=fail;
engine non-regression=pass;
quality-v2 all=False.

## Decision

Verdict: **retain_packed** (['opt074_coverage_unadmitted_missing_evidence']).
production_kept=False; retain_reason=missing_evidence.
Shipping Q4 stays `packed` /
`paired_staged`.
OPT-075 production pins are unaffected unless this sitting keeps a candidate.

## tok/s

Speedup versus the then-current packed baseline is **0** while packed is
retained.
