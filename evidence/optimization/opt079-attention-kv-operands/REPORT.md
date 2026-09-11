# OPT-079 — Convert prompt attention KV operands once per shared stage

Status: **measured**. Authority llama.cpp
`cc83d7b4824f73cfdda4dfbb47ee39804f71b328`, GGUF SHA-256 `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`. Control is production `f16_async`.
Candidate `kv_once` converts each live BF16 K/V stage to F16 once after
cp.async completion and reuses the same 16-bit shared slots. Tile dimensions,
two-stage async ring, register softmax, stream-K, query preparation and F16
arithmetic are unchanged. No global converted KV cache. No gqa6/nbatch64 retry.

`claims_throughput: false` unless production_kept. Require >=5 ms/prompt
estimated complete attention saving and a positive paired interval, or
measured rejection. Long release is OPT-080.

## Numeric policy

Candidate F16 operands and complete attention outputs must equal `f16_async`
for unchanged arithmetic. Finite BF16 values around half rounding, signed
zero, underflow and overflow are covered without extra clamping.

## Mechanism

After each stage's producer wait, the CTA cooperatively converts BF16 K/V
in place. QK MMA and P×V load packed F16 fragments. Stage ownership: no
conversion before producer completion, no consumer reads before conversion,
no next asynchronous write before the last consumer.

## Dispatch

Launches: [{'config': 'f16_async', 'dispatch': {'path': 'f16_async', 'launch': 'fattn_mma_pipeline_f16_async', 'convert_once': 0, 'occupancy': 1}, 'mean_ms': 249.6971192}, {'config': 'kv_once', 'dispatch': {'path': 'kv_once', 'launch': 'fattn_mma_pipeline_kv_once', 'convert_once': 1, 'occupancy': 1}, 'mean_ms': 238.1892289}]

## Complete P4096 attention (16 layers)

Control f16_async vs survivor `kv_once`.
Component n=10 means: control 249.697 ms,
candidate 238.189 ms.
Paired CI (control-candidate): 11.4763 ..
11.5395 ms; mean diff 11.5079 ms.
Engine P4096 pairs: 1362.361 vs
1361.761 ms.

## Quality (OPT-073)

quality-v3 absolute=fail;
engine non-regression=pass.
Nonfinite=0.

## Decision

Verdict: **keep** (['component_ci_positive', 'saving_ge_5_ms_prompt', 'e2e_non_regression', 'quality', 'numeric']).
production_kept=True; retain_reason=None.
Shipping attention pipeline is `kv_once`.

## tok/s

P4096 whole-run tok/s delta versus the then-current f16_async baseline is
**1.325183133973951**. Zero when production_kept is false.
