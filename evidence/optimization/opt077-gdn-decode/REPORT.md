# OPT-077 — Parallelize the one-token GDN state reduction

Status: **measured**. Authority llama.cpp
`cc83d7b4824f73cfdda4dfbb47ee39804f71b328`, GGUF SHA-256 `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`. Control is sequential
`prepare_recurrence_window` on `launch_gdn_prepare_tiled` /
`kSequentialWindows`. Candidates are value-tile 16 and 32, four warps/CTA,
grid `value_heads x ceil(value_width/value_tile)`. Convolution stays separate.
FP32 row-major state is unchanged. llama.cpp `gated_delta_net_cuda` motivates
parallelism; its transposed state is not used.

`claims_throughput: false` unless production_kept. Require >=0.10 ms/token
complete 48-layer GDN saving and a positive paired interval, or measured
rejection. Recurrence incremental NLL <=0.02 is not relaxed. Long release is
OPT-080.

## Numeric policy

Independent FP64 sampled state/output plus same-input sequential GPU
reference. Llama GPU recurrence is not drop-in on row-major S; conversion
was rejected. Strict GDN-002 5e-8 / 5e-9 remains the sequential reference.
Nonfinite=0.

## Mechanism

Q/K norms are precomputed once per key head in shared memory. Each CTA owns
a value tile and distributes the key reduction across four warps with a
shared-memory tree; each candidate cell is written once. Prompt GDN is
unchanged.

## Dispatch

Launches: [{'config': 'sequential', 'dispatch': {'path': 'sequential', 'launch': 'prepare_recurrence_window', 'value_tile': 0, 'grid_x': 48, 'grid_y': 1, 'warps_per_cta': 4}, 'mean_ms': 12.5164898, 'value_tile': 0}, {'config': 'tile32', 'dispatch': {'path': 'tile32', 'launch': 'prepare_recurrence_decode_tiled32', 'value_tile': 32, 'grid_x': 48, 'grid_y': 4, 'warps_per_cta': 4}, 'mean_ms': 11.893901, 'value_tile': 32}]

## Complete 48-layer GDN screen

Two capture positions on repaired OPT-071 decode captures. Control sequential
vs tiled survivor `tile32`.
Component n=10 means: control 12.516 ms,
candidate 11.894 ms.
Paired CI (control-candidate): -0.1643 ..
1.4094 ms; mean diff 0.6226 ms.
Engine D2048+32 pairs: 878.416 vs
866.012 ms.

## Quality (OPT-073) and state

quality-v3 absolute=fail;
engine non-regression=pass.
Committed-state isolation is required before install.

## Decision

Verdict: **retain_sequential** (['component_interval_not_positive']).
production_kept=False; retain_reason=uncertainty.
Shipping GDN decode stays `sequential`.

## tok/s

Speedup versus the then-current sequential baseline is **0.0** while sequential
is retained unless production_kept.
