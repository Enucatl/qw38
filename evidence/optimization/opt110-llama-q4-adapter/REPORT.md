# OPT-110 — Execute pinned-llama Q4_K MMVQ through a Quartz adapter

Status: **quality_blocked**. Authority llama.cpp
`cc83d7b4824f73cfdda4dfbb47ee39804f71b328`, GGUF SHA-256
`31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`.
Control is production `integer_q8_late` / Q8Block / `raw_gguf` / 4 warps.
Candidate `llama_q4k_mmvq` is the source-faithful GENERIC nwarps=4
Q4_K × Q8_1 MMVQ adapter plus native `block_q8_1` staging and fused SWIGLU.
OPT-093 factored association and OPT-102 aligned metadata are not used.
Encodings are not interchangeable.

`claims_throughput: false`. `claims_performance_improvement: false`.
`evidence_complete=true`. Production pin `integer_q8_late` unchanged;
candidate is diagnostic-only via engine hooks (`llama_q4k_mmvq` legal override;
`kSelectedQ4DecodePath[]` stays `integer_q8_late`).

## Claim labels and proof limits

Source-faithful pinned llama MMVQ only in the candidate TU; production pins
unchanged. Primitive screen: identical BF16 activations and raw Q4_K weights at
production M/N/K; staging and dot timed separately; **sum** is go/no-go.
Complete rotating 64-layer FFN (3+10) requires saving ≥ 0.50 ms/token with a
positive 95% paired interval before keep. D128/D2048 engine pairs and P4096 guard
run only after primitive and complete-FFN gates pass. Candidate NLL must be
measured with the full quality suite; OPT-073 production quality is not reused.
Component (primitive staging+dot) and engine (complete FFN, decode pairs) are
separate claims. Candidate pos/neg abs 0.00146484375 is half-scale association,
not a mapping bug.

## Sitting identity

- device: NVIDIA GeForce RTX 5090
- image: `qw38-cuda:13.0.2`
- llama_revision: `cc83d7b4824f73cfdda4dfbb47ee39804f71b328`
- gguf_sha256: `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`
- execution_graphs: `ffn_only` (unchanged; OPT-114)
- shipping Q4 decode pin: `integer_q8_late` (unchanged)
- shipping Q4 device layout: `raw_gguf` (unchanged)
- warps per row: **4** (unchanged)
- candidate attrs (live): MMVQ 63 regs / occ 8 / 0 spill; fused SWIGLU 48 regs
  / occ 10 / 0 spill; quant 16 regs / occ 6; SASS dp4a/idp=520

## Eligibility

OPT-106 sitting admitted. Primitive screen must win before engine integration.

## Correctness and parity

Native correctness tier passed: typed `Q8Block` vs `block_q8_1` producers
isolated (`reinterpret_forbidden=true`); FP64 sample oracle at production K/M
shapes; same-input gate/up fusion, tails, zero, nonfinites, graph/eager;
down/gate sampled rows inside tolerance. Parity phase `pass=true`;
`production_pin_unchanged=true`. `kernel_parity_pass=true`.

## Primitive screen (must win before engine integration)

3 warmups + 10 paired CUDA events on identical BF16 and layout-identical Q4_K at
5120×17408 (down) and fused 17408×5120 (gate/up). `primitive_win=true`.

| role | control ms | candidate ms | saving ms | 95% CI | positive |
|---|---:|---:|---:|---|---|
| down 5120×17408 | 0.03891 | 0.01928 | 0.01962 | [0.01761, 0.02164] | true |
| gate_up fused 17408×5120 | 0.06406 | 0.03333 | 0.03073 | [0.02575, 0.03571] | true |

combined_control_ms=0.10297
combined_candidate_ms=0.05261
combined_saving_ms=0.05036
staging_bytes: Q8Block 19584 / block_q8_1 5760 (down); gate/up 5760 each
reason=matched_primitive_staging_plus_dot_won

## Complete rotating 64-layer FFN (3+10)

`ffn_keep=true`. `positive=true`. Required saving ≥ 0.50 ms/token met.

| path | mean ms | saving ms | 95% CI | positive |
|---|---:|---:|---|---|
| integer_q8_late | 9.814 | — | — | — |
| llama_q4k_mmvq | 9.097 | **+0.717** | **[0.678, 0.756]** | **true** |

## D128 engine pairs (5 uninstrumented, +32 tokens)

control 581.7 ms → candidate 549.2 ms; **55.01 → 58.27 tok/s**; CI [29.60, 35.43];
`improved=true`.

## D2048 engine pairs (5 uninstrumented, +32 tokens)

control 600.2 ms → candidate 564.7 ms; **53.31 → 56.67 tok/s**; CI [33.19, 37.89];
`improved=true`.

## P4096 prefill guard

control 1357.64 ms vs candidate 1357.85 ms; ratio **0.99985** ≥ 0.95 (MMVQ is
decode-only); `improved=null`.

## Quality

`model_quality_pass=false`. `skipped=true`. `reason=candidate_nll_not_measured`.
Candidate NLL was not run; OPT-073 production quality is not reused as a
candidate pass. Quality baseline is Makefile-wired to `OPT110_LLAMA_OBJECT` but
`cuda-opt110-diagnostics` does not build `qw38-cuda-opt058-quality-baseline-test`;
a keep flip requires that binary plus `--q4-decode llama_q4k_mmvq` and a 7200s
quality budget.

## Independent verdicts

```json
{
  "integer_q8_late": {
    "kernel_parity_pass": true,
    "primitive_pass": true,
    "model_quality_pass": true,
    "performance_pass": true,
    "production_kept": true,
    "incomplete": false
  },
  "llama_q4k_mmvq": {
    "kernel_parity_pass": true,
    "primitive_pass": true,
    "model_quality_pass": false,
    "performance_pass": true,
    "production_kept": false,
    "incomplete": true
  }
}
```

## Adapter notes

Local mods vs pin: BF16→Q8_1 (llama quantize is float); no ggml PDL/fastdiv/ids;
fused SWIGLU stores BF16; compile-time 5120×17408 and 17408×5120; diagnostic
engine hooks only. Primitive Q4_K used 144B layout-identical synthetic blocks;
engine phases used production `raw_gguf`.

## Decision

Verdict: **quality_blocked** (`quality_unresolved`). Matched primitive **won**
and performance keep bar is **met** (complete FFN +0.717 ms/token with positive
CI; D128/D2048 throughput improved; P4096 guard passed), but candidate NLL was
**not measured**, so keep is blocked. Production `integer_q8_late` /
`raw_gguf` / 4 warps retained. Candidate adapter remains diagnostic-only.
Tok/s delta versus production: **0**.

Evidence also in
[`fixtures/opt110_llama_q4_adapter.json`](../../../fixtures/opt110_llama_q4_adapter.json);
rejection record in
[`REJECTION.md`](REJECTION.md).
