# OPT-048 — Q6_K integer-dot full vocabulary projection

## Claim labels and proof limits

Live exclusive-RTX-5090 A/B of cooperative DP4A Q6_K integer dots
(llama `vec_dot_q6_K_q8_1` / `mmvq.cu` technique, 210-byte unaligned
word loads, K distributed across warps per row) against retained packed
FP32 `quant_mmv`. Complete cost includes activation staging, final RMSNorm,
and the full 248320-row FP32 output. No vocabulary pruning, top-k logits,
or approximate argmax. Keep requires **OPT-044 production-numerics
budgets**, **full-vocab logit/NLL/near-tie quality**, **lower complete
component time**, **95% P/D128 floors versus the OPT-047 keep**, and a
**strict D2048 improvement** versus OPT-047 **32.390007 tok/s**. Decode
p95 stays inside 105% of OPT-045. This is a bounded logits-category win,
not a route to the entire 24.6 ms decode recovery. Does not substitute
for the 2K llama.cpp parity gate. Nsight is not used.

Proof limit: OPT-044 production-numerics budgets; packed FP32 path retained; cooperative K warps per row; DP4A packed integer Q6_K dots; 210-byte unaligned word loads; full 248320 vocabulary, no pruning; complete cost includes stage, final norm, full FP32 output; 95% throughput floors versus OPT-047 keep; 105% p95 ceilings versus OPT-045; does not substitute for the 2K llama.cpp parity gate

## Decision

**keep**. Production pin `integer_q8_1` with
2 warps per row. A/B winner
`integer_q8_1_w2` (0.642994106 ms vs packed 1.56411517 ms). Keep sitting
ran. Quartz P 2129.85938, D128 35.9651642, D2048 34.3371162
tok/s versus OPT-047 keep P 2128.54175, D128 33.8896103, D2048
32.390007. OPT-046 packed Q4 and OPT-047 DP4A Q8 remain.

## Quality

Independent `decode_q6_k` fixtures check packed signedness (offset -32)
and the 16-value subscale boundary. Probe staged envelopes follow OPT-044;
pathological `q6_k_257x512` stays informational. Production quality
compares every logit at captured d128/d2048/real-text positions plus NLL
and near-tie argmax.

## Throughput

Baseline is the OPT-047 keep sitting. On reject, speedup is 0 and
production stays packed.
