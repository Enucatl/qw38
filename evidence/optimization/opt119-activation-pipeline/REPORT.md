# OPT-119 — Reduce activation traffic across pipeline boundaries

Status: **keep**. Mixer Q8 pin `True`. FFN Q8 pin `True`. Hardware executed on RTX 5090.

Parent is authenticated post113_selected `ffn_only` after OPT-117 rejected graphs and OPT-118 kept lazy/overlap. Primary target is **prefill (P4096)**; D128/D2048 are non-target decode guards. Two slices: mixer RMSNorm→MMQ Q8_0 and FFN residual+norm→MMQ Q4K Q8. BF16 rounding is retained in registers. Q8_1 and Q8Block are not reused as MMQ layouts. Keep requires OPT-115 complete-request gates.

## Inventory

Standalone `quantize_mmq_q8_1` after mixer/FFN RMSNorm rereads the BF16 activation. Fusion writes MMQ Q8 in the producer epilogue and drops that reread. Last-row BF16 is retained for capture.

| Slice | encoding | parent quantize launches | candidate fused launches | BF16 reread B removed |
|---|---|---:|---:|---:|
| mixer RMSNorm→Q8 | `mmq_q8_0_float_scale` | 64 | 64 | 83886080 |
| FFN residual+norm→Q8 | `mmq_q4k_half2_scale_sum` | 64 | 64 | 83886080 |

Net activation bytes removed `167772160`. Weight sweeps are unchanged.

## Same-math, tails, quality, state

greedy-parity exact=`True` tokens `539`/`539`.
P2048 / 5120 tail / 8192 long-prompt ok=`True`.
OPT-058 `--quality` invoked; restored packed/r2=`false`; same_path_exact=`True`; candidate_nll_measured=`True`; held-out PPL ratio `1.0`; wikitext PPL ratio `1.0`; opt116=`opt116_generated_v1`.
state/memory=`True` 128k_fit=`True` session_bytes=`696743940`.

## Full-engine A/B (3 warmups + 10 AB/BA)

| Workload | parent tok/s | candidate tok/s | geo ratio | CI lower | p95 ratio | gate |
|---|---:|---:|---:|---:|---:|---|
| P4096 (target) | 2958.822682 | 2976.481811 | 1.0059699566581302 | 1.0054163586171858 | 0.0 | True |
| D128 (guard) | 54.937317289999996 | 54.93511772000001 | 0.9999600345712917 | 0.9996975646184688 | 1.000146785683193 | True |
| D2048 (guard) | 47.531241210000005 | 47.54523391 | 1.0002943789938068 | 1.0001963168982044 | 0.9999150696481286 | True |

P4096 tok/s delta vs parent `17.65912900000012`. D128 sitting delta `-0.002199569999987716`. Keep=True. Reasons: none.

Production dispatch ships mixer RMSNorm→MMQ Q8_0 and FFN residual+norm→MMQ Q4K Q8 fusion. Diagnostic kernel times cannot admit a keep.
