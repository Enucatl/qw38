# OPT-118 — Remove avoidable transfers and state materialization

Status: **keep**. Production lazy pin `True`. Overlap pin `True`. Hardware executed on RTX 5090.

Parent is authenticated post113_selected `ffn_only` after the OPT-117 graph path was rejected. Candidate is lazy host logits/hidden plus device greedy argmax, with decode D2H/D2D overlapped against KV scatter. Keep requires OPT-115 complete-request gates. Lower transfer counts without a complete-request win are diagnostic, not a throughput keep.

## Inventory

Production decode copied 248320 FP32 logits and 5120 hidden values to host before scatter. GDN decode already pointer-swaps; GDN carry D2D (158859264 B) is unused at the shipping 4096 chunk/microbatch pin. Two changes were selected from that inventory: (A) lazy host logits/hidden with device-side greedy, (B) overlap remaining copies with scatter and skip the duplicate host copy on the token-only path.

| Path | D2H bytes | D2D bytes | notes |
|---|---:|---:|---|
| eager decode | 1013760 | 0 | blocking logits+hidden before scatter |
| lazy decode | 4 | 1013760 | device stash + 4 B greedy index |
| prefill (eager) | 1013760 | 0 | fused async D2H retained |

## Same-math, quality, state

greedy-parity exact=`True` greedy_equal=`True` tokens `539`/`539`.
OPT-058 `--quality` invoked; restored packed/r2=`false`; same_path_exact=`True`; candidate_nll_measured=`True`; held-out PPL ratio `1.0`; wikitext PPL ratio `1.0`; opt116=`opt116_generated_v1`.
state/memory=`True` 128k_fit=`True` session_bytes=`428308484` extra device outputs `1013764`.

## Full-engine A/B (3 warmups + 10 AB/BA)

| Workload | parent tok/s | candidate tok/s | geo ratio | CI lower | p95 ratio | gate |
|---|---:|---:|---:|---:|---:|---|
| D128 | 54.772316360000005 | 54.94479867 | 1.003149121670253 | 1.0029343132737405 | 0.9969261981681715 | True |
| D2048 | 47.43241005 | 47.54726831 | 1.0024215125066724 | 1.002296068141967 | 0.9969541258319838 | True |
| P4096 | 2894.180296 | 2895.6553240000003 | 1.0005096682758317 | 1.000207205674646 | 0.0 | True |
| logits-every-token | 49.112730420000005 | 48.9796196 | 0.9972897182806385 | 0.9971133543481421 | 1.0031935550234137 | True |

D128/D2048 are complete-request targets (256 decode tokens). P4096 is a non-target prefill guard. logits-every-token keeps full D2H and is diagnostic for the explicit public-logits caller.

D128 tok/s delta vs parent `0.17248230999999237`. Keep=True. Reasons: none.

Sampled sampling retains the host fallback. Greedy uses device argmax with the host first-max / smaller-index tie policy; NaN never wins. `Session::logits`, `sample`, eval, cancel, and checkpoint restore materialize the committed frontier on demand.
