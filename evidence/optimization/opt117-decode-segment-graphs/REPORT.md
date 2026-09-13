# OPT-117 — Repair and evaluate complete decode-segment graphs

Status: **retain_ffn_only**. Production selector `ffn_only`. Hardware executed.

Parent is authenticated post113_selected `ffn_only`. Candidate is `decode_segments8`. Keep requires OPT-115 E2E/quality/state/memory gates and a demonstrated launch/idle gap reduction. Capture failure is a specific missing capability, not a measured graph performance loss. No weight/KV DRAM-traffic claim from launch reduction alone.

## Root cause

OPT-114's generic `cannot capture CUDA scheduler FFN graphs` hid two defects: post113 `llama_q4k_mmvq` FFN capture needs OPT-110 adapter hooks, and `enqueue_decode_layer_eager` aliased GDN/attention committed pointers onto workspace candidate slots. `stage_and_validate_chunk` returns `cudaErrorInvalidValue` when candidate equals committed, so `ffn_only` could capture after the OPT-110 link while `decode_segments8` could not. Graphs now capture against session committed vs workspace candidate. Mixer Q8 grouped descriptors are uploaded from persistent workspace host memory before capture so graphs do not record stack `cudaMemcpyAsync` nodes. Eager fused decode does not ping-pong residual pointers across layers, so capture no longer swaps `residual`/`next`. Token-to-token GDN committed/candidate swaps and by-value attention frontier force recapture; `cuFuncGetParamInfo` kernel-arg patching launches but is not byte-exact, so production replay recaptures whenever frontier, GDN bases, or KV-part/crossover topology change. Scatter/commit stays eager; polls remain at eight-layer boundaries.

Capture ok=`True`; enqueue=`ok`; end_capture=`ok`; create_message=``; segment graphs=`8`; create_ms=`8.33416653`.

## Same-math, quality, state

same_math=`True` exact=`True`; quality=`True` opt116=`opt116_generated_v1`; state/memory=`True`.

## Full-engine A/B (3 warmups + 10 AB/BA)

| Workload | parent tok/s | candidate tok/s | geo ratio | CI lower | p95 ratio | gate |
|---|---:|---:|---:|---:|---:|---|
| D128 | 57.41325531 | 56.22983514 | 0.9793874228112488 | 0.9785923169622197 | 1.054403157704339 | False |
| D2048 | 55.65610045 | 54.56690292 | 0.9804293937850275 | 0.9798236785212288 | 1.0522170637893005 | False |
| P4096 | 2892.532227 | 2894.827782 | 1.0007935403887815 | 1.0004620308156211 | 0.9992071461210472 | True |

## OPT-115 gap comparison

OPT-115 D128 idle-leaf estimate `9.974685400000169` ms. OPT-114 ~11 ms/token remains a historical hypothesis. D128 tok/s delta vs parent `-1.183420169999998`. Reasons: `['d128_throughput_gate', 'd2048_throughput_gate']`.

Working `decode_segments8` graphs are byte-exact versus eager after per-token recapture (257 launch-param updates on 256 decode tokens). Kernel-arg patching is not exact, so recapture cost is inside token wall time. That overhead is larger than any remaining launch/idle gap versus `ffn_only`, so the candidate is slower on D128/D2048 (geo CI lower bound < 1.00; decode p95 ratio > 1.05). P4096 prompt graphs are unchanged and stay within the 0.98 guard. Production retains `ffn_only`.

Keep=False. Shipping `ffn_only`.
