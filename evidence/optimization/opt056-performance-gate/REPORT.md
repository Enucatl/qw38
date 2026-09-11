# OPT-056 — Close the measured speed gap with quality evidence

## Claim labels and proof limits

This increment is the end-to-end outcome gate for **same-sitting P/D128/D2048 versus pinned llama.cpp**.
Keep requires **at least 5% throughput margin**, **decode p95 no worse than llama**,
**confidence-supported improvement**, **combined production quality on selected paths**,
and **original OPT-016 2K parity evidence**.
**candidate-task completion alone is insufficient**.
This task **does not redefine the 2K llama.cpp parity gate**.
**QLT-001 remains its own owner**.
**Session TTFT does not replace the historical workload protocol**.

Gate status: **unpassed**. `gate.passed` is False.

## Protocol

| Field | Frozen value |
|---|---|
| GGUF | `models/Qwen3.8-27B-Q4_K_M.gguf` SHA-256 `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34` |
| GPU | NVIDIA GeForce RTX 5090, compute capability 12.0, exclusive |
| Quartz image | `qw38-cuda:13.0.2` |
| llama.cpp image | `qw38-llama-authority:cuda-13.0.2` |
| llama.cpp revision | `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` |
| P | exact 4096, attribution null, graphs created, 0 warm-ups, 3 cold replicates |
| D128/D2048 | prefix 128 or 2048 then 256 predetermined tokens, 3 warm + 30 measured |
| llama.cpp P | `llama-bench -p 4096 -n 0 --no-warmup -r 3 -ngl 99` |
| llama.cpp D | `qw38-llama-decode-oracle` via pinned `llama.h` and `llama_time_us` |
| Margin | Quartz mean tok/s >= 1.05 × same-sitting llama, 95% CI on (Quartz − 1.05×llama) excludes 0 |
| Nsight | not_used |

## Combined production paths

| Pin | Selected path |
|---|---|
| `rms_norm` | `parallel_fma` |
| `q4_decode` | `packed` |
| `q8_decode` | `dp4a_q8_1` |
| `q6_decode` | `integer_q8_1` |
| `ffn_decode` | `paired_staged` |
| `query_prepare` | `hoisted` |
| `attention_pipeline` | `f16_async` |
| `gdn_preproc` | `transpose` |
| `mmq_pipeline` | `fma_async` |
| `execution_graphs` | `ffn_only` |
| `production_numerics` | `strict` |
| `prompt_microbatch_rows` | `4096` |

Component rejection retained: OPT-046 integer Q4 decode reverted; production `packed`.
Production numerics path `strict`; optimized_admitted false.

## Measured sitting

- Device: NVIDIA GeForce RTX 5090 compute 12.0
- measurement_utc: 2026-09-11T01:50:43Z
- source_revision: a929d9b211d519d8a0897d83fa22611893fbc5d9 (dirty)
- hardware_executed: True
- keep_sitting_skipped: False

| Workload | Quartz tok/s | llama tok/s | ratio | 5% bar | remaining tok/s | confidence | pass |
|---|---:|---:|---:|---:|---:|---|---|
| P 4096 | 2808.49609 | 3263.516321 | 0.8605743952836107 | 3426.6921370500004 | 618.1929866666665 | False | False |
| D128 | 37.4816246 | 68.9318767 | 0.5437487460379377 | 72.378470535 | 34.896849044333344 | False | False |
| D2048 | 35.7208481 | 67.3394327 | 0.5304596182931143 | 70.706404335 | 34.98555454983333 | False | False |

Decode p95 (ms): D128 Quartz 26.9079285 vs llama 14.576 pass=False (Quartz worse by 12.3319285 ms); D2048 Quartz 28.182457 vs llama 14.726 pass=False (Quartz worse by 13.456456999999999 ms). Run-mean p95: D128 Quartz 26.713541 vs llama 14.5896225; D2048 Quartz 28.0163727 vs llama 14.8876168.

## Remaining milliseconds versus OPT-043

| Workload | OPT-043 remaining ms | OPT-056 remaining ms |
|---|---:|---:|
| P 4096 | 701.5488738177955 | 203.34385000457132 |
| D2048 per token | 24.578147254269947 | 13.144716081053918 |

Prefill attribution wall_ms: 1436.90466
Decode D2048 attribution wall_ms: 28.6976166

Prefill category ms:

| Category | ms |
|---|---:|
| embedding | 0.119456001 |
| mixer_mmq | 310.441254 |
| gdn_core | 200.220718 |
| attention_core | 257.051208 |
| ffn_mmq | 666.404114 |
| logits | 1.01510406 |
| commit_sync | 0.718432009 |
| graph | 1.51299512 |
| other_idle | 0 |

Decode D2048 category ms:

| Category | ms |
|---|---:|
| embedding | 0.0535680018 |
| mixer_mmv | 7.09635019 |
| gdn_core | 2.37075186 |
| attention_core | 2.22873592 |
| ffn_mmv | 15.4258585 |
| logits | 0.835039973 |
| state_commit | 0.0281600002 |
| graph | 0.265611976 |
| other_idle | 0.120922089 |

## Quality

Production-optimization suite all=False.
wikitext_nll pass=True ppl_ratio=1.0000765601619601.
continuation pass=True positions=32.
recurrence pass=True incremental_nll=-0.0038120669000774043.
tasks pass=False count=8.
held-out NLL pass=True quartz_mean_nll=1.7878782710057632; llama held-out NLL was not a frozen baseline and is not invented (`llama_mean_nll`=not_run_this_sitting).
QLT-001 is not claimed complete.

| Task | greedy | expected | match |
|---|---|---|---|
| task_arithmetic | [271] | [33] | False |
| task_python_len | [271] | [32] | False |
| task_inference | [271] | [34] | False |
| task_minutes | [271] | [35] | False |
| task_sort | [271] | [33] | False |
| task_json | [271] | [32] | False |
| task_reading | [271] | [34] | False |
| task_sequence | [271] | [35] | False |

OPT-016 2K: Quartz 3012.69507 vs llama 3169.571249 remaining 156.87617899999987 tok/s below llama; gate_passed=False. This sitting does not rewrite `fixtures/opt016_parity.json`.

## Secondary Session metrics

TTFT 119.808688 ms; ITL p50 25.03173 ms; ITL p95 25.146197 ms.
These do not replace the historical OPT-021/OPT-032 workload protocol.

## State / memory

Memory-fit ok=True; checkpoint ok=True; cancellation frontier 0.

## Exact remaining gap

The gate is unpassed. Honest deficits versus this sitting's 5% llama bar:

- P: Quartz needs 3426.6921370500004 tok/s and measured 2808.49609; remaining 618.1929866666665 tok/s (203.34385000457132 ms on 4096 tokens). 95% CI on (Quartz − 1.05×llama) is [-667.6370641530767, -568.7489091802562].
- D128: Quartz needs 72.378470535 tok/s and measured 37.4816246; remaining 34.896849044333344 tok/s. Decode p95 is worse by 12.3319285 ms.
- D2048: Quartz needs 70.706404335 tok/s and measured 35.7208481; remaining 34.98555454983333 tok/s (13.144716081053918 ms per token). Decode p95 is worse by 13.456456999999999 ms.
- Quality: wikitext/continuation/recurrence/held-out NLL passed; all eight production-optimization tasks failed greedy match (every case emitted token 271).
- OPT-016 2K: Quartz is 156.87617899999987 tok/s below llama (original quartz ≥ llama bar, not the 5% margin).

Do not treat candidate-task completion as a substitute. Do not invent a pass.
