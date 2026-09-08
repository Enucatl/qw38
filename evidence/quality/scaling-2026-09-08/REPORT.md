# QLT-001 scaling gate — 2K / 8K / 32K (2026-09-08)

Status: **forecast / go-no-go evidence only**. Not CMP admission. Not 128K quality.

## Method

| Engine | Harness | Sizes | Samples |
|---|---|---|---|
| Quartz (`build/cuda/qw38-bench` @ `35295d5`) | cold prefill, smoke, 0 warmups / 1 sample, chat-rendered user prompt | ~2K / ~8K / ~32K | 1 each |
| llama.cpp `cc83d7b` (`llama-bench`) | `-n 0` prefill-only, `--no-warmup`, `-r 3`, `-ngl 99`, same GGUF | exact 2048 / 8192 / 32768 | 3 each |

Hardware: NVIDIA GeForce RTX 5090. Containers: `qw38-cuda:13.0.2` and `qw38-llama-authority:cuda-13.0.2`.

## Runtime results

| Context | Quartz tokens | Quartz time | Quartz tok/s | llama.cpp time | llama.cpp tok/s | llama / Quartz |
|---|---:|---:|---:|---:|---:|---:|
| 2K | 2052 | 41.7 s | 49.25 | 0.658 s | 3114 | 63× |
| 8K | 8196 | 204.1 s | 40.16 | 2.706 s | 3027 | 75× |
| 32K | 32772 | 21.8 min | 25.09 | 12.477 s | 2626 | 105× |

Quartz peak device memory stayed **28750 MiB** across these prefills.

Throughput falls with length (49 → 40 → 25 tok/s), so cost is **super-linear**.

## Comparison takeaway

Pinned plain llama.cpp on the **same GGUF / same GPU** is about **63–105×** faster at these prefills. That gap is for runtime planning, not quality admission.

Vs the pre-OPT stopped run (~**4.2 tok/s**): Quartz is roughly **6–12×** faster here, but still far from llama.cpp.

## 128K completion forecast

Fit on the three Quartz points: `seconds ≈ a·N + b·N²` with `a=1.982961e-02`, `b=6.111014e-07` (fit residuals [-1.599, 0.5, -0.025] s).

| Estimator | 131,072-token prefill |
|---|---|
| Quadratic fit (preferred) | **3.64 h** (218 min, effective ~10.0 tok/s) |
| Hold 32K rate (25.09 tok/s) | 1.45 h |
| Hold 8K rate (40.16 tok/s) | 0.91 h |
| Hold 2K rate (49.25 tok/s) | 0.74 h |
| Old 4.2 tok/s observation | 8.7 h |

**Planning recommendation:** budget about **3.6–4.7 hours** of exclusive RTX 5090 time for one Quartz 128K retrieval prefill (quadratic estimate + ~30% margin). llama.cpp would be on the order of **0.8 minutes** at the measured 32K rate — not a Quartz expectation.

## Artifacts

- `evidence/quality/scaling-2026-09-08/quartz-prefill-{2k,8k,32k}.json`
- `evidence/quality/scaling-2026-09-08/llama-bench-prefill-2k-8k-32k.json`
- `evidence/quality/scaling-2026-09-08/report.json`
- prompts: `prompts/scaling/prefill-{2k,8k,32k}.txt`

## Gate

QLT-001 is `pending` after OPT-012/OPT-013. **128K quality retrieval should start only after this report is explicitly accepted.**
