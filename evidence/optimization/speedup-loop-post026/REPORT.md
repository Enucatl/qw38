# Speedup-loop scout — post OPT-026

## Claim labels and proof limits

Live exclusive-RTX-5090 CUDA-event attribution against rebuilt production
objects after OPT-026 delivery. **Not** a keep/reject gate and **not** a
substitute for OPT-016. Nsight unused. Scout binaries are not retained.

## Oracle gap

| Source | Mean tok/s |
|---|---:|
| Quartz successor (`fixtures/opt026_fattn_streamk.json`) | 1746.71973 |
| llama.cpp same-sitting | 3253.993621 |
| Ratio | ~1.86× (Quartz slower) |

OPT-022–OPT-026 idea ladder is exhausted; Quartz 4K still does not meet llama.cpp.

## Fresh attribution (2026-09-09T06:10–06:11Z)

### Cold exact-2048 (graphs=0)

| Category | ms | Share |
|---|---:|---:|
| `ffn_mmq` | 390.44 | 39.2% |
| `attention_core` | 245.94 | 24.7% |
| `gdn_core` | 211.43 | 21.2% |
| `mixer_mmq` | 145.66 | 14.6% |
| wall | 997.08 | tok/s 2054.0 |

### Cold exact-4096 (graphs=64; oracle length)

| Category | ms | Share |
|---|---:|---:|
| `attention_core` | 867.45 | **36.9%** |
| `ffn_mmq` | 738.27 | 31.4% |
| `gdn_core` | 453.23 | 19.3% |
| `mixer_mmq` | 285.69 | 12.2% |
| `graph` | 0.23 | ~0% |
| wall | 2348.33 | tok/s 1744.2 |

At the 4K oracle length, **attention_core is the largest Quartz-owned sink**.
Component A/Bs agree: fattn `stream_k` ≈53.6 ms/attn-layer ×16 ≈858 ms;
FFN `shared_y_swiglu_q8` ≈10.57 ms/layer ×64 ≈676 ms.

## Next pick

Prefer transferable llama.cpp/ds4 ideas that close `attention_core` first,
then `ffn_mmq`, then `gdn_core`, then `mixer_mmq`. Do not re-admit kept
OPT-022/023/025/026 work or rejected OPT-024 mixer Q8 D2R.

Admitted second ladder (keep/reject vs oracle **1746.71973**):

1. **OPT-027** — persistent Ada+ fattn stream-K (`attention_core`)
2. **OPT-028** — Q4_K/Q6_K MMQ stream-K (`ffn_mmq`, also large mixer/attn GEMMs)
3. **OPT-029** — fuse GDN conv + gated output into fused loop (`gdn_core`)
4. **OPT-030** — Hopper/Blackwell PDL on prompt compute stream (cross-cutting)
5. **OPT-031** — 4096-row mixer + GDN prompt graphs (`mixer_mmq` / `gdn_core`)

Raw JSON: [`attribution.json`](attribution.json).
