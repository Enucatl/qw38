# OPT-061 — Real-input streaming component replay

This increment **claims no performance improvement** and makes **no throughput**
claim. It delivers reusable captured FFN/mixer inputs, hot versus rotating
weight walks, complete-family CUDA-event times, and hardware/resource evidence
for later A/B tasks. Isolated ggml MUL_MAT times are **not** fused llama FFN.

## Proof limits

- Hot `cache_mode` repeats one matrix and is diagnostic only. Production
  acceptance for later tasks must use rotating weights.
- Paired decode gate/up is **64 calls per token**, not 128 pairs.
- Capture copies are untimed setup. A new capture is cached by source/input
  identity and is not repeated per candidate.
- Nsight Compute is optional (`--launch-count 1`, no `--set full`). Timeout or
  absence does not block event replay. Useful-byte rate from a streaming read
  larger than 2×L2 is an **estimate**, never a DRAM-counter claim.
- No end-to-end sitting.

## Families

| Family | Complete scope | Notes |
|---|---|---|
| `decode-ffn` | RMSNorm, one shared Q8 stage, paired gate/up/SwiGLU, down, residual | Production pin `paired_staged` |
| `decode-mixer` | Shared RMS + Q8_1 stage and all real input projections | GDN: qkv/gate/alpha/beta; attention: q/k/v. Core/state excluded |
| `prompt-ffn` | One 4096-row complete FFN via `execute_prompt_ffn` | Sampled activation rows 0,1024,2048,4095 |

## Capture

Calibration layers **0, 3, 31, 32**; held-out **62, 63**. BF16 projection
inputs, FP32 residuals, FFN down input, output-norm input, and prompt-row
indices are stored with GGUF tensor **names** resolved through
`bind_model_weights` / `pins/tensor_inventory.json`. Hard-coded byte offsets
from OPT-046/049 harnesses are not used.

Family bundles:

- `evidence/optimization/opt061-component-replay/capture-bundle-decode-ffn.json`
- `evidence/optimization/opt061-component-replay/capture-bundle-decode-mixer.json`
- `evidence/optimization/opt061-component-replay/capture-bundle-prompt-ffn.json`

Replay command template (fill `capture_key` from provenance JSON):

```
./build/qw38-cuda-component-replay --workload {family} --cache-mode rotating \
  --capture-key {capture_key} --rows 0,1024,2048,4095
```

Decode capture key:
`ffbd7cbc40792376734d837430d71c68fd266cba9f37194f5aea8d1145c395bd`.
Prompt capture key:
`2dcfaa062bbb3af5e0f285d3124ca454c2e5962b301753caa6ed893c92dbec63`.
Reference cache key is the capture identity SHA-256. OPT-059 GPU numerics
remain immutable; this sitting validated independently decoded FP64 control
dots on a tiny Q4 case (16 rows × 4 vectors, 64 dots, zero nonfinites).

## Hardware

Pinned RTX 5090, `cudaDeviceProp` + `nvidia-smi` in `hardware.json`:

- SM count 170, L2 100663296 bytes (96 MiB), shared 48 KiB/block, 100 KiB/SM
- Power cap **400 W** (readback), clocks/temperature recorded per sitting
- Compiled paired gate/up kernel: 48 registers, 0 local bytes, occupancy 10
- `ncu` present in the CUDA image; `nsys` absent. No `--set full`. Event replay
  did not depend on counters.
- Streaming checksum over >2×L2: useful-byte **estimate** about 1015 GB/s
  (`dram_counter_claim=false`).

## Measured complete-family events (acceptance, 3 warmup + 10 rounds)

Independent rounds, control only, 64-layer set, working set 9.626 GB FFN
weights (>2×L2). Gate/up and down charged **64 calls** per round.

| Family | Hot enclosing ms | Rotating enclosing ms | Hot kernel ms | Rotating kernel ms |
|---|---|---|---|---|
| decode-ffn | 15.75 | 17.20 | 13.51 | 14.91 |
| decode-mixer | 2.61 | 5.44 | 1.59 | 4.13 |
| prompt-ffn | 693.40 | 699.91 | 692.83 | 699.46 |

Mixer is the family where hot versus rotating disagrees most (about 2.1×).
Decode and prompt FFN already stream more than L2 even when repeating one
layer, so the hot/rotating gap is small. OPT-046/047 single-matrix
microbenchmarks can therefore look unlike rotating replay; that is expected
and is not forced into a fake reconciliation.

## Ranking (OPT-060 attribution + rotating replay)

| Rank | Family | OPT-060 enclosing | Rotating replay enclosing |
|---|---|---|---|
| 1 | prompt-ffn | ~666 ms / 4096-token prompt | 699.91 ms / 64-layer 4096-row family |
| 2 | decode-ffn | ~15.43 ms / decode token | 17.20 ms / 64-layer complete FFN |
| 3 | decode-mixer | ~7.10 ms / decode token | 5.44 ms shared-input projections only |

Prompt FFN remains the largest Quartz-owned interval. Decode FFN is the
largest decode family on the complete-FFN boundary. Mixer rotating time is
lower than the OPT-060 mixer enclosing interval because this replay excludes
GDN/attention **core** (stateful). Isolated OPT-043 MUL_MAT timings stay
labeled isolated-op and are not the fused llama FFN cost.

Successor family work (OPT-062+) must not treat hot-cache wins as production
evidence. Export for dependents is in
`evidence/optimization/opt061-component-replay/provenance.json`.
