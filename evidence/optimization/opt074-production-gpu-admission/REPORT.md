# OPT-074 production-shape llama GPU numerical admission

Status: **measured**. `claims_performance_improvement: false`. Authority is pinned llama.cpp `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` and GGUF SHA-256 `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`.

OPT-059 v1/v2 historical missing-evidence records are preserved. This sitting exports full-M N=1 llama GPU projections on captured BF16 activations. `--row-limit` is not production evidence.

## Frozen OPT-059 cases

Calibration layers 0/3/31/32 at tokens 128/512. Held-out layers 62/63 at 2048/4095. Host checks sample 16 rows; dispatch uses full M. Held-out validates frozen calibration ceilings and cannot enlarge them.

## Admission table

| Family | K | M | v2 admitted | coverage | complete | held-out validates | calib abs | held-out abs |
|---|---:|---:|---|---|---|---|---:|---:|
| q4_down_k17408 | 17408 | 5120 | False | unadmitted | True | False | 0.00309331773249244 | 0.10352671070461739 |
| q4_gate_up_k5120 | 5120 | 17408 | False | unadmitted | True | False | 0.008908959963719099 | 0.04322713627061603 |
| q6_vocab_k5120 | 5120 | 248320 | True | admitted | True | True | 0.025587059251023447 | 0.031267232440086445 |
| q8_mixer_k5120 | 5120 | 10240 | False | unadmitted | True | False | 0.027848582650449316 | 0.03943854112878853 |
| q8_mixer_k6144 | 6144 | 5120 | False | unadmitted | True | False | 0.002138529470133932 | 0.02645971159584093 |

## Dispatch and staging

Quantized N=1 uses llama CUDA `mmvq` (`ggml_cuda_should_use_mmvq` on Blackwell, ne11=1). CUDA backend synchronize plus `cudaDeviceSynchronize` before host reads. Q8_1 bytes come from the private adapter `quantize_q8_1_sum_x` (`half(sum(x))`), not unsupported `ggml_cpy` F32→Q8_1. Quartz `sum_q`, llama `sum_x`, and exact integer sums (Q4 MMV recomputes) stay distinct consumers.

## Proof limit

- no throughput claim
- no production pin changes
- full-M N=1 llama GPU dispatch
- no --row-limit evidence
- held-out cannot enlarge calibration ceilings
- wildcard columns=0 is not production coverage
- absent layer-kind is explicit
- OPT-059 historical missing evidence preserved
