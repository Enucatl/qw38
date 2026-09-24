# TASK-020 provisional selection

The candidate policy is [task020-policy.json](task020-policy.json). It selects
Q4G64 for MLP gate/up/down, Q8G32 for attention and GDN projections and the
vocabulary head, and BF16 for the embedding, norms, convolution and small
gate/time controls. Projection activations remain BF16; accumulation, residual
and recurrent arithmetic remain FP32. The policy uses one resident packed view.
No clipping, activation scaling, rotation or smoothing is selected.

## Inputs and measurement

The BF16 source is
`.cache/authorities/qwen3.8-27b-transformers`; the source config and index
metadata identities are in the policy record. The tokenizer JSON identity is
`0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`.
The BF16-to-GGUF conversion command and log are in
`.codex-wake-run/429b1d9281b0.log`; it used the local
`ghcr.io/ggml-org/llama.cpp:full-cuda13` image, whose inspected repo digest
is the immutable `sha256:93e004aeaddfcab61c219c65159e28557142054f63d27d4427daf178ec663cc6`
used by the remaining scripts. Conversion did not execute BF16 inference.
The frozen `Salesforce/wikitext` `wikitext-103-raw-v1` revision is
`b08601e04326c79dfdd32d625aee71d232d685c3`. Its train split supplies
128 disjoint 512-token calibration windows. The validation split supplies 32
separate 512-token development windows; each uses 384 prompt IDs and 128 fixed
continuation IDs. The split manifests under `.cache/task020/tokens/` record
parquet shard identities, row counts, tokenizer identity, the sampler source,
window IDs and window digests. No final evaluation split or answer was used.

`task020_prepare_calibration.py` decodes the exact train windows into two
segments which reencode to the same 65,536 token IDs. A single text join would
have changed token boundaries; the discarded diagnostic is retained as
`.cache/task020/calibration-imatrix-newline-invalid.*`. Pinned llama.cpp
`llama-imatrix` runs on a BF16-derived Q4_K_M model with all 65 layers on GPU.
The combined matrix covers 496 model tensors as 992 sum/count records, each
with 65,536 observations. The
per-channel RMS summaries are in `.cache/task020/calibration-summary.json`.
This is quantized-model activation calibration, with no BF16 whole-model run.

The fixed token IDs are scored with a pinned llama.cpp CUDA build
`e6ab7c1a4` in image digest
`sha256:93e004aeaddfcab61c219c65159e28557142054f63d27d4427daf178ec663cc6`
on the RTX 5090 (SM120), driver 590.48.01. The scorer forces every model
tensor, including the embedding, to CUDA0. Each model uses the same IDs and
teacher-forced next-token negative log likelihood. First-token agreement and
greedy common-prefix length are secondary diagnostics. The score TSVs and
load logs are under `.cache/task020/gpu-*-scores.tsv` and
`.cache/task020/gpu-*-score.log`; the compact comparison is
`.cache/task020/full-gpu-comparisons.json`. The scorer source SHA-256 is
`21128e522c6c4d91dd8ffe4a47af6ac0dfff1db3f715f93e140e33c55e32e990`
and the compiled binary SHA-256 is
`3c5577019867678a02e9f3bcd09314f7d64c0fa66effd1be2b658284d1e5d789`.
Every run reports 65/65 layers on
GPU, no CPU mapped model buffer and all 32 cases scored. The external
`models/Qwen3.8-27B-Q4_K_M.gguf` is a separate reference: its tensor types
include Q8_0 attention/GDN, Q4_K MLP and Q6_K attention output/head, whereas
the freshly converted controlled Q4_K_M model has different allocation.
Their average target NLLs are 1.945632 and 1.949951 respectively.

All experimental GGUFs were quantized directly from the 51 GiB BF16 GGUF
conversion, not from the external Q4 artifact. The table gives paired mean
NLL change against the freshly converted Q4_K_M control, in nats per target
token over 4,096 target tokens. Positive is worse on these development cases.

| BF16-derived GGUF variant | Change |
| --- | ---: |
| NVFP4 dense projections | +0.025601 |
| NVFP4 dense except MLP down | +0.017027 |
| NVFP4 MLP gate/up only | +0.000113 |
| NVFP4 attention/GDN only | +0.010164 |
| MXFP4 dense projections | +0.034897 |
| NVFP4 vocabulary head only | +0.009167 |
| Q8_0 vocabulary head only | +0.000229 |
| BF16 embedding and small GDN gate/time controls with Q8_0 head | +0.000230 |
| Q8_0 attention/GDN and head; Q4_K MLP; BF16 controls | **−0.002632** |

The selected GGUF proxy averages 1.947319 nats/token, or +0.001687 against
the external artifact. It wins 10 of 32 cases against the external artifact;
its first-token agreement is 13 of 32 versus 14, and its mean greedy prefix
is 0.90625 tokens versus 0.96875. These small fixed-continuation differences
are a selection signal, not EVAL-01 acceptance or a claim of equivalent output
quality. Q4_K and Q8_0 GGUF have different logical and physical quantization
from QW38 Q4G64 and Q8G32; llama.cpp's converted GGUF also stores 353
other small/control tensors as F32, while QW38's retained policy uses BF16.
The selected policy still requires compilation,
runtime integration and acceptance in TASK-021/022/026.

## Error attribution

The component probe reads full BF16 and NVFP4 matrices and quantized GPU
model inputs from `train.window-000`: selected rows of its 384-token prompt
and the first populated decode input in layers 0, 31 and 63. It checks
BF16 tensor rows against the source checkpoint, including Qwen3.5 V-head
permutation, then independently perturbs weights, activations and both.
Results for 24 matrix/phase pairs are in
`.cache/task020/component-ablation.json`. The projection probe excludes
`output.weight.prefill.bin` from the shared trace directory. After this repair,
replay against the original GGUFs reproduced all 24 results byte for byte in
`.cache/task020/component-ablation-replay.json`; the command, seven passing
focused tests and comparison are recorded in `.codex-wake-run/5689952f8f60.log`.
The separate full 248,320-row head
probe is in `.cache/task020/head-ablation.json`. Selected relative output L2
errors (weight / activation / combined) are:

| Matrix and phase | Weight | Activation | Combined |
| --- | ---: | ---: | ---: |
| layer 0 GDN qkv prefill | 0.0351 | 0.0221 | 0.0409 |
| layer 31 attention output prefill | 0.1270 | 0.1099 | 0.1675 |
| layer 31 MLP down decode | 0.1065 | 0.0804 | 0.1317 |
| layer 63 MLP down prefill | 0.0778 | 0.0437 | 0.0891 |
| vocabulary head prefill | 0.0953 | 0.0804 | 0.1153 |
| vocabulary head decode | 0.0665 | 0.0585 | 0.0908 |

The layer 63 MLP down input reaches absolute 330.7 during prefill; the
component probe includes this outlier. The FP4 component probe computes a
per-row, per-16 activation scale; reconstruction is invariant to combining
rows into a chunk. Its scale recipe follows the pinned gguf Python UE4M3
encoder, including midpoint and exponent-15 behavior that differs from the
TASK-019 native-format reference. This probe diagnoses a rejected FP4 option;
the selected policy retains BF16 activations because of the observed FP4
perturbation and conversion costs. It therefore has no activation scale
granularity or lifetime. The unit checks cover E2M1 tie rounding, saturation,
UE4M3 boundary scales, zero blocks,
large outliers, nonfinite rejection, scale block orientation and chunk
sensitivity. The head and late attention/output ablations explain why an
unconditional FP4 choice is unsafe on the current screen. The much smaller
MLP gate/up score change alone does not justify a second physical format and
its activation conversion path.

The selected logical QW38 quantizers were independently applied to the same
BF16 GGUF source matrices in bounded row chunks. This probe uses Q4G64 for
MLP and Q8G32 for attention/GDN/head, rounds the reconstructed weight to
BF16 as the QW38 contraction does, and separately rounds the traced input to
BF16. All 12 selected full-layer matrices and the complete 248,320-row head
were measured in both phases. The 26 component results are in
`.cache/task020/selected-ablation.json`; selected combined output relative L2
errors include layer 31 attention output 0.00729/0.00697, layer 31 MLP down
0.09710/0.09856, and head 0.00430/0.00290 for prefill/decode. Their
activation-only errors are at most 0.00209 among the sampled matrices and
head, consistent with retaining BF16 transport. The Q4 MLP down at layer 63
has 0.09515/0.07538 combined error, slightly above the local FP4 component
probe; the paired continuation score, memory and conversion path govern the
family choice. This is a logical QW38 component calculation on quantized
model inputs; it does not replace integrated runtime checks.

## Feasibility and limits

TASK-019 demonstrated native NVFP4 and MXFP4 SM120 instructions and bounded
Q4G64 controls on real projection shapes. Its complete timing includes
CPU reference activation packing; at M=1024, MLP gate packing made the NVFP4
path 507.611 ms while the native GEMM kernel itself was 0.166 ms. A fast
packed FP4 contraction would therefore still require a new GPU activation
converter. The selected grouped policy retains BF16 activations and the
existing Q4/Q8 packing ABI; the proposed attention/GDN Q8 consumer and
prefill lowering are downstream integration work. The selected GGUF proxy
adds 4,488.03 MiB of CUDA model buffer over controlled Q4_K_M
(20,258.38 versus 15,770.35 MiB). This is a llama.cpp measurement, not a
QW38 kernel or integrated throughput measurement.

`task020_bench_selected.sh` measured the cost of the selective Q8 proxy
against the controlled Q4 model in the same pinned llama.cpp image, with CUDA
backend, 99 GPU layers, CUDA0 embedding override and no host model buffer.
Each model had three warmed repetitions of 384-token prompt processing and
128-token generation. Raw samples and build/hardware fields are in
`.cache/task020/bench-{q4-k-m,sensitive-q8}.json`. Mean prompt throughput
was 3,214.87 versus 3,407.17 tokens/s, respectively; generation throughput
was 78.16 versus 70.57 tokens/s. The Q8 exception improved this proxy's
prompt throughput by 5.98% and reduced generation throughput by 9.70%.
The benchmark uses llama.cpp's generated inputs, not the fixed development
continuations, and cannot establish QW38 latency. The offline BF16-GGUF-to-
control and selected-proxy conversions took 172.215 and 162.637 seconds,
respectively; those one-time conversions are excluded from inference throughput.

The inventory-derived [memory budget](task020-memory-budget.json) counts
Q4G64 MLP, Q8G32 attention/GDN and head, BF16 controls, retained inactive
MTP, FP32 recurrent state, BF16 KV for 32,768 tokens, scratch/workspace,
graphs and metadata. Estimated resident payload is 19,958.53 MiB and peak
24,073.35 MiB, leaving 5,444.65 MiB after a 2 GiB reserve against the
TASK-019 measured 31,566 MiB starting free. The estimated peak is not a
measured integrated QW38 runtime peak. A second complete Q4 projection view
would add 12,923,699,200 bytes and exceed this budget.

Source scripts are under `scripts/task020_*`; focused tests are
`tests/test_task020_*`. The reproducible long commands were run with
`wake-run`; their exact invocations and result logs are available under
`.codex-wake-run/`. The source and result artifacts above provide the full
per-case, per-component and GPU placement evidence behind this selection.
