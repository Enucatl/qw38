# TASK-030 runtime and delivery decision

Retain the TASK-026/027 accepted Q4_K/Q8 runtime as the production control.
The combined TASK-028 attention / TASK-029 Q4 fallback candidate is **not
promoted**: its frozen core-54 C92 result is INCONCLUSIVE under EVAL-01.
The [task report](tasks/TASK-030.md) records final reviews and delivery status.

The final candidate uses source `bea89024f4a6970976b94dc909ed56056d90a344`.
TASK-030 changes reporting and documentation, not inference arithmetic.
Evaluator SHA-256 is
`c720b2b419157fcb4609672d899d785da00ba6f8c4733b5a5311f3c6baa1764b`;
request benchmark SHA-256 is
`278adad399476b38385094ce4adba37a8ff30bef1f4c8b53bb226f4db53b4f4a`.
Exact build, source, replay and hardware identities are preserved in
`.cache/evaluation/qw38-language-v2/task030-support/`.

## Artifact and calculation

Both control and candidate load
`.cache/candidates/candidate-v2-q4k-rope-fixed.qw38`, manifest digest
`41c1f5e673bb24eb2fb283aa6044dbccdebecc7cd85f847815b3c02a6763fc43`,
precision policy 1027. This is the [CandidateV2 format](task021-candidate-format.md):
Q4_K primary-language MLP gate/up/down, Q8 attention/GDN/head, BF16 embeddings
and small parameters. Compiler identity is
`qw38-candidate-v2-q4k-llama-e6ab7c1a4-no-imatrix-bf16-operands`, version 0.1.2.
The artifact uses the pinned BF16 source and existing no-imatrix recipe;
TASK-030 does not recalibrate it. Source metadata hash is
`77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df`;
tokenizer JSON hash is
`0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`.
Source identity is metadata-only; never hash tensor payloads or scales.

Upload one resident view per tensor. FP32 residuals feed RMS with BF16 output,
then bounded unpack/cuBLAS prefill or grouped GEMV decode. MLP combines gate/up
with FP32 SwiGLU, stores BF16 intermediate values, and adds the down projection
to the FP32 residual. GDN keeps FP32 recurrence in `[head,value,key]` and BF16
convolution history; attention keeps BF16 KV. No whole-model floating-point
weight cache, persistent duplicate view, request-time repack or CUDA graphs.

Prefill processes at most 256 tokens per chunk, in layer order. The candidate's
attention uses Q32/K64 tiles with native BF16 QK, FP32 online softmax/PV and
cooperative dot products in segmented decode. The production control retains
its earlier attention scan. Both preserve QK normalization, partial RoPE,
absolute positions, causal masking, nonempty-session continuation and the state
ABI. Generation reads final-position logits; evaluation requests bounded rows.
Complete-token commit, poison/reset/restore and in-memory snapshot semantics
remain unchanged. Cross-schedule numerical differences are diagnostic;
same-schedule replay must be bitwise identical.

[TASK-029 NVFP4](task029-nvfp4-format.md) remains an optional implemented format,
rejected for conversion-inclusive request/decode cost. It is not the delivered
weight selection and does not consume a second resident fallback view.

## Coverage and limits

The final candidate completed core-54, the single `R-32768-s0-d0.1` quality
case, nine replay checkpoints and all seven prefill partition patterns.
Teacher-target NLL delta is +0.0012929024 nats/token over 701 targets; aggregate
and slice NLL gates pass. L12 is 12/12 and retrieval at 512/4096 is 6/6 per
horizon in both arms; fixed 32K retrieval is 1/1 in both arms. C92 is 7/15
versus 8/15: case `6082513c8dba4ec68aa68f1bf5854d09` answers E instead of A.
The aggregate accuracy-loss interval [0, 0.20] crosses the 0.02 budget, so
neither promotion nor a statistically established regression is claimed.

Validated capacity is 32,768 prompt tokens plus room for 128 continuation
tokens (32,896 allocated slots). Free memory after the final 32K workloads was
8,317,239,296 bytes, above the 2 GiB reserve. This is sampled free memory;
the traced allocation peak is 24,196,583,696 bytes over the whole process,
including load and transient allocations. Model bytes including scales/padding
are 21,462,812,800, persistent state 2,309,816,320 and scratch 256,901,120;
the peak also includes library/transient allocations. Sampled resident growth
is separately 24,769,462,272 bytes, including context/allocator overhead.
These measurements are not additive. No maximum-context search was performed. Claims cover
one GPU, one primary-language sequence and the fixed diagnostic cases; they
do not establish MTP, vision, batching, sampling or untested context coverage.

Observed memory ratios are llama.cpp/QW38; values above one favor QW38.
Allocation peaks cover the whole process, including load and transients.
Resident growth is a separate post-run sample, not a measured residency peak.

| Execution | Tracked allocation peak ratio | Sampled resident growth ratio |
| --- | ---: | ---: |
| decode-512 | 0.893191 | 0.873831 |
| decode-4096 | 0.894479 | 0.875309 |
| decode-32768 | 0.902489 | 0.884599 |
| request-256 | 0.887092 | 0.867902 |
| request-4096 | 0.894479 | 0.875309 |
| request-32768 | 0.902489 | 0.884599 |

The request observations also cover their derived prefill rows; no separate
prefill memory peak was measured. Reproduce the ratios from saved data with
`uv run --script .cache/evaluation/qw38-language-v2/task030-support/memory-ratios.py`.
Inputs are the final `task030-support/profiles.json` and the retained
`task027-support/single-run/profiles.json`, both beneath
`.cache/evaluation/qw38-language-v2/`. `task030-support/memory-ratios.json`
records their SHA-256 identities, raw byte counts and unrounded ratios; the
comparator profile also matches its original reviewed evidence digest.

All nine PERF-01 rows remain below llama.cpp speed parity. Timings are one
Nsight-instrumented execution per workload, with first-use costs, zero warmups
and no repetition statistics. Each request supplies its prefill/TTFT row.
TASK-027 llama.cpp observations are reused after input, adapter, model,
settings, hardware and timing-boundary checks. See `comparator-reuse.md` and
`request-adapter-change.diff` in the support directory. This is a speed-only
comparison, not a quality/speed Pareto claim.

## Reproduce the candidate

Use the pinned CUDA image on the RTX 5090 with the existing 400 W limit,
GCC 14, CUDA 13.4 and native `sm_120`. The current build also requires the
existing CUTLASS checkout at `098de2a652cf8f00fd70b2df54051c7eccbb855a` for
the optional NVFP4 implementation. From the repository root:

```bash
docker run --rm --gpus all -u "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" -w /workspace \
  sha256:3844dc9c37087cecd40f96e62bd4f305ad405408f0d63312bda8aff8651f2b49 \
  bash -lc 'cmake -S . -B build/pinned-release -DCMAKE_BUILD_TYPE=Release &&
    cmake --build build/pinned-release --target qw38_decode qw38_evaluate qw38_state_replay qw38_bench_request -j4'
```

`qw38-decode` accepts comma-separated token IDs or a little-endian uint32 token
file from the pinned tokenizer. It emits token IDs and uses greedy decoding;
`--generate` is a fixed cap and does not stop on EOS. Example:

```bash
docker run --rm --gpus all -u "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" -w /workspace \
  sha256:3844dc9c37087cecd40f96e62bd4f305ad405408f0d63312bda8aff8651f2b49 \
  build/pinned-release/src/qw38-decode \
  --artifact .cache/candidates/candidate-v2-q4k-rope-fixed.qw38 \
  --tokens-file .cache/evaluation/qw38-language-v2/tokens/case_000.prompt.u32le \
  --generate 8
```

The executed acceptance commands are retained in `validation.sh`,
`score-and-measure.sh` and `measure.sh` under the support directory. They are
recorded reproductions, not instructions to rerun already captured workloads.
The quality scorer uses `task030-eval-policy-rebind.json` to authenticate
unchanged fixtures and comparators against DELIVERY-01. Never launch full-216
from automation or combine outputs from different candidate identities.

## Use the retained production control

The accepted control is source commit `d2f02e2` (TASK-027), with the same
artifact and tokenizer. Changing only the artifact path does not roll back
attention, because the two runtimes share these weights. Build the control in
a separate checkout and stop the candidate process before loading it:

```bash
git worktree add --detach ../qw38-control d2f02e2
docker run --rm -u "$(id -u):$(id -g)" \
  -v "$PWD/../qw38-control:/workspace" -w /workspace \
  sha256:3844dc9c37087cecd40f96e62bd4f305ad405408f0d63312bda8aff8651f2b49 \
  bash -lc 'cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release &&
    cmake --build build/release --target qw38_decode -j4'
docker run --rm --gpus all -u "$(id -u):$(id -g)" \
  -v "$PWD/../qw38-control:/workspace:ro" -v "$PWD:/assets:ro" -w /workspace \
  sha256:3844dc9c37087cecd40f96e62bd4f305ad405408f0d63312bda8aff8651f2b49 \
  build/release/src/qw38-decode \
  --artifact /assets/.cache/candidates/candidate-v2-q4k-rope-fixed.qw38 \
  --tokens-file /assets/.cache/evaluation/qw38-language-v2/tokens/case_000.prompt.u32le \
  --generate 8
```

The control's quality/capacity acceptance remains TASK-026 and its speed
baseline remains TASK-027. The rollback recipe selects that recorded source;
TASK-030 does not claim a fresh control evaluation or rebuild measurement.
