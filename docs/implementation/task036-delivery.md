# TASK-036 fast-engine validation and delivery decision

**RETAIN_CONTROL. The FP8 candidate is not promoted, and the fast-engine goal
remains unmet.** The frozen core-54 has C92 INCONCLUSIVE and a P100 review
failure; all nine matched speed-parity targets are unmet. Production remains
the TASK-026/027 Q4_K/Q8 engine at `d2f02e2`. TASK-030's historical decision
is unchanged. See the [task report](tasks/TASK-036.md) for delivery status.

## Frozen candidate and precision

Runtime/compiler source is `aacdffe72b1a72f62afd8001784ad78d03025f5b`.
TASK-036 changes benchmark admission/labels and reporting, not inference
arithmetic. Artifact `.cache/candidates/candidate-fp8-mixer-v1.qw38` has
manifest `7fd0f81bfb7a4e6bd5ebaacda532d2992fde64ce136fef55116e7e4f292d10c7`,
policy 1029, compiler
`qw38-fp8-mixer-v1-cutlass-098de2a6-e4m3-rne-absmax448-m1n128k128` version
0.1.2. The manifest is metadata only; tensor payloads/scales are not hashed.
Source metadata hash is
`77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df`;
tokenizer hash is
`0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`.

The 208 large GDN and attention matrices use E4M3 weights compiled directly
from the pinned BF16 source, one FP32 absmax/448 reconstruction scale per
N128/K128 weight block. Per-row activation scales cover K128, with local
BF16 rounding before FP8 conversion; all-zero blocks use scale 1. RNE,
saturation, finite-input validation and scale-underflow handling are the
[TASK-035 contract](tasks/TASK-035.md). M>1 uses pinned CUTLASS groupwise
E4M3 x E4M3, FP32 accumulation, cooperative 128x128x128; M=1 uses the same
FP8 weight view with BF16 activation GEMV and does not produce unused packs.

RMS prepares one code/scale operand for GDN QKV/Z or attention QG/K/V siblings;
GDN A/B consume the same producer's BF16 companion. Gated output producers
prepare fresh output-projection operands. Plan-owned scratch survives until
the last compatible consumer on the session stream and is then reused.
Q4_K MLP gate/up/down retain TASK-034's bounded global unpack/cuBLAS prefill
fallback and packed GEMV decode. The local-unpack and NVFP4 alternatives were
rejected in TASK-034; no duplicate resident fallback weights are retained.
Q8 head uses a single normalized BF16 row and avoids a duplicate final readout.
Embeddings and small families retain their existing formats.

TASK-033 attention retains Q32/K64 QK and two local BF16 components for P x V,
FP32 accumulation/softmax, and shared FP32 decode probabilities. FP32 residual,
nonlinearity/reduction and GDN recurrence arithmetic, BF16 KV/convolution
history, absolute positions, causal masking, complete-token commit and
poison/reset/restore semantics are unchanged. Chunk size remains 256. Weights
upload once; there is no whole-model BF16 cache or persistent activation cache.

## Quality and state

Core-54 completed 54/54. Aggregate NLL delta is +0.0036741931935621993 over
701 aligned teacher targets; aggregate +0.03 and every slice +0.06 gate pass.
C92 is 7/15 versus 8/15, with the same lost SuperGPQA case
`6082513c8dba4ec68aa68f1bf5854d09`; loss interval [0, 0.20] crosses the 0.02
budget. This is INCONCLUSIVE, not evidence of population non-inferiority or
a statistically established regression. L12 is 12/12 and R-512/R-4096 each
6/6 in both arms. The sole `R-32768-s0-d0.1` extension passes 1/1 in both arms.
Other 32K fixtures remain inventory only; full-216 was not run.

All 15 P100 rows have output-bound reviews. Eleven changed candidate outputs
were reviewed by GPT-6 Codex, explicitly an AI reviewer; four unchanged
candidate and fifteen comparator judgments retain original attribution only
with identical text hashes and prompts. `case_077` incorrectly adds mapping
metadata pages to mapped-file residency and is recorded as a candidate-only
factual error. The qualitative gate is FAIL. Capped outputs stay capped;
reviewed code is not presented as executed code. JSON examples were parsed.

Same-schedule bitwise replay passed at 1/3/4/63/64/65/255/256/257 for default
decode and partitions 1/63/64/65/255/256/alternating 63,65. Reset,
snapshot/restore, interleave, nonempty prefill/decode handoff and late-failure
recovery pass. Cross-schedule full-model state/logit deltas are diagnostics;
component tolerances are not applied to the complete model. All 97 retained
TASK-035 source/evidence identities still match, preserving numerical,
fan-out, pack/tail, dispatch and binding checks. The final consumers were
rebuilt and evaluated under the hashes below.

## Measured speed and memory

One Nsight-instrumented execution per workload, zero warmups, first-use costs
included. Ratios are llama.cpp/candidate latency; each target requires >=1.
All nine are individually unmet. Request prompt phases supply prefill/TTFT;
requests produce 128 tokens and populated decode consumes 128 fixed inputs.
Authenticated TASK-027 comparator captures were reused with unchanged inputs,
settings, timing boundaries, GPU/driver/power and toolchain context. This is a
speed-only comparison, not a quality/speed Pareto claim.

| Row | Candidate ms | llama.cpp ms | Ratio | Target |
| --- | ---: | ---: | ---: | --- |
| prefill 256 | 228.071 | 109.548 | 0.480325 | unmet |
| prefill 4096 | 3326.134 | 1379.908 | 0.414868 | unmet |
| prefill 32768 | 42529.240 | 12372.765 | 0.290924 | unmet |
| decode 512 | 3866.102 | 1932.101 | 0.499754 | unmet |
| decode 4096 | 3935.183 | 1952.946 | 0.496278 | unmet |
| decode 32768 | 4857.245 | 2131.478 | 0.438824 | unmet |
| request 256 | 4106.839 | 2027.903 | 0.493787 | unmet |
| request 4096 | 7274.030 | 3335.562 | 0.458558 | unmet |
| request 32768 | 47372.679 | 14480.757 | 0.305677 | unmet |

At 32K, load/upload was 5416.886 ms, session-inclusive cold setup 5423.378 ms,
and separate CUDA initialization 294.243 ms. No OS cache eviction was used.
Populated decode setup was 42383.362 ms, excluded from its measured 128 steps.
These are observations, not repetition statistics.

The final request allocated 32,896 slots (32,768 plus 128). Sampled free
memory after execution was 8,782,807,040 bytes, above the 2 GiB reserve.
Whole-process tracked allocation peak was 23,796,265,256 bytes, including load
and transients; sampled resident growth was separately 24,303,894,528 bytes.
Model including codes/scales/padding is 21,013,686,400 bytes, persistent state
2,309,816,320 and session scratch 256,901,120. Additional prefill/output/library
allocations are included in the tracked peak, not omitted from capacity.
The FP8 codes/scales are 7,214,202,880/1,761,280 bytes with no matrix padding
for actual shapes. The prefill engine workspace is 83,410,944 bytes, including
FP32 accumulation slabs and bounded unpack storage; it is not the complete
session allocation. Reused TASK-035 compiler peak RSS is 4,033,146,880 bytes.
These categories and residency/peak measurements must not be added together.

| Execution | Allocation peak ratio | Sampled resident growth ratio |
| --- | ---: | ---: |
| decode 512 | 0.909682 | 0.892165 |
| decode 4096 | 0.910817 | 0.893482 |
| decode 32768 | 0.917671 | 0.901545 |
| request 256 | 0.903483 | 0.886125 |
| request 4096 | 0.910817 | 0.893482 |
| request 32768 | 0.917671 | 0.901545 |

Memory ratios are llama.cpp/candidate; request observations also cover their
derived prefill rows. No separate prefill peak or maximum-context search was
performed. Coverage is one GPU, one sequence, primary-language execution;
vision, MTP, batching, sampling and larger contexts are not established.

Remaining measured costs rank as: 32K prefill projections plus unpack
20.475 s GPU (including 8.181 s global unpack), attention 19.247 s, then GDN
1.757 s. Populated decode-4096 projections including FP8 GEMV take 3.020 s of
3.935 s elapsed. CPU waits overlap GPU work and are not extra GPU time.
The next smallest justified performance change is one real-shape Q4_K prefill
consumer improvement that removes the measured unpack cost without repeating
TASK-034's rejected integration; retain it only if complete consumer cost wins.
Resolve C92 uncertainty and the P100 error before promotion. No additional
experiment or roadmap sequence is launched by this task.

## Reproduction and rollback

Use RTX 5090, driver 590.48.01, existing 400 W limit, CUDA 13.4.1, GCC 14,
Release C++23/CUDA23 native sm_120 and CUTLASS
`098de2a652cf8f00fd70b2df54051c7eccbb855a`. From the repository root with the
pinned checkpoint and CUTLASS checkout already available:

```bash
docker run --rm --gpus all -u "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" -w /workspace \
  sha256:3844dc9c37087cecd40f96e62bd4f305ad405408f0d63312bda8aff8651f2b49 \
  bash -lc 'cmake -S . -B build/pinned-release -DCMAKE_BUILD_TYPE=Release &&
    cmake --build build/pinned-release --target qw38_compile qw38_decode qw38_evaluate qw38_state_replay qw38_bench_request -j4'
# Only if the frozen artifact is absent:
docker run --rm --gpus all -u "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" -w /workspace \
  sha256:3844dc9c37087cecd40f96e62bd4f305ad405408f0d63312bda8aff8651f2b49 \
  build/pinned-release/src/qw38-compile \
  --checkpoint .cache/authorities/qwen3.8-27b-transformers \
  --output .cache/candidates/candidate-fp8-mixer-v1.qw38 --format fp8-mixer
docker run --rm --gpus all -u "$(id -u):$(id -g)" \
  -v "$PWD:/workspace" -w /workspace \
  sha256:3844dc9c37087cecd40f96e62bd4f305ad405408f0d63312bda8aff8651f2b49 \
  build/pinned-release/src/qw38-decode \
  --artifact .cache/candidates/candidate-fp8-mixer-v1.qw38 \
  --tokens-file .cache/evaluation/qw38-language-v2/tokens/case_000.prompt.u32le \
  --generate 8
```

This candidate command is diagnostic, not the recommended production engine.
The decoder consumes little-endian uint32 IDs from the pinned tokenizer and
uses a fixed greedy cap without EOS stopping. To roll back, use the separate
`d2f02e2` checkout and Q4_K/Q8 artifact in the
[retained-control recipe](task030-delivery.md#use-the-retained-production-control).
Changing only the artifact does not restore the older attention runtime.

Exact acceptance scripts, logs, comparator authentication, replay JSON,
profile captures, summaries and review records are retained in
`.cache/evaluation/qw38-language-v2/task036-support/`. `validation.sh` performs
the frozen execution; `resume-scoring.sh` records the first successful paired
score and six captures; `finalize-evidence.sh` refreshes reporting only after
FP8 GEMV attribution repair. Failed scoring attempt 1 is retained; it failed
on the missing policy label, not inference. The 32K score was not repeated.
Final report scripts bind their own source hashes and unchanged run identities.
Evaluator SHA-256:
`1983d5d90c39931755a6698276d7de1f103d35b2428c1a9bc0a0197ac0e70c8e`;
state replay:
`3fb7d26addeb2cb00041b2cebe16cfbe49edf86865341ae12db3cc83c740dfb7`;
request benchmark:
`2976012634525e08a1dc0e69e992896af5f2861338a275af921d8c1737e3f8b5`.
