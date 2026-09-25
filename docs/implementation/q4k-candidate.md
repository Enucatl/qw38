# Q4_K MLP candidate, existing Quartz runtime precision

This candidate replaces only the 192 primary-language MLP gate, up, and down
matrices with the no-imatrix reference Q4_K fitting algorithm from
[llama.cpp e6ab7c1a4](https://github.com/ggml-org/llama.cpp/tree/e6ab7c1a4).
The separately compiled production port and test oracle retain attribution and
the MIT license in `third_party/llama.cpp-q4k`. The external comparator's
original calibration provenance remains unknown.

The compiler reads original checkpoint BF16 values directly. It fits eight
32-weight groups per 256-weight superblock, compresses fitted scales/minima to
six bits, stores two FP16 superblock scales, and recalculates unsigned 0–15
codes using those compressed values. Reconstruction is
`(d * group_scale) * code - dmin * group_min`, with separate FP32 operations,
then Quartz BF16 operand rounding. CPU and CUDA FP16 decoding now correctly
map `0x0001` to `2^-24`; subnormal scales are supported.

| Identity | Name | Wire value |
| --- | --- | --- |
| Logical quantizer | `Q4KCandidateV2` | `0x0105` |
| Physical layout | `CudaQ4KCandidateV2` | `0x020D` |
| Candidate policy | `CandidateV2` | `0x0403` |

Use `qw38-compile --format candidate-q4k`. Existing `candidate`, Q4G64 and
Q8G32 identities retain their meanings. The compiler revision is
`qw38-candidate-v2-q4k-llama-e6ab7c1a4-no-imatrix-bf16-operands`.
Container/manifest record grammar is unchanged. Older readers reject the new
IDs. The scale span is metadata: per tile row, little-endian FP16 `d`, FP16
`dmin`, and the reference's twelve packed scale/minimum bytes. Tiles remain
`[N/8,K/256,row]`, with 128 code bytes per row in adjacent-coordinate nibble
order (earlier coordinate low), and 16 metadata bytes. This physical nibble
order differs from llama.cpp's native block; logical codes and decoded weights
are equivalent. Padded rows have zero codes and metadata. Finite nonnegative
superblock scales, including subnormals, are accepted.

The existing candidate Q8 attention/GDN/head, remaining BF16 weights, inactive
MTP, BF16 activations and caches, FP32 accumulation/state/residual/logits,
fused gate/up/SwiGLU, down/residual, and execution schedule remain unchanged.
Additional weight bytes are exactly `192 * 17408 * 5120 / 256 * 8`, or
534,773,760 bytes (510 MiB), excluding alignment. Compilation uses bounded
eight-row source buffers; verification independently unpacks one tile and
recomputes its source quantization. No model payload hashes are introduced.

Verification includes independent pinned reference blocks and reconstruction,
zero/constant/asymmetric/outlier/subnormal cases, final code recalculation,
exhaustive finite FP16 decoding, format rejection and roundtrips, compiler
family/storage checks, and CUDA decoded-BF16 contractions/fused epilogues.
`qw38_q4k_projection_screen` compares both quantizers against original BF16
weights on the saved TASK-019 MLP inputs. `qw38_q4k_artifact_audit` compares
the complete old/new artifacts, requiring byte-identical non-MLP payloads and
unchanged arithmetic/state/scratch/graph metadata.

The frozen development screen uses the first eight existing TASK-020
validation windows (384 prompt and 128 target tokens each), the actual Quartz
runtime for both candidate arms, and the existing comparator scores with
matching token hashes. Before observing scores, the promotion condition is:
Q4_K improves over Q4G64 and comparator-relative NLL is at most +0.03
nats/token. `scripts/q4k_development_screen.py` freezes inputs and reports the
result. Passing this diagnostic permits the full language-v2 TASK-022 gate;
it does not establish acceptance. No thresholds or core fixtures change.

On all 1,024 saved TASK-019 inputs per layer-0 MLP matrix, Q4G64 → Q4_K
relative L2 projection error against Quartz's original-BF16 contraction was:

| Projection | Q4G64 | Q4_K |
| --- | ---: | ---: |
| Gate | 0.105246 | 0.069311 |
| Up | 0.105174 | 0.069484 |
| Down | 0.033951 | 0.022115 |

Raw results: `.cache/q4k-candidate/projection-errors.jsonl`. These are component
diagnostics, not model quality acceptance.

The saved TASK-020 layer 0/31/63 traces were also tested for gate, up, and
down, with all 384 prefill inputs and one decode input each. Saved FP32 inputs
were rounded to BF16 for both arms. Relative L2 error improved in all 18
projection/layer/phase comparisons, by approximately 32.5–36.1%. This does not
mean every error statistic improved: layer-0 down/decode maximum absolute
error increased from 0.003377 to 0.004947 while relative L2 fell from 0.040004
to 0.026999. Raw results:
`.cache/q4k-candidate/trace-projection-errors.jsonl`.

The pinned Release build passed 13 targeted CTests covering format, compiler,
reference, CUDA contractions, and MLP epilogues. The CUDA test includes 2,048
exact one-hot decoded BF16 operands. Evidence:
`.cache/q4k-candidate/pinned-build-tests.log`. The screen's frozen-input and
artifact-identity guards passed 19 pytest cases; targeted Ruff and shell syntax
checks passed. The earlier TASK-022 working state was checkpointed separately
as commit `73ef3b8` before this implementation.

## Full artifact and development result

The original BF16 checkpoint compiled with complete source reconstruction
verification in 1,768.31 seconds. The artifact is
`.cache/q4k-candidate/candidate.qw38`, 21,462,978,185 bytes; reported peak RSS
was 20,704,108,544 bytes (including mapped source/artifact pages). Its manifest
SHA-256 is
`6c8f9c87853c5e0f5ea30ce0d0add43fb1c8e0daf0064496984d218f3e3a86ad`.
The old Q4G64 candidate manifest remains
`94c9ed5c9260ebde73b0eb9316b6ae7726ff82fe030f2d4caa72760deec79dd1`.

The full artifact audit passed: exactly 192 primary MLP matrices changed,
534,773,760 weight bytes were added, every other payload and scale span is
byte-identical, and precision bindings, state, scratch, graph and aliases
are unchanged. Both Quartz runtime arms completed all eight frozen windows,
scoring the same 1,024 targets using the same evaluation executable.

| Arm | NLL (nats/token) |
| --- | ---: |
| Existing llama.cpp comparator | 1.773838533 |
| Quartz Q4G64 candidate | 1.919047834 |
| Quartz Q4_K candidate | 1.913282631 |

Q4_K minus Q4G64 is **−0.005765203 nats/token**; Q4_K minus comparator is
**+0.139444098 nats/token**. The frozen +0.03 promotion condition failed.
The full TASK-022 gate was therefore not launched, and TASK-022 remains
blocked. The component improvement does not close the full-model quality
gap under the retained Quartz precision and execution schedule. This screen
does not identify the remaining cause or establish quality acceptance.

### Subsequent RoPE correction

The result above belongs to the original compiler patch-1 artifact. A
subsequent [layer-3 attribution](task022-numerical-attribution.md) found that
its frozen `rope.inv_freq` payload was incorrect. A compiler patch-2 rebuild
with the same Q4_K precision policy scores 1.775325938 nats/token on the
unchanged eight-window screen, only +0.001487405 above the llama.cpp
comparator. This clears the frozen development promotion bound but does not
replace the required complete language-v2 TASK-022 gate. The patch-1 artifact
and all scores above remain historical evidence.

Evidence: `.cache/q4k-candidate/development/report.json` (per-case scores,
frozen input hashes, both artifact identities),
`.cache/q4k-candidate/artifact-audit.json`,
`.cache/q4k-candidate/compile.log`, and
`.codex-wake-run/84edb24876b9.log`. The exact launch command was
`bash scripts/run_q4k_development_screen.sh`. The evaluator SHA-256 was
`f9615dff1dd3840ac7b31ecbe8bed6d1aa79e71b376476edba67faf2cce618ed`;
compiler and audit binary identities are in
`.cache/q4k-candidate/binaries.sha256`.
