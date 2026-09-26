# TASK-035 — Prepared FP8 weights and activation reuse

## Status

DONE

## Milestone and dependency

M14 — Shared narrow operands across real consumers. Depends on TASK-034.
[FAST-01](../post-task030-plan.md) defines the intended dataflow and lifetimes.

## Chosen implementation

Use pinned CUTLASS example 87b groupwise SM120 E4M3×E4M3 with FP32 accumulation:
one activation scale per token/K128 block, one weight scale per N128/K128
block. Start with its cooperative 128×128×128 schedule. Generate weight codes,
scales and physical layout directly from the pinned BF16 source in the compiler;
upload once, with one resident view. Version quantizer/layout/policy explicitly.

Freeze this first recipe before scoring: finite absmax/448 per declared block
as FP32 reconstruction scale, FP32 divide then E4M3 RNE/saturation; an all-zero
block uses scale 1 and zero codes. Positive scale must remain representable
(clamp underflow to the smallest normal FP32); reject nonfinite weights and
propagate/reject nonfinite activations under the existing failure semantics.
Scale layout/padding must match CUTLASS ScaleConfig. No per-chunk amax, fitting
to evaluation answers or conversion of current Q8 bytes into a new quantizer.

First integrate GDN QKV/Z together, since they share the normalized input.
After that short check, apply the same path to large GDN output and attention
QG/K/V/output matrices. Keep tiny GDN A/B, head, embeddings, small parameters
and TASK-034's MLP selection. A demonstrated failing family retains its prior
format by an explicit policy exception; this is not an invitation to sweep
every format. Use same-view FP8/BF16-activation GEMV initially at M=1; compare
native small-M only if conversion or scalar arithmetic dominates that consumer.

RMS emits FP8 codes/scales once and a BF16 companion only for remaining BF16
consumers. QG/K/V reuse one pack; QKV/Z reuse another, while A/B consume the
same producer's BF16 companion. Retain existing BF16 output rounding before
the first FP8 conversion to isolate the new quantization error, implemented
locally without a global BF16 staging tensor where practical. Attention gating
and GDN gated norm produce fresh output-projection operands; fuse packing
into their producers. Avoid unconsumed FP8 packing on M=1 fallback paths.

Use explicit plan-owned spans until the last sibling completes. The consumer
accepts prepacked codes/scales and does not quantize again internally. Reuse
applies across N tiles too; different normalization weights, residual versions,
token ranges or scale recipes are different operands. No persistent activation
cache or FP8→BF16→FP8 round trip for compatible consumers. Nonlinear/reduction
math remains FP32; KV/history and recurrence retain their current precision.

Likely files: `src/compiler/compile.cpp`, `src/format/{constants,layout,pack,unpack,schema}.*`,
`cuda/{activation,prefill,decode_mmv,gdn,attention}.*`, and runtime binders in
`src/runtime/{model,prefill,prefill_gdn,prefill_attention,gdn,attention}.*`.
Use the existing artifact and CUDA boundary patterns, without a new framework.

## Checks and completion

One independent format/contraction check covers zero/tail/outlier/nonfinite
blocks, scale indexing, rounding, M=1/M=256, and identical row packing alone
or beside different rows. One fan-out check proves one pack per producer,
unchanged codes/scales across siblings, correct scratch reuse after the last
consumer and no stale pack after a changed residual. Use existing launch
instrumentation or a focused check; no permanent telemetry service.

Measure one real GDN shared-input projection group including RMS/pack and all
consumers. Check affected attention/output families numerically without a full
per-family timing matrix. Run the predeclared precision development subset and
one FAST-01 short integrated request, including prefill/decode handoff. A speed
win that fails numerical/development quality checks is rejected; isolated GEMM
speed does not establish request improvement. Bound FP8 and companion workspace,
scales/padding and transient load memory from actual layouts.

If groupwise FP8 cannot pass the relevant quality or complete-cost check,
retain affected Q8 families and the TASK-034 compact path; do not silently
weaken scale granularity or quality gates. Record why FP8 was rejected and
which repeated conversion remains. One 256-versus-512 prompt-only chunk check
is allowed after integration only if remaining traffic/launch cost justifies
it and all enlarged workspace/state-boundary checks fit; otherwise keep 256.

Complete with the selected integrated policy, both phase consumers, proven
operand ownership and short numerical/development results. TASK-036 owns the
coherent candidate's final quality, replay, capacity and performance decision.

## Completion Report

Selected `fp8-mixer-v1` for 208 large mixer matrices: direct BF16-source
compiler quantization to E4M3 with N128/K128 scales, CUTLASS 098de2a6 example
87b cooperative 128x128x128 for M>1, and same-view FP8 weights with BF16
activation for M=1. RMS and gated-output producers retain local BF16 rounding
before packing. GDN QKV/Z and attention QG/K/V share explicit plan-owned code
and scale spans; GDN A/B use the same producer's BF16 companion. Attention
output uses a fresh producer pack. Q4_K MLP bounded-unpack/cuBLAS, Q8 head,
and small/BF16 families remain selected. No 512-token experiment was run.

The independent format and fan-out check passed zero/tail/outlier/nonfinite,
scale indexing, rounding, M=1/M=256, row independence, one-pack sibling reuse,
scratch reuse, and changed-residual cases. Attention binding rejects both
mixed FP8/BF16 input-output combinations before mutation; positive FP8 and both
negative controls passed. Astra `gpt-6-astra` high review passed on pass 2/2;
R035-1 and R035-2 were resolved, with no remaining findings or evidence
requests. The reviewed implementation/evidence identities (97 entries) were
confirmed unchanged.

Implementation: main-thread GPT-6 Codex. Documentation and bookkeeping: GPT-6
Luna. `FOLLOW_UP_REQUIRED`: TASK-036 owns C92 free-generation uncertainty and
the final quality, replay, capacity, and performance decisions.

Validation used the recorded scripts under
`.cache/evaluation/qw38-language-v2/task035-support/`:

- `integration-build.sh`: pinned CUDA 13.4.1 Release targets built; CTest
  `^(fp8|format_schema|format_reader|decode_mmv_unit|prefill_projection|attention_prefill|gdn_prefill)$`
  passed 7/7 in 3.20 s.
- `compile-screen.sh`: `qw38-compile --checkpoint
  .cache/authorities/qwen3.8-27b-transformers --output
  .cache/candidates/candidate-fp8-mixer-v1.qw38 --format fp8-mixer` passed
  (882612 ms); `language_model_integration` passed in 33.66 s; the frozen
  256-token prompt plus 8-token request and all eight frozen development
  windows completed. `finish-checks.sh` corrected a scorer identity assertion
  (the Q8 head and alias produce two records); no workload was rerun for that
  correction. Its development score passed: NLL 1.7767602373244544 versus
  comparator 1.773838532533603, delta 0.0029217047908514093 against the
  predeclared 0.03 limit. Final FP8 CTest passed in 3.87 s.
- `final-group.sh`: separate-process first-use runs, zero warmups and one
  sample each, for integrated GDN layer-0 RMS/pack plus QKV/Z and A/B at M=1
  and M=256. After the review repair, `review-repairs.sh` supplied the
  corrected production-dispatch M=1 result: Q8 GPU/host 0.738048/0.745806 ms,
  FP8 0.717088/0.726020 ms. The unchanged M=256 result is Q8
  45.046/45.0552 ms and FP8 39.6792/39.6887 ms. These are short group
  diagnostics, not a request-level speedup or promotion claim.
- `review-repairs.sh`: `fp8`, `fp8_attention_binding`, and
  `attention_prefill` passed 3/3 in 14.39 s, including both mixed-format
  rejection cases. `git diff --check` passed.

The integrated request recorded 255.401 ms prompt ingestion and 194.608 ms
for seven decode steps (27.8011 ms/step). Its generated continuation differs
from TASK-034, so it does not support a matched request-speedup claim. Actual
FP8 artifact accounting: model 21,013,686,400 bytes; FP8 codes 7,214,202,880;
scales 1,761,280; largest direct upload 2,542,796,800 bytes; no second device
load copy; compiler peak RSS 4,033,146,880 bytes. M=256 engine workspace is
83,410,944 bytes, 14,204,928 above control. The projected 32K reserve is
8,717,557,760 bytes and is not a capacity result. Hardware was NVIDIA GeForce
RTX 5090, driver 590.48.01, 32607 MiB; pinned container image SHA-256
`3844dc9c37087cecd40f96e62bd4f305ad405408f0d63312bda8aff8651f2b49`.

Evidence logs, exact command scripts and binary identities are preserved in
`.cache/evaluation/qw38-language-v2/task035-support/`, including
`integration-binaries.sha256`, `final-group-binary.sha256`, and
`review-repairs-binaries.sha256`. TASK-036 owns final quality, replay,
capacity, and performance decisions; localized free-generation uncertainty
remains open there.

Key identities: artifact manifest `7fd0f81bfb7a4e6bd5ebaacda532d2992fde64ce136fef55116e7e4f292d10c7`; `build/pinned-release/src/qw38-compile` `d21482b2bbce63fdbab46c3f98cdab4742e470f0ee03c438c1eeb2067b9eae23`; `build/pinned-release/src/qw38-decode` `b087b870eaa90d4ab82d4413d28caf314c50fe2d8fc9e4d1676911bc57262936`; `build/pinned-release/src/qw38-evaluate` `e1a2b08f68abd95272baefe384e99f91f39058d7b4014ad2a2cab8883b5d7f41`; `build/pinned-release/tests/qw38_fp8_test` `d5d64f5645df99d39b34c9d3eba407c18b433f9dc083b65e8c400f73c0cfaa5c`; `build/pinned-release/tests/qw38_language_model_integration_test` `4f2ccba8869339474e6cf7c3805d677bd511f8a124558bebc57b99dff7c4962a`; `build/pinned-release/benchmarks/qw38_bench_fp8_group` pre-repair M=256 `e8d20d032661cc7791f748f7047cbaa9be1cc53f9ffc605fb00d2b49f09d5cfc`; `build/pinned-release/tests/qw38_fp8_attention_binding_test` `72e7276686d6093252be19242ee8898d19491c9ba8eed8f66775e737fa243d54`; `build/pinned-release/benchmarks/qw38_bench_fp8_group` repaired M=1 `999a6173dcf410b99d211475837380f7516c7dbc3b06cb7e3aa514aadeacd40e`.

The GPU/compiler commands ran inside the pinned container image above, with
GPU access and repository mounted at `/workspace`. The second bind mount
`-v "$PWD:$PWD"` preserved absolute development-case paths. The scorer ran
on the host from the repository root. Exact acceptance invocations were:

```sh
out=.cache/evaluation/qw38-language-v2/task035-support
cmake --build build/pinned-release --target qw38_fp8_test qw38_compile qw38_decode qw38_evaluate qw38_format_schema_test qw38_format_reader_test qw38_decode_mmv_unit_test qw38_prefill_projection_test qw38_language_model_integration_test qw38_attention_prefill_test qw38_gdn_prefill_test -j4
ctest --test-dir build/pinned-release --output-on-failure -V -R '^(fp8|format_schema|format_reader|decode_mmv_unit|prefill_projection|attention_prefill|gdn_prefill)$'
build/pinned-release/src/qw38-compile --checkpoint .cache/authorities/qwen3.8-27b-transformers --output .cache/candidates/candidate-fp8-mixer-v1.qw38 --format fp8-mixer
QW38_AUTHORITY_ARTIFACT=/workspace/.cache/candidates/candidate-fp8-mixer-v1.qw38 ctest --test-dir build/pinned-release --output-on-failure -V -R '^language_model_integration$'
build/pinned-release/src/qw38-decode --artifact .cache/candidates/candidate-fp8-mixer-v1.qw38 --tokens-file .cache/evaluation/qw38-language-v2/task034-support/prompt-256.u32le --generate 8 --profile
build/pinned-release/src/qw38-evaluate --artifact .cache/candidates/candidate-fp8-mixer-v1.qw38 --cases .cache/q4k-candidate/development/cases.tsv --output "$out/development" --eos-ids 248044,248046,248063,248064,248065
# Host, from repository root:
uv run --script "$out/score.py"

# Pinned container:
QW38_AUTHORITY_ARTIFACT=/workspace/.cache/candidates/candidate-fp8-mixer-v1.qw38 ctest --test-dir build/pinned-release --output-on-failure -V -R '^fp8$'
cmake --build build/pinned-release --target qw38_fp8_attention_binding_test qw38_fp8_test qw38_attention_prefill_test qw38_bench_fp8_group -j4
QW38_AUTHORITY_ARTIFACT=/workspace/.cache/candidates/candidate-fp8-mixer-v1.qw38 ctest --test-dir build/pinned-release --output-on-failure -V -R '^(fp8|fp8_attention_binding|attention_prefill)$'
build/pinned-release/benchmarks/qw38_bench_fp8_group .cache/authorities/qwen3.8-27b-transformers .cache/candidates/candidate-v2-q4k-rope-fixed.qw38 .cache/evaluation/qw38-language-v2/task034-support/prompt-256.u32le q8 1 "$out/repaired-q8-1.bin"
build/pinned-release/benchmarks/qw38_bench_fp8_group .cache/authorities/qwen3.8-27b-transformers .cache/candidates/candidate-v2-q4k-rope-fixed.qw38 .cache/evaluation/qw38-language-v2/task034-support/prompt-256.u32le fp8 1 "$out/repaired-fp8-1.bin"
build/pinned-release/benchmarks/qw38_bench_fp8_group .cache/authorities/qwen3.8-27b-transformers .cache/candidates/candidate-v2-q4k-rope-fixed.qw38 .cache/evaluation/qw38-language-v2/task034-support/prompt-256.u32le q8 256 "$out/final-q8-256.bin"
build/pinned-release/benchmarks/qw38_bench_fp8_group .cache/authorities/qwen3.8-27b-transformers .cache/candidates/candidate-v2-q4k-rope-fixed.qw38 .cache/evaluation/qw38-language-v2/task034-support/prompt-256.u32le fp8 256 "$out/final-fp8-256.bin"
```

These commands were wrapped by the recorded scripts to capture logs. FP8 needs
no matrix padding for the actual
dimensions. M=256 extra workspace is 14,204,928 bytes: packed codes 1,572,864 plus scales 49,152 plus FP32 accumulation 12,582,912; the existing GDN BF16
companion is 2,621,440 bytes. Projected 32K reserve 8,717,557,760 bytes uses
TASK-030 model basis 21,462,812,800 bytes and TASK-034 reserve
8,282,636,288 bytes, plus saved weights less extra workspace. This is a
projection only; TASK-036 must measure capacity.
