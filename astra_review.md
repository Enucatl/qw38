# First decode implementation review

Date: 2026-09-23. Scope: source review and the preserved TASK-018 P11 speed
diagnostic. The full TASK-018 quality result and a matched performance benchmark
are still pending.

## Recommendation

Pause advancement to TASK-019 for a bounded decode stabilization checkpoint.
Keep TASK-018 open. Establish a fast development loop and measure the current
decode path before changing its quantization or physical layout. Fix execution
bugs within the current architecture, then run the unchanged full quality gate
at its milestone. Record any proposed change to a locked architecture decision
through the ledger's amendment procedure.

## Quantization and numerical path

| Family | V0 storage and execution |
| --- | --- |
| Large mixer and MLP projections | Symmetric Q4G64 weights: signed INT4 codes in groups of 64 with FP16 group scales. |
| `lm_head` | Symmetric Q8G32 weights: signed INT8 codes in groups of 32 with FP16 group scales. |
| Embedding, normalization, convolution, and small GDN parameters | BF16 storage. |
| Projection inputs, ordinary staging, KV cache, and convolution history | BF16. |
| Residual, dot accumulations, reductions, and GDN recurrent state | FP32. |

The quantizer uses deterministic group absmax scales. Decode reconstructs each
code and rounds the result to BF16, then performs FP32 FMA accumulation. This
is compressed weight storage, not an INT4 tensor-core decode path. No activation
quantization or outlier sidecars are present. The P11 artifact reports 408 Q4,
two Q8, 458 BF16, and one FP32 tensor; those are artifact tensor counts, not
projection or kernel counts. See `src/compiler/quantization/quantizer.cpp`,
`cuda/decode_mmv.cu`, and `.cache/task018-p11-long/v0/candidate_identity.json`.

## Current token schedule

One token runs embedding gather, 64 mixer-plus-MLP layers, final RMS, Q8 head,
full 248,320-logit device-to-host copy, and CPU argmax. The 64 layers repeat
three GDN layers then one full-attention layer, 16 times. Weights, KV, GDN
state, and scratch stay on the GPU. One CUDA stream orders the operations. The
runtime already combines paired gate/up MLP projections, grouped mixer
projections, and selected projection epilogues.

Source inspection implies roughly 675 kernel launches and 65 explicit stream
synchronizations per token: 48 GDN mixer synchronizations, 16 attention prep
synchronizations, and one final logits synchronization. The GDN path currently
rejects CUDA graph capture. Every prompt token executes this full decode path,
including the vocabulary head and CPU readout. The quality evaluator ingests
each teacher-scored prompt a second time after generation. These counts are
derived from source; a profiler must determine their actual latency share.
Relevant code: `src/runtime/language_model.cpp`, `src/runtime/gdn.cpp`,
`src/runtime/attention.cpp`, `src/runtime/mlp.cpp`, and `src/runtime/evaluate.cpp`.

The P11 diagnostic (`.cache/task018-p11-long/report.md`) ran a Release evaluator
on the RTX 5090. Both engines generated the full 2,048 tokens for
`aime2025-16`. V0's observed completion interval was 71.6 seconds and
llama.cpp's request interval was 32.2 seconds. Timing boundaries differ. The
result is evidence of a gap worth investigating, not a calibrated speed ratio.
The present repeated-decode prompt path adds request cost; it does not explain
the entire long-generation gap by itself.

## Test and measurement loop

| Check | Proposed budget | Purpose |
| --- | --- | --- |
| Focused correctness | Under 30 seconds | Changed component, numerical and scorer checks. Use focused CTest or pytest selection. |
| Full-model smoke | Around 1–2 minutes including model load | Bounded fixed prompts, eight generated tokens, plus the existing full-model integration test for reset and replay. Run `scripts/run_task018_dev_smoke.sh`; it writes a partial diagnostic record. |
| Performance diagnostic | A few minutes | Fixed token streams and populated context, with identical timing boundaries. Measure host time, GPU time, launch and synchronization overhead, and kernel families separately. |
| Full quality acceptance | Explicit milestone run | All 216 frozen cases at original caps, paired scoring, retrieval, state replay, and P100 adjudication. |

The smoke took 25.9 seconds on the current RTX 5090 host, including container
startup and model load, and completed both cases at their eight-token caps.
The current full-model plan and integration CTests passed in 11.9 seconds in
the pinned container with the authority artifact set. The performance and full
acceptance budgets remain unverified. The quick smoke detects broken execution;
an eight-token AIME response is not a quality score. Existing CTest supports
`-LE extended`. The full evaluation must remain separate from the routine
development loop.

Run the short evaluator check from the repository root with
`scripts/run_task018_dev_smoke.sh`. Run the existing full-model reset/replay
check with:

```bash
docker run --rm --gpus all -v "$PWD:/workspace" -w /workspace \
  -e QW38_AUTHORITY_ARTIFACT=/workspace/build/pinned-debug/qwen-v0.qw38 \
  qw38-dev:cuda13.4.1-pinned \
  ctest --test-dir build/pinned-debug --output-on-failure \
  -R '^(language_model_plan|language_model_integration)$'
```

Immediate profiling should isolate model load, prompt ingestion, and
populated-state decode. For decode, first quantify stream sync, kernel launch,
projection, recurrent, attention, and vocabulary readout time. Change one
measured cost at a time and rerun affected correctness checks. The full
TASK-018 acceptance remains the gate for advancing the ledger. A quality
failure in a selected quantization hypothesis requires explicit architecture
review; raw speed evidence alone does not identify that hypothesis as wrong.
