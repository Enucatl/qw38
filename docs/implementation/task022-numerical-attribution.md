# TASK-022 numerical attribution after the Q4_K screen

The eight frozen development windows score 1.773838533 nats/token for the
external llama.cpp GGUF comparator and 1.913282631 for the Quartz Q4_K
candidate. Replacing all 192 Quartz MLP matrices improved its score by only
0.005765203 nats/token, leaving a +0.139444098 gap. This is a diagnostic
investigation of that gap, not a new TASK-022 acceptance run.

## The 353 F32 GGUF controls

The BF16-derived GGUF source, the selected Q8_0/Q4_K proxy, and the external
Q4_K_M comparator contain the same 353 F32 control tensor names and identical
payload bytes: 2,645,504 values in total. The conversion log shows that these
controls were read from the available BF16 checkpoint and emitted as F32.
595,183 F32 values have nonzero low 16 bits, so a blanket F32-to-BF16 cast of
the **converted** GGUF would change them. The converter applies operations such
as adding one to normalization weights and negating/exponentiating `A_log`.
Quartz retains the original BF16 controls and performs the analogous operations
in FP32 in its activation and GDN kernels. The exact payload audit is
`scripts/task022_audit_f32_controls.py` and
`.cache/task022/f32-control-audit.json`; BF16-to-GGUF conversion is logged in
`.codex-wake-run/429b1d9281b0.log`.

Therefore, the external GGUF's F32 controls do not contain an otherwise absent
FP32 checkpoint: they came from the BF16 source already on disk. A roughly
100 GB FP32 snapshot is unnecessary to compare these control values. F32
storage can still change **execution** relative to Quartz's BF16 intermediate
rounding, especially in normalization and convolution; payload identity alone
does not rule that out.

## First GDN block, matched inputs

On the selected proxy's own saved inputs, representative layer-0 QKV and GDN
output projections differ between GGUF Q8_0 and the Quartz Q8G32/BF16 operand
recipe by relative L2 of about 0.0009–0.0013. This uses 12 sampled prefill
rows and one decode row per projection. Details are in
`.cache/task022/gdn-projection-paths-proxy-inputs.json`.

An independent 385-token replay of layer 0 on the GGUF input stream reproduces
the pinned llama.cpp GDN gated-output trace within relative L2 0.0011–0.0013
at positions 0, 63, 255, 383, and 384. Switching the replay to Quartz's
Q8G32/BF16 projection and store recipe changes gated output by 0.0015–0.0030
and recurrent state by 0.0032–0.0037 relative L2 at those positions. The
replay is a numerical model on GGUF logical head layout and common saved
inputs, not a capture of Quartz's actual GDN state. It supports a small local
GDN recipe difference at layer 0; it cannot exclude effects in later GDN
layers or interactions with different incoming residuals. Details are in
`.cache/task022/gdn-state-replay.json`.

## Actual runtime residuals: first sharp jump at full attention

The pinned llama.cpp proxy and actual Quartz Q4_K runtime processed the same
first frozen 384-token prompt and token 384. We captured the 5,120-element
FP32 residual after every layer, and after full-attention mixers at layers 3,
7, 31, and 63. Relative L2 below is `||Quartz − proxy||₂ / ||proxy||₂`.

| Token position | After layer 2 | Layer 3 attention residual | After layer 3 MLP | After layer 63 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0071 | 0.0071 | 0.0072 | 0.0428 |
| 63 | 0.0108 | 0.0436 | 0.0505 | 0.1691 |
| 255 | 0.0040 | 0.0387 | 0.0404 | 0.2033 |
| 383 | 0.0041 | 0.0247 | 0.0256 | 0.2265 |
| 384 | 0.0043 | 0.0264 | 0.0282 | 0.1718 |

Layer 3 is the first full-attention layer. The first large increase on tokens
with populated context is already present **before** its MLP. For example, at
position 63 the layer-2 residual differs by 1.08%, the layer-3 attention
residual by 4.36%, and the post-MLP residual by 5.05%. The layer-3 attention
branch itself differs by 11.8% relative to the proxy branch vector at position
63, 12.9% at 255, and 8.1% at 383. At position 0, its branch difference is
only 1.1%. Later full-attention mixers and downstream layers carry larger
differences; the trace does not show a single isolated final-layer cause.

This locates the first sharp effective-model difference. It does not identify
which attention operation causes it: the two arms use different Q8 weight
formats, activation rounding, KV formats, and attention implementations, and
their layer-3 inputs already differ slightly. The one-window activation trace
cannot by itself assign the 0.13944 nats/token development gap to a single
operation or predict how a precision change would affect NLL.

Evidence: `.cache/task022/residual-trace-comparison.json` contains all 320
layer/position residual comparisons and 20 mixer comparisons. The actual
Quartz trace was produced by `src/runtime/task022_trace.cpp`; the pinned
llama.cpp trace by `scripts/task022_trace_llama.cpp`; and
`scripts/run_task022_residual_traces.sh` runs both and compares them. The
successful run is `.codex-wake-run/a37a408bd91b.log`.

## Layer 3 attribution: incorrect frozen RoPE frequencies

The next trace captured all 385 layer-2 FP32 residuals and layer-3 attention
buffers, plus the pinned proxy's FP32 graph nodes. Before RoPE, Quartz and the
proxy differ moderately: at position 63, their normalized inputs differ by
relative L2 0.0104 and Q/g projections by 0.0068. After RoPE, Q differs by
1.0434; the gated attention value differs by 0.4591. A standard NeoX rotation
of the proxy's own Q-normalized node reproduces its post-RoPE Q within
4.3e-6 relative L2 at the five sampled positions. An FP32 softmax replay using
the proxy's Q, rotated K and F16 K/V cache recipe reproduces its traced
pre-gate attention value within 0.00015–0.00055. Replaying Quartz's saved
Q/BF16 KV/gate reproduces its actual gated value within 3.2e-5. The arithmetic
implementations were therefore not needed to explain the large jump.

Quartz's compiler had frozen the wrong 32 FP32 values in `rope.inv_freq`.
The model requires `10,000,000^(-2j/64)`. At `j=1` the old payload was
0.869341314, whereas the required value is 0.604296390. Inferred rotation
phase errors from Quartz's captured Q and K agree with the old table at each
populated position. The compiler table now contains independently rounded
values for the documented formula, and compiler revision patch 2 distinguishes
the corrected artifact from the old patch-1 artifact. The compiler transform
test checks both the frozen bits and the formula; its authority integration
test verifies reconstruction with the current revision.

The corrected Q4_K artifact was compiled from the same BF16 checkpoint with
source reconstruction verification. Its manifest digest is
`41c1f5e673bb24eb2fb283aa6044dbccdebecc7cd85f847815b3c02a6763fc43`;
its compiler identity is
`qw38-candidate-v2-q4k-llama-e6ab7c1a4-no-imatrix-bf16-operands:0.1.2`.
The previous artifact and eight-window score remain preserved. On the same
frozen prompt, the old and corrected Quartz traces are **byte-identical** at
all 385 positions for the incoming residual, normalized input, Q/g/K/V
projections, gate, and V cache. RoPE is the isolated changed factor at this
layer. Relative L2 against the pinned proxy is:

| Position | Q, old → corrected | Gated value, old → corrected | Attention branch, old → corrected | Attention residual, old → corrected |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.0259 → 0.0259 | 0.0315 → 0.0315 | 0.0106 → 0.0106 | 0.0071 → 0.0071 |
| 63 | 1.0434 → 0.0177 | 0.4591 → 0.0172 | 0.1179 → 0.0086 | 0.0436 → 0.0079 |
| 255 | 1.0446 → 0.0292 | 0.5964 → 0.0284 | 0.1291 → 0.0073 | 0.0387 → 0.0034 |
| 383 | 0.9463 → 0.0291 | 0.3219 → 0.0234 | 0.0809 → 0.0077 | 0.0247 → 0.0043 |
| 384 | 0.9733 → 0.0361 | 0.3805 → 0.0272 | 0.0852 → 0.0060 | 0.0264 → 0.0033 |

On the unchanged eight-window, 1,024-target development screen, NLL fell
from 1.913282631 to **1.775325938** nats/token. The external llama.cpp
comparator was 1.773838533, so the remaining difference is +0.001487405
nats/token, inside the previously frozen +0.03 development promotion bound.
The corrected artifact uses the same candidate precision policy and decode
dispatch; the eight case IDs, prompt and target hashes, and target count were
verified before scoring. This is strong evidence that the wrong RoPE payload
caused almost all of the development NLL gap, but it is not the complete
language-v2 EVAL-01 gate. TASK-022 remains blocked until that gate and its
other required evidence pass.

Evidence: `scripts/task022_compare_layer3_traces.py` and
`.cache/task022/layer3-boundaries{,-corrected}.json` validate runtime and
proxy nodes; `scripts/task022_diagnose_layer3_rope.py` and
`.cache/task022/layer3-rope-diagnosis.json` infer Q/K phase; and
`scripts/task022_compare_rope_fix.py` with
`.cache/task022/rope-fixed-trace-comparison.json` checks exact pre-RoPE
identity. `scripts/run_task022_rope_fix_checks.sh` passed three focused
CTests in `.codex-wake-run/f3db8b966cc1.log`.
`scripts/run_task022_rope_fix_screen.sh` produced the reconstruction-verified
artifact and unchanged-screen report in `.codex-wake-run/4063eeee9ecc.log`
and `.cache/task022/rope-fixed-development-report.json`; compiler/evaluator
SHA-256 values are in `.cache/task022/rope-fixed-binaries.sha256`. The
identity-tightened scorer reproduced the report in
`.codex-wake-run/6afe52a6c385.log`.
`scripts/run_task022_rope_fixed_trace.sh` completed in
`.codex-wake-run/e8f615c9d1fe.log`.
