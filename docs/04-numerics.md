# 4. Scalar reference and tensor oracles

[Previous](03-deepseek-v4.md) · [Index](README.md) · [Next](05-gpu-implementation.md)

## Why this matters

Optimization needs a trusted intermediate-tensor oracle. End-token equality is
too late to locate a bad layout, recurrence update, or rounding decision.

## Design and dataflow

An **oracle** is a slower implementation trusted enough to tell us what the
optimized implementation should produce. “Scalar” means that the reference
uses straightforward loops and explicit indexing rather than relying on a
specialized matrix kernel. It is not intended for serving; it is intended to
make one wrong transpose or update order obvious.

Implement `BackendOps` first as simple, typed host operations. It may be slow,
but must expose named taps. A tap is a copy of an intermediate tensor at a
named boundary, saved before the next operation overwrites its buffer:

```text
SequenceInput -> embed -> for layer 0..63:
  input norm -> mixer projections -> mixer state/output -> residual
  post norm -> gate/up -> product -> down -> residual
-> final norm -> lm_head -> F32 logits
```

The arrows describe data ownership as well as computation: embeddings and
activations are temporary, weights are read-only, and session state is the only
mutable input that survives a call. Keep `ModelSpec` independent of storage. It validates dimensions, the 64-entry
layer-kind schedule, exact tensor inventory, dtype, orientation, and quant
policy. `ModelWeights` owns immutable tensors; `SessionState` alone mutates.
Use FP32 accumulation for norms, softmax, GDN recurrence, and initial comparison.

The interfaces separate questions that are easy to conflate:

- `ModelSpec` answers “what model is this?” without owning tensor bytes.
- `ModelWeights` answers “where are the immutable parameters?”
- `SequenceInput` answers “which embeddings and positions enter this call?”
- `SessionState` answers “what prefix-dependent information survives?”
- `BackendOps` answers “how is each mathematical operation executed?”

This separation lets the same scalar forward pass read plain FP32 arrays while
the CUDA backend reads tiled quantized weights. Both must implement the same
logical operations and mutate the same conceptual state.

## Oracle workflow

### Why pin every input

Reference output is meaningful only if every semantic input is fixed. A changed
tokenizer can change IDs, a changed chat template can insert another control
token, and a library update can alter a numerical kernel. Record content hashes,
not only names such as `main` or `latest`.

1. Pin checkpoint revision, Transformers revision, llama.cpp revision,
   tokenizer files, template, prompt bytes, and reference dtype.
2. Add hooks to save shapes, dtypes, and little-endian raw tensors at embedding;
   GDN post-convolution, recurrent output, and projection; attention Q/K after
   norm/RoPE and output; FFN output; final norm; logits.
3. Feed the same IDs and positions to the scalar engine. Compare max absolute,
   max relative, RMS error, cosine similarity, and first failing index.
4. Compare Transformers first, then llama.cpp as an independent oracle. A
   disagreement between authorities is investigated, not averaged away.

### A concrete first-failure example

Suppose embedding output agrees, layer 0's normalized input agrees, but its
post-convolution Q tensor differs. The search is now small: projection binding,
packed slicing, convolution history/order, activation, or layout. There is no
reason yet to inspect attention, sampling, or the LM head. If every hidden tap
agrees but logits differ, inspect final norm and output-head orientation. This
is the practical value of intermediate taps.

Use one-token fixtures at positions 0–5 for convolution warm-up; prompts of
length 1, 4, 5, 63, 64, and 65; and taps around layers 3/4 and 63. Check chunked
prefill against token-by-token decode by comparing final logits and every state
byte or tolerance-defined state value.

For a numerical comparison, **absolute error** catches a fixed-size offset,
**relative error** catches scale-dependent drift, RMS error summarizes the whole
tensor, and cosine similarity catches a changed direction. None is sufficient
alone: a nearly zero reference value makes relative error unstable, while a
cosine score can hide one bad channel. Set tolerances per boundary and dtype
before looking at the candidate result.

For two scalar values, absolute error is `abs(candidate-reference)`. Relative
error divides that difference by a safe magnitude such as
`max(abs(reference), epsilon)`. For a tensor, report the maximum and the index
where it occurs, plus summary measures. Also check for NaN and infinity before
computing statistics; otherwise a broken tensor can produce misleading output.

### Determinism and greedy checks

Greedy decoding chooses the largest logit and removes random sampling from the
test. It is useful but not sufficient: two logit vectors may differ substantially
while retaining the same largest element. Conversely, a tiny difference between
the two highest logits can change the selected token even when all numerical
errors are within the planned dtype tolerance. Keep tensor comparisons and
token-level checks as separate gates.

## DwarfStar transfer boundary

DwarfStar's CPU-only diagnostic path, tensor validation before binding, quant
dot tests, and CPU/GPU boundary comparisons transfer unchanged. Its tensor
roles, compression fixtures, and permissive fallback assumptions do not.

## Concrete work and acceptance gate

The scalar milestone passes only when one-token and multi-token prompts match
both semantic authorities at every available boundary, greedy continuations are
stable, and a save/restore at each tested position produces the uninterrupted
result. Store fixtures with source hashes and generation commands.

## Strict reference versus optimized production

OPT-044 freezes two arithmetic roles before later kernel tuning.

The **strict reference** path keeps the historical kernels, `--fmad=false` /
`-ffp-contract=off`, and the original CUD-001/CUD-003, GDN, attention, graph,
chunk-equivalence, trace, and scalar-oracle contracts. Structural and session
checks stay exact: token identity, layout round-trips, graph-versus-eager
equality on the *same* arithmetic path, and checkpoint isolation are not
accuracy compromises.

**Optimized production** may later use Q8_1 activation quantization, reordered
FP32 reductions, explicit FMA, F16 MMA operands with FP32 accumulation, and
individually validated approximate transcendentals (`fmaf`, `rsqrtf`, `__expf`).
Those approximations are admitted only against
[`pins/production_numerics_contract.json`](../pins/production_numerics_contract.json).
Default family/shape abs and RMS ceilings are
`max(strict_reference_ceiling, 1.05 * measured_llama_error + 1e-6)` versus
independent FP64 dequantized arithmetic. Cosine uses `1-cosine` with the same
construction and a `1e-7` floor. Zero-vector pairs use explicit norm/abs checks.
If llama is nonfinite on a stress case, that shape stays on the strict path;
ceilings are not inflated from pathological output. Unrepresented shapes stay
strict until covered. Production dispatch is still the strict path: there is no
unvalidated fast kernel.

End-to-end production-optimization quality uses the existing WikiText/tokenizer
pins, a 1.01 perplexity ratio versus pinned llama (stricter than the legacy 1.05
QLT-001 gate), 0.02 recurrence NLL drift, functional answers, and a held-out
1024-target span disjoint from calibration. Legacy QLT-001 verdicts stay
alongside the new suite.

Evidence: [`fixtures/opt044_production_numerics.json`](../fixtures/opt044_production_numerics.json)
and [`evidence/optimization/opt044-production-numerics/REPORT.md`](../evidence/optimization/opt044-production-numerics/REPORT.md).
This increment **claims no performance improvement**.

OPT-059 adds a measured **v2** policy
([`pins/production_numerics_v2_contract.json`](../pins/production_numerics_v2_contract.json))
from actual llama GPU error when that export exists. Ceilings use 1.25 headroom
rather than v1's 1.05. Quartz Q8_1 stores `half(sum(integer quants))`; pinned
llama GPU stores `half(sum(original x))`. The live engine summary is **mixed**
because selected Q8/Q6 pins already use approximations; unrepresented shapes stay
on the strict fallback. Missing GPU coverage stays unadmitted. OPT-046 is not
installed. Evidence:
[`evidence/optimization/opt059-gpu-numerics/REPORT.md`](../evidence/optimization/opt059-gpu-numerics/REPORT.md).
OPT-059 v2 and OPT-074 production GPU admission are **historical** diagnostics.
They are not the active kernel-admission authority.

## Kernel parity admission (OPT-081)

**Proposed**. Active kernel admission is `kernel_parity_v1`
([`pins/kernel_parity_v1_contract.json`](../pins/kernel_parity_v1_contract.json)):
a GPU candidate versus independent CPU/dequant of the **same** quantized
weights and staged/input activations. CUD-001 `3e-4` remains a historical
MMV pin, not this envelope. The active hierarchy is structural → kernel parity
→ same-math → model quality → performance → release. Kernel
admission proves implementation of the intended quantized operation. It does
not prove full-model quality and does not depend on llama GPU versus FP64
consistency. OPT-074 unadmitted rows are not active keep blockers. Evidence:
[`evidence/optimization/opt081-kernel-parity-policy/REPORT.md`](../evidence/optimization/opt081-kernel-parity-policy/REPORT.md).


## Common failures

- Comparing logits only, or loosening tolerance until a structural bug passes.
- Transposing checkpoint matrices twice.
- Applying the convolution to unpacked Q/K/V independently.
- Updating session state while producing diagnostic retries.
- Allowing a GPU fallback into the reference path.

## Exercise and expected result

Implement one GDN head for five tokens with a four-entry convolution ring.
Compare streaming and whole-sequence execution. Expected: every output and the
final recurrent/ring state agree; corrupting the oldest warm-up row first causes
a difference at the predictable convolution boundary.
