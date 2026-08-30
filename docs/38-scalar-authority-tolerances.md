# 38. Three-authority alignment and frozen tolerances

[Index](README.md) · Implementation tasks: ORA-004, ORA-001, and EDU-024 in
[`implementation_ledger.md`](../implementation_ledger.md) · Frozen gates:
[`scalar_oracle_tolerances.json`](../pins/scalar_oracle_tolerances.json) ·
Evidence: [`scalar_authority_alignment.json`](../fixtures/scalar_authority_alignment.json)

The previous two chapters established three different witnesses:

- official BF16 weights executed by pinned Transformers tell us the published
  model's intended semantics;
- the same Q4_K_M GGUF executed by pinned llama.cpp gives an independent check
  of the compressed artifact;
- Quartz's readable scalar CPU path is the implementation CUDA must preserve.

ORA-004 joins those witnesses at named intermediate tensors. This is more
diagnostic than comparing only the final token: if a result diverges, the first
bad boundary tells us whether to inspect a projection, layout conversion,
recurrence, attention, FFN, residual, or output head.

## Comparable boundaries

Two tensors are **comparable** when they represent the same mathematical value
at the same position and layer. Their array shapes and element order must also
be made identical before numeric metrics are calculated.

The calibrated history is raw tokens `[42, 3649]`, positions 0 and 1. It checks
early and late GDN layers 0 and 62, and attention layers 3, 7, and 63. Stable
boundaries include embedding, normalization, projections, GDN convolution and
recurrence, attention Q/K/RoPE/KV/context, FFN stages, both residuals, final
normalization, and all 248,320 logits.

A tap identity is not just a short name. It includes participant, position,
layer, semantic boundary, shape, and artifact/source identity. This mattered in
practice: llama.cpp emitted the name `Vcur-3` once as a flat 1,024-value
projection and again as shape `[256,4]`. The comparison selects by name **and**
shape, so it cannot silently choose the wrong tensor.

## Runtime-private boundaries

A **runtime-private boundary** is a useful internal array that another runtime
does not expose at the same semantic moment. Examples include a fused temporary,
a padded convolution return before the model slices it, or Transformers'
`SiLU(gate)` before multiplication when Quartz stores only
`SiLU(gate) × up`.

Runtime-private tensors can remain in diagnostic evidence, but they are never
compared to a merely similar array. [`transformers_taps.py`](../tools/transformers_taps.py)
therefore retains the full padded convolution result and separately records its
current-position slice. It also captures the actual input to the FFN down
projection and the gated attention context. This adds observation points; it
does not replace upstream equations.

llama.cpp's public evaluation callback does not expose every persistent state
write as a named graph node. Its same-GGUF comparisons cover 78 stable tap
identities. The official/Quartz comparison covers all 97 selected identities,
including full recurrent and convolution states. Missing independent visibility
is stated explicitly rather than filled with a self-comparison.

## Why layout normalization is not a tolerance

The same logical values may be stored in a different order. Comparing them
before applying a known permutation produces dramatic but meaningless errors.
A **layout normalization** moves elements to a shared order exactly; a numeric
**tolerance** allows small value differences after that ordering is correct.
They solve different problems.

Qwen's GDN has 16 Q/K heads and 48 value heads, or three value replicas per key
head. The original checkpoint groups the three replicas beside their key head:

```text
grouped: key0/replica0, key0/replica1, key0/replica2, key1/replica0, ...
```

The pinned GGUF converter tiles replicas first:

```text
tiled: replica0/key0, replica0/key1, ..., replica1/key0, ...
```

[`gdn_permute`](../tools/compare_scalar_authorities.py) implements that exact
16-by-3 mapping. It is tested by round-tripping every element. Q and K are not
permuted. Value-associated projection channels, convolution channels, and
convolution-state channels are normalized where required. The semantic
recurrence uses grouped order.

The audit also corrected a diagnostic label: Quartz's convolution state was
already stored channel-major as `[10240,4]`, but the trace manifest had called
the unchanged bytes `[4,10240]`. Correcting the shape changed neither bytes nor
arithmetic.

Two other bad first mappings were rejected rather than tolerated. Quartz
`attention.query` is the raw query half split from each projection head, so it
maps to raw upstream Q, not normalized Q. Quartz `attention.context` has already
been multiplied by the sigmoid output gate, so it maps to post-gate context,
not pre-gate attention output. After those fixes, the worst same-GGUF cosine
rose from an obviously wrong near-zero value to 0.999568.

## The numeric measurements

Each comparison records several views of error because one number can hide a
different failure mode.

### Absolute error

For one element, **absolute error** is:

```text
abs(expected - actual)
```

It is in the tensor's own units. Maximum absolute error finds the largest
single discrepancy and records its index. It can be dominated by one outlier.

### Relative error

**Relative error** divides absolute error by the expected magnitude (with a
small denominator floor). It answers “large compared with what we expected?”
It is useful in reports but unstable near expected zero: a tiny harmless
absolute difference divided by a tiny number can look enormous. For that reason
the maximum relative error is retained but is not an admission gate here.

### RMS error

**RMS error** squares every difference, averages the squares, and takes the
square root:

```text
sqrt(mean((expected - actual)²))
```

It measures typical error while still weighting large discrepancies. Requiring
both maximum absolute and RMS bounds prevents either a single spike or broad
low-level drift from hiding.

### Cosine similarity

**Cosine similarity** compares the direction of two complete vectors:

```text
dot(expected, actual) / (length(expected) × length(actual))
```

One means the vectors point in the same direction. It is insensitive to a
uniform scale change, so it complements rather than replaces absolute and RMS
checks. Every admitted sample must satisfy all three gates and contain no NaN
or infinity.

**Measured:** after structural diagnosis, 156 same-GGUF llama.cpp/Quartz rows
have minimum cosine 0.999568, maximum RMS 0.143929, and maximum absolute error
1.681862. The 194 original-BF16/Quartz rows have minimum cosine 0.989494,
maximum RMS 0.835602, and maximum absolute error 15.828717. The latter includes
expected Q4 quantization and accumulated late-layer differences; per-tap gates
are much narrower than these global extrema where the observed boundary permits.

## How the frozen gates were chosen

[`freeze_scalar_tolerances.py`](../tools/freeze_scalar_tolerances.py) groups the
two observed positions by authority, layer, and boundary. For each stable tap it
freezes:

- maximum absolute error at 1.10 times the diagnosed maximum, rounded upward to
  two significant digits;
- maximum RMS error by the same rule;
- minimum cosine by expanding the observed distance from one by 1.10, then
  rounding downward to six decimals;
- exactly zero NaN, positive-infinity, and negative-infinity counts.

The ten-percent margin covers ordinary implementation-level rounding movement;
it is not inferred to be a quality threshold. The rounding rule is deterministic
and tested. The calibration identities, tokens, positions, and layers are
stored beside the gates. The fixture records every raw zero-tolerance metric,
including first differing index and top-logit comparison.

These files are now immutable inputs to CUDA admission. A future optimization
may produce smaller errors. It may not regenerate the pin from its own output or
loosen a threshold to make itself pass. A model, quantization, architecture, or
layout-version change requires a separately reviewed new calibration, not a
silent edit.

## Greedy equality and near-ties

The **greedy token** is the vocabulary index with the highest logit. All three
authorities choose tokens 3649 and 1277. At position 0, the smallest winner
margin is the official checkpoint's 0.0625 logit; at position 1 it is 1.25.

A near-tie exception would be a stored case where two logits are so close that
allowed numeric movement changes their order even though tensor gates pass.
There are no such exceptions in this fixture: the exception list is exactly
empty, so greedy equality is mandatory. Merely calling a disagreement a
near-tie later cannot bypass the gate.

## Reproducing the evidence

[`run_scalar_authorities.sh`](../tools/run_scalar_authorities.sh) runs one large
GPU process at a time. It creates a unique ignored evidence directory, captures
one two-token Quartz bundle instead of rerunning the scalar model per tap, runs
the selective llama.cpp callback, runs official eager/offload capture, aligns
the results, and regenerates the checked-in fixture and pin.

The final raw identities are:

- Quartz 383,393,792-byte all-layer blob:
  `96e14a3e29af2781a9a716ec913098f2b576d988e27ab9ff5d8c3ab548261b17`;
- selected llama.cpp blob:
  `e950c76b04580d251ba2a9da5a0ba21cb73135202201f1ebd0696066ef0dc245`;
- selected official tap blob:
  `99d47367f411786f4d5f483a0a927491e412eca119bf7d7dcf0805538b1ab164`.

The large raw files remain ignored. The committed fixture retains their hashes,
all numeric summaries, exact mappings, and greedy margins.

## What this proves

- Exact model/tool identities and raw tokens produce finite, structurally
  aligned values at every selected official/Quartz semantic boundary.
- An independent same-GGUF runtime closely agrees at every boundary it exposes.
- Every recorded sample passes an immutable pre-CUDA maximum-absolute, RMS,
  cosine, finite-value, and greedy policy.
- The scalar path is now an admitted numeric oracle for the next CUDA primitive
  gate, not merely a structural self-test.

## What this does not prove

- Two tokens and five layers are not a quality evaluation, long-recurrence test,
  128K context proof, API test, or performance benchmark.
- These scalar tolerances do not automatically admit CUDA. Each CUDA primitive
  and fused path must compare at its visible boundaries without loosening them.
- A high cosine alone cannot excuse excessive absolute or RMS error.
- The official and GGUF weights are different artifacts; exact floating-point
  equality between them is neither expected nor claimed.
- llama.cpp does not expose every persistent state boundary; the official
  authority remains primary where independent visibility ends.
