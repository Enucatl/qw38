# 43. CUDA grouped-query attention for one decoded token

[Index](README.md) · Implementation tasks: ATN-001 and EDU-029 in
[`implementation_ledger.md`](../implementation_ledger.md)

Chapter 19 introduced attention with small scalar examples. This chapter follows
the first CUDA decode implementation. “Decode” means processing one new token
after a prefix already exists. The kernel starts after the learned Q/K/V and gate
projections; matrix multiplication remains a separate primitive.

## Attention as a lookup

Attention lets the new token look back at earlier tokens. Each token contributes:

- a **query**, describing what the current token is looking for;
- a **key**, describing what a cached token can be matched by; and
- a **value**, the information returned when that cached token is selected.

Quartz compares a query with every allowed key, turns those scores into weights,
and computes a weighted sum of values. “Causal” means position 3 may read cache
rows 0 through 3, never row 4 or later.

## Heads and grouped-query attention

The production model has 24 query heads but only four key/value heads. Six
adjacent query heads share one KV head:

```text
query heads  0..5  -> KV head 0
query heads 6..11  -> KV head 1
query heads 12..17 -> KV head 2
query heads 18..23 -> KV head 3
```

This is **grouped-query attention**. Quartz computes the map as
`query_head / 6`; it does not make six physical copies of each cached KV row.
Every head has 256 lanes, so one layer stores `4 × 256` keys and the same number
of values per token.

## Normalization and partial RoPE

Before scoring, each query and key head receives RMS normalization. RMS means
“root mean square”: square the lanes, average them, add the small `1e-6` safety
constant, take a square root, and rescale the head. The learned direct scale is
then applied lane by lane.

RoPE, or Rotary Position Embedding, makes the representation depend on token
position. Quartz uses **partial RoPE**: only the first 64 of each head's 256
lanes rotate. Those lanes form two 32-lane halves. Lane 0 pairs with lane 32,
lane 1 with lane 33, and so on. The remaining 192 lanes stay normalized but
unrotated. Rotating all 256 would implement a different model.

[`normalize_query`](../cuda/attention_decode.cu) and the corresponding key kernel
use position-dependent sine and cosine with the pinned theta `10,000,000`.

## Why the KV cache uses BF16

KV history dominates long-context memory. A normal FP32 number uses four bytes.
**BF16** (bfloat16) keeps the same broad exponent range but stores fewer
precision bits in two bytes. The production row contains 1,024 keys plus 1,024
values, so it costs:

```text
(1,024 K + 1,024 V) × 2 bytes = 4,096 bytes per token per layer
```

Across 16 attention layers and 131,072 tokens, that is exactly 8 GiB. The
current normalized key and raw value are rounded to BF16 before becoming a
candidate row. Attention therefore compares against the values that will
actually persist, rather than silently using an FP32 row for the current token
and BF16 rows later.

## Causal scoring and stable softmax

For each query head, [`grouped_attention`](../cuda/attention_decode.cu) computes
the dot product with its shared KV head and divides by `sqrt(256)`. It reads
committed rows strictly before the current position and the separate candidate
row at the current position. A diagnostic fills a later committed row with
large sentinels; matching output proves that future row was ignored.

Raw scores can make a direct exponential overflow. **Stable softmax** first
subtracts the largest score, exponentiates, and divides by the sum. The resulting
positive weights add to one. Finally, each output lane is multiplied by the
sigmoid of its projected query gate.

## Candidate row, commit, and cancellation

Prepare must not make partial session progress visible:

```text
committed rows 0..2 + new position 3
              |
              v
prepare -> distinct candidate K/V row + output
commit  -> copy candidate into row 3 -> advance frontier to 4
```

The committed cache and frontier remain byte-identical during prepare. Commit
copies both BF16 arrays, synchronizes the block, and advances the frontier last.
Cancellation discards the candidate row without calling commit. Full-request
atomicity across all 64 layers still belongs to SES-002.

## Two references and the failed tight gate

The device diagnostic makes two comparisons:

1. A BF16-aware local reference performs the same storage conversion. This
   isolates CUDA arithmetic and layout errors.
2. The all-FP32 scalar calculation shows the numeric effect of the required
   two-byte cache and is checked against ORA-004's already-frozen actual-model
   attention-context limits.

The first attempt applied CPU-003's `3e-6` absolute and `1e-6` RMS limits, which
were frozen for an all-FP32 scalar transcription. Layer 3 failed at about
`1.80e-5` absolute and `9.25e-6` RMS solely after BF16 key rounding. That failed
result remains in the ledger. Quartz did not loosen CPU-003; it uses the already
frozen real-runtime layer-3 context gate: `0.051` absolute, `0.0016` RMS, and
`0.999424` cosine. This is also stricter than the frozen layer-7 and layer-63
context gates.

## Measured evidence and limits

**Measured local:** layers 3, 7, and 63 at a small inspectable shape plus one
production `24 × 4 × 256` case passed on the RTX 5090 with CUDA 13.0.2. Candidate
BF16 bytes were exact, prepare preserved committed bytes/frontier, commit changed
only position 3 then advanced the frontier, and all outputs were finite.

The production case measured `1.1920929e-7` maximum error and `7.15209136e-9`
RMS against the BF16-aware reference. Its all-FP32 comparison measured about
`2.051e-4` maximum error, `1.730e-5` RMS, and `0.999999982` cosine. After three
warm-ups, 30 synchronized samples averaged about `0.0328 ms` for this four-row
post-projection core.

These timings exclude Q/K/V projections, output projection, residual and FFN
work, graph launch, and long-context memory traffic. ATN-001 proves one-token
decode semantics and state publication. ATN-002 must still implement arbitrary
causal prompt chunks, memory-bounded score processing, and the 131,072-token
capacity boundary.
