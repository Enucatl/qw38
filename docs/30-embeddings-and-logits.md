# 30. From token IDs to embeddings, and hidden state to logits

[Index](README.md) · Implementation tasks: CPU-015 and EDU-016 in
[`implementation_ledger.md`](../implementation_ledger.md)

A language model does not send integer token IDs directly through its decoder
layers, and its final layer does not directly emit text. Two learned table-like
boundaries translate between discrete vocabulary entries and continuous
floating-point vectors:

```text
token ID -> embedding -> 64 decoder layers -> final norm -> logits
```

CPU-015 implements the two ends of that path using the real pinned GGUF. The 64
layers in the middle are not joined yet, so the diagnostic deliberately feeds
token 42's embedding directly into the final norm. That is a boundary test, not
a claim that the resulting logits are a valid model continuation.

## Token IDs are integers, not words

The tokenizer from Chapter 15 converts text bytes into a sequence of token IDs.
An ID is an integer index into a fixed vocabulary. It has no arithmetic meaning:
ID 43 is not “one more semantic unit” than ID 42.

The pinned text vocabulary contains 248,320 entries, so valid IDs satisfy:

```text
0 <= token_id < 248320
```

The largest valid ID is 248,319. ID 248,320 is out of range. CPU-015 checks the
bound before touching the destination buffer, which prevents an invalid request
from reading beyond the mapped model.

## Embedding lookup

The embedding tensor has shape `[5120, 248320]` in GGUF order. Dimension zero is
the contiguous row width, so each vocabulary entry owns one 5,120-value row:

```text
embedding = decode_Q4_K_row(token_embd.weight, token_id)
```

The result is a hidden vector: 5,120 floating-point features that the first
decoder layer can transform. Individual feature lanes do not have stable names
like “noun” or “French.” Training distributes useful information across the
whole vector.

The mapped embedding tensor occupies 715,161,600 bytes (682.03125 MiB). Quartz
decodes only the requested Q4_K row into a 20 KiB FP32 activation. It does not
expand all 248,320 rows into memory.

The diagnostic checks IDs 0, 42, and 248,319. Testing the first and last rows
catches range and row-stride mistakes; an interior row catches code that might
accidentally work only at an endpoint.

## The final hidden vector

After a real token has crossed all 64 layers, it is still a 5,120-value hidden
vector. Earlier layers have mixed current-token features with recurrent or KV
history, but the representation width stays constant so every layer can use
residual additions.

CPU-015 accepts any correctly sized final hidden vector. For isolated evidence
it uses token 42's real embedding as deterministic input. CPU-004 must later
replace that shortcut with the actual layer-63 output.

## Final direct-scale RMSNorm

The model applies one final RMSNorm before comparing the hidden state with
vocabulary output rows:

```text
mean_square = sum(hidden[i]²) / 5120
normalized[i] = hidden[i] / sqrt(mean_square + 1e-6)
normalized[i] *= output_norm_scale[i]
```

The final GGUF norm is a direct scale. The converter has already accounted for
the source checkpoint's offset convention, so Quartz must not add another one.
The prepared scale and normalized workspace each contain 5,120 FP32 values, or
20 KiB.

Final normalization keeps the magnitude presented to the output matrix stable.
It does not select a token and does not change the vocabulary size.

## Vocabulary projection

The output matrix also has 248,320 rows, one candidate next token per row. Each
Q6_K row takes a dot product with the normalized hidden vector:

```text
logit[token] = dot(output.weight[token], normalized_hidden)
```

Computing every row produces 248,320 FP32 logits. A **logit** is an unnormalized
score: it can be positive or negative, and all logits do not sum to one. A
higher value means the model currently favors that token more strongly.

The Q6_K output matrix occupies 1,042,944,000 bytes (about 994.629 MiB) in the
pinned artifact. One complete scalar projection performs
`248,320 × 5,120 = 1,271,398,400` weight products. This is exact dimension
arithmetic, not a speed estimate for the later CUDA kernel.

The FP32 logits occupy 993,280 bytes (970 KiB). Together with the 20 KiB
normalized vector, the explicit output workspace is 990 KiB. The caller owns
the logits because sampling and inspection happen outside the tensor scheduler.

## Logits, probabilities, and token choice

There are several ways to turn logits into a token:

- **greedy selection** chooses the index of the largest logit;
- **softmax** converts logits into probabilities whose sum is one;
- temperature, top-k, and top-p can reshape or restrict those probabilities;
- a pseudorandom sampler then chooses according to the configured distribution.

CPU-015 reports the greedy index only as a diagnostic that all logits were
available for comparison. It does not implement public sampling. Keeping raw
FP32 logits separate is deliberate: `Session::logits` can expose scores, while
`Session::sample` can apply a policy without mutating model state.

Softmax is unnecessary for greedy selection because it preserves ordering: the
largest logit also receives the largest softmax probability. It is necessary
when sampling according to probabilities.

## Independent evidence

[`tools/generate_real_model_boundary_fixtures.py`](../tools/generate_real_model_boundary_fixtures.py)
maps the pinned artifact independently. It:

1. decodes complete Q4_K embedding rows 0, 42, and 248,319;
2. applies direct-scale final FP32 RMSNorm to row 42;
3. decodes Q6_K output rows 0, 1, 42, 1,000, and 248,319; and
4. computes selected logits with the admitted blockwise FP32 accumulation order.

Exact SHA-256 hashes in
[`fixtures/real_model_boundaries.json`](../fixtures/real_model_boundaries.json)
bind every selected physical row to the pinned GGUF. Numeric tests report
absolute, relative, and RMS error and reject NaN or infinity.

The native diagnostic computes and checks finiteness for all 248,320 logits;
selected independently decoded rows provide semantic evidence without making
the Python generator repeat the entire 1.27-billion-product projection. The
full greedy result is therefore a native whole-output check, not an independent
continuation authority.

## Failure behavior

Two negative cases exercise opposite ends:

- token ID 248,320 fails before writing any embedding lane;
- a normalized workspace one value too short fails before writing either the
  normalized vector or any logit.

After exact structural validation, the scalar vocabulary projection writes
logits row by row. A hypothetical later arithmetic failure could leave a partial
temporary logits buffer, which the caller must discard. No GDN, KV, token, or
position state is mutated by these boundary functions.

## Proof boundary

**External:** the official model contract defines vocabulary size, residual
width, embedding lookup, final RMSNorm, and vocabulary projection order.

**Measured:** real endpoint and interior embedding rows, final normalization,
and selected output logits match independently decoded pinned-GGUF evidence;
all 248,320 native logits are finite; exact bounds and malformed workspaces fail
before writes. The scalar native boundary completed in 2.21 seconds with
1,045,280 KiB maximum RSS on this host run.

**Estimated:** embedding/output tensor sizes, 20 KiB row/norm sizes, 970 KiB
logit storage, and 1,271,398,400 scalar products follow directly from exact
artifact dimensions and formats.

CPU-015 does not connect the embedding to layer 0 or layer 63 to final norm. It
also does not prove real continuation logits, token-wise execution, chunked
prefill, sampling, or session commit. Those remain CPU-004 and later gates.
