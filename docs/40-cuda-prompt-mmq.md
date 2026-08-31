# 40. Tiled CUDA multiplication for prompt rows

[Index](README.md) · Implementation tasks: CUD-002 and EDU-026 in
[`implementation_ledger.md`](../implementation_ledger.md)

[Chapter 39](39-cuda-quant-mmv.md) multiplied one activation vector by a packed
weight matrix. Prompt processing starts with several activation vectors—one for
each prompt token. CUD-002 adds a small two-dimensional CUDA tile that handles
those **prompt rows** together and reuses each decoded weight.

## From MMV to MMQ

The earlier MMV input has shape `[columns]` and produces `[output_rows]`. The new
operation receives a matrix of prompt activations:

```text
prompt BF16: [prompt_rows, columns]
weights:     [output_rows, columns]
result FP32: [prompt_rows, output_rows]
```

The second operation is matrix-matrix multiplication, abbreviated **MMQ** here
because one matrix is quantized. “Token-major” means all outputs for prompt
token zero come first, then all outputs for token one, and so on. The explicit
index in [`quant_mmq`](../cuda/quant_mmv.cu) is:

```text
output[prompt_row * output_rows + output_row]
```

Writing this rule down matters. Accidentally swapping the two indices can
produce all the right numbers in the wrong places.

## What a tile is

A **tile** is a small rectangle of the complete result assigned to one CUDA
thread block. Quartz's first correctness tile covers:

- eight output-weight rows, one per warp; and
- four prompt rows per warp.

That makes an `8 × 4` output tile. Each warp lane walks its share of the input
columns. It decodes one Q4_K or Q6_K weight and applies that same value to as
many as four prompt activations before moving to the next column. This is
**weight reuse**: the packed byte interpretation is paid once for several dot
products instead of once per prompt token.

The tile dimensions are a semantic starting point, not a claim that `8 × 4` is
fastest. Later profiling will test production dimensions, register pressure,
memory traffic, and larger row buckets before choosing tuned dispatches.

## Shared staging, separate prompt rows

Every prompt row uses the same BF16-to-Q8 rule admitted by CUD-001. Its Q8 blocks
are stored consecutively:

```text
row 0 Q8 blocks | row 1 Q8 blocks | row 2 Q8 blocks | ...
```

[`q8_prompt_workspace_bytes`](../cuda/quant_mmv.cu) calculates exactly that
scratch allocation. It returns zero for zero prompt rows or a column count that
cannot contain whole 256-value Q4_K/Q6_K blocks. The scratch is transient: it is
not model weight storage, KV history, GDN recurrence, or session state.

## Tails are ordinary inputs

A **tail** is the partly filled tile at the edge of a matrix. Four prompt rows
fit one tile exactly, but real chunks need not be divisible by four. Similarly,
the number of output rows need not be divisible by eight.

The device checks both indices before reading or writing. The admitted fixtures
use 1, 3, 5, and 9 prompt rows and 17 or 257 output rows. Those cases exercise:

- a single prompt row;
- a partial first prompt tile;
- one full tile plus one tail row;
- two full tiles plus one tail row; and
- partially occupied output blocks.

No padding value is exposed as a model result.

## Scalar comparison and frozen limits

The diagnostic creates a different deterministic BF16 activation for every
prompt row. It stages those rows on both host and GPU, requires every Q8 scale
and signed integer to match exactly, then compares every token-major output
against the readable CPU-001 decoder and sequential FP32 dot product.

MMQ retains the CUD-001 staging rule but owns a separate reduction boundary. The
first draft reused MMV's output limits and rejected the larger Q4_K fixture:
its maximum absolute error was only `0.000427246094`, but that exceeded the MMV
ceiling of `0.0003`. Before optimization, CUD-002 therefore froze:

- maximum absolute error at `5e-4`;
- maximum RMS error at `2.5e-4`;
- zero non-finite outputs; and
- exact transient Q8 staging.

Relative error remains report-only because outputs near zero can turn a tiny
absolute difference into a large ratio. The immutable contract is
[`pins/cuda_mmq_contract.json`](../pins/cuda_mmq_contract.json).

## Measured evidence and boundary

**Measured local:** all four Q4_K/Q6_K prompt cases passed on the RTX 5090 with
CUDA 13.0.2. They used three warm-ups and 30 synchronized CUDA-event samples.
The observed means were about 0.0061–0.0124 ms for these deliberately small
correctness shapes. [`fixtures/cuda_quant_mmq.json`](../fixtures/cuda_quant_mmq.json)
retains the complete summaries and labels them as diagnostic timing rather than
production prefill throughput.

CUD-002 proves arbitrary positive prompt-row counts, token-major output layout,
tail safety, exact staging, and scalar-equivalent Q4_K/Q6_K results for the
tested shapes. It does not yet prove full model projections, causal scheduling,
GDN scans, attention prefill, production dispatch choices, 128K capacity, or a
speed advantage. Those claims remain separate implementation-ledger gates.
