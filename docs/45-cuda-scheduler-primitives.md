# 45. CUDA prerequisites for the real scheduler

[Index](README.md) · Implementation tasks: CUD-003 and EDU-031 in
[`implementation_ledger.md`](../implementation_ledger.md)

The previous CUDA chapters proved the large mathematical pieces separately:
matrix multiplication, GDN recurrence, and attention. A decoder layer cannot
run by merely placing those pieces next to each other. It also needs small
operations that translate storage formats and join one stage to the next.

CUD-003 supplies that glue for the pinned model. It remains deliberately narrow:
these are named Qwen3.8 operations, not a general tensor framework.

## Two different meanings of Q8

The model file contains **resident weights**: values loaded once and reused for
every token. Most large matrices are Q4_K, but the pinned artifact's GDN and
attention input projections use GGUF **Q8_0**. A Q8_0 block stores one FP16 scale
and 32 signed bytes. To recover element `i`, Quartz computes:

```text
weight[i] = fp16_scale × signed_byte[i]
```

The earlier MMV kernel also creates a **transient activation** in an eight-bit
form. “Transient” means temporary: it is regenerated from the current BF16
hidden vector for a matrix multiplication, then its workspace is reused. That
local block has an FP32 scale and therefore is not the same byte format as a
Q8_0 model weight. Calling both “Q8” describes their eight-bit values, not an
interchangeable file layout.

[`quant_mmv.cu`](../cuda/quant_mmv.cu) now decodes Q8_0 weights in MMV and MMQ
while retaining the same transient-activation quantizer and FP32 accumulation.
The specialized entry points require columns divisible by 256. All production
projection widths satisfy that rule; an incompatible shape fails before launch.

## An embedding row is a lookup, not a matvec

The embedding table has one quantized row per vocabulary token. Selecting token
`42` means decoding row 42 into the 5,120-value hidden vector. There is no dot
product and no reason to quantize an input activation. The
[`launch_quant_row_decode`](../cuda/quant_mmv.h) boundary checks the format,
row, and shape, then writes BF16. The diagnostic decodes the same Q4_K row with
the scalar routine and requires every resulting BF16 bit to be identical.

## Why BF16 and FP32 meet at explicit boundaries

BF16 uses two bytes per stored activation, which reduces traffic and temporary
memory. It has fewer precision bits than FP32, so Quartz performs reductions and
sensitive equations in FP32 and rounds only at a declared storage boundary.
[`scheduler_primitives.cu`](../cuda/scheduler_primitives.cu) provides:

- **RMSNorm:** square the 5,120 BF16 hidden values, reduce their mean in FP32,
  add epsilon, take the reciprocal square root, apply the learned FP32 scale,
  and store BF16.
- **Residual add:** add an FP32 branch result to the earlier BF16 hidden value,
  then publish the next BF16 hidden vector. A residual connection lets a layer
  modify the stream instead of replacing it entirely.
- **SwiGLU:** compute `SiLU(gate) × up` for 17,408 FP32 pairs and store BF16.
  SiLU is `x / (1 + exp(-x))`; the second projection controls how much of that
  gated value continues into the down projection.
- **FP32-to-BF16 conversion:** an explicit rounding boundary for projection
  results whose next consumer expects BF16.

The device diagnostic builds BF16-aware scalar answers: it rounds at the same
named boundaries rather than comparing with an unrealistically all-FP32 model.

## Splitting packed attention output

One Q8_0 attention projection emits query and output-gate values together. For
each of 24 heads, the first 256 values are the query and the next 256 are its
gate:

```text
[head 0 query | head 0 gate | head 1 query | head 1 gate | ...]
```

Attention needs all queries in one contiguous array while its final gating
needs all gates in another. `launch_split_attention_query_gate` performs that
layout conversion. Production first/last boundary values must match exactly;
this catches a tempting but wrong “split the whole array in half” interpretation.

## Why GDN has tiled and grouped head orders

GDN has 16 key heads and three value replicas per key head, for 48 value heads.
The GGUF projection stores value heads in **tiled** or replica-major order:

```text
tiled_head = replica × 16 + key_head
```

The recurrence groups the three replicas belonging to a key head together:

```text
grouped_head = key_head × 3 + replica
```

These arrays contain the same conceptual heads in different orders. Confusing
them would produce finite but incorrect text, making the error harder to spot
than a crash. The new tiled GDN prepare entry maps projection values as the
recurrence reads them. Its diagnostic runs grouped and tiled inputs through the
same state transition and requires byte-exact outputs and recurrent state.

Two other kernels bridge this boundary. Gate preparation applies the folded
decay equation and sigmoid update gate while changing tiled head order to
grouped order. Gated output RMS-normalizes the grouped recurrent result, applies
the corresponding tiled gate, and stores the tiled BF16 vector expected by the
output projection.

## Frozen evidence and proof boundary

**Measured local:** Q8_0 MMV cases measured at most `7.62939453e-5` absolute
error and `2.69527864e-5` RMS against the scalar decoder. A three-row Q8_0 MMQ
case measured `5.34057617e-5` maximum and `2.00998238e-5` RMS. In every case the
temporary activation bytes were exact. These values are below the already
frozen CUD-001/CUD-002 gates.

**Measured local:** production-sized RMSNorm, residual add, and SwiGLU matched
their BF16-aware reference exactly. Three warm-ups and 30 CUDA-event samples of
that three-kernel sequence averaged about `0.0622 ms`. This is component timing,
not layer or token throughput.

**Measured local:** production attention splitting was exact. GDN gate
preparation measured `2.98023224e-8` maximum error and `1.22614319e-8` RMS, with
no non-finite values. Embedding decode was BF16 byte-exact, and the tiled GDN
path was byte-identical to grouped recurrence input.

The authenticated contract is
[`cuda_scheduler_primitives_contract.json`](../pins/cuda_scheduler_primitives_contract.json),
and retained measurements are in
[`cuda_scheduler_primitives.json`](../fixtures/cuda_scheduler_primitives.json).
The diagnostic test is
[`scheduler_primitives_test.cu`](../cuda/scheduler_primitives_test.cu).

CUD-003 proves that the pinned weight formats and visible layer boundaries can
feed the admitted CUDA cores. It does **not** yet upload all 851 tensors, run the
64-layer hybrid schedule, produce vocabulary logits, commit a complete session,
or demonstrate the 32 GiB fit. Those remain SCH-001 and later gates.
