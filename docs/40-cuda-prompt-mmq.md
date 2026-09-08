# 40. Tiled CUDA multiplication for prompt rows

[Index](README.md) · Implementation tasks: CUD-002, OPT-009, OPT-015, OPT-017, and EDU-026 in
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

The CUD-002 tile is a semantic starting point, not a claim that `8 × 4` is
fastest. OPT-009 later measured production dimensions, register pressure,
memory traffic, and larger row buckets on SM120, then froze kind-specific
dispatch tables. Those tables are component launch evidence, not an end-to-end
prefill claim.

## Shared staging, separate prompt rows

Q4_K and Q6_K prompt rows use the same BF16-to-Q8 rule admitted by CUD-001. Their
Q8 blocks are stored consecutively:

```text
row 0 Q8 blocks | row 1 Q8 blocks | row 2 Q8 blocks | ...
```

[`q8_prompt_workspace_bytes`](../cuda/quant_mmv.cu) calculates exactly that
scratch allocation. It returns zero for zero prompt rows or a column count that
cannot contain whole 256-value Q4_K/Q6_K blocks. The scratch is transient: it is
not model weight storage, KV history, GDN recurrence, or session state.
Production Q8_0 prompt rows skip that Q8Block workspace. Tiny prompts
(`prompt_rows < 8`) still multiply packed Q8_0 weights directly by BF16
activations on the OPT-009 tiled kernel. MMA production (`prompt_rows >= 8`)
stages activations to Q8_1 shared memory inside the kernel and still does not
call `launch_quant_mmq`.

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
tested shapes. It does not by itself prove full model projections, causal
scheduling, GDN scans, attention prefill, production dispatch choices, 128K
capacity, or a speed advantage.

## OPT-009 production weight-tile reuse

Production `matrix_prompt` now launches true multi-row tiles instead of rereading
the packed matrix once per prompt row.

**Q8_0 tiled path stays BF16.** SCH-002 introduced a Q8_0-by-BF16 kernel
specifically because sending those weights through Q8-staged `launch_quant_mmq`
changed persistent state and logits. That first kernel still mapped `blockIdx.y`
to one prompt row, so `grid.y` equalled `prompt_rows` and each output-row warp
reread every weight. OPT-009 moved the row-wise kernel to test-only
[`launch_q8_mmq_bf16_reference`](../cuda/quant_mmv.cu) and introduced templated
`q8_mmq_bf16_tiled` as [`launch_q8_mmq_bf16_variant`](../cuda/quant_mmv.cu). A
256-thread block still uses eight warps, one warp per output row. `blockIdx.y`
owns a prompt-row tile. For each column in the existing lane stride, the warp
decodes the Q8_0 weight once and applies that scalar to every in-range prompt
row with the decode `__fmul_rn(weight, __bfloat162float(activation))` and
`__fadd_rn` chain. Each prompt row is warp-shuffle-reduced independently in the
16…1 order. Token-major layout and tail guards are unchanged. The tiled
variant never requantizes Q8_0 activations through `launch_quant_mmq`. After
OPT-017, that variant is the visible unloosened byte-exact reference, not
production for `prompt_rows >= 8`.

**Q4_K and Q6_K keep the CUD-002 path.** They still stage transient Q8 blocks
and call `launch_quant_mmq`. OPT-009 only extends the compile-time tiles from
`{1,2,4,8}` to `{1,2,4,8,16,32,64}` and remeasures which tile wins. Illegal
tiles fail closed with `cudaErrorInvalidValue`.

**Selection is kind-specific.** `selected_mmq_prompt_tile(kind, prompt_rows)` is
a pure function of those two arguments. The one-argument wrapper still returns
the Q4_K table so CUD-002 assertions stay in one place. OPT-004's eight-row
ceiling remains historical MMQ evidence; it is not the production table.

The exclusive RTX 5090 sweep used CUDA 13.0.2, `sm_120`, and image
`qw38-cuda:13.0.2`. Each candidate had three unrecorded warm-ups, 30
synchronized CUDA-event samples, and three replicates. The winner is the lowest
arithmetic mean among candidates that launch, keep
`cudaOccupancyMaxActiveBlocksPerMultiprocessor >= 1`, and write only zeros on
the synthetic sweep. Prompt-row buckets are `1, 2, 4, 8, 16, 32, 64, 256,
4096`. Q8_0 winners come from attention Q/gate `12288 × 5120`. Q4_K winners
minimize the sum of FFN gate/up `17408 × 5120` and down `5120 × 17408`. Q6_K
winners come from attention output `5120 × 6144`.

Regenerated fixture winners:

| Kind | Prompt-row buckets | Selected tile |
|---|---:|---:|
| Q8_0 | 1 | 1 |
| Q8_0 | 2 | 2 |
| Q8_0 | 4, 8, 16, 32, 64, 256, 4096 | 4 |
| Q4_K | 1 | 1 |
| Q4_K | 2 | 2 |
| Q4_K | 4, 8 | 4 |
| Q4_K | 16, 32, 64, 256 | 8 |
| Q4_K | 4096 | 4 |
| Q6_K | 1 | 1 |
| Q6_K | 2 | 2 |
| Q6_K | 4 | 4 |
| Q6_K | 8, 16, 32, 64, 256, 4096 | 8 |

Q4_K's 4,096-row winner is tile 4 because that tile minimizes the joint sum of
the two FFN shapes. Tile 8 is faster on `17408 × 5120` alone and is not the
production choice.

On the retained 2026-09-07 RTX 5090 OPT-009 record, the tiled Q8_0 graphs used
`grid.y = ceil(prompt_rows / 4)`: 16 at 64 rows and 1,024 at 4,096 rows. The
reference graphs used `grid.y = prompt_rows`. Occupancy was at least one active
block per SM (Q8_0 5, Q4_K 3, Q6_K 2) with 0 compiler-reported local bytes per
thread. Component means: Q8_0 `12288 × 5120` 64-row 1.12 ms versus reference
2.59 ms; 256-row 4.62 ms versus 11.79 ms; Q4_K joint 4,096-row selected 614.2 ms
versus tile-8 706.1 ms.

`launch_q8_mmq_bf16_variant` outputs stay byte-identical to the retained
row-wise kernel. That OPT-009 admitting gate is unloosened. Q4_K/Q6_K keep
the frozen CUD-002 envelope: maximum absolute error `5e-4`, RMS `2.5e-4`,
exact Q8 staging, and zero non-finites.

**Measured component evidence, OPT-009:** weight-tile reuse, measured SM120
kind×bucket selection, occupancy, Q8_0 byte equality, Q4_K/Q6_K frozen
envelopes, and those component timing predicates. This is not an end-to-end
prefill or decode speedup, not the comparative 5% gate, not execution of a 128K
prefill, and not 128K retrieval quality. QLT-001 remains blocked. Contract,
fixture, and raw sweep:
[`pins/cuda_prompt_mmq_contract.json`](../pins/cuda_prompt_mmq_contract.json),
[`fixtures/cuda_prompt_mmq.json`](../fixtures/cuda_prompt_mmq.json), and
[`evidence/profiling/opt009-mmq-tile-sweep-raw.txt`](../evidence/profiling/opt009-mmq-tile-sweep-raw.txt).

## OPT-015 and the 2K FFN/MMQ sink

**Measured, RTX 5090, OPT-014:** one cold exact-2048 production `sync_tokens`
spent 32294.9512 ms in `ffn_mmq`, 77.0% of 41963.8828 ms host wall
(~48.80 tok/s). Those numbers are copied from
[`fixtures/cuda_prefill_attribution.json`](../fixtures/cuda_prefill_attribution.json);
this chapter does not add a new GPU sample.

At 2048 prompt rows, production Q4_K MMQ selects tile 4 because
`prompt_rows > 256` returns 4. The OPT-009 table has no 2048 bucket: tile 8
wins Q4_K at 16–256 rows, and tile 4 is the measured 4096-row winner for the
joint FFN shapes. 2048 therefore reuses each decoded weight across only four
prompt rows. The kernel remains scalar `__fmul_rn`/`__fadd_rn` weight-tile
reuse on 256-thread blocks. Occupancy is admitted; this is not tensor-core
MMA.

**External:** pinned llama.cpp `cc83d7b` Q4_K MMQ uses 256 threads × 128
output rows × prompt-tile J in `{8…128}`, Q8_1 shared-memory staging, and MMA
tensor-core dots. **Estimated** from that OPT-014 `ffn_mmq` interval: if GDN
and attention became free, Quartz would still be
`2048 / (32294.9512 / 1000) ≈ 63.4` tok/s versus the scaling `llama-bench`
exact-2048 mean 3114.049476 tok/s.

**Proposed** Rank 1 for the later 2K throughput gate is llama.cpp-style
Q4_K/Q6_K MMA MMQ under `plan.md:66-68` file-level provenance, keeping
`launch_quant_mmq` / CUD-002 as the visible reference and the frozen CUD-002
envelope. That sequence is the checked-in recovery map, not a throughput gate
and not llama.cpp parity:
[`evidence/optimization/opt015-2k-recovery/REPORT.md`](../evidence/optimization/opt015-2k-recovery/REPORT.md),
[`pins/opt015_recovery_contract.json`](../pins/opt015_recovery_contract.json),
and [`fixtures/opt015_recovery.json`](../fixtures/opt015_recovery.json).

## OPT-017 mixer Q8_0 MMA

Production mixer Q8_0 prompt MMQ (`matrix_prompt` → `launch_q8_mmq_bf16`)
uses an MMA/shared-memory path on `sm_120` when `prompt_rows >= 8`. Smaller
mixer prompts keep the OPT-009 tiled variant at
`selected_mmq_prompt_tile(kQ8_0, prompt_rows)`. Decode `q8_mmv_bf16` is
unchanged. Production Q8_0 still does not go through `launch_quant_mmq` or
`launch_quant_mmq_mma`.

The MMA kernel reuses the Q4_K/Q6_K geometry: 256 threads, 128 output rows, and
prompt-tile J in `{32, 64, 128}`. Each K-step of 32 columns is one GGUF Q8_0
block (34 bytes, FP16 scale, signed int8 values). Activations are staged to
Q8_1 shared memory with the existing warp-max / 127 rule. That SRAM layout is
not Quartz `Q8Block` and uses no extra `cudaMalloc`. The integer MMA is
`mma.sync.aligned.m16n8k32` then `acc += dA * dB * float(C)`. Output remains
token-major FP32. Partial prompt and output tiles are guarded.

**Visible unloosened OPT-009 references.**
`launch_q8_mmq_bf16_variant` versus `launch_q8_mmq_bf16_reference` remains
byte-exact on the existing prompt-row and shape set. Production MMA is not
memcmp'd to those BF16 kernels. `pins/cuda_prompt_mmq_contract.json`
`q8_bf16_reference_exact` stays `true`.

**CUD-002 admission versus staged Q8.** MMA is admitted against GPU staged Q8
MMQ: diagnostic `launch_quant_mmq_variant` with `kQ8_0` and a test-only
`Q8Block` workspace. Frozen CUD-002 numbers are unchanged: maximum absolute
error `5e-4`, RMS `2.5e-4`, and zero non-finites. Versus unstaged BF16
`launch_q8_mmq_bf16_variant` is informational only; INT8 MMA stages activations,
so that BF16 delta is not a pass/fail gate and is not a loosened envelope.

**Measured, RTX 5090, 2026-09-08:** mixer-shape J sweep at 2048 prompt rows on
`12288×5120`, `10240×5120`, and `5120×6144` (3 CUDA-event replicates, 0
warm-ups) pinned `q8_mma_prompt_tile_2048 = 128` with occupancy 2 and
shape-mean 21.80 ms. Component speed at 2048 rows on `12288×5120`: MMA 8.89 ms
versus tiled variant 42.97 ms. Worst staged-Q8 envelope on the component cases
was `gpu_staged_max_abs = 0.000282287598`, `gpu_staged_rms = 7.70256956e-05`,
zero non-finites.

**Measured remasurement, not the OPT-016 gate:** three cold unperturbed
exact-2048 Quartz `sync_tokens` replicates mean **375.743988** tok/s (walls
5453.1001 / 5449.18408 / 5449.27734 ms) versus live llama.cpp `cc83d7b`
`llama-bench` avg_ts **3203.276277**. `owns_opt016_parity_gate` is false;
`would_pass_opt016` is informational false. One post-remasurement OPT-014
attribution reconstructed wall 5383.53369 ms: `ffn_mmq` 2525.01465 ms,
`gdn` 1647.58936 ms, `attention` 1207.48132 ms. Mixer Q8_0 time stays inside
`gdn` and `attention` until OPT-020.

**External:** llama.cpp revision `cc83d7b4824f73cfdda4dfbb47ee39804f71b328`
Q8_0 MMQ MMA in `mmq.cuh`, `mma.cuh`, `mmq-config-ampere.cuh`,
`mmq-load-tiles.cuh`, and `mmq-vec-dot.cuh` (MIT, The ggml authors). ds4
`cuda/mmq` is the ggml-free launcher pattern only; it is not vendored and is
not a same-GGUF baseline.

**Proof boundary:** numeric envelopes unloosened; OPT-009 Q8_0 reference
remains byte-exact; **OPT-016 remains the parity gate owner**; this is not an
end-to-end 2K tok/s gate and not an 8K/32K/128K throughput gate. Contract,
fixture, and report:
[`pins/opt017_mixer_mma_contract.json`](../pins/opt017_mixer_mma_contract.json),
[`fixtures/opt017_mixer_mma.json`](../fixtures/opt017_mixer_mma.json), and
[`evidence/optimization/opt017-mixer-q8-mma/REPORT.md`](../evidence/optimization/opt017-mixer-q8-mma/REPORT.md).
