# Offline CUDA dispatch tuning

[Index](README.md) · Implementation tasks: OPT-004, OPT-009, and EDU-040 in
[`implementation_ledger.md`](../implementation_ledger.md) · Evidence:
[`cuda/quant_mmv.cu`](../cuda/quant_mmv.cu),
[`cuda/dispatch_tuning_test.cu`](../cuda/dispatch_tuning_test.cu),
[`fixtures/cuda_dispatch_tuning.json`](../fixtures/cuda_dispatch_tuning.json),
the [OPT-004 raw sweep](../evidence/profiling/opt004-dispatch-sweep-raw.txt),
[`pins/cuda_prompt_mmq_contract.json`](../pins/cuda_prompt_mmq_contract.json),
[`fixtures/cuda_prompt_mmq.json`](../fixtures/cuda_prompt_mmq.json),
and the [OPT-009 raw sweep](../evidence/profiling/opt009-mmq-tile-sweep-raw.txt)

## What a dispatch table does

A CUDA kernel is a GPU function. A **launch shape** says how its threads are
grouped. The same arithmetic can run with several correct launch shapes, but
hardware scheduling, register use, and unused threads make their speeds differ.

Quartz does not tune while serving a request. An offline experiment measures a
small fixed candidate set on the production RTX 5090. The winning choices are
then written into ordinary C++ conditionals called a **dispatch table**. Runtime
selection is therefore deterministic, inspectable, and free of benchmark noise.

## MMV row buckets

MMV means matrix-vector multiplication: one activation row is multiplied by a
weight matrix during token decode. One 32-thread **warp** owns one output row.
The candidate is how many independent row warps share a thread block: 4, 8, or
16. This does not split or reorder an output's dot product; each warp still reads
and reduces its row exactly as before.

A **row bucket** maps a matrix's output-row count to the measured launch choice.
Qwen uses eight admitted row counts, from the 48-value GDN gates through the
248,320-token output projection:

| Maximum output rows | Warps per block |
|---:|---:|
| 48 | 4 |
| 1,024 | 8 |
| 5,120 | 16 |
| 10,240 | 8 |
| 12,288 | 4 |
| 17,408 | 8 |
| larger | 4 |

The 6,144-row shape falls inside the 10,240 bucket and also selected 8 warps.
The final bucket covers the 248,320-row vocabulary projection. These are
SM120/Qwen3.8 choices, not portable advice for another model or GPU.

## Prompt-row tiles and chunks

MMQ means matrix-matrix multiplication here: several prompt activation rows use
the same weight matrix. A **prompt tile** lets one warp decode a weight once and
reuse it across several prompt rows. More reuse reduces weight decoding, but
also increases per-thread registers and wastes work when the prompt is smaller
than the tile.

OPT-004 measured Q4_K MMQ tiles `{1,2,4,8}` on prompt chunks of 1, 2, 4, 8, 16,
32, and 64 rows for the FFN gate/up shape `17408 × 5120` only. Each of the first
three selected its matching tile; every chunk of eight or more selected tile 8.
That table is historical evidence. It is not mutated, and it is not the
production MMQ authority once chunks can be 4,096 rows and Q8_0/Q6_K also need
kind-specific winners.

OPT-009 remeasures production MMQ tiles on exclusive RTX 5090 hardware. Legal
compile-time tiles are `{1,2,4,8,16,32,64}`. Prompt-row buckets are `1, 2, 4, 8,
16, 32, 64, 256, 4096`. Q8_0 winners come from BF16 attention Q/gate
`12288 × 5120`. Q4_K winners minimize the sum of FFN gate/up `17408 × 5120` and
down `5120 × 17408`. Q6_K winners come from attention output `5120 × 6144`. A
candidate that fails to launch, writes a non-zero synthetic output, or reports
occupancy 0 cannot win.

The regenerated fixture winners are:

| Kind | Prompt-row buckets | Selected tile |
|---|---:|---:|
| Q8_0 | 1 | 1 |
| Q8_0 | 2 | 2 |
| Q8_0 | 4 and above | 4 |
| Q4_K | 1 | 1 |
| Q4_K | 2 | 2 |
| Q4_K | 4, 8 | 4 |
| Q4_K | 16, 32, 64, 256 | 8 |
| Q4_K | 4096 | 4 |
| Q6_K | 1 | 1 |
| Q6_K | 2 | 2 |
| Q6_K | 4 | 4 |
| Q6_K | 8 and above | 8 |

Q4_K's 4,096-row winner is tile 4 because that tile minimizes the joint sum of
the two FFN shapes; tile 8 is faster on `17408 × 5120` alone. Production
`selected_mmq_prompt_tile(kind, prompt_rows)` hard-codes those buckets. This
tile is an internal projection choice. It does not change token order, the
OPT-008 4,096-row outer chunk, or claim that the complete 64-layer scheduler now
has a new prefill throughput.

## How selection was measured

For each candidate the diagnostic performs three unrecorded warm-ups, then 30
synchronized CUDA-event samples. The complete sweep ran three times. Selection
uses the lowest arithmetic mean across those three replicates at each admitted
shape. All candidates—including losers—remain in the JSON fixture; all 30
individual samples from the admitted run remain in the raw text evidence.

Zero-filled packed weights keep the sweep deterministic and avoid copying a
second model, while preserving the real allocation sizes, memory traffic, block
decoding instructions, and output grids. This is valid for launch-shape timing,
not a numeric authority. Existing nonzero Q4_K/Q6_K/Q8_0 fixtures and the real
full-model scheduler remain the correctness authorities.

The OPT-004 prompt result is large and stable. For example, the three-replicate
mean for 64 Q4_K rows was about 11.06 ms at tile 1 versus 4.17 ms at tile 8.
Some MMV choices are close: the 17,408-row three-replicate means were about
0.1567 ms at four warps and 0.1561 ms at eight. The raw values are retained
rather than describing small differences as universal speedups.

OPT-009 uses the same warm-up/sample/replicate rule on the four production
shapes and writes every sample to
[`evidence/profiling/opt009-mmq-tile-sweep-raw.txt`](../evidence/profiling/opt009-mmq-tile-sweep-raw.txt).
The dedicated diagnostic also times production versus the retained row-wise
Q8_0 reference at 64 and 256 rows, and the selected Q4_K joint 4,096-row tile
versus tile 8. Those component means are retained in
[`fixtures/cuda_prompt_mmq.json`](../fixtures/cuda_prompt_mmq.json) and are not
end-to-end prefill measurements.

## Reproducibility and failure modes

- The executable rejects candidate counts outside the compiled sets.
- Every output is checked to remain zero in the synthetic sweep.
- The ordinary quant diagnostic checks nonzero decoded values and frozen numeric
  tolerances through the selected production path.
- The full scheduler rechecks real-model taps, logits, session state, and greedy
  continuation after dispatch changes.
- A different GPU, toolkit, clock state, model shape, or kernel revision requires
  a new sweep; the source hashes prevent old measurements admitting new code.

## Proof boundary

**Measured:** OPT-004 selects reproducible MMV row buckets and historical ≤64-row
Q4_K MMQ prompt tiles for the pinned RTX 5090, CUDA 13.0.2, SM120 kernels, and
Qwen3.8 shapes. It proves component launch selection and preserved correctness
gates.

**Measured component evidence, OPT-009:** production MMQ prompt-tile selection
is the kind-specific SM120 table above. It proves weight-tile reuse
(`grid.y = ceil(prompt_rows / selected_tile)`), occupancy, Q8_0 byte equality to
the retained row-wise kernel, frozen CUD-002 Q4_K/Q6_K envelopes, and the
component timing predicates in the prompt-MMQ fixture. It does not prove
end-to-end prefill/decode throughput, 128K quality recovery, another GPU, or the
comparative 5% release gate. BEN-001 and CMP-002/CMP-003 own those claims;
QLT-001 remains blocked.
