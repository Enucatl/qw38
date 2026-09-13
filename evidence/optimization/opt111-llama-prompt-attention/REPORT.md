# OPT-111 — Reproduce the remaining llama F16 MMA prompt-attention differences

Status: **kept**. Authority llama.cpp `cc83d7b4824f73cfdda4dfbb47ee39804f71b328`,
GGUF SHA-256 `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`.
Control is shipping OPT-079 `kv_once`. Base candidate `opt111_base` is
convert-once plus llama decreasing-granularity `cp.async` KV load. Optional XOR
candidate `opt111_xor` adds exactly one e4b9af007 K/V shared-tile swizzle,
screened only after a measured bank-conflict/transaction hypothesis. OPT-079
convert-once is not counted as a new gain. OPT-051 `f16_async` is historical
ancestry, not the performance control.

`uses_opt098_arrays_for_keep_reject: false`. Fresh OPT-114 sitting is the
reporting baseline for tok/s deltas. OPT-108 decode-vector work is independent.

## Claim labels and proof limits

Pinned Ampere `fattn-mma-f16` (DKQ=DV=256, ncols=32) remaining delta versus
shipping `kv_once`; causality, rotary, norms, scale, gates, KV visibility, and
atomic publication unchanged; matched primitive rows 1/32/128/512/2048/4096 with
prep+combine+conversion included; row 1 is tiled and parity-only; XOR screened
only after measured bank-conflict signal; no tile/swizzle sweep; complete
16-layer P4096 attention requires ≥20 ms saving with positive 95% CI; P4096
engine five AB/BA graph pairs strictly faster within 2%; D128/D2048 within OPT-114
2% bound; candidate NLL measured via OPT-058 `--quality --quality-config`;
component and engine measurements are separate claims.

## Sitting identity

- device: NVIDIA GeForce RTX 5090
- image: `qw38-cuda:13.0.2`
- llama_revision: `cc83d7b4824f73cfdda4dfbb47ee39804f71b328`
- gguf_sha256: `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`
- execution_graphs: `ffn_only` (unchanged; OPT-114)
- shipping attention pin (after keep): `opt111_base`
- OPT-114 reporting baselines: D128 53.4602 tok/s; D2048 48.3830 tok/s; P4096
  2981.0894 tok/s

## Remaining differences (pinned fattn-mma-f16 vs kv_once)

Pinned Ampere fattn-mma-f16 DKQ=DV=256 ncols=32 vs shipping kv_once:
nthreads=128 occupancy=2 nbatch_fa=32 nbatch_K2/V2/combine=128 nstages=2
Q_in_reg=true.

| Component | Pinned llama | Shipping kv_once |
|---|---|---|
| query_tiling | ncols1=16 ncols2=2 (ncols=32) | matches |
| kv_tiling | nbatch_fa=32 | matches |
| shared_memory | stride nbatch_K2+4 half2 | linear width=256 halfs, no pad |
| mma | ldmatrix tile<16,8,half2> | 16×8 fragment fill then mma.sync |
| softmax | register online softmax | register online softmax |
| causal_mask | optional mask tile | absolute>qpos −inf |
| gqa | ncols2 | Ncols2=2 kSubgroups=3 |
| partition | stream-K np | KvParts=2 |
| fixup | fattn fixup metadata in pad | stream-K meta buffer |
| combine | nbatch_combine+4 | fattn_stream_k_combine_kernel |
| kv_conversion_lifetime | F16 KV cache | BF16 then OPT-079 convert-once (not a new gain) |

Candidate delta: llama decreasing-granularity `cp.async` KV load on top of
OPT-079 convert-once. XOR swizzle (e4b9af007) is exactly one extra candidate
after a measured bank-conflict/transaction hypothesis.

Live attrs (acceptance primitive): kv_once 195 regs / occ 1; opt111_base 202/1;
opt111_xor 223/1.

## Parity (feedback; max_abs gate)

Tiny (1/17/33), causal boundary (32), and tail (33) cases on identical prompt
buffers: max_abs=0, nonfinite=0, equal=true for `opt111_base` vs `kv_once`.

## Primitive screen (acceptance; 3 warmup + 10 samples; must win before XOR or engine)

primitive_win=true reason=matched_primitive_won

| tokens | kv_once ms | opt111_base ms | faster | positive | note |
|---:|---:|---:|---|---|---|
| 1 | 0.040355 | 0.040758 | false | false | tiled_only parity |
| 32 | 0.082451 | 0.076669 | true | true | go/no-go |
| 128 | 0.120909 | 0.111680 | true | true | go/no-go |
| 512 | 0.491059 | 0.437974 | true | true | go/no-go |
| 2048 | 4.535616 | 3.993731 | true | true | go/no-go |
| 4096 | 15.849989 | 13.825627 | true | true | go/no-go |

## XOR screen (feedback; after base wins)

xor_screened=true hypothesis=true bank_conflict_hypothesis=true
linear_ms=0.123616 xor_ms=0.120864 source=e4b9af007.
Complete keep used base; XOR primitive 4096 row 14.148 ms vs base 13.826 ms.

## Complete 16-layer P4096 attention (acceptance; 3 warmup + 10 paired)

Survivor `opt111_base`:

| path | control ms | candidate ms | saving ms | 95% CI | keep_bar |
|---|---:|---:|---:|---|---|
| opt111_base | 253.651 | 221.216 | 32.436 | [32.407, 32.464] | pass |
| opt111_xor | 253.676 | 227.453 | 26.222 | positive | not selected |

## P4096 engine (acceptance; five AB/BA graph pairs)

Control `kv_once` 1366.183 ms / **2998.38 tok/s**; candidate `opt111_base`
1344.412 ms / **3046.73 tok/s**; faster=true within_2pct=true.

| reference | tok/s |
|---|---:|
| OPT-114 baseline | 2981.09 |
| candidate | 3046.73 |
| **delta vs OPT-114** | **+65.6** |
| llama ref | 3252.58 |
| Quartz pre-OPT-111 | 3036.84 |

## D128 / D2048 guards (acceptance; one pair each)

OPT-114 baselines are the 2% non-regression bound, not keep/reject gates.

| prefix | control tok/s | candidate tok/s | OPT-114 tok/s | within 2% |
|---|---:|---:|---:|---|
| D128 | 58.423 | 58.419 | 53.460 | true |
| D2048 | 56.797 | 56.799 | 48.383 | true |

Reporting deltas vs OPT-114: D128 **+4.959**, D2048 **+8.416**.

## Quality (OPT-058; acceptance)

model_quality_pass=true candidate_nll_measured=true reason=opt111_quality_pass.
Both `kv_once` and `opt111_base` measured with `effective_attention` printed.

| case | kv_once NLL | opt111_base NLL | ppl_ratio |
|---|---:|---:|---:|
| held_out | 1.787599 | 1.787599 | 1.0 |
| wikitext | 1.524935 | 1.524935 | 1.0 |
| recurrence_incremental vs control | — | 0.0 | (max \|delta\| 0.02) |

## State, graph, and prompt-to-decode handoff (acceptance)

- graph/eager equality: pass (max_abs=0)
- graph_eager_repeat: pass (capture stream fix applied before quality)
- prompt_to_decode_kv_handoff: pass (atomic_publication, candidate_committed_visibility)
- cancellation: pass
- rotary, norms, scale, gates, causality: preserved

## Independent verdicts

From [`fixtures/opt111_llama_prompt_attention.json`](../../../fixtures/opt111_llama_prompt_attention.json)
`independent_verdicts`:

| path | kernel_parity | primitive | model_quality | performance | production_kept |
|---|---|---|---|---|---|
| kv_once | pass | pass | pass | pass | false |
| opt111_base | pass | pass | pass | pass | **true** |

## Decision

Verdict: **keep** (`matched_primitive_won`, `complete_ci_positive`,
`p4096_faster_within_2pct`, `d128_d2048_within_opt114`, `candidate_nll_measured`).
`production_kept=true`. Shipping attention pipeline=`opt111_base`.

Tok/s delta vs OPT-114: P4096 **+65.6**; D2048 **+8.416**; D128 **+4.959**
(reporting only).

Evidence also in
[`fixtures/opt111_llama_prompt_attention.json`](../../../fixtures/opt111_llama_prompt_attention.json);
contracts in
[`pins/opt111_llama_prompt_attention_contract.json`](../../../pins/opt111_llama_prompt_attention_contract.json),
[`pins/opt111_iteration_contract.json`](../../../pins/opt111_iteration_contract.json),
[`pins/opt111_llama_prompt_attention_provenance.json`](../../../pins/opt111_llama_prompt_attention_provenance.json);
host validator [`tests/test_opt111_llama_prompt_attention.py`](../../../tests/test_opt111_llama_prompt_attention.py).
