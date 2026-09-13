# OPT-115 — Complete-request memory traffic and conditional bounds

Status: **measured**. A/A verdict
`repeatable`. Hardware executed
`True`.

diagnostics only: no production selector or kernel change; fresh post113_selected is the authenticated control; pinned llama cc83d7b4824f73cfdda4dfbb47ee39804f71b328 is the benchmark authority; 3 warmups plus 10 interleaved AB/BA pairs at P4096, D128, and D2048; long-context D8192/D32768/D131040+32 are separately labeled probes in a 131072-capacity session; 1792 GB/s is listed RTX 5090 peak, not observed application bandwidth; event unions distinguish GPU busy/idle, host overlap, sync and launch submission; no throughput or optimality claim.

## Sitting identity

- device: NVIDIA GeForce RTX 5090
- llama_revision (pinned authority): `cc83d7b4824f73cfdda4dfbb47ee39804f71b328`
- llama inspection HEAD (not the pin): `1945e092030f8668ff93382799502d01490e564d`
- ds4 inspection HEAD: `c238077a87186381bf626cc531bccffe1fef79e7`
- gguf_sha256: `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`
- source: `e4e02bfea64e2d44fc6e4e7537f447c46f7c6d67` (dirty)
- nvccflags: `-O2 --fmad=false`
- execution_graphs: `ffn_only`
- gpu_blocker: None

post113_authenticated=`True`; Q4=`True`;
prompt attention=`True`; decode attention
`True`; GDN `True`;
graphs `True`. OPT-113 tok/s numbers are historical
observations, not this sitting's paired control.

## A/A same-binary post113 control

3 warmups + 10 interleaved AB/BA pairs. Combined verdict
**repeatable**.

| Workload | A tok/s | B tok/s | geo ratio | Verdict |
|---|---:|---:|---:|---|
| p4096 | 2985.6727 | 2986.1139 | 1.0001 | repeatable |
| d128 | 57.3949 | 57.3951 | 1.0000 | repeatable |
| d2048 | 55.6682 | 55.6640 | 0.9999 | repeatable |

Session loop (sync → greedy sample → eval): 
status=`measured`;
TTFT_ms=`1402.3132`;
p50=`20.6323`;
p95=`20.6806`.
CLI/server cancellation poll is `EvalControl::poll` at layer/chunk boundaries;
benchmark `execute_token`/`sync_tokens` paths omit the public Session poll
unless an `EvalControl` is supplied.

## Long-context probes

Capacity 131072. Capacity alone is not evidence of reading a populated cache.
These probes are separately labeled and are not D128/D2048 wins.

| Probe | prefix | outputs | tok/s | TTFT ms | p50 ms | p95 ms | status |
|---|---:|---:|---:|---:|---:|---:|---|
| d8192 | 8192 | 32 | 41.4200 | 3159.9185 | 24.1365 | 24.1649 | measured |
| d32768 | 32768 | 32 | 22.1180 | 21911.7754 | 45.1987 | 45.2872 | measured |
| d131040 | 131040 | 32 | 7.7232 | 242093.4380 | 129.4768 | 129.5350 | measured |

## Bytes / lifetime (decode D128 tensor-derived compulsory)

Units are bytes. Peak ms uses 1792 GB/s listed peak.

| Phase | bytes/token | units | peak_ms |
|---|---:|---|---:|
| weights | 18247717696 | bytes | 10.1829 |
| activations | 5767168 | bytes | 0.0032 |
| kv | 8454144 | bytes | 0.0047 |
| gdn_state | 158859264 | bytes | 0.0886 |
| transfers | 1013760 | bytes | 0.0006 |
| logits | 993280 | bytes | 0.0006 |
| barriers | 0 | bytes | 0.0000 |
| graphs | 0 | bytes | 0.0000 |

Resident 128K ledger: model
18973870432 B, KV
8589934592 B, GDN
158859264 B, workspace
1831836288 B, reserve
1610612736 B.

## Timeline coverage

Method: CUDA-event unions; overlapping intervals are not summed.

| Prefix | GPU busy ms | GPU idle ms | host union ms | unclassified ms | coverage |
|---|---:|---:|---:|---:|---:|
| D128 | 7.8908 | 9.9747 | 0.0000 | 0.6377 | 0.9658 |
| D2048 | 8.2642 | 10.0257 | 0.0000 | 0.4792 | 0.9747 |

## Bandwidth bounds

D128 overlap-aware peak bound
`10.1829` ms.
Independent phase sums are **not** a bound.
Assumptions: listed 1792 GB/s peak; embedding is one Q4_K row; GDN decode
does not copy the 8 GiB KV cache; measured sustainable bandwidth is
`None`.

## Pinned-source comparison

Pinned llama `cc83d7b4824f73cfdda4dfbb47ee39804f71b328`. Inspection llama HEAD
`1945e092030f8668ff93382799502d01490e564d` is_not_pin=
`True`. ds4 HEAD
`c238077a87186381bf626cc531bccffe1fef79e7`.
DeepSeek compression/indexer/MoE is not Qwen's 48-GDN/16-attention

## Ranked OPT-117–122 opportunities

| Rank | Task | Verdict | removable bytes upper | peak ms upper | primary |
|---|---|---|---:|---:|---|
| 1 | OPT-121 | go | 18247717696 | 10.1829 | d128/d2048 |
| 2 | OPT-117 | go | 0 | 9.9747 | d128/d2048 |
| 3 | OPT-120 | go | 8587902976 | 4.7924 | long-context decode |
| 4 | OPT-122 | go_conditional | 158859264 | 0.0886 | complete request after OPT-118 |
| 5 | OPT-119 | go | 5767168 | 0.0032 | prefill (P4096) then decode guard |
| 6 | OPT-118 | go | 1013760 | 0.0006 | complete sync/sample/eval request |

No keep. production_kept=True.
claims_throughput=false. claims_performance_improvement=false.

## Shared keep protocol (frozen for OPT-117–123)

parent=`post113_selected`; quality_against=`post113`;
screen 3 warmups / 5 pairs;
acceptance 3 / 10;
decode outputs 256; ratio lower bound
1.0; non-target >=
0.98; decode p95 <=
1.05× parent. Memory-only is not a
throughput keep.
