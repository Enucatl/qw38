# OPT-124 — Practical performance ceiling under explicit constraints

Status: **analysis**. Hardware executed
`True`.
universal_optimality_claimed=`False`.
supports_continuing=`False`.

analysis/diagnostics only: no production selector or kernel change; reuse OPT-123 samples only with matching identities; pinned llama cc83d7b4824f73cfdda4dfbb47ee39804f71b328 remains the benchmark authority; 1792 GB/s and 104.8 TFLOPS are listed peaks, not measured application rates; a bound that excludes unpack/scale or serial launch idle is optimistic, not attainable; independent phase sums are not a bound; no universal-optimality assertion; speculation/MTP/batching/model replacement/sparse attention are outside this approved batch

## Sitting identity

- device: NVIDIA GeForce RTX 5090
- final Quartz stack: `combined_opt118_opt119`
- historical control: `post113_selected`
- llama_revision (pinned authority): `cc83d7b4824f73cfdda4dfbb47ee39804f71b328`
- llama inspection HEAD (not the pin): `1945e092030f8668ff93382799502d01490e564d`
- gguf_sha256: `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`
- source: `a8daac9b610b331be9e5f559757fdd189a6cb04f` (dirty)
- nvccflags: `-O2 --fmad=false`
- execution_graphs: `ffn_only`
- OPT-123 identities match: `True` (reuse samples `True`)
- gpu_blocker: None

OPT-113 tok/s numbers remain historical. This comparison uses the OPT-123
combined sitting versus authenticated post113 and pinned llama on the same
request boundaries, generated token counts, capacities, and GGUF.

## Recomputed byte / compute / serial bounds

Listed peaks: 1792 GB/s DRAM, 104.8 TFLOPS FP32, 209.5 TFLOPS dense FP16
tensor (2× FP32, no sparsity). Measured sustainable bandwidth is `None`.
Independent phase sums are **not** a bound. Unpack/scale is excluded from
2×N×T, so the compute number is optimistic. Combined-stack decode D2H is
the measured 4-byte greedy index, not 1,013,760 B logits.

| Workload | byte critical ms | optimistic compute ms | serial idle ms | busy overlap ms | Quartz wall ms | llama wall ms | residual vs busy ms | idle source |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| p4096 | 10.1895 | 1051.7041 | n/a | 1051.7041 | 1395.3836 | 1318.9822 | 343.6795 | n/a |
| d128 | 10.1829 | 0.2569 | 9.9783 | 10.1829 | 18.5131 | 14.5625 | 8.3302 | combined_timeline |
| d2048 | 10.1829 | 0.2587 | 10.0028 | 10.1829 | 24.0193 | 14.8842 | 13.8365 | combined_timeline |
| d8192 | 10.1829 | 0.2645 | n/a | 10.1829 | 123.9811 | n/a | 113.7982 | n/a |
| d32768 | 10.1829 | 0.2875 | n/a | 10.1829 | 739.6772 | n/a | 729.4943 | n/a |
| d131040 | 10.1829 | 0.3797 | n/a | 10.1829 | n/a | n/a | n/a | n/a |

Serial dependency: ordinary autoregressive batch-one execution cannot overlap
tokens. Prefill may amortize a weight tile across prompt rows inside one
microbatch; decode cannot. Speculation, MTP, multi-session batching, model
replacement and sparse attention are outside this approved batch. Their
exclusion prevents a universal claim; it does not authorize implementation.

Lower-precision alternatives (OPT-120 q8q8 KV, OPT-121 Q8→Q4_K weights,
OPT-122 BF16 GDN) are **not admitted**. A smaller byte model is not an
identical-arithmetic kernel win.

## Matched llama / post113 / combined comparison

Pinned llama authentication: ok=`True`; n_gpu_layers=
`99`; flash_attn auto=`True`;
backends=`CUDA`; type_k/v=`f16`/
`f16`. Authority unchanged. Supplementary inspection
HEAD present=`True` revision=`1945e092030f8668ff93382799502d01490e564d`
measured=`False`. Failure to run a supplementary baseline
limits the strongest-local-llama clause and does not invalidate the pin.

Quality (OPT-116 successor, budgets anchored to post113): strict=
`True` successor=`True`
PPL ratio=`1.0` candidate NLL measured=
`True`.

| Workload | post113 tok/s | combined tok/s | llama tok/s | combined ms | llama ms | combined/llama |
|---|---:|---:|---:|---:|---:|---:|
| p4096 | 2917.1056 | 2935.3935 | 3105.4247 | 1395.3836 | 1318.9822 | 0.9452 |
| d128 | 53.7729 | 54.0159 | 68.6696 | 18.5131 | 14.5625 | 0.7866 |
| d2048 | 41.4485 | 41.6331 | 67.1855 | 24.0193 | 14.8842 | 0.6197 |

Long-context (separately labeled, not D128/D2048 wins): D8192 tok/s
`8.0657`; D32768
`1.3519`; D131040 OOM=
`True`. OPT-016 2K llama tok/s
`3288.2052` quartz measured=`False`.
Non-exclusive sitting (unrelated processes) is why 128K/2K OOM; this dossier
does not authorize stopping them. A smaller allocated context or skipped host
sampling is not an inference improvement under identical constraints.

## Combined-stack timeline (optional GPU)

Method: CUDA-event unions; overlapping intervals are not summed. OPT-115
post113 overlay is not silently treated as combined.

| Prefix | valid | GPU busy ms | GPU idle ms | unclassified ms | coverage | identity |
|---|---|---:|---:|---:|---:|---|
| d128 | True | 8.2922 | 9.9783 | 0.7223 | 0.9622 | combined_opt118_opt119 |
| d2048 | True | 10.5408 | 10.0028 | 0.4485 | 0.9787 | combined_opt118_opt119 |

## Mechanism disposition (OPT-117–122)

| Task | outcome | bytes upper | peak ms upper | candidate | reopen |
|---|---|---:|---:|---|---|
| OPT-117 | retain_ffn_only | 0 | 9.9747 | decode_segments8 | no |
| OPT-118 | keep | 1013760 | 0.0006 | lazy+overlap | no |
| OPT-119 | keep | 167772160 | 0.0936 | mixer_q8+ffn_q8 | no |
| OPT-120 | quality_blocked | 4026531840 | 4.7923 | q8q8 | no |
| OPT-121 | quality_blocked | 3355443200 | 1.8725 | q8_to_q4k | no |
| OPT-122 | no_material_opportunity | 150994944 | 0.0000 | none admitted (BF16/Q8 bound-rejected) | no |

OPT-118 and OPT-119 are the keepers. OPT-117/120/121/122 outcomes do not reverse on the combined stack: graphs remain ffn_only, KV dense BF16, weights unrequantized, GDN FP32. No new causal reason to reopen a rejected idea appeared in OPT-123 leave-one-out.

## Supported conclusions per workload

Labels are only: quartz_advantage, measurable_headroom,
no_demonstrated_worthwhile_headroom, insufficient_evidence. None of these is a
universal optimality claim. Materiality default is 2% of request time.

| Workload | conclusion | Quartz tok/s | llama tok/s | ratio | remaining mechanism |
|---|---|---:|---:|---:|---|
| p4096 | measurable_headroom | 2935.3935 | 3105.4247 | 0.9452 | prefill MMQ/attention efficiency versus pinned llama (OPT-119 already kept activation fusion; weight requant quality_blocked) |
| d128 | measurable_headroom | 54.0159 | 68.6696 | 0.7866 | launch/idle after OPT-117 recapture reject; optimistic weight byte bound still below both Quartz and llama walls |
| d2048 | measurable_headroom | 41.6331 | 67.1855 | 0.6197 | launch/idle plus decode-attention path versus llama flash-attn auto; hybrid_crossover@1024 is shipping and was not reopened |
| d8192 | measurable_headroom | 8.0657 | n/a | n/a | populated-cache KV traffic exists in the byte model, but OPT-120 packed KV was slower and quality_blocked; no matched llama long-context number on this sitting |
| d32768 | measurable_headroom | 1.3519 | n/a | n/a | same as D8192: KV-byte headroom in the bound, no admitted encoding |
| d131040 | insufficient_evidence | n/a | n/a | n/a | n/a |
| opt016_2k | insufficient_evidence | n/a | 3288.2052 | n/a | n/a |

supports_continuing=`False`.
The approved OPT-117–122 ladder is exhausted for quality-admitted encodings. Remaining decode gaps versus llama and versus the optimistic busy bound exceed 2%, so this is not evidence that further improvement is impractical. Continuing requires a new causal mechanism (capture/replay that avoids recapture, or a quality-passing weight encoding), not another random tile/launch. Speculation/MTP/batching are outside this batch and are not authorized here.

## Stop / reopen criteria

Stop this batch when: OPT-117–122 quality-admitted candidates are keep-or-reject complete; no new causal mechanism is identified (random tile/launch is not a reason); speculation/MTP/batching/model-replacement/sparse attention remain out of scope.

Do not stop by claiming: llama.cpp is globally optimal; the kernel-tuning ladder proved a bandwidth bound; a roofline bottleneck proved an optimal algorithm.

Reopen when: a new causal graph-capture/replay design avoids per-token recapture; a quality-passing weight encoding with a different error profile exists; gdn_core becomes DRAM-bound after some other keep; a quality-matched packed-KV encoding is actually faster on populated cache; exclusive sitting completes D131040 and OPT-016 2K for the combined stack.

Practical-stop test: complete traffic account within measurement uncertainty AND plausible remaining scoped savings below 2% request time. Decode P4096/D128/D2048 do not currently meet that joint test versus llama.

Missing for a stronger stop: combined-stack event-union timeline if identity-timeline GPU phase did not run; measured sustainable bandwidth for weight-stream access; exclusive-sitting D131040 and OPT-016 2K; quality-matched llama packed-KV / lower-precision weight baseline; built inspection-HEAD llama supplementary configs.

production_kept=True. claims_throughput=false.
claims_performance_improvement=false. No tok/s delta; this task does not
change the sitting baseline.
