# Post-098 performance recovery

This batch is the next performance program after OPT-098. It is task design,
not new benchmark evidence. The frozen starting point is the production
combination measured by OPT-098:

| Workload | Quartz | pinned llama.cpp | Quartz latency | llama latency | Gap |
|---|---:|---:|---:|---:|---:|
| P4096 | 3046.23 tok/s | 3170.93 tok/s | 1344.6 ms/prompt | 1291.7 ms/prompt | 52.9 ms/prompt |
| D128 | 37.29 tok/s | 69.13 tok/s | 26.82 ms/token | 14.46 ms/token | 12.35 ms/token |
| D2048 | 35.49 tok/s | 67.72 tok/s | 28.18 ms/token | 14.77 ms/token | 13.41 ms/token |

Prefill is within about 4.1% throughput of parity. Decode remains 1.85x to
1.91x slower and is the primary release blocker. The original OPT-056 +5%
gate and OPT-016 2K gate remain unchanged.

## What the current evidence does and does not establish

OPT-090 repaired Quartz event accounting and ranked the eager diagnostic trace,
but its pinned-llama family fields are all zero and its P4096 attribution is
invalid. Its D2048 Quartz ranking is therefore useful only for ordering Quartz
work: paired Q4 gate/up/GLU about 12.9 ms/token, the bucket containing FFN down
about 7.1, Q8 input projections about 4.7, attention core about 4.4, GDN core
about 4.0, and the smaller output/logits families after that. These nested
families are not additive and cannot be subtracted from the 28.18 ms wall.

The component replays are also non-additive. They show that complete rotating
FFN, GDN, attention and Q8 mixer bodies remain material, but they do not prove
which family explains the Quartz-versus-llama gap. OPT-099 closes this blind
spot before another kernel is promoted.

## Source-backed opportunities

1. **GDN state topology is materially different.** Quartz's selected
   `prepare_recurrence_window` assigns one thread to an output column, serially
   walks all 128 key rows, rereads state for prediction and update, and repeats
   q/k norm work for each of 48 value heads. Pinned llama.cpp
   `gated_delta_net_cuda<128>` stores transposed state, assigns a warp to a
   column, keeps four state values per lane in registers, and performs the two
   reductions with warp shuffles. OPT-077/094 did not test this layout change;
   they retained row-major state and added shared-memory tiled reductions.
2. **Q8/Q6 physical layout can limit memory efficiency.** Quartz consumes raw
   GGUF Q8_0 blocks of 34 bytes and Q6_K blocks of 210 bytes. `../ds4` has a
   production aligned-SoA Q8 path and records 235-245 GB/s versus 150-195 GB/s
   for corresponding raw warp8 kernels. That is cross-model evidence, not a
   Quartz speed claim. A Quartz keep must replace rather than duplicate the
   device representation and must support every decode and prefill consumer so
   the 1.5 GiB reserve remains intact.
3. **The Q4 gap needs a direct implementation comparison.** OPT-089 made
   `integer_q8_late` a large win, while OPT-093's paired-group factoring was a
   measured regression. Pinned llama's MMVQ still differs in lane mapping,
   reduction, Q8_1 semantics and scale/min unpack. Newer llama.cpp commit
   `73ab7599b` also makes Q4_K scale unpack branchless. OPT-102 tests those
   concrete mechanisms against the current late-w4 control; it does not reopen
   the rejected factored candidate.
4. **Attention needs a different parallelization, not the rejected GQA6 CTA.**
   OPT-095 grouped six query heads into one CTA and lost. Pinned llama selects
   its 128-thread online-softmax vector FlashAttention kernel for BF16/F16,
   head width 256 and one query at the measured prefixes. OPT-103 ports only
   that specialization and compares preparation-inclusive complete attention.
5. **Prefill needs only a final-mile win.** The current Q4 MMQ uses a 128x128
   tile with one raw-X stage because two stages exceed the opt-in shared-memory
   limit. A 64x128 tile previously lost without async X, but its smaller
   footprint can hold two raw-X stages. OPT-105 tests that single new schedule,
   not another general tile sweep.

## Ordering and stop rules

Run OPT-099 first. OPT-100 through OPT-105 may proceed after it in ledger order;
GPU measurements remain serialized. Each candidate keeps a callable control,
uses kernel_parity_v1 and the OPT-091 successor full-model quality policy, and
is promoted only by complete-component plus uninstrumented end-to-end evidence.
OPT-106 freezes the admitted combination and reruns the unchanged outcome gates.

For decode candidates, screen with one warmup and three AB/BA rounds. Acceptance
uses 3 warmups and 10 independent paired complete-component rounds; require a
positive 95% interval for control minus candidate and at least 0.10 ms/token
expected saving. Then run five uninstrumented D128 and D2048 32-token pairs;
the point estimate must improve at both prefixes and the one-sided 95% upper
bound on regression must be at most 2%. Run one P4096 pair as a gross guard.

For prompt candidates, require at least 5 ms/P4096 expected saving, 3+10 paired
complete-FFN rounds, five P4096 pairs with a strictly better point estimate and
the same 2% regression bound, plus one D128/D2048 pair as guards. Do not repeat
sampling after an inconclusive result; retain the diagnostic or reject it.

Every keep requires the complete local quality suite with candidate NLL, no new
regression versus the OPT-084/091 production baseline, no nonfinite values,
unchanged public logits/vocabulary, state/checkpoint/cancellation correctness,
and the 128K memory reserve. Same-math transformations require bitwise equality;
changed association order uses the approved independent CPU/dequant parity
envelope. A component reject is a successful task outcome when production is
restored and rejection evidence is retained.

## Shared delivery contract

Each task adds its own pin, iteration contract, fixture, runner, focused native
test, host validator and `evidence/optimization/optNNN-<slug>/REPORT.md` (plus
`REJECTION.md` when appropriate). Raw samples, selected launch identities,
kernel resources, bytes moved, graph identity, hardware/toolchain identity and
proof limits must be retained. Live runs write a new run directory and promote
only that task's evidence.

Do not modify historical fixtures, relax OPT-016/056, change the GGUF
quantization, prune the vocabulary, add speculative decoding, use approximate
attention, add an unaccounted persistent allocation, or claim that ds4 numbers
transfer to Quartz. Use pinned llama revision
`cc83d7b4824f73cfdda4dfbb47ee39804f71b328` for release comparison; current
`../llama.cpp` is source inspiration unless separately identified.
