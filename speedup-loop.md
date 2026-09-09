# Speedup loop

Operational method for closing the Quartz versus llama.cpp prefill (and
decode) gap. Prefer ideas and technique inspiration over wholesale copies.
Ledger tasks and `/run-ledger-task-cursor` remain the implementation vehicle.

## Eight steps

1. **Compare for ideas.** Compare our code with DS4 and llama.cpp for ideas
   on how to speed up prefill and potentially also decoding. Prefer ideas
   over copying entire batches of code, although wholesale adoption is
   allowed when appropriate and licensed under `plan.md` provenance.

2. **Oracle.** The steering oracle is the speed of a **4K-token** long-context
   prefill (cold exact-4096 Quartz versus same-protocol llama.cpp), as pinned
   by OPT-021. OPT-016 remains the separate 2K parity gate when that task is
   active.

3. **Add tasks.** Admit stable ledger tasks for the chosen ideas
   (`implementation_ledger.md`), with keep/reject acceptance tied to the
   oracle where applicable.

4. **Accuracy bounds.** It is allowed to relax accuracy requirements, as long
   as we stay within similar boundaries as what ds4 and llama.cpp use
   (recorded in `plan.md` / task contracts). Do not loosen references that a
   task explicitly keeps as byte-exact or frozen.

5. **Implement.** Implement the next eligible task with
   `/run-ledger-task-cursor` (or the autocallable skill under Cursor).

6. **Measure and keep/reject.** Measure the oracle speed after the task. If
   it improved, keep the idea. If not, reject it (revert production as
   required) and continue with the next task.

7. **Instrument the next pick; report tok/s delta.** Use existing
   instrumentation (OPT-020 exclusive attribution, decode timings, and
   same-protocol llama.cpp comparisons) to see which part of prefill/decode
   is taking the longest, or is longer than llama.cpp, and guide the next
   best idea from that data—for example prefer a fast-attention path only
   when attribution shows attention as the next best sink. At the end of
   each completed task, note how much faster we are in tokens/s versus the
   then-current baseline (absolute delta and ×; `0` / unchanged on reject).
   Prefer **live 4K** attribution (oracle length; graphs on) over stale 2K
   shares when ranking the next idea.

8. **Stop.** Stop when Quartz has at least equalled or beaten llama.cpp on
   the oracle (4K prefill under the frozen protocol), unless a later
   instruction changes the stop condition.

## Current state (post OPT-026)

| Item | Value |
|---|---|
| First ladder | OPT-022–OPT-026 **exhausted** |
| Successor oracle Quartz | **1746.71973** tok/s (`fixtures/opt026_fattn_streamk.json`) |
| Same-sitting llama.cpp | **3253.993621** tok/s |
| Gap | ~1.86×; stop condition **not** met |
| Next eligible | **OPT-027** |

### Live post-OPT-026 4K attribution (scout)

Cold exact-4096, graphs=64, rebuilt production objects,
`measurement_utc` 2026-09-09T06:11:25Z, wall 2348.33 ms / **1744.2** tok/s:

| Sink | ms | Share |
|---|---:|---:|
| `attention_core` | 867.45 | 36.9% |
| `ffn_mmq` | 738.27 | 31.4% |
| `gdn_core` | 453.23 | 19.3% |
| `mixer_mmq` | 285.69 | 12.2% |

Evidence: [`evidence/optimization/speedup-loop-post026/`](evidence/optimization/speedup-loop-post026/).

### Second idea ladder (admit order = try order)

| ID | Idea | Primary 4K sink |
|---|---|---|
| OPT-027 | Persistent Ada+ fattn stream-K (SM-count tiles, not `grid.z=2`) | `attention_core` |
| OPT-028 | Q4_K/Q6_K MMQ stream-K + fixup | `ffn_mmq` |
| OPT-029 | Fuse GDN conv + gated output into fused token loop | `gdn_core` |
| OPT-030 | Hopper/Blackwell PDL on prompt compute stream | cross-cutting |
| OPT-031 | 4096-row mixer + GDN prompt graphs | `mixer_mmq` / `gdn_core` |

Each OPT-027–OPT-031 keep/reject: cold exact-4096 mean tok/s strictly greater
than the then-current successor oracle, else revert and retain rejection.
Does not substitute for OPT-016.
