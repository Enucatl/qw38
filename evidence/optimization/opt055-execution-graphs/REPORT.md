# OPT-055 — Capture measured remaining launch overhead

## Claim labels and proof limits

Live exclusive-RTX-5090 measurement of remaining decode/prompt launch gaps after
OPT-054. Keep requires
**stable-address decode and prompt FFN graphs remain the shipping path**,
**layer-segment mixer/core capture stays unpopulated unless idle exceeds noise**,
**graph/eager equality on the same arithmetic path**,
**token change, frontier growth, invalidation, partial tails, and cancellation before publication**,
**node and launch counts plus measured idle/waits including poll versus null**,
**parameter uploads counted in host launch_params without extra device allocation**,
**128K post-graph reserve unchanged**, and
**does not substitute for the 2K llama.cpp parity gate**. Copied denominators
are P 2895.42773, D128 37.5605927, D2048 35.7286987.

## Decision

**no-change** — `reverted`=false;
`keep_sitting_skipped`=true;
selected_execution_graph_path=ffn_only;
winner ffn_only; below_noise=true;
D128 graphs idle 0.0855464935 ms /
wall 25.5044994 ms;
D2048 graphs idle 0.105142593 ms /
wall 27.6636486 ms;
prompt graphs idle 0 ms /
wall 1422.6759 ms;
tok/s sitting skipped.

## Protocol

| Field | Frozen value |
|---|---|
| GGUF | `models/Qwen3.8-27B-Q4_K_M.gguf` SHA-256 `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34` |
| GPU | NVIDIA GeForce RTX 5090, compute capability 12.0, exclusive |
| Quartz image | `qw38-cuda:13.0.2` |
| llama.cpp revision | `cc83d7b4824f73cfdda4dfbb47ee39804f71b328` |
| A/B | FFN graphs versus eager plus poll versus null; layer_segments unpopulated when idle is below noise |
| Cancellation | no publish after poll-8 layer boundary |
| P / D128 / D2048 | OPT-021 / OPT-032 protocols, copied on measured no-change |
| Nsight | not_used |

## Measured sitting

- Device: NVIDIA GeForce RTX 5090 compute 12.0
- measurement_utc: 2026-09-11T01:21:59Z
- extra_workspace_bytes: 0
- decode_graph_count / prompt_graph_count: 64 / 64
- decode_segment_graph_count / prompt_mixer_graph_count: 0 / 0
- decode_node_count / prompt_node_count: 384 / 448
- graph allocated_bytes: 16777216
- D128 graphs wall 25.5044994 ms idle 0.0855464935 ms; eager idle 0.306892395 ms; poll idle 0.36829567 ms
- D2048 graphs wall 27.6636486 ms idle 0.105142593 ms; eager idle 0.385662079 ms; poll idle 0.352926254 ms
- prompt graphs wall 1422.6759 ms idle 0 ms; eager 1424.9436 ms; poll 1428.71667 ms
- P Quartz mean tok/s: 2895.42773 versus OPT-054 2895.42773
- D128 Quartz mean tok/s: 37.5605927 versus OPT-054 37.5605927
- D2048 Quartz mean tok/s: 35.7286987 versus OPT-054 35.7286987
- owns_opt016_parity_gate: false
- substitutes_for_opt016: false
