# OPT-126 — Argument and ownership inventory

Captured by `capture_decode_segment_graph` / `SchedulerGraphs::update_launch_params`. Dynamic scalars live in session-owned device `DecodeLaunchState` and update in stream order. Physical GDN ping-pong bases stay stable; kernels load the committed slot. Topology is bounded to two graph variants (`warp_query` at capture frontier 0, `vec128_online` at 1024). Opaque byte-patching is unused on this path.

GDN parity design: `stable_pingpong_committed_slot`.
Launch-state bytes: `56`.
Topology count: `2`.
Shipping selector: `ffn_only`.

| Input | Consumer | Kind | Topology |
|---|---|---|---|
| `frontier_position` | `attention_rope_kv_span` | `value` | `False` |
| `visible_kv_length` | `attention_decode` | `value` | `False` |
| `gdn_committed_slot` | `gdn_prepare_tiled` | `pointer_index` | `False` |
| `token` | `embedding_row` | `value` | `False` |
| `attention_dispatch` | `vec128_vs_warp_query` | `kernel_identity` | `True` |
| `session_identity` | `scheduler_graphs` | `pointer` | `True` |
| `q8_grouped_descriptors` | `mixer_q8` | `descriptor_lifetime` | `True` |

## Ownership

- Slot 0 convolution/recurrent: session allocation (`gdn_convolution_` / `gdn_recurrent_`).
- Slot 1 convolution/recurrent: workspace candidate buffers, bound not copied.
- `gdn_committed_slot_`: session integer; XOR on successful commit only.
- `DecodeLaunchState*`: session device allocation; host staging is session-owned, not a capture-time temp.
- Grouped-Q8 descriptors: persistent workspace host memory rebound on session/workspace identity change.
- Graph executables: `SchedulerGraphs`; 8 segments × 2 topologies = 16.
