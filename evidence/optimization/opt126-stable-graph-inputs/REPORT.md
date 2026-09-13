# OPT-126 — Make decode graph inputs stable across token commits

Status: **prerequisite_verified**. Shipping selector `ffn_only`. Foundation/correctness prerequisite; not a production speed keep. `claims_throughput=false`.

## Design

GDN uses stable physical ping-pong with an explicit committed slot. Dynamic scalars (token, position, frontier, kv_bucket, slot, generation, ping-pong bases) live in session-owned device `DecodeLaunchState` and upload once per token in stream order. Topology is bounded to two graph variants at the 1024 warp_query → vec128_online dispatch change. Shipping `ffn_only` is unchanged; `decode_segments8` is exercised as a diagnostic path.

GDN parity=`stable_pingpong_committed_slot`. Launch-state bytes=`56`. Selector unchanged=`True`.

## Same-math and positions

same_math ok=`True` exact=`True` state_equals=`True` matched_tokens=`8`. positions ok=`True`. isolation ok=`True` cancel=`True` retry=`True` failure=`True` candidate_isolated=`True`. restore ok=`True` stale_rejected=`True` divergent=`True`.

## Upload / indirection / memory

graph_bytes=`14680064` (OPT-117 eight-graph capture was 12582912 bytes; two topologies are expected near 2×). create_ms=`14.4733286`. launch_state_bytes=`56`. launch_state_upload_ms=`0.00360699999`. launch_state_uploads=`2`. topology_recaptures=`0`. indirection_loads_per_consumer=`2`.

Opaque captured-argument byte patching is unused. Recapture happens only when ping-pong bases or session KV identity change, not on ordinary frontier or GDN slot flips.

Verdict `prerequisite_verified`. Reasons: `[]`. Production selector remains `ffn_only`. OPT-127 owns replay admission.
