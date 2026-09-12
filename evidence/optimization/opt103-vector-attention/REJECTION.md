# OPT-103 rejection — retain warp_query

Production pin remains `kSelectedDecodeAttentionVec128Path[] = "warp_query"`
and `kSelectedVec128NParts = 16`. `warp_query_gqa6` stays rejected.

Complete 16-layer attention, D128/D2048 five-pair, and P4096 guard did not
jointly win. Keep required ≥0.10 ms/token complete attention with a positive
paired CI plus both decode prefixes.

Independent verdicts:

{
  "independent_verdicts": {
    "warp_query": {
      "kernel_parity_pass": true,
      "model_quality_pass": true,
      "performance_pass": false,
      "production_kept": false,
      "incomplete": false
    },
    "vec128_online": {
      "kernel_parity_pass": true,
      "model_quality_pass": true,
      "performance_pass": false,
      "production_kept": false,
      "incomplete": false
    }
  },
  "selected_path": "warp_query",
  "shipping_unchanged": true,
  "shipping_decode_attention_vec128": "warp_query",
  "production_kept": false,
  "winners": [],
  "status": "retain_warp_query",
  "claims_throughput": false
}

tok/s delta vs OPT-098 P4096 **3046.23 tok/s**: **0** (rejection).

status=measured_reject.
