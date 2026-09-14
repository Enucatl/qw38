# OPT-132 — Admit the combined decode stack and publish corrected headroom

Status: **keep**. Shipping `decode_segments8`. Hardware executed.

Control is authenticated post124 combined OPT-118+119 `ffn_only`. Combination is that parent plus OPT-127 `decode_segments8` only. Rejected OPT-130 occupancy selector and OPT-128/131 pins stay excluded. OPT-126+127 is one valid unit. Graph × attention interaction is not applicable because OPT-130 rejected.

## Freeze and identities

rejected_no_leak=`True`; interaction_matrix=`['control', 'combination', 'combination_minus_opt126_127']`; historical post113 and original OPT-056 +5% / OPT-123 mixed-prefill gates remain historical anchors and are not silently rewritten.

## Quality and state

OPT-058 `--quality` invoked=`True`; candidate_nll_measured=`True`; held-out PPL ratio=`1.0`; wikitext PPL ratio=`1.0`. Budgets stay anchored to OPT-116 / post113; they do not ratchet.

## Paired throughput versus post124 (OPT-125 boundaries)

Aggregate keep metric is complete-request tok/s over matched P4096/D128/D2048 with 3 warmups + 10 AB/BA, independently restored. Decode-only is reported separately. Decode p95 uses decode-only latencies. P4096 is a non-target >=0.98 guard.

| Workload | parent request | combination request | geo | CI lower | decode p95 | parent decode-only | combination decode-only | ok |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| P4096 | 2969.3615720000003 | 2964.720069 | 0.9984373113004618 | 0.9977350454866826 | 0.0 | 2969.3615720000003 | 2964.720069 | True |
| D128 | 54.253206639999995 | 58.069026550000004 | 1.070332971405245 | 1.06980933980354 | 0.932604796762568 | 55.25844995 | 59.262087629999996 | True |
| D2048 | 41.70515062 | 44.27479554 | 1.061614642394542 | 1.0607707908190183 | 0.9345729768607498 | 46.74614183 | 50.00062256 | True |

Aggregate CI lower `{'pass': True, 'incomplete': False, 'ci_lower': 1.0327977828262134, 'min_ratio': 0.9969250083071287, 'min_ratio_threshold': 0.95, 'threshold': 1.0, 'log_mean': 0.04206562467473164, 'df': 29, 'critical': 1.699}`; ok=`True`.

## Llama headroom (matched decode-only / prefill)

Pinned llama `cc83d7b4824f73cfdda4dfbb47ee39804f71b328`. D128/D2048 compare decode-only Quartz with the decode oracle. P4096 compares prefill with llama-bench `-n 0`. Published OPT-124 D2048 0.620× mixed complete-request Quartz with decode-only llama and does **not** survive these boundaries.

{
  "d128_decode_only": {
    "delta_tok_s": -9.504246570000006,
    "llama": 68.7663342,
    "quartz": 59.262087629999996,
    "ratio": 0.8617892507930137
  },
  "d2048_decode_only": {
    "delta_tok_s": -17.239003940000003,
    "llama": 67.2396265,
    "quartz": 50.00062256,
    "ratio": 0.7436183863989785
  },
  "opt056_plus_5pct_not_this_gate": true,
  "opt125_published_d2048_0_620_does_not_survive": true,
  "p4096_prefill": {
    "delta_tok_s": -129.07843000000003,
    "llama": 3093.798499,
    "quartz": 2964.720069,
    "ratio": 0.9582783332393103
  }
}

## Long-cache and OPT-016 2K

Long-context resource_blocked=`True`; probes=`['d8192', 'd32768', 'd131040']`. OPT-016 2K resource_blocked=`True`; opt016_gate_passed=`False` (historical OPT-056 +5%, not this keep gate). A capacity calculation was not substituted for live fit.

## Activity versus OPT-125 budget

Method `cuda_event_engine_attribution`; nsys=`False`; proven inactive `0.0`. Leaf gaps remain unobserved, not GPU idle. Component savings are not added. OPT-124 `supports_continuing=false` described an exhausted ladder, not proof that optimization is impossible.

{
  "d128": {
    "device_active_ms_per_token": 0.7650098250001219,
    "intervals": {
      "charged_count": 96,
      "device_active_ms": 9.180117900001463,
      "device_inactive_proven_ms": 0.0,
      "host_device_overlap_ms": 0.0,
      "host_union_ms": 0.0,
      "inactive_method": "not_proven_from_leaf_gaps; CUPTI/Nsight Systems required for hardware inactivity independent of event leaves",
      "interval_coverage": 0.002135857508194442,
      "leaf_gap_is_not_gpu_idle": true,
      "leaf_gap_ms_opt115_label": 4289.709215299999,
      "leaf_gap_relabeled": "unobserved_or_unrecorded_device_work",
      "leaf_kernel_count": 36,
      "never_count_cpu_wait_overlapping_gpu_twice": true,
      "opt115_classify_timeline_not_independent_inactivity_proof": true,
      "overlap_not_summed": true,
      "profiler": "cuda_event_engine_attribution",
      "record_count": 96,
      "tokens": 12,
      "unobserved_ms": 4288.9146120999985,
      "wall_ms": 4298.09473
    },
    "native": {
      "cupti_linked": false,
      "decode_only_tok_s": 59.5612755,
      "instrumented_decode_only_wall_ms": 4298.09473,
      "keep": false,
      "method": "cuda_event_engine_attribution",
      "pool_overflow": false,
      "prefix": 128,
      "profiler_perturbation_ms": 9.57080078,
      "record_count": 96,
      "request_tok_s": 58.3543129,
      "schema_version": 1,
      "stack": "post124_plus_opt127_decode_segments8",
      "task": "OPT-132",
      "tokens": 256,
      "uninstrumented_decode_only_wall_ms": 4288.52393,
      "windows": {
        "early": "0:4",
        "late": "n-4:n",
        "middle": "mid-2:mid+2"
      },
      "workload": "activity-windows"
    },
    "proven_device_inactive_ms": 0.0,
    "record_count": 96,
    "unobserved_ms_per_token": 357.4095510083332,
    "windows": {
      "early": {
        "charged_count": 32,
        "device_active_ms": 2.98138790000003,
        "device_inactive_proven_ms": 0.0,
        "host_device_overlap_ms": 0.0,
        "host_union_ms": 0.0,
        "inactive_method": "not_proven_from_leaf_gaps; CUPTI/Nsight Systems required for hardware inactivity independent of event leaves",
        "interval_coverage": 1.0,
        "leaf_gap_is_not_gpu_idle": true,
        "leaf_gap_ms_opt115_label": 63.00401930000001,
        "leaf_gap_relabeled": "unobserved_or_unrecorded_device_work",
        "leaf_kernel_count": 12,
        "never_count_cpu_wait_overlapping_gpu_twice": true,
        "opt115_classify_timeline_not_independent_inactivity_proof": true,
        "overlap_not_summed": true,
        "profiler": "cuda_event_engine_attribution",
        "record_count": 32,
        "tokens": 4,
        "unobserved_ms": 0.0,
        "wall_ms": 2.98138790000003
      },
      "late": {
        "charged_count": 32,
        "device_active_ms": 3.0727600000018356,
        "device_inactive_proven_ms": 0.0,
        "host_device_overlap_ms": 0.0,
        "host_union_ms": 0.0,
        "inactive_method": "not_proven_from_leaf_gaps; CUPTI/Nsight Systems required for hardware inactivity independent of event leaves",
        "interval_coverage": 1.0,
        "leaf_gap_is_not_gpu_idle": true,
        "leaf_gap_ms_opt115_label": 65.04442999999901,
        "leaf_gap_relabeled": "unobserved_or_unrecorded_device_work",
        "leaf_kernel_count": 12,
        "never_count_cpu_wait_overlapping_gpu_twice": true,
        "opt115_classify_timeline_not_independent_inactivity_proof": true,
        "overlap_not_summed": true,
        "profiler": "cuda_event_engine_attribution",
        "record_count": 32,
        "tokens": 4,
        "unobserved_ms": 0.0,
        "wall_ms": 3.0727600000018356
      },
      "middle": {
        "charged_count": 32,
        "device_active_ms": 3.1259699999995973,
        "device_inactive_proven_ms": 0.0,
        "host_device_overlap_ms": 0.0,
        "host_union_ms": 0.0,
        "inactive_method": "not_proven_from_leaf_gaps; CUPTI/Nsight Systems required for hardware inactivity independent of event

## Corrections versus original OPT-115–124 evidence

- Preserve original OPT-115–124 artifacts; this report links the corrected OPT-125 catalog rather than rewriting those files.
- OPT-123 D8192/D32768 request rates mixed prefill into decode tok/s.
- OPT-124 0.787×/0.620× mixed complete-request Quartz with decode-only llama.

Blocked gates: ['long_context_resource_blocked', 'opt016_2k_resource_blocked', 'nsight_systems_absent'].

Reasons: [].

See `fixtures/opt132_combined_decode.json` and `build/optimization-runs/opt132/`.

