# OPT-027 rejection

Persistent Ada+ fattn stream-K was measured on exclusive RTX 5090
and did not strictly beat OPT-026 `stream_k` (`grid.z=2`), or the 4K oracle
did not strictly beat the OPT-026 Quartz baseline.

- quality sitting mean tok/s: 1734.68005
- OPT-026 baseline: 1746.71973
- A/B winner: stream_k
- A/B win: false
- rule: strictly_greater_mean_tok_s (equality is a reject); A/B loss is a reject
- production prompt fattn remains OPT-026 stream_k (grid.z=2)
- selected_persistent_fattn_path: off
- successor_oracle: false (denominator remains 1746.71973)
- ladder_exhausted: false
- measurement_utc: 2026-09-09T06:42:08Z
