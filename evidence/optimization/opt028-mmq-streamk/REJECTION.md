# OPT-028 rejection

Q4_K/Q6_K MMQ stream-K was measured on exclusive RTX 5090 and did not
strictly beat 2D tiling, or the 4K oracle did not strictly beat the OPT-026
Quartz baseline.

- quality sitting mean tok/s: 1729.0459
- OPT-026 baseline: 1746.71973
- A/B winner: off
- A/B win: false
- rule: strictly_greater_mean_tok_s (equality is a reject); A/B loss is a reject
- production Q4_K/Q6_K MMA remains 2D tiling
- successor_oracle: false (denominator remains 1746.71973)
- ladder_exhausted: false
- measurement_utc: 2026-09-09T08:06:02Z
