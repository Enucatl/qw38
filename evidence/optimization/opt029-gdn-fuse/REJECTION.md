# OPT-029 rejection

Fusing tiled causal convolution and/or gated-output into the warp-column
GDN token loop was measured on exclusive RTX 5090 and did not strictly beat
the split sequence, or the 4K oracle did not strictly beat the OPT-026
Quartz baseline.

- quality sitting mean tok/s: 1725.36658
- OPT-026 baseline: 1746.71973
- A/B winner: off
- A/B win: false
- rule: strictly_greater_mean_tok_s (equality is a reject); A/B loss is a reject
- production GDN core remains split conv + warp-column + gdn_gated_output_rows
- successor_oracle: false (denominator remains 1746.71973)
- ladder_exhausted: false
- measurement_utc: 2026-09-09T10:38:37Z
