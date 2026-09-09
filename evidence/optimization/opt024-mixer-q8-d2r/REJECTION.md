# OPT-024 rejection

Aligned-SoA D2R for large mixer Q8_0 GEMMs was measured on exclusive RTX 5090
and did not strictly beat quality MMQ on every timed large shape, or the 4K
oracle did not strictly beat the OPT-023 Quartz baseline.

- quality sitting mean tok/s: 1684.9541
- OPT-023 baseline: 1687.86169
- A/B winner: quality_mma
- A/B win: false
- rule: strictly_greater_mean_tok_s (equality is a reject); A/B loss is a reject
- production large mixer GEMMs remain OPT-022 I=128 J=128 quality MMA
- successor_oracle: false (OPT-025+ denominator remains 1687.86169)
- measurement_utc: 2026-09-09T04:20:55Z
