# OPT-030 rejection

Hopper/Blackwell PDL serialization of ungraphed mixer/GDN/attention prompt
launches was measured on exclusive RTX 5090 and did not strictly beat ordinary
`<<<>>>` stream launches, or the 4K oracle did not strictly beat the OPT-026
Quartz baseline.

- quality sitting mean tok/s: 1734.4137
- OPT-026 baseline: 1746.71973
- A/B winner: pdl
- A/B win: true
- rule: strictly_greater_mean_tok_s (equality is a reject); A/B loss is a reject
- production launches remain ordinary `<<<>>>`; `kSelectedPdlPath` is off
- successor_oracle: false (denominator remains 1746.71973)
- ladder_exhausted: false
- measurement_utc: 2026-09-09T11:34:14Z
