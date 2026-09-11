# OPT-086 — Re-evaluate installed Q8 and MMQ paths

Status: **measured**. Authority llama.cpp
`cc83d7b4824f73cfdda4dfbb47ee39804f71b328`, GGUF SHA-256 `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`. Independent keep/revert for Q8
`r1_w4` vs shipping `r2_w2` and MMQ `fma_async` vs shipping `fma_async_x`
tile 128×128. Ladder: kernel parity (OPT-082) → quality (OPT-084 freeze) →
performance. `opt074_coverage_unadmitted` is **not** a blocker.
`opt074_family_admission_required=False`.

## Independent verdicts

### Q8

| Ident | kernel_parity_pass | model_quality_pass | performance_pass | production_kept |
|---|---|---|---|---|
| r1_w4 | True | True | True | True |
| r2_w2 | True | True | False | False |

Decision: **revert**.
selected=`r1_w4`.
shipping_unchanged=False.
claims_throughput=False.

### MMQ

| Ident | kernel_parity_pass | model_quality_pass | performance_pass | production_kept |
|---|---|---|---|---|
| fma_async | True | True | False | False |
| fma_async_x | True | True | True | True |

Decision: **keep**.
selected=`fma_async_x`.
shipping_unchanged=True.
claims_throughput=True.

Top-level claims_throughput=True.
opt074_coverage_unadmitted_blocker=false.

## Kernel parity (OPT-082)

Q8 source=this_sitting_plus_opt082_fixture; host_ok=True;
typed `sum_q` vs `sum_x` rejected=True.
Q8 catalog 20.
MMQ source=this_sitting_plus_opt082_fixture; catalog 38.
Documented Q8 M17 K2048 association misses do not fail kernel admission when
selectors are correct and nonfinite count is 0. Unaligned MMQ is fallback
association, not the candidate kernel.

## Quality (OPT-083/084)

Identity-cached OPT-084 suite with `--quality`. Shipping idents use the
Quartz-vs-baseline freeze. Replacement controls use the same freeze rules.
A known baseline defect does not reject a non-worsening kernel.

## Performance

Q8: complete rotating mixer, 48 GDN + 16 attention groups, OPT-058
invalidation on both sides. X-prefetch held at shipping MMQ until the MMQ
half decides. this_sitting=True.
Control `r1_w4` 7.453 ms vs shipping `r2_w2`
7.504 ms. Paired CI
-0.0824 .. -0.0203 ms.
Engine 867.392 vs 869.183 ms.

MMQ: 64 complete FFNs, tile 128×128, Q8 decision held fixed.
this_sitting=True.
Control `fma_async` 886.348 ms vs shipping
`fma_async_x` 838.197 ms. Paired CI
46.6675 .. 49.6356 ms.
Engine 1414.274 vs 1366.312 ms.

Historical ranking only (not a keep): Q8 [7.4549503999999995, 7.5063936];
MMQ [887.291583, 838.5226258999999]. A keep/revert requires
this sitting's confirmation. Historical OPT-070 report is unmodified.

## tok/s

No OPT-069/080 P/D oracle rerun. Official tok/s delta versus the then-current
shipping baseline is **0** unless a family both changes shipping and this
sitting claims throughput. claims_throughput is true per family only when
that family's shipping ident is `production_kept` on this sitting's measured
evidence.
