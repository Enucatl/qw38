# EVAL-003 — Make OPT-046 Q4 decode evidence representative and sampled

Status: pending. Dependencies: OPT-046, EVAL-002. Coupled IDs: none. Primary
target: OPT-046 A/B and P/D evidence.

## Objective

Reduce the wall time and repeated host work in the cooperative Q4_K decode
oracle without weakening the documented final decision. The current A/B
diagnostic performs full host FP64 passes over large synthetic production
shapes, repeats block decoding for original and staged activations, and
compiles the decode oracle separately for each prefix.

## Design

1. Keep full host references for small probes and boundary cases. For large
   gate/up and down shapes, compare deterministic output-row samples: first,
   last, tail, and a fixed pseudo-random selection seeded by the contract.
   Report the exact sampled row IDs and policy in raw evidence and the fixture.
2. Decode each selected Q4_K weight row once and reuse the decoded row for both
   original-BF16 and staged-activation FP64 references. Keep the existing
   original/staged error envelopes; record full versus sampled point counts.
   Permit one explicitly labeled full-reference deep/release run, but do not
   make it the implementation-loop default.
3. Retain deterministic synthetic probes, and add actual representative
   early/middle/late layer captures required by the shared performance
   protocol for production gate/up/down shapes. The model argument must either
   load those captures or be rejected as unused; synthetic-only evidence cannot
   claim capture coverage.
4. Split candidate work into tiers. Smoke runs only small probes; correctness
   screens all candidates with reduced samples and no P/D sitting; acceptance
   performs the frozen 3-warmup/30-sample production A/B for packed and
   numerical/occupancy-eligible survivors, then runs P, D128, and D2048 only
   for the selected survivor. A packed winner continues to skip P/D.
5. Compile prefill once and decode once, then invoke the same binaries for all
   required prefixes. Preserve the pinned GGUF, llama revision, same-sitting
   control policy, OPT-044 budgets, complete/prequant accounting, frequency
   weighting, and 95%/105% cross-workload guards.

## Acceptance

- Large host-reference evidence proves the deterministic sample policy,
  selected rows, decoded-row reuse, and full/sampled point counts; small and
  boundary references remain complete.
- Production A/B evidence contains real early/middle/late captures in addition
  to synthetic probes, or records an explicit bounded failure rather than
  claiming capture coverage.
- Correctness-tier execution is materially shorter than acceptance and never
  launches P/D. Acceptance still records 3 warmups/30 samples where required,
  and P/D run only after a candidate wins the component gate.
- Prefill and decode compile phases occur once per sitting; D128 and D2048
  reuse the decode binary and retain separate prefix results.
- Existing OPT-046 keep/reject predicates, OPT-044 numerical envelopes, and
  retained packed fallback remain unchanged. Any sampling compromise is
  documented as an oracle proof boundary, with an optional deep full-reference
  release check.

## Proof boundary

Sampled large-shape references detect deterministic layout, tail, quantization,
and reduction failures but do not prove every output row. Full small/boundary
coverage and the optional deep run cover that limitation. Performance claims
still require the acceptance tier and complete end-to-end guards.
