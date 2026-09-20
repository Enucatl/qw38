# TASK-22 — Remediate Phase 1 review findings

## Control

- Primary ID: `TASK-22`
- Coupled IDs: `none`
- Dependencies: `TASK-21` (DONE at admission)
- Status: `TODO`
- Ledger acceptance: Correct the review findings, make their validations substantive, resolve the TASK-17 mapping-estimate blocker, and qualify or independently prove the TASK-21 freeze claim.

## Goal and boundaries

Repair the evidence, contracts, and checkers identified by 22 independent task
reviews plus one independent whole-system review. This is a corrective
documentation/evidence increment, not runtime implementation or performance
work. Its outcome must make later workers able to rely on the Phase 1 corpus
without treating a JSON-fence echo as independent proof.

- Constraints: retain the BF16 checkpoint/config as authority; preserve all
  existing unselected alternatives, unknown SKU values, and no-winner status;
  make no Quartz, llama.cpp, GGUF, CUDA-kernel, benchmark, or model-weight change.
- Non-goals: no selected quantization/layout/mapping winner; no NLL, tok/s,
  occupancy, or bandwidth measurement; no vision-encoder expansion; no broad
  prose rewrite; do not remove TASK-01–20 historical draft banners merely for
  consistency.
- Plan impact: `none`.
- Affected interfaces: Phase 1 Markdown contracts and their stdlib checker
  commands only; no production API or binary format is introduced.

## Repository evidence and implementation decisions

The independent review reports are the admission record. Retain a finding only
when the following cited evidence reproduces it; where reviewers disagreed, the
decision below governs implementation.

### Mandatory correctness and contract repairs

1. **TASK-02 / downstream MTP:** either pin authoritative source evidence that
   recognizes this checkpoint's MTP fields/tensors and proves concat order,
   alignment, block type, and KV state, or relabel those exact claims unresolved.
   Do not represent generic MTP defaults as checkpoint-specific proof. Update
   TASK-03/04/06/07/11 consumers only as required by the chosen evidence result.

2. **TASK-03/06/07:** redraw GDN Diagram 6 with its actual prior-state/current
   causal inputs; make forced activation the only unqualified semantic minimum;
   label region-cut totals as an explicit assumed interface accounting; correct
   the output-projection reduction dimension to include K=6144 and compute RoPE
   maximum phase as `max_position_embeddings - 1`.

3. **TASK-05:** run `--check-analysis` once against the authoritative checkpoint
   and record command, exit code, elapsed time, hash, and shard/index identity.
   Revise the dossier contract to state that per-tensor retained evidence is
   absmax/RMS/directional summaries, rather than add unused full per-tensor
   `GlobalStats`. Remove the unused schema-key tuple constants and dead `names`
   accumulator from the two analysis scripts.

4. **TASK-09/10/13/14/15:** specify unselected, parseable zero-point and
   sidecar/view descriptors for `i8_row_asym`, `extract_high`, and `mixed_group`;
   include count/encoding/binding/range information without choosing tile,
   alignment, or bit order. Add omitted MTP norm loads and state reads to the
   decode plan. Add the required nine-row decode-vs-prefill comparison and remove
   superseded numeric locks from the TASK-14 dossier. Redraw TASK-15's diagram so
   it does not imply a selected portable/specialized view.

5. **TASK-16/17 — severe downstream blocker:**
   - Correct TASK-16 occupancy so resident warps cannot exceed `W_max`; model
     register allocation with explicit allocation scope/granularity or a
     SKU-specific allocation function; qualify fusion-footprint inequalities as
     conditional rather than universal.
   - For every one of the 18 TASK-17 mappings, add a keyed record containing work
     citation/partition, HBM citation, access/decomposition, synchronization,
     symbolic `R_t`, `C_cta`, permitted symbolic `T_cta` candidates, occupancy/
     hiding/wave expressions, and decode/prefill mode label. Correct
     `map_mlp_grid_T` synchronization and represent the MTP split's norm/cat and
     GEMM pipeline stages. Do not insert SKU numeric limits, launch choices, or a
     mapping winner.

6. **TASK-18/21:** lock a corpus-agnostic teacher-forced measurement identity:
   tokenizer/version, special-token/loss-mask policy, event-weighted aggregation,
   window/overlap accounting, and reset/continuation rule. For TASK-21, either
   change "checked"/freeze wording to mechanical consistency only, or implement
   an independent semantic verifier deriving dimensions, GDN/KV state, work,
   storage, and node/stage multiplicities from authoritative upstream structures.
   The preferred repair is the independent verifier, because the ledger claims
   those categories were checked.

### Validation and proof-quality repairs

Make each checker reject a focused mutated copy that removes or alters the
relevant content. Parse/rendered-table contracts where their task promised table
semantics, rather than merely searching global substrings. In particular cover:

- TASK-02 equations and MTP clauses; TASK-03 Diagram 6; TASK-04 lifetime catalog
  and ranked candidates; TASK-05 numeric/table provenance; TASK-06 region-cut
  inclusion inventory; TASK-07 sensitive-op rows; TASK-09 metadata descriptors;
  TASK-13 Loads/state-I/O; TASK-14 stage comparison; TASK-15 upstream inherited
  fields; TASK-16 exactly-one JSON fence; TASK-17 all mapping records; TASK-18
  upstream contracts; TASK-19 per-row `HYPOTHESIS`; and TASK-20 rendered backlog
  fields.
- Replace duplicated static upstream arrays in TASK-18 and TASK-21 checkers with
  direct parsing of the relevant authoritative first JSON fence where feasible.
  Keep local constants only for values derived directly from the authority config.
- Correct only current-status metadata that is internally contradictory in
  TASK-12, TASK-14, TASK-19, and TASK-21 dossiers, and update TASK-21's own
  published freeze status. Do not mass-change historical draft banners.

## Invariants

- Primary language-model dimensions, inventory totals, and unselected-alternative
  flags must remain unchanged unless authoritative checkpoint/source evidence
  proves an existing claim wrong.
- All new or changed claims retain an evidence label; performance claims remain
  absent.
- The severe TASK-17 warning remains visible until all 18 complete records and
  their checker coverage are verified.
- Every changed downstream document/checker is reconciled in the same increment.

## Acceptance and validation

- Run each affected checker with the authoritative config and compare its fresh
  JSON to the document fence where that remains the contract.
- Run `python3 scripts/inventory_bf16_checkpoint.py --check-inventory docs/architecture/model-inventory.md` and `python3 scripts/analyze_bf16_tensors.py --check-analysis docs/architecture/bf16-tensor-analysis.md`; record the latter's full-scan evidence.
- Run all focused commands recorded by TASK-02 through TASK-21 that consume a
  changed contract, plus `python3 -m py_compile` for every changed script.
- Add or use temporary mutated document copies proving every listed strengthened
  validation fails; do not commit fixtures unless a checker needs a durable one.
- Run the TASK-21 review checker only after its claimed verification scope and all
  affected corpus contracts agree. Verify no selected flag, open-question count,
  or performance measurement changed.
- Definition of done: every mandatory repair and proof-quality item above has a
  passing focused command and a recorded negative mutation check; TASK-17 has no
  missing mapping estimate; the severe warning can be cleared by the independent
  verifier.

## Performance evidence

N/A — this task corrects study contracts and verification tooling. It makes no
throughput, quality, or hardware-performance claim and must report tok/s delta as
`N/A` with this reason.

## Run record

### Planning

- Agent/model: review coordinator with 22 fresh same-model task auditors and one fresh same-model system auditor
- Outcome: admitted from evidence-backed post-freeze review; TASK-17 mapping-estimate omission is retained as a severe blocker. The system auditor did not independently flag it; its direct dossier/document evidence controls.
- Performance evidence applied: N/A

### Implementation

- Not started.

### Verification

- Not started.

### Final outcome

- Status: not started
- Candidate measured delta: N/A
- Shipping delta: N/A
- Quality result: not required
- Evidence completeness: unavailable until implementation and independent verification
- Tok/s delta: N/A — no meaningful throughput measurement in this documentation/evidence remediation task
- Commit: not created
- Push: not attempted
