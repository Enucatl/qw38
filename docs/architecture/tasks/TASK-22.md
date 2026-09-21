# TASK-22 — Remediate Phase 1 review findings

## Control

- Primary ID: `TASK-22`
- Coupled IDs: `none`
- Dependencies: `TASK-21` (DONE at admission)
- Status: `DONE`
- Ledger acceptance: Correct the review findings, make their validations substantive, resolve the TASK-17 mapping-estimate blocker, and qualify or independently prove the TASK-21 freeze claim.

## Goal and boundaries

Repair the evidence labels, contracts, and stdlib checkers identified by the
post-freeze review. The result is a trustworthy, mechanically checked Phase 1
corpus; it is not a runtime or performance increment.

- Constraints: the BF16 checkpoint/config remains the only model authority;
  preserve every unselected alternative, unknown SKU value, open question, and
  no-winner flag; make no change to `docs/architecture/plan.md`.
- Non-goals: no runtime implementation; no Quartz, llama.cpp/GGML, GGUF payload,
  CUDA-kernel, benchmark, model-weight change, new weight study, NLL, tok/s,
  occupancy, or bandwidth measurement (the required TASK-05 validation-only
  rescan checks the already-published analysis); no selected quantization, layout, mapping, launch,
  tile, artifact, corpus, or hardware winner; no vision expansion; no mass
  removal of TASK-01–20 historical draft banners.
- Plan impact: `none`.
- Affected interfaces: Phase 1 Markdown contracts, JSON summary fences, task
  dossier status metadata, and stdlib checker CLIs only.

## Repository evidence

- `docs/architecture/plan.md:35-52` makes the BF16 checkpoint authoritative and
  excludes runtime/kernel inspection during Phase 1; `plan.md` needs no change.
- `docs/architecture/task_ledger.md:804-853` contains TASK-22 exactly once,
  status TODO at admission, dependent only on DONE TASK-21, with the five
  remediation criteria copied into this dossier.
- `docs/architecture/model-semantics.md:296-316` states checkpoint-specific MTP
  alignment, concat order, full-attention block behavior, and KV state, while
  `scripts/check_model_semantics.py:300-332` proves only config arithmetic and
  flags. The local checkpoint/config establishes MTP presence and tensor shapes,
  not those four behavioral claims.
- `docs/architecture/dataflow.md:294-329` draws Diagram 6 with self-loops rather
  than separate prior-state/current-value producers.
- `docs/architecture/work-and-traffic.md:264-297` calls region-cut the primary
  bound even though only the forced set is a semantic minimum.
- `docs/architecture/numerical-sensitivity.md:171-186` omits the GDN output
  projection reduction width `K=6144` and uses `T_max`, not `T_max-1`, for the
  maximum zero-based RoPE position.
- `scripts/analyze_bf16_tensors.py:156-269` declares seven schema-key tuples that
  are never read; `scripts/inventory_bf16_checkpoint.py:220-231,341-377`
  accumulates `names` but never emits or validates it.
- `docs/architecture/runtime-format-design.md:103-110,152-161` names zero-point,
  sparse sidecar, mixed-width, and view capabilities without a parseable
  descriptor contract; TASK-10 only repeats those names at
  `docs/architecture/model-compiler-plan.md:265-271`.
- `docs/architecture/decode-plan.md:219-239` omits MTP pre-FC/final norm loads and
  does not enumerate all state reads in the stage table.
- `docs/architecture/prefill-plan.md:353-384` provides six semantic-node rows but
  only prose for the required nine stage-kind comparisons. The TASK-14 dossier
  retains superseded locked arrays around lines 311-323 and says
  `decode_schedule_deferred` despite using decode as comparison evidence.
- `docs/architecture/layout-strategy.md:533-554` visually sends several consumers
  only to `specialized`, implying a selected view despite
  `decode_prefill_distinct_views_selected=false`.
- `docs/architecture/cuda-hardware-model.md:222-242` omits a resident-warp limiter
  from `B_SM`; lines 300-330 make register/shared fusion inequalities universal
  and do not define register allocation scope.
- `docs/architecture/cuda-design-space.md:235-260` has one shallow mapping table;
  lines 379-458 state global formulas but do not provide a keyed six-estimate
  record for each of the 18 mappings. `map_mlp_grid_T` incorrectly uses a
  stream/event join although its T partitions are independent.
- `docs/architecture/quantization-validation.md:142-180` lacks a locked tokenizer,
  special-token/loss-mask, event aggregation, window/overlap, and reset contract.
- `scripts/check_quantization_validation.py:74-172` and
  `scripts/check_clean_sheet_review.py:50-190` duplicate upstream ID arrays.
- `scripts/check_performance_validation.py:1259-1424` and
  `scripts/check_clean_sheet_architecture.py:2281-2363` check global substrings,
  not the promised risk/backlog rows.
- `docs/architecture/clean-sheet-review.md:3,79-92,131-158` publishes an
  unverified banner and semantic “checked” language, but its checker compares
  fences/constants and does not independently derive the claimed semantics.
- `docs/architecture/tasks/TASK-{12,14,19,21}.md` each has Control status
  `IN PROGRESS` while its Final outcome and ledger row are DONE.

## Performance evidence

N/A — this is documentation/evidence remediation. It makes no timing,
throughput, quality, bottleneck-ranking, or hardware-performance claim. No
performance-evidence checklist, CUDA/hardware gate, or tok/s measurement applies.

## Implementation decisions

### 1. MTP evidence closure

Choose the unresolved branch. Do not inspect a runtime implementation to rescue
the claim.

- In `model-semantics.md`, retain as OBSERVED only config fields, tensor names,
  shapes, and sharing flags. Mark MTP concat order, token alignment, decoder
  block behavior, and independent KV state `UNKNOWN`. Equations (23)–(24) remain
  a clearly named **conditional analysis model**, not checkpoint-proven semantics.
- In `check_model_semantics.py`, add structured fields
  `mtp_semantics_status="conditional_unverified"` and four false proof flags;
  parse the MTP section and equations so removal, reorder, or loss of the UNKNOWN
  qualification fails.
- In `dataflow.md`, `lifetime-and-state.md`, `work-and-traffic.md`,
  `numerical-sensitivity.md`, and `semantic-graph.md`, and their paired checkers,
  label MTP-only graph/state/work/count results as conditional on TASK-02's model.
  Keep all existing numbers as conditional accounting; do not silently convert
  them to OBSERVED or delete the language-only results.
- Downstream documents changed for other findings (`runtime-format-design.md`,
  `decode-plan.md`, `prefill-plan.md`, `cuda-design-space.md`,
  `quantization-validation.md`, `performance-validation.md`,
  `clean-sheet-architecture.md`, `experiment-backlog.md`, and
  `clean-sheet-review.md`) must cite that inherited conditional status wherever
  they claim complete-map MTP behavior. No new downstream numeric total is added.

### 2. Mathematical and traffic corrections

- Redraw dataflow Diagram 6 with distinct prior-state nodes and current producers:
  prior K/V feed attention before current `k_rope`/`v_full` append; prior C plus
  current `qkv` feed FIR then produce next C; prior S plus current
  `alpha,beta,k_hat,v_lin` produce `o` and next S. The checker parses the Mermaid
  edges inside Diagram 6, not global node names.
- In TASK-06, rename region-cut to `assumed_region_interface_accounting`; only
  the forced set is the unqualified semantic minimum. Preserve totals and add a
  summary status field instead of recomputing them.
- In TASK-07, split the 5120 reduction family from GDN `W_out` and add a
  `gemm_k6144` sensitive-op row for equation (20). Set maximum zero-based RoPE
  phase to `max_position_embeddings-1 = 262143`. Update row counts/partitions
  and all affected summary fences/checker assertions; do not change severity
  without evidence.

### 3. TASK-05 evidence and dead declarations

- In `docs/architecture/tasks/TASK-05.md`, correct the retained-evidence contract:
  per tensor retains absmax, RMS, and directional summaries used to form family
  aggregates; a full per-tensor `GlobalStats` object is not retained.
- Remove only the seven unused schema tuple constants from
  `analyze_bf16_tensors.py` and the unused `names` list/append from
  `inventory_bf16_checkpoint.py`. Keep `GLOBAL_INT_KEYS`, which is used.
- Strengthen `check_analysis` to parse the cited numeric/provenance tables rather
  than accept global substrings. The implementation stage runs one authoritative
  full `--check-analysis` scan and records exit status, elapsed time, index hash,
  analysis-document hash, index metadata total, and all 18 shard identities in
  this dossier. That scan is implementation evidence, not planning work.

### 4. Parseable metadata and schedule/layout repairs

- In TASK-09, add JSON objects for three logical descriptors:
  - `i8_row_asym`: one signed-int8 zero point per logical output row, count
    `d_out`, range `[-128,127]`, bound by row ordinal and referenced by a
    metadata-blob span; its scale remains a separate per-row field.
  - `extract_high`: explicit `count`, BF16-LE value encoding, index-encoding enum
    (`flat_u32` or `flat_u64`) with selected encoding recorded per emitted view,
    logical-flat-index binding, range `0 <= index < numel`, and payload spans.
  - `mixed_group`: one `u8` width code per logical quant group, count equal to
    group count, binding by group ordinal, and allowed values equal to that
    recipe's narrow/wide formats.
  Add a view descriptor with logical tensor ID, view ID/role, backend tag,
  payload/metadata/sidecar offset+length spans, access/consumer binding,
  ordering/tile nullable references, and integrity reference. Alignment, tile,
  view presence, and packed code bit order remain unselected.
- TASK-10 emits and validates those descriptors; TASK-15 consumes the view fields
  without selecting a view. Their checkers parse exact fields, ranges, counts,
  and bindings.
- TASK-13 adds `mtp.pre_fc_norm_embedding`, `mtp.pre_fc_norm_hidden`, and
  `mtp.norm` to the appropriate load sets and makes each stage's state-read set
  explicit. Its checker parses the nine-row stage table.
- TASK-14 adds a rendered nine-row decode-vs-prefill stage table keyed by
  `stage_kind_ids`. Remove stale numeric arrays and `decode_schedule_deferred`
  locks from the dossier; the live document/checker is authoritative for derived
  values. Keep decode as comparison evidence only.
- TASK-15 redraws its Mermaid graph so every consumer points to an unresolved
  view-choice node, then to portable and specialized candidates symmetrically;
  no edge may imply a required specialized view.

### 5. CUDA symbolic model and all 18 mapping records

- Add `B_warp=floor(W_max/W_cta)` to TASK-16 and include it in `B_SM`. Define
  register allocation as SKU function
  `R_alloc_cta=A_reg(R_t,T_cta,G_reg,scope_reg)`; both granularity and scope stay
  UNKNOWN until SKU fill. Use `R_alloc_cta` in `B_reg`.
- Make F11/F12 conditional: `R_f >= max(R1,R2)` or
  `C_f >= max(C1,C2)` only when the constituent live allocations remain live in
  the fused schedule under the same accounting scope. Otherwise the fused
  footprint is re-derived; compiler recomputation/lifetime changes prevent a
  universal inequality.
- Add one `mapping_records` object keyed by all 18 `mapping_ids`. Every record has
  `node_type`, `work {citation, partition}`, `hbm {citation, expression}`,
  `access {layout_objects, decompositions, owner_expression}`,
  `synchronization {class, scope, dependency}`, `resources {R_t, C_cta,
  T_cta_candidates}`, `occupancy {B_warp, B_SM, O, W_need}`,
  `waves {N_grid, N_waves, eta_wave}`, and `mode_fit`. Values are symbolic or
  cited; no SKU number or selected launch is permitted.
- Set `map_mlp_grid_T` synchronization to `sync_none`: disjoint T-row CTAs have
  no producer/consumer edge. For `map_mtp_split_norm_gemm`, encode four stages
  (embedding norm, hidden norm, concat/materialize, FC GEMM) with events from
  both norm producers to concat and concat to GEMM. This remains an unselected
  split hypothesis.
- The TASK-17 checker requires exact key equality with `mapping_ids`, validates
  every nested field and citation, and fails if any one record/estimate is
  missing. The severe warning is cleared only after this checker and a focused
  negative mutation pass.

### 6. Quality identity and freeze claim

- TASK-18 locks a corpus-agnostic identity artifact containing tokenizer
  implementation/version plus hashes of tokenizer assets, exact input IDs,
  explicit special-token IDs, and an explicit boolean loss mask. Artificial
  context-only tokens are masked; a target is scored exactly when its mask is 1.
- Aggregate by events: sum language and MTP negative log-likelihood numerators
  and divide by their scored-event counts; never average document/window means.
  Partition scored targets into non-overlapping ranges; windows may overlap only
  for left context, and each target has exactly one owner window. Reset at every
  document boundary; never continue state across documents. Each window starts
  from reset and reconstructs its declared left context, making overlap and
  continuation reproducible without selecting a corpus.
- `check_quantization_validation.py` parses current first JSON fences from
  TASK-07, TASK-08, and TASK-10 for inherited risk/recipe/profile IDs. Local
  constants remain only for TASK-18-owned methodology fields or config-derived
  values.
- Choose the narrow TASK-21 repair. `clean-sheet-review.md`, its checker, the
  TASK-21 dossier, and TASK-21 ledger result must say the freeze records
  mechanical first-fence/schema/key/count/flag consistency only. Keep the token
  `FROZEN_FOR_COMPARATIVE_REVIEW`, but explicitly state it is not independent
  semantic proof. Replace duplicated upstream arrays with direct first-fence
  parsing. Change the review banner to a final verified-mechanical status; leave
  TASK-01–20 banners untouched.

### 7. Proof-quality and status repairs

- Each affected checker parses the relevant section/table/diagram and rejects
  duplicate IDs, missing rows/fields, wrong cell values, and misplaced labels.
  Substring presence elsewhere is insufficient.
- Add `scripts/check_phase1_review_remediation.py`, a stdlib negative-mutation
  harness. It copies documents to a temporary directory, applies one exact
  mutation per promised contract, invokes the owning checker, and requires a
  nonzero exit. Cases cover TASK-02 equations/MTP; TASK-03 Diagram 6; TASK-04
  lifetime row and ranked candidates; TASK-05 numeric provenance; TASK-06
  region inventory/status; TASK-07 sensitive-op row; TASK-09/10 descriptors;
  TASK-11 conditional MTP field; TASK-13 loads/state I/O; TASK-14 all nine stage
  rows; TASK-15 inherited view fields/diagram; TASK-16 exactly one JSON fence;
  TASK-17 each record plus both sync repairs; TASK-18 upstream contracts and
  identity; TASK-19 per-risk-row `HYPOTHESIS`; TASK-20 every rendered backlog
  field; and TASK-21 mechanical-only freeze wording. No fixtures are committed.
- TASK-19's checker parses the eight risk rows and requires `HYPOTHESIS` in each
  row. TASK-20's checker parses each experiment block as a field/value table and
  deep-compares all rendered fields to the generated experiment object.
- Set only the Control status in TASK-12, TASK-14, TASK-19, and TASK-21 dossiers
  from `IN PROGRESS` to `DONE`. Preserve stage-local historical `unverified`
  run-record text. Update TASK-21's current published review status as above.

### Exact file set

Documents: `docs/architecture/model-semantics.md`, `dataflow.md`,
`lifetime-and-state.md`, `bf16-tensor-analysis.md`, `work-and-traffic.md`,
`numerical-sensitivity.md`, `runtime-format-design.md`, `model-compiler-plan.md`,
`semantic-graph.md`, `decode-plan.md`, `prefill-plan.md`, `layout-strategy.md`,
`cuda-hardware-model.md`, `cuda-design-space.md`,
`quantization-validation.md`, `performance-validation.md`,
`clean-sheet-architecture.md`, `experiment-backlog.md`, and
`clean-sheet-review.md`.

Checkers: the paired `scripts/check_*.py` for TASK-02/03/04/06/07/09/10/11/
13/14/15/16/17/18/19/20/21, `scripts/analyze_bf16_tensors.py`,
`scripts/inventory_bf16_checkpoint.py`, and new
`scripts/check_phase1_review_remediation.py`.

Dossiers/ledger: `docs/architecture/tasks/TASK-05.md`, `TASK-12.md`,
`TASK-14.md`, `TASK-19.md`, `TASK-21.md`, this dossier, and
`docs/architecture/task_ledger.md`. No other file is in scope.

### Invariants and rejected alternatives

- Primary language dimensions, inventory totals, language-only results, and all
  selected/open flags remain unchanged unless the authoritative config/checkpoint
  proves an arithmetic error.
- MTP numeric rows remain available only as conditional accounting; they are not
  silently promoted to checkpoint truth.
- The 18 mapping IDs and all open mapping/layout/SKU decisions remain unchanged.
- Rejected: runtime/source inspection to prove MTP semantics (outside boundary);
  a new independent TASK-21 semantic engine (unnecessary after narrowing the
  claim); numeric SKU/launch choices; durable mutated fixtures; broad prose or
  banner cleanup.
- Discovered ledger work: `none`.
- Unresolved decisions: `none`.

## Acceptance and validation

Acceptance requires all semantic repairs above, deep-equal first JSON fences,
and every negative mutation case to pass.

- Focused commands:

```sh
python3 scripts/check_model_semantics.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --semantics docs/architecture/model-semantics.md
python3 scripts/check_dataflow.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --dataflow docs/architecture/dataflow.md
python3 scripts/check_lifetime_and_state.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --lifetime docs/architecture/lifetime-and-state.md
python3 scripts/check_work_and_traffic.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --work-traffic docs/architecture/work-and-traffic.md
python3 scripts/check_numerical_sensitivity.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --numerical-sensitivity docs/architecture/numerical-sensitivity.md
python3 scripts/check_runtime_format_design.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --runtime-format-design docs/architecture/runtime-format-design.md
python3 scripts/check_model_compiler_plan.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --model-compiler-plan docs/architecture/model-compiler-plan.md
python3 scripts/check_semantic_graph.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --semantic-graph docs/architecture/semantic-graph.md
python3 scripts/check_decode_plan.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --decode-plan docs/architecture/decode-plan.md
python3 scripts/check_prefill_plan.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --prefill-plan docs/architecture/prefill-plan.md
python3 scripts/check_layout_strategy.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --layout-strategy docs/architecture/layout-strategy.md
python3 scripts/check_cuda_hardware_model.py --check docs/architecture/cuda-hardware-model.md
python3 scripts/check_cuda_design_space.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --cuda-design-space docs/architecture/cuda-design-space.md
python3 scripts/check_quantization_validation.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --quantization-validation docs/architecture/quantization-validation.md
python3 scripts/check_performance_validation.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --performance-validation docs/architecture/performance-validation.md
python3 scripts/check_clean_sheet_architecture.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --architecture docs/architecture/clean-sheet-architecture.md --experiment-backlog docs/architecture/experiment-backlog.md
python3 scripts/check_clean_sheet_review.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --review docs/architecture/clean-sheet-review.md
python3 scripts/check_phase1_review_remediation.py --config .cache/authorities/qwen3.8-27b-transformers/config.json
```

- Checkpoint evidence commands (implementation stage only):

```sh
python3 scripts/inventory_bf16_checkpoint.py --checkpoint .cache/authorities/qwen3.8-27b-transformers --check-inventory docs/architecture/model-inventory.md
/usr/bin/time -f 'elapsed_seconds=%e exit_status=%x' python3 scripts/analyze_bf16_tensors.py --checkpoint .cache/authorities/qwen3.8-27b-transformers --check-analysis docs/architecture/bf16-tensor-analysis.md
sha256sum .cache/authorities/qwen3.8-27b-transformers/model.safetensors.index.json docs/architecture/bf16-tensor-analysis.md
```

- Repository-wide commands:

```sh
python3 -m compileall -q scripts
git diff --check
git diff --exit-code -- docs/architecture/plan.md
```

- Native/CUDA/hardware gates: not applicable; no runtime, kernel, GPU, or
  benchmark work is permitted.
- Definition of done: all 18 mapping records are complete and mutation-tested;
  TASK-17's severe warning is cleared; MTP uncertainty is explicit; TASK-21's
  freeze is accurately mechanical; full analysis evidence is recorded; no
  winner/open-question/performance value changes; all commands above pass.

## Run record

### Planning

- Agent/model: `gpt-5.6-sol`; effort `medium`.
- UTC/time/tokens/cost: elapsed `563.331s`; input `2946583`; cached input
  `2766464`; output `27841`; reasoning output `6679`; final cumulative total
  `2974424`; cost unavailable. Telemetry record: session
  `01a0c08d-1b8d-76f3-878c-bbac04a6f456`, source
  `/home/user/.codex/sessions/2026/09/20/rollout-2026-09-20T20-41-08-01a0c08d-1b8d-76f3-878c-bbac04a6f456.jsonl`.
- Outcome: existing dossier validated and amended into a decision-complete guide;
  MTP unresolved branch and mechanical-only TASK-21 freeze selected; coupled IDs
  none; TASK-22 marked IN PROGRESS; no implementation performed.
- Performance evidence applied: N/A — no performance claim.
- Commands: all exited 0 unless the command was an informational `rg` with no
  matching optional files. Exact planning commands:

```sh
sed -n '1,240p' /home/user/qw38/.agents/skills/run-ledger-task-codex/SKILL.md
sed -n '1,260p' /home/user/qw38/.agents/skills/run-ledger-task-codex/references/task-dossier-template.md
sed -n '1,320p' docs/architecture/tasks/TASK-22.md
sed -n '1,360p' docs/architecture/plan.md
sed -n '1,360p' docs/architecture/task_ledger.md
git status --short --branch
rg --files | sed -n '1,260p'
rg -n -C 12 '^## TASK-22|^\*\*Status:\*\*|^\*\*Depends on:\*\*|^\*\*Completion criteria:\*\*|^## Ledger history' docs/architecture/task_ledger.md | tail -n 180
rg -n 'TASK-22|Phase 1 review|mapping-estimate|JSON-fence echo|independent review' docs scripts
rg -n 'FROZEN_FOR_COMPARATIVE_REVIEW|DRAFT|freeze|checked|severe|mapping estimate|mapping_estimate' docs/architecture scripts
for f in scripts/check_*.py; do printf '%s ' "$f"; rg -c '```json|json fence|JSON fence|first.*fence|extract.*json' "$f" || true; done
rg --files -g 'pyproject.toml' -g 'uv.lock' -g 'requirements*.txt' -g 'Makefile' -g 'justfile' -g 'tox.ini'
sed -n '804,900p' docs/architecture/task_ledger.md
tail -n 80 docs/architecture/task_ledger.md
rg -n '^## TASK-22' docs/architecture/task_ledger.md
wc -l docs/architecture/task_ledger.md
git log -5 --oneline -- docs/architecture/task_ledger.md docs/architecture/tasks/TASK-22.md
git show --stat --oneline 8dca537
git show --format=fuller --no-ext-diff 8dca537 -- docs/architecture/tasks/TASK-22.md docs/architecture/task_ledger.md | sed -n '1,300p'
rg -n 'SCHEMA_KEYS|TOP_KEYS|GLOBAL_STATS_KEYS|FAMILY_KEYS|LAYER_COMPARISON|DIRECTIONAL_KEYS|names\s*[:=]|names\.append' scripts/inventory_bf16_checkpoint.py scripts/analyze_bf16_tensors.py
rg -n '\bnames\b' scripts/analyze_bf16_tensors.py scripts/inventory_bf16_checkpoint.py
rg -n 'BANNER|unverified|verified' scripts/check_clean_sheet_review.py docs/architecture/clean-sheet-review.md docs/architecture/tasks/TASK-{12,14,19,21}.md | head -n 220
rg -n -C 3 'i8_row_asym|extract_high|mixed_group' docs/architecture/quantization-design-space.md scripts/check_quantization_design_space.py docs/architecture/model-compiler-plan.md
rg -n -i 'mtp|multi.token|multi token|multi-token' docs/architecture/{model-semantics,dataflow,lifetime-and-state,work-and-traffic,numerical-sensitivity,semantic-graph,decode-plan,prefill-plan}.md | head -n 260
rg -n 'Diagram 6|K=|6144|max_position|max phase|region.cut|forced activation|prior|current' docs/architecture/{dataflow,work-and-traffic,numerical-sensitivity}.md scripts/{check_dataflow,check_work_and_traffic,check_numerical_sensitivity}.py
rg -n 'GlobalStats|SCHEMA|KEYS|names|per.tensor|absmax|direction' scripts/analyze_bf16_tensors.py docs/architecture/{bf16-tensor-analysis.md,tasks/TASK-05.md} | head -n 300
rg -n 'i8_row_asym|extract_high|mixed_group|zero.point|sidecar|view descriptor|descriptor' docs/architecture/{runtime-format-design,model-compiler-plan,decode-plan,layout-strategy}.md scripts/{check_runtime_format_design,check_model_compiler_plan,check_decode_plan,check_layout_strategy}.py | head -n 360
rg -n 'MTP|norm|Loads|state|prefill|comparison|nine|9' docs/architecture/{decode-plan,prefill-plan}.md scripts/{check_decode_plan,check_prefill_plan}.py | head -n 360
rg -n 'W_max|resident warp|register|allocation|fusion|map_mlp_grid_T|synchron|R_t|C_cta|T_cta|occup|hiding|wave|18|mapping' docs/architecture/{cuda-hardware-model,cuda-design-space}.md scripts/{check_cuda_hardware_model,check_cuda_design_space}.py | head -n 440
rg -n 'teacher.forced|tokenizer|special.token|loss.mask|event.weight|window|overlap|reset|continuation|corpus' docs/architecture/quantization-validation.md scripts/check_quantization_validation.py | head -n 300
for f in docs/architecture/tasks/TASK-{12,14,19,21}.md; do echo "$f"; rg -n '^#|Status:|Final outcome|Draft conclusions|Verification|DONE|IN PROGRESS|TODO|unverified|verified' "$f" | tail -n 45; done
nl -ba docs/architecture/model-semantics.md | sed -n '1,90p;270,430p'
nl -ba scripts/check_model_semantics.py | sed -n '1,180p;300,650p'
nl -ba docs/architecture/dataflow.md | sed -n '150,270p;294,335p'
rg -n 'Diagram 6|diagram_6|gdn|prior|state' docs/architecture/dataflow.md scripts/check_dataflow.py | head -n 180
rg -n 'MTP|catalog|lifetime|ranked|candidate' docs/architecture/lifetime-and-state.md scripts/check_lifetime_and_state.py | head -n 220
nl -ba docs/architecture/work-and-traffic.md | sed -n '90,135p;255,315p'
nl -ba docs/architecture/numerical-sensitivity.md | sed -n '100,220p'
nl -ba scripts/analyze_bf16_tensors.py | sed -n '140,290p;1030,1160p;1420,1490p'
nl -ba scripts/inventory_bf16_checkpoint.py | sed -n '180,500p'
nl -ba docs/architecture/runtime-format-design.md | sed -n '80,210p;300,430p'
rg -n 'recipe|representation|metadata|capabil|consumer|check_|substring|table' scripts/check_runtime_format_design.py | head -n 260
rg -n 'i8_row_asym|extract_high|mixed_group|zero|sidecar|view|descriptor' docs/architecture/model-compiler-plan.md scripts/check_model_compiler_plan.py
nl -ba docs/architecture/decode-plan.md | sed -n '130,260p'
nl -ba docs/architecture/prefill-plan.md | sed -n '175,225p;320,390p'
rg -n 'portable|specialized|view|mermaid|flowchart' docs/architecture/layout-strategy.md | head -n 200
nl -ba docs/architecture/cuda-hardware-model.md | sed -n '220,380p;420,540p'
rg -n 'W_max|resident|register|alloc|fusion|footprint|JSON_FENCE|check_' scripts/check_cuda_hardware_model.py | head -n 300
nl -ba docs/architecture/cuda-design-space.md | sed -n '70,485p'
rg -n 'MAPPING_IDS|map_mlp_grid_T|mapping|synchron|R_t|C_cta|T_cta|occup|wave|mtp' scripts/check_cuda_design_space.py | head -n 400
rg -n 'six|estimate|mapping|map_mlp_grid_T|R_t|C_cta|T_cta|sync|MTP' docs/architecture/tasks/TASK-17.md | head -n 320
nl -ba docs/architecture/quantization-validation.md | sed -n '80,260p'
rg -n 'teacher|tokenizer|special|mask|event|window|overlap|reset|continu|upstream|TASK|_IDS|first_json' scripts/check_quantization_validation.py | head -n 360
nl -ba docs/architecture/clean-sheet-review.md | sed -n '1,240p'
nl -ba scripts/check_clean_sheet_review.py | sed -n '1,220p;1000,1130p;1500,1750p'
rg -n 'backlog|table|row|HYPOTHESIS|first_json|substring|check_' scripts/check_clean_sheet_architecture.py | head -n 360
rg -n 'HYPOTHESIS|risk|table|row|substring|check_' scripts/check_performance_validation.py | head -n 320
rg -n 'decode_schedule_deferred|n_stage_kinds_compared|nine|9|comparison|locked.*[0-9]|[0-9].*lock' docs/architecture/tasks/TASK-14.md | head -n 260
nl -ba docs/architecture/tasks/TASK-05.md | sed -n '220,275p;430,470p;6310,6360p'
nl -ba docs/architecture/experiment-backlog.md | sed -n '70,180p'
nl -ba docs/architecture/performance-validation.md | sed -n '300,370p'
git diff --check && git diff --name-only && rg -n '^## TASK-22|^\*\*Status:\*\*|^\*\*Depends on:\*\*' docs/architecture/task_ledger.md | tail -n 8 && rg -n 'Coupled IDs|Status:|Unresolved decisions|Agent/model|UTC/time/tokens/cost' docs/architecture/tasks/TASK-22.md && git diff --exit-code -- docs/architecture/plan.md && git status --short --branch
```

No executable checker, checkpoint scan, benchmark, or model payload command was
run during planning.

### Implementation

- Attempt 1 — agent/model: `gpt-5.6-sol`; effort `low`.
- Changes present when the worker stopped: the dossier-scoped Phase 1
  documents, checkers, task metadata, and new
  `scripts/check_phase1_review_remediation.py`; no commit or push.
- Commands: the worker terminated before returning its command log; exact
  outcomes unavailable. `git diff --check` passed after termination.
- UTC/time/tokens/cost: elapsed `5476.786s`; input `18477760`; cached input
  `18273664`; output `44243`; reasoning output `5416`; final cumulative total
  `18522003`; cost unavailable. Telemetry record: session
  `01a0c097-06b0-7fa1-927b-d3be89fa5afd`, source
  `/home/user/.codex/sessions/2026/09/20/rollout-2026-09-20T20-51-58-01a0c097-06b0-7fa1-927b-d3be89fa5afd.jsonl`.
- Result: interrupted by `usage_limit_exceeded`; recovery condition was quota
  reset, now satisfied. A fresh implementation attempt is required before
  documentation or verification.
- Attempt 2 — agent/model: `gpt-5.6-sol`; effort `low` (historical attempt;
  future stages use the active `gpt-5.6-luna` high model per user direction).
- Changes and checks reported before interruption: 39 modified files plus new
  `scripts/check_phase1_review_remediation.py`; all 17 focused acceptance
  checkers passed; the negative-mutation harness passed all cases, including
  18 TASK-17 mappings and both synchronization repairs; checkpoint inventory
  passed; `git diff --check` passed; `docs/architecture/plan.md` unchanged.
- Checkpoint scan: interrupted during shard 17 of 18 after `4370.18s`; it did
  not complete. Reported hashes were index
  `77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df` and
  analysis document
  `41ce1b4585acf6b33afd6f023d3baf32815ee80b51ca392159819945605ef25d`.
- Commands: the worker was interrupted before appending its command log; the
  outcomes above are its partial report and remain implementation evidence,
  not verifier conclusions.
- UTC/time/tokens/cost: elapsed `4438.697s`; input `976229`; cached input
  `914176`; output `3253`; reasoning output `310`; final cumulative total
  `979482`; cost unavailable. Telemetry record: session
  `01a0c28c-e150-7d42-b6fc-b646e6a5fdc5`, source
  `/home/user/.codex/sessions/2026/09/21/rollout-2026-09-21T06-00-07-01a0c28c-e150-7d42-b6fc-b646e6a5fdc5.jsonl`.
- Result: intentionally interrupted after the worker was stuck in the long
  full-scan command; implementation remains incomplete and unverified.

- Attempt 3 — recovery agent; active model/configuration; no commit or push.
- Changed paths present for this attempt (exact worktree paths; `plan.md` is
  unchanged):

```text
docs/architecture/clean-sheet-architecture.md
docs/architecture/clean-sheet-review.md
docs/architecture/cuda-design-space.md
docs/architecture/cuda-hardware-model.md
docs/architecture/dataflow.md
docs/architecture/decode-plan.md
docs/architecture/experiment-backlog.md
docs/architecture/layout-strategy.md
docs/architecture/lifetime-and-state.md
docs/architecture/model-compiler-plan.md
docs/architecture/model-semantics.md
docs/architecture/numerical-sensitivity.md
docs/architecture/performance-validation.md
docs/architecture/prefill-plan.md
docs/architecture/quantization-validation.md
docs/architecture/runtime-format-design.md
docs/architecture/semantic-graph.md
docs/architecture/task_ledger.md
docs/architecture/tasks/TASK-05.md
docs/architecture/tasks/TASK-12.md
docs/architecture/tasks/TASK-14.md
docs/architecture/tasks/TASK-19.md
docs/architecture/tasks/TASK-21.md
docs/architecture/tasks/TASK-22.md
docs/architecture/work-and-traffic.md
scripts/analyze_bf16_tensors.py
scripts/check_clean_sheet_architecture.py
scripts/check_clean_sheet_review.py
scripts/check_cuda_design_space.py
scripts/check_cuda_hardware_model.py
scripts/check_dataflow.py
scripts/check_layout_strategy.py
scripts/check_lifetime_and_state.py
scripts/check_model_compiler_plan.py
scripts/check_model_semantics.py
scripts/check_numerical_sensitivity.py
scripts/check_quantization_validation.py
scripts/check_runtime_format_design.py
scripts/check_semantic_graph.py
scripts/check_work_and_traffic.py
scripts/inventory_bf16_checkpoint.py
scripts/check_phase1_review_remediation.py
```

- Exact non-scan focused command outcomes (all exit `0`; the remediation
  harness reported `phase1 review remediation negative mutations: PASS` and
  every listed mutation as rejected by its owning checker):

```text
python3 scripts/check_model_semantics.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --semantics docs/architecture/model-semantics.md — exit 0
python3 scripts/check_dataflow.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --dataflow docs/architecture/dataflow.md — exit 0
python3 scripts/check_lifetime_and_state.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --lifetime docs/architecture/lifetime-and-state.md — exit 0
python3 scripts/check_work_and_traffic.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --work-traffic docs/architecture/work-and-traffic.md — exit 0
python3 scripts/check_numerical_sensitivity.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --numerical-sensitivity docs/architecture/numerical-sensitivity.md — exit 0
python3 scripts/check_runtime_format_design.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --runtime-format-design docs/architecture/runtime-format-design.md — exit 0
python3 scripts/check_model_compiler_plan.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --model-compiler-plan docs/architecture/model-compiler-plan.md — exit 0
python3 scripts/check_semantic_graph.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --semantic-graph docs/architecture/semantic-graph.md — exit 0
python3 scripts/check_decode_plan.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --decode-plan docs/architecture/decode-plan.md — exit 0
python3 scripts/check_prefill_plan.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --prefill-plan docs/architecture/prefill-plan.md — exit 0
python3 scripts/check_layout_strategy.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --layout-strategy docs/architecture/layout-strategy.md — exit 0
python3 scripts/check_cuda_hardware_model.py --check docs/architecture/cuda-hardware-model.md — exit 0
python3 scripts/check_cuda_design_space.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --cuda-design-space docs/architecture/cuda-design-space.md — exit 0
python3 scripts/check_quantization_validation.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --quantization-validation docs/architecture/quantization-validation.md — exit 0
python3 scripts/check_performance_validation.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --performance-validation docs/architecture/performance-validation.md — exit 0
python3 scripts/check_clean_sheet_architecture.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --architecture docs/architecture/clean-sheet-architecture.md --experiment-backlog docs/architecture/experiment-backlog.md — exit 0
python3 scripts/check_clean_sheet_review.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --review docs/architecture/clean-sheet-review.md — exit 0
python3 scripts/check_phase1_review_remediation.py --config .cache/authorities/qwen3.8-27b-transformers/config.json — exit 0
python3 scripts/inventory_bf16_checkpoint.py --checkpoint .cache/authorities/qwen3.8-27b-transformers --check-inventory docs/architecture/model-inventory.md — exit 0
python3 -m compileall -q scripts — exit 0
git diff --check — exit 0
git diff --exit-code -- docs/architecture/plan.md — exit 0
```

- Exact BF16 validation command (single ground-truth run for TASK-22; not to
  be rerun in later stages):

```text
/usr/bin/time -f 'elapsed_seconds=%e exit_status=%x' python3 scripts/analyze_bf16_tensors.py --checkpoint .cache/authorities/qwen3.8-27b-transformers --check-analysis docs/architecture/bf16-tensor-analysis.md
exit 0
elapsed_seconds=4516.47
processed 1199/1199 tensors across model-00001-of-00018.safetensors through model-00018-of-00018.safetensors
```

- Exact hash command and output:

```text
sha256sum .cache/authorities/qwen3.8-27b-transformers/model.safetensors.index.json docs/architecture/bf16-tensor-analysis.md
77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df  .cache/authorities/qwen3.8-27b-transformers/model.safetensors.index.json
41ce1b4585acf6b33afd6f023d3baf32815ee80b51ca392159819945605ef25d  docs/architecture/bf16-tensor-analysis.md
exit 0
```

- Result: implementation and required implementation-stage validation are
  complete. No verifier verdict was produced. No scan will be rerun.
- UTC/time/tokens/cost: agent elapsed `4848.795s`; model `gpt-5.6-luna`;
  effort `high`; input `1239640`; cached input `1155328`; output `9691`;
  reasoning output `3942`; final cumulative total `1249331`; cost unavailable.
  Telemetry record: session `01a0c2d1-cdaf-71e2-a0c7-71c0a0a9aade`, source
  `/home/user/.codex/sessions/2026/09/21/rollout-2026-09-21T07-15-24-01a0c2d1-cdaf-71e2-a0c7-71c0a0a9aade.jsonl`.
- Final metadata-only recheck after updating this run record:
  `git diff --check` exit `0`; `git diff --exit-code --
  docs/architecture/plan.md` exit `0`; `python3 -m compileall -q scripts`
  exit `0`. No implementation file changed after the recorded scan.

### Documentation

- Agent/model: `gpt-5.6-luna`; effort `high`; inherited active model/configuration
  with no override.
- Changes and evidence: reviewed the completed implementation diff against this
  dossier and the ledger. No implementation, runtime, benchmark, model-weight,
  plan, or semantic acceptance claim was changed. No link repair was required.
  This stage appended this Documentation run record only.
- Exact documentation-stage commands and outcomes:

```text
git diff --check
diff_check_exit=0

git diff --exit-code -- docs/architecture/plan.md
plan_diff_exit=0

git diff --name-only
exit 0; 41 tracked changed paths listed below

git ls-files --others --exclude-standard
exit 0; scripts/check_phase1_review_remediation.py

rg -n '^## TASK-22' docs/architecture/task_ledger.md
exit 0; one match at line 804

rg -n '^\- Status:|^\*\*Status:\*\*' docs/architecture/tasks/TASK-{05,12,14,19,21,22}.md docs/architecture/task_ledger.md
exit 0; TASK-12/14/19/21 Control and Final outcome statuses agree at DONE;
TASK-22 remains IN PROGRESS; no ledger status was advanced.
```

- Changed paths reviewed in the implementation handoff (exact worktree paths;
  the documentation stage itself changed only this dossier):

```text
docs/architecture/clean-sheet-architecture.md
docs/architecture/clean-sheet-review.md
docs/architecture/cuda-design-space.md
docs/architecture/cuda-hardware-model.md
docs/architecture/dataflow.md
docs/architecture/decode-plan.md
docs/architecture/experiment-backlog.md
docs/architecture/layout-strategy.md
docs/architecture/lifetime-and-state.md
docs/architecture/model-compiler-plan.md
docs/architecture/model-semantics.md
docs/architecture/numerical-sensitivity.md
docs/architecture/performance-validation.md
docs/architecture/prefill-plan.md
docs/architecture/quantization-validation.md
docs/architecture/runtime-format-design.md
docs/architecture/semantic-graph.md
docs/architecture/task_ledger.md
docs/architecture/tasks/TASK-05.md
docs/architecture/tasks/TASK-12.md
docs/architecture/tasks/TASK-14.md
docs/architecture/tasks/TASK-19.md
docs/architecture/tasks/TASK-21.md
docs/architecture/tasks/TASK-22.md
docs/architecture/work-and-traffic.md
scripts/analyze_bf16_tensors.py
scripts/check_clean_sheet_architecture.py
scripts/check_clean_sheet_review.py
scripts/check_cuda_design_space.py
scripts/check_cuda_hardware_model.py
scripts/check_dataflow.py
scripts/check_layout_strategy.py
scripts/check_lifetime_and_state.py
scripts/check_model_compiler_plan.py
scripts/check_model_semantics.py
scripts/check_numerical_sensitivity.py
scripts/check_phase1_review_remediation.py
scripts/check_quantization_validation.py
scripts/check_runtime_format_design.py
scripts/check_semantic_graph.py
scripts/check_work_and_traffic.py
scripts/inventory_bf16_checkpoint.py
```

- Draft conclusions: `unverified`; this stage supplies no verifier verdict.
- Remaining unverified items: the implementation-stage checker and single BF16
  scan records have not received independent verification; TASK-22 remains
  `IN PROGRESS`. The recorded BF16 scan is ground truth and was not rerun.
  MTP behavioral semantics remain explicitly `conditional_unverified`, and
  historical TASK-01–20 draft banners remain where the dossier requires them.
- Unresolved design decisions: none recorded. The independent verification gate
  remains pending; no winner, performance claim, or open question was closed.
- UTC/time/tokens/cost: elapsed `315.415s`; input `866592`; cached input
  `740352`; output `8382`; reasoning output `4703`; final cumulative total
  `874974`; cost unavailable. Telemetry record: session
  `01a0c31c-7711-7e42-9609-00afef749b7a`, source
  `/home/user/.codex/sessions/2026/09/21/rollout-2026-09-21T08-36-57-01a0c31c-7711-7e42-9609-00afef749b7a.jsonl`.

### Verification

- Attempt: independent integration verification, 2026-09-21 UTC.
- Agent/model: `gpt-5.6-luna`; effort `high`; inherited active
  model/configuration with no override.
- Independent diff review: completed against this dossier, `plan.md`, the ledger, and the complete shared-worktree diff. Changed paths were 41 tracked paths plus the untracked remediation harness, all within the exact file set above. No runtime, kernel, benchmark, payload, or plan path was changed.

#### Focused and repository-wide commands

The following exact commands were run independently; every command exited 0.
The one initial local typo using `qwen3.8-27B` was corrected and is not a repository result.

```text
python3 scripts/check_model_semantics.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --semantics docs/architecture/model-semantics.md
python3 scripts/check_dataflow.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --dataflow docs/architecture/dataflow.md
python3 scripts/check_lifetime_and_state.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --lifetime docs/architecture/lifetime-and-state.md
python3 scripts/check_work_and_traffic.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --work-traffic docs/architecture/work-and-traffic.md
python3 scripts/check_numerical_sensitivity.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --numerical-sensitivity docs/architecture/numerical-sensitivity.md
python3 scripts/check_runtime_format_design.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --runtime-format-design docs/architecture/runtime-format-design.md
python3 scripts/check_model_compiler_plan.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --model-compiler-plan docs/architecture/model-compiler-plan.md
python3 scripts/check_semantic_graph.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --semantic-graph docs/architecture/semantic-graph.md
python3 scripts/check_decode_plan.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --decode-plan docs/architecture/decode-plan.md
python3 scripts/check_prefill_plan.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --prefill-plan docs/architecture/prefill-plan.md
python3 scripts/check_layout_strategy.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --layout-strategy docs/architecture/layout-strategy.md
python3 scripts/check_cuda_hardware_model.py --check docs/architecture/cuda-hardware-model.md
python3 scripts/check_cuda_design_space.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --cuda-design-space docs/architecture/cuda-design-space.md
python3 scripts/check_quantization_validation.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --quantization-validation docs/architecture/quantization-validation.md
python3 scripts/check_performance_validation.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --performance-validation docs/architecture/performance-validation.md
python3 scripts/check_clean_sheet_architecture.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --architecture docs/architecture/clean-sheet-architecture.md --experiment-backlog docs/architecture/experiment-backlog.md
python3 scripts/check_clean_sheet_review.py --config .cache/authorities/qwen3.8-27b-transformers/config.json --review docs/architecture/clean-sheet-review.md
python3 scripts/check_phase1_review_remediation.py --config .cache/authorities/qwen3.8-27b-transformers/config.json
python3 scripts/inventory_bf16_checkpoint.py --checkpoint .cache/authorities/qwen3.8-27b-transformers --check-inventory docs/architecture/model-inventory.md
python3 -m compileall -q scripts
git diff --check
git diff --exit-code -- docs/architecture/plan.md
```

The remediation harness independently rejected all 37 promised mutations:
TASK-05 (1), TASK-02/03/04/06/07/09/10/11/13/14/15/16/18/19/21 (15), every
one of the 18 TASK-17 mapping records, both TASK-17 synchronization repairs,
and the TASK-20 rendered experiment field. It printed
`phase1 review remediation negative mutations: PASS`.

No Ruff executable or repository Ruff configuration is present:
`command -v ruff` produced no path. No formatting command was run and no
formatting changed files.

#### Independent raw-file, mapping, and formula checks

The following raw inspections and calculations were run independently of the
owning checkers:

```text
sha256sum .cache/authorities/qwen3.8-27b-transformers/model.safetensors.index.json docs/architecture/bf16-tensor-analysis.md
77042094076611b69791a610065f28b7013b8c621795fa86ddccc8bac7d1b9df  .cache/authorities/qwen3.8-27b-transformers/model.safetensors.index.json
41ce1b4585acf6b33afd6f023d3baf32815ee80b51ca392159819945605ef25d  docs/architecture/bf16-tensor-analysis.md
```

Those hashes exactly match the single recorded ground-truth run. The recorded
command/result was independently checked from this dossier and artifact state:
`analyze_bf16_tensors.py --check-analysis` processed `1199/1199` tensors across
18 shards, exited 0, and took `4516.47s`; the expensive scan was not rerun.

The first JSON fence of `cuda-design-space.md` was parsed as a raw file. It
contains 18 unique mapping IDs and 18 unique records with no missing or extra
IDs; each of the six node types has exactly three records; every record has
`work`, `hbm`, `access`, `synchronization`, `resources`, `occupancy`, `waves`,
and `mode_fit`; all six estimate dimensions are represented; all records use
symbolic `R_t` and `C_cta` and `[32, 64, 128, 256]` candidates; selected count
is zero; and mode counts are `decode_primary=3`, `prefill_primary=2`,
`both=13`. Raw synchronization inspection found `map_mlp_grid_T` as
`sync_none`/`none` and `map_mtp_split_norm_gemm` with the four stages and three
norm/concat/GEMM dependency events.

Independent config arithmetic reproduced the published values:

```text
n_full_layers=16 n_linear_layers=48 d_qkv=10240
mac_full_proj=104857600 mac_attn_coeff=12288 mac_lin_proj=84377600
mac_lin_conv=40960 mac_gdn=2359296 mac_lin_out=31457280
mac_lin_token=118235136 mac_mlp=267386880 mac_lm=1271398400
mac_mtp_fc=52428800 kv_one=4096 kv_all=69632 c_one=61440
s_one=3145728 s_f32=150994944 lm_bytes=2542796800 mtp_bytes=104857600
```

Raw MTP inspection found `mtp_semantics_status` equal to
`conditional_unverified`, with all four MTP proof flags false. Diagram 6 has
distinct prior-state/current-producer edges for KV, C/FIR, and S recurrence;
the region status is
`forced_is_semantic_minimum; region_cut_is_assumed_region_interface_accounting`;
the numerical document contains `K=6144` and zero-based RoPE phase 262143.
CUDA raw inspection found `B_SM` includes `B_warp`, `W_need` is present,
register allocation scope is explicitly SKU-scoped, and fusion inequalities
are qualified rather than universal.

Status and immutability checks passed: TASK-12, TASK-14, TASK-19, and TASK-21
Control/Final statuses are `DONE`; TASK-22 and its sole ledger row remain
`IN PROGRESS`; `TASK-22` occurs once in the ledger; and both
`git diff --exit-code -- docs/architecture/plan.md` and an independent byte
comparison to `HEAD:docs/architecture/plan.md` passed.

- Formatting changed files: none.
- UTC/time/tokens/cost: elapsed `843.056s`; input `2798062`; cached input
  `2680320`; output `25663`; reasoning output `5504`; final cumulative total
  `2823725`; cost unavailable. Telemetry record: session
  `01a0c322-1790-7672-9468-cc03814b828a`, source
  `/home/user/.codex/sessions/2026/09/21/rollout-2026-09-21T08-43-06-01a0c322-1790-7672-9468-cc03814b828a.jsonl`.
- Verdict: **PASS**. All TASK-22 acceptance claims have traceable source-file
  evidence and executed assertions; the TASK-17 severe mapping warning is
  cleared; MTP uncertainty and TASK-21 mechanical-only freeze scope are
  explicit; the plan is immutable; and no expensive BF16 scan was rerun.
- Remaining risk: none identified within TASK-22's stated documentation and
  evidence scope. MTP behavioral semantics remain intentionally
  `conditional_unverified`, not proven.

### Retries and escalation

None.

### Delivery

- Agent/model: inherited active model/configuration; no override.
- Scope: TASK-22 only; coupled IDs `none`.
- Outcome: TASK-22 marked `DONE` after independent verification `PASS`.
- UTC/time/tokens/cost: `2026-09-21T09:01:28Z`; `telemetry_unavailable`

### Final outcome

- Status: `DONE`
- Acceptance evidence: all focused checkers, negative mutations, inventory,
  compile, diff, independent raw checks, and the single full BF16 scan passed;
  independent verification is recorded as `PASS` above.
- Candidate measured delta: N/A — no performance measurement.
- Shipping delta: N/A — documentation/evidence remediation only.
- Quality result: not required.
- Evidence completeness: implementation and independent verification evidence
  complete.
- Tok/s delta: N/A — no meaningful throughput measurement exists for this task.
- Commit: delivery commit on `clean-sheet` (see git log).
- Push: `origin/clean-sheet`.
- First-pass acceptance: implementation first-pass acceptance was unavailable
  after prior interruptions; independent integration acceptance is PASS.
- Total elapsed/tokens/cost: `telemetry_unavailable`.
- Remaining risk or recovery condition: none within the stated scope. The full
  BF16 scan is authoritative for this task and must not be rerun; no unresolved
  implementation decisions remain.
