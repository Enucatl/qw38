# Task Dossier Template

Use this template when no dossier exists or when the failure loop requires a
repair that needs extra structure. A compact pre-authored dossier with
**Outcome**, **Implementation**, and **Acceptance** sections is valid for
admission under `run-ledger-task-cursor` when decision-complete; do not expand
it into this long form unless a repair truly needs the extra sections.
Throughput, recovery, keep/reject, and bottleneck-ranking tasks must satisfy
[performance-evidence-checklist.md](performance-evidence-checklist.md); read
[performance-evidence-review-cases.md](performance-evidence-review-cases.md)
before ranking or verifying a timing claim.

When creating from this template, replace all guidance; do not leave
placeholders or unresolved choices. Keep the dossier after delivery as versioned
evidence.

```markdown
# <PRIMARY-ID> — <ledger description>

## Control

- Primary ID: `<PRIMARY-ID>`
- Coupled IDs: `<IDs or none>`
- Dependencies: `<IDs or none; all DONE at admission>`
- Status: `IN PROGRESS`
- Ledger acceptance: <verbatim acceptance condition>

## Goal and boundaries

<Desired observable outcome.>

- Constraints: <repository, compatibility, performance, and safety constraints>
- Non-goals: <explicit exclusions>
- Plan impact: `none` <or stop: approved plan change required>
- Affected interfaces: <public APIs, formats, commands, or none>

## Repository evidence

- `<path:line>` — <fact this establishes>

## Performance evidence

Required for throughput, recovery, keep/reject, or bottleneck-ranking tasks;
write `N/A` with reason for instruction-only or non-timing work. Draft
conclusions stay `unverified` until verification.

- Measurement identity: <engine/commit, loaded binary hash, build flags/image/tool versions, GGUF, selectors, input token hash, prefix, output/eval counts, first-token convention, allocated capacity, populated length, sampling/output policy, graph mode, clocks/residents, warmups/samples, exact numerator/start/end events>
- Metric class: <prefill | decode-only | complete-request | component | public sample/eval/output; a ratio requires matching identities>
- Coverage: <tables, units, clock domain, capture bounds, streams, expected/observed graph/node counts, family mapping, unclassified work; OPT-136 coverage.json path or `unavailable`; graph envelopes are not leaf kernel time>
- Time accounting: <disjoint union method; nested/overlap handling; no parent+child or CPU-wait+GPU double count>
- Contradiction register: <conflicting sources, identities, quantitative disagreement, disposition, resolving check, or none>
- Claim types: each conclusion tagged `measured` | `derived` | `hypothesis` | `incomplete` | `historical` (derivations include formula, units, and input links)
- Target/guard roles: <region, primary metric, targets, guards, thresholds, policy ID; apply OPT-135 when opted in; or not opted in>
- Evidence completeness: <complete | incomplete | unavailable checks>
- Screen eligibility: <candidate hypothesis, matched boundary, bounded
  correctness cases, production-shaped inputs, and disconfirming experiment;
  source/SASS mechanism may remain unknown>
- Shipping evidence: <correctness, quality, state/memory, and whole-engine
  target/guard gates; causal mechanism is required only for causal claims>

## Implementation decisions

<Decision-complete approach, including exact files and behavior.>

- Invariants: <properties that must remain true>
- Rejected alternatives: <material alternatives and why rejected>
- Discovered ledger work: <linked IDs or none>
- Unresolved decisions: `none`

## Acceptance and validation

- Acceptance conditions: <testable conditions, including every coupled task>
- Tests/fixtures to add or change: <exact paths and cases>
- Focused commands: `<exact commands>`
- Candidate quality: `<required NLL / OPT-058 commands, or explicitly not required with reason>`
- Repository-wide commands: `<exact commands>`
- Native/CUDA/hardware gates: `<exact commands or not applicable with reason>`
- Documentation/evidence updates: <exact paths or none>
- Definition of done: <observable completion boundary>

## Run record

### Planning

- Agent/model: <model and effort>
- UTC/time/tokens/cost: <values when exposed; otherwise unavailable>
- Outcome: <decisions and files changed>
- Performance evidence applied: <yes, N/A with reason, or planning stop>

### Implementation

- Agent/model: <model and effort>
- Changes: <paths and behavior>
- Commands: `<command>` — <pass/fail and salient output>
- UTC/time/tokens/cost: <values when exposed; otherwise unavailable>

### Documentation

- Agent/model: <model and effort, or skipped with reason>
- Changes and evidence: <paths and links>
- Draft conclusions: `unverified` (verification has not run; do not include a verifier verdict)
- UTC/time/tokens/cost: <values when exposed; otherwise unavailable>

### Verification

- Attempt: <number>
- Agent/model: <model and effort>
- Diff review: <scope and acceptance result>
- Independent raw-record checks: <largest claimed gap, each claimed eliminated gap, coverage, units, identities, calculations>
- Commands: `<exact command>` — <pass/fail and salient output>
- Formatting changed files: <paths and rerun commands, or none>
- Verdict: `pass` or `fail` — <reason; failed evidence checks return to implementation; missing hardware is incomplete/blocked>
- UTC/time/tokens/cost: <values when exposed; otherwise unavailable>

### Retries and escalation

<Each failure, diagnosis, dossier amendment, repair, and result; or none.>

### Final outcome

- Status: `DONE` or `BLOCKED`
- Acceptance evidence: <links for primary and coupled IDs>
- Candidate measured delta: <identity-matched metric/window, or N/A with reason>
- Shipping delta: <zero on reject/revert; N/A for diagnostics; identity-matched on keep>
- Quality result: <pass/fail/not required/incomplete>
- Evidence completeness: <complete | incomplete | unavailable checks>
- Throughput delta (when applicable): <baseline fixture + mean tok/s> →
  <post mean tok/s>; delta `<post-baseline>` tok/s (`<post/baseline>×`);
  matching metric identity and window; on reject: candidate measured post and
  shipping delta `0` (reverted); optional same-sitting llama.cpp tok/s only on
  a matching metric identity
- Commit: <hash and subject, not created, or local-only after push failure>
- Push: <upstream and result, not attempted, or failure>
- First-pass acceptance: <yes/no>
- Total elapsed/tokens/cost: <values when exposed; otherwise unavailable>
- Remaining risk or recovery condition: <text or none>
```

When planning a throughput idea, cite current rebuilt measurements only after
measurement identity and graph-aware coverage are established. Name the covered
sink the increment targets. Do not rank OPT-020-style CUDA-event categories, or
rely on a prior ranked list, once newer measurements exist or coverage is
incomplete.
