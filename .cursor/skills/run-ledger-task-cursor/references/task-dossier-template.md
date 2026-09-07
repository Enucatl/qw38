# Task Dossier Template

The planning agent creates `tasks/<PRIMARY-ID>.md` from this template. Replace
all guidance; do not leave placeholders or unresolved choices. Keep the dossier
after delivery as versioned evidence.

```markdown
# <PRIMARY-ID> — <ledger description>

## Control

- Primary ID: `<PRIMARY-ID>`
- Coupled IDs: `<IDs or none>`
- Dependencies: `<IDs or none; all done at admission>`
- Status: `in_progress`
- Ledger acceptance: <verbatim acceptance condition>

## Goal and boundaries

<Desired observable outcome.>

- Constraints: <repository, compatibility, performance, and safety constraints>
- Non-goals: <explicit exclusions>
- Plan impact: `none` <or stop: approved plan change required>
- Affected interfaces: <public APIs, formats, commands, or none>

## Repository evidence

- `<path:line>` — <fact this establishes>

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
- Repository-wide commands: `<exact commands>`
- Native/CUDA/hardware gates: `<exact commands or not applicable with reason>`
- Documentation/evidence updates: <exact paths or none>
- Definition of done: <observable completion boundary>

## Run record

### Planning

- Agent/model: <model and effort>
- UTC/time/tokens/cost: <values when exposed; otherwise unavailable>
- Outcome: <decisions and files changed>

### Implementation

- Agent/model: <model and effort>
- Changes: <paths and behavior>
- Commands: `<command>` — <pass/fail and salient output>
- UTC/time/tokens/cost: <values when exposed; otherwise unavailable>

### Documentation

- Agent/model: <model and effort, or skipped with reason>
- Changes and evidence: <paths and links>
- UTC/time/tokens/cost: <values when exposed; otherwise unavailable>

### Verification

- Attempt: <number>
- Agent/model: <model and effort>
- Diff review: <scope and acceptance result>
- Commands: `<exact command>` — <pass/fail and salient output>
- Formatting changed files: <paths and rerun commands, or none>
- Verdict: `pass` or `fail` — <reason>
- UTC/time/tokens/cost: <values when exposed; otherwise unavailable>

### Retries and escalation

<Each failure, diagnosis, dossier amendment, repair, and result; or none.>

### Final outcome

- Status: `done` or `blocked`
- Acceptance evidence: <links for primary and coupled IDs>
- Commit: <hash and subject, not created, or local-only after push failure>
- Push: <upstream and result, not attempted, or failure>
- First-pass acceptance: <yes/no>
- Total elapsed/tokens/cost: <values when exposed; otherwise unavailable>
- Remaining risk or recovery condition: <text or none>
```
