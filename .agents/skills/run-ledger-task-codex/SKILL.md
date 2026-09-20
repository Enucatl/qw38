---
name: run-ledger-task-codex
description: Execute the next eligible or an explicitly selected task in docs/architecture/task_ledger.md through fresh planning, implementation, documentation, independent verification, commit, and push using Codex subagents. Use for requests to advance that ledger under Codex; do not use for ad hoc changes that are not tracked in the ledger.
---

# Run Ledger Task (Codex)

Act as a lightweight coordinator for exactly one primary ledger increment. The
repository's `docs/architecture/plan.md` and `docs/architecture/task_ledger.md`
are authoritative. Keep a permanent dossier at
`docs/architecture/tasks/<PRIMARY-ID>.md`; read
[the dossier template](references/task-dossier-template.md) before planning.
Before planning or implementation, check whether
`docs/architecture/tasks/<PRIMARY-ID>.md`
already exists. When it exists, read it in full and treat its resolved
decisions, file boundaries, acceptance commands, non-goals, and run-record
constraints as the implementation guide. Do not replace or silently weaken an
existing dossier; amend it only when the failure loop explicitly permits a
dossier repair.

Use native subagents sequentially because they share the working tree. Give each
stage fresh context (`fork_turns: "none"`) containing only the task ID, dossier
path, role, repository constraints, and output contract. Do not perform stage
work in the coordinator. Skip a stage when it has no real work.

After each stage completes, run the deterministic
[`account_usage.py`](scripts/account_usage.py) helper against the local Codex
session directory. Filter by the child agent path when known, capture its JSON
record, and append the model, effort, elapsed time, and final cumulative token
fields to that stage's dossier record. If no matching token event exists, record
`telemetry_unavailable` with the session path; do not invent or ask the stage to
estimate usage. This helper reads runtime-owned JSONL as an operational aid, so
missing or changed fields are a recorded limitation rather than a run failure.

## Performance evidence (prefill/decode)

When the increment is a throughput or recovery idea (OPT-* keep/reject,
parity, or similar), do **not** pick the next idea from a stale ranked list
alone. Guide selection and planning with live instrumentation that satisfies
the [performance evidence checklist](references/performance-evidence-checklist.md).
The eight [adversarial review cases](references/performance-evidence-review-cases.md)
are required reading before ranking or verifying a bottleneck claim.

1. Establish measurement identity and graph-aware coverage on a rebuilt
   diagnostic against **current** production objects before ranking any sink.
   A rebuilt binary is necessary but not sufficient. Stale binaries are not
   evidence. OPT-020-style CUDA-event categories cannot rank production graph
   execution until coverage of that graph is established.
2. Rank only covered, identity-matched sinks by measured milliseconds (and
   share of wall) after reconstructing a disjoint union. Prefer the largest
   Quartz-owned covered sink that still has a transferable llama.cpp/ds4
   technique under `plan.md` provenance. Missing coverage is not zero excess.
3. A Quartz-versus-llama comparison requires matching metric identities
   (same prefill / decode-only / complete-request / component boundary) and
   mapped corresponding work. Unmapped families stay `null`/`unknown`.
4. Decode work uses decode timing / BEN-001 probes the same way, after the
   same coverage and identity checks: longest covered Quartz-owned decode
   sink first.
5. Record the chosen sink, identities, coverage, contradiction register,
   claim types, and measured numbers in the planning dossier
   (`Repository evidence` / `Implementation decisions` /
   `Performance evidence`). If admitting a new ledger task after a discovery
   stop, name the sink that justified it only when those checks pass.

## Admission

Inspect the repository before mutation. Accept zero or one task ID:

- With an explicit ID, use that task.
- Without an ID, scan task entries from top to bottom and select the first
  `TODO` task whose listed dependencies are all `DONE`. Ledger order
  order is the deterministic priority; do not infer a different priority from
  task names or perceived importance. Report the selected ID before mutation.
- If more than one ID was supplied, or no eligible TODO task exists, reject
  the run without changing files.

Continue only when all of these hold:

- The ID occurs exactly once in the ledger, has status `TODO`, and all listed
  dependencies have status `DONE`.
- The worktree is clean, including untracked files.
- The current branch has a configured upstream.
- The task does not require an unapproved change to `docs/architecture/plan.md`.
- If `docs/architecture/tasks/<ID>.md` exists, it is readable and internally consistent with the
  selected ledger row; unresolved decisions or contradictory acceptance text
  are a planning stop, not an invitation to infer silently.
- For throughput, recovery, keep/reject, or bottleneck-ranking increments,
  apply the [performance evidence checklist](references/performance-evidence-checklist.md)
  at admission. Unresolved measurement identity, coverage, or material
  contradictions in an existing dossier are a planning stop.

Also reject unknown IDs and terminal or already-active explicitly selected
tasks. Report the exact failed gate and the evidence inspected.

## Stages

1. Spawn a `gpt-5.6-sol` agent at medium reasoning to inspect the repository and
   create or, when it already exists, validate and amend the decision-complete
   dossier. Its output contract is the dossier path,
   coupled IDs, changed files, decisions made, and unresolved decisions. Apply
   the [performance evidence checklist](references/performance-evidence-checklist.md)
   when the increment ranks a bottleneck or reports a timing delta. Verify
   that every coupled ID exists, is `TODO`, has satisfied dependencies, and
   represents documentation or evidence inseparable from the primary increment.
   Then mark the primary and coupled tasks `IN PROGRESS`. Do not continue if any
   implementation choice remains unresolved or the dossier is inconsistent with
   the ledger or plan.
2. Spawn a `gpt-5.6-terra` agent at medium reasoning to implement only the
   dossier's code, tests, and fixtures and run focused validation. Use Luna
   medium instead only when the planning dossier explicitly classifies every
   implementation change as mechanical. Use Sol low instead when implementation
   involves CUDA kernels, concurrency, memory ownership, numerical invariants,
   security boundaries, or designing new acceptance evidence. The agent must
   append its changes and exact command outcomes to the dossier, without
   committing.
3. If documentation or evidence changes are required, spawn a fresh Luna agent
   at medium reasoning for prose, links, and mechanical index updates. Use Terra
   medium when the stage creates or interprets fixtures, measurements, hashes,
   contracts, pins, ledger history, or acceptance claims. This stage prepares a
   draft; label conclusions `unverified` and apply the
   [performance evidence checklist](references/performance-evidence-checklist.md).
   Do not supply a verifier verdict that has not been produced. It records its
   work in the dossier and does not commit.
4. Spawn a fresh `gpt-5.6-sol` integration verifier at low reasoning. It
   independently reviews the complete diff against the dossier, ledger
   acceptance condition, `docs/architecture/plan.md`, and repository boundaries. It may run
   formatting but makes no semantic fixes. It runs the dossier's focused and
   repository-wide gates, including `uv run ruff format .`, Ruff checks,
   required pytest selections, native builds/tests, and named CUDA or hardware
   gates. Pass task paths and required phases without a pre-decided verdict.
   The verifier independently queries raw records for the largest claimed gap
   and every claimed eliminated gap; checks coverage, units, and identities;
   records calculations; and assesses the documentation draft plus all
   artifacts against the
   [performance evidence checklist](references/performance-evidence-checklist.md)
   and the eight
   [review cases](references/performance-evidence-review-cases.md). A second
   agent restating generated JSON is not independent evidence. It must trace
   every acceptance claim to an executed assertion or an independently inspected
   artifact; stdout labels, fixture status fields, and dossier claims are not
   sufficient evidence by themselves. Failed evidence checks return to
   implementation. Missing hardware is `incomplete`/`blocked`, not a measured
   rejection or no opportunity. If formatting changes files, it reruns affected
   tests. It appends exact commands, outcomes, and a clear pass/fail verdict to
   the dossier.
5. Only after a passing verification, spawn a fresh Luna delivery agent at
   medium reasoning. It confirms scope and acceptance evidence as determined by
   verification, changes the primary and every coupled task from `IN PROGRESS`
   to `DONE`, adds the final UTC ledger entry, records the outcome in the
   dossier, creates one commit, and pushes the current branch to its configured
   upstream. Delivery may publish the verified verdict but must not invent or
   revise scientific conclusions; a semantic report change returns to
   verification. Apply the
   [performance evidence checklist](references/performance-evidence-checklist.md)
   reporting split: candidate measured delta, shipping delta, quality result,
   and evidence completeness stay separate.

The delivery commit uses a Google-style subject of at most 50 characters and an
intent-focused body. It must not force-push, rebase, merge, amend, or
automatically handle a non-fast-forward rejection. If push fails, stop and
preserve the local commit.

Every successfully verified task must receive exactly one delivery commit and a
successful push to the configured upstream before the next ledger task may
begin. Invoking this skill, explicitly or implicitly, authorizes that ordinary
delivery commit and push. A push failure is a delivery failure: preserve the
local commit, report it, and do not advance the ledger. This does not authorize
a change to `docs/architecture/plan.md`.

## Failure Loop

Keep retries bounded and record every attempt in the dossier:

- After the first ordinary verification failure, spawn one fresh Terra medium
  implementation repair agent using the dossier and verifier findings, then
  verify again with a fresh Sol low agent.
- For a repeated failure, or an architectural failure on any attempt, spawn one
  Sol medium diagnostic agent to amend an inadequate dossier, then one fresh
  Terra medium repair and one fresh Sol low verification pass.
- On any further failure, unavailable dependency, material ambiguity, or needed
  architecture change, set the primary and applicable coupled tasks to
  `BLOCKED`, add the reason and recovery condition to the dossier and ledger,
  and do not commit or push.

When implementation discovers additional work, stop that stage. Have the
coordinator add a stable task to the ledger and link it from the dossier before
resuming; do not silently expand scope.

## Completion Report

Report the task and coupled IDs, final status, commit and push result, verifier
commands, retries, and dossier path. Also report per-stage model, elapsed time,
retry count, first-pass acceptance, and token/cost data when the runtime exposes
them. Apply the
[performance evidence checklist](references/performance-evidence-checklist.md)
when reporting deltas. Every task must include a tok/s delta section in the
dossier and report, with matching metric identity and window on every ratio:
candidate measured delta, shipping delta, quality result, and evidence
completeness as separate fields. Name the then-current baseline fixture and
mean tok/s, the post-task fixture and mean tok/s, absolute delta
(`post - baseline`), and relative speedup (`post / baseline` or percent) only
for the identity-matched metric being claimed. Diagnostics use `N/A` for
shipping throughput changes. Rejected candidates may have measured gains while
shipping delta is zero; do not conflate a rejected measured result with zero
shipping impact. If the task has no meaningful throughput measurement, write
`N/A` for the values and explain why; never fabricate a number. Treat cost
savings as unproven until three to five representative tasks show unchanged
acceptance quality against the Sol-only baseline.
