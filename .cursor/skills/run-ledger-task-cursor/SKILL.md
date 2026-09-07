---
name: run-ledger-task-cursor
description: >-
  Execute the next eligible or an explicitly selected repository
  implementation-ledger task through Cursor Task subagents using
  cursor-grok-4.6-high for every stage (planning, implementation,
  documentation, verification, delivery). Use when advancing
  implementation_ledger.md via Cursor Agent CLI; do not use for ad hoc
  changes that are not tracked there. Prefer this over run-ledger-task
  when running under Cursor rather than Codex.
disable-model-invocation: true
---

# Run Ledger Task (Cursor)

Act as a lightweight coordinator for exactly one primary ledger increment. The
repository's `plan.md` and `implementation_ledger.md` are authoritative. Keep a
permanent dossier at `tasks/<PRIMARY-ID>.md`; read
[the dossier template](references/task-dossier-template.md) before planning.

## Runtime mapping

This skill is the Cursor port of `.agents/skills/run-ledger-task`. Differences:

- Spawn stages with the Cursor **Task** tool, not Codex native subagents.
- Use model slug `cursor-grok-4.6-high` for **every** stage, repair, and
  diagnostic agent. Do not substitute another family, effort, or the
  `cursor-grok-4.6-high-fast` variant. Never use a `-fast` model for this
  skill. If `cursor-grok-4.6-high` is unavailable or rejected, stop and report
  the failure to the user; do not fall back.
- Prefer `subagent_type: "generalPurpose"` unless a stage is pure codebase
  exploration, in which case `explore` is allowed for planning-only reads.
- Give each stage **fresh context**: do not `resume` a prior agent; pass a
  self-contained prompt with only the task ID, dossier path, role, repository
  constraints, and output contract. Set `run_in_background: false` and wait for
  each stage before starting the next.
- Do not perform stage work in the coordinator. Skip a stage when it has no
  real work.
- There is no Codex `account_usage.py` equivalent here. After each stage,
  record `model: cursor-grok-4.6-high` and `telemetry_unavailable` (Cursor
  session) in the dossier unless the runtime clearly exposes token/cost
  fields; never invent usage.

## Admission

Inspect the repository before mutation. Accept zero or one task ID:

- With an explicit ID, use that task.
- Without an ID, scan the `Gates and Tasks` table from top to bottom and select
  the first `pending` task whose listed dependencies are all `done`. Ledger row
  order is the deterministic priority; do not infer a different priority from
  task names or perceived importance. Report the selected ID before mutation.
- If more than one ID was supplied, or no eligible pending task exists, reject
  the run without changing files.

Continue only when all of these hold:

- The ID occurs exactly once in the ledger, has status `pending`, and all listed
  dependencies have status `done`.
- The worktree is clean, including untracked files.
- The current branch has a configured upstream.
- The task does not require an unapproved change to `plan.md`.

Also reject unknown IDs and terminal or already-active explicitly selected
tasks. Report the exact failed gate and the evidence inspected.

## Stages

Every spawn below uses Task with `model: "cursor-grok-4.6-high"` only.

1. Spawn a planning agent to inspect the repository and create a
   decision-complete dossier. Its output contract is the dossier path, coupled
   IDs, changed files, decisions made, and unresolved decisions. Verify that
   every coupled ID exists, is `pending`, has satisfied dependencies, and
   represents documentation or evidence inseparable from the primary increment.
   Then mark the primary and coupled tasks `in_progress`. Do not continue if any
   implementation choice remains unresolved or the dossier is inconsistent with
   the ledger or plan.
2. Spawn an implementation agent to implement only the dossier's code, tests,
   and fixtures and run focused validation. The agent must append its changes
   and exact command outcomes to the dossier, without committing.
3. If documentation or evidence changes are required, spawn a fresh
   documentation agent for prose, links, mechanical index updates, and any
   fixtures, measurements, hashes, contracts, pins, ledger history, or
   acceptance claims the dossier assigns to this stage. It records its work in
   the dossier and does not commit.
4. Spawn a fresh integration verifier. It independently reviews the complete
   diff against the dossier, ledger acceptance condition, `plan.md`, and
   repository boundaries. It may run formatting but makes no semantic fixes. It
   runs the dossier's focused and repository-wide gates, including
   `uv run ruff format .`, Ruff checks, required pytest selections, native
   builds/tests, and named CUDA or hardware gates. It must trace every
   acceptance claim to an executed assertion or an independently inspected
   artifact; stdout labels, fixture status fields, and dossier claims are not
   sufficient evidence by themselves. If formatting changes files, it reruns
   affected tests. It appends exact commands, outcomes, and a clear pass/fail
   verdict to the dossier.
5. Only after a passing verification, spawn a fresh delivery agent. It confirms
   scope and acceptance evidence, changes the primary and every coupled task
   from `in_progress` to `done`, adds the final UTC ledger entry, records the
   outcome in the dossier, creates one commit, and pushes the current branch to
   its configured upstream.

The delivery commit uses a Google-style subject of at most 50 characters and an
intent-focused body. It must not force-push, rebase, merge, amend, or
automatically handle a non-fast-forward rejection. If push fails, stop and
preserve the local commit.

Explicit `/run-ledger-task-cursor` (or `$run-ledger-task-cursor`) invocation
authorizes the ordinary final commit and push. Implicit activation does not:
obtain user confirmation immediately before spawning the delivery agent.
Neither form authorizes a `plan.md` change.

## Failure Loop

Keep retries bounded and record every attempt in the dossier:

- After the first ordinary verification failure, spawn one fresh
  implementation repair agent using the dossier and verifier findings, then
  verify again with a fresh verifier agent.
- For a repeated failure, or an architectural failure on any attempt, spawn one
  diagnostic agent to amend an inadequate dossier, then one fresh repair and
  one fresh verification pass.
- On any further failure, unavailable dependency, material ambiguity, or needed
  architecture change, set the primary and applicable coupled tasks to
  `blocked`, add the reason and recovery condition to the dossier and ledger,
  and do not commit or push.

When implementation discovers additional work, stop that stage. Have the
coordinator add a stable task to the ledger and link it from the dossier before
resuming; do not silently expand scope.

## Completion Report

Report the task and coupled IDs, final status, commit and push result, verifier
commands, retries, and dossier path. Also report per-stage model
(`cursor-grok-4.6-high`), elapsed time when known, retry count, first-pass
acceptance, and token/cost data only when the runtime exposes them.
