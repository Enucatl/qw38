---
name: run-ledger-task
description: Execute one repository implementation-ledger task through fresh planning, implementation, documentation, independent verification, and authorized delivery agents. Use for requests to run a task ID from implementation_ledger.md; do not use for ad hoc changes that are not tracked there.
---

# Run Ledger Task

Act as a lightweight coordinator for exactly one primary ledger increment. The
repository's `plan.md` and `implementation_ledger.md` are authoritative. Keep a
permanent dossier at `tasks/<PRIMARY-ID>.md`; read
[the dossier template](references/task-dossier-template.md) before planning.

Use native subagents sequentially because they share the working tree. Give each
stage fresh context (`fork_turns: "none"`) containing only the task ID, dossier
path, role, repository constraints, and output contract. Do not perform stage
work in the coordinator. Skip a stage when it has no real work.

## Admission

Parse one task ID from the request and inspect the repository before mutation.
Reject the run without changing files unless all of these hold:

- The ID occurs exactly once in the ledger, has status `pending`, and all listed
  dependencies have status `done`.
- The worktree is clean, including untracked files.
- The current branch has a configured upstream.
- The task does not require an unapproved change to `plan.md`.

Also reject missing or extra IDs, unknown IDs, and terminal or already-active
tasks. Report the exact failed gate and the evidence inspected.

## Stages

1. Spawn a `gpt-5.6-sol` agent at medium reasoning to inspect the repository and
   create a decision-complete dossier. Its output contract is the dossier path,
   coupled IDs, changed files, decisions made, and unresolved decisions. Verify
   that every coupled ID exists, is `pending`, has satisfied dependencies, and
   represents documentation or evidence inseparable from the primary increment.
   Then mark the primary and coupled tasks `in_progress`. Do not continue if any
   implementation choice remains unresolved or the dossier is inconsistent with
   the ledger or plan.
2. Spawn a `gpt-5.6-luna` agent at medium reasoning to implement only the
   dossier's code, tests, and fixtures and run focused validation. It must append
   its changes and exact command outcomes to the dossier, without committing.
3. If documentation or evidence changes are required, spawn a fresh Luna agent
   at medium reasoning to make them, including applicable handbook, README,
   evidence links, and ledger-history updates. It records its work in the
   dossier and does not commit.
4. Spawn a fresh Luna integration verifier at medium reasoning. It independently
   reviews the complete diff against the dossier, ledger acceptance condition,
   `plan.md`, and repository boundaries. It may run formatting but makes no
   semantic fixes. It runs the dossier's focused and repository-wide gates,
   including `uv run ruff format .`, Ruff checks, required pytest selections,
   native builds/tests, and named CUDA or hardware gates. If formatting changes
   files, it reruns affected tests. It appends exact commands, outcomes, and a
   clear pass/fail verdict to the dossier.
5. Only after a passing verification, spawn a fresh Luna delivery agent at
   medium reasoning. It confirms scope and acceptance evidence, changes the
   primary and every coupled task from `in_progress` to `done`, adds the final
   UTC ledger entry, records the outcome in the dossier, creates one commit, and
   pushes the current branch to its configured upstream.

The delivery commit uses a Google-style subject of at most 50 characters and an
intent-focused body. It must not force-push, rebase, merge, amend, or
automatically handle a non-fast-forward rejection. If push fails, stop and
preserve the local commit.

Explicit `$run-ledger-task` invocation authorizes the ordinary final commit and
push. Implicit activation does not: obtain user confirmation immediately before
spawning the delivery agent. Neither form authorizes a `plan.md` change.

## Failure Loop

Keep retries bounded and record every attempt in the dossier:

- After the first ordinary verification failure, spawn one fresh Luna
  implementation repair agent using the dossier and verifier findings, then
  verify again with a fresh Luna agent.
- For a repeated failure, or an architectural failure on any attempt, spawn one
  Sol medium diagnostic agent to amend an inadequate dossier, then one fresh
  Luna repair and one fresh Luna verification pass.
- On any further failure, unavailable dependency, material ambiguity, or needed
  architecture change, set the primary and applicable coupled tasks to
  `blocked`, add the reason and recovery condition to the dossier and ledger,
  and do not commit or push.

When implementation discovers additional work, stop that stage. Have the
coordinator add a stable task to the ledger and link it from the dossier before
resuming; do not silently expand scope.

## Completion Report

Report the task and coupled IDs, final status, commit and push result, verifier
commands, retries, and dossier path. Also report per-stage model, elapsed time,
retry count, first-pass acceptance, and token/cost data when the runtime exposes
them. Treat cost savings as unproven until three to five representative tasks
show unchanged acceptance quality against the Sol-only baseline.
