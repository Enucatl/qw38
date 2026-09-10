---
name: run-ledger-task-cursor
description: >-
  Execute the next eligible or an explicitly selected repository
  implementation-ledger task through Cursor Task subagents. Use when advancing
  implementation_ledger.md, running the next ledger task, continuing
  ledger work, unblocking or delivering a pending ledger ID, or when
  the user mentions run-ledger-task-codex / run-ledger-task-cursor under
  Cursor. Do not use for ad hoc changes that are not tracked in the
  ledger. Prefer this over run-ledger-task-codex when running under Cursor
  rather than Codex.
---

# Run Ledger Task (Cursor)

Act as a lightweight coordinator for exactly one primary ledger increment. The
repository's `plan.md` and `implementation_ledger.md` are authoritative. Keep a
permanent dossier at `tasks/<PRIMARY-ID>.md`; read
[the dossier template](references/task-dossier-template.md) when creating a new
dossier or repairing an inadequate one.
Before planning or implementation, check whether `tasks/<PRIMARY-ID>.md`
already exists. When it exists, read it in full and use its resolved decisions,
file boundaries, acceptance commands, non-goals, and run-record constraints as
the implementation guide. Do not replace or silently weaken an existing
dossier; amend it only when the failure loop explicitly permits a dossier
repair.

## Pre-authored dossiers

Many recovery tasks (for example OPT-043–OPT-056) ship with a high-quality
pre-authored dossier in compact form: **Outcome**, **Implementation**, and
**Acceptance** sections with concrete paths, commands, and boundaries. Treat
that format as authoritative when it is decision-complete.

**Admit without a planning subagent** when the coordinator confirms all of:

- Outcome states the observable deliverable and matches the ledger row.
- Implementation names exact files, behaviors, and non-goals; no material
  choice is left open.
- Acceptance lists testable conditions, artifact paths, and focused commands.
- No placeholders, `TBD`, or contradictory text versus `plan.md` or the ledger.
- Coupled IDs are named explicitly or clearly `none`.
- For throughput / keep-reject tasks whose dossier already cites a sink and
  measured numbers, those numbers are current enough for this increment; when
  the task itself produces the measurements (for example OPT-043), sink ranking
  in the dossier is not required before implementation.

On admission, the coordinator (not a subagent) appends a brief **Run record →
Planning** entry (`skipped — pre-authored dossier admitted`), marks the primary
and any coupled tasks `in_progress`, and proceeds directly to implementation.

**Spawn a planning subagent only when** the dossier is missing, fails the
checklist above, or the failure loop explicitly requires dossier repair.
Planning repairs should amend the existing dossier minimally; do not rewrite a
good pre-authored dossier into the long template unless the repair truly needs
extra structure.

## Performance steering (prefill/decode)

When the increment is a throughput or recovery idea (OPT-* keep/reject,
parity, or similar), do **not** pick the next idea from a stale ranked list
alone. Guide selection and planning with live instrumentation:

1. Prefer the current OPT-020-style exclusive CUDA-event attribution
   (`mixer_mmq`, `gdn_core`, `attention_core`, `ffn_mmq`, plus existing
   categories) on a rebuilt diagnostic against **current** production
   objects. Stale binaries are not evidence.
2. Rank sinks by measured milliseconds (and share of wall). Prefer the
   largest Quartz-owned sink that still has a transferable llama.cpp/ds4
   technique under `plan.md` provenance.
3. When a same-protocol llama.cpp category or whole-prefill/decode comparison
   exists, prefer ideas that close the largest Quartz-versus-llama gap, not
   only the largest Quartz-internal share.
4. Decode work uses decode timing / BEN-001 probes the same way: longest
   Quartz-owned decode sink first.
5. Record the chosen sink and the measured numbers in the planning dossier
   (`Repository evidence` / `Implementation decisions`). If admitting a new
   ledger task after a discovery stop, name the sink that justified it.

Examples: after mixer quality lands, if attribution shows `ffn_mmq` then
`attention_core` dominating, the next idea must target those—not a lower
sink—unless a dossier proves the larger sinks are already llama-competitive
or plan-forbidden.

## Runtime mapping

This skill is the Cursor port of `.agents/skills/run-ledger-task-codex`. Differences:

- Spawn stages with the Cursor **Task** tool, not Codex native subagents.
- Use model slug `cursor-grok-4.6-high` for planning and implementation, and
  `composer-2.5` for documentation, verification/testing, delivery, and their
  repairs. Do not substitute `cursor-grok-4.6-high-fast`, other `-fast`
  variants, or `auto`. If either required
  model is unavailable or rejected, stop and report the failure; do not fall
  back silently.
- Prefer `subagent_type: "generalPurpose"` unless a stage is pure codebase
  exploration, in which case `explore` is allowed for planning-only reads.
- Give each stage **fresh context**: do not `resume` a prior agent; pass a
  self-contained prompt with only the task ID, dossier path, role, repository
  constraints, and output contract. Set `run_in_background: false` and wait for
  each stage before starting the next.
- Do not perform stage work in the coordinator. Skip a stage when it has no
  real work.
- The Cursor Task surface may not expose token/cost fields directly. When the
  Cursor SDK or Agents API is available, source `run.usage` / `result.usage`
  (token counts), `result.duration_ms`, and `agent.get_usage()` or the usage API
  (billed cost). Otherwise record the actually assigned model
  (`cursor-grok-4.6-high` or `composer-2.5`) and
  `telemetry_unavailable` in the dossier; never invent usage.

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
- If `tasks/<ID>.md` exists, it is readable and internally consistent with the
  selected ledger row; unresolved decisions or contradictory acceptance text
  are a planning stop, not an invitation to infer silently.

Also reject unknown IDs and terminal or already-active explicitly selected
tasks. Report the exact failed gate and the evidence inspected.

## Stages

1. **Planning (conditional).** If `tasks/<PRIMARY-ID>.md` passes the
   [pre-authored dossier](#pre-authored-dossiers) checklist, the coordinator
   admits it, records the skipped planning entry, and marks the primary and
   coupled tasks `in_progress`. Otherwise spawn a planning agent with
   `model: "cursor-grok-4.6-high"` to inspect the repository and create or
   amend a decision-complete dossier. A planning agent's output contract is the
   dossier path, coupled IDs, changed files, decisions made, and unresolved
   decisions. Verify that every coupled ID exists, is `pending`, has satisfied
   dependencies, and represents documentation or evidence inseparable from the
   primary increment. Then mark the primary and coupled tasks `in_progress`. Do
   not continue if any implementation choice remains unresolved or the dossier
   is inconsistent with the ledger or plan.
2. Spawn an implementation agent with `model: "cursor-grok-4.6-high"` to implement only the dossier's code, tests,
   and fixtures and run focused validation. The agent must append its changes
   and exact command outcomes to the dossier, without committing.
3. If documentation or evidence changes are required, spawn a fresh
   documentation agent with `model: "composer-2.5"` for prose, links, mechanical index updates, and any
   fixtures, measurements, hashes, contracts, pins, ledger history, or
   acceptance claims the dossier assigns to this stage. It records its work in
   the dossier and does not commit.
4. Spawn a fresh integration verifier with `model: "composer-2.5"`. It independently reviews the complete
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
5. Only after a passing verification, spawn a fresh delivery agent with
   `model: "composer-2.5"`. It confirms
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
  `cursor-grok-4.6-high` implementation repair agent using the dossier and verifier
  findings, then verify again with a fresh `composer-2.5` verifier agent.
- For a repeated failure, or an architectural failure on any attempt, spawn one
  `cursor-grok-4.6-high` diagnostic agent to amend an inadequate dossier, then one fresh
  `cursor-grok-4.6-high` repair and one fresh `composer-2.5` verification pass.
- On any further failure, unavailable dependency, material ambiguity, or needed
  architecture change, set the primary and applicable coupled tasks to
  `blocked`, add the reason and recovery condition to the dossier and ledger,
  and do not commit or push.

When implementation discovers additional work, stop that stage. Have the
coordinator add a stable task to the ledger and link it from the dossier before
resuming; do not silently expand scope.

## Completion Report

Report the task and coupled IDs, final status, commit and push result, verifier
commands, retries, and dossier path. Note whether planning was skipped
(pre-authored dossier admitted) or ran via subagent. Also report per-stage model
(`cursor-grok-4.6-high` for planning/implementation and `composer-2.5` for
documentation, verification/testing, and delivery), elapsed time when known,
retry count, first-pass acceptance, and token/cost data only when the runtime
exposes them.

For throughput / keep-reject / oracle-steered tasks, always include a
**tok/s delta versus the then-current baseline** at the end of the completion
report (and in the dossier Final outcome / delivery ledger History entry):

- Name the baseline (fixture path + mean tok/s) and the post-task mean tok/s.
- Report absolute delta (`post - baseline`) and relative speedup
  (`post / baseline`, or percent).
- On a reject/revert, still report the measured post number and state that
  speedup is `0` (baseline unchanged).
- When an OPT-021-style llama.cpp same-sitting number exists, also report
  Quartz-versus-llama tok/s (informational unless that task owns the gate).

Example: `4K Quartz 1680.8 tok/s vs baseline 967.3 (+713.5, 1.74×); llama 3227.5`.
