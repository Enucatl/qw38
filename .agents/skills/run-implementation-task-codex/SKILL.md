---
name: run-implementation-task-codex
description: Execute and safely deliver exactly one eligible QW38 implementation task from docs/implementation/task_ledger.md using fresh Codex subagents. Use when advancing the sequential implementation ledger; do not use for ad hoc changes or architecture redesign.
---

# Run Implementation Task (Codex)

Coordinate exactly one `TASK-XXX` from the implementation ledger. The task
file is the complete implementation contract; unlike the architecture-ledger
workflow, this skill has no planning dossier stage.

## Authority

Use these sources, in descending task relevance:

- `docs/implementation/tasks/<TASK-ID>.md` for scope and acceptance;
- `docs/architecture/architecture-v0.md` for model and runtime architecture;
- `docs/implementation/technology-baseline.md` for toolchain, platform, and
  container requirements;
- `docs/implementation/code-standards.md` for C++ and CUDA rules.

The coordinator handles admission and delivery. Use fresh subagents with
`fork_turns: "none"` for implementation, code review, repair, and any requested
evidence collection. Pass only the task ID and path, relevant normative paths,
role, findings when applicable, and required output. Run agents sequentially
because they share the working tree.

Use `gpt-6-luna` at high reasoning for implementation and repair, and at
medium reasoning for separately requested command collection. Use
`gpt-6-sol` at high reasoning for independent code review and difficult
diagnosis.

## Admission

Accept zero or one explicit task ID.

- With an ID, select exactly that task.
- Without one, scan the ledger from top to bottom and select the first `TODO`
  task whose dependencies are all `DONE`.
- Ledger order is the only implicit priority.

Before any mutation, reject the invocation when any of these gates fail:

- more than one ID was supplied, the ID is unknown, its status is not `TODO`,
  its dependencies are incomplete, or no eligible task exists;
- the task file or any normative document is missing;
- the ledger row and task file disagree on identity, name, status, or
  dependencies;
- the worktree is not clean, including untracked files, or the current branch
  has no configured upstream;
- the contract contains an unresolved `TBD`, placeholder, contradiction, or
  lacks objective acceptance criteria;
- an acceptance requirement needs an architectural decision not specified by
  Architecture V0; report `TASK_CONTRACT_INCOMPLETE`.

Report the exact failed gate and evidence inspected. Do not infer an
architecture decision. For an unspecified architectural requirement, report:

```text
TASK_CONTRACT_INCOMPLETE

Requirement:
Missing Architecture V0 decision:
Evidence inspected:
```

After admission, change the selected ledger row to `IN_PROGRESS`, and update
the task-file status when the task file has a status field. Do not activate
downstream tasks.

## Implementation

Spawn a fresh Luna implementation subagent. Require it to:

1. read the complete task file and only the relevant sections of the normative
   documents;
2. implement only the task, obeying `LOCKED` decisions and preserving the
   distinction between tuning defaults and required semantics;
3. run focused commands on the final candidate that establish every acceptance
   criterion and each required test, benchmark, or diagnostic, keeping
   correctness tests separate from benchmarks;
4. record exact commands, results, logs, artifact/binary identities, and
   hardware context in the task's Completion Report; and
5. leave commits and pushes to the coordinator.

Unrelated discoveries belong in the Completion Report as
`FOLLOW_UP_REQUIRED`; do not expand scope, create roadmap tasks, or edit
downstream specifications. If missing work prevents acceptance, stop and
block the task.

If a locked decision is impossible, inconsistent, or incompatible with the
required semantics, require this report and stop without repair:

```text
ARCHITECTURE_BLOCKER

Decision ID:
Attempted implementation:
Observed problem:
Evidence:
Why this is architectural rather than tuning:
Smallest plausible alternative:
Affected downstream tasks:
```

## Evidence and independent review

The implementation agent leaves complete command evidence for Sol. Missing
hardware or a skipped required check is incomplete evidence, not a pass. Apply
the verification scope in `code-standards.md`; do not run unrelated suites or
repeat unchanged commands merely for a second witness.

A fresh Sol reviewer reads the task contract, relevant architecture and
code standards, the complete diff, changed code and its production callers,
tests, and the evidence. Review independently for:

- correct behavior at numerical, state, lifetime, error, and CUDA boundaries
  relevant to the change, including failure and continuation paths;
- compliance with Architecture V0's locked decisions, the technology baseline,
  and `code-standards.md`'s design philosophy and concrete rules;
- the smallest implementation that fully meets the task, without speculative
  abstractions, extra dependencies, or unrelated changes;
- discriminating tests and complete, current evidence for every acceptance
  criterion, including resource or schedule effects where relevant.

Sol inspects code and evidence; Luna runs builds, tests, benchmarks, and any
targeted commands Sol requests. Sol must not accept a passing test log as proof
of code correctness. Return all material findings in one pass with file/line,
the violated contract or failure scenario, and a concrete correction:

```text
REVIEW: PASS|CHANGES_REQUIRED|BLOCKED
Reviewed candidate:
Acceptance and evidence gaps:
Code findings (severity, file/line, contract, impact, correction):
Targeted evidence requested:
```

Handle `CHANGES_REQUIRED` first by sending the findings to the same Sol reviewer
and asking it to make only the requested code or documentation corrections
directly. It may not change architecture, expand scope, commit, or push. Luna
continues to run builds, tests, benchmarks, and targeted evidence commands;
after affected evidence is refreshed, the same Sol reviewer examines the
complete revised candidate and reports whether the findings are resolved.

When Sol requests evidence without a code change, batch its requests into one
fresh Luna evidence subagent. It collects output without changing code; the same
Sol reviewer then examines the unchanged candidate and new evidence. A failed
check enters the repair path if unused. Allow one Sol repair round and one
supplemental evidence round. If code findings remain after repair, or evidence
remains incomplete after the supplemental round, mark the task `BLOCKED`.
Unavailable required hardware also marks it `BLOCKED`.

An architectural failure, missing required dependency, or material contract
ambiguity also marks the task `BLOCKED`. For every `BLOCKED` outcome, record the
reason in both the task file and ledger, then stop. Do not use repair to invent
a design.

## Delivery

Only after Sol review passes, the coordinator:

1. confirm that the reviewed implementation and evidence are unchanged;
2. record the review result in the Completion Report;
3. change the task and ledger from `IN_PROGRESS` to `DONE`;
4. create exactly one commit for the task; and
5. push the current branch to its configured upstream.

If code or evidence changes after review, refresh affected evidence and obtain
a new Sol review before committing. The coordinator makes only status and
review-record edits after approval.

Use the git-commit skill to create a proper git commit message. Never
force-push, rebase, merge, amend, or automatically resolve a non-fast-forward
rejection. If push fails, preserve the local commit, report the failure, do not
claim successful delivery, and do not start another task.

## Final report

Report the task ID and name, final status, implementation/review models,
whether repair or separate evidence collection was used, exact acceptance
commands and results, commit hash if created, push result, task-file path, and
any blocker or follow-up.
