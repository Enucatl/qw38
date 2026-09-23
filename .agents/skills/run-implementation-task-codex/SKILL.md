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

Start every implementation, repair, verification, diagnostic, and delivery
stage as a new Codex subagent with `fork_context: false`. Never resume an
agent or reuse one in another role. Give each subagent only the task ID, task
path, relevant normative paths, role, any findings needed for that role, and
the required output contract. Because subagents share the working tree, run
these stages sequentially.

Use exactly `gpt-6-luna` at high reasoning for implementation and repair,
and independent verification. Use exactly `gpt-6-sol` at high reasoning only
for difficult diagnosis that exceeds the implementation agent's capacity.
Use exactly `gpt-6-luna` at medium reasoning for final documentation and delivery.

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
3. run every focused test and required benchmark/diagnostic, keeping
   correctness tests separate from benchmark measurements;
4. record exact commands and outcomes, artifact/binary identities, and
   hardware context in the task's Completion Report; and
5. leave commits and pushes to the delivery stage.

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

## Independent verification and one repair

After implementation, spawn a fresh Luna verification subagent. It must
independently read the contract and normative documents, inspect the complete
diff and Completion Report, and run every acceptance command. It must verify
the evidence rather than trust reported results, make no semantic fixes, and
return:

```text
VERIFICATION: PASS|FAIL
Commands run:
Results:
Unmet acceptance criteria:
File/line findings:
```

On the first ordinary verification failure, spawn one fresh Luna repair
subagent with the verifier findings. It may repair only those findings, may
not change architecture, and must not commit or push. Then spawn a fresh Luna
verifier and repeat the full independent gate.

If verification fails again, or if any stage encounters an architectural
failure, missing required dependency, or material contract ambiguity, mark the
task `BLOCKED`, record the reason in both the task file and ledger, and stop.
Do not use repair to invent a design. Missing hardware is incomplete evidence,
not a passing result.

## Delivery

Only after verification passes, spawn a fresh delivery subagent. It may
only:

1. confirm that the verified diff is unchanged;
2. record the final verification result in the Completion Report;
3. change the task and ledger from `IN_PROGRESS` to `DONE`;
4. create exactly one commit for the task; and
5. push the current branch to its configured upstream.

Delivery must not make semantic implementation changes. If substantive
content changed after verification, invalidate the result and run a fresh
verification before delivery.

Use the git-commit skill to create a proper git commit message.
Never force-push, rebase, merge, amend, or automatically resolve a non-fast-forward rejection. If push fails, preserve
the local commit, report the failure, do not claim successful delivery, and do
not start another task.

## Final report

Report the task ID and name, final status, implementation and verification
models, whether repair was used, exact verification commands, commit hash if
created, push result, task-file path, and any blocker or follow-up status.
