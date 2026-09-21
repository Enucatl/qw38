---
name: run-implementation-task
description: Execute and safely deliver exactly one eligible QW38 implementation task from docs/implementation/task_ledger.md using fresh Cursor Task subagents. Use when advancing the sequential implementation ledger; do not use for ad hoc work or architecture redesign.
---

# Run Implementation Task

Coordinate exactly one `TASK-XXX`. The task file is the complete implementation contract: there is no planning stage or dossier.

## Authority and models

Use these sources, in descending task relevance:

- `docs/implementation/tasks/<TASK-ID>.md` for scope and acceptance.
- `docs/architecture/architecture-v0.md` for model/runtime architecture.
- `docs/implementation/technology-baseline.md` for toolchain, platform, and containers.
- `docs/implementation/code-standards.md` for C++/CUDA rules.

Start every implementation, repair, diagnostic, verification, and delivery stage as a new Cursor Task subagent with fresh context. Never resume an agent, reuse one in another role, or pass conversational history. Each prompt contains only the task ID, task path, normative paths, role, relevant prior-stage findings when needed, and expected output.

Use exactly:

- `cursor-grok-4.6-high` for implementation, repair, and difficult diagnostic or contract analysis.
- `composer-2.5` for independent verification, verification after repair, and delivery.

Never use `auto`, `-fast`, GPT-5.6, or another fallback. If either model is unavailable, stop and report it.

## Select and admit

Accept zero or one explicit task ID. With an ID, select exactly it. Without one, scan the ledger top to bottom and select the first `TODO` whose dependencies are all `DONE`. Ledger order is the only implicit priority.

Before any mutation, reject the invocation if:

- more than one ID was supplied, the ID is unknown, its status is not `TODO`, its dependencies are incomplete, or no eligible task exists;
- `docs/implementation/tasks/<TASK-ID>.md` or any normative document is missing;
- the ledger row and task file disagree on identity, name, status, or dependencies;
- the worktree is not clean, including untracked files, or the current branch has no upstream;
- the contract contains an unresolved `TBD`, placeholder, contradiction, or lacks objective acceptance criteria.

If acceptance requires an architectural decision not specified by Architecture V0, stop with `TASK_CONTRACT_INCOMPLETE`. Do not invent architecture.

After admission, change the ledger row to `IN_PROGRESS` and do the same in the task file when it has a status field. Do not activate downstream tasks.

## Implement

Start a fresh `cursor-grok-4.6-high` implementation agent. Require it to read the complete task file and only relevant normative sections, implement only the task, obey LOCKED decisions, preserve tuning/default distinctions, run every focused test, update the Completion Report, and neither commit nor push. It must not redesign Architecture V0.

Unrelated discoveries go in the Completion Report as `FOLLOW_UP_REQUIRED`; do not expand scope, create roadmap tasks, or edit downstream specifications. If missing work prevents acceptance, block the current task.

If a LOCKED decision is impossible, inconsistent, or incompatible with required semantics, require this report:

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

Then stop, mark the task `BLOCKED`, record the blocker in the ledger and task file, and do not change Architecture V0, invoke repair, commit, or push.

## Verify and repair once

After implementation success, start a fresh `composer-2.5` verifier. It independently reads the contract and normative documents, inspects the complete diff and Completion Report, and runs every acceptance command. It must verify evidence rather than trust reported test results, make no semantic fixes, and return:

```text
VERIFICATION: PASS|FAIL
Commands run:
Results:
Unmet acceptance criteria:
File/line findings:
```

On the first ordinary failure, start one fresh `cursor-grok-4.6-high` repair agent with the verifier findings. It may repair only those failures, may not change architecture, and must not commit or push. Then start a fresh `composer-2.5` verifier and repeat full independent verification.

If verification fails again, mark the task `BLOCKED`, record the remaining failure, and stop. At any point, route an architectural failure directly through `ARCHITECTURE_BLOCKER`; never use repair to invent a design.

## Deliver

Only after verification passes, start a fresh `composer-2.5` delivery agent. It may only:

1. confirm the verified diff is unchanged;
2. record the final verification result in the Completion Report;
3. change the task and ledger from `IN_PROGRESS` to `DONE`;
4. create exactly one commit for the task and push the current branch to its configured upstream.

The commit uses a concise Google-style subject, at most 50 characters where practical, and a body explaining intent. Delivery makes no semantic implementation change. If substantive content changed after verification, invalidate the result and run a fresh verification before delivery.

Never force-push, rebase, merge, amend, or resolve a non-fast-forward failure automatically. If push fails, preserve the local commit, report failure, do not claim successful delivery, and do not start another task.

## Final report

Report the task ID and name, final status, implementation and verification models, whether repair was used, exact verification commands, commit hash if created, push result, task-file path, and any blocker or follow-up status.
