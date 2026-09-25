---
name: run-implementation-task-codex
description: Execute and deliver exactly one eligible QW38 implementation task from docs/implementation/task_ledger.md, with main-thread implementation, independent Astra verification, and final Luna delivery. Use when advancing the sequential implementation ledger; do not use for ad hoc changes or architecture redesign.
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

The main thread handles admission, implementation, repairs, evidence
collection, and any blocked-task records. During that work, spawn only one
separate verification agent:
`gpt-6-astra` at high reasoning, with `fork_turns: "none"`. Give it the task ID
and path, relevant normative paths, all changed and new files, and evidence
locations, and the review output required below. Reuse that Astra agent for the
second review pass if needed. After verification passes, spawn a fresh
`gpt-6-luna` agent at medium reasoning, with `fork_turns: "none"`, for
documentation, bookkeeping, commit, and push. Do not delegate implementation,
repair, or evidence commands.

## Admission

Accept zero or one explicit task ID.

- With an ID, select exactly that task.
- Without one, scan the ledger from top to bottom and select the first `TODO`
  task whose dependencies are all `DONE`.
- Ledger order is the only implicit priority.

An explicitly selected task may be `IN_PROGRESS` when the user manually
approves continuing that task. Treat approval as specific to the task ID and
continuation request; a request to start or select a task is not, by itself,
approval to resume existing work. On approval, continue from the recorded state
and preserve prior implementation, evidence, and completion-report content; do
not reset or repeat completed work unless the task's remaining acceptance
criteria require it. Without explicit approval, stop at admission, report the
current status and ask whether the user wants to continue that task. `IN_PROGRESS`
tasks are never selected implicitly when no ID is supplied.

Before any mutation, reject the invocation when any of these gates fail:

- more than one ID was supplied, the ID is unknown, its status is neither
  `TODO` nor an explicitly approved `IN_PROGRESS` continuation, its
  dependencies are incomplete, or no eligible task exists;
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

After admitting a `TODO` task, change its ledger row to `IN_PROGRESS` and update
the task-file status when it has a status field. For an approved
`IN_PROGRESS` continuation, preserve the existing status and resume the
unfinished work in place. Do not activate downstream tasks.

## Implementation

In the main thread:

1. read the complete task file and only the relevant sections of the normative
   documents;
2. implement only the task, obeying `LOCKED` decisions and preserving the
   distinction between tuning defaults and required semantics;
3. run focused commands on the final candidate that establish every acceptance
   criterion and each required test, benchmark, or diagnostic, keeping
   correctness tests separate from benchmarks;
4. preserve exact commands, results, logs, artifact/binary identities, and
   hardware context for Astra to review and Luna to record in the task's
   Completion Report; and
5. leave the final Completion Report, `DONE` status changes, commit, and push
   to the final Luna agent.

Unrelated discoveries belong in the Completion Report as
`FOLLOW_UP_REQUIRED`; do not expand scope, create roadmap tasks, or edit
downstream specifications. If missing work prevents acceptance, stop and
block the task.

If a locked decision is impossible, inconsistent, or incompatible with the
required semantics, report this and stop without repair:

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

The main thread makes complete command evidence available to Astra. Missing
hardware or a skipped required check is incomplete evidence, not a pass. Apply
the verification scope in `code-standards.md`; do not run unrelated suites or
repeat unchanged commands merely for a second witness.

Spawn the Astra reviewer only when the candidate and its evidence are ready.
Give it clear, self-contained context: the task ID and contract, relevant
normative paths, the full candidate diff and all new untracked files, evidence
and log locations, and this review brief. Ask it to read the contract, relevant
architecture and code standards, the complete diff, changed code and its
production callers, tests, and evidence. Review independently for:

- correct behavior at numerical, state, lifetime, error, and CUDA boundaries
  relevant to the change, including failure and continuation paths;
- compliance with Architecture V0's locked decisions, the technology baseline,
  and `code-standards.md`'s design philosophy and concrete rules;
- the smallest implementation that fully meets the task, without speculative
  abstractions, extra dependencies, or unrelated changes;
- discriminating tests and complete, current evidence for every acceptance
  criterion, including resource or schedule effects where relevant.

Astra inspects code and evidence without editing files or running delivery
steps. The main thread runs any targeted commands Astra requests. Astra must
not accept a passing test log as proof of code correctness. Return all material
findings in one pass with file/line, the violated contract or failure scenario,
and a concrete correction:

```text
REVIEW: PASS|CHANGES_REQUIRED|BLOCKED
Reviewed candidate:
Acceptance and evidence gaps:
Code findings (severity, file/line, contract, impact, correction):
Targeted evidence requested:
```

`PASS` requires no code findings, acceptance gaps, or evidence requests. Astra
uses `CHANGES_REQUIRED` for remediable code or evidence gaps and `BLOCKED` for
an architectural failure or unavailable required hardware. After the first
`CHANGES_REQUIRED`, the main thread makes the necessary corrections and runs
affected or targeted checks. Send the same Astra agent the findings, revised
complete candidate, and refreshed evidence for one second review. If that
review still reports any code issue, acceptance gap, or incomplete evidence,
stop for human intervention and mark the task `BLOCKED`; do not start another
repair or review round.

An architectural failure, missing required dependency, or material contract
ambiguity also marks the task `BLOCKED`. For every `BLOCKED` outcome, record the
reason in both the task file and ledger, then stop. Do not use repair to invent
a design.

## Delivery

Only after Astra review passes, the main thread confirms that the reviewed
implementation and evidence are unchanged. Then spawn the final Luna agent
with the task ID and path, ledger path, review result, exact acceptance
commands and evidence, and the intended delivery steps. Luna:

1. records the command evidence and review result in the Completion Report;
2. changes the ledger row and any task-file status from `IN_PROGRESS` to `DONE`;
3. pauses for the main thread to check those edits; then
4. as part of delivery, the main thread removes task-specific cache folders
   for tasks whose ledger status is `DONE`, including the task just completed
   and earlier completed tasks. For a ledger ID such as `TASK-018`, match only
   `.cache/task018` and `.cache/task018-*` directories. First ensure the
   Completion Report preserves any required evidence and no live process is
   using the folders.
   Keep caches for `TODO`, `IN_PROGRESS`, and `BLOCKED` tasks, along with shared
   caches that are not task-specific. If permissions prevent removal, report
   the remaining path and size rather than broadening the cleanup.
5. stages only task files, checks the staged diff, creates exactly one commit
   for the task, and pushes the current branch to its configured upstream.

Luna may edit only task documentation and ledger status; it must not change
implementation code, tests, or underlying command results. The main thread
checks the documentation for accuracy and confirms that the reviewed code and
evidence stayed unchanged before directing Luna to commit and push. If a code or
evidence change becomes necessary, the main thread handles it and obtains an
Astra review within the two-pass limit before delivery continues. If both
review passes have already been used, stop for human intervention.

Luna uses the `git-commit-message` skill to create a proper commit message.
Never force-push, rebase, merge, amend, or automatically resolve a
non-fast-forward rejection. If push fails, preserve the local commit, report the
failure, do not claim successful delivery, and do not start another task.

## Final report

Report the task ID and name, final status, main-thread implementation model,
Astra review result and number of passes, Luna delivery result, exact
acceptance commands and results, commit hash if created, push result,
task-file path, and any blocker or follow-up.
