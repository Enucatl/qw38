---
name: no-poll
description: Coordinate independent subagents and long-running external jobs efficiently. Use for builds, full test suites, deployments, MCP jobs, or other work that may outlive a normal command response; do not use for short, synchronous tasks.
---

# Async work coordination

Use this skill to reduce unnecessary model involvement while preserving clear
ownership, evidence, and safe delivery. It governs execution mechanics only;
task-specific skills and user instructions remain authoritative for scope,
models, verification, and external mutation.

## Delegate by ownership

Delegate work when it has a self-contained outcome that can be reviewed from
its artifacts or report. Give each subagent the goal, relevant paths and
constraints, permitted mutations, acceptance commands, and a concise output
contract. Do not delegate an agent merely to wait for another process.

- Parallelize independent read-only investigation, review, or validation when
  their outputs do not depend on each other.
- Serialize agents that edit a shared working tree unless their file boundaries
  are proven disjoint and their results can be integrated safely.
- Keep the parent responsible for combining findings, deciding next actions,
  and handling external side effects such as commits or deployments when the
  task requires it.
- Start fresh role-specific agents when independence matters; do not ask a
  verifier to validate its own implementation.

## Long-running commands and jobs

Prefer one blocking wait over repeated model-driven status checks. For local
commands, use the execution environment's long wait/yield facility or a
blocking process wait, with a timeout appropriate to the declared command.
Capture the command, exit status, duration, and relevant output for later
verification.

For remote builds, deployments, MCP operations, and asynchronous APIs, choose
the first available completion mechanism in this order:

1. A blocking wait command or tool-level completion notification.
2. A callback, webhook, event stream, queued resume, or native watcher that
   resumes work after completion.
3. A lightweight local watcher that polls on behalf of the agent and writes a
   durable result record.
4. Direct polling by the agent only when none of the above exists.

Never use model turns merely to poll a long-running operation. Prefer a
blocking wait or event-driven notification. If the external system only
exposes a polling API, delegate that polling to a lightweight script or
process with an appropriate interval, and resume Codex only when the state
changes or the operation completes. Set the first wait long enough for normal
completion, then inspect or resume only on a completion signal, timeout, or
material failure.

## Status-only APIs

When an external system exposes only a status endpoint, polling is necessary,
but the model should not perform each poll. Launch a small watcher that:

- polls at a service-appropriate interval, with backoff where useful;
- exits on a terminal state or an explicit deadline;
- writes an atomic, durable result record containing the job ID, final status,
  timestamps, and final response or error; and
- returns a nonzero exit status for failed, cancelled, or timed-out jobs.

Keep the watcher alive independently of the model when the environment
supports it. Have the orchestration layer wait for the watcher or arrange a
queued follow-up/resume after its completion. On resume, consume the persisted
record before making a decision; do not begin a fresh polling loop.

Use an explicit status vocabulary. Treat unknown, malformed, or stale status
as an error requiring inspection, not as successful completion. Do not expose
secrets in watcher arguments, logs, or result records.

Illustrative pattern for a status-only API (adapt paths, terminal states, and
interval to the service):

```bash
while :; do
    result="$(service status "$job_id")"
    case "$result" in
        running) sleep 30 ;;
        *) printf '%s\n' "$result" >"$result_path.tmp" &&
           mv "$result_path.tmp" "$result_path"
           exit 0 ;;
    esac
done
```

## Completion and recovery

When a long operation finishes, verify its actual result rather than trusting
its state label. Read the persisted result or command output, check the exit
code and required artifacts, then run any task-required follow-up validation.
On timeout or failure, preserve diagnostic output, report the command/job
identity and terminal evidence, and stop or retry only within the task's
authorization and retry policy.
