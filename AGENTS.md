# Agent instructions

## Long-running commands

- Any command reasonably expected to take longer than 10 seconds to execute MUST use the `wake-run` skill. This includes tests, GPU evaluations, builds, benchmarks, and similar work.
- Follow the installed `wake-run` skill's launch and continuation instructions. Finalize the command before launching it, and resume work when the skill delivers its completion notification.
