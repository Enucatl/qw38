# TASK-001 — Reproducible CUDA build foundation

## Status
TODO

## Milestone
M0 — Reproducible implementation foundation

## Purpose
Create the architecture-neutral container, build, test, and GPU-smoke foundation on which every V0 task runs.

## Depends on
- None.

## Normative references
- `docs/architecture/architecture-v0.md` — Implementation baseline
- `docs/implementation/technology-baseline.md`
- `docs/implementation/code-standards.md`

## Architecture decisions consumed
| Decision ID | Contract | Type |
|---|---|---|
| I-01–I-05 | Linux/Docker/NVIDIA runtime, CUDA 13.4.x, RTX 5090 `sm_120`, C++23/CUDA C++23 | LOCKED |
| I-07 | No old-GPU or old-toolchain compatibility in V0 | LOCKED |

## Starting point
The repository has documentation and Python analysis scripts only: no Dockerfile, CMake project, native sources, native tests, benchmarks, or engine CLI.

## Scope
- Pin a CUDA 13.4.x development image/base distribution and compatible host compiler/CMake versions; use an immutable digest when obtainable.
- Add the minimal CMake project with separate ordinary C++ and narrow CUDA targets, Debug/Release support, native `sm_120`, C++23, CUDA C++23, CTest, and a distinct benchmark target area.
- Add host `std::expected` compile smoke, CUDA kernel/runtime smoke, and documented container build, test, and GPU-run commands.
- Validate and report driver/container compatibility from inside the container.

## Out of scope
- Engine functionality, model loading, `.qw38`, performance kernels, old targets, PTX fallback, packaging, CI, or a generic dependency framework.

## Required interfaces
- One documented container build command, one CMake configure/build command, and one normal CTest command.
- A trivial CUDA executable that launches a kernel, checks its result, and reports the active device and compiled target evidence.

## Required semantics
Smoke failures return nonzero. The CUDA test must execute on hardware, not merely compile.

## Data representation
No model data. Generated build output remains outside source-controlled source directories.

## Implementation constraints
- Keep host-only code outside CUDA compilation.
- Do not add compatibility shims for pre-C++23 or pre-Blackwell systems.
- Benchmarks must not run as correctness tests.

## Tuning defaults
- Native CUDA architecture: `120` / `sm_120`.

## Expected files/modules
Root build/container files plus minimal `src/`, `cuda/`, `tests/`, and `benchmarks/` skeletons justified by the existing empty native tree.

## Tests required
### Unit tests
- Compile and execute a host C++23 `std::expected` smoke.
### Reference/numerical tests
- None.
### Integration tests
- Build Debug and Release in the container; run CTest; run the CUDA smoke on RTX 5090; inspect the produced binary for native `sm_120` code.

## Benchmark required
No.

## Acceptance criteria
- [ ] Image/tool versions are pinned and documented.
- [ ] Host and CUDA C++23 targets compile.
- [ ] CUDA smoke executes correctly on RTX 5090.
- [ ] Binary inspection proves the intended Blackwell target.
- [ ] The normal test command passes.
- [ ] No engine behavior or compatibility work was added.

## Architecture blocker rule
If a locked decision prevents correct implementation, stop and report `ARCHITECTURE_BLOCKER` with: Decision ID; Attempted implementation; Observed problem; Evidence; Why this is architectural rather than tuning; Smallest plausible alternative; Affected downstream tasks. Do not silently substitute a design.

## Completion report
### Result
DONE | BLOCKED
### Changes made
### Tests run
Exact commands and results.
### Benchmark results
Not required.
### Architecture blocker
None, or the full blocker report.
### Follow-up observations
Concrete downstream observations only; do not redesign future tasks.

