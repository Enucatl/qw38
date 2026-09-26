# Replan QW38 after TASK-027 for fast delivery

You are GPT-6 Astra. Revise the remaining implementation plan in this repository. **Do not start implementation, run benchmarks, build, regenerate artifacts, or launch evaluations.** This task produces an actionable revised plan and reconciled planning documents.

## Objective and authority

Our objective is to build a fast, memory-efficient, usable inference engine quickly. We now have enough architecture knowledge, working code, quality evidence, and TASK-027 profiling to make educated engineering choices. We do not need rigorous proof of every choice before implementing it.

Treat this as my instruction to revise TASK-028 onward, including their ordering, scope, dependencies, evidence requirements, and completion criteria. Existing future task contracts are editable planning material, not obstacles to this replan. Preserve completed TASK-001–027 records and their historical results. Do not mark future implementation complete.

Aim for the straightforward design likely to get us most of the way there. Prefer a concrete, defensible default with a cheap check and fallback over a research program to establish the optimum. “80% of the way” describes engineering progress, not permission to promise an unmeasured percentage of peak performance.

## Read the relevant evidence

Inspect repository instructions, the implementation ledger, TASK-027's completion report and compact profile summaries, and the current TASK-028–032 specifications. Consult TASK-019/020 findings and TASK-021–026 implementation reports where they inform actual choices. Trace relevant runtime/kernel code sufficiently to distinguish existing functionality from missing work. Do not reread every document or parse all raw traces by default.

Known starting points to verify and use:

- The accepted Q4_K/Q8 engine works, but TASK-027 shows substantial performance gaps.
- At 32K, prefill takes about 208.46 seconds versus 12.37 seconds for the comparator; the attention scan alone accounts for about 177.87 seconds. This is already sufficient reason to prioritize a better attention algorithm/schedule. Do not defer this obvious bottleneck merely because the old ledger puts long-context work later.
- At populated decode length 4096, projections account for about 3.19 seconds of 4.28 seconds host elapsed. At 32768, attention and projections both matter.
- Current prefill expands bounded packed-weight tiles to BF16 for cuBLAS. Native compute-compatible representations could remove substantial conversion and traffic, but faster projection kernels alone cannot fix the dominant 32K attention cost.
- Native SM120 NVFP4/MXFP4 feasibility has been demonstrated. Production GPU activation packing and complete consumer integration remain distinct work. Prior family screening makes MLP gate/up a plausible FP4 starting point; sensitive families do not justify an indiscriminate all-FP4 conversion.
- Capacity currently fits. Reducing unnecessary traffic, conversions, temporary materialization, and launches is more immediately useful than shrinking every persistent buffer. The 144 MiB GDN state is not an automatic priority.

Use these observations to choose a direction. Do not rerun the baseline to rediscover them.

## Design the calculation chain as a whole

Give the revised tasks a concrete intended dataflow, rather than a list of isolated kernel experiments:

1. Choose quantization, scale representation, physical layout, and consuming kernels together. Prepare resident weight representations once. Avoid request-time full-weight repacking and unbudgeted duplicate views.
2. Generate activation codes/scales on the GPU where needed. Reuse compatible packed operands across projections. Fuse normalization/packing or producer/consumer epilogues when the dependencies make it straightforward and useful.
3. Define where values live and how long: persistent weights/KV/recurrent state in device memory; reusable bounded workspace for cross-kernel intermediates; registers/shared memory for local tiles, reductions, and accumulators where practical. Do not assume intermediates survive arbitrary kernel boundaries in registers/shared memory or that we can explicitly pin everything in cache.
4. Address the long-context attention scan with a reasonably efficient tiled, reuse-oriented attention path. Inspect what exists and select a practical implementation route; avoid a broad algorithm bake-off.
5. Use suitable prefill and decode consumers. Large-M tensor-core performance does not establish the best one-token decode path. A simple GEMV fallback is acceptable. Start with a reasonable dispatch rule and change it when an actual result gives a reason.
6. Reuse existing kernels, installed libraries, and established implementation patterns before writing new infrastructure. Avoid giant fusion projects, generalized autotuners, or new frameworks.

Preserve model equations, tensor/tokenizer identity, causal behavior, session/state correctness, and primary-language scope. Keep existing FP32 residual/accumulation/recurrent arithmetic and BF16 KV/history as defaults unless a specific, worthwhile change is explicitly planned. Do not add unrelated features.

## Use proportionate validation

- Engineering judgment is sufficient to select the first implementation. Label uncertain choices as hypotheses and proceed; uncertainty alone is not a blocker.
- Run basic correctness and numerical checks once for the affected path. Repetition needs a reason: a failure, changed code, or a specific unresolved concern.
- Preserve the existing single-run performance policy: one execution per selected workload/engine, zero warmups, no repetitions, no medians/p99/bootstrap/confidence-interval requirements. Label instrumentation and first-use costs; report observations without claims of statistical significance.
- Use a small representative end-to-end check to reject obvious regressions and confirm that the intended path runs. Add a focused measurement only when it could change an immediate implementation decision. Do not require exhaustive per-stage attribution, format comparisons, ablations, instruction audits, or tile searches for every change.
- Reuse TASK-027 evidence where valid. Do not rerun the comparator or every context length at every intermediate step without a concrete need.
- Distinguish development checks from final promotion. Consolidate applicable model-quality, long-context, replay, memory-capacity, and matched performance checks around a coherent final candidate instead of repeating the whole suite after each small task. Preserve meaningful quality criteria; smaller development checks must not be relabeled as full acceptance.
- Keep the 216-case suite optional and human-initiated. Do not launch or require it. Identify the existing routine quality coverage to retain at final promotion, and explicitly reconcile any changed timing or ownership of those obligations.
- Record commands, observed results, and enough identity/context to interpret them. Avoid creating a new evidence bureaucracy.

Simplify inherited acceptance criteria that force research before useful implementation. Missing proof of an optimum is not a blocker. A demonstrated correctness failure, unsupported required operation, or capacity failure is.

## Deliverables

Edit `docs/implementation/task_ledger.md` and the remaining task specifications to create the shortest coherent delivery sequence. Merge, reorder, narrow, or replace future tasks when useful; reconcile task IDs, dependencies, milestones, and the critical path. Update directly conflicting policy/architecture text only where necessary so future agents do not inherit contradictory requirements. Preserve historical evidence and distinguish this amendment from prior policies.

For each revised task, specify concisely:

- The concrete working behavior it delivers and its dependencies.
- The chosen first implementation and the code areas likely involved.
- Essential representation, memory-lifetime, and producer/consumer contracts.
- The smallest useful one-run validation and a clear completion condition.
- A fallback or trigger for reconsideration only where there is a real uncertainty.

Make a recommendation. Do not return a menu of equally weighted architectures or ask me to choose routine engineering details. Do not preserve six tasks merely because six currently exist. Prefer a few substantial integration steps over many isolated experiments; retain separate tasks only when they produce useful, reviewable milestones.

Finish with a concise account of the new sequence, why it targets the known bottlenecks, which assumptions we are willing to implement without stronger proof, and which expensive experiments/checks were removed or deferred. Verify the edited plan's internal consistency without running engine workloads. Stop after the planning documents are revised; do not execute the first implementation task.
