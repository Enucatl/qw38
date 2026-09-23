# Assessment after the first 15 implementation tasks

Reviewed 2026-09-22 at source commit
`130063c903d4d31610b6491a5f35ac2ca2eaf9d5`, after the requested 30-minute delay.
This is a repository assessment and implementation handoff, not an implementation
of the recommendations. Only this document is changed by the review.

## Overall assessment

The implementation is on track at the component level. The intended format,
compiler, quantizers, CUDA ownership, MLP, GDN, and segmented attention pieces
exist. The repair batch materially improved explicit operand validation,
canonical artifact ownership, numerical edge cases, residual preservation,
state snapshots, and deterministic continuation. A freshly rebuilt Release fast
suite passes all 36 tests. TASK-016 is the right next integration checkpoint.

However, “the review items were fixed” should not yet mean “the foundation is
ready for unrestricted composition.” There are residual correctness issues in
verification policy and session lifetime/failure handling. Several repairs were
tested as isolated helpers or successful local operations without proving the
contract of their production callers. Excluded benchmark targets have also
drifted from the repaired interfaces.

The strongest conclusion from this batch is to strengthen integration contracts
and completion evidence, not replace the architecture. There is no current
evidence justifying a different quantizer, recurrent-state precision, activation
transport, packed weight view, or prefill algorithm. Full-model source agreement,
behavioral quality, production prefill, and end-to-end performance remain future
TASK-017–023 evidence; their absence is not a defect in TASK-001–015.

The most urgent work below should precede accepting TASK-016. Other items can
be incorporated into their named downstream task without expanding that task
into an architecture rewrite. Priorities describe engineering impact: P1 blocks
trustworthy integration or artifact verification; P2 is a bounded correctness,
resource, evidence, or maintenance repair. “Confirmed” means established from
current source or an executed check. New counterexample tests proposed below
were not written or executed during this documentation-only assessment.

## Verification performed and limits

The worktree was initially clean. The review used independent read-only runtime,
compiler/format, and architecture/documentation assessments, then checked their
claims against the current source. References below use paths and line anchors
at the reviewed commit; symbol names remain the durable navigation aid.

Environment: NVIDIA GeForce RTX 5090, driver `590.48.01`, reported memory
`32607 MiB`. Existing Release configuration uses GCC 14 and CUDA 13.4.1.
The development image was resolved and executed by immutable local image ID:
`sha256:be0903b40ab2e14ec1b5ba285655219cc9e521cd1455b070a6dfdb68ed4e4bfb`.

Executed successfully:

```bash
docker run --gpus all --rm -u 175200003:175200003 \
  -v /home/user/qw38:/workspace -w /workspace \
  sha256:be0903b40ab2e14ec1b5ba285655219cc9e521cd1455b070a6dfdb68ed4e4bfb \
  bash -lc 'cmake --build build/release -j 4 && ctest --test-dir build/release --output-on-failure -LE extended'
```

Result: build passed; **36/36 tests passed**, CTest elapsed time 25.61 seconds.
An initial invocation incorrectly used UID/GID 1000 and failed to link due to
output permissions; retrying with the actual workspace UID/GID above resolved
that invocation error. It was not a source/build-system failure.

Additional targeted checks exposed excluded-target regressions:

```text
cmake --build build/release --target qw38_bench_mlp qw38_bench_gdn_mixer -j 4
  FAIL: benchmarks/mlp_bench.cpp:46: TensorView has no member 'writable'

cmake --build build/release --target qw38_bench_gdn_mixer -j 4
  PASS
build/release/benchmarks/qw38_bench_gdn_mixer
  FAIL: invalid_argument field=s S must be FP32 [48,48,128,128]
        in [layer,value_head,value,key] order
```

Both checks used the same container and Release build. The GDN invocation
failed during setup, so no performance measurement is claimed. No full authority
checkpoint scan, extended full-vocabulary run, Debug rebuild, sanitizer sweep,
or end-to-end model evaluation was performed. The passing fast suite establishes
existing regression coverage, not absence of the gaps below.

### Verification cadence recommendation

Run the full authoritative-checkpoint reconstruction/quantization pass once
when promoting a new artifact for release or sharing, and once on a candidate
when a code change can alter the bytes or the correctness decision. Concrete
triggers include changes to:

- checkpoint parsing/classification, source tensor selection, or source identity
  metadata (never source tensor content hashes; see the binding rule below);
- quantizer equations/rounding, Q4/Q8 packing, BF16 transforms, layouts, or
  emitted tensor bytes;
- artifact schema, graph bindings, policy selection, or compiler identity fields;
- reconstruction, semantic-schema comparison, or the full-verification path.

For a code change, run it on the candidate before merging or promoting that
change if the change affects an authoritative artifact's result or whether it
passes verification. When releasing an artifact, run it once for that exact
checkpoint, compiler revision, format policy, and output artifact. A later
release of a different checkpoint or output is a new candidate and needs its
own run. If those identities and bytes are unchanged, retain the prior
verification record instead of repeating the hour-long pass.

Examples that do **not** trigger it: documentation-only edits; changes limited
to runtime execution kernels after compilation; benchmark-only changes; and
tests that use synthetic/small checkpoint fixtures. Run the fast suite and
focused fixture tests for those changes. Do not add a nightly full-checkpoint
run by default. This is a promotion or focused engineering checkpoint, not a
per-commit or per-PR CI job.

Every commit runs the fast suite and small fixture tests. Those tests exercise
identity, policy, and corruption behavior without scanning or quantizing the
50+ GB authority checkpoint. Metadata-only inspection stays cheap. The normal
artifact compile necessarily reads and transforms its inputs; the separate
reconstruction verifier must be opt-in so it does not add a second full source
read and quantization pass by default. At the reviewed commit,
`CompileOptions::verify_reconstruction` defaults to `true`
(`src/compiler/compile.hpp:24`) and the CLI disables it only with `--no-verify`
(`src/compiler/main.cpp:29`). Change that default/flag semantics as part of the
follow-up so code matches this cadence. Print whether verification ran and which
artifact, source-metadata, compiler, and policy identities it covered. Never
include a BF16 source tensor or shard content hash in that record.

## Required repairs

### AR-01 — Withdrawn by explicit source-checkpoint policy

The earlier proposal to hash cached BF16 source tensors or their shard files is
rejected and must not be implemented, tested against the authority checkpoint,
or suggested again. The authoritative rule is in the
[code requirements guide](docs/implementation/code-standards.md#cached-bf16-checkpoint-payload-policy):
after the one-time successful download, never compute, store, or compare a
content digest over cached BF16 tensor contents, directly or by hashing
containing shard files, for any reason. The `.qw38` artifact's own integrity
records remain separate and required by its format.

Keep `source_hash` metadata-only. Configuration, index, names, shapes, dtypes,
declared sizes, and safetensors headers may be validated without reading or
hashing payload contents. Do not add source-content identities, payload hash
audits, release-time shard scans, or mutation tests over cached BF16 files.
This item is closed as a policy decision; it is not a future implementation task.

### AR-02 — P1, confirmed: check artifact policy and schema from metadata

**Evidence.** `verify_compiled_artifact` ignores its policy argument at
`src/compiler/compile.cpp:886`. Later it selects quantization/layout/geometry
from each artifact record. `verify_identity_artifact` delegates with
`IdentityBf16` at lines 971–976, but that policy is ignored, so a quantized
artifact may pass the function named for identity artifacts if its payloads
reconstruct as expected. The verifier also compares tensor bytes without first
establishing that the artifact's complete tensor directory and graph bindings
match the expected model schema. `src/format/schema.cpp:2468` validates
individual graph records; it does not prove complete-model membership.

This is a policy/schema validation gap. It is not a request to hash the cached
BF16 checkpoint. The current `verify_compiled_artifact` also performs the
separate, expensive payload reconstruction pass; keep that work explicitly
separate from the schema decision described here.

**Implementation.** Add a metadata-only comparison that derives the expected
schema from parsed checkpoint/config/index metadata and the requested policy,
using the existing schema construction path. It must not read, hash, or digest
BF16 tensor payloads or their shard files. Compare exact tensor membership,
logical geometry, family storage/quantizer/layout/mapping, semantic scope, graph
bindings, and canonical sharing. Ignore writer-assigned offsets, ordering where
nonsemantic, and integrity-record bytes for this semantic comparison. Report
the differing field and tensor/binding. Avoid a duplicate model-schema table.
Have the payload reconstruction verifier call this check before it reads source
payloads, while allowing tests to invoke the metadata check alone. The compile
option that controls full reconstruction remains separate.

**Acceptance.** Use small synthetic metadata/artifact fixtures in ordinary CI:
identity artifacts pass only `IdentityBf16`, production artifacts pass only
`ProductionV0`, and mismatches fail before any source tensor payload is read.
Cover
missing/swapped bindings, extra tensors, wrong family format, and changed shape
with unchanged element count. Regenerate `.qw38` integrity records in mutated
fixtures where needed so the semantic comparison is reached; these are hashes
of test artifacts, not cached BF16 source data. Require mismatches to be rejected
from metadata before any source tensor payload is read. No authoritative checkpoint,
source payload hash, full quantization, or full reconstruction is required for
AR-02. Keep the separate full reconstruction cadence above for cases that
explicitly need payload-level evidence.

### AR-03 — P1, confirmed: define failed-state recovery and complete-token commit

**Evidence.** `cuda/stream.cpp:84` performs the real synchronization before
returning its injected failure. `Session::reset` can therefore zero all device
state and return before resetting host counters (`src/runtime/session.cpp:368`).
Restore can replace device state and leave old metadata (`:441`). Tests at
`tests/runtime_session_integration_test.cpp:409` and `:431` check unchanged host
counters, then immediately retry; they never establish a usable intermediate
session. GDN mutates history/S before its final commit
(`src/runtime/gdn.cpp:1302`, `:1364`). Attention commits populated length after
preparation (`src/runtime/attention.cpp:937`) before its core can fail (`:1000`).

**Decision proposed.** Use explicit session health and a completion boundary.
Validate ordinary argument/position/plan errors before enqueueing mutation.
After work may have mutated persistent state, any execution, synchronization,
or subsequent commit failure poisons the session. Reject further execution and
snapshot creation until a successful reset or valid snapshot restore reestablishes
coherent state. An unrecoverable CUDA context error requires recreation. Do not
copy all persistent state for automatic rollback on every token.

For TASK-017, maintain one globally committed token position. Treat progress
through individual layers as pending until all layers and required output work
complete successfully. Failure after layer N cannot be retried as though layers
0 through N-1 had not advanced. Snapshot only at a completed operation boundary;
retain explicit component-test support for partial graphs.

**Implementation.** Put health/commit ownership in session execution state and
route all session-backed stateful operations through it. Specify recovery
behavior for reset/restore failure as well as normal decode failure. Low-level
explicit-view APIs must either carry the corresponding execution context or
document that their caller owns recovery; do not let them bypass production
session health.

**Acceptance.** Inject failures after device mutation in reset, restore, GDN,
and attention core after preparation. Verify execute/save rejection while
poisoned and exact continuation against a clean control after successful
recovery. Verify preflight rejection does not poison or mutate. Extend coverage
at TASK-017 to a failure after an earlier layer successfully advanced. Test
device bytes and host metadata together, not counters alone.

### AR-04 — P1, confirmed: make plan-bound session metadata survive supported moves

**Evidence.** `src/runtime/session.hpp:87` defaults move construction/assignment.
Metadata arrays are embedded in the session, while slots retain pointers into
them (`src/runtime/session.cpp:644`, `:653`, `:665`). Moving a bound session moves
its device allocations but copies those arrays to new addresses. Existing plans
then update the moved-from counters; destruction of that object leaves dangling
host pointers. Current move tests do not exercise this bound-plan sequence.

**Decision proposed.** Allocate a small move-stable session execution-state
object during creation, containing counters and AR-03 health/commit metadata.
Moving ownership then preserves the addresses borrowed by existing plans.
Device allocations already retain their addresses. Plans remain borrowed:
model/session destruction, shutdown, or replacement invalidates their plans.
Document that move assignment invalidates plans belonging to the replaced
destination, while plans for the moved source follow its ownership.

**Acceptance.** Bind GDN and attention plans, move the session, destroy the
moved-from object, execute the existing plans, and inspect the destination's
snapshot/counters. Cover move assignment separately. No hot-path allocation is
introduced. Do not turn this into shared ownership of every tensor or a generic
runtime object registry.

### AR-05 — P1, confirmed: enforce one session stream at every session-backed binder

**Evidence.** `bind_mlp_plan(Model, Session, ...)` at
`src/runtime/mlp.cpp:456` and attention preparation at
`src/runtime/attention.cpp:761` accept an arbitrary stream. GDN front and
recurrence session binders also lack the complete mixer's check. Complete GDN
does compare against the session stream (`src/runtime/gdn.cpp:1124`).
Save/reset/restore synchronize the owning stream, so alternate-stream execution
can race lifecycle operations or scratch reuse.

**Implementation.** Prefer deriving the production execution stream from the
session. If existing signatures retain an explicit stream, apply a small common
preflight requiring the same native stream and matching model/device ownership
at every session-backed binder. Explicit-view diagnostic binders may retain
caller-managed ordering, stated in their contract.

**Acceptance.** Reject a second same-device stream before launches or mutation
for MLP, attention, GDN front/recurrence, and complete GDN. Conditionally test a
different-device model/stream when hardware permits. Preserve valid binding
through supported Runtime movement/sharing. A single-device success test does
not establish stream ownership.

### AR-06 — P2, confirmed: close remaining checked-geometry holes

**Evidence.** `src/runtime/view.hpp:80` loops over unchecked rank in an eight-entry
extent array and multiplies sizes unchecked when converting to WorkspaceView.
The comment explicitly introduces this production constructor for test
convenience. `extents()` at line 44 and `regions()` at line 96 expose spans with
unchecked public counts. `Session::kv_populated` at `session.hpp:113` indexes an
array without the setter's bounds check.

Separately, `src/compiler/transforms.cpp:20` multiplies element count by two
unchecked; dense and convolution transforms first multiply dimensions unchecked
at lines 82, 103, 125, and 145. For example, an empty span with dense dimensions
`N=8, K=2^61` passes positivity/divisibility checks and wraps its element count
to zero, then enters a loop that accesses empty storage. This is a public-helper
counterexample, not a claim that validated checkpoint geometry currently reaches
that path.

**Implementation.** Replace fallible implicit workspace conversion with a
checked factory or an explicitly test-local helper. Bound rank/count/index before
array access; check products before evaluating them. Do not silently clamp
invalid descriptors. Use checked results for indexing and size validation in
both transform directions. Preserve typed allocation/error boundaries.

**Acceptance.** Small host tests for rank 9/255, region count 13/255, index
16/UINT32_MAX, element and byte-product overflow, zero dimensions, short/long
payloads, and valid round trips. Split runtime and compiler repairs if that
keeps commits focused. No large allocation or authority scan is needed; a focused
sanitizer run is useful for the original memory-access failure.

### AR-07 — P2, confirmed: keep BF16 verification bounded too

**Evidence.** Quantized verification now uses bounded chunks, but
`src/compiler/compile.cpp:959` still calls full BF16 reconstruction. The identity
mapping copies the entire payload at lines 442–443; tiled reconstruction allocates
a full vector in `src/compiler/transforms.cpp:106`. A `248320 x 5120` BF16
embedding/head is 2,542,796,800 bytes, approximately 2.37 GiB. Production
verification still copies the embedding; identity control additionally copies
tiled large weights. The existing memory regression exercises only the quantized
verifier. Sampling peak RSS after verification is already fixed and should stay.

**Implementation.** Compare identity-layout spans directly. Verify tiled BF16
with fixed-size tiles or row groups; compare convolution mappings without a full
reconstruction buffer. Keep a materializing helper only for callers explicitly
requesting materialized output. State whether a memory measurement covers owned
temporary memory or total RSS including mapped pages.

**Acceptance.** Corrupt a byte in each supported BF16 layout and require a
failure using small fixtures in ordinary CI. Measure row-major and tiled BF16
verification in isolated processes over increasing synthetic sizes, showing
bounded owned scratch; do not use 2.37 GiB authority tensors for this test. Avoid
process-lifetime RSS high-water marks hiding additional allocations. The single
full-model reconstruction run belongs at artifact promotion or after a relevant
compiler/quantization change, not on every commit.

### AR-08 — P2, confirmed: define the closed Runtime API contract

**Evidence.** `Runtime::shutdown` clears `stream_`
(`src/runtime/runtime.cpp:43`). `upload` dereferences it at line 63;
`create_session` passes it to a creation path that dereferences it. The public
header documents shutdown but no precondition making those calls invalid.

**Implementation.** Return a typed closed-runtime error from fallible operations
after shutdown, before loading artifacts or allocating device resources. Define
the closed-state contract for `stream()` explicitly, either checked access or a
clearly documented internal precondition. Apply the same lifecycle wording to
borrowed plans invalidated by session shutdown under AR-04.

**Acceptance.** Repeated shutdown is safe; upload and session creation after
shutdown reject without a null dereference or allocation. Keep this a small
lifecycle fix, not a general restart framework.

### AR-09 — P2, confirmed: finish attention's direct residual output path

**Evidence.** `src/runtime/attention.cpp:980` copies the residual to the output
and then uses the legacy in-place epilogue at line 990. Architecture V0's
attention step 6 and the runtime header describe direct output projection plus
residual addition. The separate residual/output MMV interface already exists
and is used by MLP/GDN.

**Implementation.** Bind the original residual as read-only input and the other
residual buffer as output, and remove the D2D copy. Preserve validated nonaliasing
and live ranges. This implements the selected schedule and needs no new fusion
or architectural experiment.

**Acceptance.** Q4 and BF16-control attention preserve the input residual and
produce the same numerical result. Verify the execution trace has six attention
kernels and no residual D2D copy. Reuse existing attention comparisons and add
only the discriminating input-preservation/schedule assertion that is missing.

### AR-10 — P2, reproduced: repair benchmark consumers and stale launch accounting

**Evidence.** The MLP benchmark fails compilation at
`benchmarks/mlp_bench.cpp:46` because `TensorView::writable` was removed. The GDN
benchmark builds but fails binding: its rank-one per-layer S descriptor at
`benchmarks/gdn_mixer_bench.cpp:190` violates the current full-state contract.
It also declares nine kernels and one device copy per iteration at lines
215–216, although the repaired GDN implementation now has eight kernels and no
residual copy. These targets are `EXCLUDE_FROM_ALL`, explaining why the normal
build and fast suite pass.

**Implementation.** Update benchmark fixtures to the current const/mutable view,
state, and plan contracts, preferably using production session binding where
appropriate. If a deliberately per-layer fixture is retained, provide a valid
explicit contract rather than weakening production validation. Correct current
launch/copy accounting. Check other excluded consumers when a shared public API
changes. Keep benchmarks outside correctness timing gates.

**Acceptance.** Build all affected excluded targets and run bounded setup/smoke
checks successfully. For GDN, establish actual launch/copy counts by trace or
existing capture inspection rather than changing another unexplained constant.
Use `scripts/run_benchmark_evidence.sh` for any subsequently published timings;
there is no need to run a long benchmark sweep to close the API repair.

## Architecture and integration decisions

### AR-11 — Resolve execution completion before composing 64 layers

**Confirmed behavior / design decision.** GDN synchronizes at mixer completion
(`src/runtime/gdn.cpp:1364`), attention synchronizes midway after preparation
(`src/runtime/attention.cpp:937`), and MLP returns after enqueue. Straight
composition introduces approximately 64 stream synchronizations per token
(48 GDN plus 16 attention preparations), while still lacking a full-token commit
contract. This is an API/schedule observation, not a measured latency claim.

**Proposed solution.** Establish explicit enqueue versus completion semantics
alongside AR-03. Public completed-operation wrappers can preserve convenient
component tests. The integrated scheduler should own pending layer progress and
one token completion/commit boundary on the session stream; internal consumers
use known pending lengths/positions in ordered work. Start with eager execution
and measure it. Preserve the selected kernel regions, precision boundaries, and
stable allocations. Do not optimize away synchronization before the failure and
snapshot contract is implemented and tested.

The capture special case at `src/runtime/gdn.cpp:1358` returns success without
committing cursor/position. Replaying captured work then lacks a coherent host
continuation protocol. Until production capture has an explicit design, reject
it before mutation or isolate it as a diagnostic-only enqueue facility whose
caller owns state. Treat failed capture-status queries as errors, not evidence
that normal execution is safe. CUDA graph support remains deferred as allowed
by `architecture-v0.md:357`.

**Acceptance.** Document return-value completion guarantees for MLP, GDN,
attention, and the future token executor. Exercise AR-03's mid-token failures,
snapshot boundaries, duplicate/skipped positions, and unsupported capture. At
TASK-017, record actual synchronization count and eager token latency without
claiming a throughput improvement from source inspection alone.

### AR-12 — Choose one semantic operand-resolution boundary

**Design debt, not a demonstrated wrong-output bug.** `GraphBinding` in
`src/format/schema.hpp:119` has node/role/tensor/layer identifiers but no operand
slot. Some operands have the same role and shape: Q/K norm, gate/up, and GDN
parameter pairs. Runtime binders currently construct checkpoint-style logical
names. TASK-016/017 expect artifact-based bindings and complete graph validation;
uncoordinated name reconstruction across new consumers will weaken that goal.

**Recommended minimal solution.** Keep V0's existing wire format and establish
one model-plan construction boundary that resolves canonical logical names into
explicit, typed semantic slots/tensor IDs. Validate those resolutions against
graph membership, role, shape, layout, and layer exactly once. Execution should
consume the resulting immutable plans. Document canonical names as part of this
V0 compatibility contract. Do not select operands by first matching shape/role
or record order. If source-name-independent artifacts are required instead,
introduce versioned explicit operand slots through a recorded format amendment;
do not silently reinterpret the current ABI.

**Acceptance.** Swap same-shaped operands or graph associations, remove a
required norm, duplicate a binding, and permute nonsemantic directory order.
Reject invalid semantics while accepting equivalent ordering. Test two distinct
layer instances to detect accidental reuse. Pair this with AR-02's verification
checks, but keep format-valid partial test artifacts distinct from a
complete-model plan's stronger requirements.

### AR-13 — Strengthen the next checkpoint and independent source evidence

**Upcoming acceptance work.** Retain TASK-016's small scope: one GDN-style and
one attention-style language layer, common MLP/residual/scratch/session machinery,
consecutive tokens, and continuation. Require actual model-upload/session-backed
bindings, not exclusively handwritten exact-size views. Commit `130063c` fixed
arena-sized views in those binders, directly demonstrating why this distinction
matters. Cover multiple state indices, a second session, session movement under
AR-04, full relevant state byte comparisons, and untouched-state isolation.
On mismatch, report the first differing materialized boundary.

Before accepting TASK-017/018, make independent source-model agreement explicit.
`architecture-v0.md:363` requires it; TASK-017's current “BF16 control/reference”
wording permits two engine paths sharing the same semantic bug. Use a pinned
external source runner or frozen source-produced golden outputs for initial and
continued-token logits plus selected residual/state checkpoints. Record source
weights/config/tokenizer, source software revision, token IDs, precision settings,
and output hashes. Keep the diagnostic framework outside inference dependencies.

**Acceptance.** First establish BF16 engine agreement with that independent
source, with declared tolerances and arithmetic-boundary differences. Only then
attribute V0-versus-BF16 deviations to quantization. Budget the already planned
one-layer-resident BF16 diagnostic instead of assuming full BF16 model residency
on the 32 GiB GPU. Record untested coverage. Preserve exact equality for identical
deterministic snapshot/replay tests separately from numerical tolerance for
different schedules or implementations.

### AR-14 — Specify how a failed quality gate can be repaired

**Planning issue.** TASK-018 requires quality acceptance or BLOCKED and prohibits
early experiments. Prefill and TASK-023 depend on passing it, while EXP-A at
TASK-024 depends on TASK-023. Architecture V0 nevertheless directs quality
failures toward targeted wider-precision diagnosis. The stop policy is coherent;
the missing piece is a concrete resumption procedure for the next agent.

**Proposed solution.** Preserve the failing baseline and frozen evaluation suite.
First separate source/semantic implementation bugs from failures of the selected
quantization/transport hypothesis. Fix implementation bugs under the existing
contract. For a hypothesis failure, propose a narrow architecture amendment
naming the decision, failing evidence, diagnostic comparison, affected weight
families, artifact identity, and memory implications. Record the decision
authority and explicit acceptance in the ledger before changing a locked policy
or task ordering; merely writing the proposal does not authorize that change.
Re-run the unchanged
quality gate before accepting a replacement baseline and resuming the ledger.
Leave unrelated experiments and performance tuning deferred. This review proposes
the procedure; it does not authorize a particular new precision policy.

**Acceptance.** TASK-018 and the ledger explain the exact path from a hypothetical
failing Q4 family to diagnosis, amendment, retest, and resumed implementation.
No relaxed thresholds, implicit early EXP-A execution, or unsupported DONE status.

## Guidance and evidence maintenance

### AR-15 — Update code standards with concrete contracts learned from this batch

Apply this to `docs/implementation/code-standards.md` and relevant task acceptance
templates before another long sequence of implementation agents. Keep the
existing model-specific C++ design and proportional verification rules. Add
concise requirements covering the following independently useful points:

1. **Validate at the actual boundary.** Public bind/launch descriptions specify
   device/memory space, constness, dtype, logical and physical geometry, layout,
   byte extent/alignment, and permitted overlap. Grouped operands each have their
   own descriptor. Bound counts and checked arithmetic precede indexing,
   multiplication, allocation, or span construction.
2. **Separate capacity from logical extent.** Derive checked token/layer subviews
   from arenas; do not loosen semantic shape checks to accept oversized storage.
   A binder change must exercise real Session views, not only synthetic views.
3. **Make lifetime and completion explicit.** Every borrowed plan names its
   owners, movement/invalidation rules, stream/device, and completion guarantee.
   Stateful execution defines failed/recoverable/closed states and commits.
   Failure tests inspect bytes and metadata together and include later-stage
   failure after earlier mutation.
4. **Validate once, execute a frozen plan.** Keep mutable BindViews/builders
   separate from validated execution plans. Prefer private plan state or another
   concrete construction invariant so callers cannot casually change a validated
   pointer/layout and bypass checks. Recheck changing position,
   capacity, and health at execution. Do not add a generic operator framework.
5. **Prove the production call path.** A repaired helper needs at least one
   discriminating test through its real consumer. Identity hashing existing and
   passing in isolation is not proof that compilation uses it. Changes to a
   shared interface must identify affected tests, CLIs, and excluded benchmarks.
6. **Make numerical tests distinguish mistakes.** State input/store precision and
   rounding placement. Use independent, asymmetric golden cases for orientation,
   heads, operand identity, forbidden quantized values, overflow, and non-finite
   results. Avoid references that share the indexing/rounding logic being tested.
7. **Cover every resource branch.** A bounded-memory claim includes BF16,
   quantized, identity, reconstruction, and error paths that actually run in that
   operation. Isolate memory regressions from prior RSS high-water marks. Do not
   optimize test time by dropping content identity at production boundaries.
8. **Account for schedule changes.** Added copies, launches, synchronizations,
   and hot allocations are visible contract changes. Maintain truthful current
   diagnostic accounting. Build/smoke affected benchmark consumers without
   placing benchmark timings inside correctness tests.
9. **Close requirements with evidence.** Completion reports include a compact
   locked-requirement → implementation → discriminating test mapping. An unmet
   locked requirement is a blocker or explicit amendment, never a contradictory
   follow-up note beneath DONE. Give repaired findings stable IDs and link the
   production caller and regression proving closure.

**Acceptance.** The revised standards are short enough for agents to apply,
reference the concrete regression classes above, and do not mandate blanket
Debug+Release/full-authority/sanitizer reruns for every repair. TASK-016's updated
acceptance should demonstrate the rules on actual layer composition. These are
targeted prevention measures, not a request for a repository-wide style rewrite.

### AR-16 — Reconcile current authority with historical completion reports

**Confirmed drift.**

- `docs/implementation/task_ledger.md:19` says experiments are TASK-023–030;
  actual experiments are TASK-024–031. TASK-023 is the baseline and TASK-031 is
  EXP-H.
- `docs/implementation/code-standards.md:3` still points to a future nonexistent
  `implementation_ledger.md` instead of the existing `task_ledger.md`.
- TASK-008:90 describes the repaired intermediate QK/RoPE BF16 store;
  TASK-010:105 describes the repaired in-place `h_mid` behavior;
  TASK-013:113–115 describes two qkv/z launches, an output copy, and in-place MLP.
- `review.md` reuses item numbers, including 50–54, without item-level closure
  status. A reference to “review item 51” is ambiguous.
- TASK-009/013 explicitly withdraw insufficiently identified historical timings,
  while TASK-010:89 and TASK-012:87 still call mutable tags/binary paths identity.

**Implementation.** Correct ledger links/ranges. Preserve historical reports but
mark superseded contract descriptions and measurements clearly. Add a compact
repair index using stable IDs or priority plus exact title, linked repair commits,
current contract, focused regression, and remaining limitation. Include these AR
items when they are closed. Do not claim every original item is independently
re-certified by this review. Do not rerun old benchmarks merely to tidy history.

**Acceptance.** Reading any completed task leads an implementation agent to the
current contract and distinguishes historical observations from current evidence.
Each claim of remediation identifies both the code path and test that establishes
it; unresolved residuals remain visible.

## Suggested implementation order

1. Correct authoritative pointers and add the targeted guideline requirements
   (AR-15/16), then resolve the session execution/lifetime design in AR-03/04/05/11.
   Implement these coupled state changes in coherent, reviewable increments.
2. Repair policy verification (AR-02); repair the
   checked descriptor/transform boundaries and closed Runtime behavior (AR-06/08).
   Compiler and runtime repairs can be reviewed independently.
3. Complete bounded BF16 verification, attention direct residual output, and
   affected benchmark consumers (AR-07/09/10). Use focused tests and one final
   fast integration run after shared interfaces settle.
4. Settle semantic resolution (AR-12) and execute the strengthened TASK-016
   checkpoint (AR-13). Accept it before scaling to TASK-017.
5. Carry the independent source oracle and explicit quality-recovery procedure
   into TASK-017/018 (AR-13/14). Retain the current architecture hypotheses until
   those measurements or a documented blocker justify an amendment.

Each follow-up should report its AR ID, changed contract, exact focused checks,
and remaining limits. This review intentionally supplies decisions and acceptance
properties rather than prescribing a large patch or combining all repairs into
one implementation task.
