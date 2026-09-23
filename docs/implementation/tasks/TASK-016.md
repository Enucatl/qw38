# TASK-016 — One-layer integration checkpoint

## Status
DONE
## Milestone
M6 — Integrated language decode
## Purpose
Expose binding, residual, scratch, and persistent-state errors before scaling to 64 layers.
## Depends on
- TASK-013
- TASK-015
## Normative references
- `docs/architecture/architecture-v0.md` — Semantic graph; decode schedule; materialization
- `docs/architecture/model-semantics.md`
- `docs/implementation/technology-baseline.md`
- `docs/implementation/code-standards.md`
## Architecture decisions consumed
| Decision ID | Contract | Type |
|---|---|---|
| G-01–G-02 | Layer semantics and decode physical schedule | LOCKED |
| P-01–P-02, S-01–S-02, M-01 | Precision/state/scratch contracts | LOCKED |
## Starting point
Complete GDN+MLP and attention+MLP paths work separately.
The focused review-repair checks and trace evidence are recorded in
[`task-016-prerequisite-evidence.md`](../task-016-prerequisite-evidence.md).
## Scope
Build immutable plans and deterministic fixtures for one GDN-style language layer and one full-attention-style language layer through the same runtime/session/tensor binding/scratch/residual machinery. Execute consecutive tokens and compare each materialized boundary, state update, and final residual to reference/BF16 control.
## Out of scope
64-layer model, embedding/head, scheduler optimization, prefill, fusion, general graph engine.
## Required interfaces
Concrete `LanguageLayerPlan`-equivalent tagged plan (no virtual operator framework); one decode-layer entry dispatches statically/tagged to mixer then shared MLP.
## Required semantics
Layer ordering is mixer residual transition then post-mixer MLP transition. Input residual remains valid for its owning add. Only the correct state family changes.
## Data representation
Bindings come from artifact tensor IDs/layouts; two FP32 residual buffers ping-pong; arena aliases follow proven lifetimes; session state remains layer-indexed.
Resolve canonical V0 logical names once at model-plan construction into typed
tensor IDs and semantic slots. Validate each binding's graph membership, role,
shape, layout, and layer there. Never choose operands by matching shape or
directory order; test swapped same-shaped operands and distinct layer instances.
## Implementation constraints
No per-call plan construction/allocation; no cross-layer scratch leakage; deterministic one-stream ordering.
## Tuning defaults
Inherited kernel defaults only.
## Expected files/modules
Runtime language-layer plan/executor and deterministic integration fixture/tests.
## Tests required
### Unit tests
Missing/wrong family binding, scratch overlap rejection, layer/state index isolation.
### Reference/numerical tests
Stage/final comparisons for each layer kind over 1 and multiple tokens.
### Integration tests
GDN layer then attention layer in one session; reset/restore continuation; second session isolation.
## Benchmark required
No.
## Acceptance criteria
- [ ] Both layer types use common runtime/session interfaces.
- [ ] Residual ping-pong, scratch reuse, and tensor bindings are correct.
- [ ] Only expected persistent state changes.
- [ ] Multi-token and restore continuation pass.
- [ ] Bind through actual uploaded-model and Session views; cover multiple state indices, a second session, supported Session movement, full relevant state bytes, and untouched-state isolation.
- [ ] Report the first differing materialized boundary on mismatch; preserve exact equality checks for deterministic snapshot/replay.
- [ ] Demonstrate same-stream ownership and failed-operation/session-health behavior required by AR-03–AR-05.
## Architecture blocker rule
On locked conflict stop with full blocker report; do not introduce a generic graph/operator architecture.
## Completion report
### Result
DONE
### Changes made
Added `LanguageLayerPlan`, a tagged GDN/attention plan that resolves mixer and
shared MLP plans once from the uploaded `Model` and owning `Session`. Its
executor performs the mixer residual transition followed by the MLP transition
on the session stream without constructing plans or allocating storage. If the
MLP fails after the mixer has run, the executor now poisons the Session so a
snapshot or retry cannot treat partially advanced state as a completed layer.

The `hybrid.qw38` fixture is generated at test runtime with patterned nonzero
Q4 codes and FP16 scales, asymmetric BF16 operands, and distinct residual inputs
for two tokens. It executes GDN layers 0 and 1 and attention layers 3 and 7 in
one Session. Checks compare the complete GDN state, convolution-history, and KV
byte spans, verify only state indices 0 and 1 changed, and confirm indices after
1 remain byte-identical. The second Session executes all four plans with a
different residual; both Sessions' snapshots prove cross-session isolation.
The integration case also covers same-stream binding, reset replay, restore
continuation, Session movement, mixer and MLP materialized boundaries, and
post-mixer MLP failure followed by exact recovery against a clean control.
Replay mismatches report the first materialized boundary and element.

The follow-up fixture reads the uploaded artifact's Q4 codes, FP16 scales, BF16
tiles, and vector operands back into the independent CPU/BF16 references. It
compares the post-mixer and post-MLP residuals for all four layer instances on
both tokens, with the documented component tolerances and the first differing
boundary and element on failure. A complete Session snapshot before and after
each layer verifies that only its own GDN/conv or KV state index changes. A
second artifact binds a GDN output projection to the attention graph family;
the GDN layer binder rejects it without changing Session state.

Acceptance evidence mapping:

| Contract | Implementation and discriminating evidence |
| --- | --- |
| Common tagged layer interface, binding and residual order | `LanguageLayerPlan::bind` / `execute_decode_language_layer`; `language_layer_integration` runs the uploaded-model and Session path for both mixer kinds. |
| Nonzero operand identity, arithmetic and mixer-to-MLP flow | `language_layer_integration` compares 16 composed materialized boundaries against independent CPU/BF16 mixer and MLP references decoded from the uploaded artifact, across GDN layers 0/1 and attention layers 3/7 on two tokens. |
| Persistent-state isolation and full bytes | `language_layer_integration` compares eight complete before/after Session snapshots, requiring only the owning GDN/conv or KV index to change after each layer. It also checks all untouched indices and second-Session isolation. |
| Reset/restore continuation and deterministic equality | `language_layer_integration` replays two tokens after reset and compares each mixer/MLP boundary exactly; restored token-2 state matches uninterrupted execution byte for byte. |
| Failure health after a committed mixer | `language_layer_integration` injects an invalid MLP output view after GDN success, confirms execute/save rejection, then compares recovered state and metadata exactly with a clean control. |
| Negative bindings and scratch overlap | `language_layer_integration` rejects a same-shaped GDN projection bound to the attention graph family and verifies state is unchanged. `runtime_plan`, `mlp_unit`, `gdn_unit`, and `attention_unit` cover other missing/wrong bindings and scratch overlaps. |
| Scratch lifetime reuse | `runtime_plan` checks planned non-overlapping lifetimes; `runtime_state_index` checks the GDN and attention workspace views share the declared arena; `mlp_unit` checks stable MLP scratch views across calls. |
| Same-stream ownership | `language_layer_integration` rejects alternate same-device streams for GDN and attention plans; component integration suites also exercise bound Session streams. |

The fixture exposes its allowed metadata-only manifest digest in the integration
log: `d81a2dd6d4980adb9585be31ba15c65888369dd574ba518978ebe03910c34a3d`.
No payload or scale digest was computed, per the artifact digest policy.

### Tests run
Reference environment: `qw38-dev:cuda13.4.1`, image
`sha256:be0903b40ab2e14ec1b5ba285655219cc9e521cd1455b070a6dfdb68ed4e4bfb`,
CUDA 13.4.1, GCC 14.2.0, NVIDIA GeForce RTX 5090 (`sm_120`), driver
590.48.01, 32607 MiB.

Build, direct composition, and focused correctness run:

```sh
docker run --gpus all --rm -u "$(id -u):$(id -g)" -v "$PWD":/workspace -w /workspace qw38-dev:cuda13.4.1 bash -lc 'set -euo pipefail
cmake --build build/release --target qw38_language_layer_integration_test -j2
./build/release/tests/qw38_language_layer_integration_test | tee build/release/task016-language-layer.log
ctest --test-dir build/release -R "(language_layer_integration|runtime_plan|runtime_state_index|runtime_session_integration|mlp_unit|mlp_integration|gdn_unit|gdn_integration|attention_unit|attention_integration)$" --output-on-failure --output-log build/release/task016-ctest.log'
```

Result: build succeeded, direct integration executable exited 0, and 10/10
focused tests passed (`language_layer_integration`, `runtime_plan`,
`runtime_state_index`, `runtime_session_integration`, `mlp_unit`,
`mlp_integration`, `gdn_unit`, `gdn_integration`, `attention_unit`,
`attention_integration`). `git diff --check` passed. No benchmark was required
or run.

SHA-256 identities:

- Integration executable: `fb6a2b5142979813bed3e7d105d4cb9779a6d2f1d5a0ac08632348d0366b263b`
- Direct integration log: `e775f0bcb4aa4193a8402fc555fa5c171d7b2555e4d1a32336a304774be0e9d5`
- Focused ctest log: `d856cf2e66fc5513de03d39f95b43944af502e028edbed1c34ea30a6c75d819d`

### Benchmark results
Not required.
### Architecture blocker
None.
### Independent review
Fresh independent Sol review: `PASS`. The reviewer examined the complete diff,
production code and callers, architecture and code standards, integration
fixture, and recorded logs. No acceptance gaps, code findings, or additional
evidence were requested.
### Follow-up observations
None.
