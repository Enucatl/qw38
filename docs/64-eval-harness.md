# The evaluation harness boundary

[Index](README.md) · Implementation task: EVAL-001 (**blocked; partial**) ·
[`implementation_ledger.md`](../implementation_ledger.md)

Evidence links: [`tools/qw38_eval.py`](../tools/qw38_eval.py) ·
[`pins/eval_contract.json`](../pins/eval_contract.json) ·
[`fixtures/eval_harness.json`](../fixtures/eval_harness.json) ·
[`tests/test_eval.py`](../tests/test_eval.py)

`qw38-eval` is the machine-readable evaluation entry point planned for
logits, checkpoint, and CUDA trace evidence. It is intentionally separate from
quality admission: QLT-001 will choose prompts and thresholds later. This
chapter records the contract and the current implementation boundary.

## Raw-token request shape

The request boundary is an exact token sequence, not text. A caller supplies a
non-empty comma-separated list of decimal IDs in the pinned vocabulary range
`[0, 248320)`, an output directory, and an explicit source revision plus
`clean`/`dirty` state. Typed Python records in
[`tools/qw38_eval.py`](../tools/qw38_eval.py) construct the native argv without
shell interpolation. They reject whitespace, empty fields, negative or
out-of-vocabulary IDs, missing source identity, and duplicate trace filters
before execution.

The versioned native contract is frozen in
[`pins/eval_contract.json`](../pins/eval_contract.json). The intended command
families are:

```text
qw38-eval MODEL --mode logits|checkpoint --tokens CSV --output DIR \
  --source-revision REV --source-state clean|dirty [--continuation CSV]
qw38-eval-diagnostic MODEL --mode trace --tokens CSV --output DIR \
  --source-revision REV --source-state clean|dirty --trace-filter LAYER:TAP
```

**Measured (partial).** The current native executable accepts and validates the high-level request
shape and implements the logits publication path. It does not yet provide the
checkpoint or CUDA trace capture paths. Therefore the checked-in
[`fixtures/eval_harness.json`](../fixtures/eval_harness.json) labels all three
hardware smokes `smoke-not-run`; it is wiring evidence, not a checkpoint,
trace, or quality result.

## Evidence records

Logits evidence is a complete little-endian FP32 row of 248,320 values beside
`result.json`. The typed reader checks the exact schema keys, token metadata,
finite values, byte count, and SHA-256 before exposing the row as an immutable
tuple. The native result is expected to retain the model/tool identities,
positions, committed frontier, and deterministic greedy choice with lower-ID
tie breaking.

**Proposed.** Checkpoint evidence will retain an authenticated prefix checkpoint and the
continuation row, comparing tokens and FP32 logits after sequential save,
destroy, restore, and evaluation. **Proposed.** Trace evidence will use only the diagnostic
CUDA build and the five filters frozen by the contract, publishing the existing
`qw38.trace` v1 manifest/blob format. Neither path is implemented in this
increment, so no checkpoint or trace acceptance claim is made here.

All successful output directories must be newly created atomically. Malformed
requests, unsupported builds or filters, existing destinations, runtime
failures, and evidence mismatches must leave no final partial directory. The
normal build remains the public Engine/Session boundary; trace internals belong
only to the separately compiled diagnostic product.

## Verification boundary

Focused unit coverage currently exercises strict token parsing, source identity,
the subprocess/result types, and rejection of a wrong logits digest in
[`tests/test_eval.py`](../tests/test_eval.py). The native help and malformed
request exit contracts are covered in
[`tests/test_build.py`](../tests/test_build.py). These checks do not establish
CUDA logits, checkpoint, trace, quality, NLL, recurrence, retrieval, or task
thresholds. Those claims remain pending until the missing native paths and
RTX 5090 evidence are produced.
