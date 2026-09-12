# OPT-096 — Conditional eight-layer decode graphs

Status: **eligible** (`verdict=proceed`). Authority llama.cpp
`cc83d7b4824f73cfdda4dfbb47ee39804f71b328`, GGUF SHA-256 `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`.

Control `ffn_only` retains per-layer FFN graphs (64 executables). Candidate
`decode_segments8` captures eight eight-layer decode segments (layers 0–7 …
56–63) with `GraphLaunchParams` updates before replay. Embedding, final logits,
and state publication stay outside segments; poll at every eight-layer boundary.

`claims_throughput: true` only when `production_kept` selects `decode_segments8`.
Otherwise finish as no-reopen (`no_reopen_overhead_below_trigger`) and retain
`ffn_only`.

## Eligibility

OPT-090 gate: `graph_reopen_eligible=True`; verdict **`proceed`** (not
`no_reopen_overhead_below_trigger`); reason `unhidden_idle_at_both_prefixes`.

| Prefix | Removable unhidden idle (ms/token) | Threshold |
|--------|-----------------------------------|-----------|
| D128   | 0.8569                            | ≥0.50     |
| D2048  | 0.8619                            | ≥0.50     |

Source: [`fixtures/opt090_decode_attribution.json`](../../../fixtures/opt090_decode_attribution.json)
(host-only eligibility, 2026-09-12T09:47:39Z). Prior OPT-055 fresh diagnostic
idle was 0.08555/0.10514 ms/token at D128/D2048
([`fixtures/opt055_execution_graphs.json`](../../../fixtures/opt055_execution_graphs.json));
native `--workload overhead` was not executed in this environment (no host nvcc).
A GPU sitting with a fresh diagnostic trace could still finish
`no_reopen_overhead_below_trigger` per dossier step 2.

## Independent verdicts

Pending same-math, performance, quality, and state-memory phases. Placeholders:

```json
{
  "ffn_only": {
    "kernel_parity_pass": false,
    "model_quality_pass": false,
    "performance_pass": false,
    "production_kept": false,
    "incomplete": true
  },
  "decode_segments8": {
    "kernel_parity_pass": false,
    "model_quality_pass": false,
    "performance_pass": false,
    "production_kept": false,
    "incomplete": true
  }
}
```

## Numeric policy

Bitwise graph/eager same-math on parity positions
(0/1/127/128/2047/2048/4095/4096), reset, workspace swap, cancellation after
segments 1/4/8, prompt65→decode, and final legal 128K position. Nonfinite=None
(pending native same-math phase).

## Complete 64-layer decode body

Component n=None means: control n/a ms, candidate n/a ms.
Paired CI: n/a .. n/a ms (pending `body128`/`body2048` acceptance).

## Quality (OPT-073)

quality-v3 engine non-regression=None (pending release `quality` phase).

## Decision

`production_kept=False`. Shipping execution graphs `ffn_only`.
`claims_throughput=false`.

## References

- Contract: [`pins/opt096_decode_graphs_contract.json`](../../../pins/opt096_decode_graphs_contract.json)
- Iteration: [`pins/opt096_iteration_contract.json`](../../../pins/opt096_iteration_contract.json)
- Fixture: [`fixtures/opt096_decode_graphs.json`](../../../fixtures/opt096_decode_graphs.json)
- OPT-090 attribution: [`fixtures/opt090_decode_attribution.json`](../../../fixtures/opt090_decode_attribution.json)
- OPT-055 prior graphs: [`fixtures/opt055_execution_graphs.json`](../../../fixtures/opt055_execution_graphs.json)
- Tool: [`tools/opt096_decode_graphs.py`](../../../tools/opt096_decode_graphs.py)
