# OPT-113 — Coupled-stack recovery sitting

## Claim labels and proof limits

combined freeze of authenticated OPT-106 post106_control and post113_selected production survivors; OPT-107 hybrid_crossover kept; OPT-110 llama_q4k_mmvq kept; OPT-111 opt111_base kept; OPT-108 and OPT-109 rejected with no production selector leak; OPT-112 deferred_below_trigger omitted from selected and does not block OPT-113; independent kernel_parity_pass per family; quality is not one boolean; quartz_vs_baseline_quality_delta and quartz_vs_llama_quality_delta stay distinct; opt074_coverage_unadmitted is not a blocker; absolute task-accuracy fail inherited from OPT-084 does not by itself block; new regression versus OPT-084 does block; OPT-106 remains historical and is not reinterpreted; OPT-056 and OPT-016 stay blocked unless their owning conditions pass; parity gap is Tq-Tl; +5% throughput gap is Tq-Tl/1.05; do not label the parity gap as the +5% bar; decode p95 no worse than llama for the OPT-056 outcome; failed required quality stops release before long timing; diagnostic performance is not a release or keep; rejected candidates must not leak into production; internal improvement is selected versus authenticated OPT-106 control not llama; llama numbers are measured in this sitting and not reused from OPT-106; control Quartz is measured in this sitting and not reused from OPT-106; strict_ppl_ratio_max=1.01 applies because concession is inactive; release_eligible is never a synonym for opt056_pass; OPT-099 matched family attribution is diagnostic and non-additive; selected does not equal control so Quartz is measured twice.

`gate.passed` is False. OPT-106 remains historical and is
not reinterpreted. Quality is not one boolean. post113_selected does not equal
post106_control. Llama and control Quartz numbers were measured in this sitting.
OPT-112 is deferred and omitted.

## Three independent outcomes

| Outcome | Verdict |
|---|---|
| Internal improvement vs OPT-106 control | passed |
| Llama parity (Quartz >= llama, Tq-Tl) | unpassed |
| OPT-056 +5% (Tq-Tl/1.05, p95, quality, OPT-016) | unpassed |

## Measured sitting

| Workload | Selected tok/s | post106_control tok/s | llama tok/s | Δ vs control | Δ vs llama | parity ms | +5% ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| P 4096 | 3064.76758 | 3022.22437 | 3227.707756 | 42.54321000000027 | -162.9401759999996 | 67.46777350846924 | 127.89691821187193 |
| D128 | 57.2972374 | 53.5753784 | 68.6142494 | 3.721859000000002 | -11.317012000000005 | 736.925766908384 | 914.5925970067424 |
| D2048 | 55.6468849 | 49.4627914 | 67.2719456 | 6.184093500000003 | -11.625060699999992 | 794.9875888626061 | 976.1994756565596 |

Decode p95 ms: D128 Quartz 17.5574131 vs llama 14.554 vs control 18.8546886; D2048 Quartz 18.0740585 vs llama 14.638 vs control 20.3111382.

OPT-016 2K: Quartz 3152.96753 vs llama 3190.466786; gate_passed=False.

State/memory: memory_fit=True; checkpoint=True; cancellation frontier 0.

Matched family attribution is diagnostic and non-additive (OPT-099). Candidate NLL measured=True.

## Sitting identity

- device: NVIDIA GeForce RTX 5090
- image: `qw38-cuda:13.0.2` (Quartz host-native; llama-bench/decode-oracle in that image because `qw38-llama-authority:cuda-13.0.2` is not present)
- llama_revision: `cc83d7b4824f73cfdda4dfbb47ee39804f71b328`
- gguf_sha256: `31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34`
- post106_control: Q4 `integer_q8_late`, attention `kv_once`, decode `warp_query` (crossover 0)
- post113_selected: Q4 `llama_q4k_mmvq`, attention `opt111_base`, decode `hybrid_crossover` (crossover 1024)
- OPT-112 omitted (`deferred_below_trigger`); OPT-108/109 not in production pins
- geometric mean selected/control speed ratio 1.068564 (CI lower 1.002411); per-workload CIs and decode p95 vs control all passed
- `internal_improvement_with_quality` **passed** (quality_pass=True, state_memory_pass=True)

## Quality (measured candidate NLL, no stub)

held-out 1024 mean NLL 1.7875990840085783; ppl_ratio 0.9997208519718781; recurrence incremental NLL 0.0; inherited absolute task_arithmetic fail; `model_quality_pass` True.

## 128K memory_fit reconciliation

explicit_bytes=29571277792; measured_delta=29578231808; free_bytes=3521118208; reserve_ok=True; arithmetic=true; passed=true. Allocator delta 6954016 bytes versus explicit owner sum. Workspace ledger reconciled to live `1831836288` bytes (+25728 vs OPT-012 snapshot). Cap not raised.

## Independent fields

| Field | Status |
|---|---|
| kernel_parity_pass | True |
| model_quality_pass | True |
| performance_pass | False |
| production_kept | True |
| release_eligible | True |
| internal_improvement_with_quality | True |
| llama_parity | False |
| opt056_plus5 | False |
