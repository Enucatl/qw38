# OPT-123 — Combined pipeline stack sitting

Status: **keep**. Retained stack `combined_opt118_opt119`. Hardware executed on RTX 5090. Sitting was not exclusive (unrelated audioset-ast 868 MiB and diarization 3216 MiB); this dossier does not authorize stopping them.

Historical control is authenticated post113_selected `ffn_only`. Keepers are OPT-118 lazy logits+overlap and OPT-119 mixer+FFN norm Q8 fusion. Rejected paths stay out of the combination: OPT-117 `ffn_only`, OPT-120 dense KV, OPT-121 none requant, OPT-122 FP32 GDN. No new optimization is invented in this gate. OPT-113 tok/s numbers remain history.

## Freeze and rejected-path leak check

post113 authenticated=`True`; rejected_no_leak=`True`.
execution_graphs=`ffn_only`; packed_kv=`DenseBf16`; weight_requant=`none`; gdn_state=`Fp32`.
Removal rules frozen before results: `True`.

## Interaction matrix

Control = post113 (lazy/fusion off). Combined = OPT-118+OPT-119. opt118_only and opt119_only are survivors alone / leave-one-out of the other keeper. OPT-118/119 authenticated sittings are reused where identities match; opt119_only versus control is measured fresh at screen pairs.

## Inventory

Native inventory ok=`True` rejected_no_leak=`True`.
Launch savings, D2H bytes, and fusion launches are per stack in the fixture. Isolated component speedups are not claimed to survive unchanged in combination.

## Same-math, quality, state

greedy-parity exact=`True` tokens `539`/`539`.
OPT-058 `--quality` invoked; candidate_nll_measured=`True`; held-out PPL ratio `1.0`; strict=`True` successor=`True`; budgets anchored to post113, not ratcheted.
state/memory=`True` 128k_fit=`True` session_bytes=`696743940`.

## Full-engine A/B versus post113 (3 warmups + 10 AB/BA)

| Workload | post113 tok/s | combined tok/s | geo ratio | CI lower | p95 ratio | gate |
|---|---:|---:|---:|---:|---:|---|
| P4096 | 2917.105567 | 2935.3935069999998 | 1.0062696762665677 | 1.0057170149797419 | 0.0 | True |
| D128 | 53.77292633 | 54.01590042 | 1.0045185172753681 | 1.0043462897650308 | 0.9937156622930668 | True |
| D2048 | 41.4484684 | 41.633106240000004 | 1.0044546103204588 | 1.004164717340398 | 0.992929703821179 | True |

Aggregate geo `1.0050805829294533` CI lower `1.0047727439828789` ok=`True`.
P4096 delta `18.287939999999708` (1.0062692074660868×). D128 delta `0.24297409000000414` (1.0045185208725464×). D2048 delta `0.1846378400000006` (1.0044546360125577×).

## Fresh pinned llama (not OPT-113 history)

llama P4096 `3105.42474` D128 `68.6696311` D2048 `67.1854929`. reuse_historical_llama=`False`. Quartz-versus-llama is informational; this keep is versus post113.

## Separately labeled long-context and OPT-016 2K

D8192 populated-cache tok/s `8.06574726` TTFT `3010.93018` ms. D32768 tok/s `1.35194111` TTFT `21544.3184` ms. D131040+32 at capacity 131072 OOM=`True` because the sitting was not exclusive (audioset-ast 868 MiB + diarization 3216 MiB). These probes are not D128/D2048 wins.
OPT-016 2K llama `3288.205178` quartz OOM=`True` gate_passed=`False`. Original OPT-056 +5% versus llama is not claimed from this sitting.

Keep=True. Reasons: none.

The combined OPT-118+OPT-119 stack is admitted against post113.
