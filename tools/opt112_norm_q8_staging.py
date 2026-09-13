"""OPT-112 screen-first decode norm → typed Q8 staging admission."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.opt075_q4_production_admission import (  # noqa: E402
    AdmissionError,
    dump_json,
    load_json,
    utc_now,
)
from tools.run_optimization_task import (  # noqa: E402
    loop_product,
    validate_performance_admission,
    workload_for_mode,
)

LLAMA_REV = "cc83d7b4824f73cfdda4dfbb47ee39804f71b328"
GGUF_SHA = "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"
TOKEN_GENERATOR = "(42 + index * 997) % 248320"
CONTRACT = ROOT / "pins/opt112_norm_q8_staging_contract.json"
ITERATION = ROOT / "pins/opt112_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt112_norm_q8_staging.json"
REPORT = ROOT / "evidence/optimization/opt112-norm-q8-staging/REPORT.md"
EVIDENCE = REPORT.parent
OPT099_FIXTURE = ROOT / "fixtures/opt099_matched_attribution.json"
OPT090_FIXTURE = ROOT / "fixtures/opt090_decode_attribution.json"
PHASES = (
    "screen",
    "same-math",
    "decode",
    "quality",
    "body128",
    "body2048",
    "d128",
    "d2048",
    "state-memory",
    "prefill-guard",
)
CONTROL_ID = "current_bf16_q8_staging"
CANDIDATE_IDS = (
    "norm_to_typed_q8_rounded",
    "residual_norm_to_typed_q8_rounded",
)
CONFIGS: tuple[dict[str, Any], ...] = (
    {"id": CONTROL_ID, "role": "control", "implemented": True},
    {"id": CANDIDATE_IDS[0], "role": "candidate", "implemented": False},
    {"id": CANDIDATE_IDS[1], "role": "candidate", "implemented": False},
)
TRIGGER_MS = 0.50
MIN_SAVING_MS = 0.25
DECODE_TOKENS = 32
DECODE_PREFIXES = (128, 2048)
PRODUCER_FAMILIES = ("ffn_norm", "input_norm")
REMOVABLE_FAMILIES = ("activation_quant", "residual_mixer")
EXCLUDED_FAMILIES = ("logits_norm", "gdn_output_norm")
VERDICT_DEFER = "deferred_below_trigger"
CONSUMERS: tuple[dict[str, Any], ...] = (
    {
        "id": "mixer_gdn_qkv_gate_alpha_beta",
        "layer_kind": "gdn",
        "layer_count": 48,
        "producer": "input_norm",
        "encoding": "Q8_1Block",
        "reuse_group": "qkv_gate_alpha_beta",
        "bf16_externally_observed": False,
        "retain_existing_path": False,
        "notes": (
            "grouped_r1_w4 Q8_1 from workspace->normalized_; GDN conv/recurrence "
            "consume projection outputs, not the normalized BF16 vector."
        ),
    },
    {
        "id": "mixer_attn_query_gate_key_value",
        "layer_kind": "attention",
        "layer_count": 16,
        "producer": "input_norm",
        "encoding": "Q8_1Block",
        "reuse_group": "query_gate_key_value",
        "bf16_externally_observed": False,
        "retain_existing_path": False,
        "notes": (
            "grouped_r1_w4 Q8_1; query split/RoPE consume projected Q/K/V, not "
            "the RMSNorm BF16 vector."
        ),
    },
    {
        "id": "ffn_gate_up",
        "layer_kind": "ffn",
        "layer_count": 64,
        "producer": "ffn_norm",
        "encoding": "Q8_1Block",
        "alternate_encoding": "Q8Block",
        "reuse_group": "gate_up",
        "bf16_externally_observed": False,
        "retain_existing_path": False,
        "notes": (
            "Sitting pin llama_q4k_mmvq restages BF16 into block_q8_1 inside "
            "the adapter. paired_integer late-Q4 would require FP32-scale "
            "Q8Block instead; encodings are not interchangeable."
        ),
    },
    {
        "id": "logits_output_norm",
        "layer_kind": "logits",
        "layer_count": 1,
        "producer": "logits_norm",
        "encoding": None,
        "reuse_group": None,
        "bf16_externally_observed": True,
        "retain_existing_path": True,
        "notes": "Unquantized logits projection; keep the existing BF16 path.",
    },
)


def relpath(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def family_plan(phase: str, mode: str) -> dict[str, Any]:
    iteration = load_json(ITERATION)
    if phase not in iteration["workloads"]:
        raise AdmissionError(f"unknown phase {phase}")
    workload = workload_for_mode(iteration["workloads"][phase], mode)
    return {
        "phase": phase,
        "mode": mode,
        "family": str(workload.get("family", "norm-q8-staging")),
        "control": CONTROL_ID,
        "warmups": int(workload.get("warmups", 0)),
        "samples": int(workload.get("samples", 1)),
        "engine_pairs": int(workload.get("engine_pairs", 0) or 0),
        "cases": int(workload.get("cases", 1)),
        "candidates": int(workload.get("candidates", 1)),
        "tier": str(workload.get("tier", "correctness")),
        "product": loop_product(workload),
        "control_candidate_pairs": int(workload.get("control_candidate_pairs", 1)),
        "prefix": int(workload.get("prefix", 0) or 0),
        "tokens": int(workload.get("tokens", 1) or 1),
        "gpu_work": bool(workload.get("gpu_work", False)),
    }


def planned_observation(
    plan: Mapping[str, Any], *, keep: bool = False
) -> dict[str, Any]:
    samples = int(plan["samples"])
    return {
        "schema_version": 1,
        "task": "OPT-112",
        "warmups": plan["warmups"],
        "samples": samples,
        "observed_warmups": plan["warmups"],
        "observed_samples": samples,
        "observed_candidates": plan["candidates"],
        "observed_shapes": plan["cases"],
        "observed_tier": plan["tier"],
        "pairs": int(plan.get("control_candidate_pairs", 1)),
        "sample_ids": list(range(samples)),
        "acceptance_executed": str(plan["tier"]) == "acceptance",
        "keep": bool(keep),
        "capture_key": None,
    }


def empty_verdict_row(not_applicable: str | None = None) -> dict[str, Any]:
    row = {
        "kernel_parity_pass": False,
        "model_quality_pass": False,
        "performance_pass": False,
        "production_kept": False,
        "incomplete": True,
    }
    if not_applicable:
        row["not_applicable_reason"] = not_applicable
    return row


def decide_deferred_verdicts(verdict: str = VERDICT_DEFER) -> dict[str, Any]:
    rows = {
        CONTROL_ID: {
            **empty_verdict_row(verdict),
            "production_kept": True,
            "incomplete": False,
        }
    }
    for candidate in CANDIDATE_IDS:
        rows[candidate] = empty_verdict_row(verdict)
    return {
        "independent_verdicts": rows,
        "selected_path": CONTROL_ID,
        "shipping_unchanged": True,
        "fusion_implemented": False,
        "production_kept": True,
        "winners": [],
        "status": verdict,
        "claims_throughput": False,
        "claims_performance_improvement": False,
    }


def is_quartz_sample(sample: Mapping[str, Any]) -> bool:
    ident = str(sample.get("ident") or sample.get("engine") or "")
    if ident in {"pinned_llama", "llama"} or ident.startswith("llama"):
        return False
    return True


def mean_family_ms(phase: Mapping[str, Any] | None) -> dict[str, float]:
    if not phase:
        return {}
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for sample in phase.get("samples") or []:
        if not isinstance(sample, Mapping) or not is_quartz_sample(sample):
            continue
        families = sample.get("families")
        if not isinstance(families, Mapping):
            continue
        for name, value in families.items():
            try:
                amount = float(value or 0.0)
            except (TypeError, ValueError):
                continue
            totals[str(name)] = totals.get(str(name), 0.0) + amount
            counts[str(name)] = counts.get(str(name), 0) + 1
    if totals:
        return {name: totals[name] / float(counts[name]) for name in totals}
    ranked = (phase.get("ranking") or {}).get("ranked") or []
    means: dict[str, float] = {}
    for row in ranked:
        if not isinstance(row, Mapping) or not row.get("family"):
            continue
        means[str(row["family"])] = float(row.get("quartz_ms") or 0.0)
    return means


def per_token(window_ms: float, tokens: int) -> float:
    if tokens <= 0:
        return 0.0
    return float(window_ms) / float(tokens)


def prefix_screen(
    *,
    prefix: int,
    families: Mapping[str, float],
    tokens: int,
    source: str,
) -> dict[str, Any]:
    producer = {
        name: float(families.get(name, 0.0) or 0.0) for name in PRODUCER_FAMILIES
    }
    removable = {
        name: float(families.get(name, 0.0) or 0.0) for name in REMOVABLE_FAMILIES
    }
    excluded = {
        name: float(families.get(name, 0.0) or 0.0) for name in EXCLUDED_FAMILIES
    }
    staging_ms = removable["activation_quant"]
    residual_ms = removable["residual_mixer"]
    ffn_norm_ms = producer["ffn_norm"]
    input_norm_ms = producer["input_norm"]
    removable_window = staging_ms + residual_ms
    removable_ms_per_token = per_token(removable_window, tokens)
    clue_ms_per_token = per_token(ffn_norm_ms, tokens)
    above = removable_ms_per_token >= TRIGGER_MS
    return {
        "prefix": prefix,
        "tokens": tokens,
        "source": source,
        "families_ms": {
            **producer,
            **removable,
            **excluded,
        },
        "ffn_norm_ms_per_token": per_token(ffn_norm_ms, tokens),
        "input_norm_ms_per_token": per_token(input_norm_ms, tokens),
        "activation_quant_ms_per_token": per_token(staging_ms, tokens),
        "residual_mixer_ms_per_token": per_token(residual_ms, tokens),
        "upper_bound_clue_ms_per_token": clue_ms_per_token,
        "potentially_removable_ms_per_token": removable_ms_per_token,
        "threshold_ms_per_token": TRIGGER_MS,
        "above_trigger": above,
        "eager_labeled_diagnostic": True,
    }


def evaluate_screen(
    opt099: Mapping[str, Any],
    opt090: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not opt099:
        raise AdmissionError("missing OPT-099 attribution")
    by_prefix: dict[str, dict[str, Any]] = {}
    for prefix in DECODE_PREFIXES:
        key = f"d{prefix}"
        phase = opt099.get(key) or {}
        tokens = int(phase.get("output_tokens") or DECODE_TOKENS)
        families = mean_family_ms(phase)
        if not families and opt090:
            families = mean_family_ms(opt090.get(key) or {})
        by_prefix[key] = prefix_screen(
            prefix=prefix,
            families=families,
            tokens=tokens,
            source=relpath(OPT099_FIXTURE),
        )
    d128 = by_prefix["d128"]
    d2048 = by_prefix["d2048"]
    removable = min(
        float(d128["potentially_removable_ms_per_token"]),
        float(d2048["potentially_removable_ms_per_token"]),
    )
    clue = max(
        float(d128["upper_bound_clue_ms_per_token"]),
        float(d2048["upper_bound_clue_ms_per_token"]),
    )
    eligible = bool(d128["above_trigger"] and d2048["above_trigger"])
    if eligible:
        verdict = "proceed"
        reason = "removable_norm_staging_at_or_above_trigger"
    elif clue >= TRIGGER_MS:
        verdict = VERDICT_DEFER
        reason = "upper_bound_clue_is_not_removable_budget"
    else:
        verdict = VERDICT_DEFER
        reason = "removable_norm_staging_below_trigger"
    return {
        "eligible": eligible,
        "verdict": verdict,
        "reason": reason,
        "threshold_ms_per_token": TRIGGER_MS,
        "min_potentially_removable_ms_per_token": removable,
        "max_upper_bound_clue_ms_per_token": clue,
        "opt099_path": relpath(OPT099_FIXTURE),
        "opt090_path": relpath(OPT090_FIXTURE),
        "opt099_ffn_norm_clue_window_ms": float(
            (d2048.get("families_ms") or {}).get("ffn_norm") or 0.0
        ),
        "d128": d128,
        "d2048": d2048,
        "consumers": [dict(row) for row in CONSUMERS],
        "fusion_candidates": list(CANDIDATE_IDS),
        "fusion_implemented": False,
    }


def require_eligible(results: Mapping[str, Any]) -> None:
    gate = results.get("screen") or results.get("eligibility") or {}
    if gate.get("eligible"):
        return
    raise AdmissionError(
        "OPT-112 blocked: "
        + str(gate.get("verdict") or VERDICT_DEFER)
        + f" ({gate.get('reason')})"
    )


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def write_report(payload: Mapping[str, Any]) -> None:
    screen = payload.get("screen") or {}
    d128 = screen.get("d128") or {}
    d2048 = screen.get("d2048") or {}
    clue_window = _fmt(screen.get("opt099_ffn_norm_clue_window_ms"), 3)
    clue_token = _fmt(d2048.get("upper_bound_clue_ms_per_token"))
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    text = f"""# OPT-112 — Fuse decode normalization into typed Q8 staging

Status: **{payload.get("status", "pending")}**. Authority llama.cpp
`{LLAMA_REV}`, GGUF SHA-256 `{GGUF_SHA}`.

`claims_throughput: true` only if a fusion candidate is implemented and kept.
This sitting finished as `{payload.get("status", VERDICT_DEFER)}` and retained
`{CONTROL_ID}`. Fusion candidates were not implemented.

## Screen

Source `{screen.get("opt099_path", relpath(OPT099_FIXTURE))}`. OPT-099 unmatched
`ffn_norm` D2048 window {clue_window} ms is an upper-bound clue
({clue_token} ms/token), not a removable budget.

| Prefix | ffn_norm ms/token (clue) | activation_quant ms/token | residual_mixer ms/token | potentially removable ms/token |
| --- | ---: | ---: | ---: | ---: |
| D128 | {_fmt(d128.get("ffn_norm_ms_per_token"))} | {_fmt(d128.get("activation_quant_ms_per_token"))} | {_fmt(d128.get("residual_mixer_ms_per_token"))} | {_fmt(d128.get("potentially_removable_ms_per_token"))} |
| D2048 | {_fmt(d2048.get("ffn_norm_ms_per_token"))} | {_fmt(d2048.get("activation_quant_ms_per_token"))} | {_fmt(d2048.get("residual_mixer_ms_per_token"))} | {_fmt(d2048.get("potentially_removable_ms_per_token"))} |

Trigger {TRIGGER_MS:.2f} ms/token at both prefixes.
min removable={_fmt(screen.get("min_potentially_removable_ms_per_token"))}.
verdict={screen.get("verdict")}; reason={screen.get("reason")}.

Decode attribution has no first-class `activation_quant` leaf (Q8 restage sits
inside FFN/mixer projection intervals). `residual_mixer` is the foldable add
launch for `residual_norm_to_typed_q8_rounded`; the FP32 residual store still
has to exist. RMSNorm arithmetic in `ffn_norm` / `input_norm` is retained.

## Consumers

GDN (48) and attention (16) mixer inputs reuse Q8_1 among same-encoding
groups. FFN (64) currently restages through sitting `llama_q4k_mmvq`
`block_q8_1`; a `paired_integer` pin would need FP32-scale `Q8Block` instead.
`logits_norm` stays on BF16. Encodings are not cross-reused.

## Decision

production_kept={payload.get("production_kept")}.
Shipping path `{payload.get("selected_path", CONTROL_ID)}`.
fusion_implemented={payload.get("fusion_implemented", False)}.
"""
    REPORT.write_text(text, encoding="utf-8")


def run_screen_phase(*, mode: str, run_dir: Path) -> dict[str, Any]:
    if not OPT099_FIXTURE.is_file():
        raise AdmissionError(f"missing {OPT099_FIXTURE}")
    opt099 = load_json(OPT099_FIXTURE)
    opt090 = load_json(OPT090_FIXTURE) if OPT090_FIXTURE.is_file() else {}
    gate = evaluate_screen(opt099, opt090)
    plan = family_plan("screen", mode)
    observed = planned_observation(plan, keep=False)
    admission = validate_performance_admission(
        load_json(ITERATION),
        mode=mode,
        workload_name="screen",
        workload=workload_for_mode(load_json(ITERATION)["workloads"]["screen"], mode),
        stdout=json.dumps(observed),
        success=True,
    )
    decided = {} if gate["eligible"] else decide_deferred_verdicts(str(gate["verdict"]))
    return {
        "schema_version": 1,
        "task": "OPT-112",
        "phase": "screen",
        "mode": mode,
        **gate,
        "native_counts": observed,
        "admission": admission,
        "claims_throughput": False,
        "measurement_utc": utc_now(),
        "status": "eligible" if gate["eligible"] else str(gate["verdict"]),
        **decided,
    }


def run_stub_phase(phase: str, *, mode: str) -> dict[str, Any]:
    require_eligible(load_json(FIXTURE) if FIXTURE.is_file() else {})
    raise AdmissionError(
        f"OPT-112 {phase} is unimplemented because screen did not admit fusion"
    )


def run(mode: str, phase: str, run_dir: Path) -> dict[str, Any]:
    if phase not in PHASES:
        raise AdmissionError(f"unknown phase {phase}")
    run_dir.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    if FIXTURE.is_file():
        try:
            results.update(load_json(FIXTURE))
        except json.JSONDecodeError:
            results = {}
    results.update(
        {
            "schema_version": 1,
            "task": "OPT-112",
            "mode": mode,
            "llama_revision": LLAMA_REV,
            "gguf_sha256": GGUF_SHA,
            "token_generator": TOKEN_GENERATOR,
            "configurations": [row["id"] for row in CONFIGS],
            "control": CONTROL_ID,
            "candidates": list(CANDIDATE_IDS),
            "selected_path": CONTROL_ID,
            "fusion_implemented": False,
            "claims_throughput": False,
            "claims_performance_improvement": False,
            "report_path": relpath(REPORT),
        }
    )
    if phase == "screen":
        payload = run_screen_phase(mode=mode, run_dir=run_dir)
    else:
        payload = run_stub_phase(phase, mode=mode)
    results[phase] = payload
    results["updated_utc"] = utc_now()
    if phase == "screen":
        results["screen"] = payload
        results["eligibility"] = payload
        results["status"] = payload.get("status")
        results["eligible"] = bool(payload.get("eligible"))
        if not payload.get("eligible"):
            results.update(decide_deferred_verdicts(str(payload.get("verdict"))))
            results["quality"] = {
                "status": "not_required",
                "reason": VERDICT_DEFER,
                "candidate_nll_measured": False,
            }
            results["numeric"] = {"status": "not_required", "reason": VERDICT_DEFER}
    if payload.get("independent_verdicts"):
        results["independent_verdicts"] = payload["independent_verdicts"]
    if payload.get("production_kept") is not None:
        results["production_kept"] = payload["production_kept"]
    if payload.get("selected_path"):
        results["selected_path"] = payload["selected_path"]
    write_report(results)
    dump_json(FIXTURE, results)
    dump_json(run_dir / "opt112_norm_q8_staging.json", results)
    counts = payload.get("native_counts") or {}
    print(
        "QW38_OPT112_NORM_Q8_STAGING_RESULT="
        + json.dumps(
            {
                "task": "OPT-112",
                "mode": mode,
                "phase": phase,
                "status": results.get("status"),
                "eligible": (results.get("screen") or {}).get("eligible"),
                "verdict": (results.get("screen") or {}).get("verdict"),
                "reason": (results.get("screen") or {}).get("reason"),
                "selected_path": results.get("selected_path"),
                "production_kept": results.get("production_kept"),
                "fusion_implemented": results.get("fusion_implemented"),
                "claims_throughput": results.get("claims_throughput"),
                "keep": False,
                "warmups": counts.get("warmups"),
                "samples": counts.get("samples"),
                "observed_warmups": counts.get("observed_warmups"),
                "observed_samples": counts.get("observed_samples"),
                "observed_candidates": counts.get("observed_candidates"),
                "observed_shapes": counts.get("observed_shapes"),
                "observed_tier": counts.get("observed_tier"),
                "pairs": counts.get("pairs"),
                "sample_ids": counts.get("sample_ids"),
                "acceptance_executed": counts.get("acceptance_executed"),
            }
        )
    )
    print(
        "QW38_OPT112_NATIVE_COUNTS="
        + json.dumps({"schema_version": 1, "task": "OPT-112", **counts})
    )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=PHASES)
    parser.add_argument(
        "--mode", required=True, choices=("feedback", "acceptance", "release")
    )
    parser.add_argument("--run-dir", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(args.mode, args.phase, Path(args.run_dir))
    except AdmissionError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
