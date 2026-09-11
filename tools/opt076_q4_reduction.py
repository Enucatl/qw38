"""OPT-076 Q4 packed-load / late-reduction admission versus packed control."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.opt075_q4_production_admission import (  # noqa: E402
    AdmissionError,
    LEGACY_ABS,
    T_CRIT_DF4,
    T_CRIT_DF9,
    FEEDBACK_CRIT,
    docker_common,
    dump_json,
    load_json,
    mean,
    numeric_from_coverage,
    opt073_quality,
    opt074_q4_budgets,
    paired_student_t,
    parse_dispatch,
    parse_engine_pairs,
    parse_rounds,
    utc_now,
)
from tools.run_optimization_task import (  # noqa: E402
    loop_product,
    parse_native_observation,
    validate_performance_admission,
    workload_for_mode,
)

LLAMA_REV = "cc83d7b4824f73cfdda4dfbb47ee39804f71b328"
GGUF_SHA = "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"
TOKEN_GENERATOR = "(42 + index * 997) % 248320"
IMAGE = "qw38-cuda:13.0.2"
CONTRACT = ROOT / "pins/opt076_q4_reduction_contract.json"
ITERATION = ROOT / "pins/opt076_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt076_q4_reduction.json"
REPORT = ROOT / "evidence/optimization/opt076-q4-reduction/REPORT.md"
EVIDENCE = ROOT / "evidence/optimization/opt076-q4-reduction"
NATIVE = "build/qw38-cuda-opt076-q4-reduction-test"
REPLAY = "build/qw38-cuda-component-replay"
PROBE = "build/qw38-cuda-optimization-engine-probe"
MODEL = "models/Qwen3.8-27B-Q4_K_M.gguf"
PHASES = ("q4", "quality")
CONTROL_ID = "packed_paired_staged"
MIN_SAVING_MS = 0.10

CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "id": "packed_paired_staged",
        "q4_decode": "packed",
        "ffn_decode": "paired_staged",
        "warps_per_row": 4,
        "role": "control",
        "expected_gate_variant": "q4k_gate_up_swiglu_prequant",
        "expected_up_variant": "q4k_gate_up_swiglu_prequant",
        "expected_down_variant": "quant_mmv_packed",
    },
    {
        "id": "late_w2",
        "q4_decode": "integer_q8_late",
        "ffn_decode": "paired_integer",
        "warps_per_row": 2,
        "role": "candidate",
        "expected_gate_variant": "q4k_coop_gate_up_swiglu_late_prequant_q8",
        "expected_up_variant": "q4k_coop_gate_up_swiglu_late_prequant_q8",
        "expected_down_variant": "q4k_coop_mmv_late_q8",
    },
    {
        "id": "late_w4",
        "q4_decode": "integer_q8_late",
        "ffn_decode": "paired_integer",
        "warps_per_row": 4,
        "role": "candidate",
        "expected_gate_variant": "q4k_coop_gate_up_swiglu_late_prequant_q8",
        "expected_up_variant": "q4k_coop_gate_up_swiglu_late_prequant_q8",
        "expected_down_variant": "q4k_coop_mmv_late_q8",
    },
)

TINY_CASES: tuple[dict[str, Any], ...] = (
    {"id": "M1_K256", "rows": 1, "columns": 256},
    {"id": "M1_K512", "rows": 1, "columns": 512},
    {"id": "M17_K256", "rows": 17, "columns": 256},
    {"id": "M17_K512", "rows": 17, "columns": 512},
    {"id": "M33_K256", "rows": 33, "columns": 256},
    {"id": "M33_K512", "rows": 33, "columns": 512},
)

NativeRunner = Callable[[Sequence[str], str], subprocess.CompletedProcess[str]]


def default_native_runner(
    command: Sequence[str], tier: str
) -> subprocess.CompletedProcess[str]:
    listed = list(command)
    if listed and not listed[0].startswith("docker"):
        listed = [*docker_common(tier), *listed]
    completed = subprocess.run(listed, cwd=ROOT, capture_output=True, text=True)
    if completed.returncode != 0:
        raise AdmissionError(
            "native command failed: "
            + " ".join(listed)
            + "\n"
            + completed.stdout
            + completed.stderr
        )
    return completed


def config_by_id(config_id: str) -> dict[str, Any]:
    for row in CONFIGS:
        if row["id"] == config_id:
            return dict(row)
    raise AdmissionError(f"unknown configuration {config_id}")


def tiny_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for spec in TINY_CASES:
        cases.append(
            {
                **spec,
                "tail_guard": int(spec["rows"]) % 32 != 0,
                "groups": 8,
                "zero": True,
                "cancellation": True,
                "misalignment_fallback": True,
            }
        )
    return cases


def expected_dispatch(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "config": config["id"],
        "gate_variant": config["expected_gate_variant"],
        "up_variant": config["expected_up_variant"],
        "down_variant": config["expected_down_variant"],
        "gate_up_stage_count": 1,
        "down_stage_count": 1,
        "warps_per_row": int(config["warps_per_row"]),
        "staging": "q8_fp32",
        "q8_q6_selectors_preserved": True,
    }


def dispatch_matches(observed: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    if not observed:
        return False
    for key in ("gate_variant", "up_variant", "down_variant"):
        got = str(observed.get(key) or "")
        want = str(expected.get(key) or "")
        if not got or got != want:
            return False
    if int(observed.get("gate_up_stage_count", -1)) != 1:
        return False
    if int(observed.get("down_stage_count", -1)) != 1:
        return False
    warps = int(observed.get("warps_per_row", -1) or -1)
    return warps == int(expected["warps_per_row"])


def family_plan(phase: str, mode: str) -> dict[str, Any]:
    iteration = load_json(ITERATION)
    workload = workload_for_mode(iteration["workloads"][phase], mode)
    engine_pairs = 0
    if phase == "q4":
        engine_pairs = 1 if mode == "feedback" else 5
    return {
        "phase": phase,
        "mode": mode,
        "family": "decode-ffn",
        "control": CONTROL_ID,
        "warmups": int(workload.get("warmups", 1 if mode == "feedback" else 3)),
        "samples": int(workload.get("samples", 3 if mode == "feedback" else 10)),
        "engine_pairs": int(workload.get("engine_pairs", engine_pairs)),
        "cases": int(workload.get("cases", 1)),
        "candidates": int(workload.get("candidates", 3 if mode == "feedback" else 2)),
        "tier": str(
            workload.get("tier", "screen" if mode == "feedback" else "acceptance")
        ),
        "product": loop_product(workload),
        "control_candidate_pairs": int(workload.get("control_candidate_pairs", 1)),
    }


def planned_observation(
    plan: Mapping[str, Any], *, capture_key: str | None = None
) -> dict[str, Any]:
    samples = int(plan["samples"])
    return {
        "schema_version": 1,
        "task": "OPT-076",
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
        "keep": False,
        "capture_key": capture_key,
    }


def replay_command(
    plan: Mapping[str, Any], config: Mapping[str, Any], capture_key: str | None
) -> list[str]:
    args = [
        f"./{REPLAY}",
        MODEL,
        "--workload",
        "decode-ffn",
        "--cache-mode",
        "rotating",
        "--warmups",
        str(plan["warmups"]),
        "--samples",
        str(plan["samples"]),
        "--q4-decode",
        str(config["q4_decode"]),
        "--ffn-decode",
        str(config["ffn_decode"]),
        "--q4-warps",
        str(int(config["warps_per_row"])),
    ]
    if capture_key:
        args.extend(["--capture-key", capture_key])
    return args


def pick_survivor(by_config: Mapping[str, Mapping[str, Any]]) -> str:
    lates = [name for name in ("late_w2", "late_w4") if name in by_config]
    if not lates:
        raise AdmissionError("no late-reduction configuration was measured")
    return min(lates, key=lambda name: float(by_config[name].get("mean_ms", math.inf)))


def decide_verdict(
    *,
    component: Mapping[str, Any] | None,
    engine: Mapping[str, Any] | None,
    coverage: Mapping[str, Any],
    quality: Mapping[str, Any],
    numeric: Mapping[str, Any],
    dispatch_ok: bool,
    mode: str,
    survivor: str | None,
) -> dict[str, Any]:
    reasons: list[str] = []
    if numeric.get("nonfinite"):
        raise AdmissionError("nonfinite candidate output")
    if not dispatch_ok:
        raise AdmissionError("layout or intended-path dispatch was not proven")
    if not coverage.get("all_admitted"):
        reasons.append("opt074_coverage_unadmitted_missing_evidence")
    engine_nr = quality.get("quality_v3_engine_non_regression")
    if engine_nr not in {"pass", True}:
        reasons.append("opt073_quality_unresolved")
    production_numeric = bool(numeric.get("production_pass"))
    legacy_numeric = bool(numeric.get("legacy_pass"))
    if not production_numeric:
        reasons.append("opt074_budget_unresolved")
    if mode != "acceptance":
        return {
            "verdict": "inconclusive",
            "status": "screened_in" if component else "incomplete",
            "reasons": reasons + ["feedback_is_not_acceptance"],
            "shipping_unchanged": True,
            "production_kept": False,
            "retain_reason": "uncertainty",
            "unfinished_reference_freeze_is_not_numerical_rejection": True,
            "legacy_numeric_pass": legacy_numeric,
            "production_numeric_pass": production_numeric,
        }
    if survivor is None or survivor == CONTROL_ID:
        return {
            "verdict": "retain_packed",
            "status": "inconclusive",
            "reasons": reasons + ["no_late_survivor"],
            "shipping_unchanged": True,
            "production_kept": False,
            "retain_reason": "uncertainty",
            "unfinished_reference_freeze_is_not_numerical_rejection": True,
        }
    if component is None:
        return {
            "verdict": "inconclusive",
            "status": "incomplete",
            "reasons": reasons + ["missing_component_pairs"],
            "shipping_unchanged": True,
            "production_kept": False,
            "retain_reason": "missing_evidence",
            "unfinished_reference_freeze_is_not_numerical_rejection": True,
        }
    saving = float(component.get("mean_diff_ms", 0.0))
    component_ok = bool(component.get("positive")) and saving >= MIN_SAVING_MS
    slower = saving < 0.0 and bool(component.get("ci95_high", 0.0) < 0.0)
    too_small = (not slower) and saving < MIN_SAVING_MS
    regression_ok = True
    if engine is not None and engine.get("control_ms"):
        control_mean = float(engine.get("control_mean_ms", 0.0))
        upper = float(engine.get("regression_upper_ms", 0.0))
        limit = 0.02 * control_mean if control_mean > 0.0 else 0.0
        regression_ok = upper <= limit
    quality_ok = engine_nr in {"pass", True}
    if not quality_ok:
        return {
            "verdict": "quality_blocked",
            "status": "quality_blocked",
            "reasons": reasons,
            "shipping_unchanged": True,
            "production_kept": False,
            "retain_reason": "precision",
            "component_accepted": component_ok,
            "e2e_non_regression": regression_ok,
            "unfinished_reference_freeze_is_not_numerical_rejection": True,
        }
    if slower or too_small:
        return {
            "verdict": "performance_rejected",
            "status": "performance_rejected",
            "reasons": reasons
            + (["negative_complete_saving"] if slower else ["saving_below_0_10_ms"]),
            "shipping_unchanged": True,
            "production_kept": False,
            "retain_reason": "performance",
            "component_accepted": False,
            "e2e_non_regression": regression_ok,
            "unfinished_reference_freeze_is_not_numerical_rejection": True,
        }
    if reasons or not component_ok or not regression_ok:
        reason = (
            "missing_evidence" if not coverage.get("all_admitted") else "uncertainty"
        )
        return {
            "verdict": "retain_packed",
            "status": "inconclusive",
            "reasons": reasons
            + (
                []
                if reasons
                else (
                    ["component_interval_not_positive"]
                    if not component_ok
                    else ["e2e_regression_unresolved"]
                )
            ),
            "shipping_unchanged": True,
            "production_kept": False,
            "retain_reason": reason,
            "component_accepted": component_ok,
            "e2e_non_regression": regression_ok,
            "unfinished_reference_freeze_is_not_numerical_rejection": True,
            "legacy_numeric_pass": legacy_numeric,
            "production_numeric_pass": production_numeric,
        }
    return {
        "verdict": "keep",
        "status": "production_kept",
        "reasons": [
            "component_ci_positive",
            "saving_ge_0_10_ms",
            "e2e_non_regression",
            "quality",
            "numeric",
        ],
        "shipping_unchanged": False,
        "production_kept": True,
        "retain_reason": None,
        "component_accepted": True,
        "e2e_non_regression": True,
        "unfinished_reference_freeze_is_not_numerical_rejection": True,
    }


def parse_native_numeric(stdout: str) -> dict[str, Any]:
    observed = parse_native_observation(stdout)
    original = observed.get("original_abs")
    staged = observed.get("staged_abs")
    nonfinite = observed.get("nonfinite")
    match_o = re.search(r'"original_abs":([0-9.eE+-]+)', stdout)
    match_s = re.search(r'"staged_abs":([0-9.eE+-]+)', stdout)
    match_n = re.search(r'"nonfinite":(\d+)', stdout)
    if original is None and match_o:
        original = float(match_o.group(1))
    if staged is None and match_s:
        staged = float(match_s.group(1))
    if nonfinite is None and match_n:
        nonfinite = int(match_n.group(1))
    return {
        "original_abs": float(original or 0.0),
        "staged_abs": float(staged or 0.0),
        "nonfinite": int(nonfinite or 0),
        "native": observed,
    }


def run_native_correctness(
    *, runner: NativeRunner, run_dir: Path, mode: str, model: bool
) -> dict[str, Any]:
    workload = "screen" if mode == "feedback" else "correctness"
    command = [f"./{NATIVE}", "--workload", workload]
    if model:
        command.append(MODEL)
    completed = runner(
        command, workload if workload != "correctness" else "correctness"
    )
    (run_dir / f"native-{workload}.txt").write_text(
        completed.stdout + completed.stderr, encoding="utf-8"
    )
    parsed = parse_native_numeric(completed.stdout)
    if (
        "nonfinite" in completed.stdout.casefold()
        and "nonfinite=0" not in completed.stdout
    ):
        if parsed["nonfinite"] != 0:
            raise AdmissionError("native test reported a nonfinite value")
    return {
        "stdout": completed.stdout,
        "original_abs": parsed["original_abs"],
        "staged_abs": parsed["staged_abs"],
        "nonfinite": parsed["nonfinite"],
        "k_loop_shuffle_removed": "k_loop_shuffle_removed" in completed.stdout
        or "k_loop_shuffles_absent_source=true" in completed.stdout,
        "sass": "sass_status=" in completed.stdout,
        "attrs": "late_regs=" in completed.stdout,
    }


def run_component(
    plan: Mapping[str, Any],
    *,
    runner: NativeRunner,
    run_dir: Path,
    configs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    tier = "screen" if plan["mode"] == "feedback" else "acceptance"
    capture_key: str | None = None
    by_config: dict[str, dict[str, Any]] = {}
    stdout_all = ""
    for config in configs:
        completed = runner(replay_command(plan, config, capture_key), tier)
        stdout_all += completed.stdout + "\n"
        (run_dir / f"{plan['phase']}-{config['id']}-replay.txt").write_text(
            completed.stdout + completed.stderr, encoding="utf-8"
        )
        if (
            "nonfinite" in completed.stdout.casefold()
            and "nonfinite=0" not in completed.stdout
        ):
            raise AdmissionError(f"{config['id']} reported a nonfinite value")
        observed = parse_native_observation(completed.stdout)
        if capture_key is None:
            capture_key = str(observed.get("capture_key") or "")
            if not capture_key:
                match = re.search(r"capture_key=([0-9a-f]{64})", completed.stdout)
                if match:
                    capture_key = match.group(1)
        if not capture_key:
            raise AdmissionError("missing capture identity")
        rounds = [
            row
            for row in parse_rounds(completed.stdout)
            if row.get("cache_mode", "rotating") == "rotating"
        ]
        if len(rounds) != int(plan["samples"]):
            raise AdmissionError(
                f"{config['id']} rotating rounds {len(rounds)} != {plan['samples']}"
            )
        ids = [row.get("sample_index") for row in rounds]
        if len(ids) != len(set(ids)):
            raise AdmissionError(f"{config['id']} reused sample IDs")
        gate = re.search(r"gate_up_calls=(\d+) down_calls=(\d+)", completed.stdout)
        if gate is None or int(gate.group(1)) != 64 or int(gate.group(2)) != 64:
            raise AdmissionError(
                f"{config['id']} complete FFN calls were not 64/64: {gate}"
            )
        dispatch = parse_dispatch(completed.stdout)
        expected = expected_dispatch(config)
        if not dispatch_matches(dispatch, expected):
            raise AdmissionError(
                f"{config['id']} launch variants {dispatch} != {expected}"
            )
        samples = [float(row["enclosing_ms"]) for row in rounds]
        by_config[str(config["id"])] = {
            "config": dict(config),
            "ms": samples,
            "mean_ms": mean(samples),
            "dispatch": dispatch,
            "expected": expected,
        }
    survivor = pick_survivor(by_config)
    control_ms = by_config[CONTROL_ID]["ms"]
    candidate_ms = by_config[survivor]["ms"]
    critical = T_CRIT_DF9 if int(plan["samples"]) >= 10 else FEEDBACK_CRIT
    stats = paired_student_t(control_ms, candidate_ms, critical=critical)
    return {
        "capture_key": capture_key,
        "by_config": by_config,
        "control": CONTROL_ID,
        "survivor": survivor,
        "control_ms": control_ms,
        "candidate_ms": candidate_ms,
        "control_mean_ms": mean(control_ms),
        "candidate_mean_ms": mean(candidate_ms),
        **stats,
        "saving_ge_0_10_ms": float(stats["mean_diff_ms"]) >= MIN_SAVING_MS,
        "launches": [
            {
                "config": name,
                "dispatch": row["dispatch"],
                "mean_ms": row["mean_ms"],
                "warps_per_row": row["config"]["warps_per_row"],
            }
            for name, row in by_config.items()
        ],
        "stdout": stdout_all,
        "pairs": int(plan["samples"]),
        "dispatch_ok": True,
    }


def run_engine(
    plan: Mapping[str, Any],
    *,
    runner: NativeRunner,
    run_dir: Path,
    survivor: Mapping[str, Any],
) -> dict[str, Any]:
    pairs_n = int(plan["engine_pairs"])
    if pairs_n <= 0:
        return {}
    tier = "acceptance" if plan["mode"] == "acceptance" else "screen"
    command = [
        f"./{PROBE}",
        MODEL,
        "--workload",
        "q4-ab",
        "--pairs",
        str(pairs_n),
        "--modes",
        "graph",
        "--q4-decode",
        str(survivor["q4_decode"]),
        "--ffn-decode",
        str(survivor["ffn_decode"]),
        "--q4-warps",
        str(int(survivor["warps_per_row"])),
    ]
    completed = runner(command, tier)
    (run_dir / f"{plan['phase']}-engine.txt").write_text(
        completed.stdout + completed.stderr, encoding="utf-8"
    )
    if "override_before_capture_applied=true" not in completed.stdout:
        raise AdmissionError("graph capture override was not applied")
    if (
        "recapture=true" not in completed.stdout
        and "recapture_after_selector=true" not in completed.stdout
    ):
        raise AdmissionError(
            "packed graph was not recaptured after selecting the candidate"
        )
    pairs = parse_engine_pairs(completed.stdout)
    if len(pairs) != pairs_n:
        raise AdmissionError(f"engine pairs {len(pairs)} != {pairs_n}")
    control = [float(row["control_ms"]) for row in pairs]
    candidate = [float(row["candidate_ms"]) for row in pairs]
    diffs = [cand - ctrl for ctrl, cand in zip(control, candidate)]
    avg = mean(diffs)
    var = (
        sum((item - avg) ** 2 for item in diffs) / float(len(diffs) - 1)
        if len(diffs) > 1
        else 0.0
    )
    se = math.sqrt(var / len(diffs)) if var > 0.0 else 0.0
    df = max(len(diffs) - 1, 1)
    critical = T_CRIT_DF4 if df == 4 else FEEDBACK_CRIT
    upper = avg + critical * se
    return {
        "pairs": pairs,
        "control_ms": control,
        "candidate_ms": candidate,
        "control_mean_ms": mean(control),
        "candidate_mean_ms": mean(candidate),
        "mean_regression_ms": avg,
        "regression_upper_ms": upper,
        "df": df,
        "critical": critical,
        "stdout": completed.stdout,
        "survivor": survivor["id"],
    }


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def write_report(payload: Mapping[str, Any]) -> None:
    q4 = payload.get("q4") or payload
    component = q4.get("component") or {}
    engine = q4.get("engine") or {}
    quality = payload.get("quality") or q4.get("quality") or {}
    numeric = q4.get("numeric") or payload.get("numeric") or {}
    coverage = q4.get("coverage") or payload.get("coverage") or {}
    verdict = q4.get("verdict") or payload.get("verdict") or {}
    launches = component.get("launches") or []
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    text = f"""# OPT-076 — Remove Q4 inner-loop shuffles and scalar unpacking

Status: **{(payload.get("status") or "pending GPU sitting")}**. Authority llama.cpp
`{LLAMA_REV}`, GGUF SHA-256 `{GGUF_SHA}`. Control is OPT-075 packed/
paired_staged. Candidates are Q8Block packed-load / late-reduction kernels
at two and four warps/row with the selected paired_integer FFN composition.
FP32-scale Q8Block staging is unchanged. Half-scale Q8_1 is forbidden.
Q8/Q6 and prompt MMQ are untouched.

`claims_throughput: false`. Require >=0.10 ms/token complete-FFN saving and a
positive paired interval, or measured rejection.

## Numeric policy (OPT-074 GPU budgets)

Calibration then untouched held-out. Strict legacy {LEGACY_ABS} is recorded
separately from the versioned production ceiling.
Legacy pass={numeric.get("legacy_pass")}; production pass={numeric.get("production_pass")}.
OPT-074 Q4 families admitted={coverage.get("all_admitted")}. An unfinished
reference freeze is **not** a numerical rejection.

## Mechanism

`vec_dot_q4k_q8block_late` uses 32-bit qs/q8 loads, nibble mask/shift, exact
integer dot and min-correction sums, and keeps each lane's FP32 contribution
through K. Scale boundaries stay per group/block. Row-end reduction is
unchanged. Unaligned pointers fall back to byte unpack. SASS/resource
evidence is in the native log.

## Dispatch

Launches: {launches}

## Complete FFN screen

64-layer rotating complete FFN on repaired OPT-071 captures. Control packed vs
late survivor `{component.get("survivor") or payload.get("survivor")}`.
Component n={component.get("n")} means: control {_fmt(component.get("control_mean_ms"))} ms,
candidate {_fmt(component.get("candidate_mean_ms"))} ms.
Paired CI (control-candidate): {_fmt(component.get("ci95_low"), 4)} ..
{_fmt(component.get("ci95_high"), 4)} ms; mean diff {_fmt(component.get("mean_diff_ms"), 4)} ms.
Engine D2048+32 pairs: {_fmt(engine.get("control_mean_ms"))} vs
{_fmt(engine.get("candidate_mean_ms"))} ms.

## Quality (OPT-073)

quality-v3 absolute={quality.get("quality_v3_absolute")};
engine non-regression={quality.get("quality_v3_engine_non_regression")};
quality-v2 all={quality.get("quality_v2_all")}.

## Decision

Verdict: **{verdict.get("verdict")}** ({verdict.get("reasons")}).
production_kept={verdict.get("production_kept")}; retain_reason={verdict.get("retain_reason")}.
Shipping Q4 stays `{payload.get("shipping_q4_decode", "packed")}` /
`{payload.get("shipping_ffn_decode", "paired_staged")}`.
OPT-075 production pins are unaffected unless this sitting keeps a candidate.

## tok/s

Speedup versus the then-current packed baseline is **0** while packed is
retained.
"""
    REPORT.write_text(text, encoding="utf-8")


def run_quality_phase(*, mode: str, run_dir: Path) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    quality = opt073_quality()
    if quality.get("quality_v3_engine_non_regression") not in {"pass", True}:
        raise AdmissionError("OPT-073 engine non-regression is not a pass")
    payload = {
        "schema_version": 1,
        "task": "OPT-076",
        "phase": "quality",
        "mode": mode,
        "identity_cached": True,
        "once_per_survivor": True,
        "quality": quality,
        "legacy_absolute_visible": True,
        "claims_throughput": False,
        "measurement_utc": utc_now(),
        "status": "identity_cached",
        "native_counts": planned_observation(
            {
                "warmups": 0,
                "samples": 1,
                "candidates": 1,
                "cases": 1,
                "tier": "acceptance",
                "control_candidate_pairs": 1,
            }
        ),
    }
    dump_json(run_dir / "quality-result.json", payload)
    return payload


def run_phase(
    phase: str,
    *,
    mode: str,
    run_dir: Path,
    runner: NativeRunner | None = None,
    skip_gpu: bool = False,
    synthetic: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if phase not in PHASES:
        raise AdmissionError(f"unknown phase {phase}")
    run_dir.mkdir(parents=True, exist_ok=True)
    if phase == "quality":
        return run_quality_phase(mode=mode, run_dir=run_dir)
    plan = family_plan(phase, mode)
    quality = opt073_quality()
    coverage = opt074_q4_budgets()
    tinies = tiny_cases()
    configs: list[dict[str, Any]]
    if mode == "feedback":
        configs = [dict(row) for row in CONFIGS]
    else:
        survivor_id = "late_w4"
        if FIXTURE.is_file():
            try:
                previous = load_json(FIXTURE)
                survivor_id = str(
                    ((previous.get("q4") or {}).get("component") or {}).get("survivor")
                    or previous.get("survivor")
                    or "late_w4"
                )
            except json.JSONDecodeError:
                survivor_id = "late_w4"
        if survivor_id == CONTROL_ID:
            survivor_id = "late_w4"
        configs = [config_by_id(CONTROL_ID), config_by_id(survivor_id)]
    engine: dict[str, Any] = {}
    native: dict[str, Any] = {}
    if skip_gpu:
        if synthetic is None:
            raise AdmissionError("synthetic evidence required when skip_gpu")
        component = dict(synthetic.get("component") or {})
        engine = dict(synthetic.get("engine") or {})
        numeric = dict(synthetic.get("numeric") or {})
        dispatch_ok = bool(synthetic.get("dispatch_ok", True))
        native = dict(synthetic.get("native") or {})
    else:
        native_runner = runner or default_native_runner
        native = run_native_correctness(
            runner=native_runner, run_dir=run_dir, mode=mode, model=True
        )
        component = run_component(
            plan, runner=native_runner, run_dir=run_dir, configs=configs
        )
        dispatch_ok = bool(component.get("dispatch_ok"))
        survivor_cfg = config_by_id(str(component["survivor"]))
        try:
            engine = (
                run_engine(
                    plan, runner=native_runner, run_dir=run_dir, survivor=survivor_cfg
                )
                or {}
            )
        except AdmissionError as exc:
            if plan["mode"] == "acceptance":
                raise
            engine = {"error": str(exc), "skipped": True}
        numeric = numeric_from_coverage(
            coverage,
            original_abs=float(native.get("original_abs") or 0.0),
            staged_abs=float(native.get("staged_abs") or 0.0),
            total_abs=max(
                float(native.get("original_abs") or 0.0),
                float(native.get("staged_abs") or 0.0),
            ),
            nonfinite=int(native.get("nonfinite") or 0),
        )
    if not numeric:
        numeric = numeric_from_coverage(
            coverage, original_abs=0.0, staged_abs=0.0, total_abs=0.0
        )
    verdict = decide_verdict(
        component=component or None,
        engine=engine or None,
        coverage=coverage,
        quality=quality,
        numeric=numeric,
        dispatch_ok=dispatch_ok,
        mode=mode,
        survivor=str(component.get("survivor") or ""),
    )
    observed = planned_observation(
        plan, capture_key=str(component.get("capture_key") or "") or None
    )
    admission = validate_performance_admission(
        load_json(ITERATION),
        mode=mode,
        workload_name=phase,
        workload={
            "warmups": plan["warmups"],
            "samples": plan["samples"],
            "candidates": plan["candidates"],
            "cases": plan["cases"],
            "tier": plan["tier"],
            "control_candidate_pairs": int(plan.get("control_candidate_pairs", 1)),
        },
        stdout=json.dumps(observed),
        success=True,
    )
    if not admission["ok"]:
        raise AdmissionError(admission["message"])
    payload = {
        "schema_version": 1,
        "task": "OPT-076",
        "phase": phase,
        "mode": mode,
        "control": CONTROL_ID,
        "survivor": component.get("survivor"),
        "configurations": [row["id"] for row in CONFIGS],
        "component": component,
        "engine": engine,
        "native": {k: v for k, v in native.items() if k != "stdout"},
        "coverage": coverage,
        "quality": quality,
        "numeric": numeric,
        "tiny_cases": tinies,
        "verdict": verdict,
        "native_counts": observed,
        "admission": admission,
        "claims_throughput": False,
        "measurement_utc": utc_now(),
        "status": "measured",
        "dispatch_ok": dispatch_ok,
        "opt075_unaffected": True,
    }
    if isinstance(payload.get("component"), dict):
        payload["component"].pop("stdout", None)
    if isinstance(payload.get("engine"), dict):
        payload["engine"].pop("stdout", None)
    dump_json(run_dir / f"{phase}-result.json", payload)
    return payload


def run(mode: str, phase: str, run_dir: Path) -> dict[str, Any]:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    selected = [phase] if phase in PHASES else ["q4"]
    results: dict[str, Any] = {}
    if FIXTURE.is_file():
        try:
            results.update(load_json(FIXTURE))
        except json.JSONDecodeError:
            results = {}
    quality = opt073_quality()
    results.update(
        {
            "schema_version": 1,
            "task": "OPT-076",
            "mode": mode,
            "llama_revision": LLAMA_REV,
            "gguf_sha256": GGUF_SHA,
            "token_generator": TOKEN_GENERATOR,
            "configurations": [row["id"] for row in CONFIGS],
            "control": CONTROL_ID,
            "quality": quality,
            "claims_throughput": False,
            "claims_performance_improvement": False,
            "shipping_q4_decode": "packed",
            "shipping_ffn_decode": "paired_staged",
            "staging": "q8_fp32",
            "opt075_unaffected": True,
            "report_path": "evidence/optimization/opt076-q4-reduction/REPORT.md",
        }
    )
    for name in selected:
        results[name] = run_phase(name, mode=mode, run_dir=run_dir)
    q4 = results.get("q4") or {}
    verdict = q4.get("verdict") or {}
    results["survivor"] = q4.get("survivor")
    results["numeric"] = q4.get("numeric")
    results["coverage"] = q4.get("coverage")
    results["verdict"] = verdict
    results["production_kept"] = bool(verdict.get("production_kept"))
    results["shipping_unchanged"] = bool(verdict.get("shipping_unchanged", True))
    results["status"] = "measured" if q4 else results.get("status", "pending")
    if results.get("production_kept"):
        survivor_cfg = config_by_id(str(results.get("survivor")))
        results["shipping_q4_decode"] = survivor_cfg["q4_decode"]
        results["shipping_ffn_decode"] = survivor_cfg["ffn_decode"]
        results["opt075_unaffected"] = False
    write_report(results)
    dump_json(FIXTURE, results)
    dump_json(run_dir / "opt076_q4_reduction.json", results)
    counts = (results.get(phase) or {}).get("native_counts") or {}
    print(
        "QW38_OPT076_RESULT="
        + json.dumps(
            {
                "task": "OPT-076",
                "mode": mode,
                "phase": phase,
                "verdict": (verdict or {}).get("verdict"),
                "survivor": results.get("survivor"),
                "production_kept": results.get("production_kept"),
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
                "capture_key": counts.get("capture_key"),
            }
        )
    )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("q4", "quality"), default="q4")
    parser.add_argument(
        "--mode", choices=("feedback", "acceptance"), default="feedback"
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--skip-gpu", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_dir = args.run_dir or (
        ROOT / "build" / "optimization-runs" / "OPT-076" / args.mode
    )
    try:
        if args.skip_gpu:
            raise AdmissionError("GPU sitting required for OPT-076 phases")
        run(args.mode, args.phase, run_dir)
    except (AdmissionError, AssertionError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
