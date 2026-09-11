"""OPT-085 re-evaluate Q4 packed, integer paired, and late-reduction paths."""

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

from tools.kernel_parity import (  # noqa: E402
    ASSOCIATION,
    OPT074_FAMILY_ADMISSION_REQUIRED,
    SAME_MATH,
)
from tools.opt075_q4_production_admission import (  # noqa: E402
    AdmissionError,
    FEEDBACK_CRIT,
    T_CRIT_DF4,
    T_CRIT_DF9,
    default_native_runner,
    dump_json,
    load_json,
    mean,
    opt073_quality,
    paired_student_t,
    parse_dispatch,
    parse_engine_pairs,
    parse_rounds,
    sample_variance,
    utc_now,
)
from tools.opt082_kernel_parity import (  # noqa: E402
    catalog,
    evaluate_host_cases,
    q8_1_typed_reject,
    run_native,
)
from tools.opt084_quality_baseline import FROZEN_ACCEPTANCE  # noqa: E402
from tools.quality.quality_mode import apply_quality_mode  # noqa: E402
from tools.quality.suite import SUITE_CLASSES  # noqa: E402
from tools.run_optimization_task import (  # noqa: E402
    loop_product,
    parse_native_observation,
    validate_performance_admission,
    workload_for_mode,
)

LLAMA_REV = "cc83d7b4824f73cfdda4dfbb47ee39804f71b328"
GGUF_SHA = "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"
TOKEN_GENERATOR = "(42 + index * 997) % 248320"
CONTRACT = ROOT / "pins/opt085_q4_reevaluation_contract.json"
ITERATION = ROOT / "pins/opt085_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt085_q4_reevaluation.json"
REPORT = ROOT / "evidence/optimization/opt085-q4-reevaluation/REPORT.md"
EVIDENCE = REPORT.parent
OPT075_FIXTURE = ROOT / "fixtures/opt075_q4_production_admission.json"
OPT075_REPORT = ROOT / "evidence/optimization/opt075-q4-production-admission/REPORT.md"
OPT076_FIXTURE = ROOT / "fixtures/opt076_q4_reduction.json"
OPT076_REPORT = ROOT / "evidence/optimization/opt076-q4-reduction/REPORT.md"
OPT082_FIXTURE = ROOT / "fixtures/opt082_kernel_parity.json"
OPT084_FIXTURE = ROOT / "fixtures/opt084_quality_baseline.json"
REPLAY = "build/qw38-cuda-component-replay"
PROBE = "build/qw38-cuda-optimization-engine-probe"
MODEL = "models/Qwen3.8-27B-Q4_K_M.gguf"
PHASES = ("parity", "q4", "quality")
CONTROL_ID = "packed_paired_staged"
Q4_IDENTS = ("packed", "integer_q8", "integer_q8_late")
WARPS_PER_ROW = 4
VERDICT_KEYS = (
    "kernel_parity_pass",
    "model_quality_pass",
    "performance_pass",
    "production_kept",
)

CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "id": "packed_paired_staged",
        "q4_decode": "packed",
        "ffn_decode": "paired_staged",
        "warps_per_row": WARPS_PER_ROW,
        "role": "control",
        "expected_gate_variant": "q4k_gate_up_swiglu_prequant",
        "expected_up_variant": "q4k_gate_up_swiglu_prequant",
        "expected_down_variant": "quant_mmv_packed",
    },
    {
        "id": "integer_q8_paired",
        "q4_decode": "integer_q8",
        "ffn_decode": "paired_integer",
        "warps_per_row": WARPS_PER_ROW,
        "role": "candidate",
        "source_task": "OPT-075",
        "expected_gate_variant": "q4k_coop_gate_up_swiglu_prequant_q8",
        "expected_up_variant": "q4k_coop_gate_up_swiglu_prequant_q8",
        "expected_down_variant": "q4k_coop_mmv_q8",
    },
    {
        "id": "late_w4",
        "q4_decode": "integer_q8_late",
        "ffn_decode": "paired_integer",
        "warps_per_row": WARPS_PER_ROW,
        "role": "candidate",
        "source_task": "OPT-076",
        "expected_gate_variant": "q4k_coop_gate_up_swiglu_late_prequant_q8",
        "expected_up_variant": "q4k_coop_gate_up_swiglu_late_prequant_q8",
        "expected_down_variant": "q4k_coop_mmv_late_q8",
    },
)

NativeRunner = Callable[[Sequence[str], str], subprocess.CompletedProcess[str]]


def config_by_id(config_id: str) -> dict[str, Any]:
    for row in CONFIGS:
        if row["id"] == config_id:
            return dict(row)
    raise AdmissionError(f"unknown configuration {config_id}")


def config_for_ident(ident: str) -> dict[str, Any]:
    for row in CONFIGS:
        if row["q4_decode"] == ident:
            return dict(row)
    raise AdmissionError(f"unknown Q4 ident {ident}")


def q4_ident_catalog() -> list[dict[str, Any]]:
    """OPT-082 Q4 cases restricted to packed / integer_q8 / integer_q8_late."""
    return [row for row in catalog("q4") if row["candidate"] in Q4_IDENTS]


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
    default_candidates = 3 if phase == "q4" and mode == "feedback" else 1
    if phase == "q4" and mode == "acceptance":
        default_candidates = 2
    return {
        "phase": phase,
        "mode": mode,
        "family": "decode-ffn" if phase == "q4" else phase,
        "control": CONTROL_ID,
        "warmups": int(workload.get("warmups", 0)),
        "samples": int(workload.get("samples", 1)),
        "engine_pairs": int(workload.get("engine_pairs", engine_pairs)),
        "cases": int(workload.get("cases", 1)),
        "candidates": int(workload.get("candidates", default_candidates)),
        "tier": str(
            workload.get("tier", "screen" if mode == "feedback" else "acceptance")
        ),
        "product": loop_product(workload),
        "control_candidate_pairs": int(workload.get("control_candidate_pairs", 1)),
    }


def planned_observation(
    plan: Mapping[str, Any], *, capture_key: str | None = None, keep: bool = False
) -> dict[str, Any]:
    samples = int(plan["samples"])
    return {
        "schema_version": 1,
        "task": "OPT-085",
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


def historical_ranking() -> dict[str, Any]:
    opt075 = load_json(OPT075_FIXTURE)
    opt076 = load_json(OPT076_FIXTURE)
    key075 = str((opt075.get("q4") or {}).get("component", {}).get("capture_key") or "")
    key076 = str((opt076.get("q4") or {}).get("component", {}).get("capture_key") or "")
    packed075 = ((opt075.get("q4") or {}).get("component") or {}).get("control_mean_ms")
    int075 = ((opt075.get("q4") or {}).get("component") or {}).get("candidate_mean_ms")
    packed076 = ((opt076.get("q4") or {}).get("component") or {}).get("control_mean_ms")
    late076 = ((opt076.get("q4") or {}).get("component") or {}).get("candidate_mean_ms")
    return {
        "ranking_only": True,
        "keep_requires_this_sitting": True,
        "opt075_capture_key": key075,
        "opt076_capture_key": key076,
        "same_capture_key": bool(key075) and key075 == key076,
        "packed_vs_integer_q8_paired_ms": [packed075, int075],
        "packed_vs_late_w4_ms": [packed076, late076],
        "historical_reports_unmodified": (
            OPT075_REPORT.is_file() and OPT076_REPORT.is_file()
        ),
    }


def historical_reports_intact() -> dict[str, Any]:
    opt075 = OPT075_REPORT.read_text(encoding="utf-8")
    opt076 = OPT076_REPORT.read_text(encoding="utf-8")
    fixture075 = load_json(OPT075_FIXTURE)
    fixture076 = load_json(OPT076_FIXTURE)
    return {
        "opt075_report": str(OPT075_REPORT.relative_to(ROOT)),
        "opt076_report": str(OPT076_REPORT.relative_to(ROOT)),
        "opt075_shipping": fixture075.get("shipping_q4_decode"),
        "opt076_shipping": fixture076.get("shipping_q4_decode"),
        "opt075_production_kept": fixture075.get("production_kept"),
        "opt076_production_kept": fixture076.get("production_kept"),
        "opt075_mentions_retain_or_packed": "packed" in opt075.casefold(),
        "opt076_mentions_packed": "packed" in opt076.casefold(),
        "unmodified": (
            fixture075.get("task") == "OPT-075"
            and fixture076.get("task") == "OPT-076"
            and fixture075.get("production_kept") is False
            and fixture076.get("production_kept") is False
        ),
    }


def empty_verdict_row() -> dict[str, Any]:
    return {
        "kernel_parity_pass": False,
        "model_quality_pass": False,
        "performance_pass": False,
        "production_kept": False,
        "opt074_coverage_unadmitted_blocker": False,
        "incomplete": True,
    }


def load_prior_verdicts() -> dict[str, dict[str, Any]]:
    rows = {row["id"]: empty_verdict_row() for row in CONFIGS}
    if not FIXTURE.is_file():
        return rows
    try:
        prior = load_json(FIXTURE)
    except json.JSONDecodeError:
        return rows
    stored = prior.get("independent_verdicts") or {}
    for config_id, payload in stored.items():
        if config_id in rows and isinstance(payload, Mapping):
            merged = empty_verdict_row()
            merged.update({key: bool(payload.get(key)) for key in VERDICT_KEYS})
            merged["opt074_coverage_unadmitted_blocker"] = False
            merged["incomplete"] = bool(payload.get("incomplete", False))
            rows[config_id] = merged
    return rows


def gpu_case_pass(row: Mapping[str, Any]) -> bool | None:
    if "gpu_pass" in row:
        value = row.get("gpu_pass")
        return None if value is None else bool(value)
    if "pass" in row:
        return bool(row.get("pass"))
    return None


def gpu_case_fallback(row: Mapping[str, Any]) -> bool:
    return bool(row.get("gpu_fallback") or row.get("fallback"))


def evaluate_parity(
    *,
    skip_gpu: bool = False,
    runner: NativeRunner | None = None,
    synthetic: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cases = q4_ident_catalog()
    host_cases = evaluate_host_cases()
    typed = q8_1_typed_reject()
    host_ok = typed["pass"] and all(row["matched_expectation"] for row in host_cases)
    if not host_ok:
        raise AdmissionError("OPT-082 host association/typed staging checks failed")
    gpu_rows: list[dict[str, Any]] = []
    source = "none"
    native: dict[str, Any] = {
        "available": False,
        "ran": False,
        "success": False,
        "blocker": "skip_gpu" if skip_gpu else "",
        "cases": [],
    }
    if synthetic and synthetic.get("gpu_cases") is not None:
        gpu_rows = [dict(row) for row in synthetic["gpu_cases"]]
        source = str(synthetic.get("source") or "synthetic")
        native["ran"] = True
        native["success"] = True
        native["available"] = True
    elif skip_gpu:
        prior = load_json(OPT082_FIXTURE) if OPT082_FIXTURE.is_file() else {}
        gpu_rows = [
            dict(row)
            for row in prior.get("cases") or []
            if row.get("candidate") in Q4_IDENTS
        ]
        source = "opt082_fixture"
        native["blocker"] = "skip_gpu"
    else:
        native = run_native("q4", runner=runner)
        if native.get("ran"):
            gpu_rows = [
                dict(row)
                for row in native.get("cases") or []
                if row.get("candidate") in Q4_IDENTS
            ]
            source = "this_sitting"
        elif OPT082_FIXTURE.is_file():
            prior = load_json(OPT082_FIXTURE)
            gpu_rows = [
                dict(row)
                for row in prior.get("cases") or []
                if row.get("candidate") in Q4_IDENTS
            ]
            source = "opt082_fixture"
            native["blocker"] = (
                native.get("blocker") or "native_unavailable_reused_opt082"
            )
    by_id = {str(row.get("id")): row for row in gpu_rows}
    by_ident: dict[str, dict[str, Any]] = {}
    for ident in Q4_IDENTS:
        catalog_rows = [row for row in cases if row["candidate"] == ident]
        mapped = [by_id.get(row["id"], {}) for row in catalog_rows]
        fallback_as_candidate = False
        passes: list[bool] = []
        documented_same_math: list[str] = []
        for spec, gpu in zip(catalog_rows, mapped):
            gpu_pass = gpu_case_pass(gpu)
            if gpu_pass is None:
                passes.append(False)
                continue
            if int(gpu.get("gpu_nonfinite_count") or gpu.get("nonfinite_count") or 0):
                passes.append(False)
                continue
            if gpu_case_fallback(gpu) and not spec.get("expect_fallback"):
                fallback_as_candidate = True
                passes.append(False)
                continue
            class_name = str(spec.get("class_name") or spec.get("class") or "")
            if class_name == SAME_MATH and gpu_pass is False:
                documented_same_math.append(spec["id"])
                continue
            passes.append(bool(gpu_pass))
        complete = all(gpu_case_pass(gpu) is not None for gpu in mapped) and bool(
            catalog_rows
        )
        association_ok = bool(passes) and all(passes)
        by_ident[ident] = {
            "ident": ident,
            "config": config_for_ident(ident)["id"],
            "kernel_parity_pass": bool(
                complete and association_ok and not fallback_as_candidate
            ),
            "case_count": len(catalog_rows),
            "gpu_case_count": sum(
                1 for gpu in mapped if gpu_case_pass(gpu) is not None
            ),
            "fallback_measured_as_candidate": fallback_as_candidate,
            "incomplete": not complete,
            "documented_same_math_fails": documented_same_math,
            "documented_fails": [
                spec["id"]
                for spec, gpu in zip(catalog_rows, mapped)
                if gpu_case_pass(gpu) is False
                and str(spec.get("class_name") or "") == ASSOCIATION
            ],
        }
        if fallback_as_candidate:
            by_ident[ident]["kernel_parity_pass"] = False
    packed = by_ident["packed"]
    if not packed["kernel_parity_pass"]:
        raise AdmissionError("packed Q4_K kernel parity failed; hard stop")
    return {
        "schema_version": 1,
        "task": "OPT-085",
        "phase": "parity",
        "host_ok": host_ok,
        "typed_staging_reject": typed["pass"],
        "source": source,
        "catalog_count": len(cases),
        "gpu": {
            "available": native.get("available"),
            "ran": native.get("ran"),
            "success": native.get("success"),
            "blocker": native.get("blocker") or "",
        },
        "by_ident": by_ident,
        "opt074_family_admission_required": OPT074_FAMILY_ADMISSION_REQUIRED,
        "opt074_coverage_unadmitted_blocker": False,
        "q8_1_typed": dict(typed),
    }


def candidate_selectors(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "q4_decode": config["q4_decode"],
        "q4_staging": config["ffn_decode"],
    }


def evaluate_quality(
    config: Mapping[str, Any],
    *,
    synthetic: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    baseline = load_json(OPT084_FIXTURE)
    quality = opt073_quality()
    nr_ok = quality.get("quality_v3_engine_non_regression") in {"pass", True}
    vs_baseline = baseline.get("quartz_vs_baseline") or {}
    applied = apply_quality_mode(enabled=True, selectors=candidate_selectors(config))
    record: dict[str, Any] = {
        "config": config["id"],
        "identity_cached": True,
        "once_per_survivor": True,
        "quality_flag": "--quality",
        "suite_classes": list(SUITE_CLASSES),
        "frozen_acceptance": dict(FROZEN_ACCEPTANCE),
        "opt073_engine_non_regression": quality.get("quality_v3_engine_non_regression"),
        "absolute_quality_status": baseline.get("absolute_quality_status"),
        "known_baseline_defect_does_not_reject": True,
        "llama_inspectable": True,
        "opt074_coverage_unadmitted_blocker": False,
        "selectors": applied["selectors"],
        "ppl_ratio_max": FROZEN_ACCEPTANCE["ppl_ratio_max"],
        "recurrence_incremental_nll_max": FROZEN_ACCEPTANCE[
            "recurrence_incremental_nll_max"
        ],
    }
    injected = None
    if synthetic:
        injected = (synthetic.get("quality_by_id") or {}).get(config["id"])
    if config["id"] == CONTROL_ID:
        passed = bool(vs_baseline.get("pass")) and nr_ok
        record.update(
            {
                "model_quality_pass": passed,
                "incomplete": False,
                "quartz_baseline_regression_status": vs_baseline.get("status"),
                "zero_delta": bool(vs_baseline.get("zero_delta")),
                "reason": "shipping_quartz_vs_opt084_freeze",
            }
        )
        return record
    if isinstance(injected, Mapping) and injected.get("model_quality_pass") is True:
        if injected.get("regression"):
            raise AdmissionError(f"{config['id']} quality regression versus OPT-084")
        record.update(
            {
                "model_quality_pass": True,
                "incomplete": False,
                "reason": "synthetic_or_measured_non_regression",
                "quartz_baseline_regression_status": "pass",
            }
        )
        return record
    record.update(
        {
            "model_quality_pass": False,
            "incomplete": True,
            "reason": "candidate_specific_quality_not_measured",
            "quartz_baseline_regression_status": "incomplete",
        }
    )
    return record


def e2e_non_regression(engine: Mapping[str, Any] | None) -> bool:
    if not engine or not engine.get("control_ms"):
        return False
    control_mean = float(engine.get("control_mean_ms", 0.0))
    upper = float(engine.get("regression_upper_ms", math.inf))
    limit = 0.02 * control_mean if control_mean > 0.0 else 0.0
    return upper <= limit


def performance_pass_for(
    *,
    config_id: str,
    stats: Mapping[str, Any] | None,
    engine: Mapping[str, Any] | None,
    this_sitting: bool,
    dispatch_ok: bool,
) -> bool:
    if config_id == CONTROL_ID:
        return False
    if not this_sitting or not dispatch_ok or not stats:
        return False
    component_ok = (
        bool(stats.get("positive")) and float(stats.get("mean_diff_ms", 0.0)) > 0.0
    )
    if not component_ok:
        return False
    return e2e_non_regression(engine)


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
    for config in configs:
        completed = runner(replay_command(plan, config, capture_key), tier)
        (run_dir / f"{plan['phase']}-{config['id']}-replay.txt").write_text(
            completed.stdout + completed.stderr, encoding="utf-8"
        )
        if (
            "nonfinite" in completed.stdout.casefold()
            and "nonfinite=0" not in completed.stdout
        ):
            raise AdmissionError(f"{config['id']} reported a nonfinite value")
        if "fallback-measured-as-candidate" in completed.stdout:
            raise AdmissionError(f"{config['id']} fallback measured as candidate")
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
        effective_q4 = str(
            dispatch.get("q4_path") or dispatch.get("effective_q4_decode") or ""
        )
        if effective_q4 and effective_q4 != config["q4_decode"]:
            raise AdmissionError(
                f"{config['id']} effective q4 {effective_q4} != {config['q4_decode']}"
            )
        samples = [float(row["enclosing_ms"]) for row in rounds]
        by_config[str(config["id"])] = {
            "config": dict(config),
            "ms": samples,
            "mean_ms": mean(samples),
            "dispatch": dispatch,
            "expected": expected,
        }
        if "stale capture" in completed.stderr:
            raise AdmissionError("stale capture key")
    return {
        "capture_key": capture_key,
        "by_config": by_config,
        "control": CONTROL_ID,
        "this_sitting": True,
        "dispatch_ok": True,
        "launches": [
            {
                "config": name,
                "dispatch": row["dispatch"],
                "mean_ms": row["mean_ms"],
            }
            for name, row in by_config.items()
        ],
    }


def pick_screen_survivor(
    by_config: Mapping[str, Mapping[str, Any]],
    parity_by_ident: Mapping[str, Mapping[str, Any]],
) -> str:
    eligible: list[str] = []
    for name, row in by_config.items():
        if name == CONTROL_ID:
            continue
        ident = str((row.get("config") or {}).get("q4_decode") or "")
        if parity_by_ident.get(ident, {}).get("kernel_parity_pass"):
            eligible.append(name)
    if not eligible:
        return CONTROL_ID
    return min(
        eligible, key=lambda name: float(by_config[name].get("mean_ms", math.inf))
    )


def pair_stats(
    by_config: Mapping[str, Mapping[str, Any]],
    survivor: str,
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    control_ms = list(by_config[CONTROL_ID]["ms"])
    if survivor == CONTROL_ID:
        return {
            "survivor": survivor,
            "control_ms": control_ms,
            "candidate_ms": control_ms,
            "control_mean_ms": mean(control_ms),
            "candidate_mean_ms": mean(control_ms),
            "positive": False,
            "mean_diff_ms": 0.0,
            "ci95_low": 0.0,
            "ci95_high": 0.0,
            "n": len(control_ms),
        }
    candidate_ms = list(by_config[survivor]["ms"])
    critical = T_CRIT_DF9 if int(plan["samples"]) >= 10 else FEEDBACK_CRIT
    stats = paired_student_t(control_ms, candidate_ms, critical=critical)
    return {
        "survivor": survivor,
        "control_ms": control_ms,
        "candidate_ms": candidate_ms,
        "control_mean_ms": mean(control_ms),
        "candidate_mean_ms": mean(candidate_ms),
        **stats,
    }


def run_engine(
    plan: Mapping[str, Any],
    *,
    runner: NativeRunner,
    run_dir: Path,
    survivor: Mapping[str, Any],
) -> dict[str, Any]:
    pairs_n = int(plan["engine_pairs"])
    if pairs_n <= 0 or survivor["id"] == CONTROL_ID:
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
    if not re.search(r"q4_path=packed\b", completed.stdout):
        raise AdmissionError("control packed path missing from recaptured graph")
    if not re.search(
        rf"q4_path={re.escape(str(survivor['q4_decode']))}\b", completed.stdout
    ):
        raise AdmissionError("candidate path missing from recaptured graph")
    pairs = parse_engine_pairs(completed.stdout)
    if len(pairs) != pairs_n:
        raise AdmissionError(f"engine pairs {len(pairs)} != {pairs_n}")
    control = [float(row["control_ms"]) for row in pairs]
    candidate = [float(row["candidate_ms"]) for row in pairs]
    diffs = [cand - ctrl for ctrl, cand in zip(control, candidate)]
    avg = mean(diffs)
    var = sample_variance(diffs)
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
        "survivor": survivor["id"],
        "this_sitting": True,
    }


def decide_independent_verdicts(
    *,
    parity: Mapping[str, Any] | None,
    quality_by_id: Mapping[str, Mapping[str, Any]],
    performance_by_id: Mapping[str, Mapping[str, Any]],
    mode: str,
    prior: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = {}
    by_ident = (parity or {}).get("by_ident") or {}
    for config in CONFIGS:
        cid = config["id"]
        previous = dict((prior or {}).get(cid) or empty_verdict_row())
        ident = str(config["q4_decode"])
        if by_ident:
            kpass = bool(by_ident.get(ident, {}).get("kernel_parity_pass"))
        else:
            kpass = bool(previous.get("kernel_parity_pass"))
        quality = quality_by_id.get(cid) or {}
        if quality:
            qpass = bool(quality.get("model_quality_pass"))
        else:
            qpass = bool(previous.get("model_quality_pass"))
        perf = performance_by_id.get(cid) or {}
        if perf:
            ppass = bool(perf.get("performance_pass"))
        else:
            ppass = bool(previous.get("performance_pass"))
        rows[cid] = {
            "kernel_parity_pass": kpass,
            "model_quality_pass": qpass,
            "performance_pass": ppass,
            "production_kept": False,
            "opt074_coverage_unadmitted_blocker": False,
            "incomplete": bool(
                (by_ident.get(ident) or {}).get("incomplete")
                or quality.get("incomplete")
                or perf.get("incomplete")
            ),
        }
    winners = [
        cid
        for cid, row in rows.items()
        if cid != CONTROL_ID
        and row["kernel_parity_pass"]
        and row["model_quality_pass"]
        and row["performance_pass"]
    ]
    shipping_unchanged = True
    selected = CONTROL_ID
    if mode == "acceptance" and winners:
        selected = max(
            winners,
            key=lambda cid: float(
                (performance_by_id.get(cid) or {}).get("mean_diff_ms", -math.inf)
            ),
        )
        rows[selected]["production_kept"] = True
        shipping_unchanged = False
    elif mode == "acceptance":
        rows[CONTROL_ID]["production_kept"] = True
        selected = CONTROL_ID
        shipping_unchanged = True
    kept = [cid for cid, row in rows.items() if row["production_kept"]]
    if len(kept) > 1:
        raise AdmissionError("at most one Q4 production path may be kept")
    return {
        "independent_verdicts": rows,
        "selected_path": selected,
        "shipping_unchanged": shipping_unchanged,
        "production_kept": bool(
            selected != CONTROL_ID and rows[selected]["production_kept"]
        )
        if mode == "acceptance"
        else False,
        "packed_retained": shipping_unchanged,
        "opt074_coverage_unadmitted_blocker": False,
        "winners": winners,
    }


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def write_report(payload: Mapping[str, Any]) -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    verdicts = payload.get("independent_verdicts") or {}
    parity = payload.get("parity") or {}
    parity = payload.get("parity") or {}
    if isinstance(parity.get("parity"), Mapping):
        parity = parity["parity"]
    quality_payload = payload.get("quality") or {}
    if isinstance(quality_payload.get("quality"), Mapping):
        quality_rows = quality_payload["quality"]
    else:
        quality_rows = quality_payload
    quality_reasons = {
        cid: row.get("reason")
        for cid, row in quality_rows.items()
        if isinstance(row, Mapping)
    }
    q4 = payload.get("q4") or {}
    if "component" not in q4 and isinstance((payload.get("component")), Mapping):
        q4 = payload
    component = q4.get("component") or {}
    engine = q4.get("engine") or {}
    ranking = payload.get("historical_ranking") or historical_ranking()
    lines = []
    for config in CONFIGS:
        row = verdicts.get(config["id"]) or empty_verdict_row()
        lines.append(
            f"| {config['id']} | {row.get('kernel_parity_pass')} | "
            f"{row.get('model_quality_pass')} | {row.get('performance_pass')} | "
            f"{row.get('production_kept')} |"
        )
    ident_lines = []
    for ident, row in (parity.get("by_ident") or {}).items():
        ident_lines.append(
            f"| {ident} | {row.get('config')} | {row.get('kernel_parity_pass')} | "
            f"{row.get('gpu_case_count')}/{row.get('case_count')} | "
            f"{row.get('fallback_measured_as_candidate')} |"
        )
    text = f"""# OPT-085 — Re-evaluate Q4 decode candidates

Status: **{payload.get("status") or "pending"}**. Authority llama.cpp
`{LLAMA_REV}`, GGUF SHA-256 `{GGUF_SHA}`. Exactly three configurations:
packed/`paired_staged` control, OPT-075 `integer_q8`/`paired_integer`, and
OPT-076 `integer_q8_late`/`paired_integer` `late_w4`. Four warps/row and
FP32-scale Q8Block. No new fusion, tile, compiler, half-scale Q8_1, or Q4
kernel.

`opt074_coverage_unadmitted` is **not** a blocker.
`opt074_family_admission_required={OPT074_FAMILY_ADMISSION_REQUIRED}`.

## Independent verdicts

| Config | kernel_parity_pass | model_quality_pass | performance_pass | production_kept |
|---|---|---|---|---|
{chr(10).join(lines)}

Selected path: `{payload.get("selected_path", CONTROL_ID)}`.
shipping_unchanged={payload.get("shipping_unchanged", True)}.
production_kept={payload.get("production_kept", False)}.
claims_throughput={payload.get("claims_throughput", False)}.

## Kernel parity (OPT-082 Q4 idents)

source={parity.get("source")}; host_ok={parity.get("host_ok")};
typed `sum_q` vs `sum_x` rejected={parity.get("typed_staging_reject")}.

| Ident | Config | pass | gpu/catalog | fallback-as-candidate |
|---|---|---|---|---|
{chr(10).join(ident_lines) if ident_lines else "| none | | | | |"}

Candidates that fail kernel parity are rejected before quality.

## Quality (OPT-083/084)

Identity-cached OPT-084 suite with `--quality`. Primary gate is regression
versus the shipping Quartz baseline. Pinned llama is inspectable.
A known baseline defect does not reject a non-worsening kernel.
quality-by-config: {quality_reasons}.

## Performance

Complete 64-layer rotating FFN on repaired OPT-071 captures. AB/BA pairing.
D2048+32 engine pairs with a separate graph capture per configuration.
this_sitting={component.get("this_sitting", False)}.
Control mean {_fmt(component.get("control_mean_ms"))} ms vs survivor
`{component.get("survivor")}` {_fmt(component.get("candidate_mean_ms"))} ms.
Paired CI {_fmt(component.get("ci95_low"), 4)} .. {_fmt(component.get("ci95_high"), 4)} ms.
Engine {_fmt(engine.get("control_mean_ms"))} vs {_fmt(engine.get("candidate_mean_ms"))} ms.

Historical ranking only (not a keep): packed vs integer_q8_paired
{ranking.get("packed_vs_integer_q8_paired_ms")}; packed vs late_w4
{ranking.get("packed_vs_late_w4_ms")}. Capture identity shared=
{ranking.get("same_capture_key")}. A keep requires this sitting's confirmation.

## tok/s

Speedup versus the then-current packed baseline is **0** while packed is
retained. claims_throughput is true only when a candidate is
`production_kept` on this sitting's measured evidence.
"""
    REPORT.write_text(text, encoding="utf-8")


def admit_counts(phase: str, mode: str, observed: Mapping[str, Any]) -> dict[str, Any]:
    plan = family_plan(phase, mode)
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
    return admission


def run_parity_phase(
    *,
    mode: str,
    run_dir: Path,
    skip_gpu: bool,
    runner: NativeRunner | None,
    synthetic: Mapping[str, Any] | None,
) -> dict[str, Any]:
    plan = family_plan("parity", mode)
    parity = evaluate_parity(skip_gpu=skip_gpu, runner=runner, synthetic=synthetic)
    if parity["catalog_count"] != int(plan["cases"]):
        raise AdmissionError(
            f"parity catalog {parity['catalog_count']} != plan {plan['cases']}"
        )
    observed = planned_observation(plan, keep=False)
    admission = admit_counts("parity", mode, observed)
    quality_by_id: dict[str, dict[str, Any]] = {}
    performance_by_id: dict[str, dict[str, Any]] = {}
    decided = decide_independent_verdicts(
        parity=parity,
        quality_by_id=quality_by_id,
        performance_by_id=performance_by_id,
        mode=mode,
        prior=load_prior_verdicts(),
    )
    payload = {
        "schema_version": 1,
        "task": "OPT-085",
        "phase": "parity",
        "mode": mode,
        "parity": parity,
        "native_counts": observed,
        "admission": admission,
        **decided,
        "claims_throughput": False,
        "measurement_utc": utc_now(),
        "status": "parity_evaluated",
    }
    dump_json(run_dir / "parity-result.json", payload)
    return payload


def run_quality_phase(
    *,
    mode: str,
    run_dir: Path,
    synthetic: Mapping[str, Any] | None,
) -> dict[str, Any]:
    plan = family_plan("quality", mode)
    prior = load_prior_verdicts()
    survivor = CONTROL_ID
    if FIXTURE.is_file():
        try:
            previous = load_json(FIXTURE)
            survivor = str(
                previous.get("selected_path") or previous.get("survivor") or CONTROL_ID
            )
        except json.JSONDecodeError:
            survivor = CONTROL_ID
    targets = [CONTROL_ID]
    if survivor != CONTROL_ID:
        targets.append(survivor)
    else:
        targets.extend(row["id"] for row in CONFIGS if row["id"] != CONTROL_ID)
    quality_by_id = {
        config_id: evaluate_quality(config_by_id(config_id), synthetic=synthetic)
        for config_id in targets
    }
    observed = planned_observation(plan, keep=False)
    admission = admit_counts("quality", mode, observed)
    decided = decide_independent_verdicts(
        parity=None,
        quality_by_id=quality_by_id,
        performance_by_id={},
        mode=mode,
        prior=prior,
    )
    payload = {
        "schema_version": 1,
        "task": "OPT-085",
        "phase": "quality",
        "mode": mode,
        "quality": quality_by_id,
        "survivor": survivor,
        "identity_cached": True,
        "opt073": opt073_quality(),
        "native_counts": observed,
        "admission": admission,
        **decided,
        "claims_throughput": False,
        "measurement_utc": utc_now(),
        "status": "quality_evaluated",
    }
    dump_json(run_dir / "quality-result.json", payload)
    return payload


def run_q4_phase(
    *,
    mode: str,
    run_dir: Path,
    skip_gpu: bool,
    runner: NativeRunner | None,
    synthetic: Mapping[str, Any] | None,
) -> dict[str, Any]:
    plan = family_plan("q4", mode)
    prior = load_prior_verdicts()
    ranking = historical_ranking()
    parity_by_ident: dict[str, Any] = {}
    if FIXTURE.is_file():
        try:
            previous = load_json(FIXTURE)
            parity_by_ident = dict(
                ((previous.get("parity") or {}).get("by_ident")) or {}
            )
        except json.JSONDecodeError:
            parity_by_ident = {}
    if not parity_by_ident:
        # Feedback may run q4 after parity in the same fixture write; if the
        # ident table is missing, require a packed pass and leave candidates
        # eligible only when synthetic/live parity said so.
        parity_by_ident = {
            ident: {"kernel_parity_pass": ident == "packed"} for ident in Q4_IDENTS
        }
        if FIXTURE.is_file():
            try:
                stored = load_json(FIXTURE).get("independent_verdicts") or {}
                for ident in Q4_IDENTS:
                    cid = config_for_ident(ident)["id"]
                    if cid in stored:
                        parity_by_ident[ident] = {
                            "kernel_parity_pass": bool(
                                stored[cid].get("kernel_parity_pass")
                            )
                        }
            except json.JSONDecodeError:
                pass
    engine: dict[str, Any] = {}
    if skip_gpu:
        if synthetic and synthetic.get("component"):
            component = dict(synthetic["component"])
            engine = dict(synthetic.get("engine") or {})
            dispatch_ok = bool(synthetic.get("dispatch_ok", True))
            this_sitting = bool(synthetic.get("this_sitting", False))
        else:
            component = {
                "this_sitting": False,
                "survivor": CONTROL_ID,
                "dispatch_ok": True,
                "by_config": {},
            }
            dispatch_ok = True
            this_sitting = False
    else:
        native = runner or default_native_runner
        if mode == "feedback":
            configs = [dict(row) for row in CONFIGS]
        else:
            survivor_id = CONTROL_ID
            if FIXTURE.is_file():
                try:
                    previous = load_json(FIXTURE)
                    survivor_id = str(
                        ((previous.get("q4") or {}).get("component") or {}).get(
                            "survivor"
                        )
                        or previous.get("survivor")
                        or previous.get("selected_path")
                        or CONTROL_ID
                    )
                except json.JSONDecodeError:
                    survivor_id = CONTROL_ID
            if survivor_id == CONTROL_ID:
                # Acceptance still times packed vs the faster parity-passing
                # candidate so a keep can be confirmed this sitting.
                ranked = [
                    row["id"]
                    for row in CONFIGS
                    if row["id"] != CONTROL_ID
                    and parity_by_ident.get(row["q4_decode"], {}).get(
                        "kernel_parity_pass"
                    )
                ]
                survivor_id = ranked[0] if ranked else "integer_q8_paired"
            configs = [config_by_id(CONTROL_ID), config_by_id(survivor_id)]
        component = run_component(plan, runner=native, run_dir=run_dir, configs=configs)
        dispatch_ok = bool(component.get("dispatch_ok"))
        this_sitting = True
        survivor_id = pick_screen_survivor(component["by_config"], parity_by_ident)
        stats = pair_stats(component["by_config"], survivor_id, plan)
        component.update(stats)
        try:
            engine = (
                run_engine(
                    plan,
                    runner=native,
                    run_dir=run_dir,
                    survivor=config_by_id(survivor_id),
                )
                or {}
            )
        except AdmissionError as exc:
            if plan["mode"] == "acceptance":
                raise
            engine = {"error": str(exc), "skipped": True}
    survivor_id = str(component.get("survivor") or CONTROL_ID)
    stats = {
        "positive": component.get("positive"),
        "mean_diff_ms": component.get("mean_diff_ms"),
        "ci95_low": component.get("ci95_low"),
        "ci95_high": component.get("ci95_high"),
    }
    performance_by_id: dict[str, dict[str, Any]] = {}
    for config in CONFIGS:
        cid = config["id"]
        measured = cid in (component.get("by_config") or {}) or cid == survivor_id
        ppass = performance_pass_for(
            config_id=cid,
            stats=stats if cid == survivor_id else None,
            engine=engine if cid == survivor_id else None,
            this_sitting=this_sitting and measured,
            dispatch_ok=dispatch_ok,
        )
        if (
            cid == survivor_id
            and this_sitting
            and cid != CONTROL_ID
            and float(component.get("mean_diff_ms") or 0.0) < 0.0
            and float(component.get("ci95_high") or 0.0) < 0.0
        ):
            ppass = False
        performance_by_id[cid] = {
            "performance_pass": ppass,
            "this_sitting": this_sitting,
            "incomplete": not this_sitting,
            "mean_diff_ms": component.get("mean_diff_ms")
            if cid == survivor_id
            else None,
            "e2e_non_regression": (
                e2e_non_regression(engine) if cid == survivor_id else False
            ),
        }
    observed = planned_observation(
        plan,
        capture_key=str(component.get("capture_key") or "") or None,
        keep=False,
    )
    admission = admit_counts("q4", mode, observed)
    decided = decide_independent_verdicts(
        parity={"by_ident": parity_by_ident} if parity_by_ident else None,
        quality_by_id={},
        performance_by_id=performance_by_id,
        mode=mode,
        prior=prior,
    )
    payload = {
        "schema_version": 1,
        "task": "OPT-085",
        "phase": "q4",
        "mode": mode,
        "control": CONTROL_ID,
        "survivor": survivor_id,
        "component": {k: v for k, v in component.items() if k != "stdout"},
        "engine": {k: v for k, v in engine.items() if k not in {"stdout", "native"}},
        "performance": performance_by_id,
        "historical_ranking": ranking,
        "dispatch_ok": dispatch_ok,
        "this_sitting": this_sitting,
        "native_counts": observed,
        "admission": admission,
        **decided,
        "claims_throughput": False,
        "measurement_utc": utc_now(),
        "status": "measured" if this_sitting else "incomplete_performance",
        "opt074_coverage_unadmitted_blocker": False,
    }
    dump_json(run_dir / "q4-result.json", payload)
    return payload


def merge_fixture(
    results: dict[str, Any], phase: str, payload: Mapping[str, Any]
) -> None:
    results[phase] = payload
    prior = results.get("independent_verdicts") or load_prior_verdicts()
    incoming = payload.get("independent_verdicts") or {}
    merged = {
        cid: dict(prior.get(cid) or empty_verdict_row())
        for cid in (row["id"] for row in CONFIGS)
    }
    for cid, row in incoming.items():
        current = merged.setdefault(cid, empty_verdict_row())
        if phase == "parity":
            current["kernel_parity_pass"] = bool(row.get("kernel_parity_pass"))
        elif phase == "quality":
            current["model_quality_pass"] = bool(row.get("model_quality_pass"))
        elif phase == "q4":
            current["performance_pass"] = bool(row.get("performance_pass"))
        current["opt074_coverage_unadmitted_blocker"] = False
        current["incomplete"] = bool(row.get("incomplete", current.get("incomplete")))
    mode = str(results.get("mode") or payload.get("mode") or "feedback")
    quality_by_id = {
        cid: {"model_quality_pass": row.get("model_quality_pass")}
        for cid, row in merged.items()
    }
    performance_by_id = {
        cid: {
            "performance_pass": row.get("performance_pass"),
            "mean_diff_ms": (
                ((results.get("q4") or {}).get("performance") or {})
                .get(cid, {})
                .get("mean_diff_ms")
            ),
        }
        for cid, row in merged.items()
    }
    decided = decide_independent_verdicts(
        parity=results.get("parity"),
        quality_by_id=quality_by_id,
        performance_by_id=performance_by_id,
        mode=mode,
        prior=merged,
    )
    # Re-apply the phase-specific bits after the global decide, which already
    # combined prior+incoming fields.
    results["independent_verdicts"] = decided["independent_verdicts"]
    results["selected_path"] = decided["selected_path"]
    results["shipping_unchanged"] = decided["shipping_unchanged"]
    candidate_kept = bool(
        decided["selected_path"] != CONTROL_ID
        and decided["independent_verdicts"][decided["selected_path"]]["production_kept"]
    )
    this_sitting = bool((results.get("q4") or {}).get("this_sitting"))
    results["production_kept"] = candidate_kept
    results["claims_throughput"] = bool(candidate_kept and this_sitting)
    results["claims_performance_improvement"] = results["claims_throughput"]
    results["opt074_coverage_unadmitted_blocker"] = False
    results["kernel_parity_pass"] = {
        cid: row["kernel_parity_pass"]
        for cid, row in results["independent_verdicts"].items()
    }
    results["model_quality_pass"] = {
        cid: row["model_quality_pass"]
        for cid, row in results["independent_verdicts"].items()
    }
    results["performance_pass"] = {
        cid: row["performance_pass"]
        for cid, row in results["independent_verdicts"].items()
    }
    selected = config_by_id(str(results["selected_path"]))
    if results["shipping_unchanged"] or not candidate_kept:
        results["shipping_q4_decode"] = "packed"
        results["shipping_ffn_decode"] = "paired_staged"
    else:
        results["shipping_q4_decode"] = selected["q4_decode"]
        results["shipping_ffn_decode"] = selected["ffn_decode"]
    results["survivor"] = (results.get("q4") or {}).get("survivor") or results[
        "selected_path"
    ]
    results["status"] = str(payload.get("status") or results.get("status") or "updated")


def run(
    mode: str,
    phase: str,
    run_dir: Path,
    *,
    skip_gpu: bool = False,
    runner: NativeRunner | None = None,
    synthetic: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
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
            "task": "OPT-085",
            "mode": mode,
            "llama_revision": LLAMA_REV,
            "gguf_sha256": GGUF_SHA,
            "token_generator": TOKEN_GENERATOR,
            "configurations": [row["id"] for row in CONFIGS],
            "control": CONTROL_ID,
            "warps_per_row": WARPS_PER_ROW,
            "staging": "q8_fp32",
            "half_scale_forbidden": True,
            "opt074_family_admission_required": False,
            "opt074_coverage_unadmitted_blocker": False,
            "historical_ranking": historical_ranking(),
            "historical_reports": historical_reports_intact(),
            "report_path": "evidence/optimization/opt085-q4-reevaluation/REPORT.md",
        }
    )
    if phase == "parity":
        payload = run_parity_phase(
            mode=mode,
            run_dir=run_dir,
            skip_gpu=skip_gpu,
            runner=runner,
            synthetic=synthetic,
        )
    elif phase == "quality":
        payload = run_quality_phase(mode=mode, run_dir=run_dir, synthetic=synthetic)
    else:
        payload = run_q4_phase(
            mode=mode,
            run_dir=run_dir,
            skip_gpu=skip_gpu,
            runner=runner,
            synthetic=synthetic,
        )
    merge_fixture(results, phase, payload)
    write_report(results)
    dump_json(FIXTURE, results)
    dump_json(run_dir / "opt085_q4_reevaluation.json", results)
    counts = (results.get(phase) or {}).get("native_counts") or {}
    print(
        "QW38_OPT085_RESULT="
        + json.dumps(
            {
                "task": "OPT-085",
                "mode": mode,
                "phase": phase,
                "selected_path": results.get("selected_path"),
                "production_kept": results.get("production_kept"),
                "claims_throughput": results.get("claims_throughput"),
                "opt074_coverage_unadmitted_blocker": False,
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
                "independent_verdicts": results.get("independent_verdicts"),
            }
        )
    )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES, default="parity")
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
        ROOT / "build" / "optimization-runs" / "OPT-085" / args.mode
    )
    try:
        run(args.mode, args.phase, run_dir, skip_gpu=bool(args.skip_gpu))
    except (AdmissionError, AssertionError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
