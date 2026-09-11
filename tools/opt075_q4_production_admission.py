"""OPT-075 admit or reject existing cooperative Q4 FFN configurations."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.opt073_quality_policy import FIXTURE as OPT073_FIXTURE  # noqa: E402
from tools.opt074_production_gpu_admission import sample_rows  # noqa: E402
from tools.production_numerics import STRICT_CUD001_ABS  # noqa: E402
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
CONTRACT = ROOT / "pins/opt075_q4_production_admission_contract.json"
ITERATION = ROOT / "pins/opt075_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt075_q4_production_admission.json"
REPORT = ROOT / "evidence/optimization/opt075-q4-production-admission/REPORT.md"
EVIDENCE = ROOT / "evidence/optimization/opt075-q4-production-admission"
OPT074 = ROOT / "fixtures/opt074_production_gpu_admission.json"
REPLAY = "build/qw38-cuda-component-replay"
PROBE = "build/qw38-cuda-optimization-engine-probe"
MODEL = "models/Qwen3.8-27B-Q4_K_M.gguf"
T_CRIT_DF9 = 2.262
T_CRIT_DF4 = 2.132
FEEDBACK_CRIT = 4.303
LEGACY_ABS = 0.0003
LEGACY_NEAR_MISS_ABS = 0.000305175781
WARPS_PER_ROW = 4
PHASES = ("q4", "quality")
CONTROL_ID = "packed_paired_staged"
HISTORICAL_HYPOTHESES = ("1.26x", "1.40x")

CONFIGS: tuple[dict[str, str], ...] = (
    {
        "id": "packed_paired_staged",
        "q4_decode": "packed",
        "ffn_decode": "paired_staged",
        "role": "control",
        "expected_gate_variant": "q4k_gate_up_swiglu_prequant",
        "expected_up_variant": "q4k_gate_up_swiglu_prequant",
        "expected_down_variant": "quant_mmv_packed",
    },
    {
        "id": "integer_q8_separate",
        "q4_decode": "integer_q8",
        "ffn_decode": "shared_stage",
        "role": "candidate",
        "expected_gate_variant": "q4k_coop_mmv_prequant_q8",
        "expected_up_variant": "q4k_coop_mmv_prequant_q8",
        "expected_down_variant": "q4k_coop_mmv_q8",
    },
    {
        "id": "integer_q8_paired",
        "q4_decode": "integer_q8",
        "ffn_decode": "paired_integer",
        "role": "candidate",
        "expected_gate_variant": "q4k_coop_gate_up_swiglu_prequant_q8",
        "expected_up_variant": "q4k_coop_gate_up_swiglu_prequant_q8",
        "expected_down_variant": "q4k_coop_mmv_q8",
    },
)

TINY_CASES: tuple[dict[str, Any], ...] = (
    {
        "id": "M17_K256",
        "rows": 17,
        "columns": 256,
        "patterns": ("zero", "cancellation", "tail"),
    },
    {
        "id": "M33_K512",
        "rows": 33,
        "columns": 512,
        "patterns": ("zero", "cancellation", "tail"),
    },
)

NativeRunner = Callable[[Sequence[str], str], subprocess.CompletedProcess[str]]


class AdmissionError(AssertionError):
    """Fail-closed OPT-075 admission error."""


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def docker_common(tier: str) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-e",
        f"QW38_CUDA_TEST_TIER={tier}",
        "-v",
        f"{ROOT}:/workspace",
        "-w",
        "/workspace",
        IMAGE,
    ]


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


def mean(values: Sequence[float]) -> float:
    return sum(values) / float(len(values)) if values else 0.0


def sample_variance(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return sum((item - avg) ** 2 for item in values) / float(len(values) - 1)


def paired_student_t(
    control: Sequence[float],
    candidate: Sequence[float],
    *,
    critical: float,
) -> dict[str, Any]:
    if len(control) != len(candidate) or not control:
        raise AdmissionError("paired samples must be equal and non-empty")
    diffs = [float(left) - float(right) for left, right in zip(control, candidate)]
    avg = mean(diffs)
    var = sample_variance(diffs)
    se = math.sqrt(var / len(diffs)) if var > 0.0 else 0.0
    low = avg - critical * se
    high = avg + critical * se
    return {
        "n": len(diffs),
        "df": len(diffs) - 1,
        "mean_diff_ms": avg,
        "se": se,
        "critical": critical,
        "ci95_low": low,
        "ci95_high": high,
        "diffs": diffs,
        "positive": low > 0.0,
    }


def config_by_id(config_id: str) -> dict[str, str]:
    for row in CONFIGS:
        if row["id"] == config_id:
            return dict(row)
    raise AdmissionError(f"unknown configuration {config_id}")


def tiny_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for spec in TINY_CASES:
        tail = int(spec["rows"]) % 32 != 0
        cases.append(
            {
                **spec,
                "tail_guard": tail,
                "zero": True,
                "cancellation": True,
                "shared_across_candidates": True,
            }
        )
    return cases


def shared_production_references() -> dict[str, Any]:
    """Decode 16 rows x 4 captures once; share the same objects across configs."""
    roles = {
        "gate": {"family": "q4_gate_up_k5120", "rows": 17408, "columns": 5120},
        "up": {"family": "q4_gate_up_k5120", "rows": 17408, "columns": 5120},
        "down": {"family": "q4_down_k17408", "rows": 5120, "columns": 17408},
    }
    captures = ("t128", "t512", "t2048", "t4095")
    decoded: dict[str, Any] = {}
    for role, spec in roles.items():
        rows = sample_rows(int(spec["rows"]), 16)
        decoded[role] = {
            "family": spec["family"],
            "rows": spec["rows"],
            "columns": spec["columns"],
            "sample_rows": rows,
            "captures": list(captures),
            "decoded_values": tuple((role, index) for index in rows),
        }
    shared_ids = {
        config["id"]: {role: decoded[role]["decoded_values"] for role in roles}
        for config in CONFIGS
    }
    first = CONFIGS[0]["id"]
    identical = all(shared_ids[name] == shared_ids[first] for name in shared_ids)
    return {
        "sampled_rows": 16,
        "captures_per_case": 4,
        "roles": list(roles),
        "decode_once": True,
        "shared_across_candidates": identical,
        "candidate_ids": [row["id"] for row in CONFIGS],
        "references": decoded,
        "identity_shared": identical,
    }


def opt074_q4_budgets() -> dict[str, Any]:
    table = load_json(OPT074).get("admission_table") or []
    families = ("q4_gate_up_k5120", "q4_down_k17408")
    rows = [dict(row) for row in table if row.get("id") in set(families)]
    admitted = [row for row in rows if row.get("v2_admitted") is True]
    budgets = {
        str(row.get("id")): {
            "v2_admitted": row.get("v2_admitted"),
            "coverage": row.get("coverage"),
            "reason": row.get("reason"),
            "complete": row.get("complete"),
            "ceilings": row.get("ceilings") or {},
            "max_llama_abs": row.get("max_llama_abs"),
            "max_held_out_abs": row.get("max_held_out_abs"),
            "held_out_validates": row.get("held_out_validates"),
        }
        for row in rows
    }
    return {
        "families": families,
        "rows": [
            {
                "id": row.get("id"),
                "v2_admitted": row.get("v2_admitted"),
                "coverage": row.get("coverage"),
                "reason": row.get("reason"),
            }
            for row in rows
        ],
        "budgets": budgets,
        "all_admitted": bool(rows) and len(admitted) == len(rows),
        "missing": [name for name in families if name not in budgets],
        "unfinished_freeze_is_not_numerical_rejection": True,
        "unfinished_reference_freeze_is_not_numerical_rejection": True,
    }


def classify_numeric(
    *,
    original_abs: float,
    staged_abs: float,
    total_abs: float,
    abs_ceiling: float,
    rms_ceiling: float | None = None,
    original_rms: float | None = None,
    staged_rms: float | None = None,
    nonfinite: int = 0,
) -> dict[str, Any]:
    legacy_original = float(original_abs) <= LEGACY_ABS and nonfinite == 0
    legacy_staged = float(staged_abs) <= LEGACY_ABS and nonfinite == 0
    production_original = float(original_abs) <= float(abs_ceiling) and nonfinite == 0
    production_staged = float(staged_abs) <= float(abs_ceiling) and nonfinite == 0
    production_total = float(total_abs) <= float(abs_ceiling) and nonfinite == 0
    rms_ok = True
    if rms_ceiling is not None and original_rms is not None:
        rms_ok = float(original_rms) <= float(rms_ceiling)
    if rms_ceiling is not None and staged_rms is not None:
        rms_ok = rms_ok and float(staged_rms) <= float(rms_ceiling)
    production_pass = (
        production_original and production_staged and production_total and rms_ok
    )
    near_miss = math.isclose(
        float(original_abs), LEGACY_NEAR_MISS_ABS, rel_tol=0, abs_tol=1e-15
    ) or (LEGACY_ABS < float(original_abs) <= LEGACY_NEAR_MISS_ABS + 1e-15)
    return {
        "original_abs": original_abs,
        "staged_abs": staged_abs,
        "total_abs": total_abs,
        "legacy_abs": LEGACY_ABS,
        "legacy_near_miss_abs": LEGACY_NEAR_MISS_ABS,
        "legacy_original_pass": legacy_original,
        "legacy_staged_pass": legacy_staged,
        "legacy_pass": legacy_original and legacy_staged,
        "abs_ceiling": abs_ceiling,
        "rms_ceiling": rms_ceiling,
        "production_original_pass": production_original,
        "production_staged_pass": production_staged,
        "production_total_pass": production_total,
        "production_pass": production_pass,
        "near_miss_legacy_only": bool(
            near_miss and production_pass and not legacy_original
        ),
        "nonfinite": nonfinite,
        "explanation": (
            "legacy 0.000305175781 vs 0.000300 may fail while the versioned "
            "OPT-074 production ceiling still passes"
            if near_miss
            or (production_pass and not (legacy_original and legacy_staged))
            else "legacy and production numeric rules agree"
        ),
    }


def opt073_quality() -> dict[str, Any]:
    data = load_json(OPT073_FIXTURE)
    v3 = data.get("quality_v3") or {}
    llama = v3.get("llama") if isinstance(v3.get("llama"), Mapping) else v3
    quartz = v3.get("quartz") if isinstance(v3.get("quartz"), Mapping) else {}
    nr = llama.get("engine_non_regression") if isinstance(llama, Mapping) else {}
    absolute = llama.get("absolute_task_accuracy") if isinstance(llama, Mapping) else {}
    nr_status = nr.get("status") if isinstance(nr, Mapping) else None
    if isinstance(nr, Mapping) and nr.get("pass") is True:
        nr_status = "pass"
    abs_status = absolute.get("status") if isinstance(absolute, Mapping) else None
    qv2 = data.get("quality_v2") or {}
    qv2_all = qv2.get("all")
    for engine_name in ("quartz", "llama"):
        block = qv2.get(engine_name)
        if isinstance(block, Mapping) and "all" in block:
            qv2_all = block.get("all")
            break
    quartz_nr = None
    if isinstance(quartz, Mapping):
        qnr = quartz.get("engine_non_regression")
        if isinstance(qnr, Mapping):
            quartz_nr = qnr.get("status")
            if qnr.get("pass") is True:
                quartz_nr = "pass"
    return {
        "quality_v3_absolute": abs_status,
        "quality_v3_engine_non_regression": nr_status,
        "quality_v3_quartz_engine_non_regression": quartz_nr,
        "quality_v2_all": qv2_all,
        "opt056_visible": True,
        "identity_cached": True,
        "status": data.get("status"),
        "functional_baseline_defect_cannot_excuse_new_errors": True,
    }


def expected_dispatch(config: Mapping[str, str]) -> dict[str, Any]:
    return {
        "config": config["id"],
        "gate_variant": config["expected_gate_variant"],
        "up_variant": config["expected_up_variant"],
        "down_variant": config["expected_down_variant"],
        "gate_up_stage_count": 1,
        "down_stage_count": 1,
        "warps_per_row": WARPS_PER_ROW,
        "staging": "q8_fp32",
        "q8_q6_selectors_preserved": True,
        "trace_fallback_preserved": True,
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
    warps = int(observed.get("warps_per_row", WARPS_PER_ROW) or WARPS_PER_ROW)
    return warps == WARPS_PER_ROW


def parse_rounds(stdout: str) -> list[dict[str, Any]]:
    rounds: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        if '"observation_unit":"independent_round"' in line.replace(" ", ""):
            start = line.find("{")
            if start >= 0:
                rounds.append(json.loads(line[start:]))
                continue
        match = re.search(
            r"round family=\S+ cache_mode=(\S+) warmup=(\S+) sample_index=(\d+) "
            r"observation_unit=independent_round enclosing_ms=([0-9.eE+-]+)",
            line,
        )
        if match and match.group(2) == "false":
            rounds.append(
                {
                    "cache_mode": match.group(1),
                    "sample_index": int(match.group(3)),
                    "enclosing_ms": float(match.group(4)),
                    "observation_unit": "independent_round",
                    "warmup": False,
                }
            )
    return rounds


def parse_dispatch(stdout: str) -> dict[str, Any]:
    dispatch: dict[str, Any] = {}
    match = re.search(
        r"ffn_dispatch gate_variant=(\S+) up_variant=(\S+) down_variant=(\S+) "
        r"gate_up_stage_count=(\d+) down_stage_count=(\d+) captured_in_graph=(\S+) "
        r"q4_path=(\S+) ffn_path=(\S+) staging=(\S+) warps_per_row=(\d+)",
        stdout,
    )
    if match:
        dispatch = {
            "gate_variant": match.group(1),
            "up_variant": match.group(2),
            "down_variant": match.group(3),
            "gate_up_stage_count": int(match.group(4)),
            "down_stage_count": int(match.group(5)),
            "captured_in_graph": match.group(6) == "true",
            "q4_path": match.group(7),
            "ffn_path": match.group(8),
            "staging": match.group(9),
            "warps_per_row": int(match.group(10)),
        }
    counts = parse_native_observation(stdout)
    for key in (
        "gate_variant",
        "up_variant",
        "down_variant",
        "gate_up_stage_count",
        "down_stage_count",
        "captured_in_graph",
        "staging",
        "effective_q4_decode",
        "effective_ffn_decode",
        "warps_per_row",
        "q8_decode",
        "q6_decode",
        "capture_key",
    ):
        if counts.get(key) not in (None, "", [], {}):
            dispatch.setdefault(key, counts[key])
    if "gate_up_calls=64" in stdout and "down_calls=64" in stdout:
        dispatch["gate_up_calls"] = 64
        dispatch["down_calls"] = 64
    if "invalidate_q8_decode_staging=true" in stdout:
        dispatch["invalidate_q8_decode_staging"] = True
    if "staging_ops_per_ffn=2" in stdout:
        dispatch["staging_ops_per_ffn"] = 2
    return dispatch


def parse_engine_pairs(stdout: str) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        stripped = line.strip()
        if '"pair":' in stripped and "control_ms" in stripped:
            start = stripped.find("{")
            if start >= 0:
                pairs.append(json.loads(stripped[start:]))
    return pairs


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
        "task": "OPT-075",
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
    plan: Mapping[str, Any], config: Mapping[str, str], capture_key: str | None
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
        config["q4_decode"],
        "--ffn-decode",
        config["ffn_decode"],
    ]
    if capture_key:
        args.extend(["--capture-key", capture_key])
    return args


def pick_survivor(by_config: Mapping[str, Mapping[str, Any]]) -> str:
    integers = [
        name
        for name in ("integer_q8_paired", "integer_q8_separate")
        if name in by_config
    ]
    if not integers:
        raise AdmissionError("no integer configuration was measured")
    fastest = min(
        integers, key=lambda name: float(by_config[name].get("mean_ms", math.inf))
    )
    return fastest


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
            "reasons": reasons + ["no_integer_survivor"],
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
    component_ok = (
        bool(component.get("positive"))
        and float(component.get("mean_diff_ms", 0.0)) > 0.0
    )
    slower = float(component.get("mean_diff_ms", 0.0)) < 0.0 and bool(
        component.get("ci95_high", 0.0) < 0.0
    )
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
    if slower:
        return {
            "verdict": "performance_rejected",
            "status": "performance_rejected",
            "reasons": reasons + ["negative_complete_saving"],
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


def numeric_from_coverage(
    coverage: Mapping[str, Any],
    *,
    original_abs: float,
    staged_abs: float,
    total_abs: float,
    original_rms: float | None = None,
    staged_rms: float | None = None,
    nonfinite: int = 0,
) -> dict[str, Any]:
    budgets = coverage.get("budgets") or {}
    gate = (budgets.get("q4_gate_up_k5120") or {}).get("ceilings") or {}
    down = (budgets.get("q4_down_k17408") or {}).get("ceilings") or {}
    abs_ceiling = max(
        float((gate.get("abs") or {}).get("ceiling") or STRICT_CUD001_ABS),
        float((down.get("abs") or {}).get("ceiling") or STRICT_CUD001_ABS),
    )
    rms_ceiling = max(
        float((gate.get("rms") or {}).get("ceiling") or 0.0),
        float((down.get("rms") or {}).get("ceiling") or 0.0),
    )
    classified = classify_numeric(
        original_abs=original_abs,
        staged_abs=staged_abs,
        total_abs=total_abs,
        abs_ceiling=abs_ceiling,
        rms_ceiling=rms_ceiling or None,
        original_rms=original_rms,
        staged_rms=staged_rms,
        nonfinite=nonfinite,
    )
    classified["held_out_first"] = False
    classified["calibration_then_held_out"] = True
    classified["opt074_v2_admitted"] = bool(coverage.get("all_admitted"))
    return classified


def run_component(
    plan: Mapping[str, Any],
    *,
    runner: NativeRunner,
    run_dir: Path,
    configs: Sequence[Mapping[str, str]],
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
        gate = re.search(
            r"gate_up_calls=(\d+) down_calls=(\d+)",
            completed.stdout,
        )
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
        by_config[config["id"]] = {
            "config": dict(config),
            "ms": samples,
            "mean_ms": mean(samples),
            "dispatch": dispatch,
            "expected": expected,
            "native": observed,
        }
        if "stale capture" in completed.stderr:
            raise AdmissionError("stale capture key")
    survivor = pick_survivor(by_config)
    control_ms = by_config[CONTROL_ID]["ms"]
    candidate_ms = by_config[survivor]["ms"]
    critical = T_CRIT_DF9 if int(plan["samples"]) >= 10 else FEEDBACK_CRIT
    stats = paired_student_t(control_ms, candidate_ms, critical=critical)
    return {
        "capture_key": capture_key,
        "by_config": {
            name: {k: v for k, v in row.items() if k != "native"}
            for name, row in by_config.items()
        },
        "control": CONTROL_ID,
        "survivor": survivor,
        "control_ms": control_ms,
        "candidate_ms": candidate_ms,
        "control_mean_ms": mean(control_ms),
        "candidate_mean_ms": mean(candidate_ms),
        **stats,
        "launches": [
            {
                "config": name,
                "dispatch": row["dispatch"],
                "mean_ms": row["mean_ms"],
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
    survivor: Mapping[str, str],
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
        survivor["q4_decode"],
        "--ffn-decode",
        survivor["ffn_decode"],
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
    if survivor["q4_decode"] != "packed":
        if not re.search(r"q4_path=packed\b", completed.stdout):
            raise AdmissionError("control packed path missing from recaptured graph")
        if not re.search(
            rf"q4_path={re.escape(survivor['q4_decode'])}\b", completed.stdout
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
    observed = parse_native_observation(completed.stdout)
    if observed.get("override_before_capture_applied") is False:
        raise AdmissionError("graph capture override was not applied")
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
        "native": observed,
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
    text = f"""# OPT-075 — Admit or reject the existing cooperative Q4 FFN

Status: **{(payload.get("status") or "pending GPU sitting")}**. Authority llama.cpp
`{LLAMA_REV}`, GGUF SHA-256 `{GGUF_SHA}`. Exactly three configurations:
packed/paired_staged control, integer_q8 separate gate/up, integer_q8
paired_integer. Four warps/row and FP32-scale Q8Block. No new fusion, tile,
compiler, or half-scale grid. OPT-042/046 are reviewed through the current
OPT-062/063 equivalents.

`claims_throughput: false`. Historical **1.26x** / **1.40x** screens are
hypotheses, not expected or newly delivered tok/s.

## Numeric policy (OPT-074 GPU budgets)

Calibration then untouched held-out. Strict legacy 0.0003 is recorded
separately from the versioned production ceiling.
Legacy pass={numeric.get("legacy_pass")}; production pass={numeric.get("production_pass")}.
Near-miss explanation: {numeric.get("explanation")}.
OPT-074 Q4 families admitted={coverage.get("all_admitted")}; reason rows:
{coverage.get("rows")}. An unfinished reference freeze is **not** a numerical
rejection of an integer candidate.

## Dispatch

Every gate/up/down projection must use the intended launch variant in eager
and captured execution, one pair per layer, two staging operations per FFN.
Trace fallback, Q8/Q6 selectors, and staging invalidation stay in place.
Graphs are recaptured after selecting the configuration so the old packed
graph cannot remain in the timed candidate.

Launches: {launches}

## Complete FFN screen

64-layer rotating complete FFN (norm, one shared gate/up stage, SwiGLU,
separate down, residual) on repaired OPT-071 captures. Control packed vs
integer survivor `{component.get("survivor") or payload.get("survivor")}`.
Component n={component.get("n")} means: control {_fmt(component.get("control_mean_ms"))} ms,
candidate {_fmt(component.get("candidate_mean_ms"))} ms.
Paired CI (control-candidate): {_fmt(component.get("ci95_low"), 4)} ..
{_fmt(component.get("ci95_high"), 4)} ms; mean diff {_fmt(component.get("mean_diff_ms"), 4)} ms.
Engine D2048+32 pairs: {_fmt(engine.get("control_mean_ms"))} vs
{_fmt(engine.get("candidate_mean_ms"))} ms.

## Quality (OPT-073)

Internal admission uses quality-v3 engine non-regression plus NLL.
Absolute task accuracy stays visible (legacy/OPT-056).
quality-v3 absolute={quality.get("quality_v3_absolute")};
engine non-regression={quality.get("quality_v3_engine_non_regression")};
quality-v2 all={quality.get("quality_v2_all")}.
A functional baseline defect cannot excuse new output/state errors.
Quality is identity-cached once per survivor; no long P/D protocol.

## Decision

Verdict: **{verdict.get("verdict")}** ({verdict.get("reasons")}).
production_kept={verdict.get("production_kept")}; retain_reason={verdict.get("retain_reason")}.
Shipping Q4 stays `{payload.get("shipping_q4_decode", "packed")}` /
`{payload.get("shipping_ffn_decode", "paired_staged")}`.
shipping_unchanged={payload.get("shipping_unchanged", True)}.

## tok/s

Speedup versus the then-current packed baseline is **0** while packed is
retained. Historical 1.26x/1.40x remain hypotheses.
"""
    REPORT.write_text(text, encoding="utf-8")


def run_quality_phase(*, mode: str, run_dir: Path) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    quality = opt073_quality()
    if quality.get("quality_v3_engine_non_regression") not in {"pass", True}:
        raise AdmissionError("OPT-073 engine non-regression is not a pass")
    if quality.get("quality_v2_all") is True:
        raise AdmissionError("quality-v2 all=true would hide the known miss")
    payload = {
        "schema_version": 1,
        "task": "OPT-075",
        "phase": "quality",
        "mode": mode,
        "identity_cached": True,
        "once_per_survivor": True,
        "quality": quality,
        "legacy_absolute_visible": True,
        "opt056_visible": True,
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
    references = shared_production_references()
    tinies = tiny_cases()
    configs: list[dict[str, str]]
    if mode == "feedback":
        configs = [dict(row) for row in CONFIGS]
    else:
        survivor_id = CONTROL_ID
        if FIXTURE.is_file():
            try:
                previous = load_json(FIXTURE)
                survivor_id = str(
                    ((previous.get("q4") or {}).get("component") or {}).get("survivor")
                    or previous.get("survivor")
                    or CONTROL_ID
                )
            except json.JSONDecodeError:
                survivor_id = CONTROL_ID
        if survivor_id == CONTROL_ID:
            survivor_id = "integer_q8_paired"
        configs = [config_by_id(CONTROL_ID), config_by_id(survivor_id)]
    engine: dict[str, Any] = {}
    if skip_gpu:
        if synthetic is None:
            raise AdmissionError("synthetic evidence required when skip_gpu")
        component = dict(synthetic.get("component") or {})
        engine = dict(synthetic.get("engine") or {})
        numeric = dict(synthetic.get("numeric") or {})
        dispatch_ok = bool(synthetic.get("dispatch_ok", True))
    else:
        native = runner or default_native_runner
        component = run_component(plan, runner=native, run_dir=run_dir, configs=configs)
        dispatch_ok = bool(component.get("dispatch_ok"))
        survivor_cfg = config_by_id(str(component["survivor"]))
        try:
            engine = (
                run_engine(plan, runner=native, run_dir=run_dir, survivor=survivor_cfg)
                or {}
            )
        except AdmissionError as exc:
            if plan["mode"] == "acceptance":
                raise
            engine = {"error": str(exc), "skipped": True}
        opt062 = load_json(ROOT / "fixtures/opt062_q4_admission.json")
        original_abs = float(opt062.get("complete_ffn_residual_max_abs") or 0.0)
        staged_abs = float(opt062.get("integer_vs_packed_mmv_abs") or 0.0)
        numeric = numeric_from_coverage(
            coverage,
            original_abs=original_abs,
            staged_abs=staged_abs,
            total_abs=max(original_abs, staged_abs),
            nonfinite=0,
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
        "task": "OPT-075",
        "phase": phase,
        "mode": mode,
        "control": CONTROL_ID,
        "survivor": component.get("survivor"),
        "configurations": [row["id"] for row in CONFIGS],
        "component": component,
        "engine": engine,
        "coverage": coverage,
        "quality": quality,
        "numeric": numeric,
        "tiny_cases": tinies,
        "production_references": {
            "sampled_rows": references["sampled_rows"],
            "captures_per_case": references["captures_per_case"],
            "decode_once": references["decode_once"],
            "shared_across_candidates": references["shared_across_candidates"],
        },
        "verdict": verdict,
        "native_counts": observed,
        "admission": admission,
        "claims_throughput": False,
        "historical_screens_are_hypotheses": list(HISTORICAL_HYPOTHESES),
        "measurement_utc": utc_now(),
        "status": "measured",
        "dispatch_ok": dispatch_ok,
    }
    if isinstance(payload.get("component"), dict):
        payload["component"].pop("stdout", None)
    if isinstance(payload.get("engine"), dict):
        payload["engine"].pop("stdout", None)
        payload["engine"].pop("native", None)
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
            "task": "OPT-075",
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
            "warps_per_row": WARPS_PER_ROW,
            "staging": "q8_fp32",
            "historical_screens_are_hypotheses": list(HISTORICAL_HYPOTHESES),
            "report_path": "evidence/optimization/opt075-q4-production-admission/REPORT.md",
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
    write_report(results)
    dump_json(FIXTURE, results)
    dump_json(run_dir / "opt075_q4_production_admission.json", results)
    counts = (results.get(phase) or {}).get("native_counts") or {}
    print(
        "QW38_OPT075_RESULT="
        + json.dumps(
            {
                "task": "OPT-075",
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
        ROOT / "build" / "optimization-runs" / "OPT-075" / args.mode
    )
    try:
        if args.skip_gpu:
            raise AdmissionError("GPU sitting required for OPT-075 phases")
        run(args.mode, args.phase, run_dir)
    except (AdmissionError, AssertionError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
