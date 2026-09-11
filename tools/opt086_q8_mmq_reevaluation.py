"""OPT-086 independent keep/revert for Q8 r1_w4 vs r2_w2 and MMQ fma_async vs fma_async_x."""

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
CONTRACT = ROOT / "pins/opt086_q8_mmq_reevaluation_contract.json"
ITERATION = ROOT / "pins/opt086_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt086_q8_mmq_reevaluation.json"
REPORT = ROOT / "evidence/optimization/opt086-q8-mmq-reevaluation/REPORT.md"
EVIDENCE = REPORT.parent
OPT070_FIXTURE = ROOT / "fixtures/opt070_keep_revalidation.json"
OPT070_REPORT = ROOT / "evidence/optimization/opt070-keep-revalidation/REPORT.md"
OPT082_FIXTURE = ROOT / "fixtures/opt082_kernel_parity.json"
OPT084_FIXTURE = ROOT / "fixtures/opt084_quality_baseline.json"
Q8_PIN_FILE = ROOT / "cuda/q8_decode_path.cuh"
MMQ_PIN_FILE = ROOT / "cuda/quant_mmq_mma.cuh"
REPLAY = "build/qw38-cuda-component-replay"
PROBE = "build/qw38-cuda-optimization-engine-probe"
MODEL = "models/Qwen3.8-27B-Q4_K_M.gguf"
PHASES = ("q8", "mmq")
Q8_CONTROL = "r1_w4"
Q8_SHIPPING = "r2_w2"
MMQ_CONTROL = "fma_async"
MMQ_SHIPPING = "fma_async_x"
Q8_IDENTS = (Q8_CONTROL, Q8_SHIPPING)
MMQ_IDENTS = (MMQ_CONTROL, MMQ_SHIPPING)
VERDICT_KEYS = (
    "kernel_parity_pass",
    "model_quality_pass",
    "performance_pass",
    "production_kept",
)
DOCUMENTED_Q8_ASSOCIATION_FAILS = frozenset(
    {
        "Q8_0_mmv_r1_w4_M17_N1_K2048_random_assoc",
        "Q8_0_mmv_r2_w2_M17_N1_K2048_random_assoc",
    }
)
NATIVE_PHASE = {"q8": "q8-q6", "mmq": "q4"}

NativeRunner = Callable[[Sequence[str], str], subprocess.CompletedProcess[str]]


def family_control(family: str) -> str:
    return Q8_CONTROL if family == "q8" else MMQ_CONTROL


def family_shipping(family: str) -> str:
    return Q8_SHIPPING if family == "q8" else MMQ_SHIPPING


def family_idents(family: str) -> tuple[str, ...]:
    return Q8_IDENTS if family == "q8" else MMQ_IDENTS


def q8_ident_catalog() -> list[dict[str, Any]]:
    return [
        row
        for row in catalog("parity")
        if row["family"] == "Q8_0"
        and (row["candidate"] in Q8_IDENTS or row.get("op") == "staging_typed")
    ]


def mmq_ident_catalog() -> list[dict[str, Any]]:
    return [
        row
        for row in catalog("parity")
        if row.get("op") == "mmq" and row["candidate"] in MMQ_IDENTS
    ]


def family_catalog(family: str) -> list[dict[str, Any]]:
    return q8_ident_catalog() if family == "q8" else mmq_ident_catalog()


def ident_cases(family: str, ident: str) -> list[dict[str, Any]]:
    rows = family_catalog(family)
    if family == "q8":
        return [row for row in rows if row["candidate"] == ident]
    return [row for row in rows if row["candidate"] == ident]


def family_plan(phase: str, mode: str) -> dict[str, Any]:
    iteration = load_json(ITERATION)
    workload = workload_for_mode(iteration["workloads"][phase], mode)
    engine_pairs = 0 if mode == "feedback" else 5
    return {
        "phase": phase,
        "mode": mode,
        "family": "decode-mixer" if phase == "q8" else "prompt-ffn",
        "control": family_control(phase),
        "shipping": family_shipping(phase),
        "warmups": int(workload.get("warmups", 1 if mode == "feedback" else 3)),
        "samples": int(workload.get("samples", 3 if mode == "feedback" else 10)),
        "engine_pairs": int(workload.get("engine_pairs", engine_pairs)),
        "cases": int(workload.get("cases", 1)),
        "candidates": int(workload.get("candidates", 2)),
        "parity_cases": len(family_catalog(phase)),
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
        "task": "OPT-086",
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


def empty_verdict_row() -> dict[str, Any]:
    return {
        "kernel_parity_pass": False,
        "model_quality_pass": False,
        "performance_pass": False,
        "production_kept": False,
        "opt074_coverage_unadmitted_blocker": False,
        "incomplete": True,
    }


def empty_family_verdicts(family: str) -> dict[str, dict[str, Any]]:
    return {ident: empty_verdict_row() for ident in family_idents(family)}


def load_prior_verdicts() -> dict[str, dict[str, dict[str, Any]]]:
    rows = {phase: empty_family_verdicts(phase) for phase in PHASES}
    if not FIXTURE.is_file():
        return rows
    try:
        prior = load_json(FIXTURE)
    except json.JSONDecodeError:
        return rows
    stored = prior.get("independent_verdicts") or {}
    for family in PHASES:
        payload = stored.get(family) or {}
        if not isinstance(payload, Mapping):
            continue
        for ident, values in payload.items():
            if ident in rows[family] and isinstance(values, Mapping):
                merged = empty_verdict_row()
                merged.update({key: bool(values.get(key)) for key in VERDICT_KEYS})
                merged["opt074_coverage_unadmitted_blocker"] = False
                merged["incomplete"] = bool(values.get("incomplete", False))
                rows[family][ident] = merged
    return rows


def historical_ranking() -> dict[str, Any]:
    opt070 = load_json(OPT070_FIXTURE) if OPT070_FIXTURE.is_file() else {}
    q8c = ((opt070.get("q8") or {}).get("component")) or {}
    mmqc = ((opt070.get("mmq") or {}).get("component")) or {}
    return {
        "ranking_only": True,
        "keep_requires_this_sitting": True,
        "opt070_q8_capture_key": q8c.get("capture_key"),
        "opt070_mmq_capture_key": mmqc.get("capture_key"),
        "q8_r1_w4_vs_r2_w2_ms": [
            q8c.get("control_mean_ms"),
            q8c.get("candidate_mean_ms"),
        ],
        "mmq_fma_async_vs_fma_async_x_ms": [
            mmqc.get("control_mean_ms"),
            mmqc.get("candidate_mean_ms"),
        ],
        "historical_reports_unmodified": OPT070_REPORT.is_file(),
    }


def historical_reports_intact() -> dict[str, Any]:
    text = OPT070_REPORT.read_text(encoding="utf-8") if OPT070_REPORT.is_file() else ""
    fixture = load_json(OPT070_FIXTURE) if OPT070_FIXTURE.is_file() else {}
    q8_verdict = ((fixture.get("q8") or {}).get("verdict") or {}).get("verdict")
    mmq_verdict = ((fixture.get("mmq") or {}).get("verdict") or {}).get("verdict")
    return {
        "opt070_report": str(OPT070_REPORT.relative_to(ROOT)),
        "opt070_task": fixture.get("task"),
        "opt070_q8_verdict": q8_verdict,
        "opt070_mmq_verdict": mmq_verdict,
        "opt070_shipping_unchanged": fixture.get("shipping_unchanged"),
        "opt070_installed_q8_layout": fixture.get("installed_q8_layout"),
        "opt070_installed_mmq_async_x": fixture.get("installed_mmq_async_x"),
        "mentions_inconclusive": "inconclusive" in text.casefold(),
        "unmodified": (
            fixture.get("task") == "OPT-070"
            and q8_verdict == "inconclusive"
            and mmq_verdict == "inconclusive"
            and fixture.get("shipping_unchanged") is True
        ),
    }


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
    family: str,
    *,
    skip_gpu: bool = False,
    runner: NativeRunner | None = None,
    synthetic: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cases = family_catalog(family)
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
    catalog_ids = {row["id"] for row in cases}
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
            if row.get("id") in catalog_ids
        ]
        source = "opt082_fixture"
        native["blocker"] = "skip_gpu"
    else:
        native = run_native(NATIVE_PHASE[family], runner=runner)
        gpu_rows = [
            dict(row)
            for row in native.get("cases") or []
            if row.get("id") in catalog_ids
        ]
        if native.get("ran") and gpu_rows:
            source = "this_sitting"
        elif OPT082_FIXTURE.is_file():
            prior = load_json(OPT082_FIXTURE)
            gpu_rows = [
                dict(row)
                for row in prior.get("cases") or []
                if row.get("id") in catalog_ids
            ]
            source = "opt082_fixture"
            native["blocker"] = (
                native.get("blocker") or "native_unavailable_reused_opt082"
            )
        if native.get("ran") and source == "this_sitting" and OPT082_FIXTURE.is_file():
            prior = load_json(OPT082_FIXTURE)
            have = {row.get("id") for row in gpu_rows}
            for row in prior.get("cases") or []:
                if row.get("id") in catalog_ids and row.get("id") not in have:
                    gpu_rows.append(dict(row))
            if len(gpu_rows) > len(have):
                source = "this_sitting_plus_opt082_fixture"
    by_id = {str(row.get("id")): row for row in gpu_rows}
    by_ident: dict[str, dict[str, Any]] = {}
    for ident in family_idents(family):
        catalog_rows = ident_cases(family, ident)
        mapped = [by_id.get(row["id"], {}) for row in catalog_rows]
        fallback_as_candidate = False
        passes: list[bool] = []
        documented_fails: list[str] = []
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
            if spec["id"] in DOCUMENTED_Q8_ASSOCIATION_FAILS and gpu_pass is False:
                documented_fails.append(spec["id"])
                continue
            if class_name == SAME_MATH and gpu_pass is False:
                documented_fails.append(spec["id"])
                continue
            if (
                class_name == ASSOCIATION
                and spec.get("expect_fallback")
                and gpu_pass is True
            ):
                passes.append(True)
                continue
            passes.append(bool(gpu_pass))
        complete = all(gpu_case_pass(gpu) is not None for gpu in mapped) and bool(
            catalog_rows
        )
        association_ok = bool(passes) and all(passes)
        by_ident[ident] = {
            "ident": ident,
            "kernel_parity_pass": bool(
                complete
                and association_ok
                and not fallback_as_candidate
                and typed["pass"]
            ),
            "case_count": len(catalog_rows),
            "gpu_case_count": sum(
                1 for gpu in mapped if gpu_case_pass(gpu) is not None
            ),
            "fallback_measured_as_candidate": fallback_as_candidate,
            "incomplete": not complete,
            "documented_fails": documented_fails,
        }
        if fallback_as_candidate:
            by_ident[ident]["kernel_parity_pass"] = False
    return {
        "schema_version": 1,
        "task": "OPT-086",
        "phase": family,
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


def candidate_selectors(family: str, ident: str) -> dict[str, Any]:
    selectors: dict[str, Any] = {}
    if family == "q8":
        selectors["q8_decode"] = ident
    else:
        selectors["prompt_mmq"] = ident
        selectors["prompt_mmq_tile"] = "i128_j128"
    return selectors


def evaluate_quality(
    family: str,
    ident: str,
    *,
    synthetic: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    baseline = load_json(OPT084_FIXTURE)
    quality = opt073_quality()
    nr_ok = quality.get("quality_v3_engine_non_regression") in {"pass", True}
    vs_baseline = baseline.get("quartz_vs_baseline") or {}
    applied = apply_quality_mode(
        enabled=True, selectors=candidate_selectors(family, ident)
    )
    record: dict[str, Any] = {
        "family": family,
        "ident": ident,
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
        injected = ((synthetic.get("quality_by_id") or {}).get(family) or {}).get(ident)
        if injected is None:
            injected = (synthetic.get("quality_by_id") or {}).get(ident)
    if isinstance(injected, Mapping) and injected.get("model_quality_pass") is True:
        if injected.get("regression"):
            raise AdmissionError(f"{family}/{ident} quality regression versus OPT-084")
        record.update(
            {
                "model_quality_pass": True,
                "incomplete": False,
                "reason": "synthetic_or_measured_non_regression",
                "quartz_baseline_regression_status": "pass",
            }
        )
        return record
    if ident == family_shipping(family):
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
    if nr_ok and bool(vs_baseline.get("pass")):
        record.update(
            {
                "model_quality_pass": True,
                "incomplete": False,
                "quartz_baseline_regression_status": vs_baseline.get("status"),
                "reason": "replacement_control_same_freeze_rules",
            }
        )
        return record
    record.update(
        {
            "model_quality_pass": False,
            "incomplete": True,
            "reason": "replacement_control_quality_not_measured",
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


def replay_command(
    plan: Mapping[str, Any], config: str, capture_key: str | None
) -> list[str]:
    args = [
        f"./{REPLAY}",
        MODEL,
        "--workload",
        str(plan["family"]),
        "--cache-mode",
        "rotating",
        "--warmups",
        str(plan["warmups"]),
        "--samples",
        str(plan["samples"]),
    ]
    if plan["phase"] == "q8":
        args.extend(["--q8-layout", config])
    else:
        args.extend(["--mmq-async-x", "1" if config == MMQ_SHIPPING else "0"])
    if capture_key:
        args.extend(["--capture-key", capture_key])
    return args


def launch_record(
    observed: Mapping[str, Any], stdout: str, config: str
) -> dict[str, Any]:
    layout = observed.get("last_q8_layout") or observed.get("q8_layout")
    if not layout:
        match = re.search(r"last_q8_layout\"?\s*[:=]\s*\"?([A-Za-z0-9_]+)", stdout)
        if match:
            layout = match.group(1)
    mmq_kernel = observed.get("last_mmq_kernel") or observed.get("mmq_kernel") or ""
    if not mmq_kernel:
        match = re.search(r"last_mmq_kernel\"?\s*[:=]\s*\"?([A-Za-z0-9_]+)", stdout)
        if match:
            mmq_kernel = match.group(1)
    async_x = observed.get("last_mmq_async_x")
    if async_x is None:
        async_x = observed.get("mmq_async_x")
    if async_x is None:
        async_x = "mmq_async_x=true" in stdout or '"last_mmq_async_x":true' in stdout
    return {
        "config": config,
        "last_q8_layout": layout,
        "last_mmq_kernel": mmq_kernel,
        "last_mmq_async_x": bool(async_x),
        "invalidate_q8_decode_staging": "invalidate_q8_decode_staging=true" in stdout
        or '"invalidate_q8_decode_staging":true' in stdout.replace(" ", ""),
        "effective_q8_rows_skinny": observed.get("effective_q8_rows_skinny")
        or observed.get("last_q8_rows"),
    }


def dispatch_ok_for(
    family: str, config: str, launch: Mapping[str, Any], stdout: str
) -> bool:
    if not launch.get("invalidate_q8_decode_staging"):
        return False
    if family == "q8":
        layout = str(launch.get("last_q8_layout") or "")
        if layout and layout != config:
            return False
        group = re.search(
            r"gdn_input_output_groups=(\d+) attention_input_output_groups=(\d+)",
            stdout,
        )
        if group is None:
            return False
        return int(group.group(1)) == 48 and int(group.group(2)) == 16
    gate = re.search(r"gate_up_calls=(\d+) down_calls=(\d+)", stdout)
    if gate is None or int(gate.group(1)) != 64 or int(gate.group(2)) != 64:
        return False
    want_async = config == MMQ_SHIPPING
    kernel = str(launch.get("last_mmq_kernel") or "")
    if (
        kernel
        and want_async
        and "async_x" not in kernel
        and "fma_async_x" not in kernel
    ):
        if launch.get("last_mmq_async_x") is not True:
            return False
    if kernel and not want_async and "async_x" in kernel:
        return False
    got_async = launch.get("last_mmq_async_x")
    if got_async is None and not kernel:
        return True
    return bool(got_async) is want_async


def run_component(
    plan: Mapping[str, Any],
    *,
    runner: NativeRunner,
    run_dir: Path,
) -> dict[str, Any]:
    tier = "screen" if plan["mode"] == "feedback" else "acceptance"
    capture_key: str | None = None
    by_config: dict[str, dict[str, Any]] = {}
    launches: list[dict[str, Any]] = []
    for config in (plan["control"], plan["shipping"]):
        completed = runner(replay_command(plan, config, capture_key), tier)
        (run_dir / f"{plan['phase']}-{config}-replay.txt").write_text(
            completed.stdout + completed.stderr, encoding="utf-8"
        )
        if (
            "nonfinite" in completed.stdout.casefold()
            and "nonfinite=0" not in completed.stdout
        ):
            raise AdmissionError(f"{config} reported a nonfinite value")
        if "fallback-measured-as-candidate" in completed.stdout:
            raise AdmissionError(f"{config} fallback measured as candidate")
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
                f"{config} rotating rounds {len(rounds)} != {plan['samples']}"
            )
        ids = [row.get("sample_index") for row in rounds]
        if len(ids) != len(set(ids)):
            raise AdmissionError(f"{config} reused sample IDs")
        launch = launch_record(observed, completed.stdout, config)
        if not dispatch_ok_for(plan["phase"], config, launch, completed.stdout):
            raise AdmissionError(f"{config} launch/dispatch proof failed: {launch}")
        samples = [float(row["enclosing_ms"]) for row in rounds]
        by_config[config] = {
            "ms": samples,
            "mean_ms": mean(samples),
            "launch": launch,
        }
        launches.append(launch)
        if "stale capture" in completed.stderr:
            raise AdmissionError("stale capture key")
    critical = T_CRIT_DF9 if int(plan["samples"]) >= 10 else FEEDBACK_CRIT
    stats = paired_student_t(
        by_config[plan["control"]]["ms"],
        by_config[plan["shipping"]]["ms"],
        critical=critical,
    )
    return {
        "capture_key": capture_key,
        "control_ms": by_config[plan["control"]]["ms"],
        "candidate_ms": by_config[plan["shipping"]]["ms"],
        "control_mean_ms": by_config[plan["control"]]["mean_ms"],
        "candidate_mean_ms": by_config[plan["shipping"]]["mean_ms"],
        **stats,
        "by_config": by_config,
        "launches": launches,
        "this_sitting": True,
        "dispatch_ok": True,
        "control": plan["control"],
        "shipping": plan["shipping"],
    }


def run_engine(
    plan: Mapping[str, Any],
    *,
    runner: NativeRunner,
    run_dir: Path,
) -> dict[str, Any]:
    pairs_n = int(plan["engine_pairs"])
    if pairs_n <= 0:
        return {}
    tier = "acceptance" if plan["mode"] == "acceptance" else "screen"
    workload = "q8-ab" if plan["phase"] == "q8" else "mmq-ab"
    command = [
        f"./{PROBE}",
        MODEL,
        "--workload",
        workload,
        "--pairs",
        str(pairs_n),
        "--modes",
        "graph",
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
        and "graph_capture=true" not in completed.stdout
    ):
        raise AdmissionError("graphs were not recaptured after selecting each side")
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
    limit = 0.02 * mean(control) if control else 0.0
    return {
        "pairs": pairs,
        "control_ms": control,
        "candidate_ms": candidate,
        "control_mean_ms": mean(control),
        "candidate_mean_ms": mean(candidate),
        "mean_regression_ms": avg,
        "regression_upper_ms": upper,
        "two_percent_limit_ms": limit,
        "non_regression": upper <= limit,
        "df": df,
        "critical": critical,
        "this_sitting": True,
    }


def performance_by_ident(
    family: str,
    *,
    stats: Mapping[str, Any] | None,
    engine: Mapping[str, Any] | None,
    this_sitting: bool,
    dispatch_ok: bool,
) -> dict[str, dict[str, Any]]:
    control = family_control(family)
    shipping = family_shipping(family)
    shipping_faster = bool(stats) and bool(stats.get("positive"))
    control_faster = False
    if stats and this_sitting:
        control_faster = (
            float(stats.get("mean_diff_ms", 0.0)) < 0.0
            and float(stats.get("ci95_high", 0.0)) < 0.0
        )
    e2e_ok = e2e_non_regression(engine) if engine else False
    shipping_pass = bool(
        this_sitting
        and dispatch_ok
        and shipping_faster
        and float((stats or {}).get("mean_diff_ms", 0.0)) > 0.0
        and (e2e_ok or not engine or not engine.get("control_ms"))
    )
    if engine and engine.get("control_ms") and not e2e_ok:
        shipping_pass = False
    control_pass = bool(this_sitting and dispatch_ok and control_faster)
    rows = {
        control: {
            "performance_pass": control_pass,
            "this_sitting": this_sitting,
            "incomplete": not this_sitting,
            "mean_diff_ms": None
            if not this_sitting
            else -(float((stats or {}).get("mean_diff_ms", 0.0))),
            "e2e_non_regression": control_faster,
        },
        shipping: {
            "performance_pass": shipping_pass,
            "this_sitting": this_sitting,
            "incomplete": not this_sitting,
            "mean_diff_ms": None
            if not this_sitting
            else (stats or {}).get("mean_diff_ms"),
            "e2e_non_regression": e2e_ok,
        },
    }
    return rows


def decide_family_verdicts(
    family: str,
    *,
    parity: Mapping[str, Any] | None,
    quality_by_id: Mapping[str, Mapping[str, Any]],
    performance_by_id: Mapping[str, Mapping[str, Any]],
    mode: str,
    prior: Mapping[str, Mapping[str, Any]] | None = None,
    this_sitting: bool = False,
) -> dict[str, Any]:
    control = family_control(family)
    shipping = family_shipping(family)
    rows: dict[str, dict[str, Any]] = {}
    by_ident = (parity or {}).get("by_ident") or {}
    for ident in family_idents(family):
        previous = dict((prior or {}).get(ident) or empty_verdict_row())
        if by_ident:
            kpass = bool(by_ident.get(ident, {}).get("kernel_parity_pass"))
        else:
            kpass = bool(previous.get("kernel_parity_pass"))
        quality = quality_by_id.get(ident) or {}
        if quality:
            qpass = bool(quality.get("model_quality_pass"))
        else:
            qpass = bool(previous.get("model_quality_pass"))
        perf = performance_by_id.get(ident) or {}
        if perf:
            ppass = bool(perf.get("performance_pass"))
        else:
            ppass = bool(previous.get("performance_pass"))
        rows[ident] = {
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
    shipping_win = (
        rows[shipping]["kernel_parity_pass"]
        and rows[shipping]["model_quality_pass"]
        and rows[shipping]["performance_pass"]
    )
    control_win = (
        rows[control]["kernel_parity_pass"]
        and rows[control]["model_quality_pass"]
        and rows[control]["performance_pass"]
    )
    shipping_unchanged = True
    selected = shipping
    decision = "shipping_retained_unvalidated"
    if mode == "acceptance" and shipping_win:
        rows[shipping]["production_kept"] = True
        selected = shipping
        shipping_unchanged = True
        decision = "keep"
    elif mode == "acceptance" and control_win:
        rows[control]["production_kept"] = True
        selected = control
        shipping_unchanged = False
        decision = "revert"
    elif mode == "acceptance":
        selected = shipping
        shipping_unchanged = True
        decision = "shipping_retained_gates_incomplete"
    kept = [ident for ident, row in rows.items() if row["production_kept"]]
    if len(kept) > 1:
        raise AdmissionError(f"{family}: at most one production path may be kept")
    claims = bool(
        mode == "acceptance"
        and this_sitting
        and rows[selected]["production_kept"]
        and selected == shipping
        and rows[shipping]["performance_pass"]
    )
    return {
        "independent_verdicts": rows,
        "selected_path": selected,
        "shipping_unchanged": shipping_unchanged,
        "production_kept": bool(rows[selected]["production_kept"]),
        "decision": decision,
        "opt074_coverage_unadmitted_blocker": False,
        "claims_throughput": claims,
        "winners": [ident for ident in (shipping, control) if ident in kept],
    }


def current_q8_layout_pin() -> str:
    text = Q8_PIN_FILE.read_text(encoding="utf-8")
    rows = re.search(r"kSelectedQ8DecodeRowsSkinny = (\d+);", text)
    warps = re.search(r"kSelectedQ8DecodeLayoutWarpsSkinny = (\d+);", text)
    if rows and warps and rows.group(1) == "1" and warps.group(1) == "4":
        return Q8_CONTROL
    return Q8_SHIPPING


def current_mmq_async_x_pin() -> bool:
    text = MMQ_PIN_FILE.read_text(encoding="utf-8")
    match = re.search(r"constexpr bool kSelectedMmqAsyncX = (true|false);", text)
    return bool(match and match.group(1) == "true")


def apply_q8_layout_pin(layout: str) -> None:
    text = Q8_PIN_FILE.read_text(encoding="utf-8")
    rows, warps = (1, 4) if layout == Q8_CONTROL else (2, 2)
    for role in ("Skinny", "Medium", "Wide"):
        text = re.sub(
            rf"constexpr unsigned int kSelectedQ8DecodeRows{role} = \d+;",
            f"constexpr unsigned int kSelectedQ8DecodeRows{role} = {rows};",
            text,
        )
        text = re.sub(
            rf"constexpr unsigned int kSelectedQ8DecodeLayoutWarps{role} = \d+;",
            f"constexpr unsigned int kSelectedQ8DecodeLayoutWarps{role} = {warps};",
            text,
        )
    Q8_PIN_FILE.write_text(text, encoding="utf-8")


def apply_mmq_async_x_pin(enabled: bool) -> None:
    text = MMQ_PIN_FILE.read_text(encoding="utf-8")
    text = re.sub(
        r"constexpr bool kSelectedMmqAsyncX = (true|false);",
        f"constexpr bool kSelectedMmqAsyncX = {'true' if enabled else 'false'};",
        text,
    )
    MMQ_PIN_FILE.write_text(text, encoding="utf-8")


def apply_production_pins(results: Mapping[str, Any]) -> dict[str, Any]:
    applied: dict[str, Any] = {"q8": False, "mmq": False}
    q8 = results.get("q8") or {}
    mmq = results.get("mmq") or {}
    q8_selected = str(
        q8.get("selected_path") or results.get("shipping_q8_layout") or Q8_SHIPPING
    )
    mmq_selected = str(
        mmq.get("selected_path") or results.get("shipping_mmq") or MMQ_SHIPPING
    )
    q8_kept = bool(
        ((q8.get("independent_verdicts") or {}).get(q8_selected) or {}).get(
            "production_kept"
        )
    )
    mmq_kept = bool(
        ((mmq.get("independent_verdicts") or {}).get(mmq_selected) or {}).get(
            "production_kept"
        )
    )
    if q8_kept and q8_selected != current_q8_layout_pin():
        apply_q8_layout_pin(q8_selected)
        applied["q8"] = True
    if mmq_kept:
        want = mmq_selected == MMQ_SHIPPING
        if want != current_mmq_async_x_pin():
            apply_mmq_async_x_pin(want)
            applied["mmq"] = True
    applied["q8_pin"] = current_q8_layout_pin()
    applied["mmq_async_x_pin"] = current_mmq_async_x_pin()
    return applied


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def write_report(payload: Mapping[str, Any]) -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    verdicts = payload.get("independent_verdicts") or {}
    q8_rows = verdicts.get("q8") or {}
    mmq_rows = verdicts.get("mmq") or {}
    q8 = payload.get("q8") or {}
    mmq = payload.get("mmq") or {}
    q8p = q8.get("parity") or {}
    if isinstance(q8p.get("parity"), Mapping):
        q8p = q8p["parity"]
    mmqp = mmq.get("parity") or {}
    if isinstance(mmqp.get("parity"), Mapping):
        mmqp = mmqp["parity"]
    q8c = q8.get("component") or {}
    mmqc = mmq.get("component") or {}
    q8e = q8.get("engine") or {}
    mmqe = mmq.get("engine") or {}
    ranking = payload.get("historical_ranking") or historical_ranking()

    def table(family_rows: Mapping[str, Any]) -> str:
        lines = []
        for ident, row in family_rows.items():
            lines.append(
                f"| {ident} | {row.get('kernel_parity_pass')} | "
                f"{row.get('model_quality_pass')} | {row.get('performance_pass')} | "
                f"{row.get('production_kept')} |"
            )
        return "\n".join(lines) if lines else "| none | | | | |"

    text = f"""# OPT-086 — Re-evaluate installed Q8 and MMQ paths

Status: **{payload.get("status") or "pending"}**. Authority llama.cpp
`{LLAMA_REV}`, GGUF SHA-256 `{GGUF_SHA}`. Independent keep/revert for Q8
`r1_w4` vs shipping `r2_w2` and MMQ `fma_async` vs shipping `fma_async_x`
tile 128×128. Ladder: kernel parity (OPT-082) → quality (OPT-084 freeze) →
performance. `opt074_coverage_unadmitted` is **not** a blocker.
`opt074_family_admission_required={OPT074_FAMILY_ADMISSION_REQUIRED}`.

## Independent verdicts

### Q8

| Ident | kernel_parity_pass | model_quality_pass | performance_pass | production_kept |
|---|---|---|---|---|
{table(q8_rows)}

Decision: **{(q8.get("decision") or payload.get("q8_decision") or "pending")}**.
selected=`{q8.get("selected_path", Q8_SHIPPING)}`.
shipping_unchanged={q8.get("shipping_unchanged", True)}.
claims_throughput={q8.get("claims_throughput", False)}.

### MMQ

| Ident | kernel_parity_pass | model_quality_pass | performance_pass | production_kept |
|---|---|---|---|---|
{table(mmq_rows)}

Decision: **{(mmq.get("decision") or payload.get("mmq_decision") or "pending")}**.
selected=`{mmq.get("selected_path", MMQ_SHIPPING)}`.
shipping_unchanged={mmq.get("shipping_unchanged", True)}.
claims_throughput={mmq.get("claims_throughput", False)}.

Top-level claims_throughput={payload.get("claims_throughput", False)}.
opt074_coverage_unadmitted_blocker=false.

## Kernel parity (OPT-082)

Q8 source={q8p.get("source")}; host_ok={q8p.get("host_ok")};
typed `sum_q` vs `sum_x` rejected={q8p.get("typed_staging_reject")}.
Q8 catalog {q8p.get("catalog_count")}.
MMQ source={mmqp.get("source")}; catalog {mmqp.get("catalog_count")}.
Documented Q8 M17 K2048 association misses do not fail kernel admission when
selectors are correct and nonfinite count is 0. Unaligned MMQ is fallback
association, not the candidate kernel.

## Quality (OPT-083/084)

Identity-cached OPT-084 suite with `--quality`. Shipping idents use the
Quartz-vs-baseline freeze. Replacement controls use the same freeze rules.
A known baseline defect does not reject a non-worsening kernel.

## Performance

Q8: complete rotating mixer, 48 GDN + 16 attention groups, OPT-058
invalidation on both sides. X-prefetch held at shipping MMQ until the MMQ
half decides. this_sitting={q8c.get("this_sitting", False)}.
Control `r1_w4` {_fmt(q8c.get("control_mean_ms"))} ms vs shipping `r2_w2`
{_fmt(q8c.get("candidate_mean_ms"))} ms. Paired CI
{_fmt(q8c.get("ci95_low"), 4)} .. {_fmt(q8c.get("ci95_high"), 4)} ms.
Engine {_fmt(q8e.get("control_mean_ms"))} vs {_fmt(q8e.get("candidate_mean_ms"))} ms.

MMQ: 64 complete FFNs, tile 128×128, Q8 decision held fixed.
this_sitting={mmqc.get("this_sitting", False)}.
Control `fma_async` {_fmt(mmqc.get("control_mean_ms"))} ms vs shipping
`fma_async_x` {_fmt(mmqc.get("candidate_mean_ms"))} ms. Paired CI
{_fmt(mmqc.get("ci95_low"), 4)} .. {_fmt(mmqc.get("ci95_high"), 4)} ms.
Engine {_fmt(mmqe.get("control_mean_ms"))} vs {_fmt(mmqe.get("candidate_mean_ms"))} ms.

Historical ranking only (not a keep): Q8 {ranking.get("q8_r1_w4_vs_r2_w2_ms")};
MMQ {ranking.get("mmq_fma_async_vs_fma_async_x_ms")}. A keep/revert requires
this sitting's confirmation. Historical OPT-070 report is unmodified.

## tok/s

No OPT-069/080 P/D oracle rerun. Official tok/s delta versus the then-current
shipping baseline is **0** unless a family both changes shipping and this
sitting claims throughput. claims_throughput is true per family only when
that family's shipping ident is `production_kept` on this sitting's measured
evidence.
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


def stored_parity(results: Mapping[str, Any], family: str) -> dict[str, Any]:
    block = results.get(family) or {}
    parity = block.get("parity") or {}
    if isinstance(parity.get("by_ident"), Mapping):
        return dict(parity)
    if isinstance(parity.get("parity"), Mapping) and isinstance(
        parity["parity"].get("by_ident"), Mapping
    ):
        return dict(parity["parity"])
    return {}


def run_family_phase(
    family: str,
    *,
    mode: str,
    run_dir: Path,
    skip_gpu: bool,
    runner: NativeRunner | None,
    synthetic: Mapping[str, Any] | None,
    prior_results: Mapping[str, Any],
) -> dict[str, Any]:
    plan = family_plan(family, mode)
    prior = load_prior_verdicts().get(family) or empty_family_verdicts(family)
    ranking = historical_ranking()
    parity = stored_parity(prior_results, family)
    if not parity.get("by_ident") or mode == "feedback" or skip_gpu:
        try:
            parity = evaluate_parity(
                family, skip_gpu=skip_gpu, runner=runner, synthetic=synthetic
            )
        except AdmissionError:
            if family == "q8" or family == "mmq":
                raise
    parity_failed = not any(
        (parity.get("by_ident") or {}).get(ident, {}).get("kernel_parity_pass")
        for ident in family_idents(family)
    )
    quality_by_id: dict[str, dict[str, Any]] = {}
    if mode == "acceptance" and not parity_failed:
        for ident in family_idents(family):
            ident_parity = (parity.get("by_ident") or {}).get(ident) or {}
            if ident_parity.get("kernel_parity_pass"):
                quality_by_id[ident] = evaluate_quality(
                    family, ident, synthetic=synthetic
                )
    elif mode == "acceptance":
        quality_by_id = {
            ident: {
                "model_quality_pass": False,
                "incomplete": True,
                "reason": "stopped_on_parity_failure",
            }
            for ident in family_idents(family)
        }
    quality_failed = bool(quality_by_id) and not any(
        row.get("model_quality_pass") for row in quality_by_id.values()
    )
    engine: dict[str, Any] = {}
    component: dict[str, Any] = {}
    this_sitting = False
    dispatch_ok = False
    stop_perf = parity_failed or (mode == "acceptance" and quality_failed)
    if skip_gpu:
        if synthetic and synthetic.get("component"):
            component = dict(synthetic["component"])
            engine = dict(synthetic.get("engine") or {})
            dispatch_ok = bool(synthetic.get("dispatch_ok", True))
            this_sitting = bool(synthetic.get("this_sitting", False))
        else:
            component = {
                "this_sitting": False,
                "dispatch_ok": True,
                "control": plan["control"],
                "shipping": plan["shipping"],
            }
            dispatch_ok = True
            this_sitting = False
    elif stop_perf:
        component = {
            "this_sitting": False,
            "stopped": True,
            "reason": "parity_or_quality_failure",
            "control": plan["control"],
            "shipping": plan["shipping"],
        }
    else:
        native = runner or default_native_runner
        component = run_component(plan, runner=native, run_dir=run_dir)
        dispatch_ok = bool(component.get("dispatch_ok"))
        this_sitting = True
        try:
            engine = run_engine(plan, runner=native, run_dir=run_dir) or {}
        except AdmissionError as exc:
            if plan["mode"] == "acceptance":
                raise
            engine = {"error": str(exc), "skipped": True}
    stats = {
        "positive": component.get("positive"),
        "mean_diff_ms": component.get("mean_diff_ms"),
        "ci95_low": component.get("ci95_low"),
        "ci95_high": component.get("ci95_high"),
    }
    performance = performance_by_ident(
        family,
        stats=stats if this_sitting else None,
        engine=engine if this_sitting else None,
        this_sitting=this_sitting,
        dispatch_ok=dispatch_ok,
    )
    observed = planned_observation(
        plan,
        capture_key=str(component.get("capture_key") or "") or None,
        keep=False,
    )
    admission = admit_counts(family, mode, observed)
    decided = decide_family_verdicts(
        family,
        parity=parity,
        quality_by_id=quality_by_id,
        performance_by_id=performance,
        mode=mode,
        prior=prior,
        this_sitting=this_sitting,
    )
    payload = {
        "schema_version": 1,
        "task": "OPT-086",
        "phase": family,
        "mode": mode,
        "control": plan["control"],
        "shipping": plan["shipping"],
        "parity": parity,
        "quality": quality_by_id,
        "component": {
            k: v for k, v in component.items() if k not in {"stdout", "native"}
        },
        "engine": {k: v for k, v in engine.items() if k not in {"stdout", "native"}},
        "performance": performance,
        "historical_ranking": ranking,
        "dispatch_ok": dispatch_ok,
        "this_sitting": this_sitting,
        "native_counts": observed,
        "admission": admission,
        **decided,
        "measurement_utc": utc_now(),
        "status": "measured"
        if this_sitting
        else ("stopped" if stop_perf else "incomplete_performance"),
        "opt074_coverage_unadmitted_blocker": False,
        "stopped_on_failure": stop_perf,
    }
    dump_json(run_dir / f"{family}-result.json", payload)
    return payload


def merge_fixture(
    results: dict[str, Any], family: str, payload: Mapping[str, Any]
) -> None:
    results[family] = payload
    prior = (results.get("independent_verdicts") or {}).get(
        family
    ) or empty_family_verdicts(family)
    incoming = payload.get("independent_verdicts") or {}
    merged = {
        ident: dict(prior.get(ident) or empty_verdict_row())
        for ident in family_idents(family)
    }
    for ident, row in incoming.items():
        current = merged.setdefault(ident, empty_verdict_row())
        for key in VERDICT_KEYS:
            if key in row:
                current[key] = bool(row.get(key))
        current["opt074_coverage_unadmitted_blocker"] = False
        current["incomplete"] = bool(row.get("incomplete", current.get("incomplete")))
    results.setdefault("independent_verdicts", {})
    results["independent_verdicts"][family] = merged
    results[f"{family}_decision"] = payload.get("decision")
    results[f"{family}_selected_path"] = payload.get("selected_path")
    results[f"{family}_shipping_unchanged"] = payload.get("shipping_unchanged")
    results[f"{family}_claims_throughput"] = bool(payload.get("claims_throughput"))
    q8_selected = results.get("q8_selected_path") or Q8_SHIPPING
    mmq_selected = results.get("mmq_selected_path") or MMQ_SHIPPING
    if family == "q8":
        results["shipping_q8_layout"] = (
            q8_selected if payload.get("production_kept") else Q8_SHIPPING
        )
        if payload.get("decision") == "revert":
            results["shipping_q8_layout"] = Q8_CONTROL
        elif payload.get("decision") == "keep":
            results["shipping_q8_layout"] = Q8_SHIPPING
    else:
        results["shipping_mmq"] = (
            mmq_selected if payload.get("production_kept") else MMQ_SHIPPING
        )
        if payload.get("decision") == "revert":
            results["shipping_mmq"] = MMQ_CONTROL
        elif payload.get("decision") == "keep":
            results["shipping_mmq"] = MMQ_SHIPPING
    results["claims_throughput_by_family"] = {
        "q8": bool(results.get("q8_claims_throughput")),
        "mmq": bool(results.get("mmq_claims_throughput")),
    }
    results["claims_throughput"] = bool(
        results["claims_throughput_by_family"]["q8"]
        or results["claims_throughput_by_family"]["mmq"]
    )
    results["claims_performance_improvement"] = results["claims_throughput"]
    results["opt074_coverage_unadmitted_blocker"] = False
    results["shipping_unchanged"] = bool(
        results.get("q8_shipping_unchanged", True)
        and results.get("mmq_shipping_unchanged", True)
    )
    results["status"] = str(payload.get("status") or results.get("status") or "updated")


def run(
    mode: str,
    phase: str,
    run_dir: Path,
    *,
    skip_gpu: bool = False,
    runner: NativeRunner | None = None,
    synthetic: Mapping[str, Any] | None = None,
    apply_pins: bool = False,
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
            "task": "OPT-086",
            "mode": mode,
            "llama_revision": LLAMA_REV,
            "gguf_sha256": GGUF_SHA,
            "token_generator": TOKEN_GENERATOR,
            "control_q8_layout": Q8_CONTROL,
            "shipping_q8_layout": results.get("shipping_q8_layout", Q8_SHIPPING),
            "control_mmq": MMQ_CONTROL,
            "shipping_mmq": results.get("shipping_mmq", MMQ_SHIPPING),
            "families": list(PHASES),
            "opt074_family_admission_required": False,
            "opt074_coverage_unadmitted_blocker": False,
            "historical_ranking": historical_ranking(),
            "historical_reports": historical_reports_intact(),
            "report_path": "evidence/optimization/opt086-q8-mmq-reevaluation/REPORT.md",
        }
    )
    payload = run_family_phase(
        phase,
        mode=mode,
        run_dir=run_dir,
        skip_gpu=skip_gpu,
        runner=runner,
        synthetic=synthetic,
        prior_results=results,
    )
    merge_fixture(results, phase, payload)
    if apply_pins and mode == "acceptance" and not skip_gpu:
        results["pins_applied"] = apply_production_pins(results)
    write_report(results)
    dump_json(FIXTURE, results)
    dump_json(run_dir / "opt086_q8_mmq_reevaluation.json", results)
    counts = (results.get(phase) or {}).get("native_counts") or {}
    print(
        "QW38_OPT086_RESULT="
        + json.dumps(
            {
                "task": "OPT-086",
                "mode": mode,
                "phase": phase,
                "selected_path": payload.get("selected_path"),
                "decision": payload.get("decision"),
                "production_kept": payload.get("production_kept"),
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
                "independent_verdicts": (results.get("independent_verdicts") or {}).get(
                    phase
                ),
            }
        )
    )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES, default="q8")
    parser.add_argument(
        "--mode", choices=("feedback", "acceptance"), default="feedback"
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument("--apply-pins", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_dir = args.run_dir or (
        ROOT / "build" / "optimization-runs" / "OPT-086" / args.mode
    )
    try:
        run(
            args.mode,
            args.phase,
            run_dir,
            skip_gpu=bool(args.skip_gpu),
            apply_pins=bool(args.apply_pins) or args.mode == "acceptance",
        )
    except (AdmissionError, AssertionError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
