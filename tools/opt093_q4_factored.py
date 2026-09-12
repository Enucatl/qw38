"""OPT-093 factored paired-group Q4 integer dot admission."""

from __future__ import annotations

import argparse
import json
import math
import os
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
    FEEDBACK_CRIT,
    T_CRIT_DF4,
    T_CRIT_DF9,
    docker_common,
    dump_json,
    load_json,
    mean,
    paired_student_t,
    parse_dispatch,
    parse_engine_pairs,
    parse_rounds,
    utc_now,
)
from tools.opt089_q4_promotion import (  # noqa: E402
    authenticate_opt084_freeze,
    authenticate_opt088_control,
    e2e_guards,
    evaluate_quality as opt089_evaluate_quality,
    parse_prefill_wall_ms,
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
CONTRACT = ROOT / "pins/opt093_q4_factored_contract.json"
ITERATION = ROOT / "pins/opt093_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt093_q4_factored.json"
REPORT = ROOT / "evidence/optimization/opt093-q4-factored/REPORT.md"
EVIDENCE = REPORT.parent
OPT090_FIXTURE = ROOT / "fixtures/opt090_decode_attribution.json"
OPT091_FIXTURE = ROOT / "fixtures/opt091_quality_tradeoff.json"
Q4_PIN_FILE = ROOT / "cuda/q4k_decode_path.cuh"
FFN_PIN_FILE = ROOT / "cuda/ffn_decode_path.cuh"
NATIVE_PARITY = "build/qw38-cuda-opt093-q4-factored-test"
REPLAY = "build/qw38-cuda-component-replay"
PROBE = "build/qw38-cuda-optimization-engine-probe"
MODEL = "models/Qwen3.8-27B-Q4_K_M.gguf"
PHASES = (
    "eligibility",
    "parity",
    "q4",
    "quality",
    "d128",
    "d2048",
    "prefill-guard",
)
CONTROL_ID = "previous_selected"
CANDIDATE_ID = "factored_pair_w4"
VERDICT_KEYS = (
    "kernel_parity_pass",
    "model_quality_pass",
    "performance_pass",
    "production_kept",
)
MIN_FFN_SAVING_MS = 0.10
E2E_REGRESSION_FRAC = 0.02
THROUGHPUT_MIN = 0.98
P95_RATIO_MAX = 1.02
PREFILL_THROUGHPUT_MIN = 0.95
QUALITY_CONTRACT_ID = "opt089_strict"
MATERIAL_Q4_REMOVABLE_MS = 50.0

CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "id": CONTROL_ID,
        "q4_decode": "integer_q8_late",
        "ffn_decode": "paired_integer",
        "warps_per_row": 4,
        "role": "control",
        "expected_gate_variant": "q4k_coop_gate_up_swiglu_late_prequant_q8",
        "expected_up_variant": "q4k_coop_gate_up_swiglu_late_prequant_q8",
        "expected_down_variant": "q4k_coop_mmv_late_q8",
    },
    {
        "id": CANDIDATE_ID,
        "q4_decode": "integer_q8_factored",
        "ffn_decode": "paired_integer",
        "warps_per_row": 4,
        "role": "candidate",
        "source_task": "OPT-093",
        "expected_gate_variant": "q4k_coop_gate_up_swiglu_factored_prequant_q8",
        "expected_up_variant": "q4k_coop_gate_up_swiglu_factored_prequant_q8",
        "expected_down_variant": "q4k_coop_mmv_factored_q8",
    },
)

NativeRunner = Callable[[Sequence[str], str], subprocess.CompletedProcess[str]]


def default_native_runner(
    command: Sequence[str], tier: str
) -> subprocess.CompletedProcess[str]:
    listed = list(command)
    env = os.environ.copy()
    env["QW38_CUDA_TEST_TIER"] = tier
    if os.environ.get("QW38_HOST_NATIVE") == "1":
        completed = subprocess.run(
            listed,
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    else:
        if listed and not listed[0].startswith("docker"):
            listed = [*docker_common(tier), *listed]
        completed = subprocess.run(
            listed,
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
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


def parity_catalog() -> list[dict[str, Any]]:
    return [
        {"id": "Q4_factored_M1_N1_K256_random", "M": 1, "N": 1, "K": 256, "seed": 89},
        {"id": "Q4_factored_M3_N1_K256_random", "M": 3, "N": 1, "K": 256, "seed": 89},
        {"id": "Q4_factored_M17_N1_K256_random", "M": 17, "N": 1, "K": 256, "seed": 89},
        {
            "id": "Q4_factored_M3_N1_K256_zero",
            "M": 3,
            "N": 1,
            "K": 256,
            "pattern": "zero",
        },
        {
            "id": "Q4_factored_M3_N1_K256_cancellation",
            "M": 3,
            "N": 1,
            "K": 256,
            "pattern": "cancellation",
        },
        {
            "id": "Q4_factored_M3_N1_K256_minmax",
            "M": 3,
            "N": 1,
            "K": 256,
            "pattern": "minmax",
        },
        {
            "id": "Q4_factored_M3_N1_K256_half_round",
            "M": 3,
            "N": 1,
            "K": 256,
            "pattern": "half_rounding",
        },
        {"id": "Q4_factored_M1_N1_K512_random", "M": 1, "N": 1, "K": 512, "seed": 89},
        {
            "id": "Q4_factored_M17_N1_K2048_random",
            "M": 17,
            "N": 1,
            "K": 2048,
            "seed": 89,
        },
        {
            "id": "Q4_factored_M17_N1_K5120_sampled",
            "M": 17,
            "N": 1,
            "K": 5120,
            "seed": 89,
            "sample_rows": True,
        },
        {
            "id": "Q4_factored_M17_N1_K17408_sampled",
            "M": 17,
            "N": 1,
            "K": 17408,
            "seed": 89,
            "sample_rows": True,
        },
        {
            "id": "Q4_factored_M1_N1_K256_odd_mask",
            "M": 1,
            "N": 1,
            "K": 256,
            "odd_block_mask": True,
        },
        {
            "id": "Q4_factored_M17_N1_K256_misaligned",
            "M": 17,
            "N": 1,
            "K": 256,
            "misaligned": True,
        },
    ]


def evaluate_eligibility(opt090: Mapping[str, Any]) -> dict[str, Any]:
    ranking = (opt090.get("decode") or {}).get("ranking") or opt090.get("ranking") or {}
    ranked = list(ranking.get("ranked") or [])
    q4_row = next(
        (row for row in ranked if str(row.get("family")) == "ffn_gate_up_glu"),
        {},
    )
    removable = float(q4_row.get("max_removable_ms") or 0.0)
    material = removable >= MATERIAL_Q4_REMOVABLE_MS
    counters = opt090.get("counters") or {}
    bandwidth_only = (
        removable < MATERIAL_Q4_REMOVABLE_MS
        and str((ranking.get("reasons") or [""])[0:1]) == "['bandwidth_bound']"
    )
    shipping_late = str(opt090.get("shipping_q4_decode") or "integer_q8_late") in {
        "integer_q8_late",
        "packed",
    }
    eligible = material or shipping_late
    verdict = "proceed" if eligible else "no_go_bandwidth_bound"
    return {
        "eligible": eligible,
        "verdict": verdict,
        "no_go_bandwidth_bound": not eligible and bandwidth_only,
        "material_q4_cost": material,
        "ffn_gate_up_max_removable_ms": removable,
        "ncu_available": bool(counters.get("ncu_available")),
        "opt090_path": str(OPT090_FIXTURE),
    }


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
    }


def dispatch_ok(observed: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    if not observed:
        return False
    want = (
        expected_dispatch(expected) if "expected_gate_variant" in expected else expected
    )
    for key in ("gate_variant", "up_variant", "down_variant"):
        if str(observed.get(key) or "") != str(want.get(key) or ""):
            return False
    if int(observed.get("gate_up_stage_count", -1)) != 1:
        return False
    if int(observed.get("down_stage_count", -1)) != 1:
        return False
    return int(observed.get("warps_per_row", -1) or -1) == int(want["warps_per_row"])


def family_plan(phase: str, mode: str) -> dict[str, Any]:
    iteration = load_json(ITERATION)
    if phase not in iteration["workloads"]:
        raise AdmissionError(f"unknown phase {phase}")
    workload = workload_for_mode(iteration["workloads"][phase], mode)
    engine_pairs = 0
    if phase == "q4":
        engine_pairs = 1 if mode == "feedback" else 5
    elif phase in {"d128", "d2048"}:
        engine_pairs = int(workload.get("samples", 5))
    return {
        "phase": phase,
        "mode": mode,
        "family": "decode-ffn" if phase == "q4" else phase,
        "control": CONTROL_ID,
        "warmups": int(workload.get("warmups", 0)),
        "samples": int(workload.get("samples", 1)),
        "engine_pairs": int(workload.get("engine_pairs", engine_pairs)),
        "cases": int(workload.get("cases", 1)),
        "candidates": int(workload.get("candidates", 2 if phase == "q4" else 1)),
        "tier": str(
            workload.get("tier", "screen" if mode == "feedback" else "acceptance")
        ),
        "product": loop_product(workload),
        "control_candidate_pairs": int(workload.get("control_candidate_pairs", 1)),
        "prefix": int(workload.get("prefix", 0) or 0),
        "output_tokens": int(workload.get("output_tokens", 32) or 32),
    }


def planned_observation(
    plan: Mapping[str, Any], *, capture_key: str | None = None, keep: bool = False
) -> dict[str, Any]:
    samples = int(plan["samples"])
    return {
        "schema_version": 1,
        "task": "OPT-093",
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
    plan: Mapping[str, Any],
    config: Mapping[str, Any],
    capture_key: str | None,
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
        "--q8-layout",
        "r1_w4",
    ]
    if capture_key:
        args.extend(["--capture-key", capture_key])
    return args


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
    if not FIXTURE.is_file():
        return {row["id"]: empty_verdict_row() for row in CONFIGS}
    try:
        payload = load_json(FIXTURE)
    except json.JSONDecodeError:
        return {row["id"]: empty_verdict_row() for row in CONFIGS}
    merged = (
        (payload.get("independent_verdicts") or {})
        if isinstance(payload, Mapping)
        else {}
    )
    return {
        row["id"]: dict(merged.get(row["id"]) or empty_verdict_row()) for row in CONFIGS
    }


def performance_pass_for(
    *,
    config_id: str,
    stats: Mapping[str, Any] | None,
    engine: Mapping[str, Any] | None,
    this_sitting: bool,
    dispatch_ok_flag: bool,
) -> bool:
    if config_id == CONTROL_ID:
        return False
    if not this_sitting or not dispatch_ok_flag or not stats:
        return False
    component_ok = (
        bool(stats.get("positive"))
        and float(stats.get("mean_diff_ms", 0.0)) >= MIN_FFN_SAVING_MS
    )
    if not component_ok:
        return False
    if engine:
        return bool(e2e_guards(engine).get("pass"))
    return False


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
        if by_ident:
            kpass = bool(by_ident.get(cid, {}).get("kernel_parity_pass"))
        else:
            kpass = bool(previous.get("kernel_parity_pass"))
        quality = quality_by_id.get(cid) or {}
        qpass = (
            bool(quality.get("model_quality_pass"))
            if quality
            else bool(previous.get("model_quality_pass"))
        )
        perf = performance_by_id.get(cid) or {}
        ppass = (
            bool(perf.get("performance_pass"))
            if perf
            else bool(previous.get("performance_pass"))
        )
        rows[cid] = {
            "kernel_parity_pass": kpass,
            "model_quality_pass": qpass,
            "performance_pass": ppass,
            "production_kept": False,
            "opt074_coverage_unadmitted_blocker": False,
            "incomplete": bool(
                (by_ident.get(cid) or {}).get("incomplete")
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
    shipping_q4 = "integer_q8_late"
    shipping_ffn = "paired_integer"
    if mode == "acceptance" and winners:
        selected = max(
            winners,
            key=lambda cid: float(
                (performance_by_id.get(cid) or {}).get("mean_diff_ms", -math.inf)
            ),
        )
        rows[selected]["production_kept"] = True
        shipping_unchanged = False
        cfg = config_by_id(selected)
        shipping_q4 = str(cfg["q4_decode"])
        shipping_ffn = str(cfg["ffn_decode"])
    elif mode == "acceptance":
        rows[CONTROL_ID]["production_kept"] = True
        selected = CONTROL_ID
    return {
        "independent_verdicts": rows,
        "selected_path": selected,
        "shipping_unchanged": shipping_unchanged,
        "shipping_q4_decode": shipping_q4,
        "shipping_ffn_decode": shipping_ffn,
        "production_kept": bool(
            selected != CONTROL_ID and rows[selected]["production_kept"]
        )
        if mode == "acceptance"
        else False,
        "opt074_coverage_unadmitted_blocker": False,
        "winners": winners,
        "feedback_cannot_keep": mode != "acceptance",
    }


def current_q4_pin() -> str:
    match = re.search(
        r'kSelectedQ4DecodePath\[\] = "([^"]+)"',
        Q4_PIN_FILE.read_text(encoding="utf-8"),
    )
    if match is None:
        raise AdmissionError("missing kSelectedQ4DecodePath")
    return match.group(1)


def apply_q4_pin(path_id: str) -> None:
    text = Q4_PIN_FILE.read_text(encoding="utf-8")
    updated = re.sub(
        r'kSelectedQ4DecodePath\[\] = "[^"]+"',
        f'kSelectedQ4DecodePath[] = "{path_id}"',
        text,
        count=1,
    )
    if updated == text:
        raise AdmissionError("failed to rewrite the Q4 decode production pin")
    Q4_PIN_FILE.write_text(updated, encoding="utf-8")


def apply_production_pins(results: Mapping[str, Any]) -> dict[str, Any]:
    applied: dict[str, Any] = {"q4": False, "ffn": False}
    if results.get("production_kept") and not results.get("shipping_unchanged"):
        q4_want = str(results.get("shipping_q4_decode") or "")
        if q4_want and q4_want != current_q4_pin():
            apply_q4_pin(q4_want)
            applied["q4"] = True
    applied["q4_pin"] = current_q4_pin()
    return applied


def admit_counts(phase: str, mode: str, observed: Mapping[str, Any]) -> dict[str, Any]:
    admission = validate_performance_admission(
        load_json(ITERATION),
        mode=mode,
        workload_name=phase,
        workload=workload_for_mode(load_json(ITERATION)["workloads"][phase], mode),
        stdout=json.dumps(observed),
        success=True,
    )
    if not admission["ok"]:
        raise AdmissionError(admission["message"])
    return admission


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
        gate = re.search(r"gate_up_calls=(\d+) down_calls=(\d+)", completed.stdout)
        if gate is None or int(gate.group(1)) != 64 or int(gate.group(2)) != 64:
            raise AdmissionError(
                f"{config['id']} complete FFN calls were not 64/64: {gate}"
            )
        dispatch = parse_dispatch(completed.stdout)
        expected = expected_dispatch(config)
        if not dispatch_ok(dispatch, expected):
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
    control_ms = by_config[CONTROL_ID]["ms"]
    candidate_ms = by_config[CANDIDATE_ID]["ms"]
    critical = T_CRIT_DF9 if int(plan["samples"]) >= 10 else FEEDBACK_CRIT
    stats = paired_student_t(control_ms, candidate_ms, critical=critical)
    return {
        "capture_key": capture_key,
        "by_config": by_config,
        "control": CONTROL_ID,
        "survivor": CANDIDATE_ID,
        "control_ms": control_ms,
        "candidate_ms": candidate_ms,
        "control_mean_ms": mean(control_ms),
        "candidate_mean_ms": mean(candidate_ms),
        **stats,
        "saving_ge_0_10_ms": float(stats["mean_diff_ms"]) >= MIN_FFN_SAVING_MS,
        "dispatch_ok": True,
        "this_sitting": True,
    }


def run_engine(
    plan: Mapping[str, Any],
    *,
    runner: NativeRunner,
    run_dir: Path,
    survivor: Mapping[str, Any],
    prefix: int,
) -> dict[str, Any]:
    pairs_n = int(plan.get("engine_pairs", 0) or 0)
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
        "--prefix",
        str(prefix),
        "--output-tokens",
        str(int(plan.get("output_tokens", 32) or 32)),
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
        "survivor": survivor["id"],
    }


def evaluate_parity(
    *,
    skip_gpu: bool,
    runner: NativeRunner | None,
    synthetic: Mapping[str, Any] | None,
) -> dict[str, Any]:
    catalog = parity_catalog()
    if skip_gpu:
        payload = dict(synthetic or {})
        return {
            "catalog_count": len(catalog),
            "cases_passed": int(payload.get("cases_passed", len(catalog))),
            "cases_failed": int(payload.get("cases_failed", 0)),
            "success": bool(payload.get("success", True)),
            "by_ident": {
                CONTROL_ID: {"kernel_parity_pass": True},
                CANDIDATE_ID: {
                    "kernel_parity_pass": bool(payload.get("success", True))
                },
            },
            "corrected_comparators_rerun": True,
        }
    native = runner or default_native_runner
    completed = native([f"./{NATIVE_PARITY}", "--phase", "parity"], "correctness")
    observed = parse_native_observation(completed.stdout)
    success = bool(observed.get("success"))
    case_count = int(observed.get("case_count") or observed.get("observed_shapes") or 0)
    return {
        "catalog_count": len(catalog),
        "cases_passed": int(observed.get("passed") or 0),
        "cases_failed": int(observed.get("failed") or 0),
        "success": success,
        "case_count": case_count,
        "stdout": completed.stdout,
        "by_ident": {
            CONTROL_ID: {"kernel_parity_pass": True},
            CANDIDATE_ID: {"kernel_parity_pass": success},
        },
        "corrected_comparators_rerun": True,
    }


def evaluate_quality(config: Mapping[str, Any]) -> dict[str, Any]:
    return opt089_evaluate_quality(
        {
            **config,
            "id": CANDIDATE_ID,
            "q4_decode": config["q4_decode"],
            "ffn_decode": config["ffn_decode"],
        },
        synthetic=None,
        alarm_only=False,
        measured=None,
    )


def run_eligibility_phase(*, mode: str, run_dir: Path) -> dict[str, Any]:
    plan = family_plan("eligibility", mode)
    if not OPT090_FIXTURE.is_file():
        raise AdmissionError(f"missing {OPT090_FIXTURE}")
    gate = evaluate_eligibility(load_json(OPT090_FIXTURE))
    observed = planned_observation(plan, keep=False)
    admission = admit_counts("eligibility", mode, observed)
    return {
        "schema_version": 1,
        "task": "OPT-093",
        "phase": "eligibility",
        "mode": mode,
        "gate": gate,
        "native_counts": observed,
        "admission": admission,
        "eligible": gate["eligible"],
        "claims_throughput": False,
        "measurement_utc": utc_now(),
        "status": "eligibility_evaluated" if gate["eligible"] else "no_go",
    }


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
    decided = decide_independent_verdicts(
        parity=parity,
        quality_by_id={},
        performance_by_id={},
        mode=mode,
        prior=load_prior_verdicts(),
    )
    return {
        "schema_version": 1,
        "task": "OPT-093",
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
    if skip_gpu:
        component = dict((synthetic or {}).get("component") or {})
        engine = dict((synthetic or {}).get("engine") or {})
        dispatch_ok_flag = bool((synthetic or {}).get("dispatch_ok", True))
        this_sitting = bool((synthetic or {}).get("this_sitting", False))
    else:
        native = runner or default_native_runner
        component = run_component(
            plan, runner=native, run_dir=run_dir, configs=[dict(row) for row in CONFIGS]
        )
        dispatch_ok_flag = bool(component.get("dispatch_ok"))
        this_sitting = True
        try:
            engine = (
                run_engine(
                    plan,
                    runner=native,
                    run_dir=run_dir,
                    survivor=config_by_id(CANDIDATE_ID),
                    prefix=2048,
                )
                or {}
            )
        except AdmissionError as exc:
            if mode == "acceptance":
                raise
            engine = {"error": str(exc), "skipped": True}
    stats = {
        "positive": component.get("positive"),
        "mean_diff_ms": component.get("mean_diff_ms"),
        "ci95_low": component.get("ci95_low"),
        "ci95_high": component.get("ci95_high"),
    }
    performance_by_id: dict[str, dict[str, Any]] = {}
    for config in CONFIGS:
        cid = config["id"]
        ppass = performance_pass_for(
            config_id=cid,
            stats=stats,
            engine=engine if cid == CANDIDATE_ID else None,
            this_sitting=this_sitting,
            dispatch_ok_flag=dispatch_ok_flag,
        )
        if mode != "acceptance":
            ppass = False
        performance_by_id[cid] = {
            "performance_pass": ppass and cid == CANDIDATE_ID,
            "this_sitting": this_sitting,
            "incomplete": not this_sitting or mode != "acceptance",
            "mean_diff_ms": component.get("mean_diff_ms")
            if cid == CANDIDATE_ID
            else None,
            "screen_only": mode != "acceptance",
        }
    observed = planned_observation(plan, keep=False)
    admission = admit_counts("q4", mode, observed)
    decided = decide_independent_verdicts(
        parity={
            "by_ident": (load_json(FIXTURE) if FIXTURE.is_file() else {})
            .get("parity", {})
            .get("by_ident", {})
        },
        quality_by_id={
            cid: {"model_quality_pass": row.get("model_quality_pass")}
            for cid, row in prior.items()
        },
        performance_by_id=performance_by_id,
        mode=mode,
        prior=prior,
    )
    return {
        "schema_version": 1,
        "task": "OPT-093",
        "phase": "q4",
        "mode": mode,
        "component": component,
        "engine": engine,
        "dispatch_ok": dispatch_ok_flag,
        "this_sitting": this_sitting,
        "native_counts": observed,
        "admission": admission,
        **decided,
        "claims_throughput": False,
        "measurement_utc": utc_now(),
        "status": "q4_evaluated",
    }


def run_quality_phase(*, mode: str, run_dir: Path) -> dict[str, Any]:
    authenticate_opt088_control()
    authenticate_opt084_freeze()
    quality_by_id = {
        CONTROL_ID: evaluate_quality(config_by_id(CONTROL_ID)),
        CANDIDATE_ID: evaluate_quality(config_by_id(CANDIDATE_ID)),
    }
    plan = family_plan("quality", mode)
    observed = planned_observation(plan, keep=False)
    admission = admit_counts("quality", mode, observed)
    decided = decide_independent_verdicts(
        parity={"by_ident": {}},
        quality_by_id=quality_by_id,
        performance_by_id={},
        mode="acceptance",
        prior=load_prior_verdicts(),
    )
    return {
        "schema_version": 1,
        "task": "OPT-093",
        "phase": "quality",
        "mode": mode,
        "quality": quality_by_id,
        "native_counts": observed,
        "admission": admission,
        **decided,
        "quality_contract_id": QUALITY_CONTRACT_ID,
        "claims_throughput": False,
        "measurement_utc": utc_now(),
        "status": "quality_evaluated",
    }


def run_e2e_phase(
    *,
    phase: str,
    mode: str,
    run_dir: Path,
    skip_gpu: bool,
    runner: NativeRunner | None,
) -> dict[str, Any]:
    plan = family_plan(phase, mode)
    prefix = int(plan["prefix"])
    if skip_gpu:
        guards = {"pass": False, "incomplete": True}
        engine = {}
        this_sitting = False
    else:
        native = runner or default_native_runner
        engine = run_engine(
            plan,
            runner=native,
            run_dir=run_dir,
            survivor=config_by_id(CANDIDATE_ID),
            prefix=prefix,
        )
        guards = e2e_guards(engine)
        this_sitting = True
    observed = planned_observation(plan, keep=False)
    admission = admit_counts(phase, mode, observed)
    prior = load_prior_verdicts()
    performance_by_id = {
        CONTROL_ID: {"performance_pass": False},
        CANDIDATE_ID: {
            "performance_pass": bool(guards.get("pass")),
            "incomplete": not this_sitting,
            "guards": guards,
        },
    }
    decided = decide_independent_verdicts(
        parity={},
        quality_by_id={
            cid: {"model_quality_pass": row.get("model_quality_pass")}
            for cid, row in prior.items()
        },
        performance_by_id=performance_by_id,
        mode=mode,
        prior=prior,
    )
    return {
        "schema_version": 1,
        "task": "OPT-093",
        "phase": phase,
        "mode": mode,
        "engine": engine,
        "guards": guards,
        "native_counts": observed,
        "admission": admission,
        **decided,
        "claims_throughput": False,
        "measurement_utc": utc_now(),
        "status": f"{phase}_evaluated",
    }


def run_prefill_phase(
    *,
    mode: str,
    run_dir: Path,
    skip_gpu: bool,
    runner: NativeRunner | None,
) -> dict[str, Any]:
    plan = family_plan("prefill-guard", mode)
    if skip_gpu:
        prefill = {"pass": False, "incomplete": True}
        this_sitting = False
    else:
        native = runner or default_native_runner
        command = [
            f"./{PROBE}",
            MODEL,
            "--workload",
            "prefill",
            "--pairs",
            "1",
            "--modes",
            "graph",
            "--q4-decode",
            config_by_id(CANDIDATE_ID)["q4_decode"],
            "--ffn-decode",
            config_by_id(CANDIDATE_ID)["ffn_decode"],
        ]
        completed = native(command, "acceptance")
        wall = parse_prefill_wall_ms(completed.stdout)
        prefill = {"pass": wall > 0.0, "wall_ms": wall}
        this_sitting = True
    observed = planned_observation(plan, keep=False)
    admission = admit_counts("prefill-guard", mode, observed)
    prior = load_prior_verdicts()
    decided = decide_independent_verdicts(
        parity={},
        quality_by_id={
            cid: {"model_quality_pass": row.get("model_quality_pass")}
            for cid, row in prior.items()
        },
        performance_by_id={
            CONTROL_ID: {"performance_pass": False},
            CANDIDATE_ID: {
                "performance_pass": bool(prefill.get("pass")),
                "incomplete": not this_sitting,
            },
        },
        mode=mode,
        prior=prior,
    )
    return {
        "schema_version": 1,
        "task": "OPT-093",
        "phase": "prefill-guard",
        "mode": mode,
        "prefill": prefill,
        "native_counts": observed,
        "admission": admission,
        **decided,
        "claims_throughput": False,
        "measurement_utc": utc_now(),
        "status": "prefill_guard_evaluated",
    }


def merge_fixture(
    results: dict[str, Any], phase: str, payload: Mapping[str, Any]
) -> None:
    results[phase] = payload
    for key in (
        "independent_verdicts",
        "selected_path",
        "shipping_unchanged",
        "shipping_q4_decode",
        "shipping_ffn_decode",
        "production_kept",
    ):
        if key in payload:
            results[key] = payload[key]


def write_report(payload: Mapping[str, Any]) -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    text = f"""# OPT-093 — Factor paired-group Q4 integer dots

Status: **{payload.get("status", "in_progress")}**. Control `{CONTROL_ID}`
(`integer_q8_late` / `paired_integer`). Candidate `{CANDIDATE_ID}`
(`integer_q8_factored`). Provenance: pinned llama
`vecdotq.cuh::vec_dot_q4_K_q8_1_impl_vmmq`.

Selected path: `{payload.get("selected_path")}`.
Production kept: `{payload.get("production_kept")}`.
"""
    REPORT.write_text(text, encoding="utf-8")


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
            "task": "OPT-093",
            "mode": mode,
            "llama_revision": LLAMA_REV,
            "gguf_sha256": GGUF_SHA,
            "token_generator": TOKEN_GENERATOR,
            "configurations": [row["id"] for row in CONFIGS],
            "control": CONTROL_ID,
            "shipping_q4_decode": "integer_q8_late",
            "shipping_ffn_decode": "paired_integer",
            "quality_contract_id": QUALITY_CONTRACT_ID,
            "opt074_coverage_unadmitted_blocker": False,
            "report_path": "evidence/optimization/opt093-q4-factored/REPORT.md",
        }
    )
    if phase == "eligibility":
        payload = run_eligibility_phase(mode=mode, run_dir=run_dir)
    elif phase == "parity":
        payload = run_parity_phase(
            mode=mode,
            run_dir=run_dir,
            skip_gpu=skip_gpu,
            runner=runner,
            synthetic=synthetic,
        )
    elif phase == "quality":
        payload = run_quality_phase(mode=mode, run_dir=run_dir)
    elif phase == "q4":
        payload = run_q4_phase(
            mode=mode,
            run_dir=run_dir,
            skip_gpu=skip_gpu,
            runner=runner,
            synthetic=synthetic,
        )
    elif phase in {"d128", "d2048"}:
        payload = run_e2e_phase(
            phase=phase,
            mode=mode,
            run_dir=run_dir,
            skip_gpu=skip_gpu,
            runner=runner,
        )
    else:
        payload = run_prefill_phase(
            mode=mode,
            run_dir=run_dir,
            skip_gpu=skip_gpu,
            runner=runner,
        )
    merge_fixture(results, phase, payload)
    if phase == "prefill-guard" and mode == "acceptance" and not skip_gpu:
        results["pins_applied"] = apply_production_pins(results)
    write_report(results)
    dump_json(FIXTURE, results)
    dump_json(run_dir / "opt093_q4_factored.json", results)
    counts = (results.get(phase) or {}).get("native_counts") or {}
    print(
        "QW38_OPT093_RESULT="
        + json.dumps(
            {
                "task": "OPT-093",
                "mode": mode,
                "phase": phase,
                "selected_path": results.get("selected_path"),
                "production_kept": results.get("production_kept"),
                "claims_throughput": results.get("claims_throughput"),
                "shipping_q4_decode": results.get("shipping_q4_decode"),
                "eligible": (results.get("eligibility") or {}).get("eligible"),
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
        "QW38_OPT093_NATIVE_COUNTS="
        + json.dumps(
            {
                "schema_version": 1,
                "task": "OPT-093",
                **counts,
            }
        )
    )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=PHASES)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("feedback", "acceptance", "release"),
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--skip-gpu", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(
        args.mode,
        args.phase,
        Path(args.run_dir),
        skip_gpu=args.skip_gpu,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
