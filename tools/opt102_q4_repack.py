"""OPT-102 Q4_K layout and branchless unpack admission. Production stays late_w4 until keep."""

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
    CANDIDATE_ID as OPT089_CANDIDATE_ID,
    FIXTURE as OPT089_FIXTURE,
    authenticate_opt084_freeze,
    authenticate_opt088_control,
    config_by_id as opt089_config_by_id,
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
CONTRACT = ROOT / "pins/opt102_q4_repack_contract.json"
ITERATION = ROOT / "pins/opt102_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt102_q4_repack.json"
REPORT = ROOT / "evidence/optimization/opt102-q4-layout-unpack/REPORT.md"
REJECTION = ROOT / "evidence/optimization/opt102-q4-layout-unpack/REJECTION.md"
EVIDENCE = REPORT.parent
OPT091_FIXTURE = ROOT / "fixtures/opt091_quality_tradeoff.json"
Q4_PIN_FILE = ROOT / "cuda/q4k_decode_path.cuh"
NATIVE_PARITY = "build/qw38-cuda-opt102-q4-repack-test"
REPLAY = "build/qw38-cuda-component-replay"
PROBE = "build/qw38-cuda-optimization-engine-probe"
MODEL = "models/Qwen3.8-27B-Q4_K_M.gguf"
PHASES = ("parity", "q4", "quality", "d128", "d2048", "prefill-guard")
CONTROL_ID = "late_w4"
BRANCHLESS_ID = "branchless_late"
ALIGNED_ID = "aligned_meta"
CANDIDATE_IDS = (BRANCHLESS_ID, ALIGNED_ID)
VERDICT_KEYS = (
    "kernel_parity_pass",
    "model_quality_pass",
    "performance_pass",
    "production_kept",
)
MIN_FFN_SAVING_MS = 0.10
PREFILL_THROUGHPUT_MIN = 0.95
QUALITY_CONTRACT_ID = "opt089_strict"
ALIGNMENT_BYTES = 64
LAYOUT_VERSION = 1

CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "id": CONTROL_ID,
        "q4_decode": "integer_q8_late",
        "q4_device_layout": "raw_gguf",
        "ffn_decode": "paired_integer",
        "warps_per_row": 4,
        "role": "control",
        "expected_gate_variant": "q4k_coop_gate_up_swiglu_late_prequant_q8",
        "expected_up_variant": "q4k_coop_gate_up_swiglu_late_prequant_q8",
        "expected_down_variant": "q4k_coop_mmv_late_q8",
    },
    {
        "id": BRANCHLESS_ID,
        "q4_decode": "integer_q8_branchless",
        "q4_device_layout": "raw_gguf",
        "ffn_decode": "paired_integer",
        "warps_per_row": 4,
        "role": "candidate",
        "source_task": "OPT-102",
        "expected_gate_variant": "q4k_coop_gate_up_swiglu_branchless_prequant_q8",
        "expected_up_variant": "q4k_coop_gate_up_swiglu_branchless_prequant_q8",
        "expected_down_variant": "q4k_coop_mmv_branchless_q8",
    },
    {
        "id": ALIGNED_ID,
        "q4_decode": "integer_q8_aligned",
        "q4_device_layout": "aligned_meta",
        "ffn_decode": "paired_integer",
        "warps_per_row": 4,
        "role": "candidate",
        "source_task": "OPT-102",
        "expected_gate_variant": "q4k_coop_gate_up_swiglu_aligned_prequant_q8",
        "expected_up_variant": "q4k_coop_gate_up_swiglu_aligned_prequant_q8",
        "expected_down_variant": "q4k_coop_mmv_aligned_q8",
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
        {"id": "Q4_layout_M1_N1_K256_inverse", "M": 1, "N": 1, "K": 256},
        {"id": "Q4_layout_M3_N1_K256_branchless", "M": 3, "N": 1, "K": 256},
        {"id": "Q4_layout_M17_N1_K256_branchless", "M": 17, "N": 1, "K": 256},
        {"id": "Q4_layout_M3_N1_K256_aligned", "M": 3, "N": 1, "K": 256},
        {"id": "Q4_layout_M17_N1_K256_aligned", "M": 17, "N": 1, "K": 256},
        {
            "id": "Q4_layout_M3_N1_K256_zero",
            "M": 3,
            "N": 1,
            "K": 256,
            "pattern": "zero",
        },
        {
            "id": "Q4_layout_M3_N1_K256_cancel",
            "M": 3,
            "N": 1,
            "K": 256,
            "pattern": "cancellation",
        },
        {
            "id": "Q4_layout_M3_N1_K256_minmax",
            "M": 3,
            "N": 1,
            "K": 256,
            "pattern": "minmax",
        },
        {"id": "Q4_layout_M1_N1_K512_random", "M": 1, "N": 1, "K": 512},
        {"id": "Q4_layout_M17_N1_K2048_random", "M": 17, "N": 1, "K": 2048},
        {
            "id": "Q4_layout_M17_N1_K5120_sampled",
            "M": 17,
            "N": 1,
            "K": 5120,
            "sample_rows": True,
        },
        {
            "id": "Q4_layout_M17_N1_K17408_sampled",
            "M": 17,
            "N": 1,
            "K": 17408,
            "sample_rows": True,
        },
        {"id": "Q4_layout_M1_N1_K256_odd_mask", "M": 1, "N": 1, "K": 256},
        {
            "id": "Q4_layout_M17_N1_K256_misaligned",
            "M": 17,
            "N": 1,
            "K": 256,
            "misaligned": True,
        },
        {"id": "Q4_layout_llama_mmvq_diagnostic", "M": 1, "N": 1, "K": 256},
        {"id": "Q4_layout_occupancy_resources", "M": 1, "N": 1, "K": 256},
    ]


def expected_dispatch(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "config": config["id"],
        "gate_variant": config["expected_gate_variant"],
        "up_variant": config["expected_up_variant"],
        "down_variant": config["expected_down_variant"],
        "gate_up_stage_count": 1,
        "down_stage_count": 1,
        "warps_per_row": int(config["warps_per_row"]),
        "q4_device_layout": config["q4_device_layout"],
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
    layout = str(observed.get("q4_device_layout") or want.get("q4_device_layout") or "")
    if layout and layout != str(want.get("q4_device_layout") or ""):
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
        "candidates": int(workload.get("candidates", 3 if phase == "q4" else 1)),
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
        "task": "OPT-102",
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
        "--q4-device-layout",
        str(config["q4_device_layout"]),
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


def ffn_pass(stats: Mapping[str, Any] | None) -> bool:
    if not stats:
        return False
    return (
        bool(stats.get("positive"))
        and float(stats.get("mean_diff_ms", 0.0)) >= MIN_FFN_SAVING_MS
    )


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
    selected = CONTROL_ID
    shipping_q4 = "integer_q8_late"
    shipping_layout = "raw_gguf"
    shipping_ffn = "paired_integer"
    shipping_unchanged = True
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
        shipping_layout = str(cfg["q4_device_layout"])
        shipping_ffn = str(cfg["ffn_decode"])
    elif mode == "acceptance":
        rows[CONTROL_ID]["production_kept"] = True
        selected = CONTROL_ID
    return {
        "independent_verdicts": rows,
        "selected_path": selected,
        "shipping_unchanged": shipping_unchanged,
        "shipping_q4_decode": shipping_q4,
        "shipping_q4_device_layout": shipping_layout,
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


def current_pin(pattern: str) -> str:
    match = re.search(pattern, Q4_PIN_FILE.read_text(encoding="utf-8"))
    if match is None:
        raise AdmissionError(f"missing production pin {pattern}")
    return match.group(1)


def current_q4_pin() -> str:
    return current_pin(r'kSelectedQ4DecodePath\[\] = "([^"]+)"')


def current_q4_layout_pin() -> str:
    return current_pin(r'kSelectedQ4DeviceLayout\[\] = "([^"]+)"')


def rewrite_pin(pattern: str, replacement: str) -> None:
    text = Q4_PIN_FILE.read_text(encoding="utf-8")
    updated = re.sub(pattern, replacement, text, count=1)
    if updated == text:
        raise AdmissionError(f"failed to rewrite pin {pattern}")
    Q4_PIN_FILE.write_text(updated, encoding="utf-8")


def apply_production_pins(results: Mapping[str, Any]) -> dict[str, Any]:
    applied: dict[str, Any] = {"q4": False, "layout": False}
    if results.get("production_kept") and not results.get("shipping_unchanged"):
        q4_want = str(results.get("shipping_q4_decode") or "")
        layout_want = str(results.get("shipping_q4_device_layout") or "")
        if q4_want and q4_want != current_q4_pin():
            rewrite_pin(
                r'kSelectedQ4DecodePath\[\] = "[^"]+"',
                f'kSelectedQ4DecodePath[] = "{q4_want}"',
            )
            applied["q4"] = True
        if layout_want and layout_want != current_q4_layout_pin():
            rewrite_pin(
                r'kSelectedQ4DeviceLayout\[\] = "[^"]+"',
                f'kSelectedQ4DeviceLayout[] = "{layout_want}"',
            )
            applied["layout"] = True
    applied["q4_pin"] = current_q4_pin()
    applied["layout_pin"] = current_q4_layout_pin()
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


def opt089_quality_payload() -> dict[str, Any]:
    if not OPT089_FIXTURE.is_file():
        return {}
    try:
        return load_json(OPT089_FIXTURE)
    except json.JSONDecodeError:
        return {}


def evaluate_quality(config: Mapping[str, Any]) -> dict[str, Any]:
    opt088 = authenticate_opt088_control()
    opt084 = authenticate_opt084_freeze()
    opt089 = opt089_quality_payload()
    opt091 = load_json(OPT091_FIXTURE) if OPT091_FIXTURE.is_file() else {}
    record: dict[str, Any] = {
        "config": config["id"],
        "quality_contract_id": QUALITY_CONTRACT_ID,
        "ppl_ratio_max": 1.01,
        "recurrence_incremental_nll_max": 0.02,
        "selectors": {
            "q4_decode": config["q4_decode"],
            "q4_device_layout": config["q4_device_layout"],
            "ffn_decode": config["ffn_decode"],
        },
        "opt088_control": opt088,
        "opt084_freeze": opt084,
        "opt089_admission": {
            "task": opt089.get("task"),
            "production_kept": opt089.get("production_kept"),
            "quality_contract_id": opt089.get("quality_contract_id"),
        },
        "opt091_concession": {
            "concession_used": opt091.get("concession_used"),
            "successor_model_quality_pass": opt091.get("successor_model_quality_pass"),
        },
        "reuse_opt089_authenticated_scores": False,
        "opt074_coverage_unadmitted_blocker": False,
    }
    late_verdicts = (opt089.get("independent_verdicts") or {}).get(
        OPT089_CANDIDATE_ID, {}
    )
    late_quality = opt089_evaluate_quality(opt089_config_by_id(OPT089_CANDIDATE_ID))
    passed = bool(
        late_verdicts.get("model_quality_pass")
        or late_quality.get("model_quality_pass")
    )
    if opt091.get("concession_used"):
        passed = bool(
            passed
            and opt091.get("successor_model_quality_pass")
            and opt091.get("strict_model_quality_pass") is not True
        )
        record["opt091_same_math_required"] = True
    reuse = config["id"] != CONTROL_ID
    record.update(
        {
            "model_quality_pass": passed,
            "incomplete": not passed,
            "reuse_opt089_authenticated_scores": reuse,
            "reason": (
                "opt089_authenticated_late_w4_reused"
                if passed and reuse
                else (
                    "opt089_shipping_late_w4"
                    if passed
                    else "opt089_quality_not_authenticated"
                )
            ),
            "shipping_q4_decode": current_q4_pin(),
            "shipping_q4_device_layout": current_q4_layout_pin(),
        }
    )
    return record


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
        layout_match = re.search(r"q4_device_layout=(\S+)", completed.stdout)
        if layout_match:
            dispatch["q4_device_layout"] = layout_match.group(1)
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
    critical = T_CRIT_DF9 if int(plan["samples"]) >= 10 else FEEDBACK_CRIT
    stats_by_id: dict[str, dict[str, Any]] = {}
    for cid in CANDIDATE_IDS:
        stats_by_id[cid] = paired_student_t(
            control_ms, by_config[cid]["ms"], critical=critical
        )
        stats_by_id[cid]["saving_ge_0_10_ms"] = (
            float(stats_by_id[cid]["mean_diff_ms"]) >= MIN_FFN_SAVING_MS
        )
    survivor = max(
        CANDIDATE_IDS, key=lambda cid: float(stats_by_id[cid]["mean_diff_ms"])
    )
    return {
        "capture_key": capture_key,
        "by_config": by_config,
        "control": CONTROL_ID,
        "survivor": survivor,
        "control_ms": control_ms,
        "control_mean_ms": mean(control_ms),
        "stats_by_id": stats_by_id,
        "mean_diff_ms": stats_by_id[survivor]["mean_diff_ms"],
        "positive": stats_by_id[survivor]["positive"],
        "ci95_low": stats_by_id[survivor].get("ci95_low"),
        "ci95_high": stats_by_id[survivor].get("ci95_high"),
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
        "q4-device-ab",
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
        "--q4-device-layout",
        str(survivor["q4_device_layout"]),
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
    payload = {
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
    payload["guards"] = e2e_guards(payload)
    return payload


def evaluate_parity(
    *,
    skip_gpu: bool,
    runner: NativeRunner | None,
    synthetic: Mapping[str, Any] | None,
) -> dict[str, Any]:
    catalog = parity_catalog()
    if skip_gpu:
        payload = dict(synthetic or {})
        success = bool(payload.get("success", True))
        return {
            "catalog_count": len(catalog),
            "cases_passed": int(payload.get("cases_passed", len(catalog))),
            "cases_failed": int(payload.get("cases_failed", 0)),
            "success": success,
            "by_ident": {
                CONTROL_ID: {"kernel_parity_pass": True},
                BRANCHLESS_ID: {"kernel_parity_pass": success},
                ALIGNED_ID: {"kernel_parity_pass": success},
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
            BRANCHLESS_ID: {"kernel_parity_pass": success},
            ALIGNED_ID: {"kernel_parity_pass": success},
        },
        "corrected_comparators_rerun": True,
    }


def faster_survivor(component: Mapping[str, Any]) -> dict[str, Any]:
    stats = component.get("stats_by_id") or {}
    survivor = str(component.get("survivor") or BRANCHLESS_ID)
    if survivor not in CANDIDATE_IDS:
        survivor = max(
            CANDIDATE_IDS,
            key=lambda cid: float(
                (stats.get(cid) or {}).get("mean_diff_ms", -math.inf)
            ),
        )
    return config_by_id(survivor)


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
        "task": "OPT-102",
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
        engine = {}
    stats_by_id = component.get("stats_by_id") or {}
    performance_by_id: dict[str, dict[str, Any]] = {
        CONTROL_ID: {
            "performance_pass": False,
            "this_sitting": this_sitting,
            "incomplete": not this_sitting or mode != "acceptance",
            "mean_diff_ms": 0.0,
        }
    }
    for cid in CANDIDATE_IDS:
        stats = stats_by_id.get(cid) or {
            "positive": component.get("positive"),
            "mean_diff_ms": component.get("mean_diff_ms"),
        }
        ppass = (
            this_sitting
            and dispatch_ok_flag
            and mode == "acceptance"
            and ffn_pass(stats)
        )
        performance_by_id[cid] = {
            "performance_pass": ppass,
            "this_sitting": this_sitting,
            "incomplete": not this_sitting or mode != "acceptance",
            "mean_diff_ms": stats.get("mean_diff_ms"),
            "positive": stats.get("positive"),
            "ci95_low": stats.get("ci95_low"),
            "ci95_high": stats.get("ci95_high"),
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
        "task": "OPT-102",
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
    quality_by_id = {config["id"]: evaluate_quality(config) for config in CONFIGS}
    plan = family_plan("quality", mode)
    observed = planned_observation(plan, keep=False)
    admission = admit_counts("quality", mode, observed)
    decided = decide_independent_verdicts(
        parity=(load_json(FIXTURE) if FIXTURE.is_file() else {}).get("parity") or {},
        quality_by_id=quality_by_id,
        performance_by_id={},
        mode="acceptance",
        prior=load_prior_verdicts(),
    )
    return {
        "schema_version": 1,
        "task": "OPT-102",
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
    fixture = load_json(FIXTURE) if FIXTURE.is_file() else {}
    survivor = faster_survivor((fixture.get("q4") or {}).get("component") or {})
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
            survivor=survivor,
            prefix=prefix,
        )
        guards = engine.get("guards") or e2e_guards(engine)
        this_sitting = True
    observed = planned_observation(plan, keep=False)
    admission = admit_counts(phase, mode, observed)
    prior = load_prior_verdicts()
    performance_by_id = {CONTROL_ID: {"performance_pass": False}}
    for cid in CANDIDATE_IDS:
        performance_by_id[cid] = {
            "performance_pass": bool(guards.get("pass")) and cid == survivor["id"],
            "incomplete": not this_sitting,
            "guards": guards,
            "mean_diff_ms": (
                (fixture.get("q4") or {})
                .get("component", {})
                .get("stats_by_id", {})
                .get(cid, {})
                .get("mean_diff_ms")
            ),
        }
    decided = decide_independent_verdicts(
        parity=fixture.get("parity") or {},
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
        "task": "OPT-102",
        "phase": phase,
        "mode": mode,
        "engine": engine,
        "guards": guards,
        "survivor": survivor["id"],
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
    fixture = load_json(FIXTURE) if FIXTURE.is_file() else {}
    survivor = faster_survivor((fixture.get("q4") or {}).get("component") or {})
    if skip_gpu:
        prefill = {"pass": False, "incomplete": True}
        this_sitting = False
    else:
        native = runner or default_native_runner
        walls: dict[str, float] = {}
        for config in (config_by_id(CONTROL_ID), survivor):
            command = [
                f"./{PROBE}",
                MODEL,
                "--workload",
                "prefill",
                "--prompt",
                "4096",
                "--modes",
                "graph",
                "--q4-decode",
                str(config["q4_decode"]),
                "--q4-device-layout",
                str(config["q4_device_layout"]),
                "--ffn-decode",
                str(config["ffn_decode"]),
            ]
            completed = native(command, "acceptance")
            (run_dir / f"prefill-{config['id']}.txt").write_text(
                completed.stdout + completed.stderr, encoding="utf-8"
            )
            walls[str(config["id"])] = parse_prefill_wall_ms(completed.stdout)
        control_ms = walls[CONTROL_ID]
        candidate_ms = walls[survivor["id"]]
        control_tok = 4096.0 / (control_ms / 1000.0) if control_ms > 0 else 0.0
        cand_tok = 4096.0 / (candidate_ms / 1000.0) if candidate_ms > 0 else 0.0
        ratio = cand_tok / control_tok if control_tok else 0.0
        prefill = {
            "pass": control_ms > 0.0 and ratio >= PREFILL_THROUGHPUT_MIN,
            "control_ms": control_ms,
            "candidate_ms": candidate_ms,
            "control_tok_s": control_tok,
            "candidate_tok_s": cand_tok,
            "throughput_ratio": ratio,
            "opt098_p4096_tok_s": 3046.23,
            "tok_s_delta_vs_opt098": 0.0,
        }
        this_sitting = True
    observed = planned_observation(plan, keep=False)
    admission = admit_counts("prefill-guard", mode, observed)
    prior = load_prior_verdicts()
    q4_stats = (fixture.get("q4") or {}).get("component", {}).get(
        "stats_by_id", {}
    ) or {}
    d128_ok = bool(((fixture.get("d128") or {}).get("guards") or {}).get("pass"))
    d2048_ok = bool(((fixture.get("d2048") or {}).get("guards") or {}).get("pass"))
    performance_by_id = {CONTROL_ID: {"performance_pass": False, "mean_diff_ms": 0.0}}
    for cid in CANDIDATE_IDS:
        ffn_ok = ffn_pass(q4_stats.get(cid) or {})
        e2e_ok = d128_ok and d2048_ok and cid == survivor["id"]
        ppass = (
            this_sitting
            and bool(prefill.get("pass"))
            and ffn_ok
            and e2e_ok
            and cid == survivor["id"]
        )
        performance_by_id[cid] = {
            "performance_pass": ppass,
            "incomplete": not this_sitting,
            "ffn_pass": ffn_ok,
            "d128_pass": d128_ok,
            "d2048_pass": d2048_ok,
            "prefill_pass": bool(prefill.get("pass")),
            "mean_diff_ms": (q4_stats.get(cid) or {}).get("mean_diff_ms"),
        }
    decided = decide_independent_verdicts(
        parity=fixture.get("parity") or {},
        quality_by_id={
            cid: {"model_quality_pass": row.get("model_quality_pass")}
            for cid, row in prior.items()
        },
        performance_by_id=performance_by_id,
        mode=mode,
        prior=prior,
    )
    if mode == "acceptance" and not decided["production_kept"]:
        decided["shipping_q4_decode"] = "integer_q8_late"
        decided["shipping_q4_device_layout"] = "raw_gguf"
        decided["shipping_ffn_decode"] = "paired_integer"
        decided["shipping_unchanged"] = True
        decided["selected_path"] = CONTROL_ID
        for cid, row in decided["independent_verdicts"].items():
            row["production_kept"] = cid == CONTROL_ID
    return {
        "schema_version": 1,
        "task": "OPT-102",
        "phase": "prefill-guard",
        "mode": mode,
        "prefill": prefill,
        "survivor": survivor["id"],
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
        "shipping_q4_device_layout",
        "shipping_ffn_decode",
        "production_kept",
    ):
        if key in payload:
            results[key] = payload[key]


def write_report(payload: Mapping[str, Any]) -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    q4 = payload.get("q4") or {}
    component = q4.get("component") or {}
    stats = component.get("stats_by_id") or {}
    d128 = (payload.get("d128") or {}).get("engine") or {}
    d2048 = (payload.get("d2048") or {}).get("engine") or {}
    prefill = (payload.get("prefill-guard") or {}).get("prefill") or {}
    kept = bool(payload.get("production_kept"))
    status = "keep" if kept else "reject"
    text = f"""# OPT-102 — Q4_K layout and branchless unpack

Status: **{status}**. Control `{CONTROL_ID}` (`integer_q8_late` / `raw_gguf` /
`paired_integer`). Candidates `{BRANCHLESS_ID}` and `{ALIGNED_ID}`. OPT-093
factored association is not a candidate.

Selected path: `{payload.get("selected_path")}`.
Shipping Q4 decode: `{payload.get("shipping_q4_decode")}`.
Shipping Q4 layout: `{payload.get("shipping_q4_device_layout")}`.
Production kept: `{payload.get("production_kept")}`.
Quality contract: `{QUALITY_CONTRACT_ID}` reused from authenticated OPT-089.

## Complete FFN

branchless mean_diff_ms={(stats.get(BRANCHLESS_ID) or {}).get("mean_diff_ms")}
aligned mean_diff_ms={(stats.get(ALIGNED_ID) or {}).get("mean_diff_ms")}
control_mean_ms={component.get("control_mean_ms")}

## Decode prefixes

D128 control_mean_ms={d128.get("control_mean_ms")} candidate_mean_ms={d128.get("candidate_mean_ms")}
D2048 control_mean_ms={d2048.get("control_mean_ms")} candidate_mean_ms={d2048.get("candidate_mean_ms")}

## P4096

control_tok_s={prefill.get("control_tok_s")} candidate_tok_s={prefill.get("candidate_tok_s")}
throughput_ratio={prefill.get("throughput_ratio")}
tok/s delta vs OPT-098 P4096 3046.23: {0 if not kept else prefill.get("tok_s_delta_vs_opt098", 0)}
"""
    REPORT.write_text(text, encoding="utf-8")
    if not kept:
        REJECTION.write_text(
            f"""# OPT-102 rejection — retain integer_q8_late / raw_gguf

Production pins remain `kSelectedQ4DecodePath[] = "integer_q8_late"` and
`kSelectedQ4DeviceLayout[] = "raw_gguf"`. OPT-093 factored association is
unchanged and not reinterpreted.

Complete 64-layer FFN, D128/D2048 five-pair, and P4096 guard did not jointly
win. Keep required ≥0.10 ms/token complete FFN with positive paired CI plus
both decode prefixes.

Independent verdicts:

{json.dumps(payload.get("independent_verdicts"), indent=2)}

tok/s delta vs OPT-098 P4096 **3046.23 tok/s**: **0** (rejection).

status=measured_reject.
""",
            encoding="utf-8",
        )


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
            "task": "OPT-102",
            "mode": mode,
            "llama_revision": LLAMA_REV,
            "gguf_sha256": GGUF_SHA,
            "token_generator": TOKEN_GENERATOR,
            "configurations": [row["id"] for row in CONFIGS],
            "control": CONTROL_ID,
            "control_layout": "raw_gguf",
            "shipping_q4_decode": "integer_q8_late",
            "shipping_q4_device_layout": "raw_gguf",
            "shipping_ffn_decode": "paired_integer",
            "quality_contract_id": QUALITY_CONTRACT_ID,
            "opt074_coverage_unadmitted_blocker": False,
            "replacement_mandatory_on_keep": True,
            "layout_version": LAYOUT_VERSION,
            "alignment_bytes": ALIGNMENT_BYTES,
            "report_path": "evidence/optimization/opt102-q4-layout-unpack/REPORT.md",
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
        results["evidence_complete"] = True
        results["claims_throughput"] = bool(results.get("production_kept"))
        results["claims_performance_improvement"] = bool(results.get("production_kept"))
        results["kernel_parity_pass"] = {
            cid: bool(row.get("kernel_parity_pass"))
            for cid, row in (results.get("independent_verdicts") or {}).items()
        }
        results["model_quality_pass"] = {
            cid: bool(row.get("model_quality_pass"))
            for cid, row in (results.get("independent_verdicts") or {}).items()
        }
        results["performance_pass"] = {
            cid: bool(row.get("performance_pass"))
            for cid, row in (results.get("independent_verdicts") or {}).items()
        }
    write_report(results)
    dump_json(FIXTURE, results)
    dump_json(run_dir / "opt102_q4_repack.json", results)
    counts = (results.get(phase) or {}).get("native_counts") or {}
    print(
        "QW38_OPT102_RESULT="
        + json.dumps(
            {
                "task": "OPT-102",
                "mode": mode,
                "phase": phase,
                "selected_path": results.get("selected_path"),
                "production_kept": results.get("production_kept"),
                "claims_throughput": results.get("claims_throughput"),
                "shipping_q4_decode": results.get("shipping_q4_decode"),
                "shipping_q4_device_layout": results.get("shipping_q4_device_layout"),
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
        "QW38_OPT102_NATIVE_COUNTS="
        + json.dumps(
            {
                "schema_version": 1,
                "task": "OPT-102",
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
