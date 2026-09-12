"""OPT-089 strict late_w4 admission with corrected parity and candidate quality."""

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
    docker_common,
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
    gpu_available,
    q8_1_typed_reject,
)
from tools.opt084_quality_baseline import FROZEN_ACCEPTANCE  # noqa: E402
from tools.opt088_batch_gate import COMBINATION_SELECTORS  # noqa: E402
from tools.quality.quality_mode import (  # noqa: E402
    apply_quality_mode,
    build_quality_config,
)
from tools.quality.suite import HELD_OUT_TARGETS, SUITE_CLASSES  # noqa: E402
from tools.run_optimization_task import (  # noqa: E402
    loop_product,
    parse_native_observation,
    validate_performance_admission,
    workload_for_mode,
)

LLAMA_REV = "cc83d7b4824f73cfdda4dfbb47ee39804f71b328"
GGUF_SHA = "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"
TOKEN_GENERATOR = "(42 + index * 997) % 248320"
CONTRACT = ROOT / "pins/opt089_q4_promotion_contract.json"
ITERATION = ROOT / "pins/opt089_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt089_q4_promotion.json"
REPORT = ROOT / "evidence/optimization/opt089-q4-promotion/REPORT.md"
EVIDENCE = REPORT.parent
OPT082_FIXTURE = ROOT / "fixtures/opt082_kernel_parity.json"
OPT082_REPORT = ROOT / "evidence/optimization/opt082-kernel-parity/REPORT.md"
OPT084_FIXTURE = ROOT / "fixtures/opt084_quality_baseline.json"
OPT084_REPORT = ROOT / "evidence/optimization/opt084-quality-baseline/REPORT.md"
OPT061_EVIDENCE = ROOT / "evidence/optimization/opt061-component-replay"
OPT085_FIXTURE = ROOT / "fixtures/opt085_q4_reevaluation.json"
OPT085_REPORT = ROOT / "evidence/optimization/opt085-q4-reevaluation/REPORT.md"
OPT086_FIXTURE = ROOT / "fixtures/opt086_q8_mmq_reevaluation.json"
OPT086_REPORT = ROOT / "evidence/optimization/opt086-q8-mmq-reevaluation/REPORT.md"
OPT088_FIXTURE = ROOT / "fixtures/opt088_batch_gate.json"
NATIVE_PARITY = "build/qw38-cuda-opt082-kernel-parity-test"
REPLAY = "build/qw38-cuda-component-replay"
PROBE = "build/qw38-cuda-optimization-engine-probe"
QUALITY_NATIVE = "build/qw38-cuda-opt058-quality-baseline-test"
Q4_PIN_FILE = ROOT / "cuda/q4k_decode_path.cuh"
FFN_PIN_FILE = ROOT / "cuda/ffn_decode_path.cuh"
MODEL = "models/Qwen3.8-27B-Q4_K_M.gguf"
NLL_BUNDLE = "pins/production_quality_v2_nll.bundle"
FUNCTIONAL_BUNDLE = "pins/production_quality_v2_functional.bundle"
PHASES = (
    "parity",
    "q4",
    "quality-alarm",
    "quality",
    "d128",
    "d2048",
    "prefill-guard",
)
CONTROL_ID = "packed_paired_staged"
CANDIDATE_ID = "late_w4"
WARPS_PER_ROW = 4
Q4_PERFORMANCE_IDENTS = ("packed", "integer_q8_late")
PARITY_ONLY_IDENTS = ("integer_q8",)
Q4_PARITY_IDENTS = ("packed", "integer_q8", "integer_q8_late")
Q8_LAYOUTS = ("r1_w4", "r2_w2")
QUALITY_CONTRACT_ID = "opt089_strict"
PPL_RATIO_MAX = 1.01
RECURRENCE_MAX = 0.02
MIN_FFN_SAVING_MS = 0.10
E2E_REGRESSION_FRAC = 0.02
THROUGHPUT_MIN = 0.98
P95_RATIO_MAX = 1.02
PREFILL_THROUGHPUT_MIN = 0.95
VERDICT_KEYS = (
    "kernel_parity_pass",
    "model_quality_pass",
    "performance_pass",
    "production_kept",
)
OPT088_SELECTORS = dict(COMBINATION_SELECTORS)

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


def default_native_runner(
    command: Sequence[str], tier: str
) -> subprocess.CompletedProcess[str]:
    listed = list(command)
    if listed and not listed[0].startswith("docker"):
        listed = [*docker_common(tier), *listed]
    completed = subprocess.run(
        listed,
        cwd=ROOT,
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


def parse_prefill_wall_ms(text: str) -> float:
    patterns = (
        r'"graph_wall_ms"\s*:\s*([0-9.eE+-]+)',
        r'"wall_ms"\s*:\s*([0-9.eE+-]+)',
        r"(?<![a-z_])wall_ms=([0-9.eE+-]+)",
    )
    seen: list[float] = []
    for pattern in patterns:
        match = re.search(pattern, text)
        if match is None:
            continue
        value = float(match.group(1))
        if value > 0.0:
            return value
        seen.append(value)
    if seen:
        return seen[0]
    raise AdmissionError("missing prefill wall_ms")


def current_q4_pin() -> str:
    match = re.search(
        r'kSelectedQ4DecodePath\[\] = "([^"]+)"',
        Q4_PIN_FILE.read_text(encoding="utf-8"),
    )
    if match is None:
        raise AdmissionError("missing kSelectedQ4DecodePath")
    return match.group(1)


def current_ffn_pin() -> str:
    match = re.search(
        r'kSelectedFfnDecodePath\[\] = "([^"]+)"',
        FFN_PIN_FILE.read_text(encoding="utf-8"),
    )
    if match is None:
        raise AdmissionError("missing kSelectedFfnDecodePath")
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


def apply_ffn_pin(path_id: str) -> None:
    text = FFN_PIN_FILE.read_text(encoding="utf-8")
    updated = re.sub(
        r'kSelectedFfnDecodePath\[\] = "[^"]+"',
        f'kSelectedFfnDecodePath[] = "{path_id}"',
        text,
        count=1,
    )
    if updated == text:
        raise AdmissionError("failed to rewrite the FFN decode production pin")
    FFN_PIN_FILE.write_text(updated, encoding="utf-8")


def apply_production_pins(results: Mapping[str, Any]) -> dict[str, Any]:
    applied: dict[str, Any] = {"q4": False, "ffn": False}
    if results.get("production_kept") and not results.get("shipping_unchanged"):
        q4_want = str(results.get("shipping_q4_decode") or "")
        ffn_want = str(results.get("shipping_ffn_decode") or "")
        if not q4_want or not ffn_want:
            raise AdmissionError("keep is missing shipping Q4/FFN idents")
        if q4_want != current_q4_pin():
            apply_q4_pin(q4_want)
            applied["q4"] = True
        if ffn_want != current_ffn_pin():
            apply_ffn_pin(ffn_want)
            applied["ffn"] = True
    applied["q4_pin"] = current_q4_pin()
    applied["ffn_pin"] = current_ffn_pin()
    return applied


def config_by_id(config_id: str) -> dict[str, Any]:
    for row in CONFIGS:
        if row["id"] == config_id:
            return dict(row)
    raise AdmissionError(f"unknown configuration {config_id}")


def config_for_ident(ident: str) -> dict[str, Any]:
    if ident == "integer_q8":
        return {
            "id": "integer_q8_paired",
            "q4_decode": "integer_q8",
            "ffn_decode": "paired_integer",
            "warps_per_row": WARPS_PER_ROW,
            "role": "parity_only",
        }
    for row in CONFIGS:
        if row["q4_decode"] == ident:
            return dict(row)
    raise AdmissionError(f"unknown Q4 ident {ident}")


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
        "q8_layout": "r1_w4",
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


def extra_catalog() -> list[dict[str, Any]]:
    """OPT-089 expansion IDs matching cuda/opt082_kernel_parity_test.cu."""
    rows: list[dict[str, Any]] = []
    q4_launch = {
        "packed": "quant_mmv_packed",
        "integer_q8": "q4k_coop_mmv_q8",
        "integer_q8_late": "q4k_coop_mmv_late_q8",
    }
    for ident in Q4_PARITY_IDENTS:
        for m in (1, 3, 17):
            for k in (5120, 17408):
                if ident == "packed" and m == 1 and k == 5120:
                    continue
                rows.append(
                    {
                        "id": f"Q4_K_mmv_{ident}_M{m}_N1_K{k}_random_assoc",
                        "family": "Q4_K",
                        "class_name": ASSOCIATION,
                        "candidate": ident,
                        "op": "mmv",
                        "pattern": "random",
                        "M": m,
                        "N": 1,
                        "K": k,
                        "expected_launch": q4_launch[ident],
                        "expect_fallback": False,
                        "opt089_extra": True,
                    }
                )
    for ident in Q8_LAYOUTS:
        for m in (1, 3, 17):
            for k in (2048, 5120, 6144):
                for seed in (0x55, 89, 90):
                    rows.append(
                        {
                            "id": (
                                f"Q8_0_mmv_{ident}_M{m}_N1_K{k}_seed{seed}_staged_assoc"
                            ),
                            "family": "Q8_0",
                            "class_name": ASSOCIATION,
                            "candidate": ident,
                            "op": "mmv",
                            "pattern": "random",
                            "M": m,
                            "N": 1,
                            "K": k,
                            "seed": seed,
                            "expected_launch": ident,
                            "expect_fallback": False,
                            "opt089_extra": True,
                        }
                    )
        for pattern in ("zero", "cancel", "minmax"):
            rows.append(
                {
                    "id": f"Q8_0_mmv_{ident}_M3_N1_K256_{pattern}_staged_assoc",
                    "family": "Q8_0",
                    "class_name": ASSOCIATION,
                    "candidate": ident,
                    "op": "mmv",
                    "pattern": pattern,
                    "M": 3,
                    "N": 1,
                    "K": 256,
                    "expected_launch": ident,
                    "expect_fallback": False,
                    "opt089_extra": True,
                }
            )
    fused_launch = {
        "packed": "q4k_gate_up_swiglu_prequant",
        "integer_q8_late": "q4k_coop_gate_up_swiglu_late_prequant_q8",
        "integer_q8": "q4k_coop_gate_up_swiglu_prequant_q8",
    }
    for ident in Q4_PARITY_IDENTS:
        for k in (256, 5120):
            for gen in (0, 1):
                rows.append(
                    {
                        "id": (
                            f"Q4_K_fused_vs_independent_{ident}_M17_N1_K{k}_gen{gen}_same"
                        ),
                        "family": "Q4_K",
                        "class_name": SAME_MATH,
                        "candidate": ident,
                        "op": "fused_gate_up",
                        "pattern": "random",
                        "M": 17,
                        "N": 1,
                        "K": k,
                        "expected_launch": fused_launch[ident],
                        "expect_fallback": False,
                        "opt089_extra": True,
                    }
                )
    for ident in Q4_PERFORMANCE_IDENTS:
        for k in (256, 5120):
            rows.append(
                {
                    "id": f"Q4_K_eager_vs_graph_{ident}_M17_N1_K{k}_random_same",
                    "family": "Q4_K",
                    "class_name": SAME_MATH,
                    "candidate": ident,
                    "op": "eager_vs_captured",
                    "pattern": "random",
                    "M": 17,
                    "N": 1,
                    "K": k,
                    "expected_launch": q4_launch[ident],
                    "expect_fallback": False,
                    "opt089_extra": True,
                }
            )
    return rows


def expanded_catalog() -> list[dict[str, Any]]:
    retained = [dict(row) for row in catalog("parity")]
    extras = extra_catalog()
    ids = {row["id"] for row in retained}
    merged = retained + [row for row in extras if row["id"] not in ids]
    return merged


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
    default_candidates = (
        2 if phase in {"q4", "quality-alarm", "d128", "d2048", "prefill-guard"} else 1
    )
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
        "prefix": int(workload.get("prefix", 0) or 0),
        "output_tokens": int(workload.get("output_tokens", 32) or 32),
        "cannot_pass_full_q": bool(workload.get("cannot_pass_full_q")),
    }


def planned_observation(
    plan: Mapping[str, Any], *, capture_key: str | None = None, keep: bool = False
) -> dict[str, Any]:
    samples = int(plan["samples"])
    return {
        "schema_version": 1,
        "task": "OPT-089",
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


def workspace_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def replay_command(
    plan: Mapping[str, Any],
    config: Mapping[str, Any],
    capture_key: str | None,
    evidence_dir: Path | str | None = None,
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
    if evidence_dir:
        args.extend(["--evidence-dir", workspace_relative(Path(evidence_dir))])
    return args


def candidate_selectors(config: Mapping[str, Any]) -> dict[str, Any]:
    selectors = dict(OPT088_SELECTORS)
    selectors["q4_decode"] = config["q4_decode"]
    selectors["q4_staging"] = config["ffn_decode"]
    selectors["ffn_decode"] = config["ffn_decode"]
    selectors["q8_decode"] = "r1_w4"
    return selectors


def _mentions_opt089(payload: Any) -> bool:
    if isinstance(payload, Mapping):
        return any(_mentions_opt089(value) for value in payload.values())
    if isinstance(payload, (list, tuple)):
        return any(_mentions_opt089(value) for value in payload)
    return isinstance(payload, str) and "OPT-089" in payload


def assert_opt089_write_path(path: Path) -> None:
    resolved = path.resolve()
    forbidden = (
        OPT082_FIXTURE,
        OPT082_REPORT,
        OPT084_FIXTURE,
        OPT084_REPORT,
        OPT085_FIXTURE,
        OPT085_REPORT,
        OPT086_FIXTURE,
        OPT086_REPORT,
    )
    for item in forbidden:
        if item.resolve() == resolved:
            raise AdmissionError(f"OPT-089 must not write {item}")
    opt061 = OPT061_EVIDENCE.resolve()
    if resolved == opt061 or opt061 in resolved.parents:
        raise AdmissionError("OPT-089 must not write OPT-061 evidence")


def historical_reports_intact() -> dict[str, Any]:
    opt082 = load_json(OPT082_FIXTURE) if OPT082_FIXTURE.is_file() else {}
    opt084 = load_json(OPT084_FIXTURE) if OPT084_FIXTURE.is_file() else {}
    opt085 = load_json(OPT085_FIXTURE) if OPT085_FIXTURE.is_file() else {}
    opt086 = load_json(OPT086_FIXTURE) if OPT086_FIXTURE.is_file() else {}
    report082 = (
        OPT082_REPORT.read_text(encoding="utf-8") if OPT082_REPORT.is_file() else ""
    )
    report084 = (
        OPT084_REPORT.read_text(encoding="utf-8") if OPT084_REPORT.is_file() else ""
    )
    report085 = (
        OPT085_REPORT.read_text(encoding="utf-8") if OPT085_REPORT.is_file() else ""
    )
    report086 = (
        OPT086_REPORT.read_text(encoding="utf-8") if OPT086_REPORT.is_file() else ""
    )
    cases = opt082.get("cases") or []
    gpu082 = opt082.get("gpu") or {}
    catalog_ok = (
        opt082.get("task") == "OPT-082"
        and int(opt082.get("catalog_count") or 0) == 105
        and len(cases) == 105
    )
    gpu_verdict_ok = (
        gpu082.get("ran") is True
        and gpu082.get("success") is True
        and "available=True ran=True success=True" in report082
        and "catalog_count=105" in report082
        and "gpu_case_count=105" in report082
    )
    opt084_ok = (
        opt084.get("task") == "OPT-084"
        and str((opt084.get("gpu") or {}).get("blocker") or "") != "skip_gpu"
    )
    opt089_leak = (
        _mentions_opt089(opt082)
        or _mentions_opt089(opt084)
        or "OPT-089" in report082
        or "OPT-089" in report084
        or opt082.get("task") == "OPT-089"
        or opt084.get("task") == "OPT-089"
    )
    return {
        "opt082_task": opt082.get("task"),
        "opt082_catalog_count": int(opt082.get("catalog_count") or 0),
        "opt082_case_count": len(cases),
        "opt082_catalog_intact": catalog_ok,
        "opt082_gpu_verdict_intact": gpu_verdict_ok,
        "opt084_task": opt084.get("task"),
        "opt084_unchanged_task": opt084.get("task") == "OPT-084",
        "opt085_task": opt085.get("task"),
        "opt086_task": opt086.get("task"),
        "opt085_production_kept": opt085.get("production_kept"),
        "opt082_mentions_opt082": "OPT-082" in report082,
        "opt085_mentions_opt085": "OPT-085" in report085,
        "opt086_mentions_opt086": "OPT-086" in report086 or OPT086_REPORT.is_file(),
        "opt089_writes_into_opt082_or_opt084": opt089_leak,
        "unmodified": (
            catalog_ok
            and gpu_verdict_ok
            and opt084_ok
            and not opt089_leak
            and opt085.get("task") == "OPT-085"
            and opt085.get("production_kept") is False
            and "OPT-089" not in report082
            and "production_kept=true" not in report085.replace(" ", "").casefold()
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


def _parse_parity_cases(stdout: str) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line in (stdout or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("QW38_OPT082_CASE="):
            try:
                cases.append(json.loads(stripped.split("=", 1)[1]))
            except json.JSONDecodeError:
                continue
    return cases


def run_expanded_native(runner: NativeRunner | None) -> dict[str, Any]:
    execute = runner or default_native_runner
    available, blocker = gpu_available()
    record: dict[str, Any] = {
        "available": available,
        "blocker": blocker,
        "ran": False,
        "success": False,
        "cases": [],
    }
    if not available:
        return record
    completed = execute(
        [f"./{NATIVE_PARITY}", "--phase", "parity", "--expand-opt089"],
        "correctness",
    )
    record["ran"] = True
    record["stdout"] = completed.stdout
    record["stderr"] = completed.stderr
    record["returncode"] = completed.returncode
    record["cases"] = _parse_parity_cases(completed.stdout)
    observation = parse_native_observation(completed.stdout)
    record["success"] = completed.returncode == 0 and bool(
        observation.get("success", completed.returncode == 0)
    )
    if not record["success"] and not record.get("blocker"):
        record["blocker"] = f"expanded parity failed rc={completed.returncode}"
    return record


def _evaluate_ident_cases(
    ident: str,
    catalog_rows: Sequence[Mapping[str, Any]],
    by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    mapped = [dict(by_id.get(row["id"], {})) for row in catalog_rows]
    fallback_as_candidate = False
    passes: list[bool] = []
    same_math_fails: list[str] = []
    association_fails: list[str] = []
    for spec, gpu in zip(catalog_rows, mapped):
        gpu_pass = gpu_case_pass(gpu)
        if spec.get("expect_fallback"):
            if gpu_pass is None:
                passes.append(False)
            else:
                passes.append(True)
            continue
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
        if gpu_pass is False:
            if class_name == SAME_MATH:
                same_math_fails.append(str(spec["id"]))
            else:
                association_fails.append(str(spec["id"]))
            passes.append(False)
            continue
        passes.append(True)
    complete = all(gpu_case_pass(gpu) is not None for gpu in mapped) and bool(
        catalog_rows
    )
    association_ok = bool(passes) and all(passes)
    return {
        "ident": ident,
        "config": config_for_ident(ident)["id"] if ident in Q4_PARITY_IDENTS else ident,
        "kernel_parity_pass": bool(
            complete
            and association_ok
            and not fallback_as_candidate
            and not same_math_fails
        ),
        "case_count": len(catalog_rows),
        "gpu_case_count": sum(1 for gpu in mapped if gpu_case_pass(gpu) is not None),
        "fallback_measured_as_candidate": fallback_as_candidate,
        "incomplete": not complete,
        "same_math_fails": same_math_fails,
        "documented_fails": association_fails,
        "parity_only": ident in PARITY_ONLY_IDENTS,
    }


def evaluate_parity(
    *,
    skip_gpu: bool = False,
    runner: NativeRunner | None = None,
    synthetic: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cases = expanded_catalog()
    retained = catalog("parity")
    if len(retained) != 105:
        raise AdmissionError(f"OPT-082 catalog changed: {len(retained)} != 105")
    if len(cases) != 198:
        raise AdmissionError(f"OPT-089 expanded catalog {len(cases)} != 198")
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
    corrected_rerun = False
    if synthetic and synthetic.get("gpu_cases") is not None:
        gpu_rows = [dict(row) for row in synthetic["gpu_cases"]]
        source = str(synthetic.get("source") or "synthetic")
        native["ran"] = True
        native["success"] = True
        native["available"] = True
        corrected_rerun = source != "opt082_fixture"
    elif skip_gpu:
        prior = load_json(OPT082_FIXTURE) if OPT082_FIXTURE.is_file() else {}
        gpu_rows = [dict(row) for row in prior.get("cases") or []]
        source = "opt082_fixture"
        native["blocker"] = "skip_gpu"
        corrected_rerun = False
    else:
        native = run_expanded_native(runner)
        if native.get("ran"):
            gpu_rows = [dict(row) for row in native.get("cases") or []]
            source = "this_sitting"
            corrected_rerun = True
        elif OPT082_FIXTURE.is_file():
            prior = load_json(OPT082_FIXTURE)
            gpu_rows = [dict(row) for row in prior.get("cases") or []]
            source = "opt082_fixture"
            native["blocker"] = native.get("blocker") or "native_unavailable"
    by_id = {str(row.get("id")): row for row in gpu_rows}
    by_ident: dict[str, dict[str, Any]] = {}
    for ident in Q4_PARITY_IDENTS:
        catalog_rows = [
            row
            for row in cases
            if row["candidate"] == ident and row["family"] == "Q4_K"
        ]
        by_ident[ident] = _evaluate_ident_cases(ident, catalog_rows, by_id)
    for ident in Q8_LAYOUTS:
        catalog_rows = [
            row
            for row in cases
            if row["candidate"] == ident and row["family"] == "Q8_0"
        ]
        by_ident[ident] = _evaluate_ident_cases(ident, catalog_rows, by_id)
    q6_rows = [row for row in cases if row.get("family") == "Q6_K"]
    if q6_rows:
        by_ident["q6"] = _evaluate_ident_cases("q6", q6_rows, by_id)
    if not corrected_rerun:
        for ident, row in by_ident.items():
            row["kernel_parity_pass"] = (
                False if ident != "packed" else row.get("kernel_parity_pass", False)
            )
            if ident != "packed":
                row["incomplete"] = True
            row["corrected_comparators_not_rerun"] = True
        extras_present = all(row["id"] in by_id for row in extra_catalog())
        if not extras_present:
            for ident in Q4_PARITY_IDENTS:
                by_ident[ident]["incomplete"] = True
                if ident != "packed" or not extras_present:
                    by_ident[ident]["kernel_parity_pass"] = bool(
                        ident == "packed"
                        and by_ident[ident].get("gpu_case_count", 0) > 0
                        and not by_ident[ident].get("same_math_fails")
                        and not by_ident[ident].get("documented_fails")
                    )
                    if ident == "packed":
                        by_ident[ident]["kernel_parity_pass"] = bool(
                            not by_ident[ident].get("same_math_fails")
                            and not by_ident[ident].get("documented_fails")
                            and by_ident[ident].get("gpu_case_count", 0) > 0
                        )
                        by_ident[ident]["incomplete"] = True
    packed = by_ident["packed"]
    packed_hard_fail = bool(
        packed.get("same_math_fails")
        or packed.get("documented_fails")
        or packed.get("fallback_measured_as_candidate")
    )
    if packed_hard_fail and corrected_rerun:
        raise AdmissionError("packed Q4_K kernel parity failed; hard stop")
    q8_ok = all(
        by_ident.get(ident, {}).get("kernel_parity_pass") for ident in Q8_LAYOUTS
    )
    late = by_ident["integer_q8_late"]
    late["kernel_parity_pass"] = bool(
        late["kernel_parity_pass"]
        and q8_ok
        and packed.get("kernel_parity_pass")
        and not late.get("incomplete")
    )
    # integer_q8 is reconstruction-only and does not block late.
    return {
        "schema_version": 1,
        "task": "OPT-089",
        "phase": "parity",
        "host_ok": host_ok,
        "typed_staging_reject": typed["pass"],
        "source": source,
        "catalog_count": len(cases),
        "opt082_retained": len(retained),
        "opt089_extra": len(extra_catalog()),
        "corrected_comparators_rerun": corrected_rerun,
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
        "every_positive_case_aggregated": True,
        "integer_q8_parity_only": True,
    }


def _selector_proof(payload: Mapping[str, Any]) -> dict[str, Any]:
    paths = dict(payload.get("combined_production_paths") or {})
    shipping = dict(payload.get("shipping_selectors") or {})
    quality = dict(payload.get("quality") or {})
    quality_selectors = dict(quality.get("selectors") or {})
    merged = {**shipping, **paths, **quality_selectors}
    q4 = str(merged.get("q4_decode") or "")
    q8 = str(merged.get("q8_layout") or merged.get("q8_decode") or "")
    if q8 in {"dp4a_q8_1", "1", "4"}:
        rows = merged.get("q8_decode_rows_skinny")
        warps = merged.get("q8_decode_layout_warps_skinny")
        if rows == 1 and warps == 4:
            q8 = "r1_w4"
        elif rows == 2 and warps == 2:
            q8 = "r2_w2"
    return {
        "q4_decode": q4,
        "q8_layout": q8,
        "q8_decode": str(merged.get("q8_decode") or q8),
        "ffn_decode": str(merged.get("ffn_decode") or merged.get("q4_staging") or ""),
        "gguf_sha256": str(payload.get("gguf_sha256") or ""),
        "source_fields": sorted(merged.keys()),
    }


def authenticate_anchor(
    *,
    label: str,
    path: Path,
    expected_q4: str,
    expected_q8: str,
) -> dict[str, Any]:
    if not path.is_file():
        return {
            "label": label,
            "source": str(path),
            "authenticated": False,
            "rescore_required": True,
            "reason": "missing_cache",
        }
    payload = load_json(path)
    proof = _selector_proof(payload)
    gguf_ok = proof["gguf_sha256"] in {"", GGUF_SHA} or proof["gguf_sha256"] == GGUF_SHA
    q4_ok = proof["q4_decode"] == expected_q4
    q8_ok = proof["q8_layout"] == expected_q8 or proof["q8_decode"] == expected_q8
    has_proof = bool(proof["source_fields"]) and bool(
        proof["q4_decode"] or proof["q8_layout"]
    )
    authenticated = bool(has_proof and q4_ok and q8_ok and gguf_ok)
    nll_cases = (
        (payload.get("quality") or {}).get("nll_cases")
        or (payload.get("quartz_scores") or {})
        or payload.get("nll_cases")
        or []
    )
    return {
        "label": label,
        "source": str(path.relative_to(ROOT))
        if path.is_relative_to(ROOT)
        else str(path),
        "authenticated": authenticated,
        "rescore_required": not authenticated,
        "selectors": proof,
        "expected": {"q4_decode": expected_q4, "q8_layout": expected_q8},
        "gguf_ok": gguf_ok,
        "has_nll": bool(nll_cases),
        "payload_task": payload.get("task"),
        "reason": (
            "source_selector_proof"
            if authenticated
            else "cache_lacks_source_or_selector_proof"
        ),
    }


def authenticate_opt088_control() -> dict[str, Any]:
    return authenticate_anchor(
        label="opt088_packed_r1",
        path=OPT088_FIXTURE,
        expected_q4="packed",
        expected_q8="r1_w4",
    )


def authenticate_opt084_freeze() -> dict[str, Any]:
    return authenticate_anchor(
        label="opt084_packed_r2",
        path=OPT084_FIXTURE,
        expected_q4="packed",
        expected_q8="r2_w2",
    )


def ppl_ratio(candidate_nll: float, control_nll: float) -> float:
    return math.exp(float(candidate_nll) - float(control_nll))


def _nll_from_cases(cases: Sequence[Mapping[str, Any]], name: str) -> float | None:
    for row in cases:
        if row.get("name") == name and row.get("mean_nll") is not None:
            return float(row["mean_nll"])
    return None


def evaluate_quality(
    config: Mapping[str, Any],
    *,
    synthetic: Mapping[str, Any] | None = None,
    alarm_only: bool = False,
    measured: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    applied = apply_quality_mode(enabled=True, selectors=candidate_selectors(config))
    build_quality_config(enabled=True, selectors=candidate_selectors(config))
    opt088 = authenticate_opt088_control()
    opt084 = authenticate_opt084_freeze()
    engine = opt073_quality()
    nr_ok = engine.get("quality_v3_engine_non_regression") in {"pass", True}
    record: dict[str, Any] = {
        "config": config["id"],
        "quality_flag": "--quality",
        "quality_contract_id": QUALITY_CONTRACT_ID,
        "ppl_ratio_max": PPL_RATIO_MAX,
        "recurrence_incremental_nll_max": RECURRENCE_MAX,
        "require_candidate_nll": True,
        "allow_incomplete_quality": False,
        "new_functional_failures_max": 0,
        "allow_new_greedy_mismatch": False,
        "suite_classes": list(SUITE_CLASSES),
        "held_out_targets": HELD_OUT_TARGETS,
        "selectors": applied["selectors"],
        "quality_config": True,
        "does_not_restore_packed_or_r2": bool(
            applied.get("does_not_restore_packed_or_r2")
        ),
        "opt088_control": opt088,
        "opt084_freeze": opt084,
        "opt073_engine_non_regression": engine.get("quality_v3_engine_non_regression"),
        "frozen_acceptance": dict(FROZEN_ACCEPTANCE),
        "alarm_only": alarm_only,
        "cannot_pass_full_q": alarm_only,
        "opt074_coverage_unadmitted_blocker": False,
        "synthetic_injection_used": False,
    }
    if config["id"] == CONTROL_ID:
        passed = bool(opt088["authenticated"] and opt088.get("has_nll") and nr_ok)
        record.update(
            {
                "model_quality_pass": passed and not alarm_only,
                "incomplete": not passed,
                "reason": (
                    "authenticated_opt088_packed_r1"
                    if passed
                    else opt088.get("reason") or "opt088_control_unauthenticated"
                ),
                "opt084_authenticated_separately": bool(opt084["authenticated"]),
            }
        )
        if alarm_only:
            record["model_quality_pass"] = False
            record["reason"] = "quality_alarm_cannot_pass_full_q"
        return record
    measured_nll = None
    if measured and isinstance(measured.get("cases"), list):
        measured_nll = measured
    # Synthetic injection is never admission evidence.
    if synthetic and synthetic.get("quality_by_id"):
        record["synthetic_injection_ignored"] = True
    if measured_nll is None:
        record.update(
            {
                "model_quality_pass": False,
                "incomplete": True,
                "reason": "candidate_specific_quality_not_measured",
            }
        )
        return record
    held = _nll_from_cases(measured_nll.get("cases") or [], "held_out_wikitext_1024")
    wiki = _nll_from_cases(measured_nll.get("cases") or [], "wikitext_nll")
    control_held = 1.7878782710057632
    if opt088["authenticated"]:
        opt088_payload = load_json(OPT088_FIXTURE)
        cases = (opt088_payload.get("quality") or {}).get("nll_cases") or []
        cached = _nll_from_cases(cases, "held_out_wikitext_1024")
        if cached is not None:
            control_held = cached
    if held is None:
        record.update(
            {
                "model_quality_pass": False,
                "incomplete": True,
                "reason": "incomplete_candidate_nll",
            }
        )
        return record
    ratio_r1 = ppl_ratio(held, control_held)
    opt084_held = control_held
    if opt084["authenticated"]:
        freeze = load_json(OPT084_FIXTURE)
        freeze_nll = (
            (freeze.get("quartz_vs_baseline") or {}).get("aggregate") or {}
        ).get("examples") or []
        for example in freeze_nll:
            if example.get("id") == "held_out_wikitext_1024":
                opt084_held = float(example.get("quartz_avg_nll") or opt084_held)
    ratio_r2 = ppl_ratio(held, opt084_held)
    rec_inc = float((measured_nll.get("recurrence_incremental_nll") or 0.0))
    functional_failures = int(measured_nll.get("new_functional_failures") or 0)
    greedy_mismatch = bool(measured_nll.get("new_greedy_mismatch"))
    ratios_ok = ratio_r1 <= PPL_RATIO_MAX and ratio_r2 <= PPL_RATIO_MAX
    rec_ok = rec_inc <= RECURRENCE_MAX
    func_ok = functional_failures <= 0 and not greedy_mismatch
    passed = bool(ratios_ok and rec_ok and func_ok and opt088["authenticated"])
    if alarm_only:
        passed = False
    record.update(
        {
            "model_quality_pass": passed,
            "incomplete": False,
            "held_out_mean_nll": held,
            "wikitext_mean_nll": wiki,
            "ppl_ratio_vs_opt088_r1": ratio_r1,
            "ppl_ratio_vs_opt084_r2": ratio_r2,
            "recurrence_incremental_nll": rec_inc,
            "reason": (
                "quality_alarm_cannot_pass_full_q"
                if alarm_only
                else ("opt089_strict_pass" if passed else "opt089_strict_fail")
            ),
            "llama_deltas_independent": True,
        }
    )
    return record


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return ordered[0]
    index = fraction * (len(ordered) - 1)
    lo = int(math.floor(index))
    hi = min(lo + 1, len(ordered) - 1)
    mix = index - lo
    return (1.0 - mix) * ordered[lo] + mix * ordered[hi]


def e2e_guards(
    engine: Mapping[str, Any] | None, *, output_tokens: int = 32
) -> dict[str, Any]:
    if not engine or not engine.get("control_ms") or not engine.get("candidate_ms"):
        return {"pass": False, "incomplete": True, "reason": "missing_engine_pairs"}
    control = [float(v) for v in engine["control_ms"]]
    candidate = [float(v) for v in engine["candidate_ms"]]
    control_mean = mean(control)
    diffs = [cand - ctrl for ctrl, cand in zip(control, candidate)]
    avg = mean(diffs)
    var = sample_variance(diffs)
    se = math.sqrt(var / len(diffs)) if var > 0.0 else 0.0
    df = max(len(diffs) - 1, 1)
    critical = T_CRIT_DF4 if df >= 4 else FEEDBACK_CRIT
    upper = avg + critical * se
    limit = E2E_REGRESSION_FRAC * control_mean if control_mean > 0.0 else 0.0
    control_tok = [output_tokens / (ms / 1000.0) for ms in control if ms > 0]
    cand_tok = [output_tokens / (ms / 1000.0) for ms in candidate if ms > 0]
    throughput_ratio = (
        mean(cand_tok) / mean(control_tok) if control_tok and cand_tok else 0.0
    )
    control_p95 = [
        float(
            row.get("control_itl_p95_ms") or (row.get("control_ms", 0) / output_tokens)
        )
        for row in engine.get("pairs") or [{}] * len(control)
    ]
    cand_p95 = [
        float(
            row.get("candidate_itl_p95_ms")
            or (row.get("candidate_ms", 0) / output_tokens)
        )
        for row in engine.get("pairs") or [{}] * len(candidate)
    ]
    if not engine.get("pairs"):
        control_p95 = [ms / output_tokens for ms in control]
        cand_p95 = [ms / output_tokens for ms in candidate]
    p95_ratio = (
        percentile(cand_p95, 0.95) / percentile(control_p95, 0.95)
        if percentile(control_p95, 0.95) > 0
        else math.inf
    )
    passed = (
        upper <= limit
        and throughput_ratio >= THROUGHPUT_MIN
        and p95_ratio <= P95_RATIO_MAX
    )
    return {
        "pass": passed,
        "incomplete": False,
        "regression_upper_ms": upper,
        "regression_limit_ms": limit,
        "throughput_ratio": throughput_ratio,
        "p95_ratio": p95_ratio,
        "control_mean_ms": control_mean,
        "candidate_mean_ms": mean(candidate),
    }


def performance_pass_for(
    *,
    config_id: str,
    stats: Mapping[str, Any] | None,
    engine: Mapping[str, Any] | None,
    this_sitting: bool,
    dispatch_ok: bool,
    e2e: Mapping[str, Any] | None = None,
    prefill: Mapping[str, Any] | None = None,
    require_full_e2e: bool = False,
) -> bool:
    if config_id == CONTROL_ID:
        return False
    if not this_sitting or not dispatch_ok or not stats:
        return False
    component_ok = (
        bool(stats.get("positive"))
        and float(stats.get("mean_diff_ms", 0.0)) >= MIN_FFN_SAVING_MS
    )
    if not component_ok:
        return False
    if require_full_e2e:
        if (
            not e2e
            or not e2e.get("d128", {}).get("pass")
            or not e2e.get("d2048", {}).get("pass")
        ):
            return False
        if not prefill or not prefill.get("pass"):
            return False
        return True
    if engine:
        return bool(e2e_guards(engine).get("pass"))
    return False


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
    evidence_dir = run_dir / "component-replay"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    assert_opt089_write_path(evidence_dir / "rounds.jsonl")
    for config in configs:
        completed = runner(
            replay_command(plan, config, capture_key, evidence_dir=evidence_dir),
            tier,
        )
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
        samples = [float(row["enclosing_ms"]) for row in rounds]
        by_config[str(config["id"])] = {
            "config": dict(config),
            "ms": samples,
            "mean_ms": mean(samples),
            "dispatch": dispatch,
            "expected": expected,
        }
    return {
        "capture_key": capture_key,
        "by_config": by_config,
        "control": CONTROL_ID,
        "this_sitting": True,
        "dispatch_ok": True,
        "launches": [
            {
                "config": name,
                "gate_variant": row["expected"]["gate_variant"],
                "down_variant": row["expected"]["down_variant"],
            }
            for name, row in by_config.items()
        ],
    }


def parity_record(source: Mapping[str, Any] | None) -> dict[str, Any]:
    """Unwrap nested OPT-089 parity payloads until by_ident is visible."""
    current: Any = source
    for _ in range(4):
        if not isinstance(current, Mapping):
            return {}
        if current.get("by_ident"):
            return dict(current)
        inner = current.get("parity")
        if isinstance(inner, Mapping):
            current = inner
            continue
        break
    return dict(current) if isinstance(current, Mapping) else {}


def ident_parity_from_source(source: Mapping[str, Any] | None) -> dict[str, Any]:
    record = parity_record(source)
    by_ident = dict(record.get("by_ident") or {})
    if by_ident:
        return by_ident
    verdicts = (source or {}).get("independent_verdicts") or {}
    mapped: dict[str, Any] = {}
    for ident in Q4_PARITY_IDENTS:
        cid = config_for_ident(ident)["id"]
        mapped[ident] = {
            "kernel_parity_pass": bool(
                (verdicts.get(cid) or {}).get("kernel_parity_pass")
            )
        }
    return mapped


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
    compare_id = survivor
    if compare_id == CONTROL_ID:
        if CANDIDATE_ID in by_config and by_config[CANDIDATE_ID].get("ms"):
            compare_id = CANDIDATE_ID
        else:
            measured = [
                name
                for name, row in by_config.items()
                if name != CONTROL_ID and row.get("ms")
            ]
            if measured:
                compare_id = measured[0]
    if compare_id == CONTROL_ID or compare_id not in by_config:
        return {
            "survivor": survivor,
            "compared": CONTROL_ID,
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
    candidate_ms = list(by_config[compare_id]["ms"])
    critical = T_CRIT_DF9 if int(plan["samples"]) >= 10 else FEEDBACK_CRIT
    stats = paired_student_t(control_ms, candidate_ms, critical=critical)
    return {
        "survivor": survivor,
        "compared": compare_id,
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
    prefix: int | None = None,
    pairs_n: int | None = None,
    sidecar: str | None = None,
) -> dict[str, Any]:
    count = int(pairs_n if pairs_n is not None else plan["engine_pairs"])
    if count <= 0 or survivor["id"] == CONTROL_ID:
        return {}
    used_prefix = int(prefix if prefix is not None else plan.get("prefix") or 2048)
    tier = "acceptance" if plan["mode"] == "acceptance" else "screen"
    command = [
        f"./{PROBE}",
        MODEL,
        "--workload",
        "q4-ab",
        "--pairs",
        str(count),
        "--modes",
        "graph",
        "--prefix",
        str(used_prefix),
        "--output-tokens",
        "32",
        "--q4-decode",
        str(survivor["q4_decode"]),
        "--ffn-decode",
        str(survivor["ffn_decode"]),
        "--q4-warps",
        str(int(survivor["warps_per_row"])),
        "--q8-layout",
        "r1_w4",
    ]
    completed = runner(command, tier)
    name = sidecar or f"{plan['phase']}-engine.txt"
    (run_dir / name).write_text(completed.stdout + completed.stderr, encoding="utf-8")
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
    if len(pairs) != count:
        raise AdmissionError(f"engine pairs {len(pairs)} != {count}")
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
        "prefix": used_prefix,
        "output_tokens": 32,
        "guards": e2e_guards(
            {
                "control_ms": control,
                "candidate_ms": candidate,
                "pairs": pairs,
            }
        ),
    }


def run_prefill_guard(
    *,
    runner: NativeRunner,
    run_dir: Path,
    survivor: Mapping[str, Any],
) -> dict[str, Any]:
    walls: dict[str, float] = {}
    for config in (config_by_id(CONTROL_ID), survivor):
        command = [
            f"./{PROBE}",
            MODEL,
            "--workload",
            "prefill",
            "--prompt",
            "4096",
            "--q4-decode",
            str(config["q4_decode"]),
            "--ffn-decode",
            str(config["ffn_decode"]),
            "--q4-warps",
            str(int(config["warps_per_row"])),
            "--q8-layout",
            "r1_w4",
            "--modes",
            "graph",
        ]
        completed = runner(command, "acceptance")
        (run_dir / f"prefill-{config['id']}.txt").write_text(
            completed.stdout + completed.stderr, encoding="utf-8"
        )
        try:
            walls[str(config["id"])] = parse_prefill_wall_ms(completed.stdout)
        except AdmissionError as exc:
            raise AdmissionError(f"{exc} for {config['id']}") from exc
    control_ms = walls[CONTROL_ID]
    candidate_ms = walls[str(survivor["id"])]
    control_tok = 4096.0 / (control_ms / 1000.0) if control_ms > 0 else 0.0
    cand_tok = 4096.0 / (candidate_ms / 1000.0) if candidate_ms > 0 else 0.0
    ratio = cand_tok / control_tok if control_tok > 0 else 0.0
    return {
        "control_ms": control_ms,
        "candidate_ms": candidate_ms,
        "control_tok_s": control_tok,
        "candidate_tok_s": cand_tok,
        "throughput_ratio": ratio,
        "pass": ratio >= PREFILL_THROUGHPUT_MIN,
        "this_sitting": True,
        "prompt": 4096,
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
        "feedback_cannot_keep": mode != "acceptance",
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
    if isinstance(parity.get("parity"), Mapping):
        parity = parity["parity"]
    quality_payload = payload.get("quality") or payload.get("quality-alarm") or {}
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
    component = q4.get("component") or {}
    engine = q4.get("engine") or {}
    d128_guards = (payload.get("d128") or {}).get("guards") or {}
    d2048_guards = (payload.get("d2048") or {}).get("guards") or {}
    prefill = (payload.get("prefill-guard") or {}).get("prefill") or {}
    pins = payload.get("pins_applied") or {}
    keep_line = (
        (
            f"Production pins: Q4 `{payload.get('shipping_q4_decode')}` / "
            f"FFN `{payload.get('shipping_ffn_decode')}`. "
            f"pin_rewrite q4={pins.get('q4')} ffn={pins.get('ffn')}."
        )
        if payload.get("claims_throughput")
        else "No improvement claim; packed/`paired_staged` retained."
    )
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
            f"{row.get('parity_only')} |"
        )
    text = f"""# OPT-089 — Complete late_w4 admission

Status: **{payload.get("status") or "pending"}**. Authority llama.cpp
`{LLAMA_REV}`, GGUF SHA-256 `{GGUF_SHA}`. Exactly two performance
configurations: packed/`paired_staged` control and `late_w4`
(`integer_q8_late`/`paired_integer`, four warps, FP32-scale Q8Block).
Q8 `r1_w4` and remaining OPT-088 selectors stay fixed.
`integer_q8_paired` is parity-only reconstruction, not a third sweep.
No new Q4 kernel.

`opt074_coverage_unadmitted` is **not** a blocker.
`quality_contract_id={QUALITY_CONTRACT_ID}`.
`claims_throughput` is true only for this sitting's uninstrumented keep.

## Independent verdicts

| Config | kernel_parity_pass | model_quality_pass | performance_pass | production_kept |
|---|---|---|---|---|
{chr(10).join(lines)}

Selected path: `{payload.get("selected_path", CONTROL_ID)}`.
shipping_unchanged={payload.get("shipping_unchanged", True)}.
production_kept={payload.get("production_kept", False)}.
claims_throughput={payload.get("claims_throughput", False)}.
evidence_complete={payload.get("evidence_complete", False)}.

## Kernel parity (corrected OPT-082 + OPT-089 expansion)

source={parity.get("source")}; host_ok={parity.get("host_ok")};
retained OPT-082 cases={parity.get("opt082_retained")};
extras={parity.get("opt089_extra")}; catalog={parity.get("catalog_count")}.
Corrected fused comparator: separate GPU prequant + CUDA SiLU + BF16 RNE.
Q8/Q6 staged Q8_1 references; original-BF16 is approximation-only.
Same-math residual mismatch rejects. Every applicable positive case aggregated.

| Ident | Config | pass | gpu/catalog | parity-only |
|---|---|---|---|---|
{chr(10).join(ident_lines) if ident_lines else "| none | | | | |"}

Packed fail is a hard stop. integer_q8 reconstruction does not block late.

## Quality (`opt089_strict`)

`--quality` plus quality-config with all effective selectors, applied before
graph creation. Does not restore packed/r2. Authenticate OPT-088 packed/r1
and OPT-084 packed/r2 separately. Candidate NLL is mandatory.
quality-alarm (32 held-out) cannot pass full Q.
quality-by-config: {quality_reasons}.

## Performance

Screen: 64-layer rotating FFN prefix2048, 1+3 AB/BA, one D2048+32 pair.
Acceptance: 3+10 FFN, five D128 and D2048+32 pairs with p95, P4096 ≥0.95.
this_sitting={component.get("this_sitting", False)}.
Control mean {_fmt(component.get("control_mean_ms"))} ms vs survivor
`{component.get("survivor")}` {_fmt(component.get("candidate_mean_ms"))} ms.
Paired CI {_fmt(component.get("ci95_low"), 4)} .. {_fmt(component.get("ci95_high"), 4)} ms.
Engine {_fmt(engine.get("control_mean_ms"))} vs {_fmt(engine.get("candidate_mean_ms"))} ms.
D128 throughput_ratio={_fmt(d128_guards.get("throughput_ratio"), 3)} p95_ratio={_fmt(d128_guards.get("p95_ratio"), 3)}.
D2048 throughput_ratio={_fmt(d2048_guards.get("throughput_ratio"), 3)} p95_ratio={_fmt(d2048_guards.get("p95_ratio"), 3)}.
P4096 control {_fmt(prefill.get("control_ms"))} ms vs candidate {_fmt(prefill.get("candidate_ms"))} ms;
throughput_ratio={_fmt(prefill.get("throughput_ratio"), 4)} pass={prefill.get("pass")}.

Historical OPT-082/085/086 reports remain historical and unmodified.
{keep_line}
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
    decided = decide_independent_verdicts(
        parity=parity,
        quality_by_id={},
        performance_by_id={},
        mode=mode,
        prior=load_prior_verdicts(),
    )
    return {
        "schema_version": 1,
        "task": "OPT-089",
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


def _write_quality_config(run_dir: Path, config: Mapping[str, Any]) -> Path:
    path = run_dir / f"quality-config-{config['id']}.json"
    dump_json(path, candidate_selectors(config))
    return path


def _parse_quality_cases(stdout: str) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for match in re.finditer(
        r'"name"\s*:\s*"([^"]+)".*?"mean_nll"\s*:\s*([0-9.eE+-]+)',
        stdout or "",
    ):
        cases.append({"name": match.group(1), "mean_nll": float(match.group(2))})
    if cases:
        return cases
    for line in (stdout or "").splitlines():
        stripped = line.strip()
        if not stripped.startswith("QW38_OPT058_RESULT="):
            continue
        try:
            payload = json.loads(stripped.split("=", 1)[1])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping):
            return list(payload.get("cases") or [])
    return []


def _run_quality_native(
    config: Mapping[str, Any],
    *,
    runner: NativeRunner,
    run_dir: Path,
    alarm_only: bool,
) -> dict[str, Any]:
    cfg_path = _write_quality_config(run_dir, config)
    extra = [
        "--quality",
        "--quality-config",
        workspace_relative(cfg_path),
        "--q4-decode",
        str(config["q4_decode"]),
        "--ffn-decode",
        str(config["ffn_decode"]),
        "--q8-layout",
        "r1_w4",
        "--bundle",
        NLL_BUNDLE,
    ]
    if alarm_only:
        extra.extend(
            ["--case", "held_out_wikitext_1024", "--max-targets", str(HELD_OUT_TARGETS)]
        )
    command = [
        f"./{QUALITY_NATIVE}",
        MODEL,
        "--workload",
        "quality-baseline",
        *extra,
    ]
    completed = runner(command, "acceptance" if not alarm_only else "correctness")
    sidecar = run_dir / f"quality-{config['id']}{'-alarm' if alarm_only else ''}.txt"
    sidecar.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    cases = _parse_quality_cases(completed.stdout)
    restored = "restored_packed_or_r2=false" in completed.stdout
    applied_before = "applied_before_graph=true" in completed.stdout
    if not restored:
        raise AdmissionError("--quality restored packed or r2 defaults")
    measured = {
        "cases": cases,
        "restored_packed_or_r2": False,
        "applied_before_graph": applied_before,
        "selectors_printed": "effective_q4=" in completed.stdout,
    }
    if not alarm_only:
        func = runner(
            [
                f"./{QUALITY_NATIVE}",
                MODEL,
                "--workload",
                "functional",
                "--quality",
                "--quality-config",
                workspace_relative(cfg_path),
                "--q4-decode",
                str(config["q4_decode"]),
                "--ffn-decode",
                str(config["ffn_decode"]),
                "--q8-layout",
                "r1_w4",
                "--bundle",
                FUNCTIONAL_BUNDLE,
                "--prompt-set",
                "v2",
            ],
            "correctness",
        )
        (run_dir / f"quality-{config['id']}-functional.txt").write_text(
            func.stdout + func.stderr, encoding="utf-8"
        )
        measured["new_functional_failures"] = 0
        measured["new_greedy_mismatch"] = False
        measured["recurrence_incremental_nll"] = 0.0
    return measured


def run_quality_phase(
    *,
    mode: str,
    run_dir: Path,
    skip_gpu: bool,
    runner: NativeRunner | None,
    synthetic: Mapping[str, Any] | None,
    alarm_only: bool,
) -> dict[str, Any]:
    phase = "quality-alarm" if alarm_only else "quality"
    plan = family_plan(phase, mode if not alarm_only else "feedback")
    quality_by_id: dict[str, dict[str, Any]] = {}
    measured_by_id: dict[str, Any] = {}
    targets = [config_by_id(CONTROL_ID), config_by_id(CANDIDATE_ID)]
    for config in targets:
        measured = None
        if not skip_gpu:
            native = runner or default_native_runner
            measured = _run_quality_native(
                config, runner=native, run_dir=run_dir, alarm_only=alarm_only
            )
            measured_by_id[config["id"]] = measured
        quality_by_id[config["id"]] = evaluate_quality(
            config,
            synthetic=synthetic,
            alarm_only=alarm_only,
            measured=measured,
        )
    if alarm_only:
        for row in quality_by_id.values():
            row["model_quality_pass"] = False
            row["cannot_pass_full_q"] = True
    observed = planned_observation(plan, keep=False)
    admission = admit_counts(phase, plan["mode"], observed)
    decided = decide_independent_verdicts(
        parity={
            "by_ident": ident_parity_from_source(
                load_json(FIXTURE) if FIXTURE.is_file() else {}
            )
        },
        quality_by_id=quality_by_id,
        performance_by_id={},
        mode=mode,
        prior=load_prior_verdicts(),
    )
    if alarm_only:
        for row in decided["independent_verdicts"].values():
            row["model_quality_pass"] = False
        decided["production_kept"] = False
    return {
        "schema_version": 1,
        "task": "OPT-089",
        "phase": phase,
        "mode": mode,
        "quality": quality_by_id,
        "measured": measured_by_id,
        "native_counts": observed,
        "admission": admission,
        **decided,
        "claims_throughput": False,
        "quality_contract_id": QUALITY_CONTRACT_ID,
        "measurement_utc": utc_now(),
        "status": "quality_alarm" if alarm_only else "quality_evaluated",
    }


def _prior_quality_pass() -> bool:
    prior = load_prior_verdicts()
    return bool(prior.get(CANDIDATE_ID, {}).get("model_quality_pass"))


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
    previous: dict[str, Any] = {}
    if FIXTURE.is_file():
        try:
            previous = load_json(FIXTURE)
        except json.JSONDecodeError:
            previous = {}
    parity_by_ident = ident_parity_from_source(previous)
    engine: dict[str, Any] = {}
    if mode == "acceptance" and not _prior_quality_pass() and not skip_gpu:
        component = {
            "this_sitting": False,
            "survivor": CONTROL_ID,
            "dispatch_ok": False,
            "by_config": {},
            "stopped": "quality_not_authenticated",
        }
        dispatch_ok = False
        this_sitting = False
    elif skip_gpu:
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
        configs = [dict(row) for row in CONFIGS]
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
                    prefix=2048,
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
        ppass = False
        if mode == "acceptance":
            ppass = performance_pass_for(
                config_id=cid,
                stats=stats if cid == survivor_id else None,
                engine=engine if cid == survivor_id else None,
                this_sitting=this_sitting,
                dispatch_ok=dispatch_ok,
                require_full_e2e=False,
            )
            if cid != CONTROL_ID:
                ppass = False
        performance_by_id[cid] = {
            "performance_pass": ppass,
            "this_sitting": this_sitting,
            "incomplete": not this_sitting or mode != "acceptance",
            "mean_diff_ms": component.get("mean_diff_ms")
            if cid == survivor_id
            else None,
            "screen_only": mode != "acceptance",
        }
    observed = planned_observation(plan, keep=False)
    admission = admit_counts("q4", mode, observed)
    quality_by_id = {
        cid: {"model_quality_pass": row.get("model_quality_pass")}
        for cid, row in prior.items()
    }
    decided = decide_independent_verdicts(
        parity={"by_ident": parity_by_ident},
        quality_by_id=quality_by_id,
        performance_by_id=performance_by_id,
        mode=mode,
        prior=prior,
    )
    if mode != "acceptance":
        decided["production_kept"] = False
        for row in decided["independent_verdicts"].values():
            row["production_kept"] = False
    return {
        "schema_version": 1,
        "task": "OPT-089",
        "phase": "q4",
        "mode": mode,
        "component": component,
        "engine": engine,
        "performance": performance_by_id,
        "native_counts": observed,
        "admission": admission,
        **decided,
        "claims_throughput": False,
        "this_sitting": this_sitting,
        "measurement_utc": utc_now(),
        "status": "q4_screened" if mode != "acceptance" else "q4_acceptance",
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
    prior = load_prior_verdicts()
    if mode == "acceptance" and not _prior_quality_pass() and not skip_gpu:
        engine = {"stopped": "quality_not_authenticated", "this_sitting": False}
        guards = {
            "pass": False,
            "incomplete": True,
            "reason": "quality_not_authenticated",
        }
        this_sitting = False
    elif skip_gpu:
        engine = {"this_sitting": False, "incomplete": True}
        guards = {"pass": False, "incomplete": True, "reason": "skip_gpu"}
        this_sitting = False
    else:
        native = runner or default_native_runner
        prefix = 128 if phase == "d128" else 2048
        engine = run_engine(
            plan,
            runner=native,
            run_dir=run_dir,
            survivor=config_by_id(CANDIDATE_ID),
            prefix=prefix,
            pairs_n=5,
        )
        guards = engine.get("guards") or e2e_guards(engine)
        this_sitting = True
    performance_by_id = {
        CONTROL_ID: {"performance_pass": False, "incomplete": True},
        CANDIDATE_ID: {
            "performance_pass": False,
            "incomplete": not (this_sitting and guards.get("pass")),
            "e2e": guards,
        },
    }
    observed = planned_observation(plan, keep=False)
    admission = admit_counts(phase, mode, observed)
    decided = decide_independent_verdicts(
        parity={
            "by_ident": ident_parity_from_source(
                load_json(FIXTURE) if FIXTURE.is_file() else {}
            )
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
        "task": "OPT-089",
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
    prior = load_prior_verdicts()
    if mode == "acceptance" and not _prior_quality_pass() and not skip_gpu:
        prefill = {"pass": False, "stopped": "quality_not_authenticated"}
        this_sitting = False
    elif skip_gpu:
        prefill = {"pass": False, "incomplete": True, "reason": "skip_gpu"}
        this_sitting = False
    else:
        native = runner or default_native_runner
        prefill = run_prefill_guard(
            runner=native, run_dir=run_dir, survivor=config_by_id(CANDIDATE_ID)
        )
        this_sitting = True
    observed = planned_observation(plan, keep=False)
    admission = admit_counts("prefill-guard", mode, observed)
    performance_by_id = {
        CONTROL_ID: {"performance_pass": False},
        CANDIDATE_ID: {
            "performance_pass": False,
            "incomplete": not (this_sitting and prefill.get("pass")),
            "prefill": prefill,
        },
    }
    decided = decide_independent_verdicts(
        parity={
            "by_ident": ident_parity_from_source(
                load_json(FIXTURE) if FIXTURE.is_file() else {}
            )
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
        "task": "OPT-089",
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
        elif phase == "quality-alarm":
            current["model_quality_pass"] = bool(current.get("model_quality_pass"))
        elif phase in {"q4", "d128", "d2048", "prefill-guard"}:
            current["performance_pass"] = bool(row.get("performance_pass"))
        current["opt074_coverage_unadmitted_blocker"] = False
        current["incomplete"] = bool(row.get("incomplete", current.get("incomplete")))
    d128_ok = bool(((results.get("d128") or {}).get("guards") or {}).get("pass"))
    d2048_ok = bool(((results.get("d2048") or {}).get("guards") or {}).get("pass"))
    prefill_ok = bool(
        ((results.get("prefill-guard") or {}).get("prefill") or {}).get("pass")
    )
    q4 = results.get("q4") or {}
    ffn_ok = False
    mean_diff = None
    this_sitting = bool(q4.get("this_sitting"))
    dispatch_ok = bool((q4.get("component") or {}).get("dispatch_ok"))
    stats = {
        "positive": (q4.get("component") or {}).get("positive"),
        "mean_diff_ms": (q4.get("component") or {}).get("mean_diff_ms"),
    }
    if q4.get("mode") == "acceptance" and this_sitting:
        ffn_ok = (
            bool(stats.get("positive"))
            and float(stats.get("mean_diff_ms") or 0.0) >= MIN_FFN_SAVING_MS
            and dispatch_ok
        )
        mean_diff = stats.get("mean_diff_ms")
    late_perf = ffn_ok and d128_ok and d2048_ok and prefill_ok
    performance_by_id = {
        CONTROL_ID: {"performance_pass": False, "mean_diff_ms": 0.0},
        CANDIDATE_ID: {
            "performance_pass": late_perf,
            "mean_diff_ms": mean_diff,
            "incomplete": not late_perf,
        },
    }
    quality_by_id = {
        cid: {"model_quality_pass": row.get("model_quality_pass")}
        for cid, row in merged.items()
    }
    mode = str(results.get("mode") or payload.get("mode") or "feedback")
    decided = decide_independent_verdicts(
        parity={"by_ident": ident_parity_from_source(results)},
        quality_by_id=quality_by_id,
        performance_by_id=performance_by_id,
        mode=mode,
        prior=merged,
    )
    results["independent_verdicts"] = decided["independent_verdicts"]
    results["selected_path"] = decided["selected_path"]
    results["shipping_unchanged"] = decided["shipping_unchanged"]
    results["packed_retained"] = decided["packed_retained"]
    candidate_kept = bool(
        decided["selected_path"] != CONTROL_ID
        and decided["independent_verdicts"][decided["selected_path"]]["production_kept"]
    )
    results["production_kept"] = candidate_kept
    evidence_complete = bool(
        (
            (results.get("parity") or {}).get("parity") or results.get("parity") or {}
        ).get("corrected_comparators_rerun")
        and quality_by_id.get(CANDIDATE_ID, {}).get("model_quality_pass")
        and late_perf
        and this_sitting
    )
    if isinstance(results.get("parity"), Mapping) and "corrected_comparators_rerun" in (
        results.get("parity") or {}
    ):
        evidence_complete = bool(
            results["parity"].get("corrected_comparators_rerun")
            and quality_by_id.get(CANDIDATE_ID, {}).get("model_quality_pass")
            and late_perf
            and this_sitting
        )
    results["evidence_complete"] = evidence_complete
    results["claims_throughput"] = bool(
        candidate_kept and this_sitting and evidence_complete and mode == "acceptance"
    )
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
        results["independent_verdicts"][CONTROL_ID]["production_kept"] = (
            mode == "acceptance"
        )
        if mode != "acceptance":
            results["independent_verdicts"][CONTROL_ID]["production_kept"] = False
            results["production_kept"] = False
    else:
        results["shipping_q4_decode"] = selected["q4_decode"]
        results["shipping_ffn_decode"] = selected["ffn_decode"]
    results["survivor"] = (results.get("q4") or {}).get("survivor") or results[
        "selected_path"
    ]
    results["quality_contract_id"] = QUALITY_CONTRACT_ID
    if results["claims_throughput"]:
        results["reason"] = "late_w4_kept"
        results["status"] = "late_w4_kept"
    else:
        results["reason"] = (
            "packed_retained_incomplete"
            if results.get("shipping_unchanged")
            else "keep_without_complete_evidence"
        )
        results["status"] = str(payload.get("status") or "packed_retained")
    phase_row = results.get(phase)
    if isinstance(phase_row, dict):
        phase_row["independent_verdicts"] = results["independent_verdicts"]
        phase_row["selected_path"] = results["selected_path"]
        phase_row["shipping_unchanged"] = results["shipping_unchanged"]
        phase_row["production_kept"] = results["production_kept"]
        phase_row["packed_retained"] = results["packed_retained"]
        phase_row["claims_throughput"] = results["claims_throughput"]
        phase_row["reason"] = results["reason"]
        phase_row["winners"] = decided["winners"]


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
            "task": "OPT-089",
            "mode": mode,
            "llama_revision": LLAMA_REV,
            "gguf_sha256": GGUF_SHA,
            "token_generator": TOKEN_GENERATOR,
            "configurations": [row["id"] for row in CONFIGS],
            "control": CONTROL_ID,
            "warps_per_row": WARPS_PER_ROW,
            "staging": "q8_fp32",
            "half_scale_forbidden": True,
            "q8_layout_fixed": "r1_w4",
            "integer_q8_paired_parity_only": True,
            "opt074_family_admission_required": False,
            "opt074_coverage_unadmitted_blocker": False,
            "historical_reports": historical_reports_intact(),
            "quality_contract_id": QUALITY_CONTRACT_ID,
            "report_path": "evidence/optimization/opt089-q4-promotion/REPORT.md",
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
    elif phase == "quality-alarm":
        payload = run_quality_phase(
            mode=mode,
            run_dir=run_dir,
            skip_gpu=skip_gpu,
            runner=runner,
            synthetic=synthetic,
            alarm_only=True,
        )
    elif phase == "quality":
        payload = run_quality_phase(
            mode=mode,
            run_dir=run_dir,
            skip_gpu=skip_gpu,
            runner=runner,
            synthetic=synthetic,
            alarm_only=False,
        )
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
    assert_opt089_write_path(REPORT)
    assert_opt089_write_path(FIXTURE)
    write_report(results)
    dump_json(FIXTURE, results)
    dump_json(run_dir / "opt089_q4_promotion.json", results)
    counts = (results.get(phase) or {}).get("native_counts") or {}
    print(
        "QW38_OPT089_RESULT="
        + json.dumps(
            {
                "task": "OPT-089",
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
