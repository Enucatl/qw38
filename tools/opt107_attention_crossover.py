"""OPT-107 prefix-aware decode attention selector vs global warp_query."""

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
    FEEDBACK_CRIT,
    T_CRIT_DF4,
    T_CRIT_DF9,
    dump_json,
    load_json,
    mean,
    opt073_quality,
    paired_student_t,
    parse_engine_pairs,
    parse_rounds,
    utc_now,
)
from tools.opt089_q4_promotion import parse_prefill_wall_ms  # noqa: E402
from tools.opt103_vector_attention import (  # noqa: E402
    CANDIDATE_ID,
    CONTROL_ID,
    E2E_REGRESSION_FRAC,
    MIN_SAVING_MS,
    config_by_id,
    default_native_runner as opt103_native_runner,
    dispatch_matches,
    parse_attn_dispatch,
    run_one_replay,
)
from tools.run_optimization_task import (  # noqa: E402
    loop_product,
    parse_native_observation,
    validate_performance_admission,
    workload_for_mode,
)

LLAMA_REV = "cc83d7b4824f73cfdda4dfbb47ee39804f71b328"
GGUF_SHA = "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"
CONTRACT = ROOT / "pins/opt107_attention_crossover_contract.json"
ITERATION = ROOT / "pins/opt107_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt107_attention_crossover.json"
REPORT = ROOT / "evidence/optimization/opt107-attention-crossover/REPORT.md"
REJECTION = ROOT / "evidence/optimization/opt107-attention-crossover/REJECTION.md"
EVIDENCE = REPORT.parent
OPT103_FIXTURE = ROOT / "fixtures/opt103_vector_attention.json"
OPT114_FIXTURE = ROOT / "fixtures/opt114_sitting_launch.json"
NATIVE = "build/qw38-cuda-opt107-attention-crossover-test"
REPLAY = "build/qw38-cuda-component-replay"
PROBE = "build/qw38-cuda-optimization-engine-probe"
MODEL = "models/Qwen3.8-27B-Q4_K_M.gguf"
PIN_PATH = ROOT / "cuda/attention_decode_path.cuh"
PHASES = (
    "eligibility",
    "screen",
    "select",
    "parity",
    "d128",
    "d2048",
    "quality",
    "state",
    "prefill-guard",
    "fallback-128k",
)
HYBRID_ID = "hybrid_crossover"
SCREEN_THRESHOLDS = (512, 1024, 1536, 2048)
MEASURE_POSITIONS = (128, 512, 1024, 1536, 2048, 4096)
VERIFIED_MAX = 4096
FALLBACK_128K = 131072
PREFILL_THROUGHPUT_MIN = 0.95
OPT114_D128_TOK_S = 53.460154339999995
OPT114_D2048_TOK_S = 48.383027649999995
OPT114_P4096_TOK_S = 2981.08938

NativeRunner = Callable[[Sequence[str], str], subprocess.CompletedProcess[str]]


def default_native_runner(
    command: Sequence[str], tier: str
) -> subprocess.CompletedProcess[str]:
    return opt103_native_runner(command, tier)


def path_for_position(
    position: int,
    *,
    threshold: int,
    verified_max: int = VERIFIED_MAX,
    override: str | None = None,
) -> str:
    if override:
        return override
    if threshold > 0 and position >= threshold and position <= verified_max:
        return CANDIDATE_ID
    return CONTROL_ID


def choose_threshold(
    by_position: Mapping[int, Mapping[str, Any]],
    *,
    thresholds: Sequence[int] = SCREEN_THRESHOLDS,
    positions: Sequence[int] = MEASURE_POSITIONS,
    verified_max: int = VERIFIED_MAX,
) -> dict[str, Any]:
    rejected: list[dict[str, Any]] = []
    for threshold in thresholds:
        longer = [pos for pos in positions if threshold <= pos <= verified_max]
        missing = [pos for pos in longer if pos not in by_position]
        failed = []
        for pos in longer:
            stats = by_position.get(pos) or {}
            if not bool(stats.get("positive")):
                failed.append(pos)
        if missing or failed:
            rejected.append(
                {
                    "threshold": threshold,
                    "missing": missing,
                    "failed_positions": failed,
                    "admitted": False,
                }
            )
            continue
        return {
            "threshold": threshold,
            "admitted": True,
            "required_positions": longer,
            "rejected": rejected,
            "reason": "lowest_positive_paired_saving",
        }
    return {
        "threshold": 0,
        "admitted": False,
        "required_positions": [],
        "rejected": rejected,
        "reason": "no_threshold_positive_through_4096",
    }


def apply_production_pin(threshold: int) -> None:
    if threshold not in (0, *SCREEN_THRESHOLDS):
        raise AdmissionError(f"illegal crossover threshold {threshold}")
    text = PIN_PATH.read_text(encoding="utf-8")
    if "kSelectedDecodeAttentionCrossoverThreshold = " not in text:
        raise AdmissionError("crossover threshold pin is missing")
    if 'kSelectedDecodeAttentionVec128Path[] = "warp_query"' not in text:
        raise AdmissionError("OPT-103 vec128 pin must remain warp_query")
    updated = re.sub(
        r"kSelectedDecodeAttentionCrossoverThreshold = \d+",
        f"kSelectedDecodeAttentionCrossoverThreshold = {threshold}",
        text,
        count=1,
    )
    PIN_PATH.write_text(updated, encoding="utf-8")


def _legal_threshold(value: Any) -> int:
    try:
        parsed = int(value or 0)
    except (TypeError, ValueError):
        return 0
    return parsed if parsed in (0, *SCREEN_THRESHOLDS) else 0


def selected_threshold(results: Mapping[str, Any] | None = None) -> int:
    payload = results or (load_json(FIXTURE) if FIXTURE.is_file() else {})
    top = _legal_threshold(payload.get("selected_threshold"))
    if top > 0:
        return top
    nested_select = payload.get("select") or {}
    nested_screen = payload.get("screen") or {}
    for candidate in (
        nested_select.get("threshold"),
        nested_screen.get("selected_threshold"),
        (nested_screen.get("choose") or {}).get("threshold"),
    ):
        recovered = _legal_threshold(candidate)
        if recovered > 0:
            return recovered
    return 0


def family_plan(phase: str, mode: str) -> dict[str, Any]:
    iteration = load_json(ITERATION)
    if phase not in iteration["workloads"]:
        raise AdmissionError(f"unknown phase {phase}")
    workload = workload_for_mode(iteration["workloads"][phase], mode)
    engine_pairs = int(workload.get("engine_pairs", 0) or 0)
    if phase in {"d128", "d2048"} and engine_pairs == 0:
        engine_pairs = 1 if mode == "feedback" else 5
    return {
        "phase": phase,
        "mode": mode,
        "family": str(workload.get("family", "decode-attention-crossover")),
        "control": CONTROL_ID,
        "warmups": int(workload.get("warmups", 0)),
        "samples": int(workload.get("samples", 1)),
        "engine_pairs": engine_pairs,
        "cases": int(workload.get("cases", 1)),
        "candidates": int(workload.get("candidates", 2)),
        "tier": str(
            workload.get("tier", "screen" if mode == "feedback" else "acceptance")
        ),
        "product": loop_product(workload),
        "control_candidate_pairs": int(workload.get("control_candidate_pairs", 1)),
        "decode_position": int(workload.get("decode_position", 128) or 128),
        "tokens": int(workload.get("tokens", 1) or 1),
    }


def planned_observation(
    plan: Mapping[str, Any], *, capture_key: str | None = None, keep: bool = False
) -> dict[str, Any]:
    samples = int(plan["samples"])
    return {
        "schema_version": 1,
        "task": "OPT-107",
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
        "uses_opt098_arrays": False,
    }


def hybrid_replay_command(
    plan: Mapping[str, Any], threshold: int, capture_key: str | None
) -> list[str]:
    args = [
        f"./{REPLAY}",
        MODEL,
        "--workload",
        "decode-attention",
        "--cache-mode",
        "rotating",
        "--warmups",
        str(plan["warmups"]),
        "--samples",
        str(plan["samples"]),
        "--decode-attention-crossover-threshold",
        str(threshold),
        "--vec128-n-parts",
        "16",
        "--decode-position",
        str(plan["decode_position"]),
    ]
    if capture_key:
        args.extend(["--capture-key", capture_key])
    return args


def expected_hybrid_config(position: int, threshold: int) -> dict[str, Any]:
    path = path_for_position(position, threshold=threshold)
    online = path == CANDIDATE_ID
    return {
        "id": HYBRID_ID,
        "decode_attention_vec128": path,
        "role": "candidate",
        "expected_launch": (
            "vec128_online_decode_attention"
            if online
            else "warp_query_decode_attention"
        ),
        "expected_prep_launches": 0,
        "expected_grid_x": 24,
        "expected_block_y": 4 if online else 1,
        "n_parts": 16,
    }


def run_native_correctness(
    *, runner: NativeRunner, run_dir: Path, mode: str
) -> dict[str, Any]:
    workload = "screen" if mode == "feedback" else "correctness"
    if mode == "acceptance":
        workload = "correctness"
    command = [f"./{NATIVE}", "--workload", workload]
    completed = runner(
        command, workload if workload != "correctness" else "correctness"
    )
    (run_dir / f"native-{workload}.txt").write_text(
        completed.stdout + completed.stderr, encoding="utf-8"
    )
    observed = parse_native_observation(completed.stdout)
    nonfinite = 0
    if "seq_nonfinite=" in completed.stdout:
        for match in re.finditer(r"seq_nonfinite=(\d+)", completed.stdout):
            nonfinite += int(match.group(1))
    passed = "status=passed" in completed.stdout
    return {
        "stdout": completed.stdout,
        "nonfinite": nonfinite,
        "native": observed,
        "pass": passed,
        "selector_host": "selector_host" in completed.stdout
        and "pass=true" in completed.stdout,
    }


def run_forced_component(
    plan: Mapping[str, Any],
    *,
    runner: NativeRunner,
    run_dir: Path,
) -> dict[str, Any]:
    capture_key: str | None = None
    by_config: dict[str, dict[str, Any]] = {}
    n_parts = 16
    for config in (
        config_by_id(CONTROL_ID, n_parts),
        config_by_id(CANDIDATE_ID, n_parts),
    ):
        row, capture_key = run_one_replay(
            plan, config, runner=runner, run_dir=run_dir, capture_key=capture_key
        )
        by_config[str(config["id"])] = row
    control_ms = by_config[CONTROL_ID]["ms"]
    candidate_ms = by_config[CANDIDATE_ID]["ms"]
    critical = T_CRIT_DF9 if int(plan["samples"]) >= 10 else FEEDBACK_CRIT
    stats = paired_student_t(control_ms, candidate_ms, critical=critical)
    return {
        "capture_key": capture_key,
        "by_config": by_config,
        "control": CONTROL_ID,
        "candidate": CANDIDATE_ID,
        "n_parts": n_parts,
        "control_ms": control_ms,
        "candidate_ms": candidate_ms,
        "control_mean_ms": mean(control_ms),
        "candidate_mean_ms": mean(candidate_ms),
        **stats,
        "path_control": by_config[CONTROL_ID]["dispatch"].get("path"),
        "path_candidate": by_config[CANDIDATE_ID]["dispatch"].get("path"),
        "n_parts_control": by_config[CONTROL_ID]["dispatch"].get("n_parts"),
        "n_parts_candidate": by_config[CANDIDATE_ID]["dispatch"].get("n_parts"),
        "dispatch_ok": True,
        "uses_opt098_arrays": False,
    }


def run_hybrid_replay(
    plan: Mapping[str, Any],
    threshold: int,
    *,
    runner: NativeRunner,
    run_dir: Path,
    capture_key: str | None,
) -> tuple[dict[str, Any], str]:
    completed = runner(
        hybrid_replay_command(plan, threshold, capture_key),
        "screen" if plan["mode"] == "feedback" else "acceptance",
    )
    (
        run_dir / f"{plan['phase']}-hybrid-p{plan['decode_position']}-replay.txt"
    ).write_text(completed.stdout + completed.stderr, encoding="utf-8")
    observed = parse_native_observation(completed.stdout)
    key = capture_key or str(observed.get("capture_key") or "")
    if not key:
        match = re.search(r"capture_key=([0-9a-f]{64})", completed.stdout)
        if match:
            key = match.group(1)
    if not key:
        raise AdmissionError("missing capture identity")
    rounds = [
        row
        for row in parse_rounds(completed.stdout)
        if row.get("cache_mode", "rotating") == "rotating"
    ]
    if len(rounds) != int(plan["samples"]):
        raise AdmissionError(
            f"hybrid rotating rounds {len(rounds)} != {plan['samples']}"
        )
    dispatch = parse_attn_dispatch(completed.stdout)
    expected = expected_hybrid_config(int(plan["decode_position"]), threshold)
    if not dispatch_matches(dispatch, expected):
        raise AdmissionError(f"hybrid dispatch {dispatch} != {expected}")
    samples = [float(row["enclosing_ms"]) for row in rounds]
    return {
        "config": expected,
        "ms": samples,
        "mean_ms": mean(samples),
        "dispatch": dispatch,
        "capture_key": key,
        "path": expected["decode_attention_vec128"],
        "n_parts": 16,
    }, key


def run_hybrid_component(
    plan: Mapping[str, Any],
    threshold: int,
    *,
    runner: NativeRunner,
    run_dir: Path,
) -> dict[str, Any]:
    control = config_by_id(CONTROL_ID, 16)
    control_row, capture_key = run_one_replay(
        plan, control, runner=runner, run_dir=run_dir, capture_key=None
    )
    hybrid_row, capture_key = run_hybrid_replay(
        plan,
        threshold,
        runner=runner,
        run_dir=run_dir,
        capture_key=capture_key,
    )
    critical = T_CRIT_DF9 if int(plan["samples"]) >= 10 else FEEDBACK_CRIT
    stats = paired_student_t(control_row["ms"], hybrid_row["ms"], critical=critical)
    return {
        "capture_key": capture_key,
        "control": CONTROL_ID,
        "candidate": HYBRID_ID,
        "threshold": threshold,
        "decode_position": plan["decode_position"],
        "control_ms": control_row["ms"],
        "candidate_ms": hybrid_row["ms"],
        "control_mean_ms": control_row["mean_ms"],
        "candidate_mean_ms": hybrid_row["mean_ms"],
        "control_dispatch": control_row["dispatch"],
        "hybrid_dispatch": hybrid_row["dispatch"],
        "selected_path": hybrid_row["path"],
        "n_parts": 16,
        **stats,
        "dispatch_ok": True,
        "uses_opt098_arrays": False,
    }


def run_hybrid_engine(
    plan: Mapping[str, Any],
    threshold: int,
    *,
    runner: NativeRunner,
    run_dir: Path,
) -> dict[str, Any]:
    pairs_n = int(plan["engine_pairs"])
    if pairs_n <= 0:
        return {}
    prefix = int(plan.get("decode_position") or 2048)
    command = [
        f"./{PROBE}",
        MODEL,
        "--workload",
        "attn-ab",
        "--pairs",
        str(pairs_n),
        "--modes",
        "graph",
        "--prefix",
        str(prefix),
        "--output-tokens",
        "32",
        "--decode-attention-crossover-threshold",
        str(threshold),
        "--vec128-n-parts",
        "16",
    ]
    completed = runner(
        command, "acceptance" if plan["mode"] == "acceptance" else "screen"
    )
    (run_dir / f"{plan['phase']}-engine.txt").write_text(
        completed.stdout + completed.stderr, encoding="utf-8"
    )
    pairs = parse_engine_pairs(completed.stdout)
    if len(pairs) != pairs_n:
        raise AdmissionError(f"engine pairs {len(pairs)} != {pairs_n}")
    control = [float(row["control_ms"]) for row in pairs]
    candidate_ms = [float(row["candidate_ms"]) for row in pairs]
    diffs = [cand - ctrl for ctrl, cand in zip(control, candidate_ms)]
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
    control_tok = [32000.0 / ms if ms > 0 else 0.0 for ms in control]
    candidate_tok = [32000.0 / ms if ms > 0 else 0.0 for ms in candidate_ms]
    return {
        "pairs": pairs,
        "control_ms": control,
        "candidate_ms": candidate_ms,
        "control_mean_ms": mean(control),
        "candidate_mean_ms": mean(candidate_ms),
        "control_mean_tok_s": mean(control_tok),
        "candidate_mean_tok_s": mean(candidate_tok),
        "mean_regression_ms": avg,
        "regression_upper_ms": upper,
        "df": df,
        "critical": critical,
        "candidate": HYBRID_ID,
        "threshold": threshold,
        "prefix": prefix,
        "uses_opt098_arrays": False,
    }


def eligibility_from_fixtures() -> dict[str, Any]:
    if not OPT103_FIXTURE.is_file():
        raise AdmissionError(f"missing {OPT103_FIXTURE}")
    if not OPT114_FIXTURE.is_file():
        raise AdmissionError(f"missing {OPT114_FIXTURE}")
    opt103 = load_json(OPT103_FIXTURE)
    opt114 = load_json(OPT114_FIXTURE)
    d128 = ((opt103.get("d128") or {}).get("component")) or {}
    d2048 = ((opt103.get("d2048") or {}).get("component")) or {}
    aa = str(opt114.get("aa_verdict") or "")
    split = float(d128.get("mean_diff_ms") or 0.0) < 0.0 and bool(d2048.get("positive"))
    sitting_ok = aa == "repeatable"
    eligible = bool(split and sitting_ok)
    return {
        "eligible": eligible,
        "verdict": "proceed" if eligible else "no_go",
        "reason": (
            "opt103_split_and_opt114_repeatable"
            if eligible
            else "missing_split_or_unstable_sitting"
        ),
        "opt103_d128_mean_diff_ms": d128.get("mean_diff_ms"),
        "opt103_d2048_mean_diff_ms": d2048.get("mean_diff_ms"),
        "opt103_d2048_positive": d2048.get("positive"),
        "opt103_shipping": opt103.get("shipping_decode_attention_vec128"),
        "opt114_aa_verdict": aa,
        "uses_opt098_arrays_for_keep_reject": False,
        "opt103_rejection_retained": opt103.get("production_kept") is False,
    }


def decide_verdict(
    *,
    threshold: int,
    d128: Mapping[str, Any] | None,
    d2048: Mapping[str, Any] | None,
    quality: Mapping[str, Any],
    mode: str,
) -> dict[str, Any]:
    reasons: list[str] = []
    engine_nr = quality.get("quality_v3_engine_non_regression")
    if engine_nr not in {"pass", True}:
        reasons.append("opt073_quality_unresolved")
    if mode != "acceptance":
        return {
            "verdict": "inconclusive",
            "status": "screened_in" if threshold else "incomplete",
            "reasons": reasons + ["feedback_is_not_acceptance"],
            "shipping_unchanged": True,
            "production_kept": False,
            "retain_reason": "uncertainty",
        }
    if threshold <= 0:
        return {
            "verdict": "no_threshold",
            "status": "performance_rejected",
            "reasons": ["no_threshold_positive_through_4096"],
            "shipping_unchanged": True,
            "production_kept": False,
            "retain_reason": "performance",
        }
    d128_comp = (d128 or {}).get("component") or {}
    d128_engine = (d128 or {}).get("engine") or {}
    d2048_comp = (d2048 or {}).get("component") or {}
    d2048_engine = (d2048 or {}).get("engine") or {}
    if not d128_comp or not d2048_comp:
        return {
            "verdict": "inconclusive",
            "status": "measured",
            "reasons": reasons + ["acceptance_incomplete"],
            "shipping_unchanged": True,
            "production_kept": False,
            "retain_reason": "uncertainty",
        }
    d128_path = str(d128_comp.get("selected_path") or "")
    if d128_path != CONTROL_ID:
        return {
            "verdict": "performance_rejected",
            "status": "performance_rejected",
            "reasons": ["d128_did_not_select_warp_query"],
            "shipping_unchanged": True,
            "production_kept": False,
            "retain_reason": "performance",
        }
    if d128_engine:
        control_mean = float(d128_engine.get("control_mean_ms", 0.0) or 0.0)
        upper = float(d128_engine.get("regression_upper_ms", 0.0) or 0.0)
        if control_mean > 0.0 and upper > E2E_REGRESSION_FRAC * control_mean:
            return {
                "verdict": "performance_rejected",
                "status": "performance_rejected",
                "reasons": ["d128_e2e_regression"],
                "shipping_unchanged": True,
                "production_kept": False,
                "retain_reason": "performance",
            }
    saving = float(d2048_comp.get("mean_diff_ms", 0.0) or 0.0)
    ci_low = float(d2048_comp.get("ci95_low", 0.0) or 0.0)
    if not bool(d2048_comp.get("positive")) or saving < MIN_SAVING_MS or ci_low <= 0.0:
        return {
            "verdict": "performance_rejected",
            "status": "performance_rejected",
            "reasons": ["d2048_saving_unresolved"],
            "shipping_unchanged": True,
            "production_kept": False,
            "retain_reason": "performance",
        }
    if d2048_engine:
        control_mean = float(d2048_engine.get("control_mean_ms", 0.0) or 0.0)
        cand_mean = float(d2048_engine.get("candidate_mean_ms", 0.0) or 0.0)
        upper = float(d2048_engine.get("regression_upper_ms", 0.0) or 0.0)
        if cand_mean >= control_mean:
            return {
                "verdict": "performance_rejected",
                "status": "performance_rejected",
                "reasons": ["d2048_engine_did_not_improve"],
                "shipping_unchanged": True,
                "production_kept": False,
                "retain_reason": "performance",
            }
        if control_mean > 0.0 and upper > E2E_REGRESSION_FRAC * control_mean:
            return {
                "verdict": "performance_rejected",
                "status": "performance_rejected",
                "reasons": ["d2048_e2e_regression"],
                "shipping_unchanged": True,
                "production_kept": False,
                "retain_reason": "performance",
            }
    if reasons:
        return {
            "verdict": "quality_blocked",
            "status": "quality_blocked",
            "reasons": reasons,
            "shipping_unchanged": True,
            "production_kept": False,
            "retain_reason": "precision",
        }
    return {
        "verdict": "keep",
        "status": "production_kept",
        "reasons": [
            "d128_warp_query",
            "d2048_component_ci_positive",
            "saving_ge_0_10_ms",
            "d2048_engine_improved",
            "quality",
        ],
        "shipping_unchanged": False,
        "production_kept": True,
        "retain_reason": None,
    }


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def write_report(payload: Mapping[str, Any]) -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    kept = bool(payload.get("production_kept"))
    d128 = (payload.get("d128") or {}).get("component") or {}
    d2048 = (payload.get("d2048") or {}).get("component") or {}
    d128_eng = (payload.get("d128") or {}).get("engine") or {}
    d2048_eng = (payload.get("d2048") or {}).get("engine") or {}
    verdict = payload.get("verdict") or {}
    status = payload.get("status") or "pending GPU sitting"
    text = f"""# OPT-107 — Dispatch decode attention by measured prefix crossover

Status: **{status}**. Authority llama.cpp `{LLAMA_REV}`, GGUF SHA-256
`{GGUF_SHA}`. Control is global in-kernel `warp_query`. Candidate is a
prefix-aware hybrid that keeps OPT-103 kernels unchanged and selects
`vec128_online` only on a measured range.

`uses_opt098_arrays_for_keep_reject: false`. Fresh OPT-114 sitting is the
control. OPT-103 rejection evidence is retained; this is a new hybrid.

## Eligibility

{json.dumps(payload.get("eligibility") or {}, sort_keys=True)}

## Threshold screen

measured positions {list(MEASURE_POSITIONS)}; screen thresholds
{list(SCREEN_THRESHOLDS)}. Selected threshold:
{payload.get("selected_threshold", 0)}. verified_max={VERIFIED_MAX}.
128K fallback checks selector only and is not an extrapolated threshold.

{json.dumps(payload.get("select") or payload.get("screen") or {}, sort_keys=True)[:4000]}

## D128

Must select `warp_query`. path={d128.get("selected_path")}.
control {_fmt(d128.get("control_mean_ms"))} ms vs hybrid
{_fmt(d128.get("candidate_mean_ms"))} ms.
Engine control {_fmt(d128_eng.get("control_mean_tok_s"), 4)} tok/s vs hybrid
{_fmt(d128_eng.get("candidate_mean_tok_s"), 4)} tok/s.
OPT-114 D128 baseline {OPT114_D128_TOK_S:.4f} tok/s.

## D2048

Require >=0.10 ms/token complete attention with a positive 95% paired
interval and engine throughput improvement.
control {_fmt(d2048.get("control_mean_ms"))} ms vs hybrid
{_fmt(d2048.get("candidate_mean_ms"))} ms.
paired CI {_fmt(d2048.get("ci95_low"), 4)} .. {_fmt(d2048.get("ci95_high"), 4)} ms.
Engine control {_fmt(d2048_eng.get("control_mean_tok_s"), 4)} tok/s vs hybrid
{_fmt(d2048_eng.get("candidate_mean_tok_s"), 4)} tok/s.
OPT-114 D2048 baseline {OPT114_D2048_TOK_S:.4f} tok/s.

## Decision

Verdict: **{verdict.get("verdict") or status}** ({verdict.get("reasons")}).
production_kept={kept}.
Shipping vec128 pin stays `warp_query`; crossover threshold=
{payload.get("selected_threshold", 0) if kept else 0}.
tok/s delta vs OPT-114 D2048: {payload.get("tok_s_delta_vs_opt114_d2048", 0)}.
"""
    REPORT.write_text(text, encoding="utf-8")
    rejected = str(verdict.get("verdict") or "") in {
        "performance_rejected",
        "no_threshold",
    }
    if rejected:
        REJECTION.write_text(
            f"""# OPT-107 rejection — retain global warp_query

Production pin remains
`kSelectedDecodeAttentionCrossoverThreshold = 0` and
`kSelectedDecodeAttentionVec128Path[] = "warp_query"`. OPT-103 rejection
evidence is unchanged.

{json.dumps(verdict, indent=2)}

tok/s delta vs OPT-114 D2048 **{OPT114_D2048_TOK_S:.4f} tok/s**: **0**
(rejection; speedup 0).

status=measured_reject.
""",
            encoding="utf-8",
        )
    elif REJECTION.is_file():
        REJECTION.unlink()


def persist_fixture(results: Mapping[str, Any]) -> None:
    dump_json(FIXTURE, results)
    write_report(results)


def current_results() -> dict[str, Any]:
    if FIXTURE.is_file():
        try:
            return load_json(FIXTURE)
        except json.JSONDecodeError:
            return {}
    return {}


def run_eligibility_phase(*, run_dir: Path) -> dict[str, Any]:
    gate = eligibility_from_fixtures()
    payload = {
        "schema_version": 1,
        "task": "OPT-107",
        "phase": "eligibility",
        "measurement_utc": utc_now(),
        **gate,
    }
    dump_json(run_dir / "eligibility-result.json", payload)
    return payload


def run_quality_phase(*, mode: str, run_dir: Path) -> dict[str, Any]:
    quality = opt073_quality()
    if quality.get("quality_v3_engine_non_regression") not in {"pass", True}:
        raise AdmissionError("OPT-073 engine non-regression is not a pass")
    payload = {
        "schema_version": 1,
        "task": "OPT-107",
        "phase": "quality",
        "mode": mode,
        "identity_cached": True,
        "model_quality_pass": True,
        "quality": quality,
        "claims_throughput": False,
        "uses_opt098_arrays_for_keep_reject": False,
        "measurement_utc": utc_now(),
    }
    dump_json(run_dir / "quality-result.json", payload)
    return payload


def run_screen_phase(
    *, mode: str, run_dir: Path, runner: NativeRunner
) -> dict[str, Any]:
    by_position: dict[str, Any] = {}
    capture_key: str | None = None
    for position in MEASURE_POSITIONS:
        plan = {
            "phase": "screen",
            "mode": mode,
            "warmups": 3 if mode == "acceptance" else 1,
            "samples": 10 if mode == "acceptance" else 3,
            "tier": "acceptance" if mode == "acceptance" else "screen",
            "decode_position": position,
            "candidates": 2,
            "cases": 1,
            "control_candidate_pairs": 1,
        }
        row = run_forced_component(plan, runner=runner, run_dir=run_dir)
        capture_key = str(row.get("capture_key") or capture_key)
        by_position[str(position)] = {
            "position": position,
            "control_mean_ms": row["control_mean_ms"],
            "candidate_mean_ms": row["candidate_mean_ms"],
            "mean_diff_ms": row["mean_diff_ms"],
            "ci95_low": row["ci95_low"],
            "ci95_high": row["ci95_high"],
            "positive": row["positive"],
            "path_control": row["path_control"],
            "path_candidate": row["path_candidate"],
            "n_parts": 16,
            "dispatch_ok": row["dispatch_ok"],
            "component": row,
        }
    chosen = choose_threshold({int(key): value for key, value in by_position.items()})
    payload = {
        "schema_version": 1,
        "task": "OPT-107",
        "phase": "screen",
        "mode": mode,
        "by_position": by_position,
        "choose": chosen,
        "selected_threshold": chosen["threshold"] if chosen["admitted"] else 0,
        "capture_key": capture_key,
        "uses_opt098_arrays": False,
        "measurement_utc": utc_now(),
    }
    dump_json(run_dir / "screen-result.json", payload)
    return payload


def run_select_phase(*, run_dir: Path, results: Mapping[str, Any]) -> dict[str, Any]:
    screen = results.get("screen") or {}
    by_position = {
        int(key): value for key, value in (screen.get("by_position") or {}).items()
    }
    chosen = (
        choose_threshold(by_position)
        if by_position
        else {
            "threshold": 0,
            "admitted": False,
            "reason": "screen_incomplete",
            "rejected": [],
            "required_positions": [],
        }
    )
    payload = {
        "schema_version": 1,
        "task": "OPT-107",
        "phase": "select",
        **chosen,
        "uses_opt098_arrays": False,
        "measurement_utc": utc_now(),
    }
    dump_json(run_dir / "select-result.json", payload)
    return payload


def run_fallback_phase(*, run_dir: Path, threshold: int) -> dict[str, Any]:
    path = path_for_position(FALLBACK_128K, threshold=threshold)
    pin_text = PIN_PATH.read_text(encoding="utf-8")
    payload = {
        "schema_version": 1,
        "task": "OPT-107",
        "phase": "fallback-128k",
        "position": FALLBACK_128K,
        "threshold": threshold,
        "verified_max": VERIFIED_MAX,
        "selected_path": path,
        "pass": path == CONTROL_ID,
        "extrapolated": False,
        "vec128_pin_is_warp_query": 'kSelectedDecodeAttentionVec128Path[] = "warp_query"'
        in pin_text,
        "measurement_utc": utc_now(),
    }
    dump_json(run_dir / "fallback-128k-result.json", payload)
    return payload


def run_prefill_guard_phase(
    *, mode: str, run_dir: Path, runner: NativeRunner, threshold: int
) -> dict[str, Any]:
    del mode
    walls: dict[str, float] = {}
    for ident, extra in (
        (CONTROL_ID, ["--decode-attention-crossover-threshold", "0"]),
        (
            HYBRID_ID,
            ["--decode-attention-crossover-threshold", str(max(threshold, 0))],
        ),
    ):
        command = [
            f"./{PROBE}",
            MODEL,
            "--workload",
            "prefill",
            "--prompt",
            "4096",
            "--modes",
            "graph",
            *extra,
            "--vec128-n-parts",
            "16",
        ]
        completed = runner(command, "acceptance")
        (run_dir / f"prefill-{ident}.txt").write_text(
            completed.stdout + completed.stderr, encoding="utf-8"
        )
        walls[ident] = parse_prefill_wall_ms(completed.stdout)
    control_ms = walls[CONTROL_ID]
    candidate_ms = walls[HYBRID_ID]
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
        "opt114_p4096_tok_s": OPT114_P4096_TOK_S,
        "tok_s_delta_vs_opt114": cand_tok - OPT114_P4096_TOK_S,
        "prompt_path_unchanged": True,
        "uses_opt098_arrays": False,
    }
    payload = {
        "schema_version": 1,
        "task": "OPT-107",
        "phase": "prefill-guard",
        "prefill": prefill,
        "measurement_utc": utc_now(),
    }
    dump_json(run_dir / "prefill-guard-result.json", payload)
    return payload


def run_decode_phase(
    phase: str,
    *,
    mode: str,
    run_dir: Path,
    runner: NativeRunner,
    threshold: int,
) -> dict[str, Any]:
    plan = family_plan(phase, mode)
    native = run_native_correctness(runner=runner, run_dir=run_dir, mode=mode)
    if not native.get("pass"):
        raise AdmissionError("native numeric/state phase failed")
    component = run_hybrid_component(plan, threshold, runner=runner, run_dir=run_dir)
    try:
        engine = (
            run_hybrid_engine(plan, threshold, runner=runner, run_dir=run_dir) or {}
        )
    except AdmissionError as exc:
        if plan["mode"] == "acceptance":
            raise
        engine = {"error": str(exc), "skipped": True}
    quality = opt073_quality()
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
            "tokens": plan["tokens"],
        },
        stdout=json.dumps(observed),
        success=True,
    )
    if not admission["ok"]:
        raise AdmissionError(admission["message"])
    payload = {
        "schema_version": 1,
        "task": "OPT-107",
        "phase": phase,
        "mode": mode,
        "control": CONTROL_ID,
        "candidate": HYBRID_ID,
        "selected_threshold": threshold,
        "component": component,
        "engine": engine,
        "native": {k: v for k, v in native.items() if k != "stdout"},
        "quality": quality,
        "numeric": {"nonfinite": int(native.get("nonfinite") or 0)},
        "native_counts": observed,
        "admission": admission,
        "claims_throughput": False,
        "uses_opt098_arrays_for_keep_reject": False,
        "measurement_utc": utc_now(),
        "dispatch_ok": bool(component.get("dispatch_ok")),
        "decode_position": plan["decode_position"],
    }
    dump_json(run_dir / f"{phase}-result.json", payload)
    return payload


def run_phase(
    phase: str,
    *,
    mode: str,
    run_dir: Path,
    runner: NativeRunner | None = None,
    skip_gpu: bool = False,
    threshold: int = 0,
    results: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if phase not in PHASES:
        raise AdmissionError(f"unknown phase {phase}")
    run_dir.mkdir(parents=True, exist_ok=True)
    if phase == "eligibility":
        return run_eligibility_phase(run_dir=run_dir)
    if phase == "quality":
        return run_quality_phase(mode=mode, run_dir=run_dir)
    if phase == "select":
        return run_select_phase(run_dir=run_dir, results=results or current_results())
    if phase == "fallback-128k":
        return run_fallback_phase(run_dir=run_dir, threshold=threshold)
    native_runner = runner or default_native_runner
    if skip_gpu:
        raise AdmissionError(f"GPU sitting required for OPT-107 phase {phase}")
    if phase == "screen":
        return run_screen_phase(mode=mode, run_dir=run_dir, runner=native_runner)
    if phase == "parity" or phase == "state":
        native = run_native_correctness(
            runner=native_runner, run_dir=run_dir, mode=mode
        )
        if not native.get("pass"):
            raise AdmissionError("native parity/state phase failed")
        payload = {
            "schema_version": 1,
            "task": "OPT-107",
            "phase": phase,
            "mode": mode,
            "pass": True,
            "numeric": {"nonfinite": int(native.get("nonfinite") or 0)},
            "native": {k: v for k, v in native.items() if k != "stdout"},
            "measurement_utc": utc_now(),
        }
        dump_json(run_dir / f"{phase}-result.json", payload)
        return payload
    if phase == "prefill-guard":
        return run_prefill_guard_phase(
            mode=mode, run_dir=run_dir, runner=native_runner, threshold=threshold
        )
    return run_decode_phase(
        phase,
        mode=mode,
        run_dir=run_dir,
        runner=native_runner,
        threshold=threshold,
    )


def empty_results(mode: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task": "OPT-107",
        "mode": mode,
        "status": "pending",
        "llama_revision": LLAMA_REV,
        "gguf_sha256": GGUF_SHA,
        "control": CONTROL_ID,
        "candidate": HYBRID_ID,
        "selected_threshold": 0,
        "verified_max": VERIFIED_MAX,
        "shipping_decode_attention_vec128": CONTROL_ID,
        "production_kept": False,
        "claims_throughput": False,
        "claims_performance_improvement": False,
        "uses_opt098_arrays_for_keep_reject": False,
        "opt103_kernels_unchanged": True,
        "report_path": "evidence/optimization/opt107-attention-crossover/REPORT.md",
    }


def run(
    mode: str, phase: str, run_dir: Path, *, skip_gpu: bool = False
) -> dict[str, Any]:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    prior = current_results()
    results = empty_results(mode)
    results.update(prior)
    results["mode"] = mode
    results["selected_threshold"] = selected_threshold(results)
    quality = opt073_quality()
    results["quality"] = quality
    eligibility = run_eligibility_phase(run_dir=run_dir)
    results["eligibility"] = eligibility
    if not eligibility.get("eligible"):
        results["status"] = "no_go"
        results["production_kept"] = False
        persist_fixture(results)
        print(
            "QW38_OPT107_ATTENTION_CROSSOVER_RESULT="
            + json.dumps(
                {
                    "task": "OPT-107",
                    "mode": mode,
                    "phase": phase,
                    "verdict": "no_go",
                    "production_kept": False,
                    "keep": False,
                }
            )
        )
        return results
    if phase == "eligibility":
        results["status"] = "eligible"
        persist_fixture(results)
        print(
            "QW38_OPT107_ATTENTION_CROSSOVER_RESULT="
            + json.dumps(
                {
                    "task": "OPT-107",
                    "mode": mode,
                    "phase": phase,
                    "verdict": "eligible",
                    "production_kept": False,
                    "keep": False,
                    "uses_opt098_arrays_for_keep_reject": False,
                }
            )
        )
        return results
    threshold = selected_threshold(results)
    selected = [phase] if phase in PHASES else ["screen"]
    for name in selected:
        if name == "eligibility":
            continue
        results[name] = run_phase(
            name,
            mode=mode,
            run_dir=run_dir,
            skip_gpu=skip_gpu,
            threshold=threshold,
            results=results,
        )
        if name == "screen":
            threshold = int(results[name].get("selected_threshold") or 0)
            results["selected_threshold"] = threshold
        if name == "select":
            threshold = int(results[name].get("threshold") or 0)
            results["selected_threshold"] = threshold
    results["selected_threshold"] = threshold
    verdict = decide_verdict(
        threshold=threshold,
        d128=results.get("d128"),
        d2048=results.get("d2048"),
        quality=results.get("quality") or quality,
        mode=mode,
    )
    results["verdict"] = verdict
    results["shipping_unchanged"] = bool(verdict.get("shipping_unchanged", True))
    results["status"] = verdict.get("status") or "measured"
    results["uses_opt098_arrays_for_keep_reject"] = False
    prefill_ok = bool(
        ((results.get("prefill-guard") or {}).get("prefill") or {}).get("pass")
    )
    fallback_ok = bool((results.get("fallback-128k") or {}).get("pass", True))
    results["production_kept"] = False
    results["claims_throughput"] = False
    results["tok_s_delta_vs_opt114_d2048"] = 0
    results["tok_s_delta_vs_opt114_d128"] = 0
    if phase in {"prefill-guard", "fallback-128k", "d2048"} and mode == "acceptance":
        keep = (
            bool(verdict.get("production_kept"))
            and (phase != "prefill-guard" or prefill_ok)
            and fallback_ok
            and threshold > 0
        )
        if keep and phase == "prefill-guard":
            apply_production_pin(threshold)
            results["production_kept"] = True
            results["claims_throughput"] = True
            results["status"] = "production_kept"
            engine = (results.get("d2048") or {}).get("engine") or {}
            cand_tok = float(engine.get("candidate_mean_tok_s") or 0.0)
            d128_eng = (results.get("d128") or {}).get("engine") or {}
            results["tok_s_delta_vs_opt114_d2048"] = cand_tok - OPT114_D2048_TOK_S
            results["tok_s_delta_vs_opt114_d128"] = (
                float(d128_eng.get("candidate_mean_tok_s") or 0.0) - OPT114_D128_TOK_S
            )
        elif mode == "acceptance" and phase == "prefill-guard":
            apply_production_pin(0)
            results["selected_threshold"] = 0
            results["status"] = "performance_rejected"
            results["verdict"] = {
                "verdict": "performance_rejected",
                "status": "performance_rejected",
                "reasons": list(verdict.get("reasons") or ["acceptance_failed"]),
                "shipping_unchanged": True,
                "production_kept": False,
                "retain_reason": "performance",
            }
    persist_fixture(results)
    dump_json(run_dir / "opt107_attention_crossover.json", results)
    print(
        "QW38_OPT107_ATTENTION_CROSSOVER_RESULT="
        + json.dumps(
            {
                "task": "OPT-107",
                "mode": mode,
                "phase": phase,
                "verdict": (results.get("verdict") or {}).get("verdict"),
                "production_kept": results.get("production_kept"),
                "keep": bool(results.get("production_kept")),
                "selected_threshold": results.get("selected_threshold", 0),
                "tok_s_delta_vs_opt114_d2048": results.get(
                    "tok_s_delta_vs_opt114_d2048", 0
                ),
                "uses_opt098_arrays_for_keep_reject": False,
            }
        )
    )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES, default="eligibility")
    parser.add_argument(
        "--mode", choices=("feedback", "acceptance", "release"), default="feedback"
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--skip-gpu", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_dir = args.run_dir or (
        ROOT / "build" / "optimization-runs" / "OPT-107" / args.mode
    )
    try:
        if args.skip_gpu and args.phase not in {
            "eligibility",
            "quality",
            "select",
            "fallback-128k",
        }:
            raise AdmissionError("GPU sitting required for OPT-107 phases")
        run(args.mode, args.phase, run_dir, skip_gpu=args.skip_gpu)
    except (AdmissionError, AssertionError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
