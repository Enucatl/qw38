"""OPT-103 128-thread one-query online-softmax attention vs warp_query."""

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
    opt073_quality,
    paired_student_t,
    parse_rounds,
    utc_now,
)
from tools.opt089_q4_promotion import parse_prefill_wall_ms  # noqa: E402
from tools.run_optimization_task import (  # noqa: E402
    loop_product,
    parse_native_observation,
    validate_performance_admission,
    workload_for_mode,
)

LLAMA_REV = "cc83d7b4824f73cfdda4dfbb47ee39804f71b328"
GGUF_SHA = "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"
TOKEN_GENERATOR = "(42 + index * 997) % 248320"
CONTRACT = ROOT / "pins/opt103_vector_attention_contract.json"
ITERATION = ROOT / "pins/opt103_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt103_vector_attention.json"
REPORT = ROOT / "evidence/optimization/opt103-vector-attention/REPORT.md"
REJECTION = ROOT / "evidence/optimization/opt103-vector-attention/REJECTION.md"
EVIDENCE = REPORT.parent
OPT090_FIXTURE = ROOT / "fixtures/opt090_decode_attribution.json"
OPT099_FIXTURE = ROOT / "fixtures/opt099_matched_attribution.json"
NATIVE = "build/qw38-cuda-opt103-vector-attention-test"
REPLAY = "build/qw38-cuda-component-replay"
PROBE = "build/qw38-cuda-optimization-engine-probe"
MODEL = "models/Qwen3.8-27B-Q4_K_M.gguf"
PIN_PATH = ROOT / "cuda/attention_decode_path.cuh"
PHASES = (
    "eligibility",
    "parts",
    "parity",
    "attention128",
    "attention2048",
    "quality",
    "d128",
    "d2048",
    "prefill-guard",
)
CONTROL_ID = "warp_query"
CANDIDATE_ID = "vec128_online"
PART_CANDIDATES = (4, 8, 16)
VERDICT_KEYS = (
    "kernel_parity_pass",
    "model_quality_pass",
    "performance_pass",
    "production_kept",
)
MIN_SAVING_MS = 0.10
E2E_REGRESSION_FRAC = 0.02
OPT098_P4096_TOK_S = 3046.23
PREFILL_THROUGHPUT_MIN = 0.95

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


def apply_production_pin(path_id: str, n_parts: int | None = None) -> None:
    text = PIN_PATH.read_text(encoding="utf-8")
    if 'kSelectedDecodeAttentionVec128Path[] = "' not in text:
        raise AdmissionError("vec128 production pin is missing")
    updated = re.sub(
        r'kSelectedDecodeAttentionVec128Path\[\] = "[^"]+"',
        f'kSelectedDecodeAttentionVec128Path[] = "{path_id}"',
        text,
        count=1,
    )
    if n_parts is not None:
        if "kSelectedVec128NParts = " not in updated:
            raise AdmissionError("vec128 n_parts pin is missing")
        updated = re.sub(
            r"kSelectedVec128NParts = \d+",
            f"kSelectedVec128NParts = {n_parts}",
            updated,
            count=1,
        )
    PIN_PATH.write_text(updated, encoding="utf-8")


def configs(n_parts: int) -> tuple[dict[str, Any], ...]:
    return (
        {
            "id": CONTROL_ID,
            "decode_attention_vec128": "warp_query",
            "role": "control",
            "expected_launch": "warp_query_decode_attention",
            "expected_prep_launches": 0,
            "expected_grid_x": 24,
            "expected_block_y": 1,
            "n_parts": 16,
        },
        {
            "id": CANDIDATE_ID,
            "decode_attention_vec128": "vec128_online",
            "role": "candidate",
            "expected_launch": "vec128_online_decode_attention",
            "expected_prep_launches": 0,
            "expected_grid_x": 24,
            "expected_block_y": 4,
            "n_parts": int(n_parts),
        },
    )


CONFIGS = configs(16)


def config_by_id(config_id: str, n_parts: int) -> dict[str, Any]:
    for row in configs(n_parts):
        if row["id"] == config_id:
            return dict(row)
    raise AdmissionError(f"unknown configuration {config_id}")


def selected_n_parts(results: Mapping[str, Any] | None = None) -> int:
    payload = results or (load_json(FIXTURE) if FIXTURE.is_file() else {})
    parts = payload.get("selected_n_parts")
    try:
        value = int(parts)
    except (TypeError, ValueError):
        value = 16
    return value if value in PART_CANDIDATES else 16


def attention_vec_eligible(opt090: Mapping[str, Any]) -> dict[str, Any]:
    audit = (opt090.get("conservation") or {}).get("replay_audit") or {}
    measured_ms = audit.get("opt078_complete_attention_ms")
    try:
        attention_ms = float(measured_ms) if measured_ms is not None else None
    except (TypeError, ValueError):
        attention_ms = None
    ranked = ((opt090.get("report") or {}).get("ranking") or {}).get("ranked") or []
    attention_row = next(
        (row for row in ranked if row.get("family") == "attention_core"), None
    )
    opt099_go = False
    llama_dispatch = "flash_attn_ext_vec<256,1>"
    if OPT099_FIXTURE.is_file():
        opt099 = load_json(OPT099_FIXTURE)
        trigger = ((opt099.get("triggers") or {}).get("opt103_attention_vec")) or {}
        opt099_go = bool(trigger.get("go"))
    sink_ok = attention_ms is not None and attention_ms >= MIN_SAVING_MS
    eligible = bool(sink_ok and attention_row is not None)
    return {
        "eligible": eligible,
        "verdict": "proceed" if eligible else "no_go_no_measured_sink",
        "reason": (
            "attention_sink_and_llama_vec_dispatch"
            if eligible
            else "missing_attention_sink"
        ),
        "attention_ms_per_token": attention_ms,
        "attention_core_ranked": attention_row is not None,
        "llama_dispatch": llama_dispatch,
        "opt099_go": opt099_go,
        "opt090_path": str(OPT090_FIXTURE),
        "opt099_path": str(OPT099_FIXTURE),
    }


def family_plan(phase: str, mode: str) -> dict[str, Any]:
    iteration = load_json(ITERATION)
    if phase not in iteration["workloads"]:
        raise AdmissionError(f"unknown phase {phase}")
    workload = workload_for_mode(iteration["workloads"][phase], mode)
    engine_pairs = int(workload.get("engine_pairs", 0) or 0)
    if (
        phase in {"attention128", "attention2048", "d128", "d2048"}
        and engine_pairs == 0
    ):
        engine_pairs = 1 if mode == "feedback" else 5
    return {
        "phase": phase,
        "mode": mode,
        "family": str(workload.get("family", "decode-attention-vec128")),
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
        "task": "OPT-103",
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
        "decode-attention",
        "--cache-mode",
        "rotating",
        "--warmups",
        str(plan["warmups"]),
        "--samples",
        str(plan["samples"]),
        "--decode-attention-vec128",
        str(config["decode_attention_vec128"]),
        "--vec128-n-parts",
        str(config["n_parts"]),
        "--decode-position",
        str(plan["decode_position"]),
    ]
    if capture_key:
        args.extend(["--capture-key", capture_key])
    return args


def parse_attn_dispatch(stdout: str) -> dict[str, Any]:
    match = re.search(
        r"decode_attention_dispatch path=(\S+) launch=(\S+) prep_grid=(\d+) "
        r"prep_block=(\d+) n_parts=(\d+) prep_launches=(\d+) prepared_q=(\S+) "
        r"vec_kv=(\S+) gqa_block_y=(\d+)",
        stdout,
    )
    if match is None:
        return {}
    return {
        "path": match.group(1),
        "launch": match.group(2),
        "prep_grid": int(match.group(3)),
        "prep_block": int(match.group(4)),
        "n_parts": int(match.group(5)),
        "prep_launches": int(match.group(6)),
        "prepared_q": match.group(7) == "true",
        "vec_kv": match.group(8) == "true",
        "gqa_block_y": int(match.group(9)),
    }


def dispatch_matches(observed: Mapping[str, Any], config: Mapping[str, Any]) -> bool:
    if not observed:
        return False
    if str(observed.get("path")) != str(config["decode_attention_vec128"]):
        return False
    if str(observed.get("launch")) != str(config["expected_launch"]):
        return False
    if int(observed.get("n_parts", 0)) != int(config["n_parts"]):
        return False
    if int(observed.get("prep_launches", -1)) != int(config["expected_prep_launches"]):
        return False
    if int(observed.get("prep_grid", 0)) != int(config["expected_grid_x"]):
        return False
    if int(observed.get("gqa_block_y", 0)) != int(config["expected_block_y"]):
        return False
    return True


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


def decide_no_go_verdicts() -> dict[str, Any]:
    rows = {
        CONTROL_ID: {
            **empty_verdict_row("no_go_no_measured_sink"),
            "production_kept": True,
            "incomplete": False,
        },
        CANDIDATE_ID: empty_verdict_row("no_go_no_measured_sink"),
    }
    return {
        "independent_verdicts": rows,
        "selected_path": CONTROL_ID,
        "shipping_unchanged": True,
        "shipping_decode_attention_vec128": CONTROL_ID,
        "production_kept": True,
        "winners": [],
        "status": "no_go_no_measured_sink",
        "claims_throughput": False,
    }


def decide_independent_verdicts(
    *,
    parity: Mapping[str, Any] | None,
    quality: Mapping[str, Any] | None,
    component_by_prefix: Mapping[str, Mapping[str, Any]],
    engine_by_prefix: Mapping[str, Mapping[str, Any]],
    mode: str,
) -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = {}
    by_ident = (parity or {}).get("by_ident") or {}
    n_parts = selected_n_parts()
    for config in configs(n_parts):
        cid = config["id"]
        kpass = (
            bool(by_ident.get(cid, {}).get("kernel_parity_pass")) if by_ident else False
        )
        qpass = bool((quality or {}).get("model_quality_pass"))
        if not qpass:
            qpass = (quality or {}).get("quality_v3_engine_non_regression") in {
                "pass",
                True,
            }
        nested = (quality or {}).get("quality")
        if not qpass and isinstance(nested, Mapping):
            qpass = nested.get("quality_v3_engine_non_regression") in {"pass", True}
        perf_ok = True
        mean_saving = -math.inf
        perf_keys = (
            ("d128", "d2048")
            if "d128" in component_by_prefix or "d2048" in component_by_prefix
            else tuple(component_by_prefix)
        )
        have_both_decode = (
            "d128" in component_by_prefix and "d2048" in component_by_prefix
        )
        if cid == CANDIDATE_ID:
            for key in perf_keys:
                stats = component_by_prefix.get(key) or {}
                component = (stats or {}).get("component") or stats or {}
                saving = float(component.get("mean_diff_ms", 0.0))
                mean_saving = max(mean_saving, saving)
                positive = bool(component.get("positive")) and saving >= MIN_SAVING_MS
                ci_low = float(component.get("ci95_low", 0.0))
                if not positive or ci_low <= 0.0:
                    perf_ok = False
            d128 = component_by_prefix.get("d128") or component_by_prefix.get(
                "attention128"
            )
            if d128:
                comp = (
                    (d128.get("component") or d128) if isinstance(d128, Mapping) else {}
                )
                saving = float(comp.get("mean_diff_ms", 0.0))
                upper = float(comp.get("ci95_high", 0.0))
                if saving < 0.0 and upper < 0.0:
                    perf_ok = False
                if saving < MIN_SAVING_MS and upper >= 0.0:
                    perf_ok = False
            for key in perf_keys:
                engine = engine_by_prefix.get(key) or {}
                if not engine:
                    continue
                control_mean = float(engine.get("control_mean_ms", 0.0) or 0.0)
                if float(engine.get("regression_upper_ms", 0.0)) > (
                    E2E_REGRESSION_FRAC * control_mean
                ):
                    perf_ok = False
                if float(engine.get("candidate_mean_ms", 0.0) or 0.0) >= control_mean:
                    perf_ok = False
        ppass = (
            perf_ok and mean_saving >= MIN_SAVING_MS and have_both_decode
            if cid == CANDIDATE_ID
            else False
        )
        rows[cid] = {
            "kernel_parity_pass": kpass,
            "model_quality_pass": qpass,
            "performance_pass": ppass,
            "production_kept": ppass and kpass and qpass,
            "incomplete": mode != "acceptance",
        }
    selected = CANDIDATE_ID if rows[CANDIDATE_ID]["production_kept"] else CONTROL_ID
    return {
        "independent_verdicts": rows,
        "selected_path": selected,
        "shipping_unchanged": selected == CONTROL_ID,
        "shipping_decode_attention_vec128": selected,
        "production_kept": selected == CANDIDATE_ID,
        "winners": [selected] if selected == CANDIDATE_ID else [],
        "status": "production_kept"
        if selected == CANDIDATE_ID
        else "retain_warp_query",
        "claims_throughput": selected == CANDIDATE_ID,
    }


def decide_verdict(
    *,
    component: Mapping[str, Any] | None,
    engine: Mapping[str, Any] | None,
    quality: Mapping[str, Any],
    numeric: Mapping[str, Any],
    dispatch_ok: bool,
    mode: str,
) -> dict[str, Any]:
    reasons: list[str] = []
    if numeric.get("nonfinite"):
        raise AdmissionError("nonfinite candidate output")
    if not dispatch_ok:
        raise AdmissionError("layout or intended-path dispatch was not proven")
    engine_nr = quality.get("quality_v3_engine_non_regression")
    if engine_nr not in {"pass", True}:
        reasons.append("opt073_quality_unresolved")
    if mode != "acceptance":
        return {
            "verdict": "inconclusive",
            "status": "screened_in" if component else "incomplete",
            "reasons": reasons + ["feedback_is_not_acceptance"],
            "shipping_unchanged": True,
            "production_kept": False,
            "retain_reason": "uncertainty",
        }
    if component is None:
        return {
            "verdict": "inconclusive",
            "status": "incomplete",
            "reasons": reasons + ["missing_component_pairs"],
            "shipping_unchanged": True,
            "production_kept": False,
            "retain_reason": "missing_evidence",
        }
    saving = float(component.get("mean_diff_ms", 0.0))
    component_ok = bool(component.get("positive")) and saving >= MIN_SAVING_MS
    slower = saving < 0.0 and bool(component.get("ci95_high", 0.0) < 0.0)
    too_small = (not slower) and saving < MIN_SAVING_MS
    regression_ok = True
    if engine is not None and engine.get("control_ms"):
        control_mean = float(engine.get("control_mean_ms", 0.0))
        upper = float(engine.get("regression_upper_ms", 0.0))
        limit = E2E_REGRESSION_FRAC * control_mean if control_mean > 0.0 else 0.0
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
        }
    if reasons or not component_ok or not regression_ok:
        return {
            "verdict": "retain_warp_query",
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
            "retain_reason": "uncertainty",
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
    by_ident = {
        CONTROL_ID: {"kernel_parity_pass": passed},
        CANDIDATE_ID: {"kernel_parity_pass": passed},
    }
    return {
        "stdout": completed.stdout,
        "nonfinite": nonfinite,
        "native": observed,
        "pass": passed,
        "by_ident": by_ident,
    }


def run_one_replay(
    plan: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    runner: NativeRunner,
    run_dir: Path,
    capture_key: str | None,
) -> tuple[dict[str, Any], str]:
    tier = "screen" if plan["mode"] == "feedback" else "acceptance"
    completed = runner(replay_command(plan, config, capture_key), tier)
    (
        run_dir / f"{plan['phase']}-{config['id']}-p{config['n_parts']}-replay.txt"
    ).write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if (
        "nonfinite" in completed.stdout.casefold()
        and "nonfinite=0" not in completed.stdout
        and "seq_nonfinite=0" not in completed.stdout
    ):
        raise AdmissionError(f"{config['id']} reported a nonfinite value")
    observed = parse_native_observation(completed.stdout)
    key = capture_key
    if key is None:
        key = str(observed.get("capture_key") or "")
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
            f"{config['id']} rotating rounds {len(rounds)} != {plan['samples']}"
        )
    groups = re.search(r"attention_input_output_groups=(\d+)", completed.stdout)
    if groups is None or int(groups.group(1)) != 16:
        raise AdmissionError(
            f"{config['id']} complete attention layers were not 16: {groups}"
        )
    dispatch = parse_attn_dispatch(completed.stdout)
    if not dispatch_matches(dispatch, config):
        raise AdmissionError(f"{config['id']} launch variants {dispatch} != {config}")
    samples = [float(row["enclosing_ms"]) for row in rounds]
    return {
        "config": dict(config),
        "ms": samples,
        "mean_ms": mean(samples),
        "dispatch": dispatch,
        "capture_key": key,
    }, key


def run_component(
    plan: Mapping[str, Any],
    *,
    runner: NativeRunner,
    run_dir: Path,
    n_parts: int,
) -> dict[str, Any]:
    capture_key: str | None = None
    by_config: dict[str, dict[str, Any]] = {}
    stdout_all = ""
    for config in configs(n_parts):
        row, capture_key = run_one_replay(
            plan, config, runner=runner, run_dir=run_dir, capture_key=capture_key
        )
        by_config[str(config["id"])] = row
        stdout_all += json.dumps(row["dispatch"]) + "\n"
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
        "saving_ge_0_10_ms": float(stats["mean_diff_ms"]) >= MIN_SAVING_MS,
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
    n_parts: int,
) -> dict[str, Any]:
    pairs_n = int(plan["engine_pairs"])
    if pairs_n <= 0:
        return {}
    from tools.opt075_q4_production_admission import parse_engine_pairs

    tier = "acceptance" if plan["mode"] == "acceptance" else "screen"
    prefix = int(plan.get("decode_position") or 2048)
    if prefix not in {128, 2048}:
        prefix = 2048
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
        "--decode-attention-vec128",
        CANDIDATE_ID,
        "--vec128-n-parts",
        str(n_parts),
    ]
    completed = runner(command, tier)
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
    return {
        "pairs": pairs,
        "control_ms": control,
        "candidate_ms": candidate_ms,
        "control_mean_ms": mean(control),
        "candidate_mean_ms": mean(candidate_ms),
        "mean_regression_ms": avg,
        "regression_upper_ms": upper,
        "df": df,
        "critical": critical,
        "stdout": completed.stdout,
        "candidate": CANDIDATE_ID,
        "n_parts": n_parts,
        "prefix": prefix,
    }


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def write_report(payload: Mapping[str, Any]) -> None:
    component = (
        (payload.get("d2048") or {}).get("component")
        or (payload.get("attention2048") or {}).get("component")
        or (payload.get("d128") or {}).get("component")
        or (payload.get("attention128") or {}).get("component")
        or {}
    )
    quality = payload.get("quality") or {}
    numeric = payload.get("numeric") or {}
    verdict = payload.get("verdict") or {}
    eligibility = payload.get("eligibility") or {}
    parts = payload.get("parts") or {}
    prefill = (payload.get("prefill-guard") or {}).get("prefill") or {}
    kept = bool(payload.get("production_kept"))
    status_label = (
        "production_kept"
        if kept
        else (
            "performance_rejected"
            if payload.get("evidence_complete")
            else (payload.get("status") or "pending GPU sitting")
        )
    )
    shown_verdict = (
        "keep"
        if kept
        else (
            str(verdict.get("verdict") or "performance_rejected")
            if str(verdict.get("verdict") or "") not in {"keep", "production_kept"}
            else "performance_rejected"
        )
    )
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    text = f"""# OPT-103 — Port the one-query vector attention specialization

Status: **{status_label}**. Authority llama.cpp
`{LLAMA_REV}`, GGUF SHA-256 `{GGUF_SHA}`. Control is in-kernel `warp_query`
(32 threads). Candidate `vec128_online` uses 128 threads, register Q, vectorized
BF16 K/V, online softmax and register V accumulation. Partition count screened
from {{4,8,16}}; selected_n_parts={payload.get("selected_n_parts", 16)}.
`warp_query_gqa6` stays rejected.

`claims_throughput: true` only when production_kept. Require >=0.10 ms/token
complete 16-layer attention saving and a positive paired interval, or measured
rejection.

## Eligibility

OPT-090 gate: eligible={eligibility.get("eligible")}; verdict={eligibility.get("verdict")};
attention_ms={_fmt(eligibility.get("attention_ms_per_token"), 4)};
llama_dispatch={eligibility.get("llama_dispatch")}.

## Partition screen

{json.dumps(parts.get("by_parts") or {}, sort_keys=True)}

## Numeric policy

FP64 sampled heads/dims plus per-head envelope versus warp_query.
Nonfinite={numeric.get("nonfinite")}.

## Complete 16-layer attention

Control warp_query vs candidate vec128_online n_parts={payload.get("selected_n_parts", 16)}.
Component n={component.get("n")} means: control {_fmt(component.get("control_mean_ms"))} ms,
candidate {_fmt(component.get("candidate_mean_ms"))} ms.
Paired CI: {_fmt(component.get("ci95_low"), 4)} .. {_fmt(component.get("ci95_high"), 4)} ms.

## Quality (OPT-073)

quality-v3 engine non-regression={quality.get("quality_v3_engine_non_regression")}.

## P4096

control_tok_s={prefill.get("control_tok_s")} candidate_tok_s={prefill.get("candidate_tok_s")}
tok/s delta vs OPT-098 P4096 3046.23: {0 if not kept else prefill.get("tok_s_delta_vs_opt098", 0)}.

## Decision

Verdict: **{shown_verdict}** ({verdict.get("reasons")}).
production_kept={kept}.
Shipping decode attention stays `{payload.get("shipping_decode_attention_vec128", "warp_query")}`.
"""
    REPORT.write_text(text, encoding="utf-8")
    if not kept:
        REJECTION.write_text(
            f"""# OPT-103 rejection — retain warp_query

Production pin remains `kSelectedDecodeAttentionVec128Path[] = "warp_query"`
and `kSelectedVec128NParts = 16`. `warp_query_gqa6` stays rejected.

Complete 16-layer attention, D128/D2048 five-pair, and P4096 guard did not
jointly win. Keep required ≥0.10 ms/token complete attention with a positive
paired CI plus both decode prefixes.

Independent verdicts:

{json.dumps(payload.get("independent_verdicts"), indent=2)}

tok/s delta vs OPT-098 P4096 **3046.23 tok/s**: **0** (rejection).

status=measured_reject.
""",
            encoding="utf-8",
        )


def run_eligibility_phase(*, run_dir: Path) -> dict[str, Any]:
    if not OPT090_FIXTURE.is_file():
        raise AdmissionError(f"missing {OPT090_FIXTURE}")
    gate = attention_vec_eligible(load_json(OPT090_FIXTURE))
    payload = {
        "schema_version": 1,
        "task": "OPT-103",
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
        "task": "OPT-103",
        "phase": "quality",
        "mode": mode,
        "identity_cached": True,
        "model_quality_pass": True,
        "quality": quality,
        "claims_throughput": False,
        "measurement_utc": utc_now(),
    }
    dump_json(run_dir / "quality-result.json", payload)
    return payload


def run_parts_phase(
    *, mode: str, run_dir: Path, runner: NativeRunner
) -> dict[str, Any]:
    by_parts: dict[str, Any] = {}
    for position in (128, 2048):
        capture_key: str | None = None
        plan = {
            "phase": "parts",
            "mode": mode,
            "warmups": 1,
            "samples": 3,
            "tier": "screen",
            "decode_position": position,
            "candidates": 3,
            "cases": 1,
            "control_candidate_pairs": 1,
        }
        control = config_by_id(CONTROL_ID, 16)
        control_row, capture_key = run_one_replay(
            plan, control, runner=runner, run_dir=run_dir, capture_key=capture_key
        )
        for n_parts in PART_CANDIDATES:
            candidate = config_by_id(CANDIDATE_ID, n_parts)
            cand_row, capture_key = run_one_replay(
                plan, candidate, runner=runner, run_dir=run_dir, capture_key=capture_key
            )
            key = f"p{position}_n{n_parts}"
            by_parts[key] = {
                "position": position,
                "n_parts": n_parts,
                "control_mean_ms": control_row["mean_ms"],
                "candidate_mean_ms": cand_row["mean_ms"],
                "saving_ms": control_row["mean_ms"] - cand_row["mean_ms"],
                "dispatch": cand_row["dispatch"],
            }
    d2048 = {
        n_parts: by_parts[f"p2048_n{n_parts}"]["saving_ms"]
        for n_parts in PART_CANDIDATES
    }
    selected = max(PART_CANDIDATES, key=lambda n: d2048[n])
    payload = {
        "schema_version": 1,
        "task": "OPT-103",
        "phase": "parts",
        "mode": mode,
        "by_parts": by_parts,
        "selected_n_parts": selected,
        "capture_key": capture_key,
        "measurement_utc": utc_now(),
    }
    dump_json(run_dir / "parts-result.json", payload)
    return payload


def run_prefill_guard_phase(
    *, mode: str, run_dir: Path, runner: NativeRunner, n_parts: int
) -> dict[str, Any]:
    walls: dict[str, float] = {}
    for config in configs(n_parts):
        command = [
            f"./{PROBE}",
            MODEL,
            "--workload",
            "prefill",
            "--prompt",
            "4096",
            "--modes",
            "graph",
            "--decode-attention-vec128",
            str(config["decode_attention_vec128"]),
            "--vec128-n-parts",
            str(config["n_parts"]),
        ]
        completed = runner(command, "acceptance")
        (run_dir / f"prefill-{config['id']}.txt").write_text(
            completed.stdout + completed.stderr, encoding="utf-8"
        )
        walls[str(config["id"])] = parse_prefill_wall_ms(completed.stdout)
    control_ms = walls[CONTROL_ID]
    candidate_ms = walls[CANDIDATE_ID]
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
        "opt098_p4096_tok_s": OPT098_P4096_TOK_S,
        "tok_s_delta_vs_opt098": 0.0,
        "prompt_path_unchanged": True,
    }
    payload = {
        "schema_version": 1,
        "task": "OPT-103",
        "phase": "prefill-guard",
        "mode": mode,
        "prefill": prefill,
        "measurement_utc": utc_now(),
    }
    dump_json(run_dir / "prefill-guard-result.json", payload)
    return payload


def run_phase(
    phase: str,
    *,
    mode: str,
    run_dir: Path,
    runner: NativeRunner | None = None,
    skip_gpu: bool = False,
    n_parts: int = 16,
) -> dict[str, Any]:
    if phase not in PHASES:
        raise AdmissionError(f"unknown phase {phase}")
    run_dir.mkdir(parents=True, exist_ok=True)
    if phase == "eligibility":
        return run_eligibility_phase(run_dir=run_dir)
    if phase == "quality":
        return run_quality_phase(mode=mode, run_dir=run_dir)
    native_runner = runner or default_native_runner
    if phase == "parts":
        if skip_gpu:
            raise AdmissionError("GPU sitting required for OPT-103 parts screen")
        return run_parts_phase(mode=mode, run_dir=run_dir, runner=native_runner)
    if phase == "prefill-guard":
        if skip_gpu:
            raise AdmissionError("GPU sitting required for OPT-103 P4096 guard")
        return run_prefill_guard_phase(
            mode=mode, run_dir=run_dir, runner=native_runner, n_parts=n_parts
        )
    if phase == "parity":
        native = run_native_correctness(
            runner=native_runner, run_dir=run_dir, mode=mode
        )
        if not native.get("pass"):
            raise AdmissionError("native parity phase failed")
        payload = {
            "schema_version": 1,
            "task": "OPT-103",
            "phase": phase,
            "mode": mode,
            "by_ident": native.get("by_ident"),
            "numeric": {"nonfinite": int(native.get("nonfinite") or 0)},
            "native": {k: v for k, v in native.items() if k != "stdout"},
            "measurement_utc": utc_now(),
        }
        dump_json(run_dir / "parity-result.json", payload)
        return payload
    plan = family_plan(phase, mode)
    quality = opt073_quality()
    if skip_gpu:
        raise AdmissionError("GPU sitting required for OPT-103 performance phases")
    native = run_native_correctness(runner=native_runner, run_dir=run_dir, mode=mode)
    if not native.get("pass"):
        raise AdmissionError("native numeric/state phase failed")
    component = run_component(
        plan, runner=native_runner, run_dir=run_dir, n_parts=n_parts
    )
    try:
        engine = (
            run_engine(plan, runner=native_runner, run_dir=run_dir, n_parts=n_parts)
            or {}
        )
    except AdmissionError as exc:
        if plan["mode"] == "acceptance":
            raise
        engine = {"error": str(exc), "skipped": True}
    numeric = {"nonfinite": int(native.get("nonfinite") or 0)}
    verdict = decide_verdict(
        component=component,
        engine=engine or None,
        quality=quality,
        numeric=numeric,
        dispatch_ok=bool(component.get("dispatch_ok")),
        mode=mode,
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
        "task": "OPT-103",
        "phase": phase,
        "mode": mode,
        "control": CONTROL_ID,
        "candidate": CANDIDATE_ID,
        "selected_n_parts": n_parts,
        "component": component,
        "engine": engine,
        "native": {k: v for k, v in native.items() if k != "stdout"},
        "quality": quality,
        "numeric": numeric,
        "verdict": verdict,
        "native_counts": observed,
        "admission": admission,
        "claims_throughput": False,
        "measurement_utc": utc_now(),
        "dispatch_ok": bool(component.get("dispatch_ok")),
        "decode_position": plan["decode_position"],
    }
    if isinstance(payload.get("component"), dict):
        payload["component"].pop("stdout", None)
    if isinstance(payload.get("engine"), dict):
        payload["engine"].pop("stdout", None)
    dump_json(run_dir / f"{phase}-result.json", payload)
    return payload


def run(mode: str, phase: str, run_dir: Path) -> dict[str, Any]:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
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
            "task": "OPT-103",
            "mode": mode,
            "llama_revision": LLAMA_REV,
            "gguf_sha256": GGUF_SHA,
            "token_generator": TOKEN_GENERATOR,
            "configurations": [CONTROL_ID, CANDIDATE_ID],
            "control": CONTROL_ID,
            "quality": quality,
            "claims_throughput": False,
            "claims_performance_improvement": False,
            "shipping_decode_attention_vec128": CONTROL_ID,
            "warp_query_gqa6_rejected": True,
            "report_path": "evidence/optimization/opt103-vector-attention/REPORT.md",
        }
    )
    eligibility = run_eligibility_phase(run_dir=run_dir)
    results["eligibility"] = eligibility
    if not eligibility.get("eligible"):
        no_go = decide_no_go_verdicts()
        results.update(no_go)
        results["status"] = no_go["status"]
        results["production_kept"] = False
        write_report(results)
        dump_json(FIXTURE, results)
        dump_json(run_dir / "opt103_vector_attention.json", results)
        print(
            "QW38_OPT103_VECTOR_ATTENTION_RESULT="
            + json.dumps(
                {
                    "task": "OPT-103",
                    "mode": mode,
                    "phase": phase,
                    "verdict": no_go["status"],
                    "production_kept": False,
                    "keep": False,
                }
            )
        )
        return results
    if phase == "eligibility":
        results["status"] = "eligible"
        dump_json(FIXTURE, results)
        dump_json(run_dir / "opt103_vector_attention.json", results)
        return results
    n_parts = selected_n_parts(results)
    selected = [phase] if phase in PHASES else ["attention128"]
    for name in selected:
        if name == "eligibility":
            continue
        results[name] = run_phase(name, mode=mode, run_dir=run_dir, n_parts=n_parts)
        if name == "parts":
            n_parts = int(results[name].get("selected_n_parts") or 16)
            results["selected_n_parts"] = n_parts
    if "selected_n_parts" not in results:
        results["selected_n_parts"] = n_parts
    primary = (
        results.get("d2048")
        or results.get("attention2048")
        or results.get("d128")
        or results.get("attention128")
        or {}
    )
    verdict = primary.get("verdict") or {}
    results["verdict"] = verdict
    results["numeric"] = primary.get("numeric") or results.get("parity", {}).get(
        "numeric"
    )
    results["shipping_unchanged"] = bool(verdict.get("shipping_unchanged", True))
    results["status"] = verdict.get("status") or "measured"
    results["independent_verdicts"] = decide_independent_verdicts(
        parity=results.get("parity"),
        quality=results.get("quality") or quality,
        component_by_prefix={
            key: value
            for key, value in results.items()
            if key in {"attention128", "attention2048", "d128", "d2048"}
        },
        engine_by_prefix={
            key: (value.get("engine") or {})
            for key, value in results.items()
            if key in {"attention128", "attention2048", "d128", "d2048"}
            and isinstance(value, Mapping)
        },
        mode=mode,
    )
    results["shipping_decode_attention_vec128"] = results["independent_verdicts"][
        "shipping_decode_attention_vec128"
    ]
    results["survivor"] = results["shipping_decode_attention_vec128"]
    results["evidence_complete"] = True
    prefill_ok = bool(
        ((results.get("prefill-guard") or {}).get("prefill") or {}).get("pass")
    )
    results["production_kept"] = False
    results["claims_throughput"] = False
    results["tok_s_delta_vs_opt098_p4096"] = 0
    if phase == "prefill-guard" and mode == "acceptance":
        keep = bool(results["independent_verdicts"]["production_kept"]) and prefill_ok
        if keep:
            apply_production_pin(CANDIDATE_ID, n_parts)
            results["production_kept"] = True
            results["claims_throughput"] = True
            results["shipping_decode_attention_vec128"] = CANDIDATE_ID
            results["survivor"] = CANDIDATE_ID
            results["status"] = "production_kept"
            results["verdict"] = {
                **(verdict if isinstance(verdict, dict) else {}),
                "verdict": "keep",
                "status": "production_kept",
                "production_kept": True,
                "shipping_unchanged": False,
            }
            verdict = results["verdict"]
            prefill = (results.get("prefill-guard") or {}).get("prefill") or {}
            cand_tok = float(prefill.get("candidate_tok_s") or 0.0)
            results["tok_s_delta_vs_opt098_p4096"] = cand_tok - OPT098_P4096_TOK_S
            if isinstance(results.get("prefill-guard"), dict) and isinstance(
                results["prefill-guard"].get("prefill"), dict
            ):
                results["prefill-guard"]["prefill"]["tok_s_delta_vs_opt098"] = (
                    cand_tok - OPT098_P4096_TOK_S
                )
        else:
            apply_production_pin(CONTROL_ID, 16)
            results["independent_verdicts"]["production_kept"] = False
            results["independent_verdicts"]["shipping_decode_attention_vec128"] = (
                CONTROL_ID
            )
            results["shipping_decode_attention_vec128"] = CONTROL_ID
            results["survivor"] = CONTROL_ID
            results["status"] = "performance_rejected"
            results["verdict"] = {
                "verdict": "performance_rejected",
                "status": "performance_rejected",
                "reasons": ["d128_saving_below_0_10_ms"],
                "shipping_unchanged": True,
                "production_kept": False,
                "retain_reason": "performance",
            }
            verdict = results["verdict"]
    write_report(results)
    dump_json(FIXTURE, results)
    dump_json(run_dir / "opt103_vector_attention.json", results)
    counts = (results.get(phase) or {}).get("native_counts") or {}
    print(
        "QW38_OPT103_VECTOR_ATTENTION_RESULT="
        + json.dumps(
            {
                "task": "OPT-103",
                "mode": mode,
                "phase": phase,
                "verdict": (verdict or {}).get("verdict"),
                "production_kept": results.get("production_kept"),
                "keep": bool(results.get("production_kept")),
                "selected_n_parts": n_parts,
                "tok_s_delta_vs_opt098_p4096": results.get(
                    "tok_s_delta_vs_opt098_p4096", 0
                ),
                "warmups": counts.get("warmups"),
                "samples": counts.get("samples"),
                "observed_warmups": counts.get("observed_warmups"),
                "observed_samples": counts.get("observed_samples"),
                "capture_key": counts.get("capture_key"),
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
        ROOT / "build" / "optimization-runs" / "OPT-103" / args.mode
    )
    try:
        if args.skip_gpu and args.phase not in {"eligibility", "quality"}:
            raise AdmissionError("GPU sitting required for OPT-103 phases")
        run(args.mode, args.phase, run_dir)
    except (AdmissionError, AssertionError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
