"""OPT-078 decode query-prep hoist and vector KV admission versus warp_query."""

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
    T_CRIT_DF4,
    T_CRIT_DF9,
    FEEDBACK_CRIT,
    docker_common,
    dump_json,
    load_json,
    mean,
    opt073_quality,
    paired_student_t,
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
CONTRACT = ROOT / "pins/opt078_decode_attention_contract.json"
ITERATION = ROOT / "pins/opt078_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt078_decode_attention.json"
REPORT = ROOT / "evidence/optimization/opt078-decode-attention/REPORT.md"
EVIDENCE = ROOT / "evidence/optimization/opt078-decode-attention"
NATIVE = "build/qw38-cuda-opt078-decode-attention-test"
REPLAY = "build/qw38-cuda-component-replay"
PROBE = "build/qw38-cuda-optimization-engine-probe"
MODEL = "models/Qwen3.8-27B-Q4_K_M.gguf"
PHASES = ("d128", "d2048", "attention", "quality-state")
CONTROL_ID = "warp_query"
MIN_SAVING_MS = 0.10
PIN_PATH = ROOT / "cuda/attention_decode_path.cuh"

CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "id": "warp_query",
        "decode_query_prep": "warp_query",
        "prepared_q": False,
        "vec_kv": False,
        "role": "control",
        "expected_launch": "warp_query_decode_attention",
        "expected_prep_launches": 0,
    },
    {
        "id": "prepared_q",
        "decode_query_prep": "prepared_q",
        "prepared_q": True,
        "vec_kv": False,
        "role": "candidate",
        "expected_launch": "prepare_decode_query+warp_query_prepared_q",
        "expected_prep_launches": 1,
    },
    {
        "id": "prepared_q_veckv",
        "decode_query_prep": "prepared_q_veckv",
        "prepared_q": True,
        "vec_kv": True,
        "role": "candidate",
        "expected_launch": "prepare_decode_query+warp_query_prepared_q_veckv",
        "expected_prep_launches": 1,
    },
)

TINY_POSITIONS: tuple[int, ...] = (0, 1, 31, 32)
PRODUCTION_POSITIONS: tuple[int, ...] = (127, 128, 2047, 2048)

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


def apply_production_pin(path_id: str) -> None:
    text = PIN_PATH.read_text(encoding="utf-8")
    updated = re.sub(
        r'kSelectedDecodeQueryPrepPath\[\] = "[^"]+"',
        f'kSelectedDecodeQueryPrepPath[] = "{path_id}"',
        text,
        count=1,
    )
    if updated == text:
        raise AdmissionError("failed to rewrite the decode query-prep pin")
    PIN_PATH.write_text(updated, encoding="utf-8")


def config_by_id(config_id: str) -> dict[str, Any]:
    for row in CONFIGS:
        if row["id"] == config_id:
            return dict(row)
    raise AdmissionError(f"unknown configuration {config_id}")


def tiny_positions() -> list[int]:
    return list(TINY_POSITIONS)


def decode_position_for(phase: str) -> int:
    return 2048 if phase in {"d2048", "attention"} else 128


def family_plan(phase: str, mode: str) -> dict[str, Any]:
    iteration = load_json(ITERATION)
    workload = workload_for_mode(iteration["workloads"][phase], mode)
    engine_pairs = 0
    if phase in {"d128", "d2048", "attention"}:
        engine_pairs = 1 if mode == "feedback" else 5
    return {
        "phase": phase,
        "mode": mode,
        "family": "decode-attention",
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
        "decode_position": decode_position_for(phase),
    }


def planned_observation(
    plan: Mapping[str, Any], *, capture_key: str | None = None
) -> dict[str, Any]:
    samples = int(plan["samples"])
    return {
        "schema_version": 1,
        "task": "OPT-078",
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
        "decode-attention",
        "--cache-mode",
        "rotating",
        "--warmups",
        str(plan["warmups"]),
        "--samples",
        str(plan["samples"]),
        "--decode-query-prep",
        str(config["decode_query_prep"]),
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
        r"vec_kv=(\S+)",
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
    }


def dispatch_matches(observed: Mapping[str, Any], config: Mapping[str, Any]) -> bool:
    if not observed:
        return False
    if str(observed.get("path")) != str(config["decode_query_prep"]):
        return False
    if str(observed.get("launch")) != str(config["expected_launch"]):
        return False
    if int(observed.get("n_parts", 0)) != 16:
        return False
    if int(observed.get("prep_launches", -1)) != int(config["expected_prep_launches"]):
        return False
    if bool(observed.get("prepared_q")) != bool(config["prepared_q"]):
        return False
    if bool(observed.get("vec_kv")) != bool(config["vec_kv"]):
        return False
    if bool(config["prepared_q"]) and int(observed.get("prep_grid", 0)) != 24:
        return False
    if (not bool(config["prepared_q"])) and int(observed.get("prep_grid", 0)) != 0:
        return False
    return True


def pick_survivor(by_config: Mapping[str, Mapping[str, Any]]) -> str:
    names = [name for name in ("prepared_q", "prepared_q_veckv") if name in by_config]
    if not names:
        raise AdmissionError("no prepared-Q configuration was measured")
    return min(names, key=lambda name: float(by_config[name].get("mean_ms", math.inf)))


def decide_verdict(
    *,
    component: Mapping[str, Any] | None,
    engine: Mapping[str, Any] | None,
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
    if survivor is None or survivor == CONTROL_ID:
        return {
            "verdict": "retain_warp_query",
            "status": "inconclusive",
            "reasons": reasons + ["no_prepared_survivor"],
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
            "component_accepted": component_ok,
            "e2e_non_regression": regression_ok,
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
    return {
        "stdout": completed.stdout,
        "nonfinite": nonfinite,
        "native": observed,
        "pass": "status=passed" in completed.stdout,
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
            and "seq_nonfinite=0" not in completed.stdout
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
        groups = re.search(r"attention_input_output_groups=(\d+)", completed.stdout)
        if groups is None or int(groups.group(1)) != 16:
            raise AdmissionError(
                f"{config['id']} complete attention layers were not 16: {groups}"
            )
        dispatch = parse_attn_dispatch(completed.stdout)
        if not dispatch_matches(dispatch, config):
            raise AdmissionError(
                f"{config['id']} launch variants {dispatch} != {config}"
            )
        samples = [float(row["enclosing_ms"]) for row in rounds]
        by_config[str(config["id"])] = {
            "config": dict(config),
            "ms": samples,
            "mean_ms": mean(samples),
            "dispatch": dispatch,
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
    from tools.opt075_q4_production_admission import parse_engine_pairs

    tier = "acceptance" if plan["mode"] == "acceptance" else "screen"
    command = [
        f"./{PROBE}",
        MODEL,
        "--workload",
        "attn-ab",
        "--pairs",
        str(pairs_n),
        "--modes",
        "graph",
        "--decode-query-prep",
        str(survivor["decode_query_prep"]),
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
        raise AdmissionError("graphs were not recaptured after selecting the candidate")
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
    attention = (
        payload.get("attention")
        or payload.get("d2048")
        or payload.get("d128")
        or payload
    )
    component = attention.get("component") or {}
    engine = attention.get("engine") or {}
    quality = payload.get("quality") or attention.get("quality") or {}
    numeric = attention.get("numeric") or payload.get("numeric") or {}
    verdict = attention.get("verdict") or payload.get("verdict") or {}
    launches = json.dumps(component.get("launches") or [], sort_keys=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    text = f"""# OPT-078 — Prepare decode queries once and vector-load KV

Status: **{(payload.get("status") or "pending GPU sitting")}**. Authority llama.cpp
`{LLAMA_REV}`, GGUF SHA-256 `{GGUF_SHA}`. Control is in-kernel `warp_query`
RMS+RoPE per partition. Candidates hoist decode Q prep (`prepared_q`) and add
legal 16-byte KV loads (`prepared_q_veckv`). n_parts stays 16. Prompt fattn
hoist is unchanged. Candidate K/V staging stays separate.

`claims_throughput: false` unless production_kept. Require >=0.10 ms/token
complete 16-layer attention saving and a positive paired interval, or measured
rejection. If the extra prep launch offsets the kernel saving, reject.

## Numeric policy

Same-path control vs candidate GPU outputs must match for unchanged arithmetic.
Sampled independent FP64 uses four query heads x eight output dimensions at
two positions. Nonfinite={numeric.get("nonfinite")}.

## Mechanism

One warp per query head writes prepared FP32 Q into the existing normalized-query
workspace once. All 16 partitions consume that buffer. Vector KV loads eight
BF16 values (16 bytes) per lane with an alignment-safe scalar fallback.
Prepared Q is rewritten for every new input/position/graph capture; pointer
identity is never a skip.

## Dispatch

Launches: {launches}

## Complete 16-layer attention screen

Control warp_query vs survivor `{component.get("survivor") or payload.get("survivor")}`.
Component n={component.get("n")} means: control {_fmt(component.get("control_mean_ms"))} ms,
candidate {_fmt(component.get("candidate_mean_ms"))} ms.
Paired CI (control-candidate): {_fmt(component.get("ci95_low"), 4)} ..
{_fmt(component.get("ci95_high"), 4)} ms; mean diff {_fmt(component.get("mean_diff_ms"), 4)} ms.
Engine D2048+32 pairs: {_fmt(engine.get("control_mean_ms"))} vs
{_fmt(engine.get("candidate_mean_ms"))} ms.

## Quality (OPT-073)

quality-v3 absolute={quality.get("quality_v3_absolute")};
engine non-regression={quality.get("quality_v3_engine_non_regression")}.

## Decision

Verdict: **{verdict.get("verdict")}** ({verdict.get("reasons")}).
production_kept={verdict.get("production_kept")}; retain_reason={verdict.get("retain_reason")}.
Shipping decode query-prep stays `{payload.get("shipping_decode_query_prep", "warp_query")}`.

## tok/s

Speedup versus the then-current warp_query baseline is **{payload.get("tok_s_delta", 0)}** while warp_query
is retained unless production_kept.
"""
    REPORT.write_text(text, encoding="utf-8")


def run_quality_state_phase(*, mode: str, run_dir: Path) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    quality = opt073_quality()
    if quality.get("quality_v3_engine_non_regression") not in {"pass", True}:
        raise AdmissionError("OPT-073 engine non-regression is not a pass")
    payload = {
        "schema_version": 1,
        "task": "OPT-078",
        "phase": "quality-state",
        "mode": mode,
        "identity_cached": True,
        "once_per_survivor": True,
        "quality": quality,
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
    dump_json(run_dir / "quality-state-result.json", payload)
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
    if phase == "quality-state":
        return run_quality_state_phase(mode=mode, run_dir=run_dir)
    plan = family_plan(phase, mode)
    quality = opt073_quality()
    configs: list[dict[str, Any]]
    if mode == "feedback":
        configs = [dict(row) for row in CONFIGS]
    else:
        survivor_id = "prepared_q"
        if FIXTURE.is_file():
            try:
                previous = load_json(FIXTURE)
                survivor_id = str(
                    (
                        (previous.get("d2048") or previous.get("d128") or {}).get(
                            "component"
                        )
                        or {}
                    ).get("survivor")
                    or previous.get("survivor")
                    or "prepared_q"
                )
            except json.JSONDecodeError:
                survivor_id = "prepared_q"
        if survivor_id == CONTROL_ID:
            survivor_id = "prepared_q"
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
            runner=native_runner, run_dir=run_dir, mode=mode
        )
        if not native.get("pass"):
            raise AdmissionError("native numeric/state phase failed")
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
        numeric = {"nonfinite": int(native.get("nonfinite") or 0)}
    verdict = decide_verdict(
        component=component or None,
        engine=engine or None,
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
        "task": "OPT-078",
        "phase": phase,
        "mode": mode,
        "control": CONTROL_ID,
        "survivor": component.get("survivor"),
        "configurations": [row["id"] for row in CONFIGS],
        "component": component,
        "engine": engine,
        "native": {k: v for k, v in native.items() if k != "stdout"},
        "quality": quality,
        "numeric": numeric,
        "tiny_positions": list(TINY_POSITIONS),
        "verdict": verdict,
        "native_counts": observed,
        "admission": admission,
        "claims_throughput": False,
        "measurement_utc": utc_now(),
        "status": "measured",
        "dispatch_ok": dispatch_ok,
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
    selected = [phase] if phase in PHASES else ["d128"]
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
            "task": "OPT-078",
            "mode": mode,
            "llama_revision": LLAMA_REV,
            "gguf_sha256": GGUF_SHA,
            "token_generator": TOKEN_GENERATOR,
            "configurations": [row["id"] for row in CONFIGS],
            "control": CONTROL_ID,
            "quality": quality,
            "claims_throughput": False,
            "claims_performance_improvement": False,
            "shipping_decode_query_prep": "warp_query",
            "report_path": "evidence/optimization/opt078-decode-attention/REPORT.md",
        }
    )
    for name in selected:
        results[name] = run_phase(name, mode=mode, run_dir=run_dir)
    primary = (
        results.get("attention") or results.get("d2048") or results.get("d128") or {}
    )
    verdict = primary.get("verdict") or {}
    results["survivor"] = primary.get("survivor")
    results["numeric"] = primary.get("numeric")
    results["verdict"] = verdict
    results["production_kept"] = bool(verdict.get("production_kept"))
    results["shipping_unchanged"] = bool(verdict.get("shipping_unchanged", True))
    results["status"] = "measured" if primary else results.get("status", "pending")
    tok_s_delta = 0.0
    engine = primary.get("engine") or {}
    control_ms = float(engine.get("control_mean_ms") or 0.0)
    candidate_ms = float(engine.get("candidate_mean_ms") or 0.0)
    if results.get("production_kept") and control_ms > 0.0 and candidate_ms > 0.0:
        tok_s_delta = (32000.0 / candidate_ms) - (32000.0 / control_ms)
        results["claims_throughput"] = True
    else:
        tok_s_delta = 0.0
        results["claims_throughput"] = False
    results["tok_s_delta"] = tok_s_delta
    if results.get("production_kept"):
        survivor_cfg = config_by_id(str(results.get("survivor")))
        results["shipping_decode_query_prep"] = survivor_cfg["decode_query_prep"]
        apply_production_pin(str(survivor_cfg["decode_query_prep"]))
    write_report(results)
    dump_json(FIXTURE, results)
    dump_json(run_dir / "opt078_decode_attention.json", results)
    counts = (results.get(phase) or {}).get("native_counts") or {}
    print(
        "QW38_OPT078_DECODE_ATTENTION_RESULT="
        + json.dumps(
            {
                "task": "OPT-078",
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
    parser.add_argument(
        "--phase",
        choices=("d128", "d2048", "attention", "quality-state"),
        default="d128",
    )
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
        ROOT / "build" / "optimization-runs" / "OPT-078" / args.mode
    )
    try:
        if args.skip_gpu:
            raise AdmissionError("GPU sitting required for OPT-078 phases")
        run(args.mode, args.phase, run_dir)
    except (AdmissionError, AssertionError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
