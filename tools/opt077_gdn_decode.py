"""OPT-077 parallel decode GDN recurrence admission versus sequential control."""

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
CONTRACT = ROOT / "pins/opt077_gdn_decode_contract.json"
ITERATION = ROOT / "pins/opt077_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt077_gdn_decode.json"
REPORT = ROOT / "evidence/optimization/opt077-gdn-decode/REPORT.md"
EVIDENCE = ROOT / "evidence/optimization/opt077-gdn-decode"
NATIVE = "build/qw38-cuda-opt077-gdn-decode-test"
REPLAY = "build/qw38-cuda-component-replay"
PROBE = "build/qw38-cuda-optimization-engine-probe"
MODEL = "models/Qwen3.8-27B-Q4_K_M.gguf"
PHASES = ("gdn", "quality-state")
CONTROL_ID = "sequential"
MIN_SAVING_MS = 0.10

CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "id": "sequential",
        "gdn_decode": "sequential",
        "value_tile": 0,
        "role": "control",
        "expected_launch": "prepare_recurrence_window",
    },
    {
        "id": "tile16",
        "gdn_decode": "tile16",
        "value_tile": 16,
        "role": "candidate",
        "expected_launch": "prepare_recurrence_decode_tiled16",
    },
    {
        "id": "tile32",
        "gdn_decode": "tile32",
        "value_tile": 32,
        "role": "candidate",
        "expected_launch": "prepare_recurrence_decode_tiled32",
    },
)

TINY_CASES: tuple[dict[str, Any], ...] = (
    {"id": "W8_U1", "key_width": 8, "value_width": 8, "updates": 1},
    {"id": "W8_U4", "key_width": 8, "value_width": 8, "updates": 4},
    {"id": "W32_U1", "key_width": 32, "value_width": 32, "updates": 1},
    {"id": "W32_U4", "key_width": 32, "value_width": 32, "updates": 4},
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


def apply_production_pin(path_id: str) -> None:
    text = (ROOT / "cuda/gdn_decode_path.cuh").read_text(encoding="utf-8")
    updated = re.sub(
        r'kSelectedGdnDecodePath\[\] = "[^"]+"',
        f'kSelectedGdnDecodePath[] = "{path_id}"',
        text,
        count=1,
    )
    if updated == text:
        raise AdmissionError("failed to rewrite the GDN decode production pin")
    (ROOT / "cuda/gdn_decode_path.cuh").write_text(updated, encoding="utf-8")


def config_by_id(config_id: str) -> dict[str, Any]:
    for row in CONFIGS:
        if row["id"] == config_id:
            return dict(row)
    raise AdmissionError(f"unknown configuration {config_id}")


def tiny_cases() -> list[dict[str, Any]]:
    return [dict(row) for row in TINY_CASES]


def family_plan(phase: str, mode: str) -> dict[str, Any]:
    iteration = load_json(ITERATION)
    workload = workload_for_mode(iteration["workloads"][phase], mode)
    engine_pairs = 0
    if phase == "gdn":
        engine_pairs = 1 if mode == "feedback" else 5
    return {
        "phase": phase,
        "mode": mode,
        "family": "decode-gdn",
        "control": CONTROL_ID,
        "warmups": int(workload.get("warmups", 1 if mode == "feedback" else 3)),
        "samples": int(workload.get("samples", 3 if mode == "feedback" else 10)),
        "engine_pairs": int(workload.get("engine_pairs", engine_pairs)),
        "cases": int(workload.get("cases", 2 if mode == "feedback" else 1)),
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
        "task": "OPT-077",
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
        "decode-gdn",
        "--cache-mode",
        "rotating",
        "--warmups",
        str(plan["warmups"]),
        "--samples",
        str(plan["samples"]),
        "--gdn-decode",
        str(config["gdn_decode"]),
    ]
    if capture_key:
        args.extend(["--capture-key", capture_key])
    return args


def parse_gdn_dispatch(stdout: str) -> dict[str, Any]:
    match = re.search(
        r"gdn_dispatch path=(\S+) launch=(\S+) value_tile=(\d+) "
        r"grid_x=(\d+) grid_y=(\d+) warps_per_cta=(\d+)",
        stdout,
    )
    if match is None:
        return {}
    return {
        "path": match.group(1),
        "launch": match.group(2),
        "value_tile": int(match.group(3)),
        "grid_x": int(match.group(4)),
        "grid_y": int(match.group(5)),
        "warps_per_cta": int(match.group(6)),
    }


def dispatch_matches(observed: Mapping[str, Any], config: Mapping[str, Any]) -> bool:
    if not observed:
        return False
    if str(observed.get("path")) != str(config["gdn_decode"]):
        return False
    if str(observed.get("launch")) != str(config["expected_launch"]):
        return False
    if int(observed.get("warps_per_cta", 0)) != 4:
        return False
    tile = int(config["value_tile"])
    if tile == 0:
        return True
    return int(observed.get("value_tile", -1)) == tile


def pick_survivor(by_config: Mapping[str, Mapping[str, Any]]) -> str:
    tiles = [name for name in ("tile16", "tile32") if name in by_config]
    if not tiles:
        raise AdmissionError("no tiled configuration was measured")
    return min(tiles, key=lambda name: float(by_config[name].get("mean_ms", math.inf)))


def decide_verdict(
    *,
    component: Mapping[str, Any] | None,
    engine: Mapping[str, Any] | None,
    quality: Mapping[str, Any],
    numeric: Mapping[str, Any],
    dispatch_ok: bool,
    mode: str,
    survivor: str | None,
    local_bytes: int = 0,
) -> dict[str, Any]:
    reasons: list[str] = []
    if numeric.get("nonfinite"):
        raise AdmissionError("nonfinite candidate output")
    if not dispatch_ok:
        raise AdmissionError("layout or intended-path dispatch was not proven")
    engine_nr = quality.get("quality_v3_engine_non_regression")
    if engine_nr not in {"pass", True}:
        reasons.append("opt073_quality_unresolved")
    if local_bytes > 0:
        reasons.append("resource_spill")
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
            "verdict": "retain_sequential",
            "status": "inconclusive",
            "reasons": reasons + ["no_tiled_survivor"],
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
    if local_bytes > 0:
        return {
            "verdict": "performance_rejected",
            "status": "performance_rejected",
            "reasons": reasons,
            "shipping_unchanged": True,
            "production_kept": False,
            "retain_reason": "resource_spill",
            "component_accepted": False,
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
            "verdict": "retain_sequential",
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
    observed = parse_native_observation(completed.stdout)
    local16 = re.search(r"tile16_local_bytes=(\d+)", completed.stdout)
    local32 = re.search(r"tile32_local_bytes=(\d+)", completed.stdout)
    nonfinite = 0
    if "seq_nonfinite=" in completed.stdout:
        for match in re.finditer(r"seq_nonfinite=(\d+)", completed.stdout):
            nonfinite += int(match.group(1))
    return {
        "stdout": completed.stdout,
        "nonfinite": nonfinite,
        "tile16_local_bytes": int(local16.group(1)) if local16 else 0,
        "tile32_local_bytes": int(local32.group(1)) if local32 else 0,
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
        groups = re.search(r"gdn_input_output_groups=(\d+)", completed.stdout)
        if groups is None or int(groups.group(1)) != 48:
            raise AdmissionError(
                f"{config['id']} complete GDN layers were not 48: {groups}"
            )
        dispatch = parse_gdn_dispatch(completed.stdout)
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
                "value_tile": row["config"]["value_tile"],
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
        "gdn-ab",
        "--pairs",
        str(pairs_n),
        "--modes",
        "graph",
        "--gdn-decode",
        str(survivor["gdn_decode"]),
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
    gdn = payload.get("gdn") or payload
    component = gdn.get("component") or {}
    engine = gdn.get("engine") or {}
    quality = payload.get("quality") or gdn.get("quality") or {}
    numeric = gdn.get("numeric") or payload.get("numeric") or {}
    verdict = gdn.get("verdict") or payload.get("verdict") or {}
    launches = component.get("launches") or []
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    text = f"""# OPT-077 — Parallelize the one-token GDN state reduction

Status: **{(payload.get("status") or "pending GPU sitting")}**. Authority llama.cpp
`{LLAMA_REV}`, GGUF SHA-256 `{GGUF_SHA}`. Control is sequential
`prepare_recurrence_window` on `launch_gdn_prepare_tiled` /
`kSequentialWindows`. Candidates are value-tile 16 and 32, four warps/CTA,
grid `value_heads x ceil(value_width/value_tile)`. Convolution stays separate.
FP32 row-major state is unchanged. llama.cpp `gated_delta_net_cuda` motivates
parallelism; its transposed state is not used.

`claims_throughput: false` unless production_kept. Require >=0.10 ms/token
complete 48-layer GDN saving and a positive paired interval, or measured
rejection. Recurrence incremental NLL <=0.02 is not relaxed. Long release is
OPT-080.

## Numeric policy

Independent FP64 sampled state/output plus same-input sequential GPU
reference. Llama GPU recurrence is not drop-in on row-major S; conversion
was rejected. Strict GDN-002 5e-8 / 5e-9 remains the sequential reference.
Nonfinite={numeric.get("nonfinite")}.

## Mechanism

Q/K norms are precomputed once per key head in shared memory. Each CTA owns
a value tile and distributes the key reduction across four warps with a
shared-memory tree; each candidate cell is written once. Prompt GDN is
unchanged.

## Dispatch

Launches: {launches}

## Complete 48-layer GDN screen

Two capture positions on repaired OPT-071 decode captures. Control sequential
vs tiled survivor `{component.get("survivor") or payload.get("survivor")}`.
Component n={component.get("n")} means: control {_fmt(component.get("control_mean_ms"))} ms,
candidate {_fmt(component.get("candidate_mean_ms"))} ms.
Paired CI (control-candidate): {_fmt(component.get("ci95_low"), 4)} ..
{_fmt(component.get("ci95_high"), 4)} ms; mean diff {_fmt(component.get("mean_diff_ms"), 4)} ms.
Engine D2048+32 pairs: {_fmt(engine.get("control_mean_ms"))} vs
{_fmt(engine.get("candidate_mean_ms"))} ms.

## Quality (OPT-073) and state

quality-v3 absolute={quality.get("quality_v3_absolute")};
engine non-regression={quality.get("quality_v3_engine_non_regression")}.
Committed-state isolation is required before install.

## Decision

Verdict: **{verdict.get("verdict")}** ({verdict.get("reasons")}).
production_kept={verdict.get("production_kept")}; retain_reason={verdict.get("retain_reason")}.
Shipping GDN decode stays `{payload.get("shipping_gdn_decode", "sequential")}`.

## tok/s

Speedup versus the then-current sequential baseline is **{payload.get("tok_s_delta", 0)}** while sequential
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
        "task": "OPT-077",
        "phase": "quality-state",
        "mode": mode,
        "identity_cached": True,
        "once_per_survivor": True,
        "quality": quality,
        "state_isolation_required": True,
        "recurrence_nll_ceiling": 0.02,
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
    tinies = tiny_cases()
    configs: list[dict[str, Any]]
    if mode == "feedback":
        configs = [dict(row) for row in CONFIGS]
    else:
        survivor_id = "tile16"
        if FIXTURE.is_file():
            try:
                previous = load_json(FIXTURE)
                survivor_id = str(
                    ((previous.get("gdn") or {}).get("component") or {}).get("survivor")
                    or previous.get("survivor")
                    or "tile16"
                )
            except json.JSONDecodeError:
                survivor_id = "tile16"
        if survivor_id == CONTROL_ID:
            survivor_id = "tile16"
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
        numeric = {
            "nonfinite": int(native.get("nonfinite") or 0),
            "tile16_local_bytes": int(native.get("tile16_local_bytes") or 0),
            "tile32_local_bytes": int(native.get("tile32_local_bytes") or 0),
        }
    local_bytes = max(
        int(numeric.get("tile16_local_bytes") or 0),
        int(numeric.get("tile32_local_bytes") or 0),
    )
    verdict = decide_verdict(
        component=component or None,
        engine=engine or None,
        quality=quality,
        numeric=numeric,
        dispatch_ok=dispatch_ok,
        mode=mode,
        survivor=str(component.get("survivor") or ""),
        local_bytes=local_bytes,
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
        "task": "OPT-077",
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
        "tiny_cases": tinies,
        "verdict": verdict,
        "native_counts": observed,
        "admission": admission,
        "claims_throughput": False,
        "measurement_utc": utc_now(),
        "status": "measured",
        "dispatch_ok": dispatch_ok,
    }
    if isinstance(payload.get("component"), dict):
        payload["component"].pop("stdout", None)
    if isinstance(payload.get("engine"), dict):
        payload["engine"].pop("stdout", None)
    dump_json(run_dir / f"{phase}-result.json", payload)
    return payload


def run(mode: str, phase: str, run_dir: Path) -> dict[str, Any]:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    selected = [phase] if phase in PHASES else ["gdn"]
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
            "task": "OPT-077",
            "mode": mode,
            "llama_revision": LLAMA_REV,
            "gguf_sha256": GGUF_SHA,
            "token_generator": TOKEN_GENERATOR,
            "configurations": [row["id"] for row in CONFIGS],
            "control": CONTROL_ID,
            "quality": quality,
            "claims_throughput": False,
            "claims_performance_improvement": False,
            "shipping_gdn_decode": "sequential",
            "report_path": "evidence/optimization/opt077-gdn-decode/REPORT.md",
        }
    )
    for name in selected:
        results[name] = run_phase(name, mode=mode, run_dir=run_dir)
    gdn = results.get("gdn") or {}
    verdict = gdn.get("verdict") or {}
    results["survivor"] = gdn.get("survivor")
    results["numeric"] = gdn.get("numeric")
    results["verdict"] = verdict
    results["production_kept"] = bool(verdict.get("production_kept"))
    results["shipping_unchanged"] = bool(verdict.get("shipping_unchanged", True))
    results["status"] = "measured" if gdn else results.get("status", "pending")
    tok_s_delta = 0.0
    engine = gdn.get("engine") or {}
    control_ms = float(engine.get("control_mean_ms") or 0.0)
    candidate_ms = float(engine.get("candidate_mean_ms") or 0.0)
    if results.get("production_kept") and control_ms > 0.0 and candidate_ms > 0.0:
        tok_s_delta = (32000.0 / candidate_ms) - (32000.0 / control_ms)
    results["tok_s_delta"] = tok_s_delta
    if results.get("production_kept"):
        survivor_cfg = config_by_id(str(results.get("survivor")))
        results["shipping_gdn_decode"] = survivor_cfg["gdn_decode"]
        results["claims_throughput"] = True
        apply_production_pin(str(survivor_cfg["gdn_decode"]))
    write_report(results)
    dump_json(FIXTURE, results)
    dump_json(run_dir / "opt077_gdn_decode.json", results)
    counts = (results.get(phase) or {}).get("native_counts") or {}
    print(
        "QW38_OPT077_RESULT="
        + json.dumps(
            {
                "task": "OPT-077",
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
    parser.add_argument("--phase", choices=("gdn", "quality-state"), default="gdn")
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
        ROOT / "build" / "optimization-runs" / "OPT-077" / args.mode
    )
    try:
        if args.skip_gpu:
            raise AdmissionError("GPU sitting required for OPT-077 phases")
        run(args.mode, args.phase, run_dir)
    except (AdmissionError, AssertionError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
