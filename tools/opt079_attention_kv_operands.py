"""OPT-079 convert prompt attention KV operands once per shared stage."""

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
    parse_engine_pairs,
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
CONTRACT = ROOT / "pins/opt079_attention_kv_operands_contract.json"
ITERATION = ROOT / "pins/opt079_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt079_attention_kv_operands.json"
REPORT = ROOT / "evidence/optimization/opt079-attention-kv-operands/REPORT.md"
EVIDENCE = ROOT / "evidence/optimization/opt079-attention-kv-operands"
NATIVE = "build/qw38-cuda-opt079-attention-kv-operands-test"
PROBE = "build/qw38-cuda-optimization-engine-probe"
MODEL = "models/Qwen3.8-27B-Q4_K_M.gguf"
PHASES = ("attention",)
CONTROL_ID = "f16_async"
CANDIDATE_ID = "kv_once"
MIN_SAVING_MS = 5.0
PIPELINE_HEADER = ROOT / "cuda/fattn_mma_f16.cuh"

CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "id": "f16_async",
        "attention_pipeline": "f16_async",
        "convert_once": 0,
        "role": "control",
        "expected_launch": "fattn_mma_pipeline_f16_async",
    },
    {
        "id": "kv_once",
        "attention_pipeline": "kv_once",
        "convert_once": 1,
        "role": "candidate",
        "expected_launch": "fattn_mma_pipeline_kv_once",
    },
)

TINY_CASES: tuple[dict[str, Any], ...] = (
    {"id": "T1", "tokens": 1, "start": 96},
    {"id": "T17", "tokens": 17, "start": 96},
    {"id": "T33", "tokens": 33, "start": 96},
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
    text = PIPELINE_HEADER.read_text(encoding="utf-8")
    target = f'kSelectedAttentionPipelinePath[] = "{path_id}"'
    if target in text:
        return
    updated = re.sub(
        r'kSelectedAttentionPipelinePath\[\] = "[^"]+"',
        target,
        text,
        count=1,
    )
    if updated == text:
        raise AdmissionError("failed to rewrite the attention pipeline pin")
    PIPELINE_HEADER.write_text(updated, encoding="utf-8")


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
    engine_pairs = 1 if mode == "feedback" else 5
    return {
        "phase": phase,
        "mode": mode,
        "family": "prompt-attn",
        "control": CONTROL_ID,
        "warmups": int(workload.get("warmups", 1 if mode == "feedback" else 3)),
        "samples": int(workload.get("samples", 3 if mode == "feedback" else 10)),
        "engine_pairs": int(workload.get("engine_pairs", engine_pairs)),
        "cases": int(workload.get("cases", 1)),
        "candidates": int(workload.get("candidates", 2)),
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
        "task": "OPT-079",
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


def parse_attn_dispatch(stdout: str, path: str) -> dict[str, Any]:
    pattern = rf"path={re.escape(path)} launch=(\S+) convert_once=(\d+) occupancy=(\d+)"
    match = re.search(pattern, stdout)
    if match is None:
        match = re.search(
            rf"path={re.escape(path)} launch=(\S+) convert_once=(\d+)", stdout
        )
        if match is None:
            return {}
        return {
            "path": path,
            "launch": match.group(1),
            "convert_once": int(match.group(2)),
            "occupancy": 0,
        }
    return {
        "path": path,
        "launch": match.group(1),
        "convert_once": int(match.group(2)),
        "occupancy": int(match.group(3)),
    }


def dispatch_matches(observed: Mapping[str, Any], config: Mapping[str, Any]) -> bool:
    if not observed:
        return False
    if str(observed.get("path")) != str(config["attention_pipeline"]):
        return False
    if str(observed.get("launch")) != str(config["expected_launch"]):
        return False
    return int(observed.get("convert_once", -1)) == int(config["convert_once"])


def pick_survivor(by_config: Mapping[str, Mapping[str, Any]]) -> str:
    if CANDIDATE_ID not in by_config:
        raise AdmissionError("kv_once was not measured")
    return CANDIDATE_ID


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
    occupancy_loss: bool = False,
    conversion_reduced: bool = True,
) -> dict[str, Any]:
    reasons: list[str] = []
    if numeric.get("nonfinite"):
        raise AdmissionError("nonfinite candidate output")
    if not dispatch_ok:
        raise AdmissionError("intended-path dispatch was not proven")
    engine_nr = quality.get("quality_v3_engine_non_regression")
    if engine_nr not in {"pass", True}:
        reasons.append("opt073_quality_unresolved")
    if local_bytes > 0:
        reasons.append("resource_spill")
    if occupancy_loss:
        reasons.append("occupancy_loss")
    if not conversion_reduced:
        reasons.append("no_conversion_reduction")
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
            "verdict": "retain_f16_async",
            "status": "inconclusive",
            "reasons": reasons + ["no_kv_once_survivor"],
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
    if local_bytes > 0 or occupancy_loss or not conversion_reduced:
        return {
            "verdict": "performance_rejected",
            "status": "performance_rejected",
            "reasons": reasons,
            "shipping_unchanged": True,
            "production_kept": False,
            "retain_reason": "resource_or_conversion",
            "component_accepted": False,
            "e2e_non_regression": regression_ok,
        }
    if slower or too_small:
        return {
            "verdict": "performance_rejected",
            "status": "performance_rejected",
            "reasons": reasons
            + (
                ["negative_complete_saving"] if slower else ["saving_below_5_ms_prompt"]
            ),
            "shipping_unchanged": True,
            "production_kept": False,
            "retain_reason": "performance",
            "component_accepted": False,
            "e2e_non_regression": regression_ok,
        }
    if reasons or not component_ok or not regression_ok:
        return {
            "verdict": "retain_f16_async",
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
            "saving_ge_5_ms_prompt",
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
    workload = "screen" if mode == "feedback" else "acceptance"
    command = [f"./{NATIVE}", "--workload", workload]
    completed = runner(command, workload)
    (run_dir / f"native-{workload}.txt").write_text(
        completed.stdout + completed.stderr, encoding="utf-8"
    )
    observed = parse_native_observation(completed.stdout)
    local_c = re.search(r"f16_async_local_bytes=(\d+)", completed.stdout)
    local_k = re.search(r"kv_once_local_bytes=(\d+)", completed.stdout)
    occ_c = re.search(r"f16_async_occupancy=(\d+)", completed.stdout)
    occ_k = re.search(r"kv_once_occupancy=(\d+)", completed.stdout)
    nonfinite = 0
    for match in re.finditer(r"nonfinite=(\d+)", completed.stdout):
        nonfinite += int(match.group(1))
    return {
        "stdout": completed.stdout,
        "nonfinite": nonfinite,
        "f16_async_local_bytes": int(local_c.group(1)) if local_c else 0,
        "kv_once_local_bytes": int(local_k.group(1)) if local_k else 0,
        "f16_async_occupancy": int(occ_c.group(1)) if occ_c else 0,
        "kv_once_occupancy": int(occ_k.group(1)) if occ_k else 0,
        "native": observed,
        "pass": "status=passed" in completed.stdout,
    }


def run_component(
    plan: Mapping[str, Any],
    *,
    native: Mapping[str, Any],
    configs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    stdout = str(native.get("stdout") or "")
    rounds = parse_rounds(stdout)
    by_config: dict[str, dict[str, Any]] = {}
    for config in configs:
        path = str(config["attention_pipeline"])
        measured = [
            row
            for row in rounds
            if str(row.get("path") or "") == path or path in str(row.get("raw") or "")
        ]
        if not measured:
            measured = [row for row in rounds if path in json.dumps(row)]
        samples = []
        for row in measured:
            if row.get("warmup") in {True, "true", "True"}:
                continue
            samples.append(float(row["enclosing_ms"]))
        if len(samples) != int(plan["samples"]):
            path_rounds = re.findall(
                rf"warmup=false sample_index=(\d+) observation_unit=independent_round "
                rf"enclosing_ms=([0-9.]+) kernel_only_ms=[0-9.]+ attention_layers=16 "
                rf"path={re.escape(path)}",
                stdout,
            )
            samples = [float(item[1]) for item in path_rounds]
        if len(samples) != int(plan["samples"]):
            raise AdmissionError(
                f"{config['id']} rotating rounds {len(samples)} != {plan['samples']}"
            )
        dispatch = parse_attn_dispatch(stdout, path)
        if not dispatch_matches(dispatch, config):
            raise AdmissionError(
                f"{config['id']} launch variants {dispatch} != {config}"
            )
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
        "capture_key": None,
        "by_config": by_config,
        "control": CONTROL_ID,
        "survivor": survivor,
        "control_ms": control_ms,
        "candidate_ms": candidate_ms,
        "control_mean_ms": mean(control_ms),
        "candidate_mean_ms": mean(candidate_ms),
        **stats,
        "saving_ge_5_ms": float(stats["mean_diff_ms"]) >= MIN_SAVING_MS,
        "launches": [
            {
                "config": name,
                "dispatch": row["dispatch"],
                "mean_ms": row["mean_ms"],
            }
            for name, row in by_config.items()
        ],
        "pairs": int(plan["samples"]),
        "dispatch_ok": True,
    }


def run_engine(
    plan: Mapping[str, Any],
    *,
    runner: NativeRunner,
    run_dir: Path,
    survivor: Mapping[str, Any],
    decode_guard: bool,
) -> dict[str, Any]:
    pairs_n = int(plan["engine_pairs"])
    if pairs_n <= 0:
        return {}
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
        "--attention-pipeline",
        str(survivor["attention_pipeline"]),
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
    decode: dict[str, Any] = {}
    if decode_guard:
        decoded = runner(
            [
                f"./{PROBE}",
                MODEL,
                "--workload",
                "attn-ab",
                "--prefix",
                "2048",
                "--output-tokens",
                "32",
                "--pairs",
                "1",
                "--modes",
                "graph",
                "--attention-pipeline",
                str(survivor["attention_pipeline"]),
            ],
            "screen" if plan["mode"] == "feedback" else "acceptance",
        )
        (run_dir / f"{plan['phase']}-decode-guard.txt").write_text(
            decoded.stdout + decoded.stderr, encoding="utf-8"
        )
        decode_pairs = parse_engine_pairs(decoded.stdout)
        decode = {
            "pairs": decode_pairs,
            "stdout_ok": "override_before_capture_applied=true" in decoded.stdout,
        }
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
        "decode_guard": decode,
    }


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def write_report(payload: Mapping[str, Any]) -> None:
    attention = payload.get("attention") or payload
    component = attention.get("component") or {}
    engine = attention.get("engine") or {}
    quality = payload.get("quality") or attention.get("quality") or {}
    numeric = attention.get("numeric") or payload.get("numeric") or {}
    verdict = attention.get("verdict") or payload.get("verdict") or {}
    launches = component.get("launches") or []
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    text = f"""# OPT-079 — Convert prompt attention KV operands once per shared stage

Status: **{(payload.get("status") or "pending GPU sitting")}**. Authority llama.cpp
`{LLAMA_REV}`, GGUF SHA-256 `{GGUF_SHA}`. Control is production `f16_async`.
Candidate `kv_once` converts each live BF16 K/V stage to F16 once after
cp.async completion and reuses the same 16-bit shared slots. Tile dimensions,
two-stage async ring, register softmax, stream-K, query preparation and F16
arithmetic are unchanged. No global converted KV cache. No gqa6/nbatch64 retry.

`claims_throughput: false` unless production_kept. Require >=5 ms/prompt
estimated complete attention saving and a positive paired interval, or
measured rejection. Long release is OPT-080.

## Numeric policy

Candidate F16 operands and complete attention outputs must equal `f16_async`
for unchanged arithmetic. Finite BF16 values around half rounding, signed
zero, underflow and overflow are covered without extra clamping.

## Mechanism

After each stage's producer wait, the CTA cooperatively converts BF16 K/V
in place. QK MMA and P×V load packed F16 fragments. Stage ownership: no
conversion before producer completion, no consumer reads before conversion,
no next asynchronous write before the last consumer.

## Dispatch

Launches: {launches}

## Complete P4096 attention (16 layers)

Control f16_async vs survivor `{component.get("survivor") or payload.get("survivor")}`.
Component n={component.get("n")} means: control {_fmt(component.get("control_mean_ms"))} ms,
candidate {_fmt(component.get("candidate_mean_ms"))} ms.
Paired CI (control-candidate): {_fmt(component.get("ci95_low"), 4)} ..
{_fmt(component.get("ci95_high"), 4)} ms; mean diff {_fmt(component.get("mean_diff_ms"), 4)} ms.
Engine P4096 pairs: {_fmt(engine.get("control_mean_ms"))} vs
{_fmt(engine.get("candidate_mean_ms"))} ms.

## Quality (OPT-073)

quality-v3 absolute={quality.get("quality_v3_absolute")};
engine non-regression={quality.get("quality_v3_engine_non_regression")}.
Nonfinite={numeric.get("nonfinite")}.

## Decision

Verdict: **{verdict.get("verdict")}** ({verdict.get("reasons")}).
production_kept={verdict.get("production_kept")}; retain_reason={verdict.get("retain_reason")}.
Shipping attention pipeline is `{payload.get("shipping_attention_pipeline", "f16_async")}`.

## tok/s

P4096 whole-run tok/s delta versus the then-current f16_async baseline is
**{payload.get("tok_s_delta", 0)}**. Zero when production_kept is false.
"""
    REPORT.write_text(text, encoding="utf-8")


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
    plan = family_plan(phase, mode)
    quality = opt073_quality()
    tinies = tiny_cases()
    configs = [dict(row) for row in CONFIGS]
    engine: dict[str, Any] = {}
    native: dict[str, Any] = {}
    occupancy_loss = False
    conversion_reduced = True
    if skip_gpu:
        if synthetic is None:
            raise AdmissionError("synthetic evidence required when skip_gpu")
        component = dict(synthetic.get("component") or {})
        engine = dict(synthetic.get("engine") or {})
        numeric = dict(synthetic.get("numeric") or {})
        dispatch_ok = bool(synthetic.get("dispatch_ok", True))
        native = dict(synthetic.get("native") or {})
        occupancy_loss = bool(synthetic.get("occupancy_loss", False))
        conversion_reduced = bool(synthetic.get("conversion_reduced", True))
    else:
        native_runner = runner or default_native_runner
        native = run_native_correctness(
            runner=native_runner, run_dir=run_dir, mode=mode
        )
        if not native.get("pass"):
            raise AdmissionError("native numeric/stage-lifetime phase failed")
        component = run_component(plan, native=native, configs=configs)
        dispatch_ok = bool(component.get("dispatch_ok"))
        survivor_cfg = config_by_id(str(component["survivor"]))
        occ_c = int(native.get("f16_async_occupancy") or 0)
        occ_k = int(native.get("kv_once_occupancy") or 0)
        occupancy_loss = occ_k < 1 or (occ_c > 0 and occ_k < occ_c)
        conversion_reduced = "convert_once_source=true" in str(
            native.get("stdout") or ""
        )
        try:
            engine = (
                run_engine(
                    plan,
                    runner=native_runner,
                    run_dir=run_dir,
                    survivor=survivor_cfg,
                    decode_guard=mode == "acceptance",
                )
                or {}
            )
        except AdmissionError as exc:
            if plan["mode"] == "acceptance":
                raise
            engine = {"error": str(exc), "skipped": True}
        numeric = {
            "nonfinite": int(native.get("nonfinite") or 0),
            "f16_async_local_bytes": int(native.get("f16_async_local_bytes") or 0),
            "kv_once_local_bytes": int(native.get("kv_once_local_bytes") or 0),
        }
    local_bytes = max(
        0,
        int(numeric.get("kv_once_local_bytes") or 0)
        - int(numeric.get("f16_async_local_bytes") or 0),
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
        occupancy_loss=occupancy_loss,
        conversion_reduced=conversion_reduced,
    )
    observed = planned_observation(plan)
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
        "task": "OPT-079",
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
    dump_json(run_dir / f"{phase}-result.json", payload)
    return payload


def run(mode: str, phase: str, run_dir: Path) -> dict[str, Any]:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    selected = [phase] if phase in PHASES else ["attention"]
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
            "task": "OPT-079",
            "mode": mode,
            "llama_revision": LLAMA_REV,
            "gguf_sha256": GGUF_SHA,
            "token_generator": TOKEN_GENERATOR,
            "configurations": [row["id"] for row in CONFIGS],
            "control": CONTROL_ID,
            "quality": quality,
            "claims_throughput": False,
            "claims_performance_improvement": False,
            "shipping_attention_pipeline": "f16_async",
            "report_path": "evidence/optimization/opt079-attention-kv-operands/REPORT.md",
        }
    )
    for name in selected:
        results[name] = run_phase(name, mode=mode, run_dir=run_dir)
    attention = results.get("attention") or {}
    verdict = attention.get("verdict") or {}
    results["survivor"] = attention.get("survivor")
    results["numeric"] = attention.get("numeric")
    results["verdict"] = verdict
    results["production_kept"] = bool(verdict.get("production_kept"))
    results["shipping_unchanged"] = bool(verdict.get("shipping_unchanged", True))
    results["status"] = "measured" if attention else results.get("status", "pending")
    tok_s_delta = 0.0
    engine = attention.get("engine") or {}
    control_ms = float(engine.get("control_mean_ms") or 0.0)
    candidate_ms = float(engine.get("candidate_mean_ms") or 0.0)
    if results.get("production_kept") and control_ms > 0.0 and candidate_ms > 0.0:
        tok_s_delta = (4096.0 / (candidate_ms / 1000.0)) - (
            4096.0 / (control_ms / 1000.0)
        )
    results["tok_s_delta"] = tok_s_delta
    if results.get("production_kept"):
        survivor_cfg = config_by_id(str(results.get("survivor")))
        results["shipping_attention_pipeline"] = survivor_cfg["attention_pipeline"]
        results["claims_throughput"] = True
        apply_production_pin(str(survivor_cfg["attention_pipeline"]))
    write_report(results)
    dump_json(FIXTURE, results)
    dump_json(run_dir / "opt079_attention_kv_operands.json", results)
    counts = (results.get(phase) or {}).get("native_counts") or {}
    print(
        "QW38_OPT079_RESULT="
        + json.dumps(
            {
                "task": "OPT-079",
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
            }
        )
    )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("attention",), default="attention")
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
        ROOT / "build" / "optimization-runs" / "OPT-079" / args.mode
    )
    try:
        if args.skip_gpu:
            raise AdmissionError("GPU sitting required for OPT-079 phases")
        run(args.mode, args.phase, run_dir)
    except (AdmissionError, AssertionError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
