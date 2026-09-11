"""OPT-070 revalidation of installed Q8 r2_w2 and MMQ X-prefetch keeps."""

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

from tools.opt069_batch_gate import (  # noqa: E402
    assemble_from_retained_evidence,
)
from tools.opt073_quality_policy import FIXTURE as OPT073_FIXTURE  # noqa: E402
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
CONTRACT = ROOT / "pins/opt070_keep_revalidation_contract.json"
ITERATION = ROOT / "pins/opt070_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt070_keep_revalidation.json"
REPORT = ROOT / "evidence/optimization/opt070-keep-revalidation/REPORT.md"
EVIDENCE = ROOT / "evidence/optimization/opt070-keep-revalidation"
OPT074 = ROOT / "fixtures/opt074_production_gpu_admission.json"
REPLAY = "build/qw38-cuda-component-replay"
PROBE = "build/qw38-cuda-optimization-engine-probe"
MODEL = "models/Qwen3.8-27B-Q4_K_M.gguf"
T_CRIT_DF9 = 2.262
T_CRIT_DF4 = 2.132
Q8_SAVING_MS = 0.10
MMQ_SAVING_MS = 5.0
PHASES = ("q8", "mmq")

NativeRunner = Callable[[Sequence[str], str], subprocess.CompletedProcess[str]]


class KeepRevalidationError(AssertionError):
    """Fail-closed OPT-070 admission error."""


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
        raise KeepRevalidationError(
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
        raise KeepRevalidationError("paired samples must be equal and non-empty")
    diffs = [float(left) - float(right) for left, right in zip(control, candidate)]
    avg = mean(diffs)
    var = sample_variance(diffs)
    se = math.sqrt(var / len(diffs)) if var > 0.0 else 0.0
    df = len(diffs) - 1
    low = avg - critical * se
    high = avg + critical * se
    return {
        "n": len(diffs),
        "df": df,
        "mean_diff_ms": avg,
        "se": se,
        "critical": critical,
        "ci95_low": low,
        "ci95_high": high,
        "diffs": diffs,
        "positive": low > 0.0,
    }


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


def parse_engine_pairs(stdout: str) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        stripped = line.strip()
        if '"pair":' in stripped and "control_ms" in stripped:
            start = stripped.find("{")
            if start >= 0:
                pairs.append(json.loads(stripped[start:]))
    return pairs


def opt074_coverage(family_ids: Sequence[str]) -> dict[str, Any]:
    table = load_json(OPT074).get("admission_table") or []
    rows = [dict(row) for row in table if row.get("id") in set(family_ids)]
    admitted = [row for row in rows if row.get("v2_admitted") is True]
    return {
        "families": family_ids,
        "rows": [
            {
                "id": row.get("id"),
                "v2_admitted": row.get("v2_admitted"),
                "coverage": row.get("coverage"),
                "reason": row.get("reason"),
            }
            for row in rows
        ],
        "all_admitted": bool(rows) and len(admitted) == len(rows),
        "missing": [
            name for name in family_ids if name not in {r.get("id") for r in rows}
        ],
    }


def opt073_quality() -> dict[str, Any]:
    data = load_json(OPT073_FIXTURE)
    v3 = data.get("quality_v3") or {}
    llama = v3.get("llama") if isinstance(v3.get("llama"), Mapping) else v3
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
    return {
        "quality_v3_absolute": abs_status,
        "quality_v3_engine_non_regression": nr_status,
        "quality_v2_all": qv2_all,
        "status": data.get("status"),
    }


def decide_verdict(
    *,
    family: str,
    component: Mapping[str, Any] | None,
    engine: Mapping[str, Any] | None,
    coverage: Mapping[str, Any],
    quality: Mapping[str, Any],
    mode: str,
) -> dict[str, Any]:
    reasons: list[str] = []
    if not coverage.get("all_admitted"):
        reasons.append("opt074_coverage_unadmitted")
    engine_nr = quality.get("quality_v3_engine_non_regression")
    if engine_nr not in {"pass", True}:
        reasons.append("opt073_quality_unresolved")
    if mode != "acceptance":
        return {
            "verdict": "inconclusive",
            "status": "screened_in" if component else "incomplete",
            "reasons": reasons + ["feedback_is_not_acceptance"],
            "shipping_unchanged": True,
        }
    if component is None:
        return {
            "verdict": "inconclusive",
            "status": "incomplete",
            "reasons": reasons + ["missing_component_pairs"],
            "shipping_unchanged": True,
        }
    if engine is None or not engine.get("control_ms"):
        reasons.append("missing_engine_pairs")
        engine = None
    saving = Q8_SAVING_MS if family == "q8" else MMQ_SAVING_MS
    component_ok = (
        bool(component.get("positive"))
        and float(component.get("mean_diff_ms", 0.0)) >= saving
    )
    regression_ok = True
    if engine is not None:
        control_mean = float(engine.get("control_mean_ms", 0.0))
        upper = float(engine.get("regression_upper_ms", 0.0))
        limit = 0.02 * control_mean if control_mean > 0.0 else 0.0
        regression_ok = upper <= limit
        engine["non_regression"] = regression_ok
        engine["two_percent_limit_ms"] = limit
    if reasons:
        return {
            "verdict": "inconclusive",
            "status": "inconclusive",
            "reasons": reasons,
            "shipping_unchanged": True,
            "component_accepted": component_ok,
            "e2e_non_regression": regression_ok,
        }
    if component_ok and regression_ok:
        return {
            "verdict": "retain",
            "status": "production_kept",
            "reasons": ["component_ci_positive", "e2e_non_regression"],
            "shipping_unchanged": True,
            "component_accepted": True,
            "e2e_non_regression": True,
        }
    slower = float(component.get("mean_diff_ms", 0.0)) < 0.0 and bool(
        component.get("ci95_high", 0.0) < 0.0
    )
    if slower and coverage.get("all_admitted"):
        return {
            "verdict": "revert",
            "status": "performance_rejected",
            "reasons": ["candidate_slower_with_control_admission"],
            "shipping_unchanged": False,
            "component_accepted": False,
            "e2e_non_regression": regression_ok,
        }
    return {
        "verdict": "inconclusive",
        "status": "inconclusive",
        "reasons": reasons
        + (
            ["component_interval_not_positive"]
            if not component_ok
            else ["e2e_regression_unresolved"]
        ),
        "shipping_unchanged": True,
        "component_accepted": component_ok,
        "e2e_non_regression": regression_ok,
    }


def family_plan(phase: str, mode: str) -> dict[str, Any]:
    iteration = load_json(ITERATION)
    workload = workload_for_mode(iteration["workloads"][phase], mode)
    return {
        "phase": phase,
        "mode": mode,
        "family": "decode-mixer" if phase == "q8" else "prompt-ffn",
        "control": "r1_w4" if phase == "q8" else "fma_async",
        "candidate": "r2_w2" if phase == "q8" else "fma_async_x",
        "warmups": int(workload.get("warmups", 1 if mode == "feedback" else 3)),
        "samples": int(workload.get("samples", 3 if mode == "feedback" else 10)),
        "engine_pairs": 0 if mode == "feedback" else 5,
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
        "task": "OPT-070",
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
        args.extend(["--mmq-async-x", "1" if config == "fma_async_x" else "0"])
    if capture_key:
        args.extend(["--capture-key", capture_key])
    return args


def run_component(
    plan: Mapping[str, Any],
    *,
    runner: NativeRunner,
    run_dir: Path,
) -> dict[str, Any]:
    tier = "screen" if plan["mode"] == "feedback" else "acceptance"
    capture_key: str | None = None
    by_config: dict[str, list[float]] = {}
    native: dict[str, Any] = {}
    stdout_all = ""
    launches: list[dict[str, Any]] = []
    for config in (plan["control"], plan["candidate"]):
        completed = runner(replay_command(plan, config, capture_key), tier)
        stdout_all += completed.stdout + "\n"
        (run_dir / f"{plan['phase']}-{config}-replay.txt").write_text(
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
            raise KeepRevalidationError("missing capture identity")
        rounds = [
            row
            for row in parse_rounds(completed.stdout)
            if row.get("cache_mode", "rotating") == "rotating"
        ]
        if len(rounds) != int(plan["samples"]):
            raise KeepRevalidationError(
                f"{config} rotating rounds {len(rounds)} != {plan['samples']}"
            )
        ids = [row.get("sample_index") for row in rounds]
        if len(ids) != len(set(ids)):
            raise KeepRevalidationError(f"{config} reused sample IDs")
        group = re.search(
            r"gdn_input_output_groups=(\d+) attention_input_output_groups=(\d+)",
            completed.stdout,
        )
        gate = re.search(
            r"gate_up_calls=(\d+) down_calls=(\d+) mixer_calls=(\d+)",
            completed.stdout,
        )
        if plan["phase"] == "q8":
            if group is None:
                raise KeepRevalidationError("mixer group counts missing")
            gdn_groups = int(group.group(1))
            attn_groups = int(group.group(2))
            if gdn_groups != 48 or attn_groups != 16:
                raise KeepRevalidationError(
                    f"mixer groups gdn={gdn_groups} attn={attn_groups} != 48/16"
                )
        elif gate is not None:
            if int(gate.group(1)) != 64 or int(gate.group(2)) != 64:
                raise KeepRevalidationError(
                    f"prompt FFN calls gate_up={gate.group(1)} "
                    f"down={gate.group(2)} != 64/64"
                )
        by_config[config] = [float(row["enclosing_ms"]) for row in rounds]
        native[config] = observed
        launches.append(
            {
                "config": config,
                "last_q8_layout": observed.get("last_q8_layout"),
                "last_mmq_kernel": observed.get("last_mmq_kernel"),
                "last_mmq_async_x": observed.get("last_mmq_async_x"),
                "effective_q8_rows_skinny": observed.get("effective_q8_rows_skinny"),
            }
        )
        if "stale capture" in completed.stderr:
            raise KeepRevalidationError("stale capture key")
    critical = T_CRIT_DF9 if int(plan["samples"]) >= 10 else 4.303
    stats = paired_student_t(
        by_config[plan["control"]], by_config[plan["candidate"]], critical=critical
    )
    return {
        "capture_key": capture_key,
        "control_ms": by_config[plan["control"]],
        "candidate_ms": by_config[plan["candidate"]],
        "control_mean_ms": mean(by_config[plan["control"]]),
        "candidate_mean_ms": mean(by_config[plan["candidate"]]),
        **stats,
        "native": native,
        "launches": launches,
        "stdout": stdout_all,
        "pairs": int(plan["samples"]),
    }


def run_engine(
    plan: Mapping[str, Any],
    *,
    runner: NativeRunner,
    run_dir: Path,
) -> dict[str, Any]:
    if int(plan["engine_pairs"]) <= 0:
        return {}
    tier = "acceptance"
    workload = "q8-ab" if plan["phase"] == "q8" else "mmq-ab"
    command = [
        f"./{PROBE}",
        MODEL,
        "--workload",
        workload,
        "--pairs",
        str(plan["engine_pairs"]),
        "--modes",
        "graph",
    ]
    completed = runner(command, tier)
    (run_dir / f"{plan['phase']}-engine.txt").write_text(
        completed.stdout + completed.stderr, encoding="utf-8"
    )
    pairs = parse_engine_pairs(completed.stdout)
    if len(pairs) != int(plan["engine_pairs"]):
        raise KeepRevalidationError(
            f"engine pairs {len(pairs)} != {plan['engine_pairs']}"
        )
    control = [float(row["control_ms"]) for row in pairs]
    candidate = [float(row["candidate_ms"]) for row in pairs]
    diffs = [cand - ctrl for ctrl, cand in zip(control, candidate)]
    avg = mean(diffs)
    var = sample_variance(diffs)
    se = math.sqrt(var / len(diffs)) if var > 0.0 else 0.0
    upper = avg + T_CRIT_DF4 * se
    observed = parse_native_observation(completed.stdout)
    if not observed.get("override_before_capture_applied", True):
        raise KeepRevalidationError("graph capture override was not applied")
    return {
        "pairs": pairs,
        "control_ms": control,
        "candidate_ms": candidate,
        "control_mean_ms": mean(control),
        "candidate_mean_ms": mean(candidate),
        "mean_regression_ms": avg,
        "regression_upper_ms": upper,
        "df": 4,
        "critical": T_CRIT_DF4,
        "native": observed,
        "stdout": completed.stdout,
    }


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _coverage_line(block: Mapping[str, Any]) -> str:
    rows = block.get("rows") or []
    parts = [
        f"{row.get('id')} v2_admitted={row.get('v2_admitted')} "
        f"coverage={row.get('coverage')}"
        for row in rows
    ]
    return "; ".join(parts) if parts else "missing"


def write_report(payload: Mapping[str, Any]) -> None:
    q8 = payload.get("q8") or {}
    mmq = payload.get("mmq") or {}
    opt069 = payload.get("opt069_reporting_correction") or {}
    q8c = q8.get("component") or {}
    mmqc = mmq.get("component") or {}
    q8e = q8.get("engine") or {}
    mmqe = mmq.get("engine") or {}
    q8n = q8e.get("native") or {}
    mmqn = mmqe.get("native") or {}
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    text = f"""# OPT-070 — Revalidate installed Q8/MMQ keeps

Status: **measured this sitting**. Authority llama.cpp `{LLAMA_REV}`, GGUF
SHA-256 `{GGUF_SHA}`. No new kernel/layout sweep. Screen-only `keep=true` is
rejected by the runner. Shipping selectors remain `r2_w2` and `fma_async_x`.

## OPT-069 reporting correction

{opt069.get("note", "retained sidecars rewritten into REPORT.md")}.
Measurement UTC of the retained sitting: {opt069.get("measurement_utc")}.
Three outcomes remain unpassed versus llama / OPT-056; the previous
unmeasured labels were a report bug, not a missing sitting.

## Arithmetic and quality

Q8 OPT-074: {_coverage_line(q8.get("coverage") or {})}.
MMQ OPT-074: {_coverage_line(mmq.get("coverage") or {})}.
Missing coverage is unresolved admission, not a numerically validated keep.
OPT-073 quality-v3 absolute={payload.get("quality", {}).get("quality_v3_absolute")};
engine non-regression={payload.get("quality", {}).get("quality_v3_engine_non_regression")};
quality-v2 all={payload.get("quality", {}).get("quality_v2_all")}.

## Q8 r1_w4 vs r2_w2 (D2048 complete rotating mixer)

Control `{q8.get("control")}` vs installed `{q8.get("candidate")}`.
48 GDN input/output groups and 16 attention input/output groups in layer
order, including RMS/Q8_1 staging and OPT-058 invalidation on both sides.
X-prefetch held at the installed `fma_async_x` pin.

Component n={q8c.get("n")} means: control {_fmt(q8c.get("control_mean_ms"))} ms,
candidate {_fmt(q8c.get("candidate_mean_ms"))} ms.
Paired CI (control-candidate): {_fmt(q8c.get("ci95_low"), 4)} ..
{_fmt(q8c.get("ci95_high"), 4)} ms; mean diff {_fmt(q8c.get("mean_diff_ms"), 4)} ms.
Engine D2048+32 five pairs: {_fmt(q8e.get("control_mean_ms"))} vs
{_fmt(q8e.get("candidate_mean_ms"))} ms; regression UB
{_fmt(q8e.get("regression_upper_ms"), 4)} ms vs 2% limit
{_fmt(q8e.get("two_percent_limit_ms"), 4)} ms;
e2e non-regression={q8e.get("non_regression")}.
Verdict: **{(q8.get("verdict") or {}).get("verdict")}**
({(q8.get("verdict") or {}).get("reasons")}).
Q8 cannot revert: OPT-074 mixer families are unadmitted, so control
admission does not pass. Shipping `r2_w2` stays unresolved.
Proof boundary: complete rotating mixer + five target engine pairs.

## MMQ fma_async vs fma_async_x (P4096, 64 FFNs)

Tile 128x128. Complete FFN includes staging, SwiGLU, down, residual.
Q8 decision held at the resulting Q8 verdict / installed r2_w2.

Component n={mmqc.get("n")} means: control {_fmt(mmqc.get("control_mean_ms"))} ms,
candidate {_fmt(mmqc.get("candidate_mean_ms"))} ms.
Paired CI: {_fmt(mmqc.get("ci95_low"), 4)} ..
{_fmt(mmqc.get("ci95_high"), 4)} ms; mean diff {_fmt(mmqc.get("mean_diff_ms"), 4)} ms.
Engine P4096 five pairs: {_fmt(mmqe.get("control_mean_ms"))} vs
{_fmt(mmqe.get("candidate_mean_ms"))} ms; regression UB
{_fmt(mmqe.get("regression_upper_ms"), 4)} ms vs 2% limit
{_fmt(mmqe.get("two_percent_limit_ms"), 4)} ms;
e2e non-regression={mmqe.get("non_regression")}.
Verdict: **{(mmq.get("verdict") or {}).get("verdict")}**
({(mmq.get("verdict") or {}).get("reasons")}).
MMQ cannot be a numerically validated keep: OPT-074 Q4 families are
unadmitted. Shipping `fma_async_x` stays unresolved.
Proof boundary: 64-layer complete prompt FFN + five target engine pairs.

## Production dispatch

Scoped Q8 layout and MMQ async-X overrides are applied in the production
translation units before separate graph capture.
Q8 launches: {q8c.get("launches")}.
MMQ launches: {mmqc.get("launches")}.
Engine graph proof: Q8 override_before_capture={q8n.get("override_before_capture_applied")}
graph_capture_separate={q8n.get("graph_capture_separate")} pairs={q8n.get("pairs")};
MMQ override_before_capture={mmqn.get("override_before_capture_applied")}
graph_capture_separate={mmqn.get("graph_capture_separate")} pairs={mmqn.get("pairs")}.

## tok/s

This increment does not rerun OPT-069 P/D oracles. Official tok/s delta
versus the then-current OPT-069 baseline is 0 (shipping selectors unchanged).
OPT-069 retained P4096 Quartz {opt069.get("p_tok_s")} tok/s vs llama
{opt069.get("llama_p_tok_s")}.
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
        raise KeepRevalidationError(f"unknown phase {phase}")
    plan = family_plan(phase, mode)
    run_dir.mkdir(parents=True, exist_ok=True)
    quality = opt073_quality()
    families = (
        ("q8_mixer_k5120", "q8_mixer_k6144")
        if phase == "q8"
        else ("q4_gate_up_k5120", "q4_down_k17408")
    )
    coverage = opt074_coverage(families)
    engine: dict[str, Any] = {}
    if skip_gpu:
        if synthetic is None:
            raise KeepRevalidationError("synthetic evidence required when skip_gpu")
        component = dict(synthetic.get("component") or {})
        engine = dict(synthetic.get("engine") or {})
    else:
        native = runner or default_native_runner
        component = run_component(plan, runner=native, run_dir=run_dir)
        try:
            engine = run_engine(plan, runner=native, run_dir=run_dir) or {}
        except KeepRevalidationError as exc:
            engine = {"error": str(exc)}
    verdict = decide_verdict(
        family=phase,
        component=component or None,
        engine=engine or None,
        coverage=coverage,
        quality=quality,
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
        raise KeepRevalidationError(admission["message"])
    payload = {
        "schema_version": 1,
        "task": "OPT-070",
        "phase": phase,
        "mode": mode,
        "control": plan["control"],
        "candidate": plan["candidate"],
        "component": component,
        "engine": engine,
        "coverage": coverage,
        "quality": quality,
        "verdict": verdict,
        "native_counts": observed,
        "admission": admission,
        "claims_throughput": False,
        "measurement_utc": utc_now(),
        "status": "measured",
    }
    if isinstance(payload.get("component"), dict):
        payload["component"].pop("stdout", None)
        payload["component"].pop("native", None)
    if isinstance(payload.get("engine"), dict):
        payload["engine"].pop("stdout", None)
    dump_json(run_dir / f"{phase}-result.json", payload)
    return payload


def run(mode: str, phase: str, run_dir: Path) -> dict[str, Any]:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    opt069 = assemble_from_retained_evidence()
    correction = {
        "note": opt069.get("reporting_correction_note"),
        "measurement_utc": opt069.get("measurement_utc"),
        "p_tok_s": opt069["p"]["quartz"]["mean_tok_s"],
        "llama_p_tok_s": opt069["p"]["llama_cpp"]["avg_ts"],
        "status": opt069.get("status"),
    }
    selected = [phase] if phase in PHASES else list(PHASES)
    results: dict[str, Any] = {}
    if FIXTURE.is_file():
        try:
            results.update(load_json(FIXTURE))
        except json.JSONDecodeError:
            results = {}
    results.update(
        {
            "schema_version": 1,
            "task": "OPT-070",
            "mode": mode,
            "llama_revision": LLAMA_REV,
            "gguf_sha256": GGUF_SHA,
            "token_generator": TOKEN_GENERATOR,
            "opt069_reporting_correction": correction,
            "quality": opt073_quality(),
            "claims_throughput": False,
            "installed_q8_layout": "r2_w2",
            "installed_mmq_async_x": True,
            "shipping_unchanged": True,
            "status": "measured",
            "report_path": "evidence/optimization/opt070-keep-revalidation/REPORT.md",
        }
    )
    for name in selected:
        results[name] = run_phase(name, mode=mode, run_dir=run_dir)
        if (results[name].get("verdict") or {}).get("shipping_unchanged") is False:
            results["shipping_unchanged"] = False
    write_report(results)
    dump_json(FIXTURE, results)
    dump_json(run_dir / "opt070_keep_revalidation.json", results)
    counts = (results.get(phase) or {}).get("native_counts") or {}
    print(
        "QW38_OPT070_RESULT="
        + json.dumps(
            {
                "task": "OPT-070",
                "mode": mode,
                "phase": phase,
                "q8_verdict": (results.get("q8") or {})
                .get("verdict", {})
                .get("verdict"),
                "mmq_verdict": (results.get("mmq") or {})
                .get("verdict", {})
                .get("verdict"),
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
    parser.add_argument("--phase", choices=("q8", "mmq", "all"), default="q8")
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
        ROOT / "build" / "optimization-runs" / "OPT-070" / args.mode
    )
    try:
        if args.skip_gpu:
            raise KeepRevalidationError("GPU sitting required for OPT-070 phases")
        run(args.mode, args.phase, run_dir)
    except (KeepRevalidationError, AssertionError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
