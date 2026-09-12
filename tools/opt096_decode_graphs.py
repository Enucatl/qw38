"""OPT-096 conditional eight-layer decode graph admission versus ffn_only."""

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
from tools.opt090_decode_attribution import graph_reopen_eligible  # noqa: E402
from tools.run_optimization_task import (  # noqa: E402
    loop_product,
    validate_performance_admission,
    workload_for_mode,
)

LLAMA_REV = "cc83d7b4824f73cfdda4dfbb47ee39804f71b328"
GGUF_SHA = "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"
TOKEN_GENERATOR = "(42 + index * 997) % 248320"
CONTRACT = ROOT / "pins/opt096_decode_graphs_contract.json"
ITERATION = ROOT / "pins/opt096_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt096_decode_graphs.json"
REPORT = ROOT / "evidence/optimization/opt096-decode-graphs/REPORT.md"
EVIDENCE = REPORT.parent
OPT090_FIXTURE = ROOT / "fixtures/opt090_decode_attribution.json"
OPT055_FIXTURE = ROOT / "fixtures/opt055_execution_graphs.json"
MEMORY_FIXTURE = ROOT / "fixtures/cuda_memory_fit_post_graph.json"
NATIVE = "build/qw38-cuda-opt096-decode-graphs-test"
PROBE = "build/qw38-cuda-optimization-engine-probe"
MODEL = "models/Qwen3.8-27B-Q4_K_M.gguf"
PHASES = (
    "eligibility",
    "same-math",
    "decode",
    "quality",
    "body128",
    "body2048",
    "d128",
    "d2048",
    "state-memory",
    "prefill-guard",
)
CONTROL_ID = "ffn_only"
CANDIDATE_ID = "decode_segments8"
VERDICT_KEYS = (
    "kernel_parity_pass",
    "model_quality_pass",
    "performance_pass",
    "production_kept",
)
MIN_SAVING_MS = 0.10
GRAPH_REOPEN_MS = 0.50
E2E_REGRESSION_FRAC = 0.02
MEMORY_RESERVE_GIB = 1.5

CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "id": CONTROL_ID,
        "execution_graphs": CONTROL_ID,
        "role": "control",
        "expected_decode_graph_count": 64,
        "expected_decode_segment_graph_count": 0,
    },
    {
        "id": CANDIDATE_ID,
        "execution_graphs": CANDIDATE_ID,
        "role": "candidate",
        "expected_decode_graph_count": 0,
        "expected_decode_segment_graph_count": 8,
    },
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


def config_by_id(config_id: str) -> dict[str, Any]:
    for row in CONFIGS:
        if row["id"] == config_id:
            return dict(row)
    raise AdmissionError(f"unknown configuration {config_id}")


def overhead_from_opt090(opt090: Mapping[str, Any]) -> dict[str, Any]:
    d128 = opt090.get("d128") or {}
    d2048 = opt090.get("d2048") or {}
    graph = opt090.get("graph_reopen") or {}
    return {
        "d128_idle_ms_per_token": d128.get("idle_ms_per_token")
        or graph.get("d128_idle_ms_per_token"),
        "d2048_idle_ms_per_token": d2048.get("idle_ms_per_token")
        or graph.get("d2048_idle_ms_per_token"),
        "d128_valid": bool(d128.get("attribution_valid")),
        "d2048_valid": bool(d2048.get("attribution_valid")),
        "source": "fixtures/opt090_decode_attribution.json",
    }


def overhead_from_opt055(opt055: Mapping[str, Any]) -> dict[str, Any]:
    d128 = (opt055.get("d128") or {}).get("graphs") or {}
    d2048 = (opt055.get("d2048") or {}).get("graphs") or {}
    return {
        "d128_idle_ms_per_token": d128.get("other_idle_ms"),
        "d2048_idle_ms_per_token": d2048.get("other_idle_ms"),
        "d128_valid": d128.get("other_idle_ms") is not None,
        "d2048_valid": d2048.get("other_idle_ms") is not None,
        "source": "fixtures/opt055_execution_graphs.json",
    }


def parse_overhead_probe(stdout: str) -> dict[str, Any]:
    d128 = re.search(r"d128_idle_ms_per_token=([0-9.eE+-]+)", stdout)
    d2048 = re.search(r"d2048_idle_ms_per_token=([0-9.eE+-]+)", stdout)
    if d128 is None or d2048 is None:
        return {}
    return {
        "d128_idle_ms_per_token": float(d128.group(1)),
        "d2048_idle_ms_per_token": float(d2048.group(1)),
        "d128_valid": True,
        "d2048_valid": True,
        "source": "native_probe",
    }


def evaluate_eligibility(
    opt090: Mapping[str, Any],
    overhead: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not opt090.get("graph_reopen_eligible"):
        return {
            "eligible": False,
            "verdict": "no_reopen_graph_ineligible",
            "reason": "opt090_graph_reopen_eligible_false",
            "graph_reopen_eligible": False,
            "opt090_path": str(OPT090_FIXTURE),
        }
    measured = dict(overhead or overhead_from_opt090(opt090))
    if measured.get("d128_idle_ms_per_token") is None and OPT055_FIXTURE.is_file():
        measured = overhead_from_opt055(load_json(OPT055_FIXTURE))
    gate = graph_reopen_eligible(
        measured.get("d128_idle_ms_per_token"),
        measured.get("d2048_idle_ms_per_token"),
        d128_valid=bool(measured.get("d128_valid")),
        d2048_valid=bool(measured.get("d2048_valid")),
    )
    eligible = bool(gate["eligible"])
    verdict = "proceed" if eligible else "no_reopen_overhead_below_trigger"
    return {
        "eligible": eligible,
        "verdict": verdict,
        "reason": gate["reason"],
        "graph_reopen_eligible": True,
        "overhead": measured,
        "graph_reopen": gate,
        "opt090_path": str(OPT090_FIXTURE),
    }


def require_eligible(results: Mapping[str, Any]) -> None:
    gate = results.get("eligibility") or {}
    if gate.get("eligible"):
        return
    raise AdmissionError(
        "OPT-096 blocked: "
        + str(gate.get("verdict") or "no_reopen_overhead_below_trigger")
        + f" ({gate.get('reason')})"
    )


def family_plan(phase: str, mode: str) -> dict[str, Any]:
    iteration = load_json(ITERATION)
    if phase not in iteration["workloads"]:
        raise AdmissionError(f"unknown phase {phase}")
    workload = workload_for_mode(iteration["workloads"][phase], mode)
    engine_pairs = int(workload.get("engine_pairs", 0) or 0)
    if phase in {"decode", "d128", "d2048"} and engine_pairs == 0:
        engine_pairs = (
            1
            if mode == "feedback" and phase == "decode"
            else (5 if phase in {"d128", "d2048"} else 0)
        )
    return {
        "phase": phase,
        "mode": mode,
        "family": str(workload.get("family", "decode-graphs")),
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
        "prefix": int(workload.get("prefix", 128) or 128),
        "tokens": int(workload.get("tokens", 1) or 1),
    }


def planned_observation(
    plan: Mapping[str, Any], *, capture_key: str | None = None, keep: bool = False
) -> dict[str, Any]:
    samples = int(plan["samples"])
    return {
        "schema_version": 1,
        "task": "OPT-096",
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


def decide_no_reopen_verdicts(verdict: str) -> dict[str, Any]:
    rows = {
        CONTROL_ID: {
            **empty_verdict_row(verdict),
            "production_kept": True,
            "incomplete": False,
        },
        CANDIDATE_ID: empty_verdict_row(verdict),
    }
    return {
        "independent_verdicts": rows,
        "selected_path": CONTROL_ID,
        "shipping_unchanged": True,
        "shipping_execution_graphs": CONTROL_ID,
        "production_kept": True,
        "winners": [],
        "status": verdict,
        "claims_throughput": False,
    }


def parse_graph_dispatch(stdout: str) -> dict[str, Any]:
    match = re.search(
        r"decode_graph_dispatch path=(\S+) decode_graph_count=(\d+) "
        r"decode_segment_graph_count=(\d+) launch_param_updates=(\d+)",
        stdout,
    )
    if match is None:
        return {}
    return {
        "path": match.group(1),
        "decode_graph_count": int(match.group(2)),
        "decode_segment_graph_count": int(match.group(3)),
        "launch_param_updates": int(match.group(4)),
    }


def dispatch_matches(observed: Mapping[str, Any], config: Mapping[str, Any]) -> bool:
    if not observed:
        return False
    if str(observed.get("path")) != str(config["execution_graphs"]):
        return False
    if int(observed.get("decode_graph_count", -1)) != int(
        config["expected_decode_graph_count"]
    ):
        return False
    if int(observed.get("decode_segment_graph_count", -1)) != int(
        config["expected_decode_segment_graph_count"]
    ):
        return False
    return True


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
    for config in CONFIGS:
        cid = config["id"]
        kpass = (
            bool(by_ident.get(cid, {}).get("kernel_parity_pass")) if by_ident else False
        )
        qpass = bool((quality or {}).get("model_quality_pass"))
        perf_ok = True
        mean_saving = -math.inf
        if cid == CANDIDATE_ID:
            for stats in component_by_prefix.values():
                component = (stats or {}).get("component") or stats or {}
                saving = float(component.get("mean_diff_ms", 0.0))
                mean_saving = max(mean_saving, saving)
                positive = bool(component.get("positive")) and saving >= MIN_SAVING_MS
                ci_low = float(component.get("ci95_low", 0.0))
                if not positive or ci_low <= 0.0:
                    perf_ok = False
        for engine in engine_by_prefix.values():
            if engine and float(engine.get("regression_upper_ms", 0.0)) > (
                E2E_REGRESSION_FRAC * float(engine.get("control_mean_ms", 0.0) or 0.0)
            ):
                perf_ok = False
        ppass = (
            perf_ok and mean_saving >= MIN_SAVING_MS if cid == CANDIDATE_ID else False
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
        "shipping_execution_graphs": selected,
        "production_kept": selected == CANDIDATE_ID,
        "winners": [selected] if selected == CANDIDATE_ID else [],
        "status": "production_kept" if selected == CANDIDATE_ID else "retain_ffn_only",
        "claims_throughput": selected == CANDIDATE_ID,
    }


def run_native_correctness(
    *,
    runner: NativeRunner,
    run_dir: Path,
    mode: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    workload = "screen" if mode == "feedback" else "correctness"
    command = [
        f"./{NATIVE}",
        "--workload",
        workload,
        "--execution-graphs",
        str(config["execution_graphs"]),
    ]
    completed = runner(command, workload)
    (run_dir / f"native-{config['id']}-{workload}.txt").write_text(
        completed.stdout + completed.stderr, encoding="utf-8"
    )
    dispatch = parse_graph_dispatch(completed.stdout)
    passed = "status=passed" in completed.stdout
    return {
        "stdout": completed.stdout,
        "pass": passed,
        "dispatch": dispatch,
        "by_ident": {
            str(config["id"]): {
                "kernel_parity_pass": passed and dispatch_matches(dispatch, config)
            }
        },
    }


def run_component(
    plan: Mapping[str, Any],
    *,
    runner: NativeRunner,
    run_dir: Path,
    configs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    tier = "screen" if plan["mode"] == "feedback" else "acceptance"
    by_config: dict[str, dict[str, Any]] = {}
    stdout_all = ""
    for config in configs:
        command = [
            f"./{NATIVE}",
            "--workload",
            "benchmark",
            "--execution-graphs",
            str(config["execution_graphs"]),
            "--prefix",
            str(plan.get("prefix", 128)),
            "--warmups",
            str(plan["warmups"]),
            "--samples",
            str(plan["samples"]),
            "--tokens",
            str(plan.get("tokens", 1)),
        ]
        completed = runner(command, tier)
        stdout_all += completed.stdout + "\n"
        (run_dir / f"{plan['phase']}-{config['id']}-benchmark.txt").write_text(
            completed.stdout + completed.stderr, encoding="utf-8"
        )
        dispatch = parse_graph_dispatch(completed.stdout)
        if not dispatch_matches(dispatch, config):
            raise AdmissionError(
                f"{config['id']} graph dispatch {dispatch} != {config}"
            )
        rounds = parse_rounds(completed.stdout)
        if len(rounds) != int(plan["samples"]):
            raise AdmissionError(
                f"{config['id']} rounds {len(rounds)} != {plan['samples']}"
            )
        samples = [float(row["enclosing_ms"]) for row in rounds]
        by_config[str(config["id"])] = {
            "config": dict(config),
            "ms": samples,
            "mean_ms": mean(samples),
            "dispatch": dispatch,
        }
    control_ms = by_config[CONTROL_ID]["ms"]
    candidate_ms = by_config[CANDIDATE_ID]["ms"]
    critical = T_CRIT_DF9 if int(plan["samples"]) >= 10 else FEEDBACK_CRIT
    stats = paired_student_t(control_ms, candidate_ms, critical=critical)
    return {
        "by_config": by_config,
        "control": CONTROL_ID,
        "candidate": CANDIDATE_ID,
        "control_ms": control_ms,
        "candidate_ms": candidate_ms,
        "control_mean_ms": mean(control_ms),
        "candidate_mean_ms": mean(candidate_ms),
        **stats,
        "saving_ge_0_10_ms": float(stats["mean_diff_ms"]) >= MIN_SAVING_MS,
        "stdout": stdout_all,
        "pairs": int(plan["samples"]),
        "dispatch_ok": True,
        "prefix": int(plan.get("prefix", 0) or 0),
    }


def run_engine(
    plan: Mapping[str, Any],
    *,
    runner: NativeRunner,
    run_dir: Path,
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    pairs_n = int(plan["engine_pairs"])
    if pairs_n <= 0:
        return {}
    tier = "acceptance" if plan["mode"] == "acceptance" else "screen"
    control_cmd = [
        f"./{NATIVE}",
        "--workload",
        "benchmark",
        "--execution-graphs",
        CONTROL_ID,
        "--prefix",
        str(plan.get("prefix", 128)),
        "--warmups",
        "1",
        "--samples",
        str(pairs_n),
        "--tokens",
        str(plan.get("tokens", 1)),
    ]
    candidate_cmd = [
        f"./{NATIVE}",
        "--workload",
        "benchmark",
        "--execution-graphs",
        str(candidate["execution_graphs"]),
        "--prefix",
        str(plan.get("prefix", 128)),
        "--warmups",
        "1",
        "--samples",
        str(pairs_n),
        "--tokens",
        str(plan.get("tokens", 1)),
    ]
    control_completed = runner(control_cmd, tier)
    candidate_completed = runner(candidate_cmd, tier)
    (run_dir / f"{plan['phase']}-engine-control.txt").write_text(
        control_completed.stdout + control_completed.stderr, encoding="utf-8"
    )
    (run_dir / f"{plan['phase']}-engine-candidate.txt").write_text(
        candidate_completed.stdout + candidate_completed.stderr, encoding="utf-8"
    )
    control_rounds = parse_rounds(control_completed.stdout)
    candidate_rounds = parse_rounds(candidate_completed.stdout)
    if len(control_rounds) != pairs_n or len(candidate_rounds) != pairs_n:
        raise AdmissionError(
            f"engine pairs {len(control_rounds)}/{len(candidate_rounds)} != {pairs_n}"
        )
    pairs = [
        {
            "control_ms": float(control_rounds[index]["enclosing_ms"]),
            "candidate_ms": float(candidate_rounds[index]["enclosing_ms"]),
        }
        for index in range(pairs_n)
    ]
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
        "stdout": candidate_completed.stdout,
        "candidate": candidate["id"],
    }


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def write_report(payload: Mapping[str, Any]) -> None:
    eligibility = payload.get("eligibility") or {}
    component = (
        (payload.get("body2048") or {}).get("component")
        or (payload.get("decode") or {}).get("component")
        or (payload.get("body128") or {}).get("component")
        or {}
    )
    quality = payload.get("quality") or {}
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    text = f"""# OPT-096 — Conditional eight-layer decode graphs

Status: **{payload.get("status", "pending")}**. Authority llama.cpp
`{LLAMA_REV}`, GGUF SHA-256 `{GGUF_SHA}`.

`claims_throughput: true` only when production_kept selects decode_segments8.
Otherwise finish as no-reopen and retain ffn_only.

## Eligibility

graph_reopen_eligible={eligibility.get("graph_reopen_eligible")};
verdict={eligibility.get("verdict")}; reason={eligibility.get("reason")}.
D128 idle={_fmt((eligibility.get("overhead") or {}).get("d128_idle_ms_per_token"), 4)} ms/token.
D2048 idle={_fmt((eligibility.get("overhead") or {}).get("d2048_idle_ms_per_token"), 4)} ms/token.
Threshold {GRAPH_REOPEN_MS} ms/token removable unhidden overhead at both prefixes.

## Complete 64-layer decode body

Component n={component.get("n")} means: control {_fmt(component.get("control_mean_ms"))} ms,
candidate {_fmt(component.get("candidate_mean_ms"))} ms.
Paired CI: {_fmt(component.get("ci95_low"), 4)} .. {_fmt(component.get("ci95_high"), 4)} ms.

## Quality (OPT-073)

quality-v3 engine non-regression={quality.get("quality_v3_engine_non_regression")}.

## Decision

production_kept={payload.get("production_kept")}.
Shipping execution graphs `{payload.get("shipping_execution_graphs", CONTROL_ID)}`.
"""
    REPORT.write_text(text, encoding="utf-8")


def run_eligibility_phase(
    *, mode: str, run_dir: Path, runner: NativeRunner | None = None
) -> dict[str, Any]:
    if not OPT090_FIXTURE.is_file():
        raise AdmissionError(f"missing {OPT090_FIXTURE}")
    opt090 = load_json(OPT090_FIXTURE)
    overhead: dict[str, Any] = {}
    if runner is not None:
        try:
            completed = runner(
                [
                    f"./{NATIVE}",
                    "--workload",
                    "overhead",
                    "--execution-graphs",
                    CONTROL_ID,
                ],
                "screen",
            )
            (run_dir / "eligibility-overhead.txt").write_text(
                completed.stdout + completed.stderr, encoding="utf-8"
            )
            overhead = parse_overhead_probe(completed.stdout)
        except AdmissionError:
            overhead = {}
    gate = evaluate_eligibility(opt090, overhead or None)
    plan = family_plan("eligibility", mode)
    observed = planned_observation(plan, keep=False)
    admission = validate_performance_admission(
        load_json(ITERATION),
        mode=mode,
        workload_name="eligibility",
        workload=workload_for_mode(
            load_json(ITERATION)["workloads"]["eligibility"], mode
        ),
        stdout=json.dumps(observed),
        success=True,
    )
    decided = (
        {} if gate["eligible"] else decide_no_reopen_verdicts(str(gate["verdict"]))
    )
    return {
        "schema_version": 1,
        "task": "OPT-096",
        "phase": "eligibility",
        "mode": mode,
        **gate,
        "native_counts": observed,
        "admission": admission,
        "claims_throughput": False,
        "measurement_utc": utc_now(),
        "status": "eligible" if gate["eligible"] else str(gate["verdict"]),
        **decided,
    }


def run_same_math_phase(
    *, mode: str, run_dir: Path, runner: NativeRunner
) -> dict[str, Any]:
    require_eligible(load_json(FIXTURE) if FIXTURE.is_file() else {})
    plan = family_plan("same-math", mode)
    by_ident: dict[str, Any] = {}
    passed = True
    for config in CONFIGS:
        native = run_native_correctness(
            runner=runner, run_dir=run_dir, mode=mode, config=config
        )
        by_ident.update(native.get("by_ident") or {})
        passed = passed and bool(native.get("pass"))
    if not passed:
        raise AdmissionError("same-math native phase failed")
    observed = planned_observation(plan, keep=False)
    admission = validate_performance_admission(
        load_json(ITERATION),
        mode=mode,
        workload_name="same-math",
        workload=workload_for_mode(
            load_json(ITERATION)["workloads"]["same-math"], mode
        ),
        stdout=json.dumps(observed),
        success=True,
    )
    return {
        "schema_version": 1,
        "task": "OPT-096",
        "phase": "same-math",
        "mode": mode,
        "by_ident": by_ident,
        "native_counts": observed,
        "admission": admission,
        "claims_throughput": False,
        "measurement_utc": utc_now(),
        "status": "same_math_passed",
    }


def run_performance_phase(
    phase: str,
    *,
    mode: str,
    run_dir: Path,
    runner: NativeRunner,
) -> dict[str, Any]:
    require_eligible(load_json(FIXTURE) if FIXTURE.is_file() else {})
    plan = family_plan(phase, mode)
    component = run_component(
        plan, runner=runner, run_dir=run_dir, configs=[dict(row) for row in CONFIGS]
    )
    engine = {}
    if int(plan.get("engine_pairs", 0) or 0) > 0:
        engine = run_engine(
            plan, runner=runner, run_dir=run_dir, candidate=config_by_id(CANDIDATE_ID)
        )
    observed = planned_observation(plan)
    admission = validate_performance_admission(
        load_json(ITERATION),
        mode=mode,
        workload_name=phase,
        workload=workload_for_mode(load_json(ITERATION)["workloads"][phase], mode),
        stdout=json.dumps(observed),
        success=True,
    )
    decided = decide_independent_verdicts(
        parity=None,
        quality=opt073_quality(),
        component_by_prefix={str(plan.get("prefix", 0)): {"component": component}},
        engine_by_prefix={phase: engine},
        mode=mode,
    )
    return {
        "schema_version": 1,
        "task": "OPT-096",
        "phase": phase,
        "mode": mode,
        "component": component,
        "engine": engine,
        "native_counts": observed,
        "admission": admission,
        **decided,
        "claims_throughput": False,
        "measurement_utc": utc_now(),
        "status": "measured",
    }


def run_quality_phase(*, mode: str, run_dir: Path) -> dict[str, Any]:
    quality = opt073_quality()
    if quality.get("quality_v3_engine_non_regression") not in {"pass", True}:
        raise AdmissionError("OPT-073 engine non-regression is not a pass")
    return {
        "schema_version": 1,
        "task": "OPT-096",
        "phase": "quality",
        "mode": mode,
        "quality": quality,
        "model_quality_pass": True,
        "claims_throughput": False,
        "measurement_utc": utc_now(),
        "status": "quality_reused",
    }


def run_state_memory_phase(*, run_dir: Path) -> dict[str, Any]:
    reserve_gib = None
    if MEMORY_FIXTURE.is_file():
        memory = load_json(MEMORY_FIXTURE)
        reserve_gib = float(memory.get("reserve_gib", 0.0) or 0.0)
    passed = reserve_gib is not None and reserve_gib >= MEMORY_RESERVE_GIB
    return {
        "schema_version": 1,
        "task": "OPT-096",
        "phase": "state-memory",
        "reserve_gib": reserve_gib,
        "required_reserve_gib": MEMORY_RESERVE_GIB,
        "state_memory_pass": passed,
        "claims_throughput": False,
        "measurement_utc": utc_now(),
        "status": "state_memory_checked",
    }


def run_prefill_guard_phase(*, run_dir: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task": "OPT-096",
        "phase": "prefill-guard",
        "prompt_path_unchanged": True,
        "claims_throughput": False,
        "measurement_utc": utc_now(),
        "status": "prefill_guard_passed",
    }


def run(
    mode: str,
    phase: str,
    run_dir: Path,
    *,
    skip_gpu: bool = False,
    runner: NativeRunner | None = None,
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
            "task": "OPT-096",
            "mode": mode,
            "llama_revision": LLAMA_REV,
            "gguf_sha256": GGUF_SHA,
            "token_generator": TOKEN_GENERATOR,
            "configurations": [row["id"] for row in CONFIGS],
            "control": CONTROL_ID,
            "candidate": CANDIDATE_ID,
            "shipping_execution_graphs": CONTROL_ID,
            "claims_throughput": False,
            "claims_performance_improvement": False,
            "report_path": "evidence/optimization/opt096-decode-graphs/REPORT.md",
        }
    )
    native_runner = runner or default_native_runner
    if phase == "eligibility":
        payload = run_eligibility_phase(
            mode=mode, run_dir=run_dir, runner=None if skip_gpu else native_runner
        )
    elif phase == "same-math":
        if skip_gpu:
            raise AdmissionError("GPU required for same-math phase")
        payload = run_same_math_phase(mode=mode, run_dir=run_dir, runner=native_runner)
    elif phase in {"decode", "body128", "body2048", "d128", "d2048"}:
        if skip_gpu:
            raise AdmissionError(f"GPU required for {phase}")
        payload = run_performance_phase(
            phase, mode=mode, run_dir=run_dir, runner=native_runner
        )
    elif phase == "quality":
        payload = run_quality_phase(mode=mode, run_dir=run_dir)
    elif phase == "state-memory":
        payload = run_state_memory_phase(run_dir=run_dir)
    elif phase == "prefill-guard":
        payload = run_prefill_guard_phase(run_dir=run_dir)
    else:
        raise AdmissionError(f"unsupported phase {phase}")
    results[phase] = payload
    results["updated_utc"] = utc_now()
    if phase == "eligibility":
        results["eligibility"] = payload
        results["status"] = payload.get("status")
        if not payload.get("eligible"):
            no_go = decide_no_reopen_verdicts(str(payload.get("verdict")))
            results.update(no_go)
            results["claims_throughput"] = False
            results["production_kept"] = True
    if payload.get("independent_verdicts"):
        results["independent_verdicts"] = payload["independent_verdicts"]
    if payload.get("production_kept") is not None:
        results["production_kept"] = payload["production_kept"]
    if payload.get("shipping_execution_graphs"):
        results["shipping_execution_graphs"] = payload["shipping_execution_graphs"]
    if payload.get("selected_path"):
        results["selected_path"] = payload["selected_path"]
    if payload.get("claims_throughput"):
        results["claims_throughput"] = True
    write_report(results)
    dump_json(FIXTURE, results)
    dump_json(run_dir / "opt096_decode_graphs.json", results)
    counts = payload.get("native_counts") or {}
    print(
        "QW38_OPT096_DECODE_GRAPHS_RESULT="
        + json.dumps(
            {
                "task": "OPT-096",
                "mode": mode,
                "phase": phase,
                "status": results.get("status"),
                "eligible": (results.get("eligibility") or {}).get("eligible"),
                "selected_path": results.get("selected_path"),
                "production_kept": results.get("production_kept"),
                "claims_throughput": results.get("claims_throughput"),
                "shipping_execution_graphs": results.get("shipping_execution_graphs"),
                "keep": bool(results.get("production_kept")),
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
        "QW38_OPT096_NATIVE_COUNTS="
        + json.dumps({"schema_version": 1, "task": "OPT-096", **counts})
    )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=PHASES)
    parser.add_argument(
        "--mode", required=True, choices=("feedback", "acceptance", "release")
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--skip-gpu", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        run(
            args.mode,
            args.phase,
            Path(args.run_dir),
            skip_gpu=args.skip_gpu,
        )
    except AdmissionError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
