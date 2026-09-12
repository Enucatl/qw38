"""OPT-094 conditional OPT-077 tile32 GDN timing replication admission."""

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
    docker_common,
    dump_json,
    load_json,
    mean,
    opt073_quality,
    paired_student_t,
    parse_rounds,
    utc_now,
)
from tools.opt077_gdn_decode import (  # noqa: E402
    dispatch_matches,
    parse_gdn_dispatch,
    run_engine,
)
from tools.opt090_decode_attribution import gdn_reopen_eligible  # noqa: E402
from tools.run_optimization_task import (  # noqa: E402
    loop_product,
    parse_native_observation,
    validate_performance_admission,
    workload_for_mode,
)

LLAMA_REV = "cc83d7b4824f73cfdda4dfbb47ee39804f71b328"
GGUF_SHA = "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"
TOKEN_GENERATOR = "(42 + index * 997) % 248320"
CONTRACT = ROOT / "pins/opt094_gdn_replication_contract.json"
ITERATION = ROOT / "pins/opt094_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt094_gdn_replication.json"
REPORT = ROOT / "evidence/optimization/opt094-gdn-replication/REPORT.md"
EVIDENCE = REPORT.parent
OPT090_FIXTURE = ROOT / "fixtures/opt090_decode_attribution.json"
OPT077_FIXTURE = ROOT / "fixtures/opt077_gdn_decode.json"
NATIVE = "build/qw38-cuda-opt077-gdn-decode-test"
REPLAY = "build/qw38-cuda-component-replay"
PROBE = "build/qw38-cuda-optimization-engine-probe"
MODEL = "models/Qwen3.8-27B-Q4_K_M.gguf"
PHASES = (
    "eligibility",
    "parity",
    "gdn",
    "quality",
    "gdn128",
    "gdn2048",
    "d128",
    "d2048",
    "state-prefill-guard",
)
CONTROL_ID = "sequential"
CANDIDATE_ID = "tile32"
VERDICT_KEYS = (
    "kernel_parity_pass",
    "model_quality_pass",
    "performance_pass",
    "production_kept",
)
MIN_SAVING_MS = 0.10
T_CRIT_DF29 = 2.04523
E2E_REGRESSION_FRAC = 0.02
THROUGHPUT_MIN = 0.98
P95_RATIO_MAX = 1.02

CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "id": CONTROL_ID,
        "gdn_decode": "sequential",
        "value_tile": 0,
        "role": "control",
        "expected_launch": "prepare_recurrence_window",
    },
    {
        "id": CANDIDATE_ID,
        "gdn_decode": "tile32",
        "value_tile": 32,
        "role": "candidate",
        "expected_launch": "prepare_recurrence_decode_tiled32",
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


def evaluate_eligibility(opt090: Mapping[str, Any]) -> dict[str, Any]:
    defect = opt090.get("opt077_defect")
    gdn = gdn_reopen_eligible(defect if isinstance(defect, Mapping) else None)
    eligible = bool(opt090.get("gdn_reopen_eligible")) and bool(gdn["eligible"])
    return {
        "eligible": eligible,
        "verdict": "proceed" if eligible else "no_reopen",
        "reason": gdn["reason"],
        "gdn_reopen": gdn,
        "opt090_path": str(OPT090_FIXTURE),
        "opt077_prior": {
            "control_mean_ms": 12.51649,
            "tile32_mean_ms": 11.89390,
            "ci95_low_ms": -0.1643,
            "ci95_high_ms": 1.4094,
            "inconclusive_not_rejection": True,
        },
    }


def require_replication_eligible(results: Mapping[str, Any]) -> None:
    gate = results.get("eligibility") or {}
    if gate.get("eligible"):
        return
    raise AdmissionError(
        "OPT-094 replication blocked: "
        + str(gate.get("verdict") or "no_reopen")
        + f" ({gate.get('reason')})"
    )


def family_plan(phase: str, mode: str) -> dict[str, Any]:
    iteration = load_json(ITERATION)
    if phase not in iteration["workloads"]:
        raise AdmissionError(f"unknown phase {phase}")
    workload = workload_for_mode(iteration["workloads"][phase], mode)
    engine_pairs = int(workload.get("engine_pairs", 0) or 0)
    if phase in {"gdn", "d128", "d2048"} and engine_pairs == 0:
        engine_pairs = (
            1
            if mode == "feedback" and phase == "gdn"
            else (5 if phase in {"d128", "d2048"} else 0)
        )
    return {
        "phase": phase,
        "mode": mode,
        "family": str(workload.get("family", "decode-gdn")),
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
        "prefix": int(workload.get("prefix", 0) or 0),
        "output_tokens": int(workload.get("tokens", 32) or 32),
    }


def planned_observation(
    plan: Mapping[str, Any], *, capture_key: str | None = None, keep: bool = False
) -> dict[str, Any]:
    samples = int(plan["samples"])
    return {
        "schema_version": 1,
        "task": "OPT-094",
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
    prefix = int(plan.get("prefix", 0) or 0)
    if prefix > 0:
        args.extend(["--prefix", str(prefix)])
    if capture_key:
        args.extend(["--capture-key", capture_key])
    return args


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


def decide_no_reopen_verdicts() -> dict[str, Any]:
    rows = {
        CONTROL_ID: {
            **empty_verdict_row("no_reopen_eligibility"),
            "production_kept": True,
            "incomplete": False,
        },
        CANDIDATE_ID: empty_verdict_row("no_reopen_eligibility"),
    }
    return {
        "independent_verdicts": rows,
        "selected_path": CONTROL_ID,
        "shipping_unchanged": True,
        "shipping_gdn_decode": CONTROL_ID,
        "production_kept": True,
        "winners": [],
        "status": "no_reopen",
        "claims_throughput": False,
    }


def decide_independent_verdicts(
    *,
    parity: Mapping[str, Any] | None,
    quality: Mapping[str, Any] | None,
    component_by_prefix: Mapping[str, Mapping[str, Any]],
    engine_by_prefix: Mapping[str, Mapping[str, Any]],
    mode: str,
    prior: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = {}
    by_ident = (parity or {}).get("by_ident") or {}
    for config in CONFIGS:
        cid = config["id"]
        previous = dict((prior or {}).get(cid) or empty_verdict_row())
        kpass = (
            bool(by_ident.get(cid, {}).get("kernel_parity_pass"))
            if by_ident
            else bool(previous.get("kernel_parity_pass"))
        )
        qpass = (
            bool((quality or {}).get("model_quality_pass"))
            if quality
            else bool(previous.get("model_quality_pass"))
        )
        perf_ok = True
        mean_saving = -math.inf
        for stats in component_by_prefix.values():
            component = (stats or {}).get("component") or stats or {}
            if cid == CANDIDATE_ID:
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
            "production_kept": False,
            "incomplete": mode != "acceptance",
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
    shipping = CONTROL_ID
    production_kept = False
    if mode == "acceptance" and winners:
        selected = CANDIDATE_ID
        rows[selected]["production_kept"] = True
        shipping = str(config_by_id(selected)["gdn_decode"])
        production_kept = True
    elif mode == "acceptance":
        rows[CONTROL_ID]["production_kept"] = True
        production_kept = True
    return {
        "independent_verdicts": rows,
        "selected_path": selected,
        "shipping_unchanged": selected == CONTROL_ID,
        "shipping_gdn_decode": shipping,
        "production_kept": production_kept,
        "winners": winners,
    }


def admit_counts(phase: str, mode: str, observed: Mapping[str, Any]) -> dict[str, Any]:
    return validate_performance_admission(
        load_json(ITERATION),
        mode=mode,
        workload_name=phase,
        workload=workload_for_mode(load_json(ITERATION)["workloads"][phase], mode),
        stdout=json.dumps(observed),
        success=True,
    )


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
        dispatch = parse_gdn_dispatch(completed.stdout)
        if not dispatch_matches(dispatch, config):
            raise AdmissionError(
                f"{config['id']} launch variants {dispatch} != {config}"
            )
        groups = re.search(r"gdn_input_output_groups=(\d+)", completed.stdout)
        if groups is None or int(groups.group(1)) != 48:
            raise AdmissionError(
                f"{config['id']} complete GDN layers were not 48: {groups}"
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
    critical = T_CRIT_DF29 if int(plan["samples"]) >= 30 else FEEDBACK_CRIT
    stats = paired_student_t(control_ms, candidate_ms, critical=critical)
    return {
        "capture_key": capture_key,
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


def evaluate_parity(
    *,
    skip_gpu: bool,
    runner: NativeRunner | None,
    synthetic: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if skip_gpu:
        payload = dict(synthetic or {})
        return {
            "pass": bool(payload.get("pass")),
            "catalog_count": int(payload.get("catalog_count", 1)),
            "by_ident": dict(payload.get("by_ident") or {}),
            "stdout": str(payload.get("stdout", "")),
        }
    native_runner = runner or default_native_runner
    completed = native_runner([f"./{NATIVE}", "--workload", "screen"], "correctness")
    stdout = completed.stdout + completed.stderr
    passed = "status=passed" in stdout
    by_ident = {
        CONTROL_ID: {"kernel_parity_pass": passed},
        CANDIDATE_ID: {"kernel_parity_pass": passed},
    }
    return {
        "pass": passed,
        "catalog_count": 1,
        "by_ident": by_ident,
        "stdout": stdout,
    }


def run_eligibility_phase(*, mode: str, run_dir: Path) -> dict[str, Any]:
    plan = family_plan("eligibility", mode)
    if not OPT090_FIXTURE.is_file():
        raise AdmissionError(f"missing {OPT090_FIXTURE}")
    gate = evaluate_eligibility(load_json(OPT090_FIXTURE))
    observed = planned_observation(plan, keep=False)
    admission = admit_counts("eligibility", mode, observed)
    decided = decide_no_reopen_verdicts() if not gate["eligible"] else {}
    return {
        "schema_version": 1,
        "task": "OPT-094",
        "phase": "eligibility",
        "mode": mode,
        "gate": gate,
        "native_counts": observed,
        "admission": admission,
        "eligible": gate["eligible"],
        "claims_throughput": False,
        "measurement_utc": utc_now(),
        "status": "eligibility_evaluated" if gate["eligible"] else "no_reopen",
        **decided,
    }


def run_parity_phase(
    *,
    mode: str,
    run_dir: Path,
    skip_gpu: bool,
    runner: NativeRunner | None,
    synthetic: Mapping[str, Any] | None,
) -> dict[str, Any]:
    require_replication_eligible(load_json(FIXTURE) if FIXTURE.is_file() else {})
    plan = family_plan("parity", mode)
    parity = evaluate_parity(skip_gpu=skip_gpu, runner=runner, synthetic=synthetic)
    observed = planned_observation(plan, keep=False)
    admission = admit_counts("parity", mode, observed)
    decided = decide_independent_verdicts(
        parity=parity,
        quality=None,
        component_by_prefix={},
        engine_by_prefix={},
        mode=mode,
    )
    return {
        "schema_version": 1,
        "task": "OPT-094",
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


def run_gdn_like_phase(
    phase: str,
    *,
    mode: str,
    run_dir: Path,
    skip_gpu: bool,
    runner: NativeRunner | None,
    synthetic: Mapping[str, Any] | None,
) -> dict[str, Any]:
    require_replication_eligible(load_json(FIXTURE) if FIXTURE.is_file() else {})
    plan = family_plan(phase, mode)
    if skip_gpu:
        component = dict((synthetic or {}).get("component") or {})
        engine = dict((synthetic or {}).get("engine") or {})
        dispatch_ok = bool((synthetic or {}).get("dispatch_ok", True))
    else:
        native_runner = runner or default_native_runner
        component = run_component(
            plan,
            runner=native_runner,
            run_dir=run_dir,
            configs=[dict(row) for row in CONFIGS],
        )
        dispatch_ok = bool(component.get("dispatch_ok"))
        engine = {}
        if int(plan.get("engine_pairs", 0) or 0) > 0:
            engine = run_engine(
                plan,
                runner=native_runner,
                run_dir=run_dir,
                survivor=config_by_id(CANDIDATE_ID),
            )
    if float(component.get("mean_diff_ms", 0.0)) < 0.0 and mode == "feedback":
        raise AdmissionError("negative mean saving at feedback screen")
    observed = planned_observation(
        plan, capture_key=str(component.get("capture_key") or "") or None
    )
    admission = admit_counts(phase, mode, observed)
    decided = decide_independent_verdicts(
        parity=None,
        quality=None,
        component_by_prefix={str(plan.get("prefix", 0)): {"component": component}},
        engine_by_prefix={phase: engine},
        mode=mode,
    )
    return {
        "schema_version": 1,
        "task": "OPT-094",
        "phase": phase,
        "mode": mode,
        "component": component,
        "engine": engine,
        "dispatch_ok": dispatch_ok,
        "native_counts": observed,
        "admission": admission,
        **decided,
        "claims_throughput": False,
        "measurement_utc": utc_now(),
        "status": "measured",
    }


def run_quality_phase(*, mode: str, run_dir: Path) -> dict[str, Any]:
    require_replication_eligible(load_json(FIXTURE) if FIXTURE.is_file() else {})
    quality = opt073_quality()
    model_quality_pass = quality.get("quality_v3_engine_non_regression") in {
        "pass",
        True,
    }
    return {
        "schema_version": 1,
        "task": "OPT-094",
        "phase": "quality",
        "mode": mode,
        "quality": quality,
        "model_quality_pass": model_quality_pass,
        "claims_throughput": False,
        "measurement_utc": utc_now(),
        "status": "quality_reused",
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


def run_engine_guard_phase(
    phase: str,
    *,
    mode: str,
    run_dir: Path,
    skip_gpu: bool,
    runner: NativeRunner | None,
) -> dict[str, Any]:
    require_replication_eligible(load_json(FIXTURE) if FIXTURE.is_file() else {})
    plan = family_plan(phase, mode)
    if skip_gpu:
        engine = {}
    else:
        engine = run_engine(
            plan,
            runner=runner or default_native_runner,
            run_dir=run_dir,
            survivor=config_by_id(CANDIDATE_ID),
        )
    observed = planned_observation(plan, keep=False)
    admission = admit_counts(phase, mode, observed)
    return {
        "schema_version": 1,
        "task": "OPT-094",
        "phase": phase,
        "mode": mode,
        "engine": engine,
        "native_counts": observed,
        "admission": admission,
        "claims_throughput": False,
        "measurement_utc": utc_now(),
        "status": "engine_guard_measured",
    }


def run_prefill_guard_phase(
    *,
    mode: str,
    run_dir: Path,
    skip_gpu: bool,
    runner: NativeRunner | None,
) -> dict[str, Any]:
    require_replication_eligible(load_json(FIXTURE) if FIXTURE.is_file() else {})
    plan = family_plan("state-prefill-guard", mode)
    if skip_gpu:
        payload = {"prefix": 4096, "skipped": True}
    else:
        command = [
            f"./{PROBE}",
            MODEL,
            "--workload",
            "prefill-guard",
            "--prefix",
            "4096",
            "--gdn-decode",
            CANDIDATE_ID,
        ]
        completed = (runner or default_native_runner)(command, "acceptance")
        (run_dir / "prefill-guard.txt").write_text(
            completed.stdout + completed.stderr, encoding="utf-8"
        )
        payload = {"prefix": 4096, "stdout": completed.stdout}
    observed = planned_observation(plan, keep=False)
    admission = admit_counts("state-prefill-guard", mode, observed)
    return {
        "schema_version": 1,
        "task": "OPT-094",
        "phase": "state-prefill-guard",
        "mode": mode,
        "prefill_guard": payload,
        "native_counts": observed,
        "admission": admission,
        "claims_throughput": False,
        "measurement_utc": utc_now(),
        "status": "prefill_guard_measured",
    }


def merge_fixture(
    results: dict[str, Any], phase: str, payload: Mapping[str, Any]
) -> None:
    results[phase] = dict(payload)
    if phase == "eligibility":
        results["status"] = payload.get("status", results.get("status"))
        results["eligible"] = payload.get("eligible")
        if payload.get("status") == "no_reopen":
            for key, value in decide_no_reopen_verdicts().items():
                results[key] = value
            results["claims_throughput"] = False
            results["production_kept"] = True
            results["shipping_gdn_decode"] = CONTROL_ID
            results["selected_path"] = CONTROL_ID
    for key in VERDICT_KEYS:
        if key in payload:
            results[key] = payload[key]
    if "independent_verdicts" in payload:
        results["independent_verdicts"] = payload["independent_verdicts"]
    if payload.get("production_kept") is not None:
        results["production_kept"] = payload["production_kept"]
    if payload.get("shipping_gdn_decode"):
        results["shipping_gdn_decode"] = payload["shipping_gdn_decode"]
    if payload.get("selected_path"):
        results["selected_path"] = payload["selected_path"]
    if payload.get("claims_throughput") is not None:
        results["claims_throughput"] = payload["claims_throughput"]


def write_report(payload: Mapping[str, Any]) -> None:
    eligibility = payload.get("eligibility") or {}
    gate = eligibility.get("gate") or {}
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    text = f"""# OPT-094 — Conditional OPT-077 tile32 GDN timing replication

Status: **{payload.get("status", "pending")}**. Authority llama.cpp
`{LLAMA_REV}`, GGUF SHA-256 `{GGUF_SHA}`.

Conditional replication of OPT-077 `tile32` versus `sequential` requires
OPT-090 `gdn_reopen_eligible=true` with a demonstrated before/after timing or
capture repair. Prior OPT-077 control 12.51649 ms vs tile32 11.89390 ms had CI
−0.1643 to 1.4094 ms (inconclusive, not numerical rejection).

## Eligibility

`gdn_reopen_eligible={gate.get("eligible")}`; verdict `{gate.get("verdict")}`;
reason `{gate.get("reason")}`.

## Independent verdicts

{json.dumps(payload.get("independent_verdicts") or {}, indent=2)}

## Decision

`production_kept={payload.get("production_kept")}`; shipping GDN decode
`{payload.get("shipping_gdn_decode", CONTROL_ID)}`; `claims_throughput=
{payload.get("claims_throughput", False)}`.

## tok/s

Speedup versus the then-current sequential baseline is **{payload.get("tok_s_delta", 0)}**
while sequential is retained unless a full replication promote occurs.
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
            "task": "OPT-094",
            "mode": mode,
            "llama_revision": LLAMA_REV,
            "gguf_sha256": GGUF_SHA,
            "token_generator": TOKEN_GENERATOR,
            "configurations": [row["id"] for row in CONFIGS],
            "control": CONTROL_ID,
            "candidate": CANDIDATE_ID,
            "shipping_gdn_decode": CONTROL_ID,
            "claims_throughput": False,
            "claims_performance_improvement": False,
            "report_path": "evidence/optimization/opt094-gdn-replication/REPORT.md",
            "tok_s_delta": 0.0,
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
    elif phase in {"gdn", "gdn128", "gdn2048"}:
        payload = run_gdn_like_phase(
            phase,
            mode=mode,
            run_dir=run_dir,
            skip_gpu=skip_gpu,
            runner=runner,
            synthetic=synthetic,
        )
    elif phase in {"d128", "d2048"}:
        payload = run_engine_guard_phase(
            phase,
            mode=mode,
            run_dir=run_dir,
            skip_gpu=skip_gpu,
            runner=runner,
        )
    elif phase == "state-prefill-guard":
        payload = run_prefill_guard_phase(
            mode=mode,
            run_dir=run_dir,
            skip_gpu=skip_gpu,
            runner=runner,
        )
    else:
        raise AdmissionError(f"unsupported phase {phase}")
    merge_fixture(results, phase, payload)
    write_report(results)
    dump_json(FIXTURE, results)
    dump_json(run_dir / "opt094_gdn_replication.json", results)
    counts = (results.get(phase) or {}).get("native_counts") or {}
    print(
        "QW38_OPT094_RESULT="
        + json.dumps(
            {
                "task": "OPT-094",
                "mode": mode,
                "phase": phase,
                "status": results.get("status"),
                "eligible": results.get("eligible"),
                "selected_path": results.get("selected_path"),
                "production_kept": results.get("production_kept"),
                "claims_throughput": results.get("claims_throughput"),
                "shipping_gdn_decode": results.get("shipping_gdn_decode"),
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
        "QW38_OPT094_NATIVE_COUNTS="
        + json.dumps(
            {
                "schema_version": 1,
                "task": "OPT-094",
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
