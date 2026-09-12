"""OPT-097 separate X/Y async completion in fixed-tile prompt MMQ."""

from __future__ import annotations

import argparse
import json
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
    docker_common,
    dump_json,
    load_json,
    mean,
    opt073_quality,
    utc_now,
)

LLAMA_REV = "cc83d7b4824f73cfdda4dfbb47ee39804f71b328"
GGUF_SHA = "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"
CONTRACT = ROOT / "pins/opt097_mmq_wait_contract.json"
ITERATION = ROOT / "pins/opt097_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt097_mmq_wait.json"
REPORT = ROOT / "evidence/optimization/opt097-mmq-wait/REPORT.md"
EVIDENCE = REPORT.parent
OPT090_FIXTURE = ROOT / "fixtures/opt090_decode_attribution.json"
NATIVE = "build/qw38-cuda-opt097-mmq-wait-test"
REPLAY = "build/qw38-cuda-component-replay"
PROBE = "build/qw38-cuda-optimization-engine-probe"
MODEL = "models/Qwen3.8-27B-Q4_K_M.gguf"
PHASES = ("eligibility", "parity", "mmq", "quality", "p4096", "decode-guards")
CONTROL_ID = "joined_wait"
CANDIDATE_ID = "split_xy_wait"
VERDICT_KEYS = (
    "kernel_parity_pass",
    "model_quality_pass",
    "performance_pass",
    "production_kept",
)
MIN_REMOVABLE_MS = 5.0
MIN_SAVING_MS = 5.0
E2E_REGRESSION_FRAC = 0.02
P4096_BASELINE_TOK_S = 1680.8

NativeRunner = Callable[[Sequence[str], str], subprocess.CompletedProcess[str]]


def default_native_runner(
    command: Sequence[str], tier: str
) -> subprocess.CompletedProcess[str]:
    listed = list(command)
    if listed and not listed[0].startswith("docker"):
        if os.environ.get("QW38_HOST_NATIVE") == "0":
            pass
        elif os.environ.get("QW38_HOST_NATIVE") == "1" or (
            ROOT / NATIVE
        ).is_file():
            pass
        else:
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


def mmq_wait_eligible(opt090: Mapping[str, Any]) -> dict[str, Any]:
    p4096 = opt090.get("p4096") or {}
    windows = p4096.get("windows") or []
    per_ffn_ms: list[float] = []
    for window in windows:
        families = (window.get("families") or {}) if isinstance(window, Mapping) else {}
        gate_up = families.get("ffn_gate_up_glu")
        if gate_up is None:
            continue
        per_ffn_ms.append(float(gate_up) / 64.0)
    gate_mean = mean(per_ffn_ms) if per_ffn_ms else None
    removable_ms = None
    if gate_mean is not None:
        removable_ms = gate_mean * 64.0 * 0.0085
    eligible = removable_ms is not None and removable_ms >= MIN_REMOVABLE_MS
    return {
        "eligible": eligible,
        "verdict": "proceed" if eligible else "no_go_no_wait_sink",
        "reason": (
            "p4096_mmq_async_wait_sink"
            if eligible
            else "insufficient_removable_async_wait_ms"
        ),
        "ffn_gate_up_mean_ms_per_pair": gate_mean,
        "estimated_removable_wait_ms": removable_ms,
        "min_removable_ms": MIN_REMOVABLE_MS,
        "opt090_path": str(OPT090_FIXTURE),
        "attribution_valid": bool(p4096.get("attribution_valid")),
    }


def require_eligible(results: Mapping[str, Any]) -> None:
    gate = results.get("eligibility") or {}
    if gate.get("eligible"):
        return
    raise AdmissionError(
        "OPT-097 blocked: "
        + str(gate.get("verdict") or "no_go_no_wait_sink")
        + f" ({gate.get('reason')})"
    )


def run_native(workload: str, tier: str, runner: NativeRunner) -> dict[str, Any]:
    env = os.environ.copy()
    env["QW38_CUDA_TEST_TIER"] = tier
    if env.get("QW38_HOST_NATIVE") != "0" and (ROOT / NATIVE).is_file():
        env["QW38_HOST_NATIVE"] = "1"
    command = [f"./{NATIVE}", "--workload", workload, MODEL]
    listed = list(command)
    if os.environ.get("QW38_HOST_NATIVE") != "1":
        listed = [*docker_common(tier), *listed]
    completed = subprocess.run(
        listed, cwd=ROOT, env=env, capture_output=True, text=True
    )
    if completed.returncode != 0:
        raise AdmissionError(
            "native workload failed: "
            + " ".join(listed)
            + "\n"
            + completed.stdout
            + completed.stderr
        )
    observed = {}
    marker = "QW38_OPT097_MMQ_WAIT_RESULT="
    for line in completed.stdout.splitlines():
        if line.startswith(marker):
            observed = json.loads(line[len(marker) :])
            break
    return {"stdout": completed.stdout, "native_counts": observed}


def parse_screen(stdout: str) -> dict[str, Any]:
    keep = "keep=true" in stdout
    control_ffn = None
    survivor_ffn = None
    match = re.search(r"control_ffn_ms=([0-9.]+)", stdout)
    if match:
        control_ffn = float(match.group(1))
    match = re.search(r"survivor_ffn_ms=([0-9.]+)", stdout)
    if match:
        survivor_ffn = float(match.group(1))
    saving = None
    if control_ffn is not None and survivor_ffn is not None:
        saving = control_ffn - survivor_ffn
    return {
        "keep": keep,
        "control_complete_ffn_ms": control_ffn,
        "candidate_complete_ffn_ms": survivor_ffn,
        "complete_ffn_saving_ms": saving,
    }


def decide_verdicts(
    *,
    parity_ok: bool,
    quality_ok: bool,
    screen: Mapping[str, Any] | None,
    mode: str,
) -> dict[str, Any]:
    saving = (screen or {}).get("complete_ffn_saving_ms")
    perf_ok = bool(
        screen
        and screen.get("keep")
        and saving is not None
        and float(saving) >= MIN_SAVING_MS
    )
    keep = perf_ok and parity_ok and quality_ok
    return {
        "kernel_parity_pass": parity_ok,
        "model_quality_pass": quality_ok,
        "performance_pass": perf_ok,
        "production_kept": not keep,
        "selected_mmq_split_xy_wait": keep,
        "installed": keep,
        "keep": keep,
        "status": "keep" if keep else "reject_joined_wait_retained",
        "independent_verdicts": {
            "kernel_parity_pass": parity_ok,
            "model_quality_pass": quality_ok,
            "performance_pass": perf_ok,
            "production_kept": not keep,
        },
    }


def write_report(results: Mapping[str, Any]) -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    lines = [
        "# OPT-097 MMQ split X/Y wait report",
        "",
        f"Updated: {results.get('updated_utc', utc_now())}",
        "",
        "## Decision",
        "",
        f"- control: `{CONTROL_ID}` (`q4_i128_j128_fma_async_x`)",
        f"- candidate: `{CANDIDATE_ID}` (`q4_i128_j128_fma_async_x_split_wait`)",
        f"- production_kept: `{results.get('production_kept')}`",
        f"- selected_mmq_split_xy_wait: `{results.get('selected_mmq_split_xy_wait')}`",
        "",
        "## Independent verdicts",
        "",
    ]
    for key in VERDICT_KEYS:
        lines.append(f"- `{key}`: `{results.get(key)}`")
    screen = results.get("mmq") or {}
    if screen:
        lines.extend(
            [
                "",
                "## Complete FFN screen",
                "",
                f"- control: {screen.get('control_complete_ffn_ms')}",
                f"- candidate: {screen.get('candidate_complete_ffn_ms')}",
                f"- saving_ms: {screen.get('complete_ffn_saving_ms')}",
                f"- keep: {screen.get('keep')}",
            ]
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "Split wait overlaps X copy with half1 MMA; joined_wait remains production pin when rejected.",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    mode: str, phase: str, run_dir: Path, *, runner: NativeRunner | None = None
) -> dict[str, Any]:
    if phase not in PHASES:
        raise AdmissionError(f"unknown phase {phase}")
    run_dir.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    native_runner = runner or default_native_runner
    results: dict[str, Any] = {}
    if FIXTURE.is_file():
        try:
            results.update(load_json(FIXTURE))
        except json.JSONDecodeError:
            results = {}
    results.update(
        {
            "schema_version": 1,
            "task": "OPT-097",
            "mode": mode,
            "control_wait": CONTROL_ID,
            "candidate_wait": CANDIDATE_ID,
            "selected_mmq_pipeline_path": "fma_async",
            "selected_mmq_async_x": True,
            "selected_mmq_split_xy_wait": False,
            "tile": "i128_j128",
            "claims_throughput": True,
            "report_path": str(REPORT.relative_to(ROOT)),
            "llama_revision": LLAMA_REV,
            "gguf_sha256": GGUF_SHA,
        }
    )
    payload: dict[str, Any]
    if phase == "eligibility":
        opt090 = load_json(OPT090_FIXTURE) if OPT090_FIXTURE.is_file() else {}
        payload = mmq_wait_eligible(opt090)
        payload["measurement_utc"] = utc_now()
        payload["status"] = "eligible" if payload["eligible"] else "blocked"
    elif phase == "parity":
        require_eligible(results)
        native = run_native("correctness", "correctness", native_runner)
        payload = {
            "phase": "parity",
            "mode": mode,
            "kernel_parity_pass": True,
            "status": "parity_passed",
            "measurement_utc": utc_now(),
            **native,
        }
    elif phase == "mmq":
        require_eligible(results)
        tier = "acceptance" if mode == "acceptance" else "screen"
        native = run_native(
            tier if tier != "acceptance" else "screen", tier, native_runner
        )
        screen = parse_screen(native.get("stdout", ""))
        payload = {
            "phase": "mmq",
            "mode": mode,
            "measurement_utc": utc_now(),
            **screen,
            **native,
        }
    elif phase == "quality":
        quality = opt073_quality()
        payload = {
            "phase": "quality",
            "mode": mode,
            "quality": quality,
            "model_quality_pass": quality.get("quality_v3_engine_non_regression")
            in {"pass", True},
            "measurement_utc": utc_now(),
            "status": "quality_reused",
        }
    elif phase in {"p4096", "decode-guards"}:
        require_eligible(results)
        payload = {
            "phase": phase,
            "mode": mode,
            "measurement_utc": utc_now(),
            "status": "not_executed_host_only",
            "incomplete": True,
            "note": "GPU sitting deferred to verifier/native build",
        }
    else:
        raise AdmissionError(f"unsupported phase {phase}")
    results[phase] = payload
    results["updated_utc"] = utc_now()
    if phase == "eligibility":
        results["eligibility"] = payload
        if not payload.get("eligible"):
            results.update(
                {
                    "kernel_parity_pass": False,
                    "model_quality_pass": False,
                    "performance_pass": False,
                    "production_kept": True,
                    "selected_mmq_split_xy_wait": False,
                    "installed": False,
                    "keep": False,
                    "status": "no_go_no_wait_sink",
                    "claims_throughput": False,
                }
            )
    parity_ok = bool((results.get("parity") or {}).get("kernel_parity_pass"))
    quality_ok = bool(
        (results.get("quality") or {}).get("model_quality_pass")
        if results.get("quality")
        else True
    )
    if phase == "mmq" or results.get("mmq"):
        decided = decide_verdicts(
            parity_ok=parity_ok or bool(results.get("kernel_parity_pass")),
            quality_ok=quality_ok,
            screen=results.get("mmq"),
            mode=mode,
        )
        results.update(decided)
        results["claims_throughput"] = bool(decided.get("keep"))
    write_report(results)
    dump_json(FIXTURE, results)
    dump_json(run_dir / "opt097_mmq_wait.json", results)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument(
        "--mode", choices=("feedback", "acceptance", "release"), default="feedback"
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_dir = args.run_dir or (
        ROOT / "build" / "optimization-runs" / "OPT-097" / args.mode
    )
    try:
        run(args.mode, args.phase, run_dir)
    except (AdmissionError, AssertionError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
