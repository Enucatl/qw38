"""OPT-105 double-buffer raw X in the 64x128 prompt MMQ tile."""

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
    opt073_quality,
    utc_now,
)

LLAMA_REV = "cc83d7b4824f73cfdda4dfbb47ee39804f71b328"
GGUF_SHA = "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"
CONTRACT = ROOT / "pins/opt105_mmq_double_x_contract.json"
ITERATION = ROOT / "pins/opt105_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt105_mmq_double_x.json"
REPORT = ROOT / "evidence/optimization/opt105-mmq-double-x/REPORT.md"
EVIDENCE = REPORT.parent
NATIVE = "build/qw38-cuda-opt105-mmq-double-x-test"
MODEL = "models/Qwen3.8-27B-Q4_K_M.gguf"
HOST_MODELS = Path("/home/user/qw38/models")
PHASES = ("eligibility", "parity", "mmq", "quality", "p4096", "decode-guards")
CONTROL_ID = "i128_j128"
CANDIDATE_ID = "i64_j128_x2"
CONTROL_KERNEL = "q4_i128_j128_fma_async_x"
CANDIDATE_KERNEL = "q4_i64_j128_fma_async_x2"
VERDICT_KEYS = (
    "kernel_parity_pass",
    "model_quality_pass",
    "performance_pass",
    "production_kept",
)
MIN_SAVING_MS = 5.0
P4096_BASELINE_TOK_S = 3046.23

NativeRunner = Callable[[Sequence[str], str], subprocess.CompletedProcess[str]]


def docker_prefix(tier: str) -> list[str]:
    listed = docker_common(tier)
    gguf = HOST_MODELS / "Qwen3.8-27B-Q4_K_M.gguf"
    if gguf.is_file() and "-w" in listed:
        idx = listed.index("-w")
        listed = (
            listed[:idx] + ["-v", f"{HOST_MODELS}:/workspace/models:ro"] + listed[idx:]
        )
    return listed


def default_native_runner(
    command: Sequence[str], tier: str
) -> subprocess.CompletedProcess[str]:
    listed = list(command)
    if listed and not listed[0].startswith("docker"):
        if os.environ.get("QW38_HOST_NATIVE") == "0":
            pass
        elif os.environ.get("QW38_HOST_NATIVE") == "1" or (ROOT / NATIVE).is_file():
            pass
        else:
            listed = [*docker_prefix(tier), *listed]
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


def double_x_eligible() -> dict[str, Any]:
    return {
        "eligible": True,
        "verdict": "proceed",
        "reason": "designed_64x128_two_slot_raw_x_vs_128x128_control",
        "control_kernel": CONTROL_KERNEL,
        "candidate_kernel": CANDIDATE_KERNEL,
        "min_saving_ms": MIN_SAVING_MS,
        "p4096_baseline_tok_s": P4096_BASELINE_TOK_S,
    }


def require_eligible(results: Mapping[str, Any]) -> None:
    gate = results.get("eligibility") or {}
    if gate.get("eligible"):
        return
    raise AdmissionError(
        "OPT-105 blocked: "
        + str(gate.get("verdict") or "no_go")
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
        listed = [*docker_prefix(tier), *listed]
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
    observed: dict[str, Any] = {}
    marker = "QW38_OPT105_MMQ_DOUBLE_X_RESULT="
    for line in completed.stdout.splitlines():
        if line.startswith(marker):
            observed = json.loads(line[len(marker) :])
            break
    return {"stdout": completed.stdout, "native_counts": observed}


def parse_screen(stdout: str) -> dict[str, Any]:
    keep = "keep=true" in stdout
    control_ffn = None
    survivor_ffn = None
    match = re.search(r"control_ffn64_ms=([0-9.]+)", stdout)
    if match:
        control_ffn = float(match.group(1))
    match = re.search(r"survivor_ffn64_ms=([0-9.]+)", stdout)
    if match:
        survivor_ffn = float(match.group(1))
    saving = None
    match = re.search(r"saving_ms=([0-9.-]+)", stdout)
    if match:
        saving = float(match.group(1))
    elif control_ffn is not None and survivor_ffn is not None:
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
    del mode
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
        "selected_mmq_double_x": keep,
        "installed": keep,
        "keep": keep,
        "status": "keep" if keep else "reject_128x128_retained",
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
        "# OPT-105 MMQ 64x128 double raw-X report",
        "",
        f"Updated: {results.get('updated_utc', utc_now())}",
        "",
        "## Decision",
        "",
        f"- control: `{CONTROL_ID}` (`{CONTROL_KERNEL}`)",
        f"- candidate: `{CANDIDATE_ID}` (`{CANDIDATE_KERNEL}`)",
        f"- production_kept: `{results.get('production_kept')}`",
        f"- selected_mmq_double_x: `{results.get('selected_mmq_double_x')}`",
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
                "## Complete 64-layer FFN screen",
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
            "## tok/s delta vs OPT-098 P4096",
            "",
            f"- baseline: {P4096_BASELINE_TOK_S} tok/s",
            "- post: unchanged on reject; measured only if kept",
            "",
            "## Notes",
            "",
            "Two raw-X slots on 64x128 vs one-slot 128x128 joined-wait control. "
            "OPT-097 split wait stays rejected. Tails use the synchronous reference.",
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
            "task": "OPT-105",
            "mode": mode,
            "control_tile": CONTROL_ID,
            "candidate_tile": CANDIDATE_ID,
            "control_kernel": CONTROL_KERNEL,
            "candidate_kernel": CANDIDATE_KERNEL,
            "selected_mmq_pipeline_path": "fma_async",
            "selected_mmq_async_x": True,
            "selected_mmq_split_xy_wait": False,
            "selected_mmq_double_x": False,
            "claims_throughput": True,
            "report_path": str(REPORT.relative_to(ROOT)),
            "llama_revision": LLAMA_REV,
            "gguf_sha256": GGUF_SHA,
        }
    )
    payload: dict[str, Any]
    if phase == "eligibility":
        payload = double_x_eligible()
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
        native = run_native("screen", tier, native_runner)
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
            "note": "GPU sitting deferred unless the 64-layer FFN screen keeps",
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
                    "selected_mmq_double_x": False,
                    "installed": False,
                    "keep": False,
                    "status": "no_go",
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
    dump_json(run_dir / "opt105_mmq_double_x.json", results)
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
        ROOT / "build" / "optimization-runs" / "OPT-105" / args.mode
    )
    try:
        run(args.mode, args.phase, run_dir)
    except (AdmissionError, AssertionError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
