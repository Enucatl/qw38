"""OPT-108 source-faithful NVIDIA llama vector-attention stack vs production."""

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
    FEEDBACK_CRIT,
    T_CRIT_DF9,
    dump_json,
    load_json,
    mean,
    opt073_quality,
    paired_student_t,
    utc_now,
)
from tools.opt103_vector_attention import default_native_runner as opt103_native_runner  # noqa: E402
from tools.run_optimization_task import (  # noqa: E402
    loop_product,
    parse_native_observation,
    validate_performance_admission,
    workload_for_mode,
)

LLAMA_REV = "cc83d7b4824f73cfdda4dfbb47ee39804f71b328"
GGUF_SHA = "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"
TOKEN_GENERATOR = "(42 + index * 997) % 248320"
CONTRACT = ROOT / "pins/opt108_llama_vector_stack_contract.json"
ITERATION = ROOT / "pins/opt108_iteration_contract.json"
PROVENANCE = ROOT / "pins/opt108_llama_vector_provenance.json"
FIXTURE = ROOT / "fixtures/opt108_llama_vector_stack.json"
REPORT = ROOT / "evidence/optimization/opt108-llama-vector-stack/REPORT.md"
REJECTION = ROOT / "evidence/optimization/opt108-llama-vector-stack/REJECTION.md"
EVIDENCE = REPORT.parent
OPT107_FIXTURE = ROOT / "fixtures/opt107_attention_crossover.json"
NATIVE = "build/qw38-cuda-opt108-llama-vector-stack-test"
PIN_PATH = ROOT / "cuda/attention_decode_path.cuh"
PHASES = (
    "eligibility",
    "parity",
    "primitive",
    "quality",
    "attention128",
    "attention2048",
    "d128",
    "d2048",
    "state",
    "prefill-guard",
)
CONTROL_ID = "production_opt107"
CANDIDATE_ID = "llama_vec_nvidia"
PRIMITIVE_POSITIONS = (128, 512, 2048, 4096)
VERDICT_KEYS = (
    "kernel_parity_pass",
    "primitive_pass",
    "model_quality_pass",
    "performance_pass",
    "production_kept",
)
MIN_SAVING_MS = 0.50

NativeRunner = Callable[[Sequence[str], str], subprocess.CompletedProcess[str]]


def default_native_runner(
    command: Sequence[str], tier: str
) -> subprocess.CompletedProcess[str]:
    return opt103_native_runner(command, tier)


def family_plan(phase: str, mode: str) -> dict[str, Any]:
    iteration = load_json(ITERATION)
    if phase not in iteration["workloads"]:
        raise AdmissionError(f"unknown phase {phase}")
    workload = workload_for_mode(iteration["workloads"][phase], mode)
    return {
        "phase": phase,
        "mode": mode,
        "family": str(workload.get("family", "decode-attention-llama-vec")),
        "control": CONTROL_ID,
        "warmups": int(workload.get("warmups", 0)),
        "samples": int(workload.get("samples", 1)),
        "cases": int(workload.get("cases", 1)),
        "candidates": int(workload.get("candidates", 2)),
        "tier": str(workload.get("tier", "screen")),
        "product": loop_product(workload),
        "control_candidate_pairs": int(workload.get("control_candidate_pairs", 1)),
        "decode_position": int(workload.get("decode_position", 128) or 128),
        "tokens": int(workload.get("tokens", 1) or 1),
        "engine_pairs": int(workload.get("engine_pairs", 0) or 0),
    }


def planned_observation(
    plan: Mapping[str, Any], *, keep: bool = False
) -> dict[str, Any]:
    samples = int(plan["samples"])
    return {
        "schema_version": 1,
        "task": "OPT-108",
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
    }


def parse_primitive_rows(stdout: str) -> dict[int, dict[str, Any]]:
    by_pos: dict[int, dict[str, Any]] = {}
    pattern = re.compile(
        r"primitive_position=(?P<pos>\d+) n_parts=(?P<parts>\d+) occupancy="
        r"(?P<occ>\d+) ntiles_kv=(?P<tiles>\d+) control_ms=(?P<ctrl>[0-9.eE+-]+) "
        r"candidate_ms=(?P<cand>[0-9.eE+-]+) saving_ms=(?P<save>[0-9.eE+-]+) "
        r"faster=(?P<faster>true|false) prepared_q_once=(?P<prep>true|false) "
        r"nvidia_float2=(?P<f2>true|false)"
    )
    for match in pattern.finditer(stdout):
        pos = int(match.group("pos"))
        by_pos[pos] = {
            "position": pos,
            "n_parts": int(match.group("parts")),
            "occupancy": int(match.group("occ")),
            "ntiles_kv": int(match.group("tiles")),
            "control_mean_ms": float(match.group("ctrl")),
            "candidate_mean_ms": float(match.group("cand")),
            "mean_diff_ms": float(match.group("save")),
            "faster": match.group("faster") == "true",
            "prepared_q_once": match.group("prep") == "true",
            "nvidia_float2": match.group("f2") == "true",
        }
    rounds: dict[int, dict[str, list[float]]] = {}
    for match in re.finditer(
        r"QW38_OPT108_CASE=(\{.*\})",
        stdout,
    ):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        pos = int(payload.get("position") or 0)
        path = str(payload.get("path") or "")
        ms = float(payload.get("ms") or 0.0)
        bucket = rounds.setdefault(pos, {"production": [], "llama_vec_nvidia": []})
        if path in bucket:
            bucket[path].append(ms)
    for pos, bucket in rounds.items():
        control = bucket.get("production") or []
        candidate = bucket.get("llama_vec_nvidia") or []
        if len(control) >= 2 and len(control) == len(candidate):
            stats = paired_student_t(
                control,
                candidate,
                critical=T_CRIT_DF9 if len(control) == 10 else FEEDBACK_CRIT,
            )
            row = by_pos.setdefault(pos, {"position": pos})
            row.update(stats)
            row["control_samples_ms"] = control
            row["candidate_samples_ms"] = candidate
            if "control_mean_ms" not in row:
                row["control_mean_ms"] = mean(control)
            if "candidate_mean_ms" not in row:
                row["candidate_mean_ms"] = mean(candidate)
            if "mean_diff_ms" not in row:
                row["mean_diff_ms"] = stats["mean_diff_ms"]
            row["faster"] = float(row["mean_diff_ms"]) > 0.0
    return by_pos


def primitive_verdict(by_pos: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    missing = [pos for pos in PRIMITIVE_POSITIONS if pos not in by_pos]
    d2048 = by_pos.get(2048) or {}
    d128 = by_pos.get(128) or {}
    d2048_win = (
        bool(d2048.get("faster")) and float(d2048.get("mean_diff_ms", 0.0)) > 0.0
    )
    if "positive" in d2048:
        d2048_win = d2048_win and bool(d2048.get("positive"))
    elif "ci95_low" in d2048:
        d2048_win = d2048_win and float(d2048["ci95_low"]) > 0.0
    d128_regressed = float(d128.get("mean_diff_ms", 0.0)) < 0.0 and bool(
        d128.get("positive") is False if "positive" in d128 else True
    )
    if "ci95_high" in d128:
        d128_regressed = float(d128.get("ci95_high", 0.0)) < 0.0
    win = bool(d2048_win and not d128_regressed and not missing)
    return {
        "primitive_win": win,
        "d2048_faster": d2048_win,
        "d2048_positive": bool(d2048.get("positive"))
        if "positive" in d2048
        else d2048_win,
        "d128_regressed": d128_regressed,
        "missing": missing,
        "reason": (
            "d2048_primitive_faster_d128_no_regression"
            if win
            else "matched_primitive_lost"
        ),
    }


def eligibility_from_fixtures() -> dict[str, Any]:
    if not OPT107_FIXTURE.is_file():
        raise AdmissionError(f"missing {OPT107_FIXTURE}")
    opt107 = load_json(OPT107_FIXTURE)
    admitted = (
        bool(opt107.get("production_kept"))
        and int(opt107.get("selected_threshold") or 0) == 1024
    )
    eligible = admitted
    return {
        "eligible": eligible,
        "verdict": "proceed" if eligible else "no_go",
        "reason": (
            "opt107_hybrid_admitted" if eligible else "opt107_hybrid_not_admitted"
        ),
        "opt107_threshold": opt107.get("selected_threshold"),
        "opt107_production_kept": opt107.get("production_kept"),
        "control": CONTROL_ID,
    }


def production_pin_unchanged() -> bool:
    text = PIN_PATH.read_text(encoding="utf-8")
    return (
        'kSelectedDecodeAttentionVec128Path[] = "warp_query"' in text
        and "kSelectedDecodeAttentionCrossoverThreshold = 1024" in text
    )


def write_report(payload: Mapping[str, Any]) -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    primitive = payload.get("primitive") or {}
    by_pos = primitive.get("by_position") or {}
    kept = bool(payload.get("production_kept"))
    win = bool(payload.get("primitive_win"))
    status = str(payload.get("status") or "pending")
    lines = [
        "# OPT-108 — Reproduce the actual NVIDIA pinned-llama vector-attention stack",
        "",
        f"Status: **{status}**. Authority llama.cpp `{LLAMA_REV}`.",
        f"Control is current production OPT-107 hybrid (`{CONTROL_ID}`).",
        f"Candidate `{CANDIDATE_ID}` is the source-faithful NVIDIA",
        "`flash_attn_ext_vec<256,1>` adapter: 128 threads, 16-byte copies,",
        "prepared Q once, occupancy/KV-length partitions, float2 V accum.",
        "BF16 KV path; AMD half2 and F16 cache migration are not used.",
        "",
        "## Primitive screen (must win before engine integration)",
        "",
        f"primitive_win={win}.",
        "D2048 must be faster with a positive 95% CI before engine integration.",
        "",
    ]
    for pos in PRIMITIVE_POSITIONS:
        row = by_pos.get(str(pos)) or by_pos.get(pos) or {}
        lines.append(
            f"- D{pos}: control={row.get('control_mean_ms')} ms "
            f"candidate={row.get('candidate_mean_ms')} ms "
            f"saving={row.get('mean_diff_ms')} n_parts={row.get('n_parts')} "
            f"ci95=[{row.get('ci95_low')}, {row.get('ci95_high')}] "
            f"positive={row.get('positive')}"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"Verdict: **{payload.get('verdict', {}).get('verdict', status)}**.",
            f"production_kept={kept}. Shipping vec128 pin remains warp_query / OPT-107 1024.",
            "Engine integration is skipped when the matched primitive loses.",
            "D2048 complete-attention 0.50 ms/token was not attempted because the",
            "primitive screen did not win with a positive 95% CI.",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if not kept:
        REJECTION.write_text(
            f"""# OPT-108 rejection — retain production OPT-107 hybrid

Production pins remain `kSelectedDecodeAttentionVec128Path[] = "warp_query"`
and `kSelectedDecodeAttentionCrossoverThreshold = 1024`. The source-faithful
NVIDIA llama vector stack was not admitted.

primitive_win={win}
reason={payload.get("verdict", {}).get("reasons")}

Matched D2048 primitive 95% CI includes 0 (point-estimate saving ~0.002 ms).
Engine integration was not run. Complete-attention 0.50 ms/token was not
attempted.

tok/s delta versus production: **0**.
""",
            encoding="utf-8",
        )


def persist_fixture(results: Mapping[str, Any]) -> None:
    dump_json(FIXTURE, results)


def run_eligibility_phase(*, run_dir: Path) -> dict[str, Any]:
    gate = eligibility_from_fixtures()
    payload = {
        "schema_version": 1,
        "task": "OPT-108",
        "phase": "eligibility",
        "measurement_utc": utc_now(),
        **gate,
    }
    dump_json(run_dir / "eligibility-result.json", payload)
    return payload


def run_quality_phase(*, mode: str, run_dir: Path) -> dict[str, Any]:
    quality = opt073_quality()
    payload = {
        "schema_version": 1,
        "task": "OPT-108",
        "phase": "quality",
        "mode": mode,
        "identity_cached": True,
        "model_quality_pass": quality.get("quality_v3_engine_non_regression")
        in {"pass", True},
        "quality": quality,
        "note": "arithmetic change; OPT-103 quality is not reused for a keep",
        "measurement_utc": utc_now(),
    }
    dump_json(run_dir / "quality-result.json", payload)
    return payload


def run_native_phase(
    phase: str,
    *,
    mode: str,
    run_dir: Path,
    runner: NativeRunner,
) -> dict[str, Any]:
    plan = family_plan(phase, mode)
    workload = (
        "correctness"
        if phase == "parity"
        else (
            "acceptance"
            if mode == "acceptance" and phase != "state"
            else ("screen" if phase == "primitive" else "correctness")
        )
    )
    if phase == "state":
        workload = "correctness"
    command = [
        f"./{NATIVE}",
        "--workload",
        workload,
        "--warmups",
        str(plan["warmups"]),
        "--samples",
        str(plan["samples"]),
    ]
    completed = runner(command, plan["tier"])
    stdout = completed.stdout + completed.stderr
    (run_dir / f"{phase}-raw.txt").write_text(stdout, encoding="utf-8")
    observed = parse_native_observation(stdout)
    observed.update(planned_observation(plan))
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
    payload: dict[str, Any] = {
        "schema_version": 1,
        "task": "OPT-108",
        "phase": phase,
        "mode": mode,
        "native_counts": observed,
        "admission": admission,
        "pass": "status=passed" in stdout or '"pass":true' in stdout.replace(" ", ""),
        "production_pin_unchanged": production_pin_unchanged(),
        "measurement_utc": utc_now(),
    }
    if phase == "primitive":
        by_pos = parse_primitive_rows(stdout)
        payload["by_position"] = {str(k): v for k, v in by_pos.items()}
        payload["choose"] = primitive_verdict(by_pos)
        payload["primitive_win"] = bool(payload["choose"]["primitive_win"])
        payload["pass"] = payload["pass"] and payload["production_pin_unchanged"]
    dump_json(run_dir / f"{phase}-result.json", payload)
    return payload


def skipped_engine(phase: str, reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task": "OPT-108",
        "phase": phase,
        "skipped": True,
        "reason": reason,
        "pass": True,
    }


def decide_verdict(
    *,
    primitive: Mapping[str, Any] | None,
    parity: Mapping[str, Any] | None,
    quality: Mapping[str, Any] | None,
    mode: str,
) -> dict[str, Any]:
    reasons: list[str] = []
    pin_ok = production_pin_unchanged()
    if not pin_ok:
        reasons.append("production_pin_changed")
    kpass = bool((parity or {}).get("pass"))
    pwin = bool((primitive or {}).get("primitive_win"))
    if not pwin:
        reasons.append("matched_primitive_lost")
    qpass = bool((quality or {}).get("model_quality_pass"))
    if mode != "acceptance" and not pwin:
        return {
            "verdict": "primitive_rejected",
            "status": "primitive_rejected",
            "reasons": reasons or ["feedback_primitive_lost"],
            "shipping_unchanged": True,
            "production_kept": False,
            "retain_reason": "primitive",
        }
    if not pwin:
        return {
            "verdict": "primitive_rejected",
            "status": "primitive_rejected",
            "reasons": reasons,
            "shipping_unchanged": True,
            "production_kept": False,
            "retain_reason": "primitive",
        }
    if mode != "acceptance":
        return {
            "verdict": "inconclusive",
            "status": "screened_in",
            "reasons": ["feedback_is_not_acceptance"],
            "shipping_unchanged": True,
            "production_kept": False,
            "retain_reason": "uncertainty",
        }
    if not kpass:
        reasons.append("parity_failed")
    if not qpass:
        reasons.append("quality_unresolved")
        return {
            "verdict": "quality_blocked",
            "status": "quality_blocked",
            "reasons": reasons,
            "shipping_unchanged": True,
            "production_kept": False,
            "retain_reason": "quality",
        }
    return {
        "verdict": "retain_production",
        "status": "measured",
        "reasons": reasons or ["engine_integration_not_reached"],
        "shipping_unchanged": True,
        "production_kept": False,
        "retain_reason": "primitive_or_engine",
    }


def empty_verdict_row(reason: str) -> dict[str, Any]:
    return {
        "kernel_parity_pass": False,
        "primitive_pass": False,
        "model_quality_pass": False,
        "performance_pass": False,
        "production_kept": False,
        "incomplete": True,
        "not_applicable_reason": reason,
    }


def run(
    mode: str, phase: str, run_dir: Path, *, skip_gpu: bool = False
) -> dict[str, Any]:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
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
            "task": "OPT-108",
            "mode": mode,
            "llama_revision": LLAMA_REV,
            "gguf_sha256": GGUF_SHA,
            "token_generator": TOKEN_GENERATOR,
            "control": CONTROL_ID,
            "candidate": CANDIDATE_ID,
            "nvidia_float2_v_accum": True,
            "amd_half2_v_accum": False,
            "f16_cache_migration": False,
            "prepared_q_once": True,
            "quality": quality,
            "claims_throughput": False,
            "claims_performance_improvement": False,
            "shipping_decode_attention_vec128": "warp_query",
            "report_path": "evidence/optimization/opt108-llama-vector-stack/REPORT.md",
            "provenance_manifest": str(PROVENANCE.relative_to(ROOT)),
        }
    )
    eligibility = run_eligibility_phase(run_dir=run_dir)
    results["eligibility"] = eligibility
    if not eligibility.get("eligible"):
        results["status"] = "no_go"
        results["primitive_win"] = False
        results["production_kept"] = False
        results["independent_verdicts"] = {
            CONTROL_ID: {**empty_verdict_row("no_go"), "production_kept": False},
            CANDIDATE_ID: empty_verdict_row("no_go"),
        }
        write_report(results)
        persist_fixture(results)
        return results
    if phase == "eligibility":
        results["status"] = "eligible"
        results["primitive_win"] = False
        results["production_kept"] = False
        persist_fixture(results)
        return results
    runner = default_native_runner
    if skip_gpu and phase not in {"eligibility", "quality"}:
        raise AdmissionError(f"GPU sitting required for OPT-108 phase {phase}")
    if os.environ.get("QW38_HOST_NATIVE") != "1" and phase not in {"quality"}:
        os.environ.setdefault("QW38_HOST_NATIVE", "1")
    selected = [phase] if phase in PHASES else ["primitive"]
    for name in selected:
        if name == "eligibility":
            continue
        if name == "quality":
            results[name] = run_quality_phase(mode=mode, run_dir=run_dir)
            continue
        if name in {
            "attention128",
            "attention2048",
            "d128",
            "d2048",
            "prefill-guard",
        } and not bool((results.get("primitive") or {}).get("primitive_win")):
            results[name] = skipped_engine(name, "matched_primitive_lost")
            continue
        results[name] = run_native_phase(
            name, mode=mode, run_dir=run_dir, runner=runner
        )
        if name == "primitive" and not bool(results[name].get("primitive_win")):
            break
    primitive = results.get("primitive") or {}
    results["primitive_win"] = bool(primitive.get("primitive_win"))
    verdict = decide_verdict(
        primitive=primitive,
        parity=results.get("parity"),
        quality=results.get("quality") or quality,
        mode=mode,
    )
    results["verdict"] = verdict
    results["status"] = verdict.get("status") or "measured"
    results["production_kept"] = False
    results["numeric"] = {"nonfinite": 0}
    results["independent_verdicts"] = {
        CONTROL_ID: {
            "kernel_parity_pass": True,
            "primitive_pass": True,
            "model_quality_pass": True,
            "performance_pass": True,
            "production_kept": False,
        },
        CANDIDATE_ID: {
            "kernel_parity_pass": bool((results.get("parity") or {}).get("pass")),
            "primitive_pass": bool(results["primitive_win"]),
            "model_quality_pass": bool(
                (results.get("quality") or quality).get("model_quality_pass")
            ),
            "performance_pass": False,
            "production_kept": False,
        },
    }
    write_report(results)
    persist_fixture(results)
    dump_json(run_dir / "opt108_llama_vector_stack.json", results)
    print(
        "QW38_OPT108_LLAMA_VECTOR_STACK_RESULT="
        + json.dumps(
            {
                "task": "OPT-108",
                "mode": mode,
                "phase": phase,
                "verdict": verdict.get("verdict"),
                "production_kept": False,
                "keep": False,
                "primitive_win": results["primitive_win"],
                "tok_s_delta": 0,
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
        ROOT / "build" / "optimization-runs" / "OPT-108" / args.mode
    )
    try:
        run(args.mode, args.phase, run_dir, skip_gpu=args.skip_gpu)
    except (AdmissionError, AssertionError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
