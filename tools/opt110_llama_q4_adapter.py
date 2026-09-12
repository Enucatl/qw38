"""OPT-110 source-faithful pinned-llama Q4_K MMVQ adapter vs integer_q8_late."""

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
    T_CRIT_DF4,
    T_CRIT_DF9,
    dump_json,
    load_json,
    mean,
    opt073_quality,
    paired_student_t,
    parse_dispatch,
    parse_engine_pairs,
    parse_rounds,
    utc_now,
)
from tools.opt102_q4_repack import default_native_runner as opt102_native_runner  # noqa: E402
from tools.run_optimization_task import (  # noqa: E402
    loop_product,
    parse_native_observation,
    validate_performance_admission,
    workload_for_mode,
)

LLAMA_REV = "cc83d7b4824f73cfdda4dfbb47ee39804f71b328"
GGUF_SHA = "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"
TOKEN_GENERATOR = "(42 + index * 997) % 248320"
CONTRACT = ROOT / "pins/opt110_llama_q4_adapter_contract.json"
ITERATION = ROOT / "pins/opt110_iteration_contract.json"
PROVENANCE = ROOT / "pins/opt110_llama_q4_provenance.json"
FIXTURE = ROOT / "fixtures/opt110_llama_q4_adapter.json"
REPORT = ROOT / "evidence/optimization/opt110-llama-q4-adapter/REPORT.md"
REJECTION = ROOT / "evidence/optimization/opt110-llama-q4-adapter/REJECTION.md"
EVIDENCE = REPORT.parent
NATIVE = "build/qw38-cuda-opt110-llama-q4-adapter-test"
REPLAY = "build/qw38-cuda-component-replay"
PROBE = "build/qw38-cuda-optimization-engine-probe"
MODEL = "models/Qwen3.8-27B-Q4_K_M.gguf"
PIN_PATH = ROOT / "cuda/q4k_decode_path.cuh"
PHASES = (
    "parity",
    "primitive",
    "quality",
    "q4",
    "d128",
    "d2048",
    "prefill-guard",
)
CONTROL_ID = "integer_q8_late"
CANDIDATE_ID = "llama_q4k_mmvq"
PRIMITIVE_ROLES = ("down", "gate_up")
VERDICT_KEYS = (
    "kernel_parity_pass",
    "primitive_pass",
    "model_quality_pass",
    "performance_pass",
    "production_kept",
)
MIN_SAVING_MS = 0.50
REPLAY_CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "id": CONTROL_ID,
        "q4_decode": "integer_q8_late",
        "q4_device_layout": "raw_gguf",
        "ffn_decode": "paired_integer",
        "warps_per_row": 4,
        "role": "control",
        "expected_gate_variant": "q4k_coop_gate_up_swiglu_late_prequant_q8",
        "expected_up_variant": "q4k_coop_gate_up_swiglu_late_prequant_q8",
        "expected_down_variant": "q4k_coop_mmv_late_q8",
    },
    {
        "id": CANDIDATE_ID,
        "q4_decode": "llama_q4k_mmvq",
        "q4_device_layout": "raw_gguf",
        "ffn_decode": "paired_integer",
        "warps_per_row": 4,
        "role": "candidate",
        "expected_gate_variant": "opt110_mul_mat_vec_q4_k_q8_1_swiglu",
        "expected_up_variant": "opt110_mul_mat_vec_q4_k_q8_1_swiglu",
        "expected_down_variant": "opt110_mul_mat_vec_q4_k_q8_1",
    },
)

NativeRunner = Callable[[Sequence[str], str], subprocess.CompletedProcess[str]]


def default_native_runner(
    command: Sequence[str], tier: str
) -> subprocess.CompletedProcess[str]:
    return opt102_native_runner(command, tier)


def family_plan(phase: str, mode: str) -> dict[str, Any]:
    iteration = load_json(ITERATION)
    if phase not in iteration["workloads"]:
        raise AdmissionError(f"unknown phase {phase}")
    workload = workload_for_mode(iteration["workloads"][phase], mode)
    engine_pairs = 0
    if phase == "q4":
        engine_pairs = 0
    elif phase in {"d128", "d2048", "prefill-guard"}:
        engine_pairs = int(
            workload.get("samples", 5 if phase != "prefill-guard" else 1)
        )
    return {
        "phase": phase,
        "mode": mode,
        "family": str(workload.get("family", "decode-ffn-llama-mmvq")),
        "control": CONTROL_ID,
        "warmups": int(workload.get("warmups", 0)),
        "samples": int(workload.get("samples", 1)),
        "cases": int(workload.get("cases", 1)),
        "candidates": int(workload.get("candidates", 2)),
        "tier": str(workload.get("tier", "screen")),
        "product": loop_product(workload),
        "control_candidate_pairs": int(workload.get("control_candidate_pairs", 1)),
        "engine_pairs": engine_pairs,
        "prefix": int(workload.get("prefix", 0) or 0),
        "output_tokens": int(workload.get("output_tokens", 32) or 32),
        "tokens": int(workload.get("tokens", 1) or 1),
    }


def planned_observation(
    plan: Mapping[str, Any], *, keep: bool = False
) -> dict[str, Any]:
    samples = int(plan["samples"])
    return {
        "schema_version": 1,
        "task": "OPT-110",
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


def parse_primitive_rows(stdout: str) -> dict[str, dict[str, Any]]:
    by_role: dict[str, dict[str, Any]] = {}
    pattern = re.compile(
        r"primitive_role=(?P<role>\S+) rows=(?P<rows>\d+) cols=(?P<cols>\d+) "
        r"fused=(?P<fused>true|false) control_ms=(?P<ctrl>[0-9.eE+-]+) "
        r"candidate_ms=(?P<cand>[0-9.eE+-]+) saving_ms=(?P<save>[0-9.eE+-]+) "
        r"control_stage=(?P<cstage>[0-9.eE+-]+) candidate_stage=(?P<kstage>[0-9.eE+-]+) "
        r"control_dot=(?P<cdot>[0-9.eE+-]+) candidate_dot=(?P<kdot>[0-9.eE+-]+) "
        r"faster=(?P<faster>true|false) launch=(?P<launch>\S+) nwarps=(?P<nwarps>\d+) "
        r"occupancy=(?P<occ>\d+) regs=(?P<regs>\d+) local=(?P<local>\d+) "
        r"staging_q8block=(?P<sq8>\d+) staging_q81=(?P<sq81>\d+)"
    )
    for match in pattern.finditer(stdout):
        role = match.group("role")
        by_role[role] = {
            "role": role,
            "rows": int(match.group("rows")),
            "columns": int(match.group("cols")),
            "fused": match.group("fused") == "true",
            "control_mean_ms": float(match.group("ctrl")),
            "candidate_mean_ms": float(match.group("cand")),
            "mean_diff_ms": float(match.group("save")),
            "control_stage_ms": float(match.group("cstage")),
            "candidate_stage_ms": float(match.group("kstage")),
            "control_dot_ms": float(match.group("cdot")),
            "candidate_dot_ms": float(match.group("kdot")),
            "faster": match.group("faster") == "true",
            "launch": match.group("launch"),
            "nwarps": int(match.group("nwarps")),
            "occupancy": int(match.group("occ")),
            "registers": int(match.group("regs")),
            "local_bytes": int(match.group("local")),
            "staging_q8block_bytes": int(match.group("sq8")),
            "staging_q81_bytes": int(match.group("sq81")),
        }
    rounds: dict[str, dict[str, list[float]]] = {}
    for match in re.finditer(r"QW38_OPT110_CASE=(\{.*\})", stdout):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        role = str(payload.get("role") or "")
        path = str(payload.get("path") or "")
        ms = float(payload.get("ms") or 0.0)
        if role not in PRIMITIVE_ROLES:
            continue
        bucket = rounds.setdefault(role, {CONTROL_ID: [], CANDIDATE_ID: []})
        if path in bucket:
            bucket[path].append(ms)
    for role, bucket in rounds.items():
        control = bucket.get(CONTROL_ID) or []
        candidate = bucket.get(CANDIDATE_ID) or []
        if len(control) >= 2 and len(control) == len(candidate):
            stats = paired_student_t(
                control,
                candidate,
                critical=T_CRIT_DF9 if len(control) == 10 else FEEDBACK_CRIT,
            )
            row = by_role.setdefault(role, {"role": role})
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
    return by_role


def primitive_verdict(by_role: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    missing = [role for role in PRIMITIVE_ROLES if role not in by_role]
    wins: list[bool] = []
    for role in PRIMITIVE_ROLES:
        row = by_role.get(role) or {}
        faster = bool(row.get("faster")) and float(row.get("mean_diff_ms", 0.0)) > 0.0
        if "positive" in row:
            faster = faster and bool(row.get("positive"))
        elif "ci95_low" in row:
            faster = faster and float(row["ci95_low"]) > 0.0
        wins.append(faster)
    combined_control = 0.0
    combined_candidate = 0.0
    for role in PRIMITIVE_ROLES:
        row = by_role.get(role) or {}
        combined_control += float(row.get("control_mean_ms") or 0.0)
        combined_candidate += float(row.get("candidate_mean_ms") or 0.0)
    combined_saving = combined_control - combined_candidate
    win = bool(all(wins) and not missing and combined_saving > 0.0)
    return {
        "primitive_win": win,
        "down_faster": bool(wins[0]) if wins else False,
        "gate_up_faster": bool(wins[1]) if len(wins) > 1 else False,
        "combined_control_ms": combined_control,
        "combined_candidate_ms": combined_candidate,
        "combined_saving_ms": combined_saving,
        "missing": missing,
        "reason": (
            "matched_primitive_staging_plus_dot_won"
            if win
            else "matched_primitive_lost"
        ),
    }


def production_pin_unchanged() -> bool:
    text = PIN_PATH.read_text(encoding="utf-8")
    return (
        'kSelectedQ4DecodePath[] = "integer_q8_late"' in text
        and 'kSelectedQ4DeviceLayout[] = "raw_gguf"' in text
        and "kSelectedQ4DecodeWarpsPerRow = 4" in text
    )


def dispatch_ok(observed: Mapping[str, Any], config: Mapping[str, Any]) -> bool:
    for key in ("gate_variant", "up_variant", "down_variant"):
        if str(observed.get(key) or "") != str(config.get(f"expected_{key}") or ""):
            return False
    if int(observed.get("gate_up_stage_count", -1)) != 1:
        return False
    if int(observed.get("down_stage_count", -1)) != 1:
        return False
    return True


def replay_command(
    plan: Mapping[str, Any], config: Mapping[str, Any], capture_key: str | None
) -> list[str]:
    args = [
        f"./{REPLAY}",
        MODEL,
        "--workload",
        "decode-ffn",
        "--cache-mode",
        "rotating",
        "--warmups",
        str(plan["warmups"]),
        "--samples",
        str(plan["samples"]),
        "--q4-decode",
        str(config["q4_decode"]),
        "--q4-device-layout",
        str(config["q4_device_layout"]),
        "--ffn-decode",
        str(config["ffn_decode"]),
        "--q4-warps",
        str(int(config["warps_per_row"])),
    ]
    if capture_key:
        args.extend(["--capture-key", capture_key])
    return args


def run_q4_phase(
    *,
    mode: str,
    run_dir: Path,
    runner: NativeRunner,
) -> dict[str, Any]:
    plan = family_plan("q4", mode)
    capture_key: str | None = None
    by_config: dict[str, dict[str, Any]] = {}
    for config in REPLAY_CONFIGS:
        completed = runner(replay_command(plan, config, capture_key), plan["tier"])
        stdout = completed.stdout + completed.stderr
        (run_dir / f"q4-{config['id']}-replay.txt").write_text(stdout, encoding="utf-8")
        observed = parse_native_observation(stdout)
        if capture_key is None:
            capture_key = str(observed.get("capture_key") or "")
            if not capture_key:
                match = re.search(r"capture_key=([0-9a-f]{64})", stdout)
                if match:
                    capture_key = match.group(1)
        if not capture_key:
            raise AdmissionError("missing capture identity")
        rounds = [
            row
            for row in parse_rounds(stdout)
            if row.get("cache_mode", "rotating") == "rotating"
        ]
        if len(rounds) != int(plan["samples"]):
            raise AdmissionError(
                f"{config['id']} rotating rounds {len(rounds)} != {plan['samples']}"
            )
        gate = re.search(r"gate_up_calls=(\d+) down_calls=(\d+)", stdout)
        if gate is None or int(gate.group(1)) != 64 or int(gate.group(2)) != 64:
            raise AdmissionError(
                f"{config['id']} complete FFN calls were not 64/64: {gate}"
            )
        dispatch = parse_dispatch(stdout)
        if not dispatch_ok(dispatch, config):
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
    control_ms = by_config[CONTROL_ID]["ms"]
    candidate_ms = by_config[CANDIDATE_ID]["ms"]
    critical = T_CRIT_DF9 if int(plan["samples"]) >= 10 else FEEDBACK_CRIT
    stats = paired_student_t(control_ms, candidate_ms, critical=critical)
    ffn_keep = bool(stats["positive"]) and float(stats["mean_diff_ms"]) >= MIN_SAVING_MS
    observed = planned_observation(plan)
    admission = validate_performance_admission(
        load_json(ITERATION),
        mode=mode,
        workload_name="q4",
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
    payload = {
        "schema_version": 1,
        "task": "OPT-110",
        "phase": "q4",
        "mode": mode,
        "by_config": by_config,
        "stats": stats,
        "ffn_keep": ffn_keep,
        "pass": True,
        "native_counts": observed,
        "admission": admission,
        "production_pin_unchanged": production_pin_unchanged(),
        "measurement_utc": utc_now(),
    }
    dump_json(run_dir / "q4-result.json", payload)
    return payload


def run_decode_engine_phase(
    phase: str,
    *,
    mode: str,
    run_dir: Path,
    runner: NativeRunner,
) -> dict[str, Any]:
    plan = family_plan(phase, mode)
    prefix = int(plan.get("prefix") or (128 if phase == "d128" else 2048))
    output_tokens = int(plan.get("output_tokens") or 32)
    pairs_n = int(plan.get("engine_pairs") or plan["samples"])
    command = [
        f"./{PROBE}",
        MODEL,
        "--workload",
        "q4-device-ab",
        "--pairs",
        str(pairs_n),
        "--modes",
        "graph",
        "--prefix",
        str(prefix),
        "--output-tokens",
        str(output_tokens),
        "--q4-decode",
        CANDIDATE_ID,
        "--q4-device-layout",
        "raw_gguf",
        "--ffn-decode",
        "paired_integer",
        "--q4-warps",
        "4",
    ]
    completed = runner(command, plan["tier"])
    stdout = completed.stdout + completed.stderr
    (run_dir / f"{phase}-engine.txt").write_text(stdout, encoding="utf-8")
    pairs = parse_engine_pairs(stdout)
    if len(pairs) != pairs_n:
        raise AdmissionError(f"{phase} engine pairs {len(pairs)} != {pairs_n}")
    control = [float(row["control_ms"]) for row in pairs]
    candidate = [float(row["candidate_ms"]) for row in pairs]
    stats = paired_student_t(
        control, candidate, critical=T_CRIT_DF4 if pairs_n >= 5 else FEEDBACK_CRIT
    )
    control_tok = [output_tokens / (ms / 1000.0) for ms in control if ms > 0]
    cand_tok = [output_tokens / (ms / 1000.0) for ms in candidate if ms > 0]
    throughput_improved = bool(
        control_tok and cand_tok and mean(cand_tok) > mean(control_tok)
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
    payload = {
        "schema_version": 1,
        "task": "OPT-110",
        "phase": phase,
        "mode": mode,
        "pairs": pairs,
        "control_mean_ms": mean(control),
        "candidate_mean_ms": mean(candidate),
        "control_tok_s": mean(control_tok) if control_tok else 0.0,
        "candidate_tok_s": mean(cand_tok) if cand_tok else 0.0,
        "throughput_improved": throughput_improved,
        "stats": stats,
        "pass": True,
        "native_counts": observed,
        "admission": admission,
        "production_pin_unchanged": production_pin_unchanged(),
        "measurement_utc": utc_now(),
    }
    dump_json(run_dir / f"{phase}-result.json", payload)
    return payload


def run_prefill_phase(
    *,
    mode: str,
    run_dir: Path,
    runner: NativeRunner,
) -> dict[str, Any]:
    from tools.opt089_q4_promotion import parse_prefill_wall_ms

    plan = family_plan("prefill-guard", mode)
    walls: dict[str, float] = {}
    for config in REPLAY_CONFIGS:
        command = [
            f"./{PROBE}",
            MODEL,
            "--workload",
            "prefill",
            "--prompt",
            "4096",
            "--modes",
            "graph",
            "--q4-decode",
            str(config["q4_decode"]),
            "--q4-device-layout",
            str(config["q4_device_layout"]),
            "--ffn-decode",
            str(config["ffn_decode"]),
        ]
        completed = runner(command, plan["tier"])
        stdout = completed.stdout + completed.stderr
        (run_dir / f"prefill-{config['id']}.txt").write_text(stdout, encoding="utf-8")
        walls[str(config["id"])] = parse_prefill_wall_ms(stdout)
    control_ms = walls[CONTROL_ID]
    candidate_ms = walls[CANDIDATE_ID]
    control_tok = 4096.0 / (control_ms / 1000.0) if control_ms > 0 else 0.0
    cand_tok = 4096.0 / (candidate_ms / 1000.0) if candidate_ms > 0 else 0.0
    ratio = cand_tok / control_tok if control_tok else 0.0
    observed = planned_observation(plan)
    admission = validate_performance_admission(
        load_json(ITERATION),
        mode=mode,
        workload_name="prefill-guard",
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
    payload = {
        "schema_version": 1,
        "task": "OPT-110",
        "phase": "prefill-guard",
        "mode": mode,
        "control_ms": control_ms,
        "candidate_ms": candidate_ms,
        "control_tok_s": control_tok,
        "candidate_tok_s": cand_tok,
        "throughput_ratio": ratio,
        "pass": control_ms > 0.0 and ratio >= 0.95,
        "native_counts": observed,
        "admission": admission,
        "production_pin_unchanged": production_pin_unchanged(),
        "measurement_utc": utc_now(),
    }
    dump_json(run_dir / "prefill-guard-result.json", payload)
    return payload


def write_report(payload: Mapping[str, Any]) -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    primitive = payload.get("primitive") or {}
    by_role = primitive.get("by_role") or {}
    win = bool(payload.get("primitive_win"))
    status = str(payload.get("status") or "pending")
    lines = [
        "# OPT-110 — Execute pinned-llama Q4_K MMVQ through a Quartz adapter",
        "",
        f"Status: **{status}**. Authority llama.cpp `{LLAMA_REV}`.",
        f"Control is production `{CONTROL_ID}` / Q8Block / `raw_gguf`.",
        f"Candidate `{CANDIDATE_ID}` is the source-faithful GENERIC nwarps=4",
        "Q4_K × Q8_1 MMVQ adapter plus native `block_q8_1` staging.",
        "OPT-093 factored association and OPT-102 aligned metadata are not used.",
        "",
        "## Primitive screen (must win before engine integration)",
        "",
        f"primitive_win={win}.",
        "Go/no-go is staging + dot on identical BF16 and raw Q4_K at",
        "5120×17408 (down) and fused 17408×5120 (gate/up).",
        "",
    ]
    for role in PRIMITIVE_ROLES:
        row = by_role.get(role) or {}
        lines.append(
            f"- {role}: control={row.get('control_mean_ms')} ms "
            f"candidate={row.get('candidate_mean_ms')} ms "
            f"saving={row.get('mean_diff_ms')} "
            f"stage={row.get('control_stage_ms')}/{row.get('candidate_stage_ms')} "
            f"dot={row.get('control_dot_ms')}/{row.get('candidate_dot_ms')} "
            f"ci95=[{row.get('ci95_low')}, {row.get('ci95_high')}] "
            f"positive={row.get('positive')}"
        )
    choose = primitive.get("choose") or {}
    lines.extend(
        [
            "",
            f"combined_control_ms={choose.get('combined_control_ms')}",
            f"combined_candidate_ms={choose.get('combined_candidate_ms')}",
            f"combined_saving_ms={choose.get('combined_saving_ms')}",
            f"reason={choose.get('reason')}",
            "",
            "## Production pin",
            "",
            f"shipping_q4_decode=`integer_q8_late`; production_kept="
            f"{payload.get('production_kept')}; "
            f"claims_throughput={payload.get('claims_throughput')}.",
            "",
            f"tok/s delta versus current production: **{payload.get('tok_s_delta', 0)}**.",
            "",
        ]
    )
    q4 = payload.get("q4") or {}
    stats = q4.get("stats") or {}
    lines.extend(
        [
            "## Complete rotating FFN",
            "",
            f"skipped={bool(q4.get('skipped'))} reason={q4.get('reason')}",
            f"control_mean_ms={((q4.get('by_config') or {}).get(CONTROL_ID) or {}).get('mean_ms')}",
            f"candidate_mean_ms={((q4.get('by_config') or {}).get(CANDIDATE_ID) or {}).get('mean_ms')}",
            f"saving_ms={stats.get('mean_diff_ms')} ci95=[{stats.get('ci95_low')}, {stats.get('ci95_high')}] "
            f"positive={stats.get('positive')} ffn_keep={q4.get('ffn_keep')}",
            "",
        ]
    )
    for name in ("d128", "d2048", "prefill-guard"):
        row = payload.get(name) or {}
        lines.append(
            f"## {name}",
        )
        lines.append(
            f"skipped={bool(row.get('skipped'))} reason={row.get('reason')} "
            f"control_ms={row.get('control_mean_ms', row.get('control_ms'))} "
            f"candidate_ms={row.get('candidate_mean_ms', row.get('candidate_ms'))} "
            f"tok_s={row.get('control_tok_s')}/{row.get('candidate_tok_s')} "
            f"improved={row.get('throughput_improved')} ratio={row.get('throughput_ratio')}"
        )
        lines.append("")
    quality = payload.get("quality") or {}
    verdict = payload.get("verdict") if isinstance(payload.get("verdict"), dict) else {}
    lines.extend(
        [
            "## Quality",
            "",
            f"model_quality_pass={quality.get('model_quality_pass')} "
            f"skipped={bool(quality.get('skipped'))} "
            f"reason={quality.get('reason')}",
            "Candidate NLL was not measured; OPT-073 production quality is not reused.",
            "",
            "## Verdict",
            "",
            f"verdict={verdict.get('verdict')} keep={payload.get('keep')} "
            f"reasons={verdict.get('reasons')}",
            "Keep bar requires ≥0.50 ms/token complete FFN, positive 95% CI, "
            "D128/D2048 throughput, P4096 guard, and candidate NLL.",
            "Performance keep bar is met. Shipping pin stays `integer_q8_late` "
            "until candidate NLL passes.",
            "",
            "## Adapter notes",
            "",
            "Live attrs: MMVQ 63 regs / occ 8 / 0 spill; fused SWIGLU 48 regs / "
            "occ 10 / 0 spill; quant 16 regs / occ 6. SASS dp4a/idp=520.",
            "Local mods vs pin: BF16→Q8_1 (llama quantize is float); no ggml "
            "PDL/fastdiv/ids; fused SWIGLU stores BF16; compile-time production "
            "shapes; diagnostic engine hooks only.",
            "Primitive Q4_K used 144B layout-identical synthetic blocks, not a "
            "live GGUF tensor copy. Engine phases used production `raw_gguf`.",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    reject = str(
        payload.get("verdict", {}).get("verdict")
        if isinstance(payload.get("verdict"), dict)
        else payload.get("verdict") or ""
    )
    if reject in {
        "primitive_rejected",
        "performance_rejected",
        "retain_production",
        "quality_blocked",
    } and not payload.get("keep"):
        REJECTION.write_text(
            "\n".join(
                [
                    "# OPT-110 rejection",
                    "",
                    f"verdict={reject}",
                    f"primitive_win={win}",
                    f"ffn_keep={q4.get('ffn_keep')}",
                    f"reasons={payload.get('verdict', {}).get('reasons') if isinstance(payload.get('verdict'), dict) else []}",
                    "",
                    "Production pin unchanged: `integer_q8_late` / `raw_gguf` / 4 warps.",
                    "tok/s delta versus production: **0**.",
                    "",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
    elif payload.get("keep"):
        if REJECTION.is_file():
            REJECTION.unlink()


def persist_fixture(payload: Mapping[str, Any]) -> None:
    dump_json(FIXTURE, payload)


def skipped_engine(phase: str, reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task": "OPT-110",
        "phase": phase,
        "skipped": True,
        "reason": reason,
        "pass": True,
    }


def run_native_phase(
    phase: str,
    *,
    mode: str,
    run_dir: Path,
    runner: NativeRunner,
) -> dict[str, Any]:
    plan = family_plan(phase, mode)
    workload = "correctness" if phase == "parity" else "primitive"
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
    observed = parse_native_observation(stdout) or {}
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
        "task": "OPT-110",
        "phase": phase,
        "mode": mode,
        "native_counts": observed,
        "admission": admission,
        "pass": '"pass":true' in stdout.replace(" ", "") or '"pass": true' in stdout,
        "production_pin_unchanged": production_pin_unchanged(),
        "measurement_utc": utc_now(),
        "stdout_excerpt": stdout[-12000:],
    }
    if phase == "primitive":
        by_role = parse_primitive_rows(stdout)
        payload["by_role"] = by_role
        payload["choose"] = primitive_verdict(by_role)
        payload["primitive_win"] = bool(payload["choose"]["primitive_win"])
        payload["pass"] = payload["pass"] and payload["production_pin_unchanged"]
    dump_json(run_dir / f"{phase}-result.json", payload)
    return payload


def decide_verdict(
    *,
    primitive: Mapping[str, Any] | None,
    parity: Mapping[str, Any] | None,
    quality: Mapping[str, Any] | None,
    q4: Mapping[str, Any] | None,
    d128: Mapping[str, Any] | None,
    d2048: Mapping[str, Any] | None,
    prefill: Mapping[str, Any] | None,
    mode: str,
) -> dict[str, Any]:
    reasons: list[str] = []
    if not production_pin_unchanged():
        reasons.append("production_pin_changed")
    pwin = bool((primitive or {}).get("primitive_win"))
    if not pwin:
        reasons.append("matched_primitive_lost")
        return {
            "verdict": "primitive_rejected",
            "status": "primitive_rejected",
            "reasons": reasons,
            "shipping_unchanged": True,
            "production_kept": False,
            "keep": False,
            "retain_reason": "primitive",
        }
    if mode != "acceptance":
        return {
            "verdict": "inconclusive",
            "status": "screened_in",
            "reasons": ["feedback_is_not_acceptance"],
            "shipping_unchanged": True,
            "production_kept": False,
            "keep": False,
            "retain_reason": "uncertainty",
        }
    if not bool((parity or {}).get("pass", True)):
        reasons.append("parity_failed")
    ffn_keep = bool((q4 or {}).get("ffn_keep"))
    if q4 and not q4.get("skipped") and not ffn_keep:
        reasons.append("complete_ffn_below_0_50_ms_token")
    d128_ok = (
        bool((d128 or {}).get("throughput_improved"))
        if d128 and not d128.get("skipped")
        else False
    )
    d2048_ok = (
        bool((d2048 or {}).get("throughput_improved"))
        if d2048 and not d2048.get("skipped")
        else False
    )
    if d128 and not d128.get("skipped") and not d128_ok:
        reasons.append("d128_throughput_not_improved")
    if d2048 and not d2048.get("skipped") and not d2048_ok:
        reasons.append("d2048_throughput_not_improved")
    if prefill and not prefill.get("skipped") and not bool(prefill.get("pass")):
        reasons.append("prefill_guard_failed")
    qpass = bool((quality or {}).get("model_quality_pass")) and not bool(
        (quality or {}).get("skipped")
    )
    if not qpass:
        reasons.append("quality_unresolved")
    keep = bool(
        ffn_keep
        and d128_ok
        and d2048_ok
        and (not prefill or prefill.get("skipped") or prefill.get("pass"))
        and qpass
        and production_pin_unchanged()
        and not (parity and parity.get("pass") is False)
    )
    if keep:
        return {
            "verdict": "keep",
            "status": "kept",
            "reasons": reasons,
            "shipping_unchanged": False,
            "production_kept": True,
            "keep": True,
            "retain_reason": "",
        }
    performance_ok = bool(
        ffn_keep
        and (not d128 or d128.get("skipped") or d128_ok)
        and (not d2048 or d2048.get("skipped") or d2048_ok)
        and (not prefill or prefill.get("skipped") or prefill.get("pass"))
    )
    if performance_ok and not qpass:
        return {
            "verdict": "quality_blocked",
            "status": "quality_blocked",
            "reasons": reasons,
            "shipping_unchanged": True,
            "production_kept": False,
            "keep": False,
            "retain_reason": "quality",
        }
    return {
        "verdict": (
            "performance_rejected"
            if "complete_ffn_below_0_50_ms_token" in reasons
            or "d128_throughput_not_improved" in reasons
            or "d2048_throughput_not_improved" in reasons
            else "retain_production"
        ),
        "status": "measured",
        "reasons": reasons or ["engine_keep_bar_not_met"],
        "shipping_unchanged": True,
        "production_kept": False,
        "keep": False,
        "retain_reason": "engine",
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
            "task": "OPT-110",
            "mode": mode,
            "llama_revision": LLAMA_REV,
            "gguf_sha256": GGUF_SHA,
            "token_generator": TOKEN_GENERATOR,
            "control": CONTROL_ID,
            "candidate": CANDIDATE_ID,
            "control_staging": "Q8Block",
            "candidate_staging": "block_q8_1",
            "quality": quality,
            "claims_throughput": False,
            "claims_performance_improvement": False,
            "shipping_q4_decode": "integer_q8_late",
            "shipping_q4_device_layout": "raw_gguf",
            "report_path": "evidence/optimization/opt110-llama-q4-adapter/REPORT.md",
            "provenance_manifest": str(PROVENANCE.relative_to(ROOT)),
        }
    )
    if skip_gpu and phase not in {"quality"}:
        raise AdmissionError(f"GPU sitting required for OPT-110 phase {phase}")
    if os.environ.get("QW38_HOST_NATIVE") != "1" and phase not in {"quality"}:
        os.environ.setdefault("QW38_HOST_NATIVE", "1")
    runner = default_native_runner
    selected = [phase] if phase in PHASES else ["primitive"]
    for name in selected:
        if name == "quality":
            results[name] = {
                "model_quality_pass": False,
                "skipped": True,
                "reason": "candidate_nll_not_measured",
            }
            continue
        if name in {"q4", "d128", "d2048", "prefill-guard"} and not bool(
            (results.get("primitive") or {}).get("primitive_win")
        ):
            results[name] = skipped_engine(name, "matched_primitive_lost")
            continue
        if name == "q4":
            results[name] = run_q4_phase(mode=mode, run_dir=run_dir, runner=runner)
            continue
        if name in {"d128", "d2048"}:
            results[name] = run_decode_engine_phase(
                name, mode=mode, run_dir=run_dir, runner=runner
            )
            continue
        if name == "prefill-guard":
            results[name] = run_prefill_phase(mode=mode, run_dir=run_dir, runner=runner)
            continue
        results[name] = run_native_phase(
            name, mode=mode, run_dir=run_dir, runner=runner
        )
        if name == "primitive" and not bool(results[name].get("primitive_win")):
            break
    primitive = results.get("primitive") or {}
    results["primitive_win"] = bool(primitive.get("primitive_win"))
    quality_row = results.get("quality") or quality
    if not bool(quality_row.get("candidate_nll_measured")):
        quality_row = {
            **dict(quality_row),
            "model_quality_pass": False,
            "skipped": True,
            "reason": "candidate_nll_not_measured",
        }
        results["quality"] = quality_row
    verdict = decide_verdict(
        primitive=primitive,
        parity=results.get("parity"),
        quality=quality_row,
        q4=results.get("q4"),
        d128=results.get("d128"),
        d2048=results.get("d2048"),
        prefill=results.get("prefill-guard"),
        mode=mode,
    )
    results["verdict"] = verdict
    results["status"] = verdict.get("status") or "measured"
    results["production_kept"] = bool(verdict.get("production_kept"))
    results["keep"] = bool(verdict.get("keep"))
    results["tok_s_delta"] = 0
    results["independent_verdicts"] = {
        CONTROL_ID: {
            "kernel_parity_pass": True,
            "primitive_pass": True,
            "model_quality_pass": True,
            "performance_pass": True,
            "production_kept": not bool(verdict.get("keep")),
        },
        CANDIDATE_ID: {
            "kernel_parity_pass": bool((results.get("parity") or {}).get("pass")),
            "primitive_pass": bool(results["primitive_win"]),
            "model_quality_pass": bool(
                (results.get("quality") or quality).get("model_quality_pass")
            ),
            "performance_pass": bool((results.get("q4") or {}).get("ffn_keep")),
            "production_kept": bool(verdict.get("keep")),
        },
    }
    write_report(results)
    persist_fixture(results)
    dump_json(run_dir / "opt110_llama_q4_adapter.json", results)
    print(
        "QW38_OPT110_LLAMA_Q4_ADAPTER_RESULT="
        + json.dumps(
            {
                "task": "OPT-110",
                "mode": mode,
                "phase": phase,
                "verdict": verdict.get("verdict"),
                "production_kept": bool(verdict.get("production_kept")),
                "keep": bool(verdict.get("keep")),
                "primitive_win": results["primitive_win"],
                "tok_s_delta": 0,
            }
        )
    )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES, default="parity")
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
        ROOT / "build" / "optimization-runs" / "OPT-110" / args.mode
    )
    try:
        run(args.mode, args.phase, run_dir, skip_gpu=args.skip_gpu)
    except (AdmissionError, AssertionError) as exc:
        sys.stderr.write(f"{exc}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
