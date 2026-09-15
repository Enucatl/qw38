"""OPT-140 production-boundary P4096 prompt-attention replay.

Diagnostics only. No attention optimization and no new quality tolerance.
Decode-attention remains an invalid substitute for prefill attn_core.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.opt075_q4_production_admission import dump_json, load_json, utc_now  # noqa: E402
from tools.opt082_kernel_parity import gpu_available  # noqa: E402
from tools.opt139_counter_identity import (  # noqa: E402
    LAUNCH_PREFILL_ATTN,
    OPT140_INELIGIBLE,
    counter_record_from_ncu_blob,
    export_ncu_csv,
    identity_match,
    ncu_kernel_regex,
    production_prefill_attn_identity,
)
from tools.performance_evidence import replay_production_boundary_ok  # noqa: E402
from tools.run_optimization_task import (  # noqa: E402
    loop_product,
    validate_future_keep_policy,
    workload_for_mode,
)

CONTRACT = ROOT / "pins/opt140_prefill_attention_replay_contract.json"
ITERATION = ROOT / "pins/opt140_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt140_prefill_attention_replay.json"
EVIDENCE = ROOT / "evidence/optimization/opt140-prefill-attention-replay"
REPORT = EVIDENCE / "REPORT.md"
REPLAY_BIN = "build/qw38-cuda-component-replay"
MODEL = "models/Qwen3.8-27B-Q4_K_M.gguf"
SELECTOR = "decode_segments8"
PARENT = "post124_plus_opt127_decode_segments8_plus_opt137_mma"
PHASES = ("preflight", "replay", "counters", "report")
CHILD_TIMEOUT_S = 300
PROMPT_ATTN_TIMEOUT_S = 1200
AGGREGATE_DEADLINE_S = 7200
PREFILL_TOKENS = 4096
CAPACITY = 131072
EXISTING_ABS_TOL = 2.0e-3
RESULT_PREFIX = "QW38_OPT140_PREFILL_ATTENTION_REPLAY_RESULT="
REQUIRED_FIXTURE_KEYS = (
    "schema_version",
    "task",
    "status",
    "mode",
    "claims_throughput",
    "claims_performance_improvement",
    "production_kept",
    "selected_execution_graph_path",
    "prefill_tokens",
    "capacity",
    "preflight",
    "replay",
    "counters",
    "answers",
    "report_path",
)

ROUND_LINE = re.compile(
    r"round family=(?P<family>\S+)\s+cache_mode=(?P<cache>\S+)\s+"
    r"warmup=(?P<warmup>\S+)\s+sample_index=(?P<sample>-?\d+)"
    r".*enclosing_ms=(?P<enclosing>[0-9.]+)"
    r".*kernel_only_ms=(?P<kernel>[0-9.]+)"
)
PARITY_LINE = re.compile(
    r"prompt_attention_parity_summary layers=(?P<layers>\d+)\s+"
    r"max_abs=(?P<max_abs>\S+)\s+kv_max_abs=(?P<kv_max_abs>\S+)\s+"
    r"nonfinite=(?P<nonfinite>\d+)\s+equal=(?P<equal>\S+)"
)
DISPATCH_LINE = re.compile(
    r"prompt_attention_dispatch path=(?P<path>\S+) launch=(?P<launch>\S+)"
    r".*capacity=(?P<capacity>\d+)\s+tokens=(?P<tokens>\d+)"
    r".*chunk_count=(?P<chunks>\d+)\s+query_start=(?P<query_start>\d+)"
)
CAPTURE_LINE = re.compile(r"capture_key=([0-9a-f]{64})")
LAYER_PARITY = re.compile(
    r"prompt_attention_parity layer=(?P<layer>\d+)\s+slot=(?P<slot>\d+)"
    r".*causal_tail_abs=(?P<tail>\S+)\s+chunk_start_abs=(?P<start>\S+)"
    r"\s+chunk_mid_abs=(?P<mid>\S+)\s+equal=(?P<equal>\S+)"
)


class PrefillAttentionReplayError(RuntimeError):
    """OPT-140 measurement or policy failure."""


def load_contract() -> dict[str, Any]:
    payload = load_json(CONTRACT)
    if payload.get("task") != "OPT-140":
        raise PrefillAttentionReplayError("prefill-attention contract task mismatch")
    return payload


def load_iteration() -> dict[str, Any]:
    payload = load_json(ITERATION)
    validate_future_keep_policy("OPT-140", payload)
    return payload


def family_plan(mode: str, family: str) -> str:
    iteration = load_iteration()
    workload = workload_for_mode(iteration["workloads"][family], mode)
    product = loop_product(workload)
    return (
        f"task=OPT-140 mode={mode} phase={family} "
        f"loop_product={product} cases={workload.get('cases')} "
        f"candidates={workload.get('candidates')} "
        f"warmups={workload.get('warmups')} samples={workload.get('samples')} "
        f"tokens={workload.get('tokens')}"
    )


def relpath(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def sidecar(run_dir: Path, name: str) -> Path:
    return run_dir / name


def store_sidecar(
    run_dir: Path, name: str, payload: Mapping[str, Any]
) -> dict[str, Any]:
    path = sidecar(run_dir, name)
    dump_json(path, payload)
    return dict(payload)


def load_sidecar(run_dir: Path, name: str) -> dict[str, Any] | None:
    path = sidecar(run_dir, name)
    if not path.is_file():
        return None
    payload = load_json(path)
    return payload if isinstance(payload, dict) else None


def empty_fixture(mode: str = "feedback") -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task": "OPT-140",
        "status": "structure",
        "mode": mode,
        "measurement_utc": None,
        "claims_throughput": False,
        "claims_performance_improvement": False,
        "production_kept": True,
        "selected_execution_graph_path": SELECTOR,
        "parent": PARENT,
        "prefill_tokens": PREFILL_TOKENS,
        "capacity": CAPACITY,
        "existing_abs_tol": EXISTING_ABS_TOL,
        "new_quality_tolerance": False,
        "decode_attention_substitution": False,
        "preflight": None,
        "replay": None,
        "counters": None,
        "answers": {},
        "report_path": relpath(REPORT),
        "nll": "N/A",
        "throughput_shipping_delta": "N/A",
    }


def validate_fixture(payload: Mapping[str, Any]) -> None:
    for key in REQUIRED_FIXTURE_KEYS:
        if key not in payload:
            raise PrefillAttentionReplayError(f"fixture missing {key}")
    if payload.get("claims_throughput") is not False:
        raise PrefillAttentionReplayError("claims_throughput must be false")
    if payload.get("selected_execution_graph_path") != SELECTOR:
        raise PrefillAttentionReplayError("selector drifted from decode_segments8")
    if payload.get("decode_attention_substitution") is not False:
        raise PrefillAttentionReplayError(
            "decode-attention substitution must stay false"
        )
    if payload.get("new_quality_tolerance") is not False:
        raise PrefillAttentionReplayError("OPT-140 must not add a quality tolerance")


def parse_bool(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"true", "1", "yes"}


def parse_rounds(text: str) -> list[dict[str, Any]]:
    rounds: list[dict[str, Any]] = []
    for match in ROUND_LINE.finditer(text or ""):
        rounds.append(
            {
                "family": match.group("family"),
                "cache_mode": match.group("cache"),
                "warmup": parse_bool(match.group("warmup")),
                "sample_index": int(match.group("sample")),
                "enclosing_ms": float(match.group("enclosing")),
                "kernel_only_ms": float(match.group("kernel")),
            }
        )
    return rounds


def parse_layer_parity(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for match in LAYER_PARITY.finditer(text or ""):
        rows.append(
            {
                "layer": int(match.group("layer")),
                "slot": int(match.group("slot")),
                "causal_tail_abs": float(match.group("tail")),
                "chunk_start_abs": float(match.group("start")),
                "chunk_mid_abs": float(match.group("mid")),
                "equal": parse_bool(match.group("equal")),
            }
        )
    return rows


def parse_parity_summary(text: str) -> dict[str, Any] | None:
    match = PARITY_LINE.search(text or "")
    if match is None:
        return None
    return {
        "layers": int(match.group("layers")),
        "max_abs": float(match.group("max_abs")),
        "kv_max_abs": float(match.group("kv_max_abs")),
        "nonfinite": int(match.group("nonfinite")),
        "equal": parse_bool(match.group("equal")),
        "existing_abs_tol": EXISTING_ABS_TOL,
        "new_quality_tolerance": False,
    }


def parse_dispatch(text: str) -> dict[str, Any] | None:
    match = DISPATCH_LINE.search(text or "")
    if match is None:
        return None
    return {
        "path": match.group("path"),
        "launch": match.group("launch"),
        "capacity": int(match.group("capacity")),
        "tokens": int(match.group("tokens")),
        "chunk_count": int(match.group("chunks")),
        "query_start": int(match.group("query_start")),
        "decode_attention_substitution": "decode_attention_substitution=true"
        in (text or ""),
        "complete_family_replay": "complete_family_replay=true" in (text or ""),
    }


def capture_key_from(text: str) -> str | None:
    match = CAPTURE_LINE.search(text or "")
    return match.group(1) if match else None


def boundary_manifest(
    *,
    replay_family: str,
    dispatch: Mapping[str, Any] | None,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    decode_sub = replay_family.startswith("decode-")
    boundary = replay_production_boundary_ok(
        {
            "cache_mode": "rotating",
            "production_weights": True,
            "synthetic_weights": False,
            "phase": "prefill",
            "replay_family": replay_family,
        }
    )
    return {
        "phase": "prefill",
        "replay_family": replay_family,
        "production_leaves": [
            "split_attention_rows",
            "stage_chunk",
            "bf16_f16_convert",
            "fattn_mma_pipeline_opt111_base",
            "merge",
            "epilogue",
            "fp32_to_bf16",
        ],
        "excluded": [
            "mixer_mmq",
            "ffn_mmq",
            "logits",
            "commit_sync",
            "embedding",
        ],
        "complete_family_replay": True,
        "counter_kernel": LAUNCH_PREFILL_ATTN,
        "counter_kernel_is_complete_family": False,
        "empty_initial_state": True,
        "prefix_reuse": False,
        "final_token_logits_only": True,
        "capacity": CAPACITY,
        "prefill_tokens": PREFILL_TOKENS,
        "chunk_count": (dispatch or {}).get("chunk_count", 1),
        "query_start": (dispatch or {}).get("query_start", 0),
        "decode_attention_substitution": decode_sub,
        "boundary": boundary,
        "identity": identity,
        "reason": None if boundary.get("ok") and not decode_sub else OPT140_INELIGIBLE,
    }


def _blocked_gpu_phase(
    phase: str, run_dir: Path, mode: str, preflight: Mapping[str, Any]
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "task": "OPT-140",
        "phase": phase,
        "mode": mode,
        "ok": False,
        "status": "incomplete",
        "blocked": True,
        "reason": preflight.get("blocked") or ["gpu_or_ncu_unavailable"],
        "gpu_available": preflight.get("gpu_available"),
        "ncu_available": (preflight.get("ncu") or {}).get("ncu_available"),
        "claims_throughput": False,
        "family_plan": family_plan(mode, phase if phase in PHASES else "report"),
        "measured_at": utc_now(),
    }
    return store_sidecar(run_dir, f"{phase}.json", payload)


def run_preflight(run_dir: Path, mode: str) -> dict[str, Any]:
    from tools import opt136_graph_accounting as opt136
    from tools import opt138_remaining_gap_profile as opt138
    from tools.opt080_batch_gate import IMAGE, docker_common

    run_dir.mkdir(parents=True, exist_ok=True)
    available, gpu_blocker = gpu_available()
    make_rc = 1
    make_stderr = gpu_blocker or ""
    make_stdout = ""
    if available:
        import subprocess

        make = subprocess.run(
            [*docker_common(IMAGE, "smoke"), "make", "cuda-opt140-diagnostics"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=CHILD_TIMEOUT_S,
        )
        make_rc = make.returncode
        make_stdout = make.stdout or ""
        make_stderr = make.stderr or ""
    ncu = (
        opt138._query_ncu_metrics()
        if available
        else {
            "ncu_available": False,
            "error": gpu_blocker or "gpu_unavailable",
            "selected_metrics": [],
            "full_ncu_sweep": False,
        }
    )
    identity = production_prefill_attn_identity()
    identity_ok = identity_match(
        phase="prefill",
        engine="quartz",
        workload="P4096",
        kernel=LAUNCH_PREFILL_ATTN,
        expected_kernel=LAUNCH_PREFILL_ATTN,
        replay_family="prompt-attention",
        replay_boundary=None,
    )
    rejected = identity_match(
        phase="prefill",
        engine="quartz",
        workload="P4096",
        kernel=LAUNCH_PREFILL_ATTN,
        expected_kernel=LAUNCH_PREFILL_ATTN,
        replay_family="decode-attention",
        replay_boundary=None,
    )
    ok = available and make_rc == 0 and identity_ok.get("ok") is True
    payload = {
        "schema_version": 1,
        "task": "OPT-140",
        "phase": "preflight",
        "mode": mode,
        "ok": ok,
        "blocked": [] if ok else [gpu_blocker or "preflight_failed"],
        "gpu_available": available,
        "gpu_blocker": gpu_blocker,
        "make_rc": make_rc,
        "make_stdout_tail": make_stdout[-800:],
        "make_stderr_tail": make_stderr[-800:],
        "replay_bin": relpath(ROOT / REPLAY_BIN)
        if (ROOT / REPLAY_BIN).is_file()
        else None,
        "replay_sha256": opt136.sha256_file(ROOT / REPLAY_BIN),
        "ncu": ncu,
        "identity": identity,
        "identity_ok": identity_ok,
        "decode_attention_rejected": rejected,
        "claims_throughput": False,
        "family_plan": family_plan(mode, "preflight"),
        "measured_at": utc_now(),
    }
    return store_sidecar(run_dir, "preflight.json", payload)


def replay_command(*, capture_key: str | None = None) -> list[str]:
    command = [
        f"./{REPLAY_BIN}",
        MODEL,
        "--workload",
        "prompt-attention",
        "--cache-mode",
        "rotating",
        "--opt140-protocol",
        "--warmups",
        "1",
        "--samples",
        "3",
        "--evidence-dir",
        relpath(EVIDENCE / "replay"),
    ]
    if capture_key:
        command.extend(["--capture-key", capture_key])
    return command


def run_replay(run_dir: Path, mode: str) -> dict[str, Any]:
    from tools import opt136_graph_accounting as opt136

    preflight = load_sidecar(run_dir, "preflight.json") or {}
    if not preflight.get("ok"):
        return _blocked_gpu_phase("replay", run_dir, mode, preflight)
    (EVIDENCE / "replay").mkdir(parents=True, exist_ok=True)
    command = replay_command()
    completed = opt136.with_gpu_lock(
        opt136.native_command(command, tier="acceptance"),
        timeout_s=PROMPT_ATTN_TIMEOUT_S,
    )
    blob = (completed.stdout or "") + (completed.stderr or "")
    raw_path = EVIDENCE / "replay" / "prompt-attention.stdout.txt"
    raw_path.write_text(blob, encoding="utf-8")
    parsed = opt136.parse_prefixed_json(completed.stdout or "", RESULT_PREFIX)
    rounds = parse_rounds(completed.stdout or "")
    timed = [row for row in rounds if not row.get("warmup")]
    warmups = [row for row in rounds if row.get("warmup")]
    parity = parse_parity_summary(completed.stdout or "")
    layers = parse_layer_parity(completed.stdout or "")
    dispatch = parse_dispatch(completed.stdout or "")
    identity = identity_match(
        phase="prefill",
        engine="quartz",
        workload="P4096",
        kernel=(dispatch or {}).get("launch") or LAUNCH_PREFILL_ATTN,
        expected_kernel=LAUNCH_PREFILL_ATTN,
        replay_family="prompt-attention",
        replay_boundary=None,
    )
    manifest = boundary_manifest(
        replay_family="prompt-attention",
        dispatch=dispatch,
        identity=identity,
    )
    dump_json(EVIDENCE / "boundary-manifest.json", manifest)
    ok = (
        completed.returncode == 0
        and bool(parsed)
        and (parity or {}).get("equal") is True
        and len(warmups) == 1
        and len(timed) == 3
        and all(row.get("cache_mode") == "rotating" for row in rounds)
        and all(row.get("family") == "prompt-attention" for row in rounds)
        and identity.get("ok") is True
        and manifest["boundary"].get("ok") is True
    )
    payload = {
        "schema_version": 1,
        "task": "OPT-140",
        "phase": "replay",
        "mode": mode,
        "ok": ok,
        "returncode": completed.returncode,
        "command": command,
        "parsed": parsed,
        "capture_key": capture_key_from(completed.stdout or ""),
        "parity": parity,
        "layer_parity": layers,
        "dispatch": dispatch,
        "rounds": rounds,
        "warmup_count": len(warmups),
        "sample_count": len(timed),
        "protocol": {
            "warmups": 1,
            "samples": 3,
            "cache_mode": "rotating",
            "accept_hot_as_production": False,
        },
        "boundary": manifest,
        "identity": identity,
        "raw_artifact": relpath(raw_path),
        "stdout_tail": (completed.stdout or "")[-2000:],
        "stderr_tail": (completed.stderr or "")[-800:],
        "claims_throughput": False,
        "family_plan": family_plan(mode, "replay"),
        "measured_at": utc_now(),
    }
    dump_json(EVIDENCE / "replay-evidence.json", payload)
    return store_sidecar(run_dir, "replay.json", payload)


def ncu_collect_command(
    *,
    metrics: Sequence[str],
    capture_key: str | None,
    raw_report: Path | None = None,
) -> list[str]:
    from tools import opt138_remaining_gap_profile as opt138

    command = [
        *opt138._docker_ncu(),
        "ncu",
        "--csv",
        "--metrics",
        ",".join(metrics),
        "--target-processes",
        "all",
        "--replay-mode",
        "application",
        "--kernel-name",
        f"regex:{ncu_kernel_regex(LAUNCH_PREFILL_ATTN)}",
        "--launch-count",
        "1",
    ]
    if raw_report is not None:
        output = raw_report
        if output.is_absolute():
            output = Path(relpath(output))
        command.extend(["-o", str(output), "--force-overwrite"])
    command.extend(
        [
            f"./{REPLAY_BIN}",
            MODEL,
            "--workload",
            "prompt-attention",
            "--cache-mode",
            "rotating",
            "--opt140-protocol",
            "--warmups",
            "0",
            "--samples",
            "1",
            "--evidence-dir",
            relpath(EVIDENCE / "ncu-replay"),
        ]
    )
    if capture_key:
        command.extend(["--capture-key", capture_key])
    return command


def run_counters(run_dir: Path, mode: str) -> dict[str, Any]:
    from tools import opt136_graph_accounting as opt136
    from tools import opt138_remaining_gap_profile as opt138

    preflight = load_sidecar(run_dir, "preflight.json") or {}
    replay = load_sidecar(run_dir, "replay.json") or {}
    if not preflight.get("ok"):
        return _blocked_gpu_phase("counters", run_dir, mode, preflight)
    ncu = preflight.get("ncu") or opt138._query_ncu_metrics()
    metrics = list(ncu.get("selected_metrics") or [])
    raw_dir = EVIDENCE / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (EVIDENCE / "ncu-replay").mkdir(parents=True, exist_ok=True)
    report_file = raw_dir / "p4096-attn_core.ncu-rep"
    raw_csv = raw_dir / "p4096-attn_core.ncu.csv"
    raw_txt = raw_dir / "p4096-attn_core.ncu.txt"
    capture_key = replay.get("capture_key")
    reused = False
    command: list[str] = []
    blob = ""
    returncode = 0
    if not ncu.get("ncu_available") or not metrics:
        record = {
            "kernel": "quartz:attn_core:P4096",
            "engine": "quartz",
            "family": "attn_core",
            "phase": "prefill",
            "replay_family": "prompt-attention",
            "expected_kernel": LAUNCH_PREFILL_ATTN,
            "error": ncu.get("error") or "ncu_unavailable",
            "supported_mechanism": None,
            "candidate": None,
            "zero_filled": False,
            "full_ncu_sweep": False,
        }
        payload = {
            "schema_version": 1,
            "task": "OPT-140",
            "phase": "counters",
            "mode": mode,
            "ok": False,
            "status": "incomplete",
            "kernels": [record],
            "ncu": ncu,
            "claims_throughput": False,
            "family_plan": family_plan(mode, "counters"),
            "measured_at": utc_now(),
        }
        dump_json(EVIDENCE / "counter-evidence.json", payload)
        return store_sidecar(run_dir, "counters.json", payload)
    if report_file.is_file() and report_file.stat().st_size > 0:
        reused = True
        command = ["reused", relpath(report_file)]
        if raw_txt.is_file():
            blob = raw_txt.read_text(encoding="utf-8", errors="replace")
    else:
        command = ncu_collect_command(
            metrics=metrics,
            capture_key=capture_key,
            raw_report=report_file,
        )
        completed = opt136.with_gpu_lock(command, timeout_s=PROMPT_ATTN_TIMEOUT_S)
        returncode = completed.returncode
        blob = (completed.stdout or "") + (completed.stderr or "")
        raw_txt.write_text(blob, encoding="utf-8")
    if report_file.is_file() and not (raw_csv.is_file() and raw_csv.stat().st_size > 0):
        raw_csv.write_text(export_ncu_csv(report_file), encoding="utf-8")
    parse_text = (
        raw_csv.read_text(encoding="utf-8", errors="replace")
        if raw_csv.is_file() and raw_csv.stat().st_size > 0
        else blob
    )
    record = counter_record_from_ncu_blob(
        parse_text,
        kernel_id="quartz:attn_core:P4096",
        engine="quartz",
        family="attn_core",
        phase="prefill",
        replay_family="prompt-attention",
        expected_kernel=LAUNCH_PREFILL_ATTN,
        workload="P4096",
        prefix=4096,
        selected_metrics=metrics,
        command=command,
        timeout_s=PROMPT_ATTN_TIMEOUT_S,
        raw_artifact=relpath(raw_csv if raw_csv.is_file() else raw_txt),
        binary_hash=(preflight.get("replay_sha256")),
    )
    record["ncu_report"] = relpath(report_file) if report_file.is_file() else None
    record["returncode"] = returncode
    record["reused"] = reused
    record["complete_family_replay"] = True
    record["counter_kernel_is_complete_family"] = False
    record["full_ncu_sweep"] = False
    record["stdout_tail"] = blob[-1500:]
    record["stderr_tail"] = ""
    ok = record.get("error") in (None, "") and (
        reused or returncode == 0 or report_file.is_file()
    )
    payload = {
        "schema_version": 1,
        "task": "OPT-140",
        "phase": "counters",
        "mode": mode,
        "ok": ok,
        "kernels": [record],
        "ncu": ncu,
        "command": command,
        "capture_key": capture_key,
        "claims_throughput": False,
        "family_plan": family_plan(mode, "counters"),
        "measured_at": utc_now(),
    }
    dump_json(EVIDENCE / "counter-evidence.json", payload)
    return store_sidecar(run_dir, "counters.json", payload)


def write_report(
    *,
    preflight: Mapping[str, Any],
    replay: Mapping[str, Any],
    counters: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> None:
    identity = production_prefill_attn_identity()
    parity = replay.get("parity") or {}
    dispatch = replay.get("dispatch") or {}
    kernel = (counters.get("kernels") or [{}])[0]
    lines = [
        "# OPT-140 — Production-boundary P4096 prefill attention replay",
        "",
        "Diagnostics only. No attention optimization. No new quality tolerance.",
        "Profiled NCU runs are not throughput samples.",
        "",
        f"status=`{payload.get('status')}` measured_at=`{payload.get('measurement_utc')}`",
        f"parent=`{PARENT}` selector=`{SELECTOR}`",
        f"replay_family=`prompt-attention` expected_kernel=`{LAUNCH_PREFILL_ATTN}`",
        f"existing_abs_tol=`{EXISTING_ABS_TOL}` new_quality_tolerance=`false`",
        "",
        "## Boundary",
        "",
        f"- tokens=`{PREFILL_TOKENS}` capacity=`{CAPACITY}` empty_state=`true` "
        "prefix_reuse=`false` final_token_logits_only=`true`",
        f"- chunk_count=`{dispatch.get('chunk_count')}` query_start=`{dispatch.get('query_start')}`",
        "- included: split, stage, BF16/F16 convert, core, merge, epilogue, fp32_to_bf16",
        "- excluded: mixer_mmq, ffn_mmq, logits, commit_sync, embedding",
        "- decode-attention substitution is rejected for prefill attn_core",
        f"- complete family replay vs counter kernel: family=`true` "
        f"kernel=`{LAUNCH_PREFILL_ATTN}`",
        "",
        "## Replay samples",
        "",
        f"- capture_key=`{replay.get('capture_key')}`",
        f"- warmup_count=`{replay.get('warmup_count')}` sample_count=`{replay.get('sample_count')}`",
        f"- parity equal=`{parity.get('equal')}` max_abs=`{parity.get('max_abs')}` "
        f"kv_max_abs=`{parity.get('kv_max_abs')}` nonfinite=`{parity.get('nonfinite')}`",
        f"- dispatch path=`{dispatch.get('path')}` launch=`{dispatch.get('launch')}`",
        f"- raw=`{replay.get('raw_artifact')}`",
        "",
    ]
    for row in replay.get("rounds") or []:
        kind = "warmup" if row.get("warmup") else "sample"
        lines.append(
            f"- {kind}[{row.get('sample_index')}] enclosing_ms=`{row.get('enclosing_ms')}` "
            f"kernel_only_ms=`{row.get('kernel_only_ms')}` cache=`{row.get('cache_mode')}`"
        )
    lines.extend(
        [
            "",
            "## Causal tails and chunk boundaries",
            "",
        ]
    )
    for row in replay.get("layer_parity") or []:
        lines.append(
            f"- layer=`{row.get('layer')}` slot=`{row.get('slot')}` "
            f"causal_tail_abs=`{row.get('causal_tail_abs')}` "
            f"chunk_start_abs=`{row.get('chunk_start_abs')}` "
            f"chunk_mid_abs=`{row.get('chunk_mid_abs')}` equal=`{row.get('equal')}`"
        )
    units = kernel.get("units") or {}
    lines.extend(
        [
            "",
            "## Structured counters",
            "",
            f"- target=`{kernel.get('target_kernel') or identity['expected_kernel']}`",
            f"- raw=`{kernel.get('raw_artifact')}` report=`{kernel.get('ncu_report')}`",
            f"- dram_read=`{kernel.get('dram_read_bytes')}` {units.get('dram_read_bytes') or ''}",
            f"- dram_write=`{kernel.get('dram_write_bytes')}`",
            f"- dram_throughput=`{kernel.get('dram_throughput')}`%",
            f"- l2=`{kernel.get('l2_traffic')}` {units.get('l2_traffic') or ''}",
            f"- sm=`{kernel.get('sm_throughput')}`%",
            f"- occupancy=`{kernel.get('achieved_occupancy')}`%",
            f"- tensor=`{kernel.get('tensor_activity')}`",
            f"- registers=`{kernel.get('registers')}` spills=`{kernel.get('local_memory_spills')}`",
            f"- mechanism=`{kernel.get('supported_mechanism')}`",
            f"- admission=`{(kernel.get('admission') or {}).get('reason')}`",
            f"- identity mismatches=`{(kernel.get('identity') or {}).get('mismatches')}`",
            "",
            "## Status",
            "",
            f"status=`{payload.get('status')}` blocked=`{payload.get('gpu_phases_blocked')}`.",
            "Throughput alone cannot establish a mechanism. Candidate stays null.",
            "No attention optimization shipped. Production selector unchanged.",
            "",
            f"Preflight make_rc=`{preflight.get('make_rc')}` "
            f"gpu=`{preflight.get('gpu_available')}` ncu=`{(preflight.get('ncu') or {}).get('ncu_available')}`.",
            "",
        ]
    )
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def run_report(run_dir: Path, mode: str) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    preflight = load_sidecar(run_dir, "preflight.json") or {}
    replay = load_sidecar(run_dir, "replay.json") or {}
    counters = load_sidecar(run_dir, "counters.json") or {}
    blocked = bool(preflight.get("blocked")) or not preflight.get("ok")
    replay_ok = bool(replay.get("ok"))
    counter_ok = bool(counters.get("ok"))
    if blocked:
        status = "incomplete"
    elif replay_ok:
        status = "measured"
    else:
        status = "incomplete"
    fixture = empty_fixture(mode)
    fixture.update(
        {
            "status": status,
            "measurement_utc": utc_now(),
            "preflight": preflight,
            "replay": replay,
            "counters": counters,
            "gpu_phases_blocked": blocked,
            "answers": {
                "matched_p4096_replay": replay_ok,
                "decode_attention_rejected": True,
                "existing_abs_tol": EXISTING_ABS_TOL,
                "new_quality_tolerance": False,
                "warmup_plus_three_rotating": (
                    replay.get("warmup_count") == 1 and replay.get("sample_count") == 3
                ),
                "bounded_ncu": counter_ok,
                "claims_throughput": False,
                "candidate": None,
                "supported_mechanism": (
                    ((counters.get("kernels") or [{}])[0].get("supported_mechanism"))
                    if counters.get("kernels")
                    else None
                ),
            },
        }
    )
    validate_fixture(fixture)
    dump_json(FIXTURE, fixture)
    write_report(
        preflight=preflight,
        replay=replay,
        counters=counters,
        payload=fixture,
    )
    payload = {
        "schema_version": 1,
        "task": "OPT-140",
        "phase": "report",
        "mode": mode,
        "ok": status == "measured",
        "status": status,
        "fixture": relpath(FIXTURE),
        "report_path": relpath(REPORT),
        "claims_throughput": False,
        "family_plan": family_plan(mode, "report"),
        "measured_at": utc_now(),
    }
    return store_sidecar(run_dir, "report.json", payload)


def run_phase(phase: str, run_dir: Path, mode: str) -> dict[str, Any]:
    started = time.time()
    if phase == "preflight":
        result = run_preflight(run_dir, mode)
    elif phase == "replay":
        result = run_replay(run_dir, mode)
    elif phase == "counters":
        result = run_counters(run_dir, mode)
    elif phase == "report":
        result = run_report(run_dir, mode)
    else:
        raise PrefillAttentionReplayError(f"unknown phase {phase}")
    elapsed = time.time() - started
    result = dict(result)
    result["phase_elapsed_s"] = elapsed
    if elapsed > AGGREGATE_DEADLINE_S:
        result["aggregate_deadline_exceeded"] = True
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--mode", default="feedback")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv)
    os.chdir(ROOT)
    result = run_phase(args.phase, Path(args.run_dir), args.mode)
    json.dump(
        {
            "phase": args.phase,
            "ok": bool(result.get("ok", True)),
            **({"status": result.get("status")} if result.get("status") else {}),
        },
        sys.stdout,
    )
    sys.stdout.write("\n")
    sys.stdout.write(
        RESULT_PREFIX
        + json.dumps({"phase": args.phase, "ok": bool(result.get("ok", True))})
        + "\n"
    )
    return 0 if result.get("ok", True) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PrefillAttentionReplayError as exc:
        sys.stderr.write(f"{exc}\n")
        raise SystemExit(1) from exc
