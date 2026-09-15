"""OPT-149 decode mixer RMSNorm fused into Q8_1 staging screen."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.opt075_q4_production_admission import (  # noqa: E402
    dump_json,
    load_json,
    parse_engine_pairs,
    utc_now,
)
from tools.opt080_batch_gate import (  # noqa: E402
    GGUF_SHA,
    IMAGE,
    docker_common,
    git_identity,
)
from tools.opt082_kernel_parity import gpu_available  # noqa: E402
from tools.opt110_llama_q4_adapter import _parse_quality_cases  # noqa: E402
from tools.opt136_graph_accounting import with_gpu_lock  # noqa: E402
from tools.performance_keep_policy import (  # noqa: E402
    evaluate,
    freeze_hash,
    validate_opt_in_contract,
)
from tools.quality.quality_mode import build_quality_config  # noqa: E402
from tools.run_optimization_task import (  # noqa: E402
    loop_product,
    validate_future_keep_policy,
    workload_for_mode,
)

CONTRACT = ROOT / "pins/opt149_norm_q8_contract.json"
ITERATION = ROOT / "pins/opt149_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt149_norm_q8.json"
REPORT = ROOT / "evidence/optimization/opt149-norm-q8/REPORT.md"
EVIDENCE = REPORT.parent
PIN_PATH = ROOT / "cuda/opt149_norm_q8.cuh"
PROMPT_PIN = ROOT / "cuda/fattn_mma_f16.cuh"
GRAPH_PIN = ROOT / "cuda/execution_graph_path.cuh"
OPT116_FIXTURE = ROOT / "fixtures/opt116_generated_quality.json"
MEMORY = ROOT / "fixtures/cuda_memory_fit_post_graph.json"
QUALITY_NATIVE = "build/qw38-cuda-opt058-quality-baseline-test"
CHECKPOINT_NATIVE = "build/qw38-cuda-checkpoint-test"
MEMORY_NATIVE = "build/qw38-cuda-memory-fit-test"
NATIVE = "build/qw38-cuda-opt149-norm-q8-test"
PROBE = "build/qw38-cuda-optimization-engine-probe"
NLL_BUNDLE = "pins/production_quality_v2_nll.bundle"
MODEL = "models/Qwen3.8-27B-Q4_K_M.gguf"
RESULT_PREFIX = "QW38_OPT149_NORM_Q8_RESULT="
COUNTS_PREFIX = "QW38_OPT149_NATIVE_COUNTS="
PHASES = (
    "correctness",
    "screen",
    "quality",
    "state-memory",
    "performance",
    "report",
)
PARENT_STACK = "post124_plus_opt127_decode_segments8_plus_opt137_mma_plus_opt147_8x8_plus_opt148_flash_vec"
SHIPPING_PROMPT = "prefill_attention_8x8_v1"
SHIPPING_DECODE = "decode_attention_flash_vec_v1"
CANDIDATE = "norm_to_q8_1_screen_v1"
CANDIDATE_LAUNCH = "rms_norm_fp32_to_q8_1"
CONTROL = "current_bf16_q8_staging"
D128_KERNEL = "warp_query"
DECODE_PIN = ROOT / "cuda/attention_decode_path.cuh"
WARMUPS = 3
SCREEN_WARMUPS = 1
SCREEN_PAIRS = 3
PAIR_COUNT = 10
DECODE_TOKENS = 256
SCREEN_ENGINE_TOKENS = 32
CHILD_TIMEOUT_S = 300.0
LONG_TIMEOUT_S = 3600.0
AGGREGATE_DEADLINE_S = 7200.0
MMA_THRESHOLD = 8192
CROSSOVER = 1024
IDENTITY_POSITIONS = (0, 1023, 1024, 2047, 2048, 4096, 8192)
COMBINED_QUALITY = {
    "q4_decode": "llama_q4k_mmvq",
    "q4_staging": "paired_integer",
    "ffn_decode": "paired_integer",
    "q8_decode": "r1_w4",
    "q8_path": "dp4a_q8_1",
    "prompt_mmq": "fma_async_x",
    "prompt_mmq_tile": "i128_j128",
    "prompt_attention": SHIPPING_PROMPT,
    "decode_gdn": "sequential",
    "decode_attention": SHIPPING_DECODE,
    "decode_norm_q8": CONTROL,
    "prompt_pair": "off",
    "nvccflags": "-O2 --fmad=false",
    "execution_graphs": "decode_segments8",
    "chat_template": "no_thinking",
    "enable_thinking": False,
    "logit_masking": False,
}
REQUIRED_FIXTURE_KEYS = (
    "schema_version",
    "task",
    "mode",
    "llama_revision",
    "gguf_sha256",
    "parent",
    "candidate",
    "quality",
    "state_memory",
    "performance",
    "shipping_decode_attention",
    "shipping_decode_norm_q8",
    "production_kept",
    "verdict",
    "report_path",
)
SCREEN_COMPLETE_RE = re.compile(
    r"screen_complete tokens=2048 layers=(\d+) control_mean_ms=([0-9.eE+-]+) "
    r"candidate_mean_ms=([0-9.eE+-]+) saving_ms=([0-9.eE+-]+) faster=(true|false) "
    r"screened_in=(true|false) pairs=(\d+) warmups=(\d+)"
)
ROUND_RE = re.compile(
    r"round family=decode-mixer cache_mode=rotating warmup=(true|false) "
    r"sample_index=(\d+).*control_ms=([0-9.eE+-]+) candidate_ms=([0-9.eE+-]+)"
)
LAYER_RE = re.compile(
    r"layer_kind=(\S+) residual=(\S+) control_path=(\S+) candidate_path=(\S+) "
    r"launch=(\S+) q8_vs_host=(\d+) q8_vs_control=(\d+) fp64_abs=([0-9.eE+-]+) "
    r"equal=(true|false) finite=(true|false)"
)


class NormQ8Error(RuntimeError):
    """OPT-149 measurement or policy failure."""


def load_contract() -> dict[str, Any]:
    payload = load_json(CONTRACT)
    if payload.get("task") != "OPT-149":
        raise NormQ8Error("short-decode flash contract task mismatch")
    validate_opt_in_contract(payload)
    return payload


def load_iteration() -> dict[str, Any]:
    payload = load_json(ITERATION)
    validate_future_keep_policy("OPT-149", payload)
    return payload


def family_plan(mode: str, family: str) -> str:
    iteration = load_iteration()
    alias = {
        "identity": "correctness",
        "quality-long": "quality",
        "d2048": "performance",
        "d128": "performance",
        "d8192": "performance",
        "d32768": "performance",
        "p4096": "performance",
    }
    key = alias.get(family, family)
    workload = workload_for_mode(iteration["workloads"][key], mode)
    product = loop_product(workload)
    return (
        f"task=OPT-149 mode={mode} phase={key} "
        f"loop_product={product} cases={workload.get('cases')} "
        f"candidates={workload.get('candidates')} "
        f"warmups={workload.get('warmups')} samples={workload.get('samples')} "
        f"tokens={workload.get('tokens')}"
    )


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


def workspace_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def production_pins_parent() -> bool:
    decode = DECODE_PIN.read_text(encoding="utf-8")
    fusion = PIN_PATH.read_text(encoding="utf-8")
    graphs = GRAPH_PIN.read_text(encoding="utf-8")
    prompt = PROMPT_PIN.read_text(encoding="utf-8")
    return (
        'kSelectedExecutionGraphPath[] = "decode_segments8"' in graphs
        and f'kSelectedAttentionPipelinePath[] = "{SHIPPING_PROMPT}"' in prompt
        and "kSelectedDecodeAttentionCrossoverThreshold = 1024" in decode
        and "kSelectedDecodeAttentionVerifiedMax = 4096" in decode
        and "kSelectedVec128NParts = 16" in decode
        and "kSelectedOpt137DenseMma = true" in decode
        and "kOpt137MmaThreshold = 8192" in decode
        and "kSelectedDecodeAttentionFlashVec = true" in decode
        and "kSelectedDecodeNormQ81Fusion = false" in fusion
    )


def production_pin() -> str:
    match = re.search(
        r"kSelectedDecodeNormQ81Fusion = (true|false)",
        PIN_PATH.read_text(encoding="utf-8"),
    )
    if match is None:
        raise NormQ8Error("missing kSelectedDecodeNormQ81Fusion")
    return CANDIDATE if match.group(1) == "true" else CONTROL


def apply_production_pin(path: str) -> None:
    if path not in {CONTROL, CANDIDATE}:
        raise NormQ8Error(f"illegal decode-norm-q8 pin {path}")
    enabled = "true" if path == CANDIDATE else "false"
    text = PIN_PATH.read_text(encoding="utf-8")
    updated = re.sub(
        r"kSelectedDecodeNormQ81Fusion = (true|false)",
        f"kSelectedDecodeNormQ81Fusion = {enabled}",
        text,
        count=1,
    )
    PIN_PATH.write_text(updated, encoding="utf-8")


def nll_from_cases(cases: Sequence[Mapping[str, Any]], name: str) -> float | None:
    for row in cases:
        if row.get("name") == name and row.get("mean_nll") is not None:
            return float(row["mean_nll"])
    return None


def ppl_ratio(candidate_nll: float, control_nll: float) -> float:
    return math.exp(float(candidate_nll) - float(control_nll))


def parse_prefixed(text: str, prefix: str) -> dict[str, Any]:
    for line in (text or "").splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            return json.loads(stripped[len(prefix) :])
    return {}


def run_locked(command: Sequence[str], *, tier: str, timeout_s: float) -> Any:
    os.environ["QW38_CUDA_TEST_TIER"] = tier
    listed = list(command)
    if os.environ.get("QW38_HOST_NATIVE") == "1":
        cmd = listed
    else:
        cmd = [*docker_common(IMAGE, tier), *listed]
    return with_gpu_lock(cmd, timeout_s=timeout_s)


def ensure_built(run_dir: Path) -> dict[str, Any]:
    available, gpu_blocker = gpu_available()
    make_rc = 1
    make_stderr = gpu_blocker or "gpu_unavailable"
    if available:
        make = run_locked(
            ["make", "cuda-opt149-diagnostics"],
            tier="correctness",
            timeout_s=LONG_TIMEOUT_S,
        )
        make_rc = make.returncode
        make_stderr = ((make.stderr or "") + (make.stdout or ""))[-4000:]
        (run_dir / "make-diagnostics.txt").write_text(
            (make.stdout or "") + (make.stderr or ""), encoding="utf-8"
        )
    return {
        "gpu_available": available,
        "gpu_blocker": gpu_blocker,
        "make_returncode": make_rc,
        "make_stderr_tail": make_stderr,
        "ok": available and make_rc == 0 and production_pins_parent(),
    }


def skip_payload(
    run_dir: Path,
    mode: str,
    phase: str,
    *,
    reason: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "task": "OPT-149",
        "phase": phase,
        "mode": mode,
        "ok": True,
        "skipped": True,
        "skip_reason": reason,
        "candidate": CANDIDATE,
        "nll_required": False,
        "opt058_invoked": False,
        "candidate_nll_measured": False,
        "candidate_nll_not_measured": False,
        "family_plan": family_plan(mode, phase),
        "measured_at": utc_now(),
        "mechanism": "unknown",
        "supported_mechanism": None,
    }
    if extra:
        payload.update(dict(extra))
    return store_sidecar(run_dir, f"{phase}.json", payload)


def screened_in(run_dir: Path) -> bool | None:
    screen = load_sidecar(run_dir, "screen.json")
    if screen is None:
        return None
    if screen.get("skipped"):
        return False
    return bool(screen.get("screened_in")) and bool(screen.get("ok"))


def parse_screen(stdout: str) -> dict[str, Any]:
    match = SCREEN_COMPLETE_RE.search(stdout or "")
    rounds: list[dict[str, Any]] = []
    for row in ROUND_RE.finditer(stdout or ""):
        if row.group(1) == "true":
            continue
        rounds.append(
            {
                "warmup": False,
                "sample_index": int(row.group(2)),
                "control_ms": float(row.group(3)),
                "candidate_ms": float(row.group(4)),
            }
        )
    control = [item["control_ms"] for item in rounds]
    candidate = [item["candidate_ms"] for item in rounds]
    recon_control = sum(control) / len(control) if control else None
    recon_candidate = sum(candidate) / len(candidate) if candidate else None
    parsed = {
        "rounds": rounds,
        "reconstructed_control_mean_ms": recon_control,
        "reconstructed_candidate_mean_ms": recon_candidate,
        "reconstructed_saving_ms": (
            recon_control - recon_candidate
            if recon_control is not None and recon_candidate is not None
            else None
        ),
        "reconstructed_faster": (
            recon_candidate < recon_control
            if recon_control is not None and recon_candidate is not None
            else False
        ),
    }
    if match is None:
        parsed["ok"] = False
        parsed["reason"] = "screen_complete_line_missing"
        return parsed
    parsed.update(
        {
            "layers": int(match.group(1)),
            "control_mean_ms": float(match.group(2)),
            "candidate_mean_ms": float(match.group(3)),
            "saving_ms": float(match.group(4)),
            "faster": match.group(5) == "true",
            "screened_in": match.group(6) == "true",
            "pairs": int(match.group(7)),
            "warmups": int(match.group(8)),
            "ok": True,
        }
    )
    return parsed


def parse_correctness(stdout: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for match in LAYER_RE.finditer(stdout or ""):
        rows.append(
            {
                "layer_kind": match.group(1),
                "residual": match.group(2) == "true",
                "control_path": match.group(3),
                "candidate_path": match.group(4),
                "launch": match.group(5),
                "q8_vs_host": int(match.group(6)),
                "q8_vs_control": int(match.group(7)),
                "fp64_abs": float(match.group(8)),
                "equal": match.group(9) == "true",
                "finite": match.group(10) == "true",
            }
        )
    mixed = (
        "mixed_consumers" in (stdout or "")
        and "pass=true"
        in (stdout or "").split("mixed_consumers", 1)[-1].split("\n", 1)[0]
    )
    stale = (
        "stale_buffer" in (stdout or "")
        and "pass=true" in (stdout or "").split("stale_buffer", 1)[-1].split("\n", 1)[0]
    )
    graph = "graph_eager equal=true" in (stdout or "") or (
        "graph_eager_repeat pass=true" in (stdout or "")
    )
    q4 = "q4_decode=llama_q4k_mmvq" in (stdout or "") or (
        "q4_decode=llama_q4k_mmvq" in (stdout or "")
    )
    opt110 = "opt110_revived=false" in (stdout or "")
    byte_claim = "byte_size_only_claim=false" in (stdout or "")
    equal = all(row["equal"] and row["q8_vs_control"] == 0 for row in rows)
    finite = all(row["finite"] for row in rows) if rows else False
    kinds = {(row["layer_kind"], row["residual"]) for row in rows}
    return {
        "rows": rows,
        "kinds": sorted(
            f"{kind}:{'residual' if residual else 'rms'}" for kind, residual in kinds
        ),
        "required_kinds": [
            "gdn:rms",
            "attention:rms",
            "gdn:residual",
            "attention:residual",
        ],
        "kinds_ok": kinds
        >= {("gdn", False), ("attention", False), ("gdn", True), ("attention", True)},
        "equal": equal,
        "finite": finite,
        "mixed_consumers_retain": mixed,
        "stale_buffer": stale,
        "graph_eager": graph,
        "q4_llama_q4k_mmvq": q4,
        "opt110_revived": not opt110,
        "byte_size_only_claim": not byte_claim,
        "launch_candidate": any(row["launch"] == CANDIDATE_LAUNCH for row in rows)
        or any("residual_add_norm_fp32_to_q8_1" in row["launch"] for row in rows),
    }


def run_native(
    run_dir: Path, *, workload: str, tier: str, timeout_s: float
) -> dict[str, Any]:
    completed = run_locked(
        [
            f"./{NATIVE}",
            "--workload",
            workload,
            "--warmups",
            str(SCREEN_WARMUPS if workload == "screen" else 0),
            "--samples",
            str(SCREEN_PAIRS if workload == "screen" else 1),
        ],
        tier=tier,
        timeout_s=timeout_s,
    )
    stdout = (completed.stdout or "") + (completed.stderr or "")
    (run_dir / f"{workload}-native.txt").write_text(stdout, encoding="utf-8")
    result = parse_prefixed(stdout, RESULT_PREFIX)
    counts = parse_prefixed(stdout, COUNTS_PREFIX)
    return {
        "returncode": completed.returncode,
        "stdout_tail": stdout[-8000:],
        "result": result,
        "counts": counts,
        "pass": completed.returncode == 0 and bool(result.get("pass", False)),
        "raw": workspace_relative(run_dir / f"{workload}-native.txt"),
    }


def run_correctness(run_dir: Path, mode: str) -> dict[str, Any]:
    print(family_plan(mode, "correctness"), flush=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    source, dirty = git_identity()
    built = ensure_built(run_dir)
    if not built["ok"]:
        payload = {
            "schema_version": 1,
            "task": "OPT-149",
            "phase": "correctness",
            "mode": mode,
            "ok": False,
            "blocked": True,
            "blockers": [
                built.get("gpu_blocker") or f"make_rc={built['make_returncode']}"
            ],
            "build": built,
            "production_pins_parent": production_pins_parent(),
            "candidate": CANDIDATE,
            "mechanism": "unknown",
            "family_plan": family_plan(mode, "correctness"),
            "measured_at": utc_now(),
            "source": source,
            "dirty": bool(dirty),
        }
        return store_sidecar(run_dir, "correctness.json", payload)
    identity = run_native(
        run_dir, workload="correctness", tier="correctness", timeout_s=CHILD_TIMEOUT_S
    )
    state = run_native(
        run_dir, workload="state", tier="correctness", timeout_s=CHILD_TIMEOUT_S
    )
    parsed = parse_correctness(
        (run_dir / "correctness-native.txt").read_text(encoding="utf-8")
        + (run_dir / "state-native.txt").read_text(encoding="utf-8")
    )
    ok = (
        bool(identity["pass"])
        and bool(state["pass"])
        and parsed["kinds_ok"]
        and parsed["equal"]
        and parsed["finite"]
        and parsed["mixed_consumers_retain"]
        and parsed["stale_buffer"]
        and parsed["graph_eager"]
        and parsed["q4_llama_q4k_mmvq"]
        and not parsed["opt110_revived"]
        and not parsed["byte_size_only_claim"]
        and production_pins_parent()
    )
    payload = {
        "schema_version": 1,
        "task": "OPT-149",
        "phase": "correctness",
        "mode": mode,
        "ok": ok,
        "build": built,
        "identity": identity,
        "state": state,
        "parsed": parsed,
        "identity_positions": list(IDENTITY_POSITIONS),
        "candidate": CANDIDATE,
        "control": CONTROL,
        "production_pin": production_pin(),
        "production_pins_parent": production_pins_parent(),
        "mechanism": "unknown",
        "supported_mechanism": None,
        "sass_observations": False,
        "source_observations": False,
        "opt110_revived": False,
        "byte_size_only_claim": False,
        "family_plan": family_plan(mode, "correctness"),
        "measured_at": utc_now(),
        "source": source,
        "dirty": bool(dirty),
        "gpu_lock": "build/optimization-runs/qw38-gpu.lock",
        "child_timeout_s": CHILD_TIMEOUT_S,
        "aggregate_deadline_s": AGGREGATE_DEADLINE_S,
    }
    return store_sidecar(run_dir, "correctness.json", payload)


def run_engine_ab(
    run_dir: Path,
    *,
    name: str,
    prompt: int | None = None,
    prefix: int | None = None,
    output_tokens: int | None = None,
    pairs: int = PAIR_COUNT,
    decode_norm_q8: bool = True,
    tier: str = "acceptance",
) -> dict[str, Any]:
    command = [
        f"./{PROBE}",
        MODEL,
        "--workload",
        "attn-ab",
        "--pairs",
        str(pairs),
        "--modes",
        "graph",
        "--attention-pipeline-control",
        SHIPPING_PROMPT,
        "--attention-pipeline",
        SHIPPING_PROMPT,
    ]
    if decode_norm_q8:
        command.extend(["--decode-norm-q8", CANDIDATE])
    if prompt is not None:
        command.extend(["--prompt", str(prompt)])
    if prefix is not None:
        command.extend(["--prefix", str(prefix)])
    if output_tokens is not None:
        command.extend(["--output-tokens", str(output_tokens)])
    completed = run_locked(command, tier=tier, timeout_s=LONG_TIMEOUT_S)
    stdout = (completed.stdout or "") + (completed.stderr or "")
    (run_dir / f"{name}-engine.txt").write_text(stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise NormQ8Error(f"{name} engine probe failed: {stdout[-2000:]}")
    if "override_before_capture_applied=true" not in stdout:
        raise NormQ8Error(f"{name} graph capture override was not applied")
    parsed_pairs = parse_engine_pairs(stdout)
    if len(parsed_pairs) != pairs:
        raise NormQ8Error(f"{name} engine pairs {len(parsed_pairs)} != {pairs}")
    return {
        "name": name,
        "pairs": parsed_pairs,
        "command": command,
        "raw": workspace_relative(run_dir / f"{name}-engine.txt"),
        "returncode": completed.returncode,
    }


def run_screen(run_dir: Path, mode: str) -> dict[str, Any]:
    print(family_plan(mode, "screen"), flush=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    built = ensure_built(run_dir)
    correctness = load_sidecar(run_dir, "correctness.json") or {}
    if not built["ok"] or not correctness.get("ok"):
        payload = {
            "schema_version": 1,
            "task": "OPT-149",
            "phase": "screen",
            "mode": mode,
            "ok": False,
            "screened_in": False,
            "screened_out": True,
            "advance": False,
            "reason": "correctness_failed"
            if correctness.get("ok")
            else "build_or_correctness_failed",
            "build": built,
            "candidate": CANDIDATE,
            "mechanism": "unknown",
            "family_plan": family_plan(mode, "screen"),
            "measured_at": utc_now(),
        }
        if not built["ok"]:
            payload["blocked"] = True
            payload["reason"] = "build_failed"
            payload["screened_out"] = False
        return store_sidecar(run_dir, "screen.json", payload)
    native = run_native(
        run_dir, workload="screen", tier="screen", timeout_s=LONG_TIMEOUT_S
    )
    stdout = (run_dir / "screen-native.txt").read_text(encoding="utf-8")
    parsed = parse_screen(stdout)
    engine: dict[str, Any] | None = None
    engine_error: str | None = None
    if native.get("pass"):
        try:
            engine = run_engine_ab(
                run_dir,
                name="d2048-screen",
                prefix=2048,
                output_tokens=SCREEN_ENGINE_TOKENS,
                pairs=1,
                decode_norm_q8=True,
                tier="screen",
            )
        except NormQ8Error as exc:
            engine_error = str(exc)
    reconstruction = {
        "schema_version": 1,
        "task": "OPT-149",
        "workload": "d2048",
        "complete_family": True,
        "rotating_layers": 64,
        "warmups": SCREEN_WARMUPS,
        "pairs": SCREEN_PAIRS,
        "control": CONTROL,
        "candidate": CANDIDATE,
        "native_control_mean_ms": parsed.get("control_mean_ms"),
        "native_candidate_mean_ms": parsed.get("candidate_mean_ms"),
        "independent_control_mean_ms": parsed.get("reconstructed_control_mean_ms"),
        "independent_candidate_mean_ms": parsed.get("reconstructed_candidate_mean_ms"),
        "independent_saving_ms": parsed.get("reconstructed_saving_ms"),
        "rounds": parsed.get("rounds"),
        "short_engine_pair": (engine or {}).get("pairs") if engine else None,
        "mechanism": "unknown",
        "prep_included": True,
        "combine_included": True,
        "conversion_included": True,
    }
    dump_json(EVIDENCE / "d2048-pair-reconstruction.json", reconstruction)
    dump_json(run_dir / "d2048-pair-reconstruction.json", reconstruction)
    faster = bool(parsed.get("faster")) and bool(parsed.get("reconstructed_faster"))
    numerical_ok = bool(native.get("pass"))
    engine_ok = engine is not None and engine_error is None
    blocked = numerical_ok and not engine_ok
    screened = numerical_ok and faster and engine_ok
    payload = {
        "schema_version": 1,
        "task": "OPT-149",
        "phase": "screen",
        "mode": mode,
        "ok": numerical_ok and engine_ok,
        "blocked": blocked,
        "screened_in": screened,
        "screened_out": (not screened) and numerical_ok and engine_ok,
        "advance": screened,
        "warmups": SCREEN_WARMUPS,
        "pairs": SCREEN_PAIRS,
        "complete_mixer_family": True,
        "short_engine_pair": True,
        "cache_mode": "rotating",
        "target": "d2048",
        "native": native,
        "parsed": parsed,
        "engine": {"raw": engine["raw"]} if engine else None,
        "engine_error": engine_error,
        "reconstruction": reconstruction,
        "candidate": CANDIDATE,
        "control": CONTROL,
        "mechanism": "unknown",
        "supported_mechanism": None,
        "screen_allowed_without_sass": True,
        "family_plan": family_plan(mode, "screen"),
        "measured_at": utc_now(),
        "reason": (
            None
            if screened
            else (
                "engine_pair_failed"
                if blocked
                else (
                    "correctness_or_launch_failed"
                    if not numerical_ok
                    else "complete_family_time_did_not_improve"
                )
            )
        ),
    }
    return store_sidecar(run_dir, "screen.json", payload)


def run_quality_native(run_dir: Path, config_id: str) -> dict[str, Any]:
    sidecar_path = run_dir / f"quality-{config_id}.txt"
    if sidecar_path.is_file() and "held_out" in sidecar_path.read_text(
        encoding="utf-8"
    ):
        text = sidecar_path.read_text(encoding="utf-8")
        cases = _parse_quality_cases(text)
        if cases:
            return {
                "id": config_id,
                "cases": cases,
                "restored_packed_or_r2": False,
                "quality_flag": "--quality",
                "sidecar": workspace_relative(sidecar_path),
                "opt058_invoked": True,
                "reused_sidecar": True,
            }
    selectors = dict(COMBINED_QUALITY)
    selectors["decode_attention"] = SHIPPING_DECODE
    selectors["decode_norm_q8"] = config_id
    selectors["prompt_attention"] = SHIPPING_PROMPT
    cfg_path = run_dir / f"quality-config-{config_id}.json"
    dump_json(cfg_path, build_quality_config(enabled=True, selectors=selectors))
    extra = [
        "--quality",
        "--quality-config",
        workspace_relative(cfg_path),
        "--q4-decode",
        "llama_q4k_mmvq",
        "--ffn-decode",
        "paired_integer",
        "--q8-layout",
        "r1_w4",
        "--attention-pipeline",
        SHIPPING_PROMPT,
        "--bundle",
        NLL_BUNDLE,
    ]
    command = [
        f"./{QUALITY_NATIVE}",
        MODEL,
        "--workload",
        "quality-baseline",
        *extra,
    ]
    completed = run_locked(command, tier="acceptance", timeout_s=LONG_TIMEOUT_S)
    sidecar_path.write_text(
        (completed.stdout or "") + (completed.stderr or ""), encoding="utf-8"
    )
    if completed.returncode != 0:
        raise NormQ8Error(
            f"OPT-058 quality failed for {config_id}: "
            f"{(completed.stderr or '')[-2000:]}"
        )
    stdout = completed.stdout or ""
    cases = _parse_quality_cases(stdout)
    restored = "restored_packed_or_r2=false" in stdout
    if not restored:
        raise NormQ8Error("--quality restored packed or r2 defaults")
    if not cases:
        raise NormQ8Error(f"OPT-058 produced no NLL cases for {config_id}")
    return {
        "id": config_id,
        "cases": cases,
        "restored_packed_or_r2": False,
        "quality_flag": "--quality",
        "sidecar": workspace_relative(sidecar_path),
        "opt058_invoked": True,
        "opt058_argv": " ".join(command),
        "quality_config": workspace_relative(cfg_path),
    }


def run_quality(run_dir: Path, mode: str) -> dict[str, Any]:
    print(family_plan(mode, "quality"), flush=True)
    if screened_in(run_dir) is False:
        return skip_payload(
            run_dir,
            mode,
            "quality",
            reason="screened_out_retain_parent",
            extra={
                "opt116_contract_id": "opt116_generated_v1",
                "ppl_ratio_max": 1.01,
                "recurrence_incremental_nll_max": 0.02,
                "arithmetic_changed": False,
            },
        )
    control = run_quality_native(run_dir, CONTROL)
    candidate = run_quality_native(run_dir, CANDIDATE)
    candidate["same_math_copied_from_control"] = False
    control_held = nll_from_cases(control["cases"], "held_out_wikitext_1024")
    cand_held = nll_from_cases(candidate["cases"], "held_out_wikitext_1024")
    control_wiki = nll_from_cases(control["cases"], "wikitext_nll")
    cand_wiki = nll_from_cases(candidate["cases"], "wikitext_nll")
    incomplete = control_held is None or cand_held is None
    held_ratio = (
        ppl_ratio(cand_held, control_held)
        if control_held is not None and cand_held is not None
        else None
    )
    wiki_ratio = (
        ppl_ratio(cand_wiki, control_wiki)
        if control_wiki is not None and cand_wiki is not None
        else None
    )
    opt116 = load_json(OPT116_FIXTURE) if OPT116_FIXTURE.is_file() else {}
    ppl = (
        ((opt116.get("authenticate") or {}).get("contracts") or {}).get("contracts")
        or {}
    ).get("opt116_generated_v1", {})
    parent_ppl = float(((ppl.get("ppl") or {}).get("aggregate_ratio")) or 1.0)
    quality_ok = (
        not incomplete
        and held_ratio is not None
        and held_ratio <= 1.01
        and (wiki_ratio is None or wiki_ratio <= 1.01)
    )
    payload = {
        "schema_version": 1,
        "task": "OPT-149",
        "phase": "quality",
        "mode": mode,
        "ok": quality_ok,
        "skipped": False,
        "opt058_invoked": True,
        "candidate_nll_measured": not incomplete,
        "candidate_nll_not_measured": False,
        "nll_required": True,
        "opt116_contract_id": "opt116_generated_v1",
        "ppl_ratio": held_ratio,
        "wikitext_ppl_ratio": wiki_ratio,
        "ppl_ratio_max": 1.01,
        "recurrence_incremental_nll_max": 0.02,
        "parent_authenticated_ppl_ratio": parent_ppl,
        "control": control,
        "candidate": candidate,
        "quality_flag": "--quality",
        "arithmetic_changed": True,
        "family_plan": family_plan(mode, "quality"),
        "measured_at": utc_now(),
    }
    return store_sidecar(run_dir, "quality.json", payload)


def run_state_memory(run_dir: Path, mode: str) -> dict[str, Any]:
    print(family_plan(mode, "state-memory"), flush=True)
    if screened_in(run_dir) is False:
        memory = load_json(MEMORY) if MEMORY.is_file() else {}
        return skip_payload(
            run_dir,
            mode,
            "state-memory",
            reason="screened_out_retain_parent",
            extra={
                "memory_fit_parent_admitted": bool(memory.get("post_graph_admitted")),
                "128k_fit_with_graphs": bool(memory.get("post_graph_admitted")),
                "checkpoint_ok": True,
                "checkpoint_skipped": True,
                "checkpoint_native": CHECKPOINT_NATIVE,
            },
        )
    state = run_native(
        run_dir, workload="state", tier="acceptance", timeout_s=CHILD_TIMEOUT_S
    )
    checkpoint_path = str(run_dir / "opt149-checkpoint.bin")
    checkpoint = run_locked(
        [
            f"./{CHECKPOINT_NATIVE}",
            MODEL,
            checkpoint_path,
            "--attention-pipeline",
            SHIPPING_PROMPT,
            "--decode-attention",
            SHIPPING_DECODE,
            "--decode-norm-q8",
            CANDIDATE,
        ],
        tier="acceptance",
        timeout_s=LONG_TIMEOUT_S,
    )
    (run_dir / "checkpoint.txt").write_text(
        (checkpoint.stdout or "") + (checkpoint.stderr or ""), encoding="utf-8"
    )
    memory = run_locked(
        [
            f"./{MEMORY_NATIVE}",
            MODEL,
            "--attention-pipeline",
            SHIPPING_PROMPT,
            "--decode-attention",
            SHIPPING_DECODE,
            "--decode-norm-q8",
            CANDIDATE,
        ],
        tier="acceptance",
        timeout_s=LONG_TIMEOUT_S,
    )
    (run_dir / "memory-fit.txt").write_text(
        (memory.stdout or "") + (memory.stderr or ""), encoding="utf-8"
    )
    checkpoint_ok = checkpoint.returncode == 0 and "passed=true" in (
        checkpoint.stdout or ""
    )
    memory_stdout = memory.stdout or ""
    memory_ok = memory.returncode == 0 and "passed=true" in memory_stdout
    fit_line = next(
        (line for line in memory_stdout.splitlines() if line.startswith("memory_fit=")),
        "",
    )
    fit_fields = dict(field.split("=", 1) for field in fit_line.split() if "=" in field)
    ok = bool(state.get("pass")) and checkpoint_ok and memory_ok
    payload = {
        "schema_version": 1,
        "task": "OPT-149",
        "phase": "state-memory",
        "mode": mode,
        "ok": ok,
        "skipped": False,
        "state": {"state": ok},
        "native_state": state,
        "checkpoint_ok": checkpoint_ok,
        "checkpoint_native": CHECKPOINT_NATIVE,
        "memory_fit_ok": memory_ok,
        "128k_fit_with_graphs": memory_ok,
        "capacity": int(fit_fields.get("capacity") or 131072),
        "free_bytes": int(fit_fields["free_bytes"])
        if "free_bytes" in fit_fields
        else None,
        "reserve_required": int(fit_fields["reserve_required"])
        if "reserve_required" in fit_fields
        else None,
        "arithmetic": fit_fields.get("arithmetic"),
        "candidate": CANDIDATE,
        "family_plan": family_plan(mode, "state-memory"),
        "measured_at": utc_now(),
        "stdout_tail": {
            "checkpoint": (checkpoint.stdout or "")[-2000:],
            "memory": (memory.stdout or "")[-2000:],
        },
    }
    return store_sidecar(run_dir, "state-memory.json", payload)


def pair_identity(metric: str, *, prefix: int, eval_count: int) -> dict[str, Any]:
    return {
        "prefix": prefix,
        "eval_count": eval_count,
        "capacity": 131072,
        "metric_boundary": metric,
        "sampling_policy": "greedy",
        "output_policy": "fixed_eval",
        "engine_config": "quartz_graph",
        "input_trajectory": "opt149_norm_q8",
    }


def engine_pairs_to_records(
    pairs: Sequence[Mapping[str, Any]],
    *,
    workload: str,
    metric: str,
    prefix: int,
    eval_count: int,
    tokens: float,
) -> list[dict[str, Any]]:
    identity = pair_identity(metric, prefix=prefix, eval_count=eval_count)
    rows: list[dict[str, Any]] = []
    for index, pair in enumerate(pairs):
        control_ms = float(pair.get("control_ms") or 0.0)
        candidate_ms = float(pair.get("candidate_ms") or 0.0)
        if metric == "decode_only":
            control_itl = float(pair.get("control_itl_p50_ms") or 0.0)
            candidate_itl = float(pair.get("candidate_itl_p50_ms") or 0.0)
            control_rate = 1000.0 / control_itl if control_itl > 0 else 0.0
            candidate_rate = 1000.0 / candidate_itl if candidate_itl > 0 else 0.0
        else:
            control_rate = tokens / (control_ms / 1000.0) if control_ms > 0 else 0.0
            candidate_rate = (
                tokens / (candidate_ms / 1000.0) if candidate_ms > 0 else 0.0
            )
        rows.append(
            {
                "sample_id": pair.get("sample_index", index),
                "order": pair.get("order") or ("AB" if index % 2 == 0 else "BA"),
                "workload": workload,
                "metric": metric,
                "control_rate": control_rate,
                "candidate_rate": candidate_rate,
                "control_ms": control_ms,
                "candidate_ms": candidate_ms,
                "control_p95_itl": pair.get("control_itl_p95_ms"),
                "candidate_p95_itl": pair.get("candidate_itl_p95_ms"),
                "identity": dict(identity),
            }
        )
    return rows


def run_performance(run_dir: Path, mode: str) -> dict[str, Any]:
    print(family_plan(mode, "performance"), flush=True)
    if screened_in(run_dir) is False:
        return skip_payload(
            run_dir,
            mode,
            "performance",
            reason="screened_out_retain_parent",
            extra={
                "pairs": [],
                "warmups": WARMUPS,
                "aa_pairs": PAIR_COUNT,
                "target": "d2048.decode_only",
                "keep_policy_id": "target_guard_v2",
            },
        )
    records: list[dict[str, Any]] = []
    probes: dict[str, Any] = {}
    p4096 = run_engine_ab(run_dir, name="p4096", prompt=4096, decode_norm_q8=False)
    probes["p4096"] = p4096
    records.extend(
        engine_pairs_to_records(
            p4096["pairs"],
            workload="p4096",
            metric="prefill",
            prefix=0,
            eval_count=4096,
            tokens=4096.0,
        )
    )
    for name, prefix in (
        ("d128", 128),
        ("d2048", 2048),
        ("d8192", 8192),
        ("d32768", 32768),
    ):
        probe = run_engine_ab(
            run_dir, name=name, prefix=prefix, output_tokens=DECODE_TOKENS
        )
        probes[name] = probe
        records.extend(
            engine_pairs_to_records(
                probe["pairs"],
                workload=name,
                metric="decode_only",
                prefix=prefix,
                eval_count=DECODE_TOKENS,
                tokens=float(DECODE_TOKENS),
            )
        )
        records.extend(
            engine_pairs_to_records(
                probe["pairs"],
                workload=name,
                metric="complete_request",
                prefix=prefix,
                eval_count=DECODE_TOKENS,
                tokens=float(prefix + DECODE_TOKENS),
            )
        )
    payload = {
        "schema_version": 1,
        "task": "OPT-149",
        "phase": "performance",
        "mode": mode,
        "ok": True,
        "skipped": False,
        "pairs": records,
        "probes": {key: {"raw": value["raw"]} for key, value in probes.items()},
        "warmups": WARMUPS,
        "aa_pairs": PAIR_COUNT,
        "target": "d2048.decode_only",
        "keep_policy_id": "target_guard_v2",
        "decode_output_tokens": DECODE_TOKENS,
        "candidate": CANDIDATE,
        "family_plan": family_plan(mode, "performance"),
        "measured_at": utc_now(),
    }
    return store_sidecar(run_dir, "performance.json", payload)


def evaluate_keep(run_dir: Path) -> dict[str, Any]:
    contract = load_contract()
    correctness = load_sidecar(run_dir, "correctness.json") or {}
    screen = load_sidecar(run_dir, "screen.json") or {}
    quality = load_sidecar(run_dir, "quality.json") or {}
    state = load_sidecar(run_dir, "state-memory.json") or {}
    performance = load_sidecar(run_dir, "performance.json") or {}
    if correctness.get("blocked") or screen.get("blocked"):
        return {
            "verdict": "blocked",
            "production_kept": False,
            "shipping_prompt_attention": SHIPPING_PROMPT,
            "shipping_decode_attention": SHIPPING_DECODE,
            "shipping_decode_norm_q8": CONTROL,
            "reasons": list(
                correctness.get("blockers") or [str(screen.get("reason") or "blocked")]
            ),
            "performance": {},
            "tok_s_deltas": {},
            "claims_throughput": False,
            "candidate_measured_delta": None,
            "shipping_delta": 0,
            "quality_result": "not_required",
        }
    if not correctness.get("ok"):
        return {
            "verdict": "reject",
            "production_kept": False,
            "shipping_prompt_attention": SHIPPING_PROMPT,
            "shipping_decode_attention": SHIPPING_DECODE,
            "shipping_decode_norm_q8": CONTROL,
            "reasons": ["identity_or_same_math_failed"],
            "performance": {},
            "tok_s_deltas": {},
            "claims_throughput": False,
            "candidate_measured_delta": None,
            "shipping_delta": 0,
            "quality_result": "not_required",
        }
    if screen.get("screened_out") or (
        screen.get("ok") and not screen.get("screened_in")
    ):
        return {
            "verdict": "screened_out",
            "production_kept": False,
            "shipping_prompt_attention": SHIPPING_PROMPT,
            "shipping_decode_attention": SHIPPING_DECODE,
            "shipping_decode_norm_q8": CONTROL,
            "reasons": [
                str(screen.get("reason") or "complete_family_time_did_not_improve")
            ],
            "performance": {},
            "tok_s_deltas": {},
            "claims_throughput": False,
            "candidate_measured_delta": screen.get("parsed"),
            "shipping_delta": 0,
            "quality_result": "nll_not_required_screened_out",
            "screen": screen,
        }
    if screen.get("screened_in") and not quality:
        return {
            "verdict": "incomplete",
            "production_kept": False,
            "shipping_prompt_attention": SHIPPING_PROMPT,
            "shipping_decode_attention": SHIPPING_DECODE,
            "shipping_decode_norm_q8": CONTROL,
            "reasons": ["missing_required_quality_not_no_opportunity"],
            "performance": {},
            "tok_s_deltas": {},
            "claims_throughput": False,
            "candidate_measured_delta": screen.get("parsed"),
            "shipping_delta": 0,
            "quality_result": None,
        }
    if screen.get("screened_in") and quality.get("candidate_nll_not_measured"):
        return {
            "verdict": "incomplete",
            "production_kept": False,
            "shipping_prompt_attention": SHIPPING_PROMPT,
            "shipping_decode_attention": SHIPPING_DECODE,
            "shipping_decode_norm_q8": CONTROL,
            "reasons": ["missing_required_quality_not_no_opportunity"],
            "shipping_delta": 0,
            "quality_result": None,
        }
    records = {
        "pairs": list(performance.get("pairs") or []),
        "quality": {"quality": bool(quality.get("ok"))},
        "state": {"state": bool(state.get("ok"))},
    }
    policy = evaluate(contract, records)
    verdict = str(policy.get("verdict") or "incomplete")
    reasons: list[str] = []
    if not bool(quality.get("ok")) or quality.get("skipped"):
        reasons.append("quality_failed")
        if verdict not in {"incomplete", "reject"}:
            verdict = "reject"
    if not bool(state.get("ok")) or state.get("skipped"):
        reasons.append("state_memory_failed")
        if verdict not in {"incomplete", "reject"}:
            verdict = "reject"
    reasons.append("mechanism_unknown_causal_claim_withheld")
    reasons.extend(str(item) for item in (policy.get("reasons") or []))
    keep = verdict == "keep"
    return {
        "verdict": verdict,
        "production_kept": keep,
        "shipping_prompt_attention": SHIPPING_PROMPT,
        "shipping_decode_attention": SHIPPING_DECODE,
        "shipping_decode_norm_q8": CANDIDATE if keep else CONTROL,
        "reasons": reasons,
        "performance": policy.get("metrics") or {},
        "tok_s_deltas": policy.get("candidate_rates") or {},
        "keep_policy": policy,
        "contract_hash": freeze_hash(contract),
        "quality": quality,
        "state_memory": state,
        "screen": screen,
        "claims_throughput": keep,
        "candidate_measured_delta": policy.get("candidate_rates") or {},
        "shipping_delta": policy.get("candidate_rates") if keep else 0,
        "quality_result": "measured" if quality.get("candidate_nll_measured") else None,
        "mechanism": "unknown",
    }


def maybe_flip_production_pins(keep: bool) -> None:
    if keep:
        apply_production_pin(CANDIDATE)
        return
    if production_pin() != CONTROL:
        apply_production_pin(CONTROL)


def write_report(result: Mapping[str, Any], run_dir: Path) -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    verdict = str(result.get("verdict"))
    quality = result.get("quality") or load_sidecar(run_dir, "quality.json") or {}
    screen = result.get("screen") or load_sidecar(run_dir, "screen.json") or {}
    parsed = screen.get("parsed") or {}
    shipping = result.get("shipping_decode_attention")
    shipping_norm = result.get("shipping_decode_norm_q8")
    kept = bool(result.get("production_kept"))
    quality_candidate = quality.get("candidate")
    held = nll_from_cases(
        quality_candidate.get("cases")
        if isinstance(quality_candidate, Mapping)
        else [],
        "held_out_wikitext_1024",
    )
    pin_line = (
        f"Production fusion pin flipped to `{CANDIDATE}`."
        if kept
        else f"Parent retained; shipping decode-norm-q8 stays `{CONTROL}`."
    )
    lines = [
        "# OPT-149 — Decode mixer RMSNorm fused into Q8_1 staging screen",
        "",
        f"Status: **{verdict}**. Shipping decode attention `{shipping}`. "
        f"Shipping decode-norm-q8 `{shipping_norm}`. {pin_line}",
        "",
        f"Parent `{PARENT_STACK}` with kept OPT-147 prompt `{SHIPPING_PROMPT}` "
        f"and kept OPT-148 decode `{SHIPPING_DECODE}`. Control is "
        f"`{CONTROL}` (BF16 RMSNorm plus `launch_quantize_bf16_q8_1`). "
        f"Candidate `{CANDIDATE}` fuses eligible mixer input_norm into "
        "`quartz_q8_1_sum_q` with in-register BF16 rounding. FFN "
        "`llama_q4k_mmvq`, Q8Block, and logits BF16 stay unfused. OPT-110 "
        "is not revived. Q8 byte size alone is not a win.",
        "",
        "## Candidate",
        "",
        f"Candidate ID: `{CANDIDATE}`. Launch `{CANDIDATE_LAUNCH}` / "
        "`residual_add_norm_fp32_to_q8_1`. Selected group `mixer_input_norm_q8_1`.",
        "Supported mechanism: `unknown`. Screen admission does not require SASS.",
        "Missing required work is incomplete/blocked, never `no_opportunity`.",
        "",
        "## Correctness",
        "",
        "Compared fused Q8_1 versus control GPU staging for GDN and attention "
        "layers, with and without residual add, mixed-consumer retain, "
        "stale-buffer rejection, graph/eager equality, and sampled FP64 refs. "
        "Identity gate is GPU control versus candidate Q8 bytes.",
        "",
        f"Correctness sidecar ok=`{(load_sidecar(run_dir, 'correctness.json') or {}).get('ok')}`.",
        "",
        "## Screen (D2048 complete decode-mixer family)",
        "",
        f"Warmups `{SCREEN_WARMUPS}`, alternating pairs `{SCREEN_PAIRS}`, "
        "64 mixer layers, residual+norm+Q8+projections included, plus "
        "one short engine pair (prefix 2048, 32 output tokens).",
        f"Control mean `{parsed.get('control_mean_ms')}` ms. Candidate mean "
        f"`{parsed.get('candidate_mean_ms')}` ms. Saving "
        f"`{parsed.get('saving_ms')}` ms. Screened in: "
        f"`{screen.get('screened_in')}`. Reason: `{screen.get('reason')}`.",
        "",
        "Independent reconstruction of one D2048 pair set: "
        "[`d2048-pair-reconstruction.json`](d2048-pair-reconstruction.json).",
        "",
        "## Quality",
        "",
        (
            "OPT-058 invoked=`"
            + str(quality.get("opt058_invoked"))
            + "`; candidate NLL measured=`"
            + str(quality.get("candidate_nll_measured"))
            + "`; NLL required=`"
            + str(quality.get("nll_required"))
            + "`; skip_reason=`"
            + str(quality.get("skip_reason"))
            + "`. Screened-in survivors must measure candidate NLL because "
            "this is an arithmetic/output-encoding change. Screened-out retains "
            "parent and does not stub `candidate_nll_not_measured`."
        ),
        "",
        f"Held-out NLL `{held}`. ppl_ratio=`{quality.get('ppl_ratio')}`.",
        "",
        "## Keep policy",
        "",
        "`target_guard_v2` target `d2048.decode_only` with D2048 "
        "`complete_request` as the target additional guard. Guards: D128/"
        "D8192/D32768 `decode_only` and `complete_request` plus P4096 prefill. "
        "Mechanism remains unknown and is not a substitute for those gates.",
        "",
        f"Verdict `{verdict}`. production_kept=`{result.get('production_kept')}`.",
        f"claims_throughput: `{bool(result.get('claims_throughput'))}`.",
        "",
        "## Deltas",
        "",
        f"Candidate measured delta: `{result.get('candidate_measured_delta')}`.",
        f"Shipping delta: `{result.get('shipping_delta')}` "
        "(zero on screened_out/reject; parent retained).",
        f"Quality result: `{result.get('quality_result')}`.",
        "",
        "## Performance evidence checklist",
        "",
        "1. **Measurement identity** — Quartz production `decode_segments8` + kept "
        f"OPT-137 MMA + kept OPT-147 `{SHIPPING_PROMPT}` + kept OPT-148 "
        f"`{SHIPPING_DECODE}`; parent decode-norm-q8 `{CONTROL}`; candidate "
        f"launch `{CANDIDATE_LAUNCH}`; shipping after this task "
        f"`{shipping_norm}`; GGUF `{GGUF_SHA}`; llama revision "
        "`cc83d7b4824f73cfdda4dfbb47ee39804f71b328`.",
        "2. **Coverage** — native complete decode-mixer family screen at D2048 "
        "with rotating layers plus one short engine pair; correctness over "
        "GDN/attention, residual, mixed consumers, stale buffers, graph/eager.",
        "3. **Time accounting** — screen means are complete-family enclosing times "
        "including residual, RMSNorm, Q8 staging and four mixer projections, "
        "not Q8 byte-size-only or leaf-only.",
        "4. **Contradiction register** — fused Q8_1 vs BF16+restage is the "
        "OPT-112 hypothesis under test, not a freeze blocker. OPT-110 is not "
        "revived. Mechanism stays unknown without source/SASS evidence.",
        "5. **Claim types** — mechanism `unknown`; screen `measured`; keep only after "
        "quality/state/target_guard_v2.",
        "6. **Target/guard** — OPT-135 `target_guard_v2` opted in for screened-in "
        "survivors.",
        "7. **Independent verification** — D2048 pair reconstruction recomputes "
        "means from raw alternating rounds.",
        "8. **Reporting** — candidate measured delta and shipping delta are "
        "separate; shipping delta is 0 unless `kSelectedDecodeNormQ81Fusion` "
        "flips on keep.",
        "",
        "## Raw gates",
        "",
        "Sidecars: [`raw/`](raw/). Reconstruction: "
        "[`d2048-pair-reconstruction.json`](d2048-pair-reconstruction.json). "
        "Fixture: `fixtures/opt149_norm_q8.json`.",
        "",
        "## Status",
        "",
        f"verdict=`{verdict}` production_kept=`{result.get('production_kept')}` "
        f"blocked=`{verdict == 'blocked'}`.",
        f"Production fusion pin `{production_pin()}`. Decode attention "
        f"`{SHIPPING_DECODE}`. Prompt pin `{SHIPPING_PROMPT}`.",
        "",
    ]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_raw(run_dir: Path) -> None:
    raw = EVIDENCE / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    for path in sorted(run_dir.glob("*.json")):
        shutil.copy2(path, raw / path.name)
    for path in sorted(run_dir.glob("*.txt")):
        shutil.copy2(path, raw / path.name)


def write_fixture(run_dir: Path, result: Mapping[str, Any], mode: str) -> None:
    quality = load_sidecar(run_dir, "quality.json") or {}
    payload = {
        "schema_version": 1,
        "task": "OPT-149",
        "mode": mode,
        "llama_revision": "cc83d7b4824f73cfdda4dfbb47ee39804f71b328",
        "gguf_sha256": GGUF_SHA,
        "parent": PARENT_STACK,
        "parent_prompt_attention": SHIPPING_PROMPT,
        "candidate": CANDIDATE,
        "quality": quality,
        "state_memory": load_sidecar(run_dir, "state-memory.json") or {},
        "performance": load_sidecar(run_dir, "performance.json")
        or result.get("performance"),
        "shipping_prompt_attention": result.get("shipping_prompt_attention"),
        "shipping_decode_attention": result.get("shipping_decode_attention"),
        "shipping_decode_norm_q8": result.get("shipping_decode_norm_q8"),
        "production_kept": bool(result.get("production_kept")),
        "verdict": result.get("verdict"),
        "report_path": str(REPORT.relative_to(ROOT)),
        "tok_s_deltas": result.get("tok_s_deltas"),
        "candidate_measured_delta": result.get("candidate_measured_delta"),
        "shipping_delta": result.get("shipping_delta"),
        "quality_result": result.get("quality_result"),
        "reasons": result.get("reasons"),
        "correctness": load_sidecar(run_dir, "correctness.json") or {},
        "screen": load_sidecar(run_dir, "screen.json") or {},
        "keep_policy_id": "target_guard_v2",
        "claims_throughput": bool(result.get("claims_throughput")),
        "claims_performance_improvement": bool(result.get("production_kept")),
        "mechanism": "unknown",
        "opt110_revived": False,
        "byte_size_only_claim": False,
    }
    dump_json(FIXTURE, payload)
    dump_json(EVIDENCE / "answers.json", payload)


def validate_fixture(payload: Mapping[str, Any]) -> dict[str, Any]:
    missing = [key for key in REQUIRED_FIXTURE_KEYS if key not in payload]
    return {
        "ok": not missing and payload.get("task") == "OPT-149",
        "task": "OPT-149",
        "missing": missing,
    }


def run_report(run_dir: Path, mode: str) -> dict[str, Any]:
    result = evaluate_keep(run_dir)
    result["quality"] = load_sidecar(run_dir, "quality.json") or {}
    maybe_flip_production_pins(bool(result.get("production_kept")))
    write_report(result, run_dir)
    write_fixture(run_dir, result, mode)
    copy_raw(run_dir)
    payload = {
        "schema_version": 1,
        "task": "OPT-149",
        "phase": "report",
        "mode": mode,
        "ok": True,
        "verdict": result.get("verdict"),
        "production_kept": result.get("production_kept"),
        "shipping_decode_attention": result.get("shipping_decode_attention"),
        "shipping_decode_norm_q8": result.get("shipping_decode_norm_q8"),
        "tok_s_deltas": result.get("tok_s_deltas"),
        "candidate_measured_delta": result.get("candidate_measured_delta"),
        "shipping_delta": result.get("shipping_delta"),
        "quality_result": result.get("quality_result"),
        "family_plan": family_plan(mode, "report"),
        "measured_at": utc_now(),
        "production_pin": production_pin(),
    }
    stored = store_sidecar(run_dir, "report.json", payload)
    copy_raw(run_dir)
    return stored


def run_phase(run_dir: Path, mode: str, phase: str) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    if phase == "correctness":
        return run_correctness(run_dir, mode)
    if phase == "screen":
        return run_screen(run_dir, mode)
    if phase == "quality":
        return run_quality(run_dir, mode)
    if phase == "state-memory":
        return run_state_memory(run_dir, mode)
    if phase == "performance":
        return run_performance(run_dir, mode)
    if phase == "report":
        return run_report(run_dir, mode)
    raise NormQ8Error(f"unknown phase {phase}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--mode", choices=("feedback", "acceptance", "release"))
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    mode = args.mode or "feedback"
    try:
        payload = run_phase(args.run_dir, mode, args.phase)
    except NormQ8Error as exc:
        print(
            json.dumps(
                {"task": "OPT-149", "phase": args.phase, "ok": False, "error": str(exc)}
            )
        )
        return 1
    print(
        json.dumps(
            {
                "task": "OPT-149",
                "phase": args.phase,
                "ok": bool(payload.get("ok", True)),
                "verdict": payload.get("verdict"),
                "screened_in": payload.get("screened_in"),
            }
        )
    )
    return 0 if payload.get("ok", True) else 1


if __name__ == "__main__":
    sys.exit(main())
