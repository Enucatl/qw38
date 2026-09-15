"""OPT-152 dispatch-only vector coverage screen of kept OPT-148 flash-vec."""

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

CONTRACT = ROOT / "pins/opt152_vector_coverage_contract.json"
ITERATION = ROOT / "pins/opt152_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt152_vector_coverage.json"
REPORT = ROOT / "evidence/optimization/opt152-vector-coverage/REPORT.md"
EVIDENCE = REPORT.parent
PIN_PATH = ROOT / "cuda/attention_decode_path.cuh"
PROMPT_PIN = ROOT / "cuda/fattn_mma_f16.cuh"
GRAPH_PIN = ROOT / "cuda/execution_graph_path.cuh"
OPT116_FIXTURE = ROOT / "fixtures/opt116_generated_quality.json"
MEMORY = ROOT / "fixtures/cuda_memory_fit_post_graph.json"
QUALITY_NATIVE = "build/qw38-cuda-opt058-quality-baseline-test"
CHECKPOINT_NATIVE = "build/qw38-cuda-checkpoint-test"
MEMORY_NATIVE = "build/qw38-cuda-memory-fit-test"
NATIVE = "build/qw38-cuda-opt152-vector-coverage-test"
PROBE = "build/qw38-cuda-optimization-engine-probe"
NLL_BUNDLE = "pins/production_quality_v2_nll.bundle"
MODEL = "models/Qwen3.8-27B-Q4_K_M.gguf"
RESULT_PREFIX = "QW38_OPT152_VECTOR_COVERAGE_RESULT="
COUNTS_PREFIX = "QW38_OPT152_NATIVE_COUNTS="
PHASES = (
    "correctness",
    "screen",
    "quality",
    "state-memory",
    "performance",
    "report",
)
PARENT_STACK = "post148_flash_vec_parent_plus_opt137_mma"
SHIPPING_PROMPT = "prefill_attention_8x8_v1"
SHIPPING_DECODE = "decode_attention_flash_vec_v1"
GAP_ONLY = "flash_vec_gap_only_v1"
ALL_SHORT = "flash_vec_all_short_v1"
CANDIDATES = (GAP_ONLY, ALL_SHORT)
PARENT_COVERAGE = "parent"
WARMUPS = 3
SCREEN_WARMUPS = 1
SCREEN_PAIRS = 3
PAIR_COUNT = 10
DECODE_TOKENS = 256
CHILD_TIMEOUT_S = 300.0
LONG_TIMEOUT_S = 3600.0
AGGREGATE_DEADLINE_S = 7200.0
MMA_THRESHOLD = 8192
CROSSOVER = 1024
VERIFIED_MAX = 4096
IDENTITY_POSITIONS = (
    0,
    1,
    127,
    128,
    511,
    1023,
    1024,
    4095,
    4096,
    4097,
    6144,
    8190,
    8191,
    8192,
)
SCREEN_SHAPES = (128, 512, 6144)
MATERIAL = 1.02
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
    "production_kept",
    "verdict",
    "report_path",
)
SCREEN_SHAPE_RE = re.compile(
    r"screen_shape tokens=(\d+) layers=(\d+) coverage=(\S+) parent_path=(\S+) "
    r"candidate_path=(\S+) control_mean_ms=([0-9.eE+-]+) "
    r"candidate_mean_ms=([0-9.eE+-]+) saving_ms=([0-9.eE+-]+) "
    r"faster=(true|false) pairs=(\d+) warmups=(\d+)"
)
ROUND_RE = re.compile(
    r"round family=decode-attn cache_mode=rotating warmup=(true|false) "
    r"sample_index=(\d+).*control_ms=([0-9.eE+-]+) candidate_ms=([0-9.eE+-]+)"
    r".*path=(\S+) launch=\S+ tokens=(\d+)"
)
LENGTH_RE = re.compile(
    r"^(length|smoke|quality_region) tokens=(\d+) start=0 position=(\d+) "
    r"parent_path=(\S+) candidate_path=(\S+) launch=(\S*) host_expect=(\S+) "
    r"topology=(\d+) max_abs=([0-9.eE+-]+) fp64_abs=([0-9.eE+-]+) "
    r"nonfinite=(\d+) equal=(true|false) finite=(true|false).*n_parts=(\d+).* "
    r"coverage=(\S+) llama_padded=(\d+) llama_kernel=(\S+)",
    re.MULTILINE,
)
GRAPH_RE = re.compile(
    r"graph_replay coverage=(\S+) capture=(\d+) replay=(\d+) "
    r"topology_capture=(\d+) topology_replay=(\d+) crossed=(true|false) "
    r"same_topology=(true|false) live_device_position=true eager_launch=(\S+) "
    r"max_abs=([0-9.eE+-]+) nonfinite=(\d+) equal=(true|false) "
    r"host_selector_only=false pass=(true|false)"
)


class VectorCoverageError(RuntimeError):
    """OPT-152 measurement or policy failure."""


def load_contract() -> dict[str, Any]:
    payload = load_json(CONTRACT)
    if payload.get("task") != "OPT-152":
        raise VectorCoverageError("vector coverage contract task mismatch")
    validate_opt_in_contract(payload)
    return payload


def load_iteration() -> dict[str, Any]:
    payload = load_json(ITERATION)
    validate_future_keep_policy("OPT-152", payload)
    return payload


def family_plan(mode: str, family: str) -> str:
    iteration = load_iteration()
    alias = {
        "identity": "correctness",
        "quality-long": "quality",
        "d6144": "performance",
        "d512": "performance",
        "d128": "performance",
        "d2048": "performance",
        "d8192": "performance",
        "d32768": "performance",
        "p4096": "performance",
    }
    key = alias.get(family, family)
    workload = workload_for_mode(iteration["workloads"][key], mode)
    product = loop_product(workload)
    return (
        f"task=OPT-152 mode={mode} phase={key} "
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
    decode = PIN_PATH.read_text(encoding="utf-8")
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
        and "kSelectedOpt151QkPvMma = false" in decode
        and 'kSelectedDecodeAttentionFlashVecCoverage[] = "parent"' in decode
    )


def production_coverage_pin() -> str:
    match = re.search(
        r'kSelectedDecodeAttentionFlashVecCoverage\[\] = "([^"]+)"',
        PIN_PATH.read_text(encoding="utf-8"),
    )
    if match is None:
        raise VectorCoverageError("missing coverage pin")
    return match.group(1)


def apply_production_coverage(coverage: str) -> None:
    if coverage not in {PARENT_COVERAGE, "gap_only", "all_short"}:
        raise VectorCoverageError(f"illegal coverage pin {coverage}")
    text = PIN_PATH.read_text(encoding="utf-8")
    updated = re.sub(
        r'kSelectedDecodeAttentionFlashVecCoverage\[\] = "[^"]+"',
        f'kSelectedDecodeAttentionFlashVecCoverage[] = "{coverage}"',
        text,
        count=1,
    )
    PIN_PATH.write_text(updated, encoding="utf-8")


def pin_name_for_ident(ident: str) -> str:
    if ident == GAP_ONLY:
        return "gap_only"
    if ident == ALL_SHORT:
        return "all_short"
    return PARENT_COVERAGE


def nll_from_cases(cases: Sequence[Mapping[str, Any]], name: str) -> float | None:
    for row in cases:
        if row.get("name") == name and row.get("mean_nll") is not None:
            return float(row["mean_nll"])
    return None


def ppl_ratio(candidate_nll: float, control_nll: float) -> float:
    return math.exp(float(candidate_nll) - float(control_nll))


def parse_prefixed(text: str, prefix: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped.startswith(prefix):
            continue
        try:
            candidate = json.loads(stripped[len(prefix) :])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            parsed = candidate
    return parsed


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
            ["make", "cuda-opt152-diagnostics"],
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
        "task": "OPT-152",
        "phase": phase,
        "mode": mode,
        "ok": True,
        "skipped": True,
        "skip_reason": reason,
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


def survivor_of(run_dir: Path) -> str | None:
    screen = load_sidecar(run_dir, "screen.json")
    if screen is None:
        return None
    if screen.get("skipped"):
        return None
    value = screen.get("survivor")
    return str(value) if value else None


def screened_in(run_dir: Path) -> bool | None:
    screen = load_sidecar(run_dir, "screen.json")
    if screen is None:
        return None
    if screen.get("skipped"):
        return False
    return bool(screen.get("screened_in")) and bool(screen.get("ok"))


def parse_correctness(stdout: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for match in LENGTH_RE.finditer(stdout or ""):
        rows.append(
            {
                "label": match.group(1),
                "tokens": int(match.group(2)),
                "position": int(match.group(3)),
                "parent_path": match.group(4),
                "candidate_path": match.group(5),
                "launch": match.group(6),
                "host_expect": match.group(7),
                "topology": int(match.group(8)),
                "max_abs": float(match.group(9)),
                "fp64_abs": float(match.group(10)),
                "nonfinite": int(match.group(11)),
                "equal": match.group(12) == "true",
                "finite": match.group(13) == "true",
                "n_parts": int(match.group(14)),
                "coverage": match.group(15),
                "llama_padded": int(match.group(16)),
                "llama_kernel": match.group(17),
            }
        )
    graphs: list[dict[str, Any]] = []
    for match in GRAPH_RE.finditer(stdout or ""):
        graphs.append(
            {
                "coverage": match.group(1),
                "capture": int(match.group(2)),
                "replay": int(match.group(3)),
                "topology_capture": int(match.group(4)),
                "topology_replay": int(match.group(5)),
                "crossed": match.group(6) == "true",
                "same_topology": match.group(7) == "true",
                "eager_launch": match.group(8),
                "max_abs": float(match.group(9)),
                "nonfinite": int(match.group(10)),
                "equal": match.group(11) == "true",
                "pass": match.group(12) == "true",
                "live_device_position": True,
                "host_selector_only": False,
            }
        )
    length_rows = [row for row in rows if row["label"] == "length"]
    positions = sorted({row["position"] for row in length_rows})
    mma_ok = all(
        row["candidate_path"] == "dense_bf16_tile_f16_mma_decode_v1"
        for row in length_rows
        if row["position"] >= MMA_THRESHOLD
    )
    graph_crossings = {
        (1024, 4096, 8192)[i]: any(
            item["replay"] == value and item.get("pass")
            for item in graphs
            for value in (1024, 4096, 8192)
        )
        for i in range(3)
    }
    return {
        "rows": rows,
        "graphs": graphs,
        "positions": positions,
        "required_positions": list(IDENTITY_POSITIONS),
        "positions_ok": positions == list(IDENTITY_POSITIONS),
        "equal": all(row["equal"] for row in length_rows) if length_rows else False,
        "finite": all(row["finite"] for row in length_rows) if length_rows else False,
        "opt137_mma": mma_ok,
        "graph_pass": bool(graphs) and all(item["pass"] for item in graphs),
        "graph_live_device": bool(graphs)
        and all(item["live_device_position"] for item in graphs),
        "graph_crossings": graph_crossings,
        "prompt_to_decode_handoff": "prompt_to_decode_kv_handoff pass=true"
        in (stdout or ""),
        "graph_eager": "graph_eager_repeat pass=true" in (stdout or ""),
        "launch_recorded": all(row["launch"] for row in length_rows)
        if length_rows
        else False,
        "verified_max_untouched": "verified_max=4096" in (stdout or ""),
        "n_parts_16": all(
            row["n_parts"] == 16
            for row in length_rows
            if row["position"] < MMA_THRESHOLD
        )
        if length_rows
        else False,
    }


def parse_screen(stdout: str) -> dict[str, Any]:
    shapes: dict[str, dict[int, dict[str, Any]]] = {GAP_ONLY: {}, ALL_SHORT: {}}
    for match in SCREEN_SHAPE_RE.finditer(stdout or ""):
        coverage = match.group(3)
        tokens = int(match.group(1))
        row = {
            "tokens": tokens,
            "layers": int(match.group(2)),
            "coverage": coverage,
            "parent_path": match.group(4),
            "candidate_path": match.group(5),
            "control_mean_ms": float(match.group(6)),
            "candidate_mean_ms": float(match.group(7)),
            "saving_ms": float(match.group(8)),
            "faster": match.group(9) == "true",
            "pairs": int(match.group(10)),
            "warmups": int(match.group(11)),
        }
        shapes.setdefault(coverage, {})[tokens] = row
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
                "path": row.group(5),
                "tokens": int(row.group(6)),
            }
        )
    return {"shapes": shapes, "rounds": rounds, "ok": bool(shapes[GAP_ONLY])}


def apply_choice_rule(parsed: Mapping[str, Any]) -> dict[str, Any]:
    shapes = parsed.get("shapes") or {}
    gap = shapes.get(GAP_ONLY) or {}
    all_short = shapes.get(ALL_SHORT) or {}

    def mean_row(
        block: Mapping[int, Mapping[str, Any]], tokens: int
    ) -> Mapping[str, Any]:
        return block.get(tokens) or {}

    def improved(block: Mapping[int, Mapping[str, Any]], tokens: int) -> bool:
        row = mean_row(block, tokens)
        control = row.get("control_mean_ms")
        candidate = row.get("candidate_mean_ms")
        return (
            control is not None
            and candidate is not None
            and float(candidate) < float(control)
        )

    def d128_ok(block: Mapping[int, Mapping[str, Any]]) -> bool:
        row = mean_row(block, 128)
        control = row.get("control_mean_ms")
        candidate = row.get("candidate_mean_ms")
        if control is None or candidate is None or float(control) <= 0:
            return False
        return float(candidate) <= float(control) * MATERIAL

    all_short_ok = (
        improved(all_short, 512) and improved(all_short, 6144) and d128_ok(all_short)
    )
    gap_ok = improved(gap, 6144)
    survivor = None
    reason = "neither_candidate_improved_targets"
    if all_short_ok:
        survivor = ALL_SHORT
        reason = "all_short_both_targets_improved_d128_within_2pct"
    elif gap_ok:
        survivor = GAP_ONLY
        reason = "gap_only_d6144_improved"
    return {
        "all_short_ok": all_short_ok,
        "gap_only_ok": gap_ok,
        "survivor": survivor,
        "reason": reason,
        "choice_rule": (
            "prefer all_short only if D512 and D6144 family means improve "
            "and D128 loses at most 2%; else gap_only if D6144 improves"
        ),
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
            "task": "OPT-152",
            "phase": "correctness",
            "mode": mode,
            "ok": False,
            "blocked": True,
            "blockers": [
                built.get("gpu_blocker") or f"make_rc={built['make_returncode']}"
            ],
            "build": built,
            "production_pins_parent": production_pins_parent(),
            "family_plan": family_plan(mode, "correctness"),
            "measured_at": utc_now(),
            "source": source,
            "dirty": bool(dirty),
        }
        return store_sidecar(run_dir, "correctness.json", payload)
    identity = run_native(
        run_dir, workload="correctness", tier="correctness", timeout_s=LONG_TIMEOUT_S
    )
    parsed = parse_correctness(
        (run_dir / "correctness-native.txt").read_text(encoding="utf-8")
    )
    ok = (
        bool(identity["pass"])
        and parsed["positions_ok"]
        and parsed["equal"]
        and parsed["finite"]
        and parsed["opt137_mma"]
        and parsed["graph_pass"]
        and parsed["graph_live_device"]
        and parsed["prompt_to_decode_handoff"]
        and parsed["n_parts_16"]
        and parsed["verified_max_untouched"]
        and parsed["launch_recorded"]
        and production_pins_parent()
    )
    payload = {
        "schema_version": 1,
        "task": "OPT-152",
        "phase": "correctness",
        "mode": mode,
        "ok": ok,
        "build": built,
        "identity": identity,
        "parsed": parsed,
        "identity_positions": list(IDENTITY_POSITIONS),
        "candidates": list(CANDIDATES),
        "production_coverage": production_coverage_pin(),
        "production_pins_parent": production_pins_parent(),
        "mechanism": "unknown",
        "opt130_revived": False,
        "verified_max": VERIFIED_MAX,
        "family_plan": family_plan(mode, "correctness"),
        "measured_at": utc_now(),
        "source": source,
        "dirty": bool(dirty),
        "gpu_lock": "build/optimization-runs/qw38-gpu.lock",
        "child_timeout_s": CHILD_TIMEOUT_S,
        "aggregate_deadline_s": AGGREGATE_DEADLINE_S,
    }
    dump_json(
        EVIDENCE / "dispatch-records.json",
        {"rows": parsed.get("rows"), "graphs": parsed.get("graphs")},
    )
    return store_sidecar(run_dir, "correctness.json", payload)


def run_screen(run_dir: Path, mode: str) -> dict[str, Any]:
    print(family_plan(mode, "screen"), flush=True)
    correctness = load_sidecar(run_dir, "correctness.json") or {}
    if not correctness.get("ok"):
        payload = {
            "schema_version": 1,
            "task": "OPT-152",
            "phase": "screen",
            "mode": mode,
            "ok": False,
            "screened_in": False,
            "screened_out": True,
            "advance": False,
            "survivor": None,
            "reason": "correctness_failed",
            "family_plan": family_plan(mode, "screen"),
            "measured_at": utc_now(),
        }
        return store_sidecar(run_dir, "screen.json", payload)
    native = run_native(
        run_dir, workload="screen", tier="screen", timeout_s=LONG_TIMEOUT_S
    )
    stdout = (run_dir / "screen-native.txt").read_text(encoding="utf-8")
    parsed = parse_screen(stdout)
    choice = apply_choice_rule(parsed)
    reconstruction = {
        "schema_version": 1,
        "task": "OPT-152",
        "complete_family": True,
        "rotating_layers": 16,
        "warmups": SCREEN_WARMUPS,
        "pairs": SCREEN_PAIRS,
        "shapes": parsed.get("shapes"),
        "rounds": parsed.get("rounds"),
        "choice": choice,
        "mechanism": "unknown",
        "prep_included": True,
        "combine_included": True,
        "conversion_included": True,
    }
    dump_json(EVIDENCE / "family-time-reconstruction.json", reconstruction)
    dump_json(run_dir / "family-time-reconstruction.json", reconstruction)
    numerical_ok = bool(native.get("pass"))
    survivor = choice.get("survivor") if numerical_ok else None
    screened = numerical_ok and survivor is not None
    payload = {
        "schema_version": 1,
        "task": "OPT-152",
        "phase": "screen",
        "mode": mode,
        "ok": numerical_ok,
        "screened_in": screened,
        "screened_out": numerical_ok and not screened,
        "advance": screened,
        "survivor": survivor,
        "warmups": SCREEN_WARMUPS,
        "pairs": SCREEN_PAIRS,
        "complete_attention_family": True,
        "native": native,
        "parsed": parsed,
        "choice": choice,
        "reconstruction": reconstruction,
        "candidates": list(CANDIDATES),
        "mechanism": "unknown",
        "family_plan": family_plan(mode, "screen"),
        "measured_at": utc_now(),
        "reason": None if screened else choice.get("reason"),
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
    selectors["decode_attention"] = config_id
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
        raise VectorCoverageError(
            f"OPT-058 quality failed for {config_id}: "
            f"{(completed.stderr or '')[-2000:]}"
        )
    stdout = completed.stdout or ""
    cases = _parse_quality_cases(stdout)
    restored = "restored_packed_or_r2=false" in stdout
    if not restored:
        raise VectorCoverageError("--quality restored packed or r2 defaults")
    if not cases:
        raise VectorCoverageError(f"OPT-058 produced no NLL cases for {config_id}")
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
    survivor = survivor_of(run_dir)
    if survivor is None:
        raise VectorCoverageError("quality requires a frozen survivor")
    built = ensure_built(run_dir)
    if not built["ok"]:
        raise VectorCoverageError(
            built.get("gpu_blocker") or f"make_rc={built['make_returncode']}"
        )
    control = run_quality_native(run_dir, SHIPPING_DECODE)
    candidate = run_quality_native(run_dir, survivor)
    candidate["same_math_copied_from_control"] = False
    region = run_native(
        run_dir,
        workload="quality-region",
        tier="acceptance",
        timeout_s=CHILD_TIMEOUT_S,
    )
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
        and bool(region.get("pass"))
    )
    payload = {
        "schema_version": 1,
        "task": "OPT-152",
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
        "admitted_region": region,
        "survivor": survivor,
        "quality_flag": "--quality",
        "arithmetic_changed": False,
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
    survivor = survivor_of(run_dir)
    if survivor is None:
        raise VectorCoverageError("state-memory requires a frozen survivor")
    state = run_native(
        run_dir, workload="state", tier="acceptance", timeout_s=CHILD_TIMEOUT_S
    )
    checkpoint_path = str(run_dir / "opt152-checkpoint.bin")
    checkpoint = run_locked(
        [
            f"./{CHECKPOINT_NATIVE}",
            MODEL,
            checkpoint_path,
            "--attention-pipeline",
            SHIPPING_PROMPT,
            "--decode-attention",
            survivor,
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
            survivor,
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
        "task": "OPT-152",
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
        "survivor": survivor,
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
        "input_trajectory": "opt152_vector_coverage",
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


def run_engine_ab(
    run_dir: Path,
    *,
    name: str,
    survivor: str,
    prompt: int | None = None,
    prefix: int | None = None,
    output_tokens: int | None = None,
    pairs: int = PAIR_COUNT,
    decode_attention: bool = True,
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
    ]
    if decode_attention:
        command.extend(["--decode-attention", survivor])
    else:
        command.extend(
            [
                "--attention-pipeline-control",
                SHIPPING_PROMPT,
                "--attention-pipeline",
                SHIPPING_PROMPT,
            ]
        )
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
        raise VectorCoverageError(f"{name} engine probe failed: {stdout[-2000:]}")
    if "override_before_capture_applied=true" not in stdout:
        raise VectorCoverageError(f"{name} graph capture override was not applied")
    parsed_pairs = parse_engine_pairs(stdout)
    if len(parsed_pairs) != pairs:
        raise VectorCoverageError(f"{name} engine pairs {len(parsed_pairs)} != {pairs}")
    return {
        "name": name,
        "pairs": parsed_pairs,
        "command": command,
        "raw": workspace_relative(run_dir / f"{name}-engine.txt"),
        "returncode": completed.returncode,
    }


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
                "keep_policy_id": "target_guard_v2",
            },
        )
    survivor = survivor_of(run_dir)
    if survivor is None:
        raise VectorCoverageError("performance requires a frozen survivor")
    records: list[dict[str, Any]] = []
    probes: dict[str, Any] = {}
    p4096 = run_engine_ab(
        run_dir, name="p4096", survivor=survivor, prompt=4096, decode_attention=False
    )
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
        ("d512", 512),
        ("d2048", 2048),
        ("d6144", 6144),
        ("d8192", 8192),
        ("d32768", 32768),
    ):
        probe = run_engine_ab(
            run_dir,
            name=name,
            survivor=survivor,
            prefix=prefix,
            output_tokens=DECODE_TOKENS,
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
        "task": "OPT-152",
        "phase": "performance",
        "mode": mode,
        "ok": True,
        "skipped": False,
        "pairs": records,
        "probes": {key: {"raw": value["raw"]} for key, value in probes.items()},
        "warmups": WARMUPS,
        "aa_pairs": PAIR_COUNT,
        "keep_policy_id": "target_guard_v2",
        "decode_output_tokens": DECODE_TOKENS,
        "survivor": survivor,
        "family_plan": family_plan(mode, "performance"),
        "measured_at": utc_now(),
    }
    return store_sidecar(run_dir, "performance.json", payload)


def contract_for_survivor(survivor: str) -> dict[str, Any]:
    contract = dict(load_contract())
    rules = (contract.get("candidate_rules") or {}).get(survivor) or {}
    target_ids = list(rules.get("targets") or ["d6144"])
    extra_guards = list(rules.get("extra_guards") or [])
    common = ["d128", "d2048", "d8192", "d32768", "p4096"]
    targets = [
        {"workload": name, "kind": "decode", "primary_metric": "decode_only"}
        for name in target_ids
    ]
    guards: list[dict[str, Any]] = []
    for name in [*common, *extra_guards]:
        if name in target_ids:
            continue
        kind = "prefill" if name == "p4096" else "decode"
        guards.append({"workload": name, "kind": kind})
    contract["targets"] = targets
    contract["guards"] = guards
    contract["candidate"] = survivor
    contract["candidate_id"] = survivor
    contract["target_workloads"] = target_ids
    contract["guard_workloads"] = [row["workload"] for row in guards]
    if survivor == ALL_SHORT:
        contract["dispatch_region"] = "decode[0,8191]"
    else:
        contract["dispatch_region"] = "decode[1024,8191]"
    return contract


def evaluate_keep(run_dir: Path) -> dict[str, Any]:
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
            "shipping_coverage": PARENT_COVERAGE,
            "reasons": list(
                correctness.get("blockers") or [str(screen.get("reason") or "blocked")]
            ),
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
            "shipping_coverage": PARENT_COVERAGE,
            "reasons": ["identity_or_dispatch_failed"],
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
            "shipping_coverage": PARENT_COVERAGE,
            "reasons": [str(screen.get("reason") or "choice_rule_retained_parent")],
            "claims_throughput": False,
            "candidate_measured_delta": screen.get("choice"),
            "shipping_delta": 0,
            "quality_result": "nll_not_required_screened_out",
            "screen": screen,
        }
    survivor = str(screen.get("survivor") or "")
    if screen.get("screened_in") and not quality:
        return {
            "verdict": "incomplete",
            "production_kept": False,
            "shipping_prompt_attention": SHIPPING_PROMPT,
            "shipping_decode_attention": SHIPPING_DECODE,
            "shipping_coverage": PARENT_COVERAGE,
            "reasons": ["missing_required_quality_not_no_opportunity"],
            "claims_throughput": False,
            "candidate_measured_delta": screen.get("choice"),
            "shipping_delta": 0,
            "quality_result": None,
        }
    if screen.get("screened_in") and quality.get("candidate_nll_not_measured"):
        return {
            "verdict": "incomplete",
            "production_kept": False,
            "shipping_prompt_attention": SHIPPING_PROMPT,
            "shipping_decode_attention": SHIPPING_DECODE,
            "shipping_coverage": PARENT_COVERAGE,
            "reasons": ["missing_required_quality_not_no_opportunity"],
            "shipping_delta": 0,
            "quality_result": None,
        }
    contract = contract_for_survivor(survivor)
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
        "shipping_coverage": pin_name_for_ident(survivor) if keep else PARENT_COVERAGE,
        "survivor": survivor,
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
        "final_dispatch_ranges": {
            "flash_vec_lo": 0 if survivor == ALL_SHORT else CROSSOVER,
            "flash_vec_hi": 8191 if keep else VERIFIED_MAX,
            "mma_lo": MMA_THRESHOLD,
            "verified_max": VERIFIED_MAX,
        },
    }


def maybe_flip_production_pins(keep: bool, survivor: str | None) -> None:
    if keep and survivor:
        apply_production_coverage(pin_name_for_ident(survivor))
        return
    if production_coverage_pin() != PARENT_COVERAGE:
        apply_production_coverage(PARENT_COVERAGE)


def write_report(result: Mapping[str, Any], run_dir: Path) -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    verdict = str(result.get("verdict"))
    quality = result.get("quality") or load_sidecar(run_dir, "quality.json") or {}
    screen = result.get("screen") or load_sidecar(run_dir, "screen.json") or {}
    choice = screen.get("choice") or {}
    shipping = result.get("shipping_decode_attention")
    kept = bool(result.get("production_kept"))
    held = nll_from_cases(
        (quality.get("candidate") or {}).get("cases") or [], "held_out_wikitext_1024"
    )
    pin_line = (
        f"Production coverage pin flipped to `{result.get('shipping_coverage')}`."
        if kept
        else "Parent retained; coverage pin stays `parent`."
    )
    lines = [
        "# OPT-152 — Vector attention coverage throughout the sub-8K region",
        "",
        f"Status: **{verdict}**. Shipping decode `{shipping}`. {pin_line}",
        "",
        f"Parent `{PARENT_STACK}` with kept OPT-148 flash-vec in `[1024,4096]`, "
        f"warp_query below 1024 and on `[4097,8191]`, OPT-137 MMA at `>=8192`. "
        "OPT-151 remains unselected. This task is dispatch-only; OPT-148 "
        "arithmetic and `n_parts=16` are unchanged. `verified_max` stays 4096.",
        "",
        "## Candidates",
        "",
        f"`{GAP_ONLY}`: flash-vec on `[1024,8191]`, warp_query below 1024, "
        "target D6144.",
        f"`{ALL_SHORT}`: flash-vec on `[0,8191]`, targets D512 and D6144; "
        "D128 is a non-regression guard.",
        "",
        f"Frozen choice rule: `{choice.get('choice_rule')}`.",
        f"Survivor: `{result.get('survivor') or screen.get('survivor')}`.",
        "",
        "## Correctness",
        "",
        "Positions 0,1,127,128,511,1023,1024,4095,4096,4097,6144,8190,8191,8192 "
        "with layers 3/7/63, sampled FP64, candidate-row visibility, empty/tail "
        "partitions, and actual post-launch dispatch. Graph replay used live "
        "DecodeLaunchState.position across 1024/4096/8192; host-only selector "
        "equality is not the proof.",
        "",
        f"Correctness sidecar ok=`{(load_sidecar(run_dir, 'correctness.json') or {}).get('ok')}`.",
        "",
        "## Screen",
        "",
        f"Warmups `{SCREEN_WARMUPS}`, pairs `{SCREEN_PAIRS}`, shapes "
        "D128/D512/D6144, 16-layer complete family including prep/combine/"
        f"conversion. Screened in: `{screen.get('screened_in')}`. Reason: "
        f"`{screen.get('reason')}`.",
        "",
        "Independent reconstruction: "
        "[`family-time-reconstruction.json`](family-time-reconstruction.json).",
        "",
        "## Quality",
        "",
        (
            "OPT-058 invoked=`"
            + str(quality.get("opt058_invoked"))
            + "`; candidate NLL measured=`"
            + str(quality.get("candidate_nll_measured"))
            + "`; skip_reason=`"
            + str(quality.get("skip_reason"))
            + "`. Newly admitted region also exercised by native quality-region."
        ),
        "",
        f"Held-out NLL `{held}`. ppl_ratio=`{quality.get('ppl_ratio')}`.",
        "",
        "## Keep policy",
        "",
        "`target_guard_v2` with frozen survivor targets. Guards D128/D2048/"
        "D8192/D32768 decode_only and complete_request plus P4096; gap_only "
        "also guards D512. Target L>1, guard L>=0.98, decode p95<=1.05. "
        "2% materiality is reported separately and is not a keep threshold.",
        "",
        f"Verdict `{verdict}`. production_kept=`{result.get('production_kept')}`.",
        f"claims_throughput: `{bool(result.get('claims_throughput'))}`.",
        "",
        "## Deltas",
        "",
        f"Candidate measured delta: `{result.get('candidate_measured_delta')}`.",
        f"Shipping delta: `{result.get('shipping_delta')}`.",
        f"Quality result: `{result.get('quality_result')}`.",
        f"Final dispatch ranges: `{result.get('final_dispatch_ranges')}`.",
        "",
        "## Performance evidence checklist",
        "",
        "1. **Measurement identity** — parent post-148 flash-vec + OPT-137 MMA; "
        f"GGUF `{GGUF_SHA}`; llama `cc83d7b4824f73cfdda4dfbb47ee39804f71b328`.",
        "2. **Coverage** — 14 positions, graph live-position replay, D128/D512/"
        "D6144 screen, llama padded kernel labels.",
        "3. **Time accounting** — screen means are complete-family enclosing times.",
        "4. **Contradiction register** — llama padded K.ne[1] 8192 is not Quartz "
        "position; Quartz MMA remains position>=8192.",
        "5. **Claim types** — dispatch experiment; mechanism unknown.",
        "6. **Target/guard** — OPT-135 target_guard_v2 for screened-in survivors.",
        "7. **Independent verification** — family-time reconstruction from rounds.",
        "8. **Reporting** — candidate vs shipping deltas are separate.",
        "",
        "## Status",
        "",
        f"verdict=`{verdict}` production_kept=`{result.get('production_kept')}` "
        f"coverage_pin=`{production_coverage_pin()}`.",
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
        "task": "OPT-152",
        "mode": mode,
        "llama_revision": "cc83d7b4824f73cfdda4dfbb47ee39804f71b328",
        "gguf_sha256": GGUF_SHA,
        "parent": PARENT_STACK,
        "parent_prompt_attention": SHIPPING_PROMPT,
        "candidate": result.get("survivor") or GAP_ONLY,
        "quality": quality,
        "state_memory": load_sidecar(run_dir, "state-memory.json") or {},
        "performance": load_sidecar(run_dir, "performance.json")
        or result.get("performance"),
        "shipping_prompt_attention": result.get("shipping_prompt_attention"),
        "shipping_decode_attention": result.get("shipping_decode_attention"),
        "shipping_coverage": result.get("shipping_coverage"),
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
        "opt130_revived": False,
        "verified_max": VERIFIED_MAX,
        "final_dispatch_ranges": result.get("final_dispatch_ranges"),
    }
    dump_json(FIXTURE, payload)
    dump_json(EVIDENCE / "answers.json", payload)


def validate_fixture(payload: Mapping[str, Any]) -> dict[str, Any]:
    missing = [key for key in REQUIRED_FIXTURE_KEYS if key not in payload]
    return {
        "ok": not missing and payload.get("task") == "OPT-152",
        "task": "OPT-152",
        "missing": missing,
    }


def run_report(run_dir: Path, mode: str) -> dict[str, Any]:
    result = evaluate_keep(run_dir)
    result["quality"] = load_sidecar(run_dir, "quality.json") or {}
    maybe_flip_production_pins(
        bool(result.get("production_kept")),
        str(result.get("survivor") or "") or None,
    )
    write_report(result, run_dir)
    write_fixture(run_dir, result, mode)
    copy_raw(run_dir)
    payload = {
        "schema_version": 1,
        "task": "OPT-152",
        "phase": "report",
        "mode": mode,
        "ok": True,
        "verdict": result.get("verdict"),
        "production_kept": result.get("production_kept"),
        "shipping_decode_attention": result.get("shipping_decode_attention"),
        "shipping_coverage": result.get("shipping_coverage"),
        "tok_s_deltas": result.get("tok_s_deltas"),
        "candidate_measured_delta": result.get("candidate_measured_delta"),
        "shipping_delta": result.get("shipping_delta"),
        "quality_result": result.get("quality_result"),
        "family_plan": family_plan(mode, "report"),
        "measured_at": utc_now(),
        "production_coverage": production_coverage_pin(),
        "final_dispatch_ranges": result.get("final_dispatch_ranges"),
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
    raise VectorCoverageError(f"unknown phase {phase}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--mode", choices=("feedback", "acceptance", "release"))
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    mode = args.mode or "feedback"
    try:
        payload = run_phase(args.run_dir, mode, args.phase)
    except VectorCoverageError as exc:
        print(
            json.dumps(
                {"task": "OPT-152", "phase": args.phase, "ok": False, "error": str(exc)}
            )
        )
        return 1
    print(
        json.dumps(
            {
                "task": "OPT-152",
                "phase": args.phase,
                "ok": bool(payload.get("ok", True)),
                "verdict": payload.get("verdict"),
                "screened_in": payload.get("screened_in"),
                "survivor": payload.get("survivor"),
            }
        )
    )
    return 0 if payload.get("ok", True) else 1


if __name__ == "__main__":
    sys.exit(main())
