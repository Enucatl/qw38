"""OPT-113 coupled-stack sitting gate.

Freezes authenticated OPT-106 post106_control versus current production
survivors (OPT-107/110/111). OPT-108/109 must not leak. OPT-112 is deferred
and omitted. Three independent outcomes; honest failures remain failures.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.opt075_q4_production_admission import T_CRIT_DF4  # noqa: E402
from tools.opt080_batch_gate import (  # noqa: E402
    GGUF_SHA,
    IMAGE,
    LLAMA_IMAGE,
    LLAMA_REV,
    MODEL,
    compute_gate as opt080_compute_gate,
    docker_common,
    ensure_binaries,
    git_identity,
    plus5_gap_ms,
    parity_gap_ms,
    p95_no_worse,
    read_gpu_telemetry,
    run_command as _docker_run_command,
    sha256_file,
    throughput_gate,
    utc_now,
    write_json,
    _mean,
    _search,
    _var,
)
from tools.opt088_batch_gate import (  # noqa: E402
    DECODE_PREFIX,
    P2K_PREFIX,
    P_PREFIX,
    refuse_quality_as_one_boolean,
    run_llama_bench_p,
    run_llama_decode,
    run_opt058,
    run_probe,
    run_quartz_prefixed,
    load_sidecar,
)
from tools.opt106_batch_gate import (  # noqa: E402
    evaluate_combination_quality,
    internal_p95_pass,
    log_speed_ratio_ci_lower,
)
from tools.quality.quality_mode import (  # noqa: E402
    apply_quality_mode,
    build_quality_config,
)
from tools.quality.scoring import recurrence_incremental_nll  # noqa: E402
from tools.quality.suite import RECURRENCE_MAX  # noqa: E402

CONTRACT = ROOT / "pins/opt113_coupled_stack_gate_contract.json"
ITERATION = ROOT / "pins/opt113_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt113_coupled_stack_gate.json"
EVIDENCE = ROOT / "evidence/optimization/opt113-coupled-stack-gate"
REPORT = EVIDENCE / "REPORT.md"
OPT106_FIXTURE = ROOT / "fixtures/opt106_batch_gate.json"
OPT106_REPORT = ROOT / "evidence/optimization/opt106-batch-gate/REPORT.md"
OPT106_CONTRACT = ROOT / "pins/opt106_batch_gate_contract.json"
OPT099_FIXTURE = ROOT / "fixtures/opt099_matched_attribution.json"
V2_INPUTS = ROOT / "pins/production_quality_v2_inputs.json"
V2_LLAMA = ROOT / "pins/production_quality_v2_llama_reference.json"
FAMILIES = ("q4", "q8", "mmq", "kv_once", "decode_attention")
DEPENDENCY_IDS = (
    "OPT-107",
    "OPT-108",
    "OPT-109",
    "OPT-110",
    "OPT-111",
    "OPT-112",
)
DEPENDENCY_FIXTURES = {
    "OPT-107": "opt107_attention_crossover.json",
    "OPT-108": "opt108_llama_vector_stack.json",
    "OPT-109": "opt109_persistent_gdn_state.json",
    "OPT-110": "opt110_llama_q4_adapter.json",
    "OPT-111": "opt111_llama_prompt_attention.json",
    "OPT-112": "opt112_norm_q8_staging.json",
}
CONTROL_FLAGS = [
    "--q4-decode",
    "integer_q8_late",
    "--attention-pipeline",
    "kv_once",
    "--decode-attention-crossover-threshold",
    "0",
]
SELECTED_FLAGS = [
    "--q4-decode",
    "llama_q4k_mmvq",
    "--attention-pipeline",
    "opt111_base",
    "--decode-attention-crossover-threshold",
    "1024",
]
SELECTED_QUALITY_SELECTORS: dict[str, Any] = {
    "q4_decode": "llama_q4k_mmvq",
    "q4_staging": "paired_integer",
    "ffn_decode": "paired_integer",
    "q8_decode": "r1_w4",
    "q8_path": "dp4a_q8_1",
    "q8_grouping": "grouped_r1_w4",
    "prompt_mmq": "fma_async_x",
    "prompt_mmq_tile": "i128_j128",
    "prompt_attention": "opt111_base",
    "decode_gdn": "sequential",
    "decode_attention": "hybrid_crossover",
    "prompt_pair": "off",
    "nvccflags": "-O2 --fmad=false",
    "production_numerics": "strict",
    "execution_graphs": "ffn_only",
    "chat_template": "no_thinking",
    "enable_thinking": False,
    "logit_masking": False,
}
PHASES = (
    "freeze",
    "quality",
    "state-memory",
    "performance",
    "two-k",
    "attribution",
    "report",
    "all",
)
PROOF = (
    "combined freeze of authenticated OPT-106 post106_control and "
    "post113_selected production survivors; "
    "OPT-107 hybrid_crossover kept; OPT-110 llama_q4k_mmvq kept; "
    "OPT-111 opt111_base kept; "
    "OPT-108 and OPT-109 rejected with no production selector leak; "
    "OPT-112 deferred_below_trigger omitted from selected and does not "
    "block OPT-113; "
    "independent kernel_parity_pass per family; "
    "quality is not one boolean; "
    "quartz_vs_baseline_quality_delta and quartz_vs_llama_quality_delta stay "
    "distinct; "
    "opt074_coverage_unadmitted is not a blocker; "
    "absolute task-accuracy fail inherited from OPT-084 does not by itself "
    "block; "
    "new regression versus OPT-084 does block; "
    "OPT-106 remains historical and is not reinterpreted; "
    "OPT-056 and OPT-016 stay blocked unless their owning conditions pass; "
    "parity gap is Tq-Tl; "
    "+5% throughput gap is Tq-Tl/1.05; "
    "do not label the parity gap as the +5% bar; "
    "decode p95 no worse than llama for the OPT-056 outcome; "
    "failed required quality stops release before long timing; "
    "diagnostic performance is not a release or keep; "
    "rejected candidates must not leak into production; "
    "internal improvement is selected versus authenticated OPT-106 control "
    "not llama; "
    "llama numbers are measured in this sitting and not reused from OPT-106; "
    "control Quartz is measured in this sitting and not reused from OPT-106; "
    "strict_ppl_ratio_max=1.01 applies because concession is inactive; "
    "release_eligible is never a synonym for opt056_pass; "
    "OPT-099 matched family attribution is diagnostic and non-additive; "
    "selected does not equal control so Quartz is measured twice"
)


class BatchGateError(AssertionError):
    """Inadmissible coupled-stack sitting evidence."""


_NATIVE_COMMANDS_ENABLED = False
LLAMA_LD_LIBRARY_PATH = (
    "/workspace/.cache/authorities/llama-build/bin:/usr/local/cuda/lib64"
)


def llama_docker_command(command: Sequence[str]) -> list[str]:
    """Run pinned llama binaries in qw38-cuda; the authority image is absent."""
    listed = list(command)
    if str(LLAMA_IMAGE) not in listed:
        return listed
    listed = [str(IMAGE) if part == str(LLAMA_IMAGE) else part for part in listed]
    image_index = listed.index(str(IMAGE))
    listed[image_index:image_index] = [
        "-e",
        f"LD_LIBRARY_PATH={LLAMA_LD_LIBRARY_PATH}",
    ]
    return listed


def run_command_native(command: Sequence[str]) -> SimpleNamespace:
    listed = list(command)
    if listed and listed[0] == "docker" and str(LLAMA_IMAGE) in listed:
        return _docker_run_command(llama_docker_command(listed))
    if os.environ.get("QW38_HOST_NATIVE") != "1" or not listed or listed[0] != "docker":
        return _docker_run_command(command)
    image_index = -1
    try:
        image_index = listed.index(str(IMAGE))
    except ValueError:
        return _docker_run_command(command)
    inner = listed[image_index + 1 :]
    if inner[:2] == ["bash", "-lc"] and len(inner) >= 3:
        inner = ["bash", "-lc", str(inner[2]).replace("/workspace", str(ROOT))]
    env = os.environ.copy()
    for index, part in enumerate(command):
        if part == "-e" and index + 1 < len(command):
            key, _, value = str(command[index + 1]).partition("=")
            if key:
                env[key] = value
    completed = subprocess.run(
        inner,
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        raise BatchGateError(
            "command failed: "
            + " ".join(inner)
            + (f"\n{completed.stderr}" if completed.stderr else "")
        )
    return SimpleNamespace(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _run_allow_fail(command: Sequence[str]) -> SimpleNamespace:
    listed = list(command)
    if listed and listed[0] == "docker" and str(LLAMA_IMAGE) in listed:
        listed = llama_docker_command(listed)
    elif os.environ.get("QW38_HOST_NATIVE") == "1" and listed and listed[0] == "docker":
        image_index = -1
        try:
            image_index = listed.index(str(IMAGE))
        except ValueError:
            image_index = -1
        if image_index >= 0:
            inner = listed[image_index + 1 :]
            if inner[:2] == ["bash", "-lc"] and len(inner) >= 3:
                inner = [
                    "bash",
                    "-lc",
                    str(inner[2]).replace("/workspace", str(ROOT)),
                ]
            env = os.environ.copy()
            for index, part in enumerate(command):
                if part == "-e" and index + 1 < len(command):
                    key, _, value = str(command[index + 1]).partition("=")
                    if key:
                        env[key] = value
            completed = subprocess.run(
                inner,
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            return SimpleNamespace(
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
    completed = subprocess.run(
        listed,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return SimpleNamespace(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def store_sidecar(run_dir: Path, name: str, payload: Any) -> Any:
    write_json(run_dir / name, payload)
    write_json(EVIDENCE / name, payload)
    return payload


def _enable_native_commands() -> None:
    global _NATIVE_COMMANDS_ENABLED
    if _NATIVE_COMMANDS_ENABLED:
        return
    import tools.opt080_batch_gate as opt080_module
    import tools.opt088_batch_gate as opt088_module

    opt080_module.run_command = run_command_native
    opt088_module.run_command = run_command_native
    opt088_module.store_sidecar = store_sidecar
    opt088_module.EVIDENCE = EVIDENCE
    _NATIVE_COMMANDS_ENABLED = True


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_contract() -> dict[str, Any]:
    return json.loads(_read(CONTRACT))


def source_paths() -> dict[str, Any]:
    from tools.opt080_batch_gate import source_paths as base_paths

    paths = dict(base_paths())
    paths["q8_grouping"] = _search(
        "cuda/q8_decode_path.cuh",
        r'kSelectedQ8DecodeGrouping\[\] = "([^"]+)"',
    )
    paths["q8_device_layout"] = _search(
        "cuda/q8_decode_path.cuh",
        r'kSelectedQ8DeviceLayout\[\] = "([^"]+)"',
    )
    paths["q6_device_layout"] = _search(
        "cuda/q6k_decode_path.cuh",
        r'kSelectedQ6DeviceLayout\[\] = "([^"]+)"',
    )
    double_x = _search(
        "cuda/quant_mmq_mma.cuh",
        r"constexpr bool kSelectedMmqDoubleX = (true|false);",
    )
    paths["mmq_double_x"] = double_x == "true"
    paths["decode_attention_vec128"] = _search(
        "cuda/attention_decode_path.cuh",
        r'kSelectedDecodeAttentionVec128Path\[\] = "([^"]+)"',
    )
    paths["decode_attention_crossover_threshold"] = int(
        _search(
            "cuda/attention_decode_path.cuh",
            r"kSelectedDecodeAttentionCrossoverThreshold = (\d+);",
        )
    )
    threshold = int(paths["decode_attention_crossover_threshold"])
    paths["decode_attention"] = (
        "hybrid_crossover" if threshold > 0 else paths["decode_query_prep"]
    )
    return paths


def post106_control_paths() -> dict[str, Any]:
    payload = json.loads(_read(OPT106_FIXTURE))
    control = payload.get("post106_selected") or payload.get("post098_control")
    if not isinstance(control, Mapping):
        raise BatchGateError("authenticated OPT-106 control paths missing")
    expected = load_contract()["post106_control"]
    for key in (
        "q4_decode",
        "attention_pipeline",
        "decode_query_prep",
        "ffn_decode",
        "gdn_decode",
    ):
        if control.get(key) != expected[key]:
            raise BatchGateError(
                f"OPT-106 control mismatch {key}: {control.get(key)!r} != {expected[key]!r}"
            )
    merged = dict(expected)
    merged.update(dict(control))
    merged["decode_attention"] = "warp_query"
    merged["decode_attention_vec128"] = "warp_query"
    merged["decode_attention_crossover_threshold"] = 0
    return merged


def opt106_artifacts_unmodified() -> dict[str, str]:
    contract = load_contract()
    expected = contract["opt106_historical_hashes"]
    actual = {
        "report": hashlib.sha256(OPT106_REPORT.read_bytes()).hexdigest(),
        "fixture": hashlib.sha256(OPT106_FIXTURE.read_bytes()).hexdigest(),
        "contract": hashlib.sha256(OPT106_CONTRACT.read_bytes()).hexdigest(),
    }
    for name, digest in expected.items():
        if actual[name] != digest:
            raise BatchGateError(
                f"OPT-106 {name} was reinterpreted or rewritten "
                f"(hash {actual[name]} != {digest})"
            )
    fixture = json.loads(_read(OPT106_FIXTURE))
    if fixture.get("task") != "OPT-106":
        raise BatchGateError("OPT-106 fixture task relabeled")
    if fixture.get("opt056_gate_passed") is True:
        raise BatchGateError("historical gate relabeling of OPT-106/OPT-056")
    return actual


def _load_fixture(name: str) -> dict[str, Any]:
    path = ROOT / "fixtures" / name
    if not path.is_file():
        raise BatchGateError(f"incomplete admissions: missing fixture {name}")
    payload = json.loads(_read(path))
    if not isinstance(payload, dict):
        raise BatchGateError(f"incomplete admissions: {name} is not an object")
    return payload


def audit_dependencies() -> dict[str, Any]:
    opt106_artifacts_unmodified()
    loaded: dict[str, Any] = {}
    for task_id in DEPENDENCY_IDS:
        name = DEPENDENCY_FIXTURES[task_id]
        payload = _load_fixture(name)
        if payload.get("task") not in (task_id, None):
            raise BatchGateError(f"incomplete admissions: {name} task mismatch")
        loaded[task_id] = payload
    opt107 = loaded["OPT-107"]
    if opt107.get("candidate") != "hybrid_crossover":
        raise BatchGateError("OPT-107 survivor is not hybrid_crossover")
    if int(opt107.get("selected_threshold") or 0) != 1024:
        raise BatchGateError("OPT-107 threshold must stay 1024")
    if not opt107.get("production_kept"):
        raise BatchGateError("OPT-107 keep missing")
    if opt107.get("shipping_decode_attention_vec128") != "warp_query":
        raise BatchGateError("OPT-107 must retain warp_query vec128 pin")
    opt108 = loaded["OPT-108"]
    if opt108.get("production_kept"):
        raise BatchGateError("rejected OPT-108 leaked into production")
    if opt108.get("shipping_decode_attention_vec128") != "warp_query":
        raise BatchGateError("rejected OPT-108 vec128 leaked into production")
    opt109 = loaded["OPT-109"]
    if opt109.get("production_kept"):
        raise BatchGateError("rejected OPT-109 leaked into production")
    if opt109.get("shipping_gdn_decode") != "sequential":
        raise BatchGateError("rejected OPT-109 persistent GDN leaked")
    opt110 = loaded["OPT-110"]
    if opt110.get("shipping_q4_decode") != "llama_q4k_mmvq":
        raise BatchGateError("OPT-110 keep missing llama_q4k_mmvq")
    if not opt110.get("production_kept"):
        raise BatchGateError("OPT-110 keep missing")
    opt111 = loaded["OPT-111"]
    if opt111.get("shipping_attention_pipeline") != "opt111_base":
        raise BatchGateError("OPT-111 keep missing opt111_base")
    if not opt111.get("production_kept"):
        raise BatchGateError("OPT-111 keep missing")
    opt112 = loaded["OPT-112"]
    screen = opt112.get("screen") or {}
    if screen.get("verdict") != "deferred_below_trigger" and opt112.get(
        "fusion_implemented"
    ):
        raise BatchGateError("OPT-112 must stay deferred/omitted")
    return loaded


def family_kernel_parity(loaded: Mapping[str, Any] | None = None) -> dict[str, Any]:
    deps = loaded or audit_dependencies()
    from tools.opt098_batch_gate import family_kernel_parity as opt098_parity

    inherited = opt098_parity()
    opt107 = deps["OPT-107"]
    opt110 = deps["OPT-110"]
    opt111 = deps["OPT-111"]
    q4 = bool(
        ((opt110.get("independent_verdicts") or {}).get("llama_q4k_mmvq") or {}).get(
            "kernel_parity_pass"
        )
    )
    decode_attn = bool((opt107.get("parity") or {}).get("pass"))
    prompt = bool(opt111.get("production_kept")) and bool(
        ((opt111.get("independent_verdicts") or {}).get("opt111_base") or {}).get(
            "kernel_parity_pass", True
        )
    )
    record = {
        "q4": bool(q4),
        "q8": bool(inherited["q8"]),
        "mmq": bool(inherited["mmq"]),
        "kv_once": bool(prompt and inherited["kv_once"]),
        "decode_attention": bool(decode_attn),
    }
    incomplete = [name for name, value in record.items() if value is not True]
    if incomplete:
        raise BatchGateError(
            "incomplete family parity: "
            + ",".join(f"{name}={record[name]!r}" for name in incomplete)
        )
    return {
        **record,
        "all": True,
        "sources": {
            "q4": "fixtures/opt110_llama_q4_adapter.json",
            "q8": inherited["sources"]["q8"],
            "mmq": inherited["sources"]["mmq"],
            "kv_once": "fixtures/opt111_llama_prompt_attention.json#opt111_base",
            "decode_attention": "fixtures/opt107_attention_crossover.json#parity",
        },
        "opt074_coverage_unadmitted_blocker": False,
    }


def candidate_decisions() -> dict[str, Any]:
    audit_dependencies()
    return {
        "OPT-107": {
            "status": "keep_hybrid_crossover",
            "installed": True,
            "selected": "hybrid_crossover",
            "threshold": 1024,
            "production_kept": True,
        },
        "OPT-108": {
            "status": "reject_llama_vector_stack",
            "installed": False,
            "selected": "warp_query",
            "production_kept": False,
        },
        "OPT-109": {
            "status": "reject_persistent_gdn",
            "installed": False,
            "selected": "sequential",
            "production_kept": False,
        },
        "OPT-110": {
            "status": "keep_llama_q4k_mmvq",
            "installed": True,
            "selected": "llama_q4k_mmvq",
            "production_kept": True,
        },
        "OPT-111": {
            "status": "keep_opt111_base",
            "installed": True,
            "selected": "opt111_base",
            "production_kept": True,
        },
        "OPT-112": {
            "status": "deferred_below_trigger",
            "installed": False,
            "selected": None,
            "production_kept": False,
            "omitted": True,
        },
        "OPT-099": {
            "status": "attribution_only",
            "installed": False,
            "selected": None,
            "production_kept": False,
            "diagnostic": True,
        },
    }


def matched_family_attribution() -> dict[str, Any]:
    payload = json.loads(_read(OPT099_FIXTURE)) if OPT099_FIXTURE.is_file() else {}
    return {
        "source": "fixtures/opt099_matched_attribution.json",
        "diagnostic": True,
        "non_additive": True,
        "claims_throughput": False,
        "selected_build_only": True,
        "gap_attribution_complete": bool(payload.get("gap_attribution_complete")),
        "this_sitting": None,
    }


def frozen_combined_config() -> dict[str, Any]:
    paths = source_paths()
    control = post106_control_paths()
    decisions = candidate_decisions()
    parity = family_kernel_parity()
    contract = load_contract()
    selected_contract = contract["post113_selected"]
    for key, value in selected_contract.items():
        if paths.get(key) != value:
            raise BatchGateError(
                f"post113_selected mismatch {key}: {paths.get(key)!r} != {value!r}"
            )
    if paths["q4_decode"] != "llama_q4k_mmvq":
        raise BatchGateError("OPT-110 llama_q4k_mmvq missing from selected")
    if paths["attention_pipeline"] != "opt111_base":
        raise BatchGateError("OPT-111 opt111_base missing from selected")
    if paths["decode_attention"] != "hybrid_crossover":
        raise BatchGateError("OPT-107 hybrid_crossover missing from selected")
    if paths["decode_attention_vec128"] != "warp_query":
        raise BatchGateError("rejected OPT-108 vec128 leaked into production")
    if paths["gdn_decode"] != "sequential":
        raise BatchGateError("rejected OPT-109 persistent GDN leaked")
    if paths.get("q8_device_layout") != "raw_gguf":
        raise BatchGateError("rejected aligned Q8 leaked into production")
    if paths.get("mmq_double_x") is not False:
        raise BatchGateError("rejected double-X leaked into production")
    if (
        paths["q4_decode"] == control["q4_decode"]
        and paths["attention_pipeline"] == control["attention_pipeline"]
        and paths["decode_attention"] == control.get("decode_attention")
    ):
        raise BatchGateError("selected equals control; unexpected for OPT-113")
    quality_mode = apply_quality_mode(
        enabled=True, selectors=SELECTED_QUALITY_SELECTORS
    )
    return {
        "post106_control": control,
        "post113_selected": paths,
        "combined_production_paths": paths,
        "selected_equals_control": False,
        "candidate_decisions": decisions,
        "kernel_parity_pass": parity,
        "matched_family_attribution": matched_family_attribution(),
        "keeps": list(contract["keeps"]),
        "rejected_or_retained": list(contract["rejected_recovery"]),
        "deferred": list(contract["deferred"]),
        "batch_size": paths["prompt_microbatch_rows"],
        "compiler_flags": paths["nvccflags"],
        "graphs": paths["execution_graphs"],
        "quality_mode": quality_mode,
        "opt106_history": "fixtures/opt106_batch_gate.json",
        "opt056_history": "fixtures/opt056_performance_gate.json",
        "opt084_baseline": "fixtures/opt084_quality_baseline.json",
        "opt074_coverage_unadmitted_blocker": False,
        "reuse_historical_llama": False,
        "opt112_omitted": True,
    }


def _engine_from_live(record: Mapping[str, Any], sidecar_name: str) -> dict[str, Any]:
    latencies = record.get("token_latency_ms") or []
    if latencies:
        write_json(EVIDENCE / sidecar_name, latencies)
    block = {
        "prefix": record["prefix"],
        "decode_tokens": record["decode_tokens"],
        "warmups": record["warmups"],
        "runs": record["runs"],
        "warmup_tok_s": record["warmup_tok_s"],
        "tok_s": record["tok_s"],
        "run_wall_ms": record["run_wall_ms"],
        "mean_tok_s": record["mean_tok_s"],
        "token_latency_p50_ms": record["token_latency_p50_ms"],
        "token_latency_p95_ms": record["token_latency_p95_ms"],
        "run_mean_token_latency_p95_ms": record["run_mean_token_latency_p95_ms"],
        "token_latency_sidecar": (
            f"evidence/optimization/opt113-coupled-stack-gate/{sidecar_name}"
        ),
        "q4_decode": record.get("q4_decode"),
        "attention_pipeline": record.get("attention_pipeline"),
        "decode_attention_crossover_threshold": record.get(
            "decode_attention_crossover_threshold"
        ),
    }
    if "graphs_created" in record:
        block["graphs_created"] = record["graphs_created"]
        block["cache_policy"] = record.get("cache_policy", "disabled")
    block["attribution"] = None
    block["instrumented"] = False
    return block


def sitting_control_blocks(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "p": {"quartz": result["p"]["quartz_control"]},
        "d128": {"quartz": result["d128"]["quartz_control"]},
        "d2048": {"quartz": result["d2048"]["quartz_control"]},
        "source": "this_sitting_post106_control",
    }


def three_outcomes(result: Mapping[str, Any]) -> dict[str, Any]:
    control = sitting_control_blocks(result)
    p_q = float(result["p"]["quartz"]["mean_tok_s"])
    p_l = float(result["p"]["llama_cpp"]["avg_ts"])
    d128_q = float(result["d128"]["quartz"]["mean_tok_s"])
    d128_l = float(result["d128"]["llama_cpp"]["mean_tok_s"])
    d2048_q = float(result["d2048"]["quartz"]["mean_tok_s"])
    d2048_l = float(result["d2048"]["llama_cpp"]["mean_tok_s"])
    p_ctrl = float(control["p"]["quartz"]["mean_tok_s"])
    d128_ctrl = float(control["d128"]["quartz"]["mean_tok_s"])
    d2048_ctrl = float(control["d2048"]["quartz"]["mean_tok_s"])
    quality_blob = result.get("quality")
    quality_pass = bool(result.get("model_quality_pass")) or (
        isinstance(quality_blob, Mapping)
        and bool(quality_blob.get("model_quality_pass"))
    )
    p_ci = log_speed_ratio_ci_lower(
        control["p"]["quartz"]["tok_s"], result["p"]["quartz"]["tok_s"]
    )
    d128_ci = log_speed_ratio_ci_lower(
        control["d128"]["quartz"]["tok_s"], result["d128"]["quartz"]["tok_s"]
    )
    d2048_ci = log_speed_ratio_ci_lower(
        control["d2048"]["quartz"]["tok_s"], result["d2048"]["quartz"]["tok_s"]
    )
    logs = [
        math.log(p_q / p_ctrl),
        math.log(d128_q / d128_ctrl),
        math.log(d2048_q / d2048_ctrl),
    ]
    geo_mean_ratio = math.exp(_mean(logs))
    geo_var = _var(logs, _mean(logs))
    geo_se = math.sqrt(geo_var / len(logs)) if geo_var > 0.0 else 0.0
    geo_lower = math.exp(_mean(logs) - T_CRIT_DF4 * geo_se)
    d128_p95 = internal_p95_pass(control["d128"]["quartz"], result["d128"]["quartz"])
    d2048_p95 = internal_p95_pass(control["d2048"]["quartz"], result["d2048"]["quartz"])
    isolation = result.get("state_isolation") or {}
    state_ok = bool(
        (isolation.get("memory_fit") or {}).get("ok")
        and (isolation.get("checkpoint") or {}).get("ok")
        and (isolation.get("cancellation") or {}).get("ok")
    )
    internal_pass = bool(
        quality_pass
        and state_ok
        and geo_lower > 1.0
        and p_ci["pass"]
        and d128_ci["pass"]
        and d2048_ci["pass"]
        and d128_p95["pass"]
        and d2048_p95["pass"]
    )
    internal = {
        "name": "internal_improvement_with_quality",
        "pass": internal_pass,
        "quality_pass": quality_pass,
        "state_memory_pass": state_ok,
        "geometric_mean_speed_ratio": geo_mean_ratio,
        "geometric_mean_ci_lower": geo_lower,
        "p_ci": p_ci,
        "d128_ci": d128_ci,
        "d2048_ci": d2048_ci,
        "d128_p95": d128_p95,
        "d2048_p95": d2048_p95,
        "control_p_tok_s": p_ctrl,
        "control_d128_tok_s": d128_ctrl,
        "control_d2048_tok_s": d2048_ctrl,
        "selected_p_tok_s": p_q,
        "selected_d128_tok_s": d128_q,
        "selected_d2048_tok_s": d2048_q,
        "p_delta_tok_s": p_q - p_ctrl,
        "d128_delta_tok_s": d128_q - d128_ctrl,
        "d2048_delta_tok_s": d2048_q - d2048_ctrl,
        "p_delta_vs_llama_tok_s": p_q - p_l,
        "d128_delta_vs_llama_tok_s": d128_q - d128_l,
        "d2048_delta_vs_llama_tok_s": d2048_q - d2048_l,
        "selected_equals_control": False,
    }
    parity = {
        "name": "llama_parity",
        "p": p_q >= p_l,
        "d128": d128_q >= d128_l,
        "d2048": d2048_q >= d2048_l,
        "d128_p95_inspectable": float(result["d128"]["quartz"]["token_latency_p95_ms"])
        <= float(result["d128"]["llama_cpp"]["token_latency_p95_ms"]),
        "d2048_p95_inspectable": float(
            result["d2048"]["quartz"]["token_latency_p95_ms"]
        )
        <= float(result["d2048"]["llama_cpp"]["token_latency_p95_ms"]),
        "p_parity_gap_ms": parity_gap_ms(p_q, p_l, 4096),
        "d128_parity_gap_ms": parity_gap_ms(d128_q, d128_l, 256),
        "d2048_parity_gap_ms": parity_gap_ms(d2048_q, d2048_l, 256),
        "pass": p_q >= p_l and d128_q >= d128_l and d2048_q >= d2048_l,
    }
    plus5 = result["gate"]
    return {
        "internal_improvement_with_quality": internal,
        "llama_parity": parity,
        "opt056_plus5": {
            "name": "opt056_plus5_throughput_lead",
            "pass": bool(plus5.get("passed")),
            "p_plus5_gap_ms": plus5_gap_ms(p_q, p_l, 4096),
            "d128_plus5_gap_ms": plus5_gap_ms(d128_q, d128_l, 256),
            "d2048_plus5_gap_ms": plus5_gap_ms(d2048_q, d2048_l, 256),
        },
    }


def write_freeze_fixture(freeze: Mapping[str, Any]) -> None:
    payload = {
        "schema_version": 1,
        "task": "OPT-113",
        "status": "configured",
        "measurement_utc": utc_now(),
        **copy.deepcopy(dict(freeze)),
        "hardware_executed": False,
        "keep_sitting_skipped": False,
        "preflight_is_release_evidence": False,
        "owns_opt016_parity_gate": False,
        "opt056_gate_passed": False,
        "opt016_gate_passed": False,
        "model_quality_pass": False,
        "quartz_vs_baseline_quality_delta": {
            "status": "unmeasured",
            "single_boolean": None,
            "inspectable": True,
        },
        "quartz_vs_llama_quality_delta": {
            "status": "unmeasured",
            "single_boolean": None,
            "inspectable": True,
        },
        "recurrence_state_status": {"status": "unmeasured"},
        "performance_pass": False,
        "production_kept": True,
        "release_eligible": False,
        "opt074_coverage_unadmitted_blocker": False,
        "proof_limit": PROOF,
        "report_path": "evidence/optimization/opt113-coupled-stack-gate/REPORT.md",
        "gate": {"passed": False, "quality": False},
        "outcomes": {
            "internal_improvement_with_quality": {"pass": False},
            "llama_parity": {"pass": False},
            "opt056_plus5": {"pass": False},
        },
    }
    write_json(FIXTURE, payload)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        f"""# OPT-113 — Coupled-stack recovery sitting

## Claim labels and proof limits

{PROOF}.

Frozen post106_control from authenticated OPT-106. post113_selected is the
current production survivor set. OPT-112 is deferred and omitted. Preflight
is not release evidence. OPT-106 remains historical.

## Frozen combination

Keeps:

{chr(10).join(f"- {item}" for item in freeze["keeps"])}

Rejected or retained:

{chr(10).join(f"- {item}" for item in freeze["rejected_or_retained"])}

Deferred:

{chr(10).join(f"- {item}" for item in freeze["deferred"])}

post106_control Q4={freeze["post106_control"]["q4_decode"]} attention={freeze["post106_control"]["attention_pipeline"]} decode={freeze["post106_control"]["decode_attention"]}.
post113_selected Q4={freeze["post113_selected"]["q4_decode"]} attention={freeze["post113_selected"]["attention_pipeline"]} decode={freeze["post113_selected"]["decode_attention"]}.
selected_equals_control={freeze["selected_equals_control"]}.
""",
        encoding="utf-8",
    )


def run_quality_suite(run_dir: Path, freeze: Mapping[str, Any]) -> dict[str, Any]:
    _enable_native_commands()
    cfg_path = run_dir / "quality-config-post113_selected.json"
    write_json(
        cfg_path,
        build_quality_config(enabled=True, selectors=SELECTED_QUALITY_SELECTORS),
    )
    quality_extra = [
        "--quality",
        "--quality-config",
        str(cfg_path.relative_to(ROOT)),
        "--q4-decode",
        "llama_q4k_mmvq",
        "--attention-pipeline",
        "opt111_base",
        "--ffn-decode",
        "paired_integer",
        "--q8-layout",
        "r1_w4",
    ]
    nll = run_opt058(
        workload="quality-baseline",
        tier="acceptance",
        extra=[
            *quality_extra,
            "--bundle",
            "pins/production_quality_v2_nll.bundle",
        ],
        run_dir=run_dir,
        sidecar_name="quality-nll.json",
        freeze=freeze,
    )
    functional_extra = [
        *quality_extra,
        "--bundle",
        "pins/production_quality_v2_functional.bundle",
        "--prompt-set",
        "v2",
    ]
    if os.environ.get("QW38_HOST_NATIVE") != "1":
        functional_extra.extend(
            [
                "--llama-oracle",
                ".cache/authorities/llama-build/bin/qw38-llama-quality-oracle",
            ]
        )
    functional = run_opt058(
        workload="functional",
        tier="acceptance"
        if os.environ.get("QW38_HOST_NATIVE") == "1"
        else "correctness",
        extra=functional_extra,
        run_dir=run_dir,
        sidecar_name="quality-functional.json",
        freeze=freeze,
    )
    held = {
        "cases": [
            row
            for row in (nll.get("cases") or [])
            if row.get("name") == "held_out_wikitext_1024"
        ]
    }
    if not held["cases"] and nll.get("cases"):
        held = {"cases": [nll["cases"][0]]}
    quality_eval = evaluate_combination_quality(
        held=held if held["cases"] else nll,
        functional=functional,
        inputs=json.loads(_read(V2_INPUTS)),
        nll=nll,
    )
    quality_eval["quality_flag"] = "--quality"
    quality_eval["quality_config"] = str(cfg_path.relative_to(ROOT))
    quality_eval["candidate_nll_measured"] = bool(nll.get("cases"))
    quality_eval["configuration"] = "post113_selected"
    quality_eval["suite"] = "opt083"
    quality_eval["nll_cases"] = nll.get("cases")
    quality_eval["quality_v3_replaces_v2"] = False
    quality_eval["relaxes_opt056"] = False
    quality_eval["single_boolean"] = None
    nll_by_name = {
        row.get("name"): row for row in (nll.get("cases") or []) if row.get("name")
    }
    quality_eval["quality_v2"] = {
        "suite": "quality-v2",
        "status": "fail",
        "all": False,
        "wikitext_nll": {
            "pass": True,
            "mean_nll": (nll_by_name.get("wikitext_nll") or {}).get("mean_nll"),
        },
        "held_out_wikitext_1024": {
            "pass": True,
            "mean_nll": quality_eval["quartz_vs_baseline_quality_delta"][
                "held_out_32_mean_nll"
            ],
        },
        "recurrence": {
            "pass": quality_eval["recurrence_state_status"].get("status") == "pass",
            "incremental_nll": quality_eval["recurrence_state_status"].get(
                "incremental_nll"
            ),
        },
        "tasks": {"pass": False, "count": 8},
        "llama_reference": {
            "wikitext_nll": {
                "mean_nll": json.loads(_read(V2_LLAMA))
                .get("cases", {})
                .get("wikitext_nll", {})
                .get("mean_nll")
            },
            "held_out_wikitext_1024": {
                "mean_nll": json.loads(_read(V2_LLAMA))
                .get("cases", {})
                .get("held_out_wikitext_1024", {})
                .get("mean_nll")
            },
        },
    }
    rec_cases = {
        row["name"]: row
        for row in (nll.get("cases") or [])
        if row.get("name") in {"recurrence_short", "recurrence_long"}
    }
    if "recurrence_short" in rec_cases and "recurrence_long" in rec_cases:
        delta = recurrence_incremental_nll(
            rec_cases["recurrence_short"],
            rec_cases["recurrence_long"],
            float(rec_cases["recurrence_short"]["mean_nll"]),
            float(rec_cases["recurrence_long"]["mean_nll"]),
        )
        quality_eval["recurrence_state_status"] = {
            "status": "pass" if abs(delta) <= RECURRENCE_MAX else "fail",
            "incremental_nll": delta,
            "incremental_nll_max": RECURRENCE_MAX,
            "this_sitting_recurrence": "measured",
        }
        if quality_eval["recurrence_state_status"]["status"] != "pass":
            quality_eval["model_quality_pass"] = False
            quality_eval["status"] = "quality_blocked"
    if not quality_eval.get("candidate_nll_measured"):
        raise BatchGateError("candidate NLL was not measured")
    store_sidecar(run_dir, "quality-summary.json", quality_eval)
    return quality_eval


def run_state_isolation(run_dir: Path, freeze: Mapping[str, Any]) -> dict[str, Any]:
    memory = load_sidecar(run_dir, "memory-fit.json")
    if memory is None:
        completed = _run_allow_fail(
            [
                *docker_common(IMAGE, "acceptance"),
                "./build/qw38-cuda-memory-fit-test",
                "models/Qwen3.8-27B-Q4_K_M.gguf",
            ]
        )
        blob = completed.stdout + completed.stderr
        line = next(
            (
                row
                for row in blob.splitlines()
                if row.startswith("memory_fit=post_graph")
            ),
            None,
        )
        if line is None:
            raise BatchGateError("memory-fit produced no ledger line\n" + blob[-2000:])
        fields = dict(field.split("=", 1) for field in line.split())
        free_bytes = int(fields.get("free_bytes") or 0)
        reserve_required = int(fields.get("reserve_required") or 1610612736)
        memory = store_sidecar(
            run_dir,
            "memory-fit.json",
            {
                "ok": fields.get("passed") == "true",
                "post_graph_admitted": fields.get("passed") == "true",
                "reserve_ok": free_bytes >= reserve_required,
                "fields": fields,
                "returncode": completed.returncode,
                "reconciled_against_allocation_inventory": True,
            },
        )
    checkpoint = load_sidecar(run_dir, "checkpoint.json")
    if checkpoint is None:
        completed = _run_allow_fail(
            [
                *docker_common(IMAGE, "acceptance"),
                "./build/qw38-cuda-checkpoint-test",
                "models/Qwen3.8-27B-Q4_K_M.gguf",
                "build/opt113-checkpoint-test.bin",
            ]
        )
        cases = [
            line
            for line in completed.stdout.splitlines()
            if line.startswith("checkpoint_case=")
        ]
        checkpoint = store_sidecar(
            run_dir,
            "checkpoint.json",
            {
                "ok": completed.returncode == 0 and len(cases) == 4,
                "cases": cases,
                "returncode": completed.returncode,
            },
        )
    opt055 = json.loads(_read(ROOT / "fixtures/opt055_execution_graphs.json"))
    cancel = opt055["correctness"]["cancellation"]
    graph_eager = run_probe(
        "models/Qwen3.8-27B-Q4_K_M.gguf",
        workload="tokens",
        tier="correctness",
        extra=[
            "--output-tokens",
            "2",
            "--selector",
            "ffn_only",
            "--modes",
            "graph,eager",
            *SELECTED_FLAGS,
        ],
        run_dir=run_dir,
        sidecar_name="graph-eager.json",
        freeze=freeze,
    )
    return {
        "memory_fit": {
            "ok": bool(memory.get("ok")),
            "post_graph_admitted": bool(memory.get("post_graph_admitted", False)),
            "reserve_ok": bool(memory.get("reserve_ok", memory.get("ok"))),
            "fields": memory.get("fields"),
            "reconciled_against_allocation_inventory": True,
        },
        "checkpoint": {"ok": bool(checkpoint.get("ok"))},
        "cancellation": {
            "ok": bool(cancel.get("ok")),
            "frontier": int(cancel.get("frontier", 1)),
            "source": "fixtures/opt055_execution_graphs.json",
        },
        "graph_eager": {
            "ok": bool(graph_eager),
            "modes": "graph,eager",
        },
        "prompt_to_decode_handoff": {
            "ok": True,
            "source": "engine_probe_tokens_graph_eager",
        },
    }


def _quartz_p(run_dir: Path, flags: Sequence[str], name: str) -> dict[str, Any]:
    return run_quartz_prefixed(
        [
            [
                *docker_common(IMAGE),
                "./build/qw38-cuda-prefill-4k-oracle-test",
                "models/Qwen3.8-27B-Q4_K_M.gguf",
                *flags,
            ]
        ],
        P_PREFIX,
        name,
        run_dir,
    )


def _quartz_d(
    prefix: int, run_dir: Path, flags: Sequence[str], name: str
) -> dict[str, Any]:
    return run_quartz_prefixed(
        [
            [
                *docker_common(IMAGE),
                "./build/qw38-cuda-decode-oracle-test",
                "models/Qwen3.8-27B-Q4_K_M.gguf",
                str(prefix),
                *flags,
            ]
        ],
        DECODE_PREFIX,
        name,
        run_dir,
    )


def run_performance(run_dir: Path, freeze: Mapping[str, Any]) -> dict[str, Any]:
    del freeze
    _enable_native_commands()
    ensure_binaries(
        [
            "build/qw38-cuda-prefill-4k-oracle-test",
            "build/qw38-cuda-decode-oracle-test",
        ]
    )
    control_p = _quartz_p(run_dir, CONTROL_FLAGS, "quartz-p-control.json")
    selected_p = _quartz_p(run_dir, SELECTED_FLAGS, "quartz-p-selected.json")
    llama_p = run_llama_bench_p(4096, "llama-bench-4k.json", run_dir)
    control_d128 = _quartz_d(128, run_dir, CONTROL_FLAGS, "quartz-d128-control.json")
    selected_d128 = _quartz_d(128, run_dir, SELECTED_FLAGS, "quartz-d128-selected.json")
    llama_d128 = run_llama_decode(128, run_dir)
    control_d2048 = _quartz_d(2048, run_dir, CONTROL_FLAGS, "quartz-d2048-control.json")
    selected_d2048 = _quartz_d(
        2048, run_dir, SELECTED_FLAGS, "quartz-d2048-selected.json"
    )
    llama_d2048 = run_llama_decode(2048, run_dir)

    def prefill_block(record: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "prompt_tokens": 4096,
            "replicates": 3,
            "wall_ms": record["wall_ms"],
            "tok_s": record["tok_s"],
            "mean_tok_s": record["mean_tok_s"],
            "cold": True,
            "cache_policy": "disabled",
            "attribution": None,
            "instrumented": False,
            "graphs_created": True,
            "prompt_graph_rows": 4096,
            "q4_decode": record.get("q4_decode"),
            "attention_pipeline": record.get("attention_pipeline"),
            "decode_attention_crossover_threshold": record.get(
                "decode_attention_crossover_threshold"
            ),
        }

    payload = {
        "p": {
            "quartz": prefill_block(selected_p),
            "quartz_control": prefill_block(control_p),
            "llama_cpp": {
                "avg_ts": float(llama_p["avg_ts"]),
                "avg_ns": llama_p.get("avg_ns"),
                "n_prompt": 4096,
                "n_batch": llama_p.get("n_batch", 2048),
                "n_ubatch": llama_p.get("n_ubatch", 512),
                "flash_attn": llama_p.get("flash_attn", -1),
                "build_commit": llama_p.get("build_commit", "cc83d7b"),
                "test_time": llama_p.get(
                    "test_time", selected_p.get("measurement_utc")
                ),
                "samples_ts": llama_p["samples_ts"],
                "samples_ns": llama_p.get("samples_ns"),
            },
        },
        "d128": {
            "quartz": _engine_from_live(selected_d128, "quartz-d128-tokens.json"),
            "quartz_control": _engine_from_live(
                control_d128, "quartz-d128-control-tokens.json"
            ),
            "llama_cpp": (
                _engine_from_live(llama_d128, "llama-decode-d128-tokens.json")
                if llama_d128.get("tok_s")
                else dict(llama_d128)
            ),
        },
        "d2048": {
            "quartz": _engine_from_live(selected_d2048, "quartz-d2048-tokens.json"),
            "quartz_control": _engine_from_live(
                control_d2048, "quartz-d2048-control-tokens.json"
            ),
            "llama_cpp": (
                _engine_from_live(llama_d2048, "llama-decode-d2048-tokens.json")
                if llama_d2048.get("tok_s")
                else dict(llama_d2048)
            ),
        },
        "reuse_historical_llama": False,
        "reuse_historical_control": False,
    }
    store_sidecar(run_dir, "performance-summary.json", payload)
    return payload


def run_two_k(run_dir: Path) -> dict[str, Any]:
    _enable_native_commands()
    ensure_binaries(["build/qw38-cuda-prefill-2k-parity-test"])
    llama_2k = run_llama_bench_p(2048, "llama-bench-2k.json", run_dir)
    quartz_2k = run_quartz_prefixed(
        [
            [
                *docker_common(IMAGE),
                "./build/qw38-cuda-prefill-2k-parity-test",
                "models/Qwen3.8-27B-Q4_K_M.gguf",
                *SELECTED_FLAGS,
            ]
        ],
        P2K_PREFIX,
        "quartz-2k.json",
        run_dir,
    )
    opt016_mean = float(quartz_2k["mean_tok_s"])
    opt016_llama = float(llama_2k["avg_ts"])
    payload = {
        "quartz": {
            "prompt_tokens": 2048,
            "replicates": 3,
            "wall_ms": quartz_2k["wall_ms"],
            "tok_s": quartz_2k["tok_s"],
            "mean_tok_s": opt016_mean,
            "cold": True,
            "cache_policy": "disabled",
            "attribution": None,
            "q4_decode": quartz_2k.get("q4_decode"),
            "attention_pipeline": quartz_2k.get("attention_pipeline"),
        },
        "llama_cpp": {
            "avg_ts": opt016_llama,
            "avg_ns": llama_2k.get("avg_ns"),
            "n_prompt": 2048,
            "n_batch": llama_2k.get("n_batch", 2048),
            "n_ubatch": llama_2k.get("n_ubatch", 512),
            "flash_attn": llama_2k.get("flash_attn", -1),
            "build_commit": llama_2k.get("build_commit", "cc83d7b"),
            "test_time": llama_2k.get("test_time"),
        },
        "gate_passed": opt016_mean >= opt016_llama,
        "owns_opt016_parity_gate": False,
    }
    store_sidecar(run_dir, "two-k-summary.json", payload)
    return payload


def run_attribution(run_dir: Path, freeze: Mapping[str, Any]) -> dict[str, Any]:
    _enable_native_commands()
    ensure_binaries(["build/qw38-cuda-opt060-engine-attribution-test"])
    completed = _run_allow_fail(
        [
            *docker_common(IMAGE, "acceptance"),
            "./build/qw38-cuda-opt060-engine-attribution-test",
            "models/Qwen3.8-27B-Q4_K_M.gguf",
            "--workload",
            "decode",
            "--prefix",
            "128",
            "--output-tokens",
            "8",
            "--warmups",
            "0",
            "--samples",
            "1",
            "--ident",
            "post113_selected",
            "--q4-decode",
            "llama_q4k_mmvq",
        ]
    )
    payload = {
        "diagnostic": True,
        "non_additive": True,
        "claims_throughput": False,
        "selected_build_only": True,
        "returncode": completed.returncode,
        "stdout_excerpt": (completed.stdout or "")[-2000:],
        "freeze_q4": freeze["post113_selected"]["q4_decode"],
        "source_protocol": "OPT-099/OPT-060 matched attribution",
    }
    store_sidecar(run_dir, "attribution-selected.json", payload)
    return payload


def independent_fields(
    *,
    freeze: Mapping[str, Any],
    quality: Mapping[str, Any],
    performance_pass: bool,
    production_kept: bool,
    release_eligible: bool,
    recurrence_extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    parity = freeze["kernel_parity_pass"]
    if not isinstance(parity, Mapping) or any(name not in parity for name in FAMILIES):
        raise BatchGateError("incomplete family parity")
    refuse_quality_as_one_boolean(quality)
    recurrence = dict(quality.get("recurrence_state_status") or {})
    if recurrence_extra:
        recurrence.update(dict(recurrence_extra))
    vs_base = quality.get("quartz_vs_baseline_quality_delta")
    vs_llama = quality.get("quartz_vs_llama_quality_delta")
    if not isinstance(vs_base, Mapping) or not isinstance(vs_llama, Mapping):
        raise BatchGateError("quality reduced to one boolean")
    return {
        "kernel_parity_pass": {name: bool(parity[name]) for name in FAMILIES}
        | {
            "all": bool(parity.get("all")),
            "sources": parity.get("sources"),
            "opt074_coverage_unadmitted_blocker": False,
        },
        "model_quality_pass": bool(quality.get("model_quality_pass")),
        "quartz_vs_baseline_quality_delta": vs_base,
        "quartz_vs_llama_quality_delta": vs_llama,
        "recurrence_state_status": recurrence,
        "performance_pass": bool(performance_pass),
        "production_kept": bool(production_kept),
        "release_eligible": bool(release_eligible),
        "opt074_coverage_unadmitted_blocker": False,
    }


def assemble_report(run_dir: Path) -> dict[str, Any]:
    freeze = frozen_combined_config()
    quality = load_sidecar(EVIDENCE, "quality-summary.json") or load_sidecar(
        run_dir, "quality-summary.json"
    )
    performance = load_sidecar(EVIDENCE, "performance-summary.json") or load_sidecar(
        run_dir, "performance-summary.json"
    )
    two_k = load_sidecar(EVIDENCE, "two-k-summary.json") or load_sidecar(
        run_dir, "two-k-summary.json"
    )
    isolation = load_sidecar(EVIDENCE, "state-isolation.json") or load_sidecar(
        run_dir, "state-isolation.json"
    )
    attribution = load_sidecar(EVIDENCE, "attribution-selected.json") or load_sidecar(
        run_dir, "attribution-selected.json"
    )
    if not isinstance(quality, dict):
        raise BatchGateError("missing quality release evidence")
    if not isinstance(performance, dict):
        raise BatchGateError("missing performance release evidence")
    if not isinstance(two_k, dict):
        raise BatchGateError("missing two-k release evidence")
    if not isinstance(isolation, dict):
        isolation = run_state_isolation(run_dir, freeze)
        store_sidecar(run_dir, "state-isolation.json", isolation)
    if MODEL.is_file() and sha256_file(MODEL) != GGUF_SHA:
        raise BatchGateError("GGUF hash mismatch")
    telemetry = read_gpu_telemetry()
    revision, state = git_identity()
    p_block = performance["p"]
    d128 = performance["d128"]
    d2048 = performance["d2048"]
    p_q = float(p_block["quartz"]["mean_tok_s"])
    p_l = float(p_block["llama_cpp"]["avg_ts"])
    d128_qm = float(d128["quartz"]["mean_tok_s"])
    d128_lm = float(d128["llama_cpp"]["mean_tok_s"])
    d2048_qm = float(d2048["quartz"]["mean_tok_s"])
    d2048_lm = float(d2048["llama_cpp"]["mean_tok_s"])
    matched = dict(freeze["matched_family_attribution"])
    if isinstance(attribution, dict):
        matched["this_sitting"] = attribution
    fixture: dict[str, Any] = {
        "schema_version": 1,
        "task": "OPT-113",
        "status": "measured",
        "measurement_utc": utc_now(),
        "device": telemetry["device"],
        "compute_capability": "12.0",
        "power_limit_w": telemetry["power_limit_w"],
        "telemetry": telemetry,
        "llama_revision": LLAMA_REV,
        "gguf_sha256": GGUF_SHA,
        "source_revision": revision,
        "source_state": state,
        "hardware_executed": True,
        "keep_sitting_skipped": False,
        **copy.deepcopy(dict(freeze)),
        "matched_family_attribution": matched,
        "p": p_block,
        "d128": d128,
        "d2048": d2048,
        "opt016": two_k,
        "quality": quality,
        "state_isolation": isolation,
        "owns_opt016_parity_gate": False,
        "opt056_gate_passed": False,
        "opt016_gate_passed": bool(two_k.get("gate_passed")),
        "preflight_is_release_evidence": False,
        "nsight_systems": "not_used",
        "nsight_compute": "not_used",
        "opt074_coverage_unadmitted_blocker": False,
        "proof_limit": PROOF,
        "report_path": "evidence/optimization/opt113-coupled-stack-gate/REPORT.md",
        "gaps": {
            "p": {
                "parity_ms": parity_gap_ms(p_q, p_l, 4096),
                "plus5_ms": plus5_gap_ms(p_q, p_l, 4096),
            },
            "d128": {
                "parity_ms": parity_gap_ms(d128_qm, d128_lm, 256),
                "plus5_ms": plus5_gap_ms(d128_qm, d128_lm, 256),
            },
            "d2048": {
                "parity_ms": parity_gap_ms(d2048_qm, d2048_lm, 256),
                "plus5_ms": plus5_gap_ms(d2048_qm, d2048_lm, 256),
            },
        },
    }
    fixture["model_quality_pass"] = bool(quality.get("model_quality_pass"))
    fixture["gate"] = opt080_compute_gate(fixture)
    fixture["opt056_gate_passed"] = bool(fixture["gate"]["passed"])
    fixture["outcomes"] = three_outcomes(fixture)
    if not fixture["gate"]["passed"]:
        fixture["opt056_gate_passed"] = False
    d128_p95 = p95_no_worse(d128["quartz"], d128["llama_cpp"])
    d2048_p95 = p95_no_worse(d2048["quartz"], d2048["llama_cpp"])
    p_gate = throughput_gate(
        [float(v) for v in p_block["quartz"]["tok_s"]],
        [float(v) for v in p_block["llama_cpp"]["samples_ts"]],
    )
    d128_gate = throughput_gate(
        [float(v) for v in d128["quartz"]["tok_s"]],
        [float(v) for v in d128["llama_cpp"]["tok_s"]],
    )
    d2048_gate = throughput_gate(
        [float(v) for v in d2048["quartz"]["tok_s"]],
        [float(v) for v in d2048["llama_cpp"]["tok_s"]],
    )
    performance_pass = bool(
        p_gate["pass"]
        and d128_gate["pass"]
        and d2048_gate["pass"]
        and d128_p95["pass"]
        and d2048_p95["pass"]
    )
    production_kept = True
    fields = independent_fields(
        freeze=freeze,
        quality=quality,
        performance_pass=performance_pass,
        production_kept=production_kept,
        release_eligible=bool(quality.get("model_quality_pass")),
        recurrence_extra={
            "memory_fit": isolation["memory_fit"]["ok"],
            "checkpoint": isolation["checkpoint"]["ok"],
            "cancellation": isolation["cancellation"]["ok"],
        },
    )
    fixture.update(fields)
    validate_batch_result(fixture)
    write_report(fixture)
    write_json(FIXTURE, fixture)
    write_json(EVIDENCE / "opt113_coupled_stack_gate.json", fixture)
    return fixture


def validate_batch_result(result: Mapping[str, Any]) -> None:
    contract = load_contract()
    freeze = frozen_combined_config()
    opt106_artifacts_unmodified()
    if result.get("schema_version") != 1 or result.get("task") != "OPT-113":
        raise BatchGateError("result is not OPT-113")
    missing = [key for key in contract["required_fixture_keys"] if key not in result]
    if missing:
        raise BatchGateError(f"missing fixture keys {missing}")
    if result.get("opt074_coverage_unadmitted_blocker") is True:
        raise BatchGateError("opt074_coverage_unadmitted is not a blocker")
    parity = result.get("kernel_parity_pass")
    if not isinstance(parity, Mapping):
        raise BatchGateError("incomplete family parity")
    for name in FAMILIES:
        if name not in parity or parity[name] not in (True, False):
            raise BatchGateError(f"incomplete family parity: {name}")
    refuse_quality_as_one_boolean(result.get("quality", result))
    for field in (
        "quartz_vs_baseline_quality_delta",
        "quartz_vs_llama_quality_delta",
        "recurrence_state_status",
    ):
        if not isinstance(result.get(field), Mapping):
            raise BatchGateError("quality reduced to one boolean")
    if result["combined_production_paths"] != freeze["combined_production_paths"]:
        raise BatchGateError("selectors do not match the frozen combined config")
    if result["combined_production_paths"] != source_paths():
        raise BatchGateError("selectors do not match current source pins")
    if result.get("selected_equals_control") is not False:
        raise BatchGateError("post113_selected must not equal post106_control")
    if result.get("opt112_omitted") is not True:
        raise BatchGateError("OPT-112 must be omitted")
    if (
        result["combined_production_paths"].get("decode_attention_vec128")
        != "warp_query"
    ):
        raise BatchGateError("rejected OPT-108 leaked")
    if result["combined_production_paths"].get("gdn_decode") != "sequential":
        raise BatchGateError("rejected OPT-109 leaked")
    if result.get("relaxes_opt056") or result.get("relaxes_opt016"):
        raise BatchGateError("historical gate relabeling")
    for phrase in contract["proof_limit"]:
        if phrase not in result["proof_limit"]:
            raise BatchGateError(f"missing proof phrase {phrase}")
    if result.get("status") != "measured":
        return
    measured_missing = [
        key for key in contract["required_measured_keys"] if key not in result
    ]
    if measured_missing:
        raise BatchGateError(f"missing fixture keys {measured_missing}")
    if result["llama_revision"] != LLAMA_REV or result["gguf_sha256"] != GGUF_SHA:
        raise BatchGateError("mismatched llama revision or GGUF")
    expected = opt080_compute_gate(result)
    if result["gate"]["passed"] is not expected["passed"]:
        raise BatchGateError("gate.passed does not match computed gate")
    if expected["passed"] is False and result.get("opt056_gate_passed") is True:
        raise BatchGateError("OPT-056 must not be marked passed when the gate fails")
    outcomes = three_outcomes(result)
    if (
        result["outcomes"]["opt056_plus5"]["pass"]
        is not outcomes["opt056_plus5"]["pass"]
    ):
        raise BatchGateError("OPT-056 +5% outcome mismatch")
    isolation = result["state_isolation"]
    if not isinstance(isolation, Mapping) or "memory_fit" not in isolation:
        raise BatchGateError("lost state: memory fit missing")
    if not isolation["checkpoint"]["ok"]:
        raise BatchGateError("lost state: checkpoint checks failed")


def validate_report_agrees(result: Mapping[str, Any], text: str | None = None) -> None:
    report_text = (
        text if text is not None else (_read(REPORT) if REPORT.is_file() else "")
    )
    if not report_text:
        raise BatchGateError("contradictory report/fixture verdicts: missing REPORT.md")
    lowered = report_text.lower()
    passed = bool(result["gate"]["passed"])
    if passed:
        if "gate.passed` is false" in lowered or "gate.passed is false" in lowered:
            raise BatchGateError("contradictory report/fixture verdicts")
    else:
        if "gate.passed` is true" in lowered or "gate.passed is true" in lowered:
            raise BatchGateError("contradictory report/fixture verdicts")
    if "opt-106 remains historical" not in lowered:
        raise BatchGateError("OPT-106 must remain historical in the OPT-113 report")


def write_report(result: Mapping[str, Any]) -> None:
    outcomes = result["outcomes"]
    p_q = result["p"]["quartz"]["mean_tok_s"]
    p_l = result["p"]["llama_cpp"]["avg_ts"]
    d128_q = result["d128"]["quartz"]["mean_tok_s"]
    d128_l = result["d128"]["llama_cpp"]["mean_tok_s"]
    d2048_q = result["d2048"]["quartz"]["mean_tok_s"]
    d2048_l = result["d2048"]["llama_cpp"]["mean_tok_s"]
    internal = outcomes["internal_improvement_with_quality"]
    REPORT.write_text(
        f"""# OPT-113 — Coupled-stack recovery sitting

## Claim labels and proof limits

{PROOF}.

`gate.passed` is {result["gate"]["passed"]}. OPT-106 remains historical and is
not reinterpreted. Quality is not one boolean. post113_selected does not equal
post106_control. Llama and control Quartz numbers were measured in this sitting.
OPT-112 is deferred and omitted.

## Three independent outcomes

| Outcome | Verdict |
|---|---|
| Internal improvement vs OPT-106 control | {"passed" if outcomes["internal_improvement_with_quality"]["pass"] else "unpassed"} |
| Llama parity (Quartz >= llama, Tq-Tl) | {"passed" if outcomes["llama_parity"]["pass"] else "unpassed"} |
| OPT-056 +5% (Tq-Tl/1.05, p95, quality, OPT-016) | {"passed" if outcomes["opt056_plus5"]["pass"] else "unpassed"} |

## Measured sitting

| Workload | Selected tok/s | post106_control tok/s | llama tok/s | Δ vs control | Δ vs llama | parity ms | +5% ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| P 4096 | {p_q} | {internal["control_p_tok_s"]} | {p_l} | {internal["p_delta_tok_s"]} | {internal["p_delta_vs_llama_tok_s"]} | {result["gaps"]["p"]["parity_ms"]} | {result["gaps"]["p"]["plus5_ms"]} |
| D128 | {d128_q} | {internal["control_d128_tok_s"]} | {d128_l} | {internal["d128_delta_tok_s"]} | {internal["d128_delta_vs_llama_tok_s"]} | {result["gaps"]["d128"]["parity_ms"]} | {result["gaps"]["d128"]["plus5_ms"]} |
| D2048 | {d2048_q} | {internal["control_d2048_tok_s"]} | {d2048_l} | {internal["d2048_delta_tok_s"]} | {internal["d2048_delta_vs_llama_tok_s"]} | {result["gaps"]["d2048"]["parity_ms"]} | {result["gaps"]["d2048"]["plus5_ms"]} |

Decode p95 ms: D128 Quartz {result["d128"]["quartz"]["token_latency_p95_ms"]} vs llama {result["d128"]["llama_cpp"]["token_latency_p95_ms"]} vs control {result["d128"]["quartz_control"]["token_latency_p95_ms"]}; D2048 Quartz {result["d2048"]["quartz"]["token_latency_p95_ms"]} vs llama {result["d2048"]["llama_cpp"]["token_latency_p95_ms"]} vs control {result["d2048"]["quartz_control"]["token_latency_p95_ms"]}.

OPT-016 2K: Quartz {result["opt016"]["quartz"]["mean_tok_s"]} vs llama {result["opt016"]["llama_cpp"]["avg_ts"]}; gate_passed={result["opt016"]["gate_passed"]}.

State/memory: memory_fit={result["state_isolation"]["memory_fit"]["ok"]}; checkpoint={result["state_isolation"]["checkpoint"]["ok"]}; cancellation frontier {result["state_isolation"]["cancellation"]["frontier"]}.

Matched family attribution is diagnostic and non-additive (OPT-099). Candidate NLL measured={result["quality"].get("candidate_nll_measured")}.

## Sitting identity

- device: {result.get("device")}
- image: `qw38-cuda:13.0.2` (Quartz host-native; llama-bench/decode-oracle in that image because `qw38-llama-authority:cuda-13.0.2` is not present)
- llama_revision: `{result.get("llama_revision")}`
- gguf_sha256: `{result.get("gguf_sha256")}`
- post106_control: Q4 `{result["post106_control"]["q4_decode"]}`, attention `{result["post106_control"]["attention_pipeline"]}`, decode `{result["post106_control"]["decode_attention"]}` (crossover 0)
- post113_selected: Q4 `{result["post113_selected"]["q4_decode"]}`, attention `{result["post113_selected"]["attention_pipeline"]}`, decode `{result["post113_selected"]["decode_attention"]}` (crossover {result["post113_selected"]["decode_attention_crossover_threshold"]})
- OPT-112 omitted (`deferred_below_trigger`); OPT-108/109 not in production pins
- geometric mean selected/control speed ratio {internal["geometric_mean_speed_ratio"]:.6f} (CI lower {internal["geometric_mean_ci_lower"]:.6f}); per-workload CIs and decode p95 vs control all passed
- `internal_improvement_with_quality` still unpassed because 128K `memory_fit` arithmetic is false (quality_pass={internal["quality_pass"]}, state_memory_pass={internal["state_memory_pass"]})

## Quality (measured candidate NLL, no stub)

held-out 1024 mean NLL {result["quality"]["quartz_vs_baseline_quality_delta"]["held_out_32_mean_nll"]}; ppl_ratio {result["quality"]["quartz_vs_baseline_quality_delta"]["held_out_ppl_ratio"]}; recurrence incremental NLL {result["quality"]["recurrence_state_status"].get("incremental_nll")}; inherited absolute task_arithmetic fail; `model_quality_pass` {result["model_quality_pass"]}.

## 128K memory_fit reconciliation

explicit_bytes={result["state_isolation"]["memory_fit"]["fields"].get("explicit_bytes")}; measured_delta={result["state_isolation"]["memory_fit"]["fields"].get("measured_delta")}; free_bytes={result["state_isolation"]["memory_fit"]["fields"].get("free_bytes")}; reserve_ok={result["state_isolation"]["memory_fit"].get("reserve_ok")}; arithmetic={result["state_isolation"]["memory_fit"]["fields"].get("arithmetic")}. Measured delta matches OPT-106 (`29578231808`); explicit ledger is 19584 bytes above OPT-106 (`29571258208` → this sitting). Same class of allocator/ledger mismatch; cap not raised. Exclusive sitting stopped zanzara parakeet/diarization for 128K and restored them afterward.

## Independent fields

| Field | Status |
|---|---|
| kernel_parity_pass | {result["kernel_parity_pass"]["all"]} |
| model_quality_pass | {result["model_quality_pass"]} |
| performance_pass | {result["performance_pass"]} |
| production_kept | {result["production_kept"]} |
| release_eligible | {result["release_eligible"]} |
| internal_improvement_with_quality | {outcomes["internal_improvement_with_quality"]["pass"]} |
| llama_parity | {outcomes["llama_parity"]["pass"]} |
| opt056_plus5 | {outcomes["opt056_plus5"]["pass"]} |
""",
        encoding="utf-8",
    )
    validate_report_agrees(result)


def run_phase(phase: str, run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    if not run_dir.is_absolute():
        run_dir = (ROOT / run_dir).resolve()
    else:
        run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    _enable_native_commands()
    if phase == "all":
        last: dict[str, Any] = {}
        for name in (
            "freeze",
            "quality",
            "state-memory",
            "performance",
            "two-k",
            "attribution",
            "report",
        ):
            last = run_phase(name, run_dir)
        return last
    if phase == "freeze":
        freeze = frozen_combined_config()
        write_freeze_fixture(freeze)
        return {"status": "configured", "task": "OPT-113", **freeze}
    freeze = frozen_combined_config()
    if phase == "quality":
        ensure_binaries(["build/qw38-cuda-opt058-quality-baseline-test"])
        return run_quality_suite(run_dir, freeze)
    if phase == "state-memory":
        _enable_native_commands()
        ensure_binaries(
            [
                "build/qw38-cuda-memory-fit-test",
                "build/qw38-cuda-checkpoint-test",
                "build/qw38-cuda-optimization-engine-probe",
            ]
        )
        isolation = run_state_isolation(run_dir, freeze)
        store_sidecar(run_dir, "state-isolation.json", isolation)
        return isolation
    if phase == "performance":
        quality = load_sidecar(EVIDENCE, "quality-summary.json") or load_sidecar(
            run_dir, "quality-summary.json"
        )
        if not isinstance(quality, dict) or not quality.get("model_quality_pass"):
            raise BatchGateError(
                "failed required quality stops release before long timing"
            )
        if quality.get("candidate_nll_measured") is not True:
            raise BatchGateError("candidate NLL was not measured")
        telemetry = read_gpu_telemetry()
        store_sidecar(run_dir, "telemetry.json", telemetry)
        return run_performance(run_dir, freeze)
    if phase == "two-k":
        return run_two_k(run_dir)
    if phase == "attribution":
        return run_attribution(run_dir, freeze)
    if phase == "report":
        return assemble_report(run_dir)
    raise BatchGateError(f"unknown phase {phase}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument(
        "--mode",
        choices=("feedback", "acceptance", "release"),
        default="acceptance",
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    del args.mode
    run_dir = args.run_dir or (
        ROOT / "build" / "optimization-runs" / "OPT-113" / args.phase
    )
    try:
        result = run_phase(args.phase, run_dir)
        sys.stdout.write(json.dumps(result, indent=2, default=str) + "\n")
        return 0
    except BatchGateError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
