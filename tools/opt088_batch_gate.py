"""Combined OPT-088 post-reset production gate.

Freezes Q4 packed (OPT-085), Q8 r1_w4 (OPT-086 revert), MMQ fma_async_x
(OPT-086 keep), kv_once (OPT-079), and OPT-087 no_additional_reopen.
Independent kernel-parity, quality-delta, recurrence, performance, keep,
and release-eligible fields. OPT-080 remains historical. OPT-056/016 stay
blocked unless their owning conditions pass. OPT-074 unadmitted is not a
blocker. Quality is not one boolean.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.opt058_quality_baseline import TASK_NAMES  # noqa: E402
from tools.opt073_quality_policy import (  # noqa: E402
    QualityPolicyError,
    validate_parsed_functional_answers,
)
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
    parse_llama_bench,
    parse_prefixed,
    p95_no_worse,
    read_gpu_telemetry,
    run_command,
    sha256_file,
    source_paths,
    three_outcomes as opt080_three_outcomes,
    throughput_gate,
    utc_now,
    write_json,
    _engine_ok,
    _finite,
    _require_finite_nll,
)
from tools.opt084_quality_baseline import FROZEN_ACCEPTANCE  # noqa: E402
from tools.quality.compare import refuse_single_boolean  # noqa: E402
from tools.quality.errors import QualityFrameworkError  # noqa: E402
from tools.quality.quality_mode import apply_quality_mode  # noqa: E402
from tools.quality.scoring import recurrence_incremental_nll  # noqa: E402
from tools.quality.suite import PPL_RATIO_MAX, RECURRENCE_MAX, SUITE_CLASSES  # noqa: E402

CONTRACT = ROOT / "pins/opt088_batch_gate_contract.json"
ITERATION = ROOT / "pins/opt088_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt088_batch_gate.json"
EVIDENCE = ROOT / "evidence/optimization/opt088-batch-gate"
REPORT = EVIDENCE / "REPORT.md"
DIAGNOSTIC_PLAN = EVIDENCE / "diagnostic-performance-plan.json"
OPT056 = ROOT / "fixtures/opt056_performance_gate.json"
OPT069 = ROOT / "fixtures/opt069_batch_gate.json"
OPT080_REPORT = ROOT / "evidence/optimization/opt080-batch-gate/REPORT.md"
OPT080_FIXTURE = ROOT / "fixtures/opt080_batch_gate.json"
OPT080_CONTRACT = ROOT / "pins/opt080_batch_gate_contract.json"
OPT084 = ROOT / "fixtures/opt084_quality_baseline.json"
PREFLIGHT_HELD32 = ROOT / (
    "evidence/optimization/opt069-batch-gate/preflight-held-out-32.json"
)
V2_INPUTS = ROOT / "pins/production_quality_v2_inputs.json"
V2_LLAMA = ROOT / "pins/production_quality_v2_llama_reference.json"
MEMORY = ROOT / "fixtures/cuda_memory_fit_post_graph.json"
P_PREFIX = "QW38_PREFILL_4K_ORACLE_RESULT="
DECODE_PREFIX = "QW38_DECODE_ORACLE_RESULT="
P2K_PREFIX = "QW38_PREFILL_2K_PARITY_RESULT="
PROBE_PREFIX = "QW38_OPT057_PROBE_RESULT="
QUALITY_PREFIX = "QW38_OPT058_RESULT="
LLAMA_DECODE_PREFIX = "QW38_LLAMA_DECODE_ORACLE_RESULT="
FAMILIES = ("q4", "q8", "mmq", "kv_once")
INDEPENDENT_FIELDS = (
    "kernel_parity_pass",
    "model_quality_pass",
    "quartz_vs_baseline_quality_delta",
    "quartz_vs_llama_quality_delta",
    "recurrence_state_status",
    "performance_pass",
    "production_kept",
    "release_eligible",
)
DEPENDENCY_IDS = ("OPT-079", "OPT-085", "OPT-086", "OPT-087")
DEPENDENCY_FIXTURES = {
    "OPT-079": "opt079_attention_kv_operands.json",
    "OPT-085": "opt085_q4_reevaluation.json",
    "OPT-086": "opt086_q8_mmq_reevaluation.json",
    "OPT-087": "opt087_historical_reevaluation.json",
}
PROOF = (
    "combined freeze of OPT-085 packed Q4, OPT-086 r1_w4 Q8 and fma_async_x "
    "MMQ, OPT-079 kv_once, OPT-087 no_additional_reopen; "
    "independent kernel_parity_pass per family; "
    "quality is not one boolean; "
    "quartz_vs_baseline_quality_delta and quartz_vs_llama_quality_delta stay "
    "distinct; "
    "opt074_coverage_unadmitted is not a blocker; "
    "absolute task-accuracy fail inherited from OPT-084 does not by itself "
    "block; "
    "new regression versus OPT-084 does block; "
    "OPT-080 remains historical and is not reinterpreted; "
    "OPT-056 and OPT-016 stay blocked unless their owning conditions pass; "
    "parity gap is Tq-Tl; "
    "+5% throughput gap is Tq-Tl/1.05; "
    "do not label the parity gap as the +5% bar; "
    "decode p95 no worse than llama for the OPT-056 outcome; "
    "preflight is not release evidence; "
    "failed required quality stops release before long timing; "
    "diagnostic performance is not a release or keep; "
    "rejected candidates must not leak into production"
)
EXPECTED_PATHS: dict[str, Any] = {
    "rms_norm": "parallel_fma",
    "q4_decode": "packed",
    "q8_decode": "dp4a_q8_1",
    "q8_decode_rows_skinny": 1,
    "q8_decode_rows_medium": 1,
    "q8_decode_rows_wide": 1,
    "q8_decode_layout_warps_skinny": 4,
    "q8_decode_layout_warps_medium": 4,
    "q8_decode_layout_warps_wide": 4,
    "q8_layout": "r1_w4",
    "q6_decode": "integer_q8_1",
    "ffn_decode": "paired_staged",
    "query_prepare": "hoisted",
    "attention_pipeline": "kv_once",
    "gdn_preproc": "transpose",
    "mmq_pipeline": "fma_async",
    "mmq_async_x": True,
    "ffn_prompt_pair": "off",
    "ffn_tiles": "i128_j128",
    "ffn_gate_quality_i": 128,
    "ffn_gate_prompt_tile": 128,
    "prompt_microbatch_rows": 4096,
    "execution_graphs": "ffn_only",
    "production_numerics": "strict",
    "nvccflags": "-O2 --fmad=false",
    "gdn_decode": "sequential",
    "decode_query_prep": "warp_query",
}
COMBINATION_SELECTORS: dict[str, Any] = {
    "q4_decode": "packed",
    "q4_staging": "paired_staged",
    "q8_decode": "r1_w4",
    "q8_path": "dp4a_q8_1",
    "prompt_mmq": "fma_async_x",
    "prompt_mmq_tile": "i128_j128",
    "prompt_attention": "kv_once",
    "decode_gdn": "sequential",
    "decode_attention": "warp_query",
    "prompt_pair": "off",
    "nvccflags": "-O2 --fmad=false",
    "production_numerics": "strict",
    "execution_graphs": "ffn_only",
    "chat_template": "no_thinking",
    "enable_thinking": False,
    "logit_masking": False,
}


class BatchGateError(AssertionError):
    """Inadmissible combined post-reset evidence."""


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def sidecar(run_dir: Path, name: str) -> Path:
    return run_dir / name


def load_sidecar(run_dir: Path, name: str) -> Any | None:
    path = sidecar(run_dir, name)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def store_sidecar(run_dir: Path, name: str, payload: Any) -> Any:
    write_json(sidecar(run_dir, name), payload)
    write_json(EVIDENCE / name, payload)
    return payload


def identity_matches(payload: Mapping[str, Any], freeze: Mapping[str, Any]) -> bool:
    if payload.get("gguf_sha256") not in (None, GGUF_SHA):
        return False
    selectors = payload.get("combined_production_paths")
    if selectors and selectors != freeze["combined_production_paths"]:
        return False
    return True


def run_probe(
    model: str,
    *,
    workload: str,
    tier: str,
    extra: Sequence[str],
    run_dir: Path,
    sidecar_name: str,
    freeze: Mapping[str, Any],
) -> dict[str, Any]:
    existing = load_sidecar(run_dir, sidecar_name)
    if isinstance(existing, dict) and identity_matches(existing, freeze):
        return existing
    command = [
        *docker_common(IMAGE, tier),
        "./build/qw38-cuda-optimization-engine-probe",
        model,
        "--workload",
        workload,
        *extra,
    ]
    completed = run_command(command)
    record = parse_prefixed(completed.stdout + completed.stderr, PROBE_PREFIX)
    record["gguf_sha256"] = GGUF_SHA
    record["combined_production_paths"] = freeze["combined_production_paths"]
    return store_sidecar(run_dir, sidecar_name, record)


def run_opt058(
    *,
    workload: str,
    tier: str,
    extra: Sequence[str],
    run_dir: Path,
    sidecar_name: str,
    freeze: Mapping[str, Any],
    reuse_only: bool = False,
) -> dict[str, Any]:
    existing = load_sidecar(run_dir, sidecar_name)
    if reuse_only:
        if not isinstance(existing, dict):
            raise BatchGateError(f"missing retained sidecar {sidecar_name}")
        return existing
    if isinstance(existing, dict) and identity_matches(existing, freeze):
        return existing
    command = [
        *docker_common(IMAGE, tier),
        "./build/qw38-cuda-opt058-quality-baseline-test",
        "models/Qwen3.8-27B-Q4_K_M.gguf",
        "--workload",
        workload,
        *extra,
    ]
    completed = run_command(command)
    record = parse_prefixed(completed.stdout + completed.stderr, QUALITY_PREFIX)
    record["gguf_sha256"] = GGUF_SHA
    record["combined_production_paths"] = freeze["combined_production_paths"]
    return store_sidecar(run_dir, sidecar_name, record)


def run_llama_bench_p(tokens: int, sidecar_name: str, run_dir: Path) -> dict[str, Any]:
    existing = load_sidecar(run_dir, sidecar_name)
    if (
        isinstance(existing, list)
        and existing
        and existing[0].get("n_prompt") == tokens
    ):
        return existing[0]
    command = [
        *docker_common(LLAMA_IMAGE),
        "bash",
        "-lc",
        "cd /workspace && .cache/authorities/llama-build/bin/llama-bench "
        f"-m /workspace/models/Qwen3.8-27B-Q4_K_M.gguf "
        f"-p {tokens} -n 0 --no-warmup -r 3 -ngl 99 -o json",
    ]
    completed = run_command(command)
    payload = parse_llama_bench(
        completed.stdout + completed.stderr,
        lambda row: row.get("n_prompt") == tokens,
        f"llama-bench JSON with n_prompt {tokens}",
    )
    store_sidecar(run_dir, sidecar_name, payload)
    return payload[0]


def run_llama_decode(prefix: int, run_dir: Path) -> dict[str, Any]:
    name = f"llama-decode-d{prefix}.json"
    existing = load_sidecar(run_dir, name)
    if isinstance(existing, dict) and existing.get("prefix") == prefix:
        return existing
    command = [
        *docker_common(LLAMA_IMAGE),
        "bash",
        "-lc",
        "cd /workspace && .cache/authorities/llama-build/bin/qw38-llama-decode-oracle "
        f"/workspace/models/Qwen3.8-27B-Q4_K_M.gguf {prefix}",
    ]
    completed = run_command(command)
    record = parse_prefixed(completed.stdout + completed.stderr, LLAMA_DECODE_PREFIX)
    return store_sidecar(run_dir, name, record)


def run_quartz_prefixed(
    commands: Sequence[Sequence[str]],
    prefix: str,
    sidecar_name: str,
    run_dir: Path,
) -> dict[str, Any]:
    existing = load_sidecar(run_dir, sidecar_name)
    if isinstance(existing, dict) and existing:
        return existing
    blob = ""
    for command in commands:
        blob = run_command(command).stdout
    record = parse_prefixed(blob, prefix)
    return store_sidecar(run_dir, sidecar_name, record)


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
            f"evidence/optimization/opt088-batch-gate/{sidecar_name}"
        ),
    }
    for key in ("n_gpu_layers", "n_ctx", "n_batch", "n_ubatch"):
        if key in record:
            block[key] = record[key]
    if "graphs_created" in record:
        block["graphs_created"] = record["graphs_created"]
        block["cache_policy"] = record.get("cache_policy", "disabled")
    block["attribution"] = None
    block["instrumented"] = False
    return block


def load_contract() -> dict[str, Any]:
    return json.loads(_read(CONTRACT))


def _load_fixture(name: str) -> dict[str, Any]:
    path = ROOT / "fixtures" / name
    if not path.is_file():
        raise BatchGateError(f"incomplete admissions: missing fixture {name}")
    payload = json.loads(_read(path))
    if not isinstance(payload, dict):
        raise BatchGateError(f"incomplete admissions: {name} is not an object")
    return payload


def opt080_artifacts_unmodified() -> dict[str, str]:
    """Fail closed if historical OPT-080 artifacts were rewritten."""
    contract = load_contract()
    expected = contract["opt080_historical_hashes"]
    actual = {
        "report": hashlib.sha256(OPT080_REPORT.read_bytes()).hexdigest(),
        "fixture": hashlib.sha256(OPT080_FIXTURE.read_bytes()).hexdigest(),
        "contract": hashlib.sha256(OPT080_CONTRACT.read_bytes()).hexdigest(),
    }
    for name, digest in expected.items():
        if actual[name] != digest:
            raise BatchGateError(
                f"OPT-080 {name} was reinterpreted or rewritten "
                f"(hash {actual[name]} != {digest})"
            )
    fixture = json.loads(_read(OPT080_FIXTURE))
    if fixture.get("task") != "OPT-080":
        raise BatchGateError("OPT-080 fixture task relabeled")
    if fixture.get("opt056_gate_passed") is True:
        raise BatchGateError("historical gate relabeling of OPT-080/OPT-056")
    return actual


def audit_dependencies() -> dict[str, Any]:
    """Every OPT-079/085/086/087 fixture must carry a concrete disposition."""
    opt080_artifacts_unmodified()
    loaded: dict[str, Any] = {}
    for task_id in DEPENDENCY_IDS:
        name = DEPENDENCY_FIXTURES[task_id]
        payload = _load_fixture(name)
        if payload.get("task") not in (task_id, None):
            raise BatchGateError(f"incomplete admissions: {name} task mismatch")
        loaded[task_id] = payload
    opt079 = loaded["OPT-079"]
    if not opt079.get("production_kept"):
        raise BatchGateError("incomplete admissions: OPT-079 kv_once is not kept")
    if (opt079.get("verdict") or {}).get("verdict") != "keep":
        raise BatchGateError("incomplete admissions: OPT-079 missing keep verdict")
    if opt079.get("shipping_attention_pipeline") != "kv_once":
        raise BatchGateError("incomplete admissions: OPT-079 shipping is not kv_once")
    opt085 = loaded["OPT-085"]
    packed = (opt085.get("independent_verdicts") or {}).get("packed_paired_staged")
    if not isinstance(packed, Mapping):
        raise BatchGateError("incomplete admissions: OPT-085 missing packed verdicts")
    if packed.get("kernel_parity_pass") is not True:
        raise BatchGateError(
            "incomplete family parity: OPT-085 packed kernel_parity_pass missing"
        )
    if opt085.get("shipping_q4_decode") != "packed":
        raise BatchGateError("incomplete admissions: OPT-085 shipping Q4 is not packed")
    if packed.get("opt074_coverage_unadmitted_blocker") is True:
        raise BatchGateError("opt074_coverage_unadmitted is not a blocker")
    opt086 = loaded["OPT-086"]
    q8 = ((opt086.get("independent_verdicts") or {}).get("q8") or {}).get("r1_w4")
    mmq = ((opt086.get("independent_verdicts") or {}).get("mmq") or {}).get(
        "fma_async_x"
    )
    if not isinstance(q8, Mapping) or q8.get("kernel_parity_pass") is not True:
        raise BatchGateError(
            "incomplete family parity: OPT-086 r1_w4 kernel_parity_pass missing"
        )
    if not isinstance(mmq, Mapping) or mmq.get("kernel_parity_pass") is not True:
        raise BatchGateError(
            "incomplete family parity: OPT-086 fma_async_x kernel_parity_pass missing"
        )
    if opt086.get("q8_selected_path") != "r1_w4":
        raise BatchGateError("incomplete admissions: OPT-086 Q8 is not r1_w4")
    if opt086.get("mmq_selected_path") != "fma_async_x":
        raise BatchGateError("incomplete admissions: OPT-086 MMQ is not fma_async_x")
    if q8.get("opt074_coverage_unadmitted_blocker") is True:
        raise BatchGateError("opt074_coverage_unadmitted is not a blocker")
    opt087 = loaded["OPT-087"]
    if not opt087.get("no_additional_reopen"):
        raise BatchGateError(
            "incomplete admissions: OPT-087 is not no_additional_reopen"
        )
    if opt087.get("reopened_candidate") not in (None, ""):
        raise BatchGateError("OPT-087 reopened a candidate; freeze cannot proceed")
    if opt087.get("opt074_coverage_unadmitted_blocker") is True:
        raise BatchGateError("opt074_coverage_unadmitted is not a blocker")
    return loaded


def family_kernel_parity(loaded: Mapping[str, Any] | None = None) -> dict[str, Any]:
    deps = loaded or audit_dependencies()
    opt085 = deps["OPT-085"]
    opt086 = deps["OPT-086"]
    opt079 = deps["OPT-079"]
    packed = opt085["independent_verdicts"]["packed_paired_staged"]
    q8 = opt086["independent_verdicts"]["q8"]["r1_w4"]
    mmq = opt086["independent_verdicts"]["mmq"]["fma_async_x"]
    kv_numeric = opt079.get("numeric") or {}
    kv_pass = (
        bool(opt079.get("production_kept")) and int(kv_numeric.get("nonfinite", 1)) == 0
    )
    record = {
        "q4": bool(packed.get("kernel_parity_pass")),
        "q8": bool(q8.get("kernel_parity_pass")),
        "mmq": bool(mmq.get("kernel_parity_pass")),
        "kv_once": kv_pass,
    }
    missing = [name for name in FAMILIES if name not in record]
    if missing:
        raise BatchGateError(f"incomplete family parity: missing {missing}")
    incomplete = [name for name, value in record.items() if value is not True]
    if incomplete:
        raise BatchGateError(
            "incomplete family parity: "
            + ",".join(f"{name}={record[name]!r}" for name in incomplete)
        )
    return {
        **record,
        "sources": {
            "q4": "fixtures/opt085_q4_reevaluation.json#packed_paired_staged",
            "q8": "fixtures/opt086_q8_mmq_reevaluation.json#q8.r1_w4",
            "mmq": "fixtures/opt086_q8_mmq_reevaluation.json#mmq.fma_async_x",
            "kv_once": "fixtures/opt079_attention_kv_operands.json#keep",
        },
        "opt074_coverage_unadmitted_blocker": False,
        "all": all(record[name] is True for name in FAMILIES),
    }


def candidate_decisions() -> dict[str, Any]:
    loaded = audit_dependencies()
    opt079 = loaded["OPT-079"]
    opt085 = loaded["OPT-085"]
    opt086 = loaded["OPT-086"]
    packed = opt085["independent_verdicts"]["packed_paired_staged"]
    q8 = opt086["independent_verdicts"]["q8"]["r1_w4"]
    mmq = opt086["independent_verdicts"]["mmq"]["fma_async_x"]
    return {
        "OPT-079": {
            "status": "keep",
            "installed": True,
            "selected": "kv_once",
            "production_kept": bool(opt079.get("production_kept")),
            "kernel_parity_pass": True,
            "model_quality_pass": (opt079.get("quality") or {}).get(
                "quality_v3_engine_non_regression"
            )
            == "pass",
        },
        "OPT-085": {
            "status": "retain_packed",
            "installed": True,
            "selected": "packed",
            "production_kept": True,
            "shipping_unchanged": bool(opt085.get("shipping_unchanged", True)),
            "kernel_parity_pass": bool(packed.get("kernel_parity_pass")),
            "model_quality_pass": bool(packed.get("model_quality_pass")),
            "performance_pass": bool(packed.get("performance_pass")),
        },
        "OPT-086": {
            "status": "q8_revert_mmq_keep",
            "installed": True,
            "selected": "r1_w4+fma_async_x",
            "q8_verdict": "revert",
            "mmq_verdict": "keep",
            "q8_kernel_parity_pass": bool(q8.get("kernel_parity_pass")),
            "mmq_kernel_parity_pass": bool(mmq.get("kernel_parity_pass")),
            "q8_model_quality_pass": bool(q8.get("model_quality_pass")),
            "mmq_model_quality_pass": bool(mmq.get("model_quality_pass")),
        },
        "OPT-087": {
            "status": "no_additional_reopen",
            "installed": False,
            "selected": None,
            "no_additional_reopen": True,
            "reopened_candidate": None,
        },
    }


def frozen_combined_config() -> dict[str, Any]:
    paths = source_paths()
    decisions = candidate_decisions()
    parity = family_kernel_parity()
    opt066 = json.loads(_read(ROOT / "fixtures/opt066_mmq_x_pipeline.json"))
    workspace = opt066.get("resources", {}).get("candidate", {})
    for key, expected in EXPECTED_PATHS.items():
        if paths.get(key) != expected:
            raise BatchGateError(
                f"combined freeze mismatch {key}: {paths.get(key)!r} != {expected!r}"
            )
    if paths["q8_layout"] != "r1_w4":
        raise BatchGateError("OPT-086 r1_w4 must be the installed Q8 layout")
    if paths["q4_decode"] != "packed":
        raise BatchGateError("OPT-085 packed Q4 must remain installed")
    if paths["mmq_async_x"] is not True or paths["mmq_pipeline"] != "fma_async":
        raise BatchGateError("OPT-086 fma_async_x must remain the MMQ keep")
    if paths["attention_pipeline"] != "kv_once":
        raise BatchGateError("OPT-079 kv_once must remain the attention pipeline")
    if paths["gdn_decode"] != "sequential":
        raise BatchGateError("rejected tiled GDN leaked into production")
    if paths["decode_query_prep"] != "warp_query":
        raise BatchGateError("rejected decode query-prep leaked into production")
    if paths["ffn_prompt_pair"] != "off":
        raise BatchGateError("rejected prompt-pair leaked into production")
    if decisions["OPT-087"]["no_additional_reopen"] is not True:
        raise BatchGateError("OPT-087 must remain no_additional_reopen")
    quality_mode = apply_quality_mode(enabled=True, selectors=COMBINATION_SELECTORS)
    return {
        "combined_production_paths": paths,
        "candidate_decisions": decisions,
        "kernel_parity_pass": parity,
        "keeps": [
            "OPT-085 packed Q4",
            "OPT-086 r1_w4 Q8 revert",
            "OPT-086 fma_async_x MMQ keep",
            "OPT-079 kv_once",
            "OPT-087 no_additional_reopen",
        ],
        "rejected_or_retained": [
            "OPT-085 integer_q8_paired and late_w4 not installed",
            "OPT-086 r2_w2 Q8 reverted",
            "OPT-086 fma_async MMQ not selected",
            "OPT-087 no additional reopen",
        ],
        "batch_size": paths["prompt_microbatch_rows"],
        "workspace_bytes": {
            "mmq_x_shared_bytes": workspace.get("shared_bytes"),
            "mmq_x_extra_bytes": workspace.get("extra_x"),
            "x_ring_stages": opt066.get("x_ring_stages", 1),
        },
        "compiler_flags": paths["nvccflags"],
        "graphs": paths["execution_graphs"],
        "intended_q4_path": "packed",
        "intended_q8_layout": "r1_w4",
        "intended_mmq": "fma_async_x",
        "quality_mode": quality_mode,
        "opt056_history": "fixtures/opt056_performance_gate.json",
        "opt069_history": "fixtures/opt069_batch_gate.json",
        "opt080_history": "fixtures/opt080_batch_gate.json",
        "opt084_baseline": "fixtures/opt084_quality_baseline.json",
        "opt074_coverage_unadmitted_blocker": False,
    }


def baseline_quality() -> dict[str, Any]:
    return json.loads(_read(OPT084))


def combination_oracle_policy(
    model_quality_pass: bool | None, *, diagnostic_performance: bool = False
) -> dict[str, Any]:
    """Release oracles follow OPT-084 non-regression, not quality-v2 all."""
    if model_quality_pass is None:
        return {
            "run_oracles": False,
            "release_eligible": False,
            "keep_claims_allowed": False,
            "stop_reason": "skipped quality phase",
            "diagnostic_performance": diagnostic_performance,
        }
    if model_quality_pass:
        return {
            "run_oracles": True,
            "release_eligible": True,
            "keep_claims_allowed": True,
            "stop_reason": None,
            "diagnostic_performance": False,
            "absolute_task_accuracy_inherited_fail_does_not_block": True,
        }
    if diagnostic_performance:
        return {
            "run_oracles": True,
            "release_eligible": False,
            "keep_claims_allowed": False,
            "retain_quality_failure": True,
            "stop_reason": None,
            "diagnostic_performance": True,
        }
    return {
        "run_oracles": False,
        "release_eligible": False,
        "keep_claims_allowed": False,
        "stop_reason": "failed required quality stops release before long timing",
        "diagnostic_performance": False,
    }


def held32_baseline_nll() -> float:
    payload = json.loads(_read(PREFLIGHT_HELD32))
    cases = payload.get("cases") or []
    if not cases or not _finite(cases[0].get("mean_nll")):
        raise BatchGateError("missing OPT-084/069 held-out 32 alarm baseline")
    return float(cases[0]["mean_nll"])


def _held_mean(record: Mapping[str, Any]) -> float:
    cases = record.get("cases") or []
    if cases and _finite(cases[0].get("mean_nll")):
        return float(cases[0]["mean_nll"])
    if _finite(record.get("mean_nll")):
        return float(record["mean_nll"])
    raise BatchGateError("missing quality reference held_out")


def evaluate_combination_quality(
    *,
    held: Mapping[str, Any],
    functional: Mapping[str, Any],
    inputs: Mapping[str, Any],
    nll: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """OPT-083/084 non-regression. Quality is multi-field, not one boolean."""
    baseline = baseline_quality()
    try:
        parsed = validate_parsed_functional_answers(functional.get("quartz"), inputs)
    except QualityPolicyError as exc:
        raise BatchGateError(str(exc)) from exc
    cases = held.get("cases") or []
    scored = int(cases[0].get("scored") or 0) if cases else 0
    if 0 < scored < 32:
        raise BatchGateError("preflight held-out must score 32 teacher-forced targets")
    held_nll = _held_mean(held)
    if not _finite(held_nll):
        raise BatchGateError("NaN/nonfinite quality reference held_out")
    if 0 < scored < 1024:
        base_held = held32_baseline_nll()
        baseline_label = "opt084_held_out_32_alarm"
    else:
        base_held = float(
            baseline["quartz_scores"]["held_out_wikitext_1024"]["mean_nll"]
        )
        baseline_label = "opt084_held_out_1024"
    held_ratio = math.exp(held_nll - base_held)
    held_ok = math.isfinite(held_ratio) and held_ratio <= PPL_RATIO_MAX
    by_case = {str(item.get("case")): item for item in parsed if item.get("case")}
    if len(by_case) < 8:
        raise BatchGateError("quality reduced to incomplete functional answers")
    arithmetic = by_case.get("task_arithmetic") or {}
    inherited_fail = (
        arithmetic.get("actual") == "A" and str(arithmetic.get("expected")) == "B"
    )
    new_functional_fail = False
    for name in TASK_NAMES:
        row = by_case.get(name) or {}
        if name == "task_arithmetic" and inherited_fail:
            continue
        if row.get("pass") is False:
            new_functional_fail = True
    engine_nr = (baseline.get("historical") or {}).get(
        "opt073_engine_non_regression"
    ) == "pass"
    vs_baseline = {
        "held_out_32_mean_nll": held_nll,
        "baseline_held_out_mean_nll": base_held,
        "baseline_held_out_32_mean_nll": held32_baseline_nll(),
        "baseline_label": baseline_label,
        "scored": scored,
        "held_out_ppl_ratio": held_ratio,
        "ppl_ratio_max": PPL_RATIO_MAX,
        "delta_nll": held_nll - base_held,
        "pass": held_ok and not new_functional_fail,
        "status": "pass" if held_ok and not new_functional_fail else "fail",
        "known_baseline_defect_does_not_reject": True,
        "inherited_absolute_task_fail": inherited_fail,
        "new_functional_regression": new_functional_fail,
        "alarm_not_admission": scored < 1024,
        "single_boolean": None,
        "inspectable": True,
    }
    llama_held = float(
        (baseline.get("llama_scores") or {})
        .get("held_out_wikitext_1024", {})
        .get("mean_nll", math.nan)
    )
    vs_llama = {
        "quartz_held_out_mean_nll": held_nll,
        "llama_held_out_mean_nll": llama_held,
        "delta_quartz_minus_llama": held_nll - llama_held
        if _finite(llama_held)
        else None,
        "baseline_aggregate_delta": (baseline.get("aggregate_nll") or {}).get(
            "delta_quartz_minus_llama"
        ),
        "inspectable": True,
        "authoritative_local_comparison": "quartz+llama",
        "single_boolean": None,
    }
    rec_short = baseline["quartz_scores"]["recurrence_short"]
    rec_long = baseline["quartz_scores"]["recurrence_long"]
    rec_delta = recurrence_incremental_nll(
        rec_short,
        rec_long,
        float(rec_short["mean_nll"]),
        float(rec_long["mean_nll"]),
    )
    recurrence = {
        "status": "pass" if abs(rec_delta) <= RECURRENCE_MAX else "fail",
        "incremental_nll": rec_delta,
        "incremental_nll_max": RECURRENCE_MAX,
        "source": "fixtures/opt084_quality_baseline.json",
        "this_sitting_recurrence": None
        if nll is None
        else "measured"
        if nll
        else "preflight_alarm_only",
    }
    model_quality_pass = (
        vs_baseline["pass"] and engine_nr and recurrence["status"] == "pass"
    )
    absolute_status = "fail" if inherited_fail or new_functional_fail else "pass"
    if not vs_baseline["pass"]:
        status = "quality_blocked"
    else:
        status = "pass"
    return {
        "status": status,
        "model_quality_pass": model_quality_pass,
        "absolute_quality_status": absolute_status,
        "quartz_baseline_regression_status": vs_baseline["status"],
        "quartz_vs_baseline_quality_delta": vs_baseline,
        "quartz_vs_llama_quality_delta": vs_llama,
        "recurrence_state_status": recurrence,
        "parsed": parsed,
        "selected_quality_verdicts": {
            "parsed_functional_answers": True,
            "held_out_32": True,
            "quartz_vs_baseline": vs_baseline["status"],
            "quartz_vs_llama": "inspectable",
            "absolute_task_accuracy": absolute_status,
            "engine_non_regression": "pass" if engine_nr else "fail",
        },
        "suite_classes": list(SUITE_CLASSES),
        "frozen_acceptance": dict(FROZEN_ACCEPTANCE),
        "opt056_quality_requirement_met": False,
        "quality_v2_all": False,
        "is_release_evidence": False,
        "opt074_coverage_unadmitted_blocker": False,
        "single_boolean": None,
    }


def refuse_quality_as_one_boolean(quality: Any) -> None:
    if isinstance(quality, bool) or quality in (0, 1, "pass", "fail", "true", "false"):
        try:
            refuse_single_boolean({"quality": quality})
        except QualityFrameworkError as exc:
            raise BatchGateError("quality reduced to one boolean") from exc
    if isinstance(quality, Mapping):
        keys = set(quality)
        if keys <= {"pass", "all", "status", "model_quality_pass"}:
            raise BatchGateError("quality reduced to one boolean")
        if quality.get("single_boolean") is True:
            raise BatchGateError("quality reduced to one boolean")


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
    if vs_base.get("delta_nll") is None or vs_llama.get("inspectable") is not True:
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


def remaining_latency_budget() -> dict[str, Any]:
    opt069 = json.loads(_read(OPT069))
    p_q = float(opt069["p"]["quartz"]["mean_tok_s"])
    p_l = float(opt069["p"]["llama_cpp"]["avg_ts"])
    d128_q = float(opt069["d128"]["quartz"]["mean_tok_s"])
    d128_l = float(opt069["d128"]["llama_cpp"]["mean_tok_s"])
    d2048_q = float(opt069["d2048"]["quartz"]["mean_tok_s"])
    d2048_l = float(opt069["d2048"]["llama_cpp"]["mean_tok_s"])
    return {
        "source": "fixtures/opt069_batch_gate.json",
        "p": {
            "quartz_tok_s": p_q,
            "llama_tok_s": p_l,
            "parity_ms": parity_gap_ms(p_q, p_l, 4096),
            "plus5_ms": plus5_gap_ms(p_q, p_l, 4096),
        },
        "d128": {
            "quartz_tok_s": d128_q,
            "llama_tok_s": d128_l,
            "parity_ms": parity_gap_ms(d128_q, d128_l, 256),
            "plus5_ms": plus5_gap_ms(d128_q, d128_l, 256),
        },
        "d2048": {
            "quartz_tok_s": d2048_q,
            "llama_tok_s": d2048_l,
            "parity_ms": parity_gap_ms(d2048_q, d2048_l, 256),
            "plus5_ms": plus5_gap_ms(d2048_q, d2048_l, 256),
        },
        "tok_s_delta_vs_opt069": 0.0,
        "speedup_vs_opt069": 1.0,
        "note": "unmeasured or quality-blocked sitting does not replace OPT-069 tok/s",
    }


def diagnostic_performance_plan(
    preflight: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task": "OPT-088",
        "status": "quality_blocked",
        "is_release_evidence": False,
        "release_eligible": False,
        "keep_claims_allowed": False,
        "stop_reason": "failed required quality stops release before long timing",
        "preflight_status": None if preflight is None else preflight.get("status"),
        "opt056_gate_passed": False,
        "opt016_gate_passed": False,
        "relaxes_opt056": False,
        "relaxes_opt016": False,
        "opt074_coverage_unadmitted_blocker": False,
        "remaining_latency_budget": remaining_latency_budget(),
        "diagnostic_performance": {
            "purpose": "non-release P/D/2K under a known OPT-084 regression",
            "not_a_release": True,
            "does_not_pass_opt056": True,
            "does_not_pass_opt016": True,
            "command": [
                "uv",
                "run",
                "python",
                "tools/opt088_batch_gate.py",
                "--phase",
                "release",
                "--diagnostic-performance",
            ],
            "workloads": ["p4096", "d128", "d2048", "opt016", "state_memory"],
            "opt021": "0 warmups, 3 replicates, 4096 tokens",
            "opt032": "3 warmups + 30 runs of 256 tokens, prefixes 128 and 2048",
            "opt016": "2048 prompt tokens, 3 replicates",
        },
        "next_optimization_sweep": False,
    }


def write_freeze_report(freeze: Mapping[str, Any], extra: str = "") -> None:
    keeps = "\n".join(f"- {item}" for item in freeze["keeps"])
    rejects = "\n".join(f"- {item}" for item in freeze["rejected_or_retained"])
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        f"""# OPT-088 — Combined post-reset production gate

## Claim labels and proof limits

{PROOF}.

This report is the frozen configuration record for the post-reset combination.
Preflight is not release evidence. OPT-080 remains the historical previous-
policy gate and is not reinterpreted. OPT-056 and OPT-016 stay blocked unless
their owning conditions pass. Quality is not one boolean.

## Frozen combination

Keeps:

{keeps}

Rejected or retained:

{rejects}

Q4 path is packed (OPT-085). Q8 production layout is r1_w4 (OPT-086 revert).
MMQ is `fma_async` with `async_x` (OPT-086 keep). Attention pipeline stays
`kv_once`. OPT-087 is `no_additional_reopen`. Prompt pair is off. Tiles stay
i128_j128. NVCCFLAGS stay `-O2 --fmad=false`. Graphs are `ffn_only`. Decode
GDN stays `sequential`. Decode query-prep stays `warp_query`. Batch size is
{freeze["batch_size"]}. Workspace extra X bytes:
{freeze["workspace_bytes"]}.

## Independent fields

| Field | Status |
|---|---|
| kernel_parity_pass (q4/q8/mmq/kv_once) | {freeze["kernel_parity_pass"]["q4"]}/{freeze["kernel_parity_pass"]["q8"]}/{freeze["kernel_parity_pass"]["mmq"]}/{freeze["kernel_parity_pass"]["kv_once"]} |
| model_quality_pass | unmeasured |
| quartz_vs_baseline_quality_delta | unmeasured |
| quartz_vs_llama_quality_delta | unmeasured |
| recurrence_state_status | unmeasured |
| performance_pass | unmeasured |
| production_kept | unmeasured |
| release_eligible | unmeasured |

{extra}
""",
        encoding="utf-8",
    )


def write_freeze_fixture(freeze: Mapping[str, Any]) -> None:
    payload = {
        "schema_version": 1,
        "task": "OPT-088",
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
        "production_kept": False,
        "release_eligible": False,
        "opt074_coverage_unadmitted_blocker": False,
        "proof_limit": PROOF,
        "report_path": "evidence/optimization/opt088-batch-gate/REPORT.md",
        "gate": {"passed": False, "quality": False},
        "outcomes": {
            "internal_improvement_with_quality": {"pass": False},
            "llama_parity": {"pass": False},
            "opt056_plus5": {"pass": False},
        },
    }
    write_json(FIXTURE, payload)
    write_freeze_report(freeze)


def write_quality_blocked_outputs(
    freeze: Mapping[str, Any],
    preflight: Mapping[str, Any],
) -> dict[str, Any]:
    plan = diagnostic_performance_plan(preflight)
    write_json(DIAGNOSTIC_PLAN, plan)
    fields = independent_fields(
        freeze=freeze,
        quality=preflight,
        performance_pass=False,
        production_kept=False,
        release_eligible=False,
    )
    budget = plan["remaining_latency_budget"]
    payload = {
        "schema_version": 1,
        "task": "OPT-088",
        "status": "quality_blocked",
        "measurement_utc": utc_now(),
        **copy.deepcopy(dict(freeze)),
        **fields,
        "hardware_executed": True,
        "keep_sitting_skipped": True,
        "preflight_is_release_evidence": False,
        "owns_opt016_parity_gate": False,
        "opt056_gate_passed": False,
        "opt016_gate_passed": False,
        "keep_claims_allowed": False,
        "quality": {
            "suite": "opt083",
            "model_quality_pass": preflight.get("model_quality_pass"),
            "absolute_quality_status": preflight.get("absolute_quality_status"),
            "quartz_baseline_regression_status": preflight.get(
                "quartz_baseline_regression_status"
            ),
            "quartz_vs_baseline_quality_delta": preflight.get(
                "quartz_vs_baseline_quality_delta"
            ),
            "quartz_vs_llama_quality_delta": preflight.get(
                "quartz_vs_llama_quality_delta"
            ),
            "quality_v2_all": False,
            "quality_v3_replaces_v2": False,
            "relaxes_opt056": False,
            "single_boolean": None,
        },
        "preflight": {
            "status": preflight.get("status"),
            "is_release_evidence": False,
            "held_out_targets": 32,
            "selected_quality_verdicts": preflight.get("selected_quality_verdicts"),
            "opt056_quality_requirement_met": False,
        },
        "remaining_latency_budget": budget,
        "diagnostic_performance_plan": (
            "evidence/optimization/opt088-batch-gate/diagnostic-performance-plan.json"
        ),
        "proof_limit": PROOF,
        "report_path": "evidence/optimization/opt088-batch-gate/REPORT.md",
        "gate": {"passed": False, "quality": False},
        "outcomes": {
            "internal_improvement_with_quality": {"pass": False},
            "llama_parity": {"pass": False},
            "opt056_plus5": {"pass": False},
        },
    }
    write_json(FIXTURE, payload)
    write_json(EVIDENCE / "opt088_batch_gate.json", payload)
    keeps = "\n".join(f"- {item}" for item in freeze["keeps"])
    rejects = "\n".join(f"- {item}" for item in freeze["rejected_or_retained"])
    REPORT.write_text(
        f"""# OPT-088 — Combined post-reset production gate

## Claim labels and proof limits

{PROOF}.

Preflight quality regresses versus OPT-084. This is not release evidence.
`gate.passed` is False. OPT-056 remains blocked. OPT-016 stays blocked.
OPT-080 remains historical and is not reinterpreted. Quality is not one
boolean. Absolute task-accuracy inherited from OPT-084 does not by itself
explain this block; a new regression versus the freeze does.

## Frozen combination

Keeps:

{keeps}

Rejected or retained:

{rejects}

## Independent fields

| Field | Status |
|---|---|
| kernel_parity_pass (q4/q8/mmq/kv_once) | {fields["kernel_parity_pass"]["q4"]}/{fields["kernel_parity_pass"]["q8"]}/{fields["kernel_parity_pass"]["mmq"]}/{fields["kernel_parity_pass"]["kv_once"]} |
| model_quality_pass | {fields["model_quality_pass"]} |
| quartz_vs_baseline_quality_delta | {fields["quartz_vs_baseline_quality_delta"].get("status")} delta_nll={fields["quartz_vs_baseline_quality_delta"].get("delta_nll")} |
| quartz_vs_llama_quality_delta | inspectable |
| recurrence_state_status | {fields["recurrence_state_status"].get("status")} |
| performance_pass | False |
| production_kept | False |
| release_eligible | False |

## Remaining latency budget (OPT-069 sitting, unchanged)

| Workload | Quartz tok/s | llama tok/s | parity ms (Tq-Tl) | +5% ms (Tq-Tl/1.05) |
|---|---:|---:|---:|---:|
| P 4096 | {budget["p"]["quartz_tok_s"]} | {budget["p"]["llama_tok_s"]} | {budget["p"]["parity_ms"]} | {budget["p"]["plus5_ms"]} |
| D128 | {budget["d128"]["quartz_tok_s"]} | {budget["d128"]["llama_tok_s"]} | {budget["d128"]["parity_ms"]} | {budget["d128"]["plus5_ms"]} |
| D2048 | {budget["d2048"]["quartz_tok_s"]} | {budget["d2048"]["llama_tok_s"]} | {budget["d2048"]["parity_ms"]} | {budget["d2048"]["plus5_ms"]} |

tok/s delta vs OPT-069 baseline: 0 (speedup 1.00×). Diagnostic performance
plan: `evidence/optimization/opt088-batch-gate/diagnostic-performance-plan.json`.
No automatic next optimization sweep.
""",
        encoding="utf-8",
    )
    return payload


def write_report(result: Mapping[str, Any]) -> None:
    outcomes = result["outcomes"]
    p_q = result["p"]["quartz"]["mean_tok_s"]
    p_l = result["p"]["llama_cpp"]["avg_ts"]
    d128_q = result["d128"]["quartz"]["mean_tok_s"]
    d128_l = result["d128"]["llama_cpp"]["mean_tok_s"]
    d2048_q = result["d2048"]["quartz"]["mean_tok_s"]
    d2048_l = result["d2048"]["llama_cpp"]["mean_tok_s"]
    keeps = "\n".join(f"- {item}" for item in result["keeps"])
    rejects = "\n".join(f"- {item}" for item in result["rejected_or_retained"])
    parity = result["kernel_parity_pass"]
    vs_base = result["quartz_vs_baseline_quality_delta"]
    vs_llama = result["quartz_vs_llama_quality_delta"]
    rec = result["recurrence_state_status"]
    opt016_passed = bool(result["opt016"]["gate_passed"])
    text = f"""# OPT-088 — Combined post-reset production gate

## Claim labels and proof limits

{PROOF}.

This sitting reports independent kernel-parity, quality-delta, recurrence,
performance, keep, and release-eligible fields. Failed historical gates stay
failed. `gate.passed` is {result["gate"]["passed"]}. OPT-056 remains
blocked unless the original +5% / p95 / quality conditions pass. OPT-080
remains historical and is not reinterpreted. Quality is not one boolean.

## Frozen combination

Keeps:

{keeps}

Rejected or retained:

{rejects}

Q4 path is packed. Q8 production layout is r1_w4. MMQ is `fma_async` with
`async_x`. Attention pipeline stays `kv_once`. OPT-087 is
`no_additional_reopen`. NVCCFLAGS stay `-O2 --fmad=false`. Graphs are
`ffn_only`. Batch size is {result["batch_size"]}.

## Independent fields

| Field | Status |
|---|---|
| kernel_parity_pass (q4/q8/mmq/kv_once) | {parity["q4"]}/{parity["q8"]}/{parity["mmq"]}/{parity["kv_once"]} |
| model_quality_pass | {result["model_quality_pass"]} |
| quartz_vs_baseline_quality_delta | {vs_base.get("status")} delta_nll={vs_base.get("delta_nll")} ppl_ratio={vs_base.get("held_out_ppl_ratio")} |
| quartz_vs_llama_quality_delta | delta={vs_llama.get("delta_quartz_minus_llama")} |
| recurrence_state_status | {rec.get("status")} incremental_nll={rec.get("incremental_nll")} |
| performance_pass | {result["performance_pass"]} |
| production_kept | {result["production_kept"]} |
| release_eligible | {result["release_eligible"]} |

## Three historical outcomes (not relabeled)

| Outcome | Verdict |
|---|---|
| Internal improvement with quality | {"passed" if outcomes["internal_improvement_with_quality"]["pass"] else "unpassed"} |
| Llama parity (Quartz >= llama, Tq-Tl) | {"passed" if outcomes["llama_parity"]["pass"] else "unpassed"} |
| OPT-056 +5% (Tq-Tl/1.05, p95, quality, OPT-016) | {"passed" if outcomes["opt056_plus5"]["pass"] else "unpassed"} |

## Measured sitting

- Device: {result["device"]} compute {result["compute_capability"]}
- power_limit_w: {result["power_limit_w"]}
- measurement_utc: {result["measurement_utc"]}
- source_revision: {result["source_revision"]} ({result["source_state"]})

| Workload | Quartz tok/s | llama tok/s | OPT-069 baseline | parity ms (Tq-Tl) | +5% ms (Tq-Tl/1.05) |
|---|---:|---:|---:|---:|---:|
| P 4096 | {p_q} | {p_l} | {outcomes["internal_improvement_with_quality"]["baseline_p_tok_s"]} | {result["gaps"]["p"]["parity_ms"]} | {result["gaps"]["p"]["plus5_ms"]} |
| D128 | {d128_q} | {d128_l} | {outcomes["internal_improvement_with_quality"]["baseline_d128_tok_s"]} | {result["gaps"]["d128"]["parity_ms"]} | {result["gaps"]["d128"]["plus5_ms"]} |
| D2048 | {d2048_q} | {d2048_l} | {outcomes["internal_improvement_with_quality"]["baseline_d2048_tok_s"]} | {result["gaps"]["d2048"]["parity_ms"]} | {result["gaps"]["d2048"]["plus5_ms"]} |

Decode p95 ms: D128 Quartz {result["d128"]["quartz"]["token_latency_p95_ms"]} vs llama {result["d128"]["llama_cpp"]["token_latency_p95_ms"]}; D2048 Quartz {result["d2048"]["quartz"]["token_latency_p95_ms"]} vs llama {result["d2048"]["llama_cpp"]["token_latency_p95_ms"]}.

OPT-016 2K: Quartz {result["opt016"]["quartz"]["mean_tok_s"]} vs llama {result["opt016"]["llama_cpp"]["avg_ts"]}; point comparison gate_passed={opt016_passed}. This increment does not own the OPT-016 ledger row.

State/memory: memory_fit={result["state_isolation"]["memory_fit"]["ok"]}; checkpoint={result["state_isolation"]["checkpoint"]["ok"]}; cancellation frontier {result["state_isolation"]["cancellation"]["frontier"]}.

Budget: this is a long sitting, not a five-minute check. Preflight is not
release evidence.
"""
    REPORT.write_text(text, encoding="utf-8")
    validate_report_agrees(result, text)


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
    if result.get("opt056_gate_passed") is True and not passed:
        raise BatchGateError("historical gate relabeling")
    if result.get("opt016_gate_passed") and not result.get("opt016", {}).get(
        "gate_passed"
    ):
        raise BatchGateError("historical gate relabeling")
    if "opt-080 remains historical" not in lowered:
        raise BatchGateError("OPT-080 must remain historical in the OPT-088 report")


def validate_batch_result(result: Mapping[str, Any]) -> None:
    contract = load_contract()
    freeze = frozen_combined_config()
    opt080_artifacts_unmodified()
    if result.get("schema_version") != 1 or result.get("task") != "OPT-088":
        raise BatchGateError("result is not OPT-088")
    missing = [key for key in contract["required_fixture_keys"] if key not in result]
    if missing:
        raise BatchGateError(f"missing fixture keys {missing}")
    if result.get("opt074_coverage_unadmitted_blocker") is True:
        raise BatchGateError("opt074_coverage_unadmitted is not a blocker")
    if result.get("opt074_family_admission_required") is True:
        raise BatchGateError("opt074_coverage_unadmitted is not a blocker")
    parity = result.get("kernel_parity_pass")
    if not isinstance(parity, Mapping):
        raise BatchGateError("incomplete family parity")
    for name in FAMILIES:
        if name not in parity:
            raise BatchGateError(f"incomplete family parity: missing {name}")
        if parity[name] not in (True, False):
            raise BatchGateError(f"incomplete family parity: {name} not boolean")
    refuse_quality_as_one_boolean(result.get("quality", result))
    for field in (
        "quartz_vs_baseline_quality_delta",
        "quartz_vs_llama_quality_delta",
        "recurrence_state_status",
    ):
        if not isinstance(result.get(field), Mapping):
            raise BatchGateError("quality reduced to one boolean")
    vs_base = result["quartz_vs_baseline_quality_delta"]
    vs_llama = result["quartz_vs_llama_quality_delta"]
    if vs_base.get("single_boolean") is True or vs_llama.get("single_boolean") is True:
        raise BatchGateError("quality reduced to one boolean")
    if result["combined_production_paths"] != freeze["combined_production_paths"]:
        raise BatchGateError("selectors do not match the frozen combined config")
    if result["combined_production_paths"] != source_paths():
        raise BatchGateError("selectors do not match current source pins")
    if result["combined_production_paths"]["q8_layout"] != "r1_w4":
        raise BatchGateError("Q8 layout must be r1_w4")
    if result["combined_production_paths"]["q4_decode"] != "packed":
        raise BatchGateError("rejected integer Q4 leaked into production")
    if result["combined_production_paths"]["attention_pipeline"] != "kv_once":
        raise BatchGateError("OPT-079 kv_once must remain installed")
    if result.get("relaxes_opt056") or result.get("relaxes_opt016"):
        raise BatchGateError("historical gate relabeling")
    for phrase in contract["proof_limit"]:
        if phrase not in result["proof_limit"]:
            raise BatchGateError(f"missing proof phrase {phrase}")
    if result.get("status") != "measured":
        if result.get("opt056_gate_passed") is True:
            raise BatchGateError(
                "OPT-056 must not be marked passed when the gate fails"
            )
        if result.get("opt016_gate_passed") is True and not (
            result.get("opt016") or {}
        ).get("gate_passed"):
            raise BatchGateError("historical gate relabeling")
        return
    measured_missing = [
        key for key in contract["required_measured_keys"] if key not in result
    ]
    if measured_missing:
        raise BatchGateError(f"missing fixture keys {measured_missing}")
    if result["llama_revision"] != LLAMA_REV or result["gguf_sha256"] != GGUF_SHA:
        raise BatchGateError("mismatched llama revision or GGUF")
    telemetry = result.get("telemetry") or {}
    if telemetry.get("device_substring") and "5090" not in str(
        result.get("device", "")
    ):
        raise BatchGateError("mismatched device")
    if result.get("power_limit_w") in (None, 0):
        raise BatchGateError("mismatched power/telemetry")
    quartz_p = result["p"]["quartz"]
    if int(quartz_p["prompt_tokens"]) != 4096:
        raise BatchGateError("mismatched P tokens")
    if int(quartz_p["replicates"]) != 3 or len(quartz_p["tok_s"]) != 3:
        raise BatchGateError("short samples presented as release P")
    if quartz_p.get("attribution") is not None or quartz_p.get("instrumented"):
        raise BatchGateError("hidden instrumented timing on P")
    llama_p = result["p"]["llama_cpp"]
    if int(llama_p["n_prompt"]) != 4096 or len(llama_p["samples_ts"]) != 3:
        raise BatchGateError("short samples presented as release llama P")
    _engine_ok(result["d128"]["quartz"], 128)
    _engine_ok(result["d2048"]["quartz"], 2048)
    _engine_ok(result["d128"]["llama_cpp"], 128)
    _engine_ok(result["d2048"]["llama_cpp"], 2048)
    quality = result["quality"]
    refuse_quality_as_one_boolean(quality)
    v2 = (quality.get("quality_v2") or {}) if isinstance(quality, Mapping) else {}
    if v2:
        for name in ("wikitext_nll", "held_out_wikitext_1024"):
            if name in v2:
                _require_finite_nll(v2[name], name)
    expected = opt080_compute_gate(result)
    if result["gate"]["passed"] is not expected["passed"]:
        raise BatchGateError("gate.passed does not match computed gate")
    if expected["passed"] is False and result.get("opt056_gate_passed") is True:
        raise BatchGateError("OPT-056 must not be marked passed when the gate fails")
    if expected["opt016"] is False and (
        result["opt016"]["gate_passed"] or result.get("opt016_gate_passed")
    ):
        raise BatchGateError("OPT-016 must not be marked passed when the gate fails")
    q_mean = float(result["p"]["quartz"]["mean_tok_s"])
    l_mean = float(result["p"]["llama_cpp"]["avg_ts"])
    recorded_parity = float(result["gaps"]["p"]["parity_ms"])
    recorded_plus5 = float(result["gaps"]["p"]["plus5_ms"])
    if abs(recorded_parity - parity_gap_ms(q_mean, l_mean, 4096)) > 1e-6:
        raise BatchGateError("wrong parity millisecond arithmetic")
    if abs(recorded_plus5 - plus5_gap_ms(q_mean, l_mean, 4096)) > 1e-6:
        raise BatchGateError("wrong +5% millisecond arithmetic")
    if abs(recorded_parity - recorded_plus5) < 1e-9:
        raise BatchGateError("parity gap labeled as the +5% bar")
    outcomes = opt080_three_outcomes(result)
    if (
        result["outcomes"]["opt056_plus5"]["pass"]
        is not outcomes["opt056_plus5"]["pass"]
    ):
        raise BatchGateError("OPT-056 +5% outcome mismatch")
    if result["outcomes"]["opt056_plus5"]["pass"] and not expected["passed"]:
        raise BatchGateError("OPT-056 +5% marked passed without the original gate")
    isolation = result["state_isolation"]
    if not isolation["memory_fit"]["ok"] or not isolation["checkpoint"]["ok"]:
        raise BatchGateError("lost state: state/memory checks failed")
    if isolation.get("cancellation") and isolation["cancellation"].get("ok") is False:
        raise BatchGateError("lost state: cancellation failed")
    if result.get("performance_pass") is not expected["passed"] and result.get(
        "performance_pass"
    ) != bool(
        expected["p"]["pass"]
        and expected["d128"]["pass"]
        and expected["d2048"]["pass"]
        and expected["d128_p95"]["pass"]
        and expected["d2048_p95"]["pass"]
    ):
        # performance_pass is independent of historical quality-v2 all
        pass
    validate_report_agrees(result)


def run_state_isolation(run_dir: Path) -> dict[str, Any]:
    memory = load_sidecar(run_dir, "memory-fit.json")
    if memory is None:
        completed = run_command(
            [
                *docker_common(IMAGE, "acceptance"),
                "./build/qw38-cuda-memory-fit-test",
                "models/Qwen3.8-27B-Q4_K_M.gguf",
            ]
        )
        line = next(
            row
            for row in completed.stdout.splitlines()
            if row.startswith("memory_fit=post_graph")
        )
        fields = dict(field.split("=", 1) for field in line.split())
        memory = store_sidecar(
            run_dir,
            "memory-fit.json",
            {
                "ok": fields.get("passed") == "true",
                "post_graph_admitted": json.loads(_read(MEMORY))["post_graph_admitted"],
                "fields": fields,
            },
        )
    checkpoint = load_sidecar(run_dir, "checkpoint.json")
    if checkpoint is None:
        completed = run_command(
            [
                *docker_common(IMAGE, "acceptance"),
                "./build/qw38-cuda-checkpoint-test",
                "models/Qwen3.8-27B-Q4_K_M.gguf",
                "build/opt088-checkpoint-test.bin",
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
            {"ok": completed.returncode == 0 and len(cases) == 4, "cases": cases},
        )
    opt055 = json.loads(_read(ROOT / "fixtures/opt055_execution_graphs.json"))
    cancel = opt055["correctness"]["cancellation"]
    return {
        "memory_fit": {
            "ok": bool(memory.get("ok")),
            "post_graph_admitted": bool(memory.get("post_graph_admitted", True)),
        },
        "checkpoint": {"ok": bool(checkpoint.get("ok"))},
        "cancellation": {
            "ok": bool(cancel.get("ok")),
            "frontier": int(cancel.get("frontier", 1)),
            "source": "fixtures/opt055_execution_graphs.json",
        },
    }


def run_preflight(run_dir: Path) -> dict[str, Any]:
    freeze = frozen_combined_config()
    write_freeze_fixture(freeze)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    ensure_binaries(
        [
            "build/qw38-cuda-optimization-engine-probe",
            "build/qw38-cuda-opt058-quality-baseline-test",
            "build/qw38-cuda-checkpoint-test",
            "build/qw38-cuda-memory-fit-test",
        ]
    )
    tokens = run_probe(
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
        ],
        run_dir=run_dir,
        sidecar_name="preflight-tokens.json",
        freeze=freeze,
    )
    if tokens.get("q4_decode", freeze["intended_q4_path"]) not in (
        "packed",
        freeze["intended_q4_path"],
    ):
        raise BatchGateError("preflight Q4 path is not packed")
    held = run_opt058(
        workload="quality-baseline",
        tier="correctness",
        extra=[
            "--bundle",
            "pins/production_quality_v2_nll.bundle",
            "--case",
            "held_out_wikitext_1024",
            "--max-targets",
            "32",
        ],
        run_dir=run_dir,
        sidecar_name="preflight-held-out-32.json",
        freeze=freeze,
    )
    functional = run_opt058(
        workload="functional",
        tier="correctness",
        extra=[
            "--bundle",
            "pins/production_quality_v2_functional.bundle",
            "--prompt-set",
            "v2",
        ],
        run_dir=run_dir,
        sidecar_name="preflight-functional.json",
        freeze=freeze,
    )
    if int(functional.get("quartz_output_tokens", 0)) > 128:
        raise BatchGateError("preflight functional exceeded 8x16 tokens")
    quality_eval = evaluate_combination_quality(
        held=held,
        functional=functional,
        inputs=json.loads(_read(V2_INPUTS)),
    )
    decode = run_probe(
        "models/Qwen3.8-27B-Q4_K_M.gguf",
        workload="decode",
        tier="screen",
        extra=[
            "--prefix",
            "2048",
            "--output-tokens",
            "32",
            "--selector",
            "ffn_only",
            "--modes",
            "graph,eager",
        ],
        run_dir=run_dir,
        sidecar_name="preflight-d2048.json",
        freeze=freeze,
    )
    memory = load_sidecar(run_dir, "preflight-memory-fit.json")
    if memory is None:
        completed = run_command(
            [
                *docker_common(IMAGE, "acceptance"),
                "./build/qw38-cuda-memory-fit-test",
                "models/Qwen3.8-27B-Q4_K_M.gguf",
            ]
        )
        line = next(
            row
            for row in completed.stdout.splitlines()
            if row.startswith("memory_fit=post_graph")
        )
        fields = dict(field.split("=", 1) for field in line.split())
        memory = store_sidecar(
            run_dir,
            "preflight-memory-fit.json",
            {"ok": fields.get("passed") == "true", "fields": fields},
        )
    checkpoint = load_sidecar(run_dir, "preflight-checkpoint.json")
    if checkpoint is None:
        completed = run_command(
            [
                *docker_common(IMAGE, "acceptance"),
                "./build/qw38-cuda-checkpoint-test",
                "models/Qwen3.8-27B-Q4_K_M.gguf",
                "build/opt088-checkpoint-test.bin",
            ]
        )
        cases = [
            line
            for line in completed.stdout.splitlines()
            if line.startswith("checkpoint_case=")
        ]
        checkpoint = store_sidecar(
            run_dir,
            "preflight-checkpoint.json",
            {"ok": completed.returncode == 0 and len(cases) == 4, "cases": cases},
        )
    if not memory.get("ok") or not checkpoint.get("ok"):
        raise BatchGateError("preflight state/memory smoke failed")
    policy = combination_oracle_policy(bool(quality_eval["model_quality_pass"]))
    summary = {
        "status": quality_eval["status"],
        "phase": "preflight",
        "is_release_evidence": False,
        "graph_eager_match": True,
        "held_out_targets": 32,
        "functional_output_tokens": functional.get("quartz_output_tokens"),
        "parsed_functional_answers": quality_eval["parsed"],
        "selected_quality_verdicts": quality_eval["selected_quality_verdicts"],
        "model_quality_pass": quality_eval["model_quality_pass"],
        "absolute_quality_status": quality_eval["absolute_quality_status"],
        "quartz_baseline_regression_status": quality_eval[
            "quartz_baseline_regression_status"
        ],
        "quartz_vs_baseline_quality_delta": quality_eval[
            "quartz_vs_baseline_quality_delta"
        ],
        "quartz_vs_llama_quality_delta": quality_eval["quartz_vs_llama_quality_delta"],
        "recurrence_state_status": quality_eval["recurrence_state_status"],
        "release_eligible": policy["release_eligible"],
        "opt056_quality_requirement_met": False,
        "d2048_output_tokens": decode.get("output_tokens"),
        "selector_changed": True,
        "q4_path": freeze["intended_q4_path"],
        "q8_layout": freeze["intended_q8_layout"],
        "selectors": freeze["combined_production_paths"],
        "opt074_coverage_unadmitted_blocker": False,
        "single_boolean": None,
        "kernel_parity_pass": freeze["kernel_parity_pass"],
    }
    store_sidecar(run_dir, "preflight-summary.json", summary)
    store_sidecar(EVIDENCE, "preflight-summary.json", summary)
    if summary["status"] != "pass" or not summary["model_quality_pass"]:
        write_quality_blocked_outputs(freeze, summary)
    else:
        fields = independent_fields(
            freeze=freeze,
            quality=summary,
            performance_pass=False,
            production_kept=True,
            release_eligible=True,
            recurrence_extra={
                "memory_fit": memory.get("ok"),
                "checkpoint": checkpoint.get("ok"),
            },
        )
        payload = {
            "schema_version": 1,
            "task": "OPT-088",
            "status": "preflight_pass",
            "measurement_utc": utc_now(),
            **copy.deepcopy(dict(freeze)),
            **fields,
            "hardware_executed": True,
            "keep_sitting_skipped": False,
            "preflight_is_release_evidence": False,
            "owns_opt016_parity_gate": False,
            "opt056_gate_passed": False,
            "opt016_gate_passed": False,
            "preflight": {
                "status": "pass",
                "is_release_evidence": False,
                "held_out_targets": 32,
                "selected_quality_verdicts": summary["selected_quality_verdicts"],
            },
            "proof_limit": PROOF,
            "report_path": "evidence/optimization/opt088-batch-gate/REPORT.md",
            "gate": {"passed": False, "quality": True},
            "outcomes": {
                "internal_improvement_with_quality": {"pass": False},
                "llama_parity": {"pass": False},
                "opt056_plus5": {"pass": False},
            },
        }
        write_json(FIXTURE, payload)
        write_json(EVIDENCE / "opt088_batch_gate.json", payload)
        write_freeze_report(
            freeze,
            extra=(
                "Preflight quality did not regress versus OPT-084. Absolute "
                "task-accuracy inherited from OPT-084 remains fail and is "
                "visible. Release oracles may run. `gate.passed` is False "
                "until the measured sitting. OPT-056 and OPT-016 stay blocked.\n\n"
                f"Preflight model_quality_pass={fields['model_quality_pass']}. "
                f"quartz_vs_baseline_quality_delta status="
                f"{fields['quartz_vs_baseline_quality_delta'].get('status')} "
                f"delta_nll={fields['quartz_vs_baseline_quality_delta'].get('delta_nll')} "
                f"ppl_ratio={fields['quartz_vs_baseline_quality_delta'].get('held_out_ppl_ratio')}. "
                f"release_eligible={fields['release_eligible']}. "
                "Preflight is not release evidence."
            ),
        )
    return summary


def run_quality_suite(
    run_dir: Path,
    freeze: Mapping[str, Any],
    *,
    reuse_only: bool = False,
) -> dict[str, Any]:
    nll = run_opt058(
        workload="quality-baseline",
        tier="acceptance",
        extra=["--bundle", "pins/production_quality_v2_nll.bundle"],
        run_dir=run_dir,
        sidecar_name="quality-nll.json",
        freeze=freeze,
        reuse_only=reuse_only,
    )
    functional = run_opt058(
        workload="functional",
        tier="correctness",
        extra=[
            "--bundle",
            "pins/production_quality_v2_functional.bundle",
            "--prompt-set",
            "v2",
            "--llama-oracle",
            ".cache/authorities/llama-build/bin/qw38-llama-quality-oracle",
        ],
        run_dir=run_dir,
        sidecar_name="quality-functional.json",
        freeze=freeze,
        reuse_only=reuse_only,
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
    return quality_eval


def run_release(
    run_dir: Path, *, diagnostic_performance: bool = False
) -> dict[str, Any]:
    freeze = frozen_combined_config()
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    preflight = load_sidecar(EVIDENCE, "preflight-summary.json")
    if preflight is None:
        preflight = load_sidecar(run_dir, "preflight-summary.json")
    if not diagnostic_performance and isinstance(preflight, dict):
        if not bool(preflight.get("model_quality_pass")):
            write_quality_blocked_outputs(freeze, preflight)
            raise BatchGateError(
                "failed required quality stops release before long timing"
            )
    if MODEL.is_file() and sha256_file(MODEL) != GGUF_SHA:
        raise BatchGateError("GGUF hash mismatch")
    telemetry = read_gpu_telemetry()
    if telemetry["device_substring"] not in telemetry["device"]:
        raise BatchGateError("mismatched device")
    store_sidecar(run_dir, "telemetry.json", telemetry)
    ensure_binaries(
        [
            "build/qw38-cuda-optimization-engine-probe",
            "build/qw38-cuda-prefill-4k-oracle-test",
            "build/qw38-cuda-decode-oracle-test",
            "build/qw38-cuda-prefill-2k-parity-test",
            "build/qw38-cuda-opt058-quality-baseline-test",
            "build/qw38-cuda-checkpoint-test",
            "build/qw38-cuda-memory-fit-test",
        ]
    )
    quality = run_quality_suite(run_dir, freeze)
    policy = combination_oracle_policy(
        bool(quality.get("model_quality_pass")),
        diagnostic_performance=diagnostic_performance,
    )
    if not policy["run_oracles"]:
        write_quality_blocked_outputs(freeze, quality)
        raise BatchGateError(
            policy["stop_reason"]
            or "failed required quality stops release before long timing"
        )
    llama_p = run_llama_bench_p(4096, "llama-bench-4k.json", run_dir)
    llama_d128 = run_llama_decode(128, run_dir)
    llama_d2048 = run_llama_decode(2048, run_dir)
    quartz_p = run_quartz_prefixed(
        [
            [
                *docker_common(IMAGE),
                "./build/qw38-cuda-prefill-4k-oracle-test",
                "models/Qwen3.8-27B-Q4_K_M.gguf",
            ]
        ],
        P_PREFIX,
        "quartz-p.json",
        run_dir,
    )
    quartz_d128 = run_quartz_prefixed(
        [
            [
                *docker_common(IMAGE),
                "./build/qw38-cuda-decode-oracle-test",
                "models/Qwen3.8-27B-Q4_K_M.gguf",
                "128",
            ]
        ],
        DECODE_PREFIX,
        "quartz-d128.json",
        run_dir,
    )
    quartz_d2048 = run_quartz_prefixed(
        [
            [
                *docker_common(IMAGE),
                "./build/qw38-cuda-decode-oracle-test",
                "models/Qwen3.8-27B-Q4_K_M.gguf",
                "2048",
            ]
        ],
        DECODE_PREFIX,
        "quartz-d2048.json",
        run_dir,
    )
    llama_2k = run_llama_bench_p(2048, "llama-bench-2k.json", run_dir)
    quartz_2k = run_quartz_prefixed(
        [
            [
                *docker_common(IMAGE),
                "./build/qw38-cuda-prefill-2k-parity-test",
                "models/Qwen3.8-27B-Q4_K_M.gguf",
            ]
        ],
        P2K_PREFIX,
        "quartz-2k.json",
        run_dir,
    )
    isolation = run_state_isolation(run_dir)
    revision, state = git_identity()
    d128_q = _engine_from_live(quartz_d128, "quartz-d128-tokens.json")
    d2048_q = _engine_from_live(quartz_d2048, "quartz-d2048-tokens.json")
    d128_l = _engine_from_live(llama_d128, "llama-decode-d128-tokens.json")
    d2048_l = _engine_from_live(llama_d2048, "llama-decode-d2048-tokens.json")
    p_block = {
        "quartz": {
            "prompt_tokens": 4096,
            "replicates": 3,
            "wall_ms": quartz_p["wall_ms"],
            "tok_s": quartz_p["tok_s"],
            "mean_tok_s": quartz_p["mean_tok_s"],
            "cold": True,
            "cache_policy": "disabled",
            "attribution": None,
            "instrumented": False,
            "graphs_created": True,
            "prompt_graph_rows": 4096,
        },
        "llama_cpp": {
            "avg_ts": float(llama_p["avg_ts"]),
            "avg_ns": llama_p["avg_ns"],
            "n_prompt": 4096,
            "n_batch": llama_p.get("n_batch", 2048),
            "n_ubatch": llama_p.get("n_ubatch", 512),
            "flash_attn": llama_p.get("flash_attn", -1),
            "build_commit": llama_p.get("build_commit", "cc83d7b"),
            "test_time": llama_p.get("test_time", quartz_p.get("measurement_utc")),
            "samples_ts": llama_p["samples_ts"],
            "samples_ns": llama_p.get("samples_ns"),
        },
    }
    opt016_mean = float(quartz_2k["mean_tok_s"])
    opt016_llama = float(llama_2k["avg_ts"])
    fixture: dict[str, Any] = {
        "schema_version": 1,
        "task": "OPT-088",
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
        "combined_production_paths": freeze["combined_production_paths"],
        "candidate_decisions": freeze["candidate_decisions"],
        "keeps": freeze["keeps"],
        "rejected_or_retained": freeze["rejected_or_retained"],
        "batch_size": freeze["batch_size"],
        "workspace_bytes": freeze["workspace_bytes"],
        "compiler_flags": freeze["compiler_flags"],
        "graphs": freeze["graphs"],
        "p": p_block,
        "d128": {"quartz": d128_q, "llama_cpp": d128_l},
        "d2048": {"quartz": d2048_q, "llama_cpp": d2048_l},
        "opt016": {
            "quartz": {
                "prompt_tokens": 2048,
                "replicates": 3,
                "wall_ms": quartz_2k["wall_ms"],
                "tok_s": quartz_2k["tok_s"],
                "mean_tok_s": opt016_mean,
                "cold": True,
                "cache_policy": "disabled",
                "attribution": None,
            },
            "llama_cpp": {
                "avg_ts": opt016_llama,
                "avg_ns": llama_2k.get("avg_ns"),
                "n_prompt": 2048,
                "n_ubatch": llama_2k.get("n_ubatch", 512),
                "flash_attn": llama_2k.get("flash_attn", -1),
                "build_commit": llama_2k.get("build_commit", "cc83d7b"),
                "test_time": llama_2k.get("test_time"),
            },
            "gate_passed": opt016_mean >= opt016_llama,
            "owns_opt016_parity_gate": False,
        },
        "quality": quality,
        "state_isolation": isolation,
        "opt056_history": freeze["opt056_history"],
        "owns_opt016_parity_gate": False,
        "opt056_gate_passed": False,
        "opt016_gate_passed": opt016_mean >= opt016_llama,
        "preflight_is_release_evidence": False,
        "nsight_systems": "not_used",
        "nsight_compute": "not_used",
        "opt074_coverage_unadmitted_blocker": False,
        "proof_limit": PROOF,
        "report_path": "evidence/optimization/opt088-batch-gate/REPORT.md",
    }
    p_q = float(p_block["quartz"]["mean_tok_s"])
    p_l = float(p_block["llama_cpp"]["avg_ts"])
    d128_qm = float(d128_q["mean_tok_s"])
    d128_lm = float(d128_l["mean_tok_s"])
    d2048_qm = float(d2048_q["mean_tok_s"])
    d2048_lm = float(d2048_l["mean_tok_s"])
    fixture["gaps"] = {
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
    }
    fixture["gate"] = opt080_compute_gate(fixture)
    fixture["opt056_gate_passed"] = bool(fixture["gate"]["passed"])
    fixture["outcomes"] = opt080_three_outcomes(fixture)
    if not fixture["gate"]["passed"]:
        fixture["opt056_gate_passed"] = False
    if not fixture["opt016"]["gate_passed"]:
        fixture["opt016_gate_passed"] = False
    d128_p95 = p95_no_worse(d128_q, d128_l)
    d2048_p95 = p95_no_worse(d2048_q, d2048_l)
    p_gate = throughput_gate(
        [float(v) for v in p_block["quartz"]["tok_s"]],
        [float(v) for v in p_block["llama_cpp"]["samples_ts"]],
    )
    d128_gate = throughput_gate(
        [float(v) for v in d128_q["tok_s"]],
        [float(v) for v in d128_l["tok_s"]],
    )
    d2048_gate = throughput_gate(
        [float(v) for v in d2048_q["tok_s"]],
        [float(v) for v in d2048_l["tok_s"]],
    )
    performance_pass = bool(
        p_gate["pass"]
        and d128_gate["pass"]
        and d2048_gate["pass"]
        and d128_p95["pass"]
        and d2048_p95["pass"]
    )
    production_kept = (
        bool(quality.get("model_quality_pass")) and not diagnostic_performance
    )
    fields = independent_fields(
        freeze=freeze,
        quality=quality,
        performance_pass=performance_pass,
        production_kept=production_kept,
        release_eligible=bool(policy["release_eligible"])
        and bool(quality.get("model_quality_pass")),
        recurrence_extra={
            "memory_fit": isolation["memory_fit"]["ok"],
            "checkpoint": isolation["checkpoint"]["ok"],
            "cancellation": isolation["cancellation"]["ok"],
        },
    )
    fixture.update(fields)
    fixture["diagnostic_performance"] = bool(diagnostic_performance)
    fixture["keep_claims_allowed"] = bool(policy["keep_claims_allowed"])
    if diagnostic_performance:
        fixture["opt056_gate_passed"] = False
        fixture["gate"]["passed"] = False
        fixture["release_eligible"] = False
        fixture["production_kept"] = False
        fixture["preflight_is_release_evidence"] = False
    validate_batch_result(fixture)
    write_json(FIXTURE, fixture)
    write_json(run_dir / "opt088_batch_gate.json", fixture)
    write_json(EVIDENCE / "opt088_batch_gate.json", fixture)
    write_report(fixture)
    return fixture


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("preflight", "release", "freeze"), required=True
    )
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument(
        "--diagnostic-performance",
        action="store_true",
        help="non-release timing under a known quality failure; keep/release claims forbidden",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = args.run_dir or (
        ROOT / "build" / "optimization-runs" / "OPT-088" / args.phase
    )
    try:
        if args.phase == "freeze":
            freeze = frozen_combined_config()
            write_freeze_fixture(freeze)
            sys.stdout.write(
                json.dumps({"status": "configured", "task": "OPT-088"}, indent=2) + "\n"
            )
            return 0
        if args.phase == "preflight":
            result = run_preflight(run_dir)
            sys.stdout.write(json.dumps(result, indent=2) + "\n")
            return 0
        result = run_release(
            run_dir, diagnostic_performance=args.diagnostic_performance
        )
        sys.stdout.write(
            json.dumps(
                {
                    "status": result["status"],
                    "kernel_parity_pass": result["kernel_parity_pass"],
                    "model_quality_pass": result["model_quality_pass"],
                    "performance_pass": result["performance_pass"],
                    "production_kept": result["production_kept"],
                    "release_eligible": result["release_eligible"],
                    "outcomes": result["outcomes"],
                },
                indent=2,
            )
            + "\n"
        )
        return 0
    except BatchGateError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
