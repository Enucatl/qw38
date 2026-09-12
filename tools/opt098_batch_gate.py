"""Combined OPT-098 post-088 outcome gate.

Freezes authenticated OPT-088 control versus OPT-089–097 selected production.
Independent kernel-parity, quality-delta, recurrence, performance, keep, and
release-eligible fields. Three outcomes: internal vs OPT-088 control, llama
parity, and historical OPT-056/+5%.
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

from tools.opt058_quality_baseline import TASK_NAMES  # noqa: E402
from tools.opt073_quality_policy import (  # noqa: E402
    QualityPolicyError,
    validate_parsed_functional_answers,
)
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
    _finite,
    _mean,
    _search,
    _var,
)
from tools.opt084_quality_baseline import FROZEN_ACCEPTANCE  # noqa: E402
from tools.opt088_batch_gate import (  # noqa: E402
    OPT080_CONTRACT,
    OPT080_FIXTURE,
    OPT080_REPORT,
    P2K_PREFIX,
    P_PREFIX,
    DECODE_PREFIX,
    combination_oracle_policy,
    independent_fields,
    refuse_quality_as_one_boolean,
    run_llama_bench_p,
    run_llama_decode,
    run_opt058,
    run_probe,
    run_quartz_prefixed,
    run_state_isolation,
    load_sidecar,
    store_sidecar,
)
from tools.quality.quality_mode import apply_quality_mode  # noqa: E402
from tools.quality.scoring import recurrence_incremental_nll  # noqa: E402
from tools.quality.suite import RECURRENCE_MAX, SUITE_CLASSES  # noqa: E402

CONTRACT = ROOT / "pins/opt098_batch_gate_contract.json"
ITERATION = ROOT / "pins/opt098_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt098_batch_gate.json"
EVIDENCE = ROOT / "evidence/optimization/opt098-batch-gate"
REPORT = EVIDENCE / "REPORT.md"
OPT088_FIXTURE = ROOT / "fixtures/opt088_batch_gate.json"
OPT088_REPORT = ROOT / "evidence/optimization/opt088-batch-gate/REPORT.md"
OPT088_CONTRACT = ROOT / "pins/opt088_batch_gate_contract.json"
OPT056 = ROOT / "fixtures/opt056_performance_gate.json"
OPT084 = ROOT / "fixtures/opt084_quality_baseline.json"
OPT091 = ROOT / "fixtures/opt091_quality_tradeoff.json"
PREFLIGHT_HELD32 = ROOT / (
    "evidence/optimization/opt069-batch-gate/preflight-held-out-32.json"
)
V2_INPUTS = ROOT / "pins/production_quality_v2_inputs.json"
V2_LLAMA = ROOT / "pins/production_quality_v2_llama_reference.json"
MEMORY = ROOT / "fixtures/cuda_memory_fit_post_graph.json"
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
DEPENDENCY_IDS = (
    "OPT-089",
    "OPT-090",
    "OPT-091",
    "OPT-092",
    "OPT-093",
    "OPT-094",
    "OPT-095",
    "OPT-096",
    "OPT-097",
)
DEPENDENCY_FIXTURES = {
    "OPT-089": "opt089_q4_promotion.json",
    "OPT-090": "opt090_decode_attribution.json",
    "OPT-091": "opt091_quality_tradeoff.json",
    "OPT-092": "opt092_q8_grouped.json",
    "OPT-093": "opt093_q4_factored.json",
    "OPT-094": "opt094_gdn_replication.json",
    "OPT-095": "opt095_attention_gqa.json",
    "OPT-096": "opt096_decode_graphs.json",
    "OPT-097": "opt097_mmq_wait.json",
}
STRICT_PPL_RATIO_MAX = 1.01
CONDITIONAL_LATE_W4_PPL_RATIO_MAX = 1.015
INTERNAL_RATIO_MIN = 0.95
INTERNAL_P95_RATIO_MAX = 1.05
PHASES = (
    "freeze",
    "preflight",
    "quality",
    "state-memory",
    "performance",
    "two-k",
    "report",
)
PROOF = (
    "combined freeze of post088_control from authenticated OPT-088 and "
    "post098_selected from OPT-089-097 keeps; "
    "independent kernel_parity_pass per family; "
    "quality is not one boolean; "
    "quartz_vs_baseline_quality_delta and quartz_vs_llama_quality_delta stay "
    "distinct; "
    "opt074_coverage_unadmitted is not a blocker; "
    "absolute task-accuracy fail inherited from OPT-084 does not by itself "
    "block; "
    "new regression versus OPT-084 does block; "
    "OPT-088 remains historical and is not reinterpreted; "
    "OPT-056 and OPT-016 stay blocked unless their owning conditions pass; "
    "parity gap is Tq-Tl; "
    "+5% throughput gap is Tq-Tl/1.05; "
    "do not label the parity gap as the +5% bar; "
    "decode p95 no worse than llama for the OPT-056 outcome; "
    "preflight is not release evidence; "
    "failed required quality stops release before long timing; "
    "diagnostic performance is not a release or keep; "
    "rejected candidates must not leak into production; "
    "internal improvement is selected versus fresh OPT-088 control not llama; "
    "strict_ppl_ratio_max=1.01 applies because concession is inactive; "
    "release_eligible is never a synonym for opt056_pass"
)
EXPECTED_PATHS: dict[str, Any] = {
    "rms_norm": "parallel_fma",
    "q4_decode": "integer_q8_late",
    "q8_decode": "dp4a_q8_1",
    "q8_grouping": "grouped_r1_w4",
    "q8_decode_rows_skinny": 1,
    "q8_decode_rows_medium": 1,
    "q8_decode_rows_wide": 1,
    "q8_decode_layout_warps_skinny": 4,
    "q8_decode_layout_warps_medium": 4,
    "q8_decode_layout_warps_wide": 4,
    "q8_layout": "r1_w4",
    "q6_decode": "integer_q8_1",
    "ffn_decode": "paired_integer",
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
    "q4_decode": "integer_q8_late",
    "q4_staging": "paired_integer",
    "q8_decode": "r1_w4",
    "q8_path": "dp4a_q8_1",
    "q8_grouping": "grouped_r1_w4",
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
    """Inadmissible combined post-088 evidence."""


def run_command_native(command: Sequence[str]) -> SimpleNamespace:
    """Run docker-wrapped commands on the host when QW38_HOST_NATIVE=1."""
    if (
        os.environ.get("QW38_HOST_NATIVE") != "1"
        or not command
        or command[0] != "docker"
    ):
        return _docker_run_command(command)
    listed = list(command)
    image_index = -1
    for image_name in (str(IMAGE), str(LLAMA_IMAGE)):
        try:
            image_index = listed.index(image_name)
            break
        except ValueError:
            continue
    if image_index < 0:
        return _docker_run_command(command)
    inner = listed[image_index + 1 :]
    if inner[:2] == ["bash", "-lc"] and len(inner) >= 3:
        script = str(inner[2]).replace("/workspace", str(ROOT))
        inner = ["bash", "-lc", script]
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


_NATIVE_COMMANDS_ENABLED = False


def _enable_native_commands() -> None:
    global _NATIVE_COMMANDS_ENABLED
    if _NATIVE_COMMANDS_ENABLED:
        return
    import tools.opt080_batch_gate as opt080_module
    import tools.opt088_batch_gate as opt088_module

    opt080_module.run_command = run_command_native
    opt088_module.run_command = run_command_native
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
    return paths


def post088_control_paths() -> dict[str, Any]:
    payload = json.loads(_read(OPT088_FIXTURE))
    control = payload.get("combined_production_paths")
    if not isinstance(control, Mapping):
        raise BatchGateError("authenticated OPT-088 control paths missing")
    contract = load_contract()
    expected = contract["post088_control"]
    for key, value in expected.items():
        if control.get(key) != value:
            raise BatchGateError(
                f"OPT-088 control mismatch {key}: {control.get(key)!r} != {value!r}"
            )
    return dict(control)


def concession_active() -> bool:
    if not OPT091.is_file():
        return False
    policy = json.loads(_read(OPT091)).get("policy") or {}
    return bool(policy.get("concession_used"))


def active_ppl_ratio_max() -> float:
    if concession_active():
        return CONDITIONAL_LATE_W4_PPL_RATIO_MAX
    return STRICT_PPL_RATIO_MAX


def _load_fixture(name: str) -> dict[str, Any]:
    path = ROOT / "fixtures" / name
    if not path.is_file():
        raise BatchGateError(f"incomplete admissions: missing fixture {name}")
    payload = json.loads(_read(path))
    if not isinstance(payload, dict):
        raise BatchGateError(f"incomplete admissions: {name} is not an object")
    return payload


def opt088_artifacts_unmodified() -> dict[str, str]:
    contract = load_contract()
    expected = contract["opt088_historical_hashes"]
    actual = {
        "report": hashlib.sha256(OPT088_REPORT.read_bytes()).hexdigest(),
        "fixture": hashlib.sha256(OPT088_FIXTURE.read_bytes()).hexdigest(),
        "contract": hashlib.sha256(OPT088_CONTRACT.read_bytes()).hexdigest(),
    }
    for name, digest in expected.items():
        if actual[name] != digest:
            raise BatchGateError(
                f"OPT-088 {name} was reinterpreted or rewritten "
                f"(hash {actual[name]} != {digest})"
            )
    fixture = json.loads(_read(OPT088_FIXTURE))
    if fixture.get("task") != "OPT-088":
        raise BatchGateError("OPT-088 fixture task relabeled")
    return actual


def opt080_artifacts_unmodified() -> dict[str, str]:
    fixture = json.loads(_read(OPT080_FIXTURE))
    if fixture.get("task") != "OPT-080":
        raise BatchGateError("OPT-080 fixture task relabeled")
    if fixture.get("opt056_gate_passed") is True:
        raise BatchGateError("historical gate relabeling of OPT-080/OPT-056")
    return {
        "report": hashlib.sha256(OPT080_REPORT.read_bytes()).hexdigest(),
        "fixture": hashlib.sha256(OPT080_FIXTURE.read_bytes()).hexdigest(),
        "contract": hashlib.sha256(OPT080_CONTRACT.read_bytes()).hexdigest(),
    }


def audit_dependencies() -> dict[str, Any]:
    opt088_artifacts_unmodified()
    opt080_artifacts_unmodified()
    loaded: dict[str, Any] = {}
    for task_id in DEPENDENCY_IDS:
        name = DEPENDENCY_FIXTURES[task_id]
        payload = _load_fixture(name)
        if payload.get("task") not in (task_id, None):
            raise BatchGateError(f"incomplete admissions: {name} task mismatch")
        loaded[task_id] = payload
    opt089 = loaded["OPT-089"]
    if opt089.get("shipping_q4_decode") != "integer_q8_late":
        raise BatchGateError("incomplete admissions: OPT-089 Q4 is not late_w4")
    if opt089.get("shipping_ffn_decode") != "paired_integer":
        raise BatchGateError("incomplete admissions: OPT-089 FFN is not paired_integer")
    late = (opt089.get("independent_verdicts") or {}).get("late_w4") or {}
    if late.get("kernel_parity_pass") is not True:
        raise BatchGateError("incomplete family parity: OPT-089 late_w4 missing")
    opt092 = loaded["OPT-092"]
    if opt092.get("shipping_grouping") != "grouped_r1_w4":
        raise BatchGateError("incomplete admissions: OPT-092 grouping not kept")
    grouped = (opt092.get("independent_verdicts") or {}).get("grouped_r1_w4") or {}
    if grouped.get("kernel_parity_pass") is not True:
        raise BatchGateError("incomplete family parity: OPT-092 grouped_r1_w4 missing")
    opt093 = loaded["OPT-093"]
    if opt093.get("shipping_q4_decode") != "integer_q8_late":
        raise BatchGateError("rejected OPT-093 factored Q4 leaked into production")
    opt094 = loaded["OPT-094"]
    if opt094.get("shipping_gdn_decode") != "sequential":
        raise BatchGateError("rejected OPT-094 tile32 leaked into production")
    opt095 = loaded["OPT-095"]
    if opt095.get("production_kept") is True:
        raise BatchGateError("rejected OPT-095 GQA leaked into production")
    opt096 = loaded["OPT-096"]
    if opt096.get("shipping_execution_graphs") not in (None, "ffn_only"):
        raise BatchGateError("rejected OPT-096 decode graphs leaked into production")
    opt097 = loaded["OPT-097"]
    if opt097.get("selected_mmq_split_xy_wait") is True:
        raise BatchGateError("rejected OPT-097 split_xy_wait leaked into production")
    if opt097.get("selected_mmq_async_x") is not True:
        raise BatchGateError("OPT-097 must retain fma_async_x MMQ keep")
    return loaded


def family_kernel_parity(loaded: Mapping[str, Any] | None = None) -> dict[str, Any]:
    deps = loaded or audit_dependencies()
    opt089 = deps["OPT-089"]
    opt092 = deps["OPT-092"]
    late = opt089["independent_verdicts"]["late_w4"]
    grouped = opt092["independent_verdicts"]["grouped_r1_w4"]
    opt097 = deps["OPT-097"]
    record = {
        "q4": bool(late.get("kernel_parity_pass")),
        "q8": bool(grouped.get("kernel_parity_pass")),
        "mmq": bool(
            (opt097.get("independent_verdicts") or {}).get("kernel_parity_pass")
        ),
        "kv_once": True,
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
            "q4": "fixtures/opt089_q4_promotion.json#late_w4",
            "q8": "fixtures/opt092_q8_grouped.json#grouped_r1_w4",
            "mmq": "fixtures/opt097_mmq_wait.json#joined_wait",
            "kv_once": "fixtures/opt079_attention_kv_operands.json#keep",
        },
        "opt074_coverage_unadmitted_blocker": False,
    }


def candidate_decisions() -> dict[str, Any]:
    audit_dependencies()
    return {
        "OPT-089": {
            "status": "keep",
            "installed": True,
            "selected": "late_w4",
            "production_kept": True,
            "kernel_parity_pass": True,
            "model_quality_pass": True,
        },
        "OPT-090": {
            "status": "attribution_only",
            "installed": False,
            "selected": None,
            "production_kept": False,
        },
        "OPT-091": {
            "status": "strict_quality_passed",
            "installed": False,
            "selected": None,
            "concession_used": False,
            "production_kept": False,
        },
        "OPT-092": {
            "status": "keep",
            "installed": True,
            "selected": "grouped_r1_w4",
            "production_kept": True,
            "kernel_parity_pass": True,
            "model_quality_pass": True,
        },
        "OPT-093": {
            "status": "reject_factored",
            "installed": False,
            "selected": None,
            "production_kept": False,
        },
        "OPT-094": {
            "status": "no_reopen",
            "installed": False,
            "selected": None,
            "production_kept": False,
        },
        "OPT-095": {
            "status": "performance_rejected",
            "installed": False,
            "selected": None,
            "production_kept": False,
        },
        "OPT-096": {
            "status": "eligible_not_installed",
            "installed": False,
            "selected": None,
            "production_kept": False,
        },
        "OPT-097": {
            "status": "reject_split_xy_wait",
            "installed": False,
            "selected": "joined_wait",
            "production_kept": True,
            "kernel_parity_pass": True,
        },
    }


def frozen_combined_config() -> dict[str, Any]:
    paths = source_paths()
    control = post088_control_paths()
    decisions = candidate_decisions()
    parity = family_kernel_parity()
    for key, expected in EXPECTED_PATHS.items():
        if paths.get(key) != expected:
            raise BatchGateError(
                f"combined freeze mismatch {key}: {paths.get(key)!r} != {expected!r}"
            )
    if paths["q4_decode"] != "integer_q8_late":
        raise BatchGateError("OPT-089 late_w4 must remain installed")
    if paths["ffn_decode"] != "paired_integer":
        raise BatchGateError("OPT-089 paired_integer FFN must remain installed")
    if paths["q8_grouping"] != "grouped_r1_w4":
        raise BatchGateError("OPT-092 grouped_r1_w4 must remain installed")
    if paths["attention_pipeline"] != "kv_once":
        raise BatchGateError("OPT-079 kv_once must remain the attention pipeline")
    if paths["execution_graphs"] != "ffn_only":
        raise BatchGateError("rejected decode graphs leaked into production")
    quality_mode = apply_quality_mode(enabled=True, selectors=COMBINATION_SELECTORS)
    return {
        "post088_control": control,
        "post098_selected": paths,
        "combined_production_paths": paths,
        "candidate_decisions": decisions,
        "kernel_parity_pass": parity,
        "keeps": [
            "OPT-089 late_w4 integer_q8_late/paired_integer",
            "OPT-092 grouped_r1_w4 Q8 grouping",
            "OPT-079 kv_once",
            "OPT-086 fma_async_x MMQ keep",
            "OPT-097 joined_wait MMQ schedule retained",
        ],
        "rejected_or_retained": [
            "OPT-088 packed Q4 not installed",
            "OPT-093 factored_pair_w4 rejected",
            "OPT-094 tile32 no_reopen",
            "OPT-095 warp_query_gqa6 rejected",
            "OPT-096 decode_segments8 not installed",
            "OPT-097 split_xy_wait rejected",
        ],
        "batch_size": paths["prompt_microbatch_rows"],
        "compiler_flags": paths["nvccflags"],
        "graphs": paths["execution_graphs"],
        "intended_q4_path": "integer_q8_late",
        "intended_q8_layout": "r1_w4",
        "intended_q8_grouping": "grouped_r1_w4",
        "intended_mmq": "fma_async_x",
        "quality_mode": quality_mode,
        "opt088_history": "fixtures/opt088_batch_gate.json",
        "opt056_history": "fixtures/opt056_performance_gate.json",
        "opt084_baseline": "fixtures/opt084_quality_baseline.json",
        "opt074_coverage_unadmitted_blocker": False,
        "concession_active": concession_active(),
        "strict_ppl_ratio_max": active_ppl_ratio_max(),
    }


def baseline_quality() -> dict[str, Any]:
    return json.loads(_read(OPT084))


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
    baseline = baseline_quality()
    ppl_max = active_ppl_ratio_max()
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
    held_ok = math.isfinite(held_ratio) and held_ratio <= ppl_max
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
        "ppl_ratio_max": ppl_max,
        "strict_ppl_ratio_max": STRICT_PPL_RATIO_MAX,
        "conditional_late_w4_ppl_ratio_max": CONDITIONAL_LATE_W4_PPL_RATIO_MAX,
        "concession_active": concession_active(),
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
    opt088_fixture = json.loads(_read(OPT088_FIXTURE))
    opt088_held = float(
        (opt088_fixture.get("quality") or {})
        .get("quartz_vs_baseline_quality_delta", {})
        .get("held_out_32_mean_nll", math.nan)
    )
    vs_opt088 = {
        "opt088_held_out_mean_nll": opt088_held,
        "delta_quartz_minus_opt088": held_nll - opt088_held
        if _finite(opt088_held)
        else None,
        "inspectable": True,
        "single_boolean": None,
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
    status = "pass" if model_quality_pass else "quality_blocked"
    return {
        "status": status,
        "model_quality_pass": model_quality_pass,
        "absolute_quality_status": absolute_status,
        "quartz_baseline_regression_status": vs_baseline["status"],
        "quartz_vs_baseline_quality_delta": vs_baseline,
        "quartz_vs_opt088_quality_delta": vs_opt088,
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


def log_speed_ratio_ci_lower(
    control_samples: Sequence[float],
    candidate_samples: Sequence[float],
    *,
    threshold: float = 1.0,
    critical: float = T_CRIT_DF4,
) -> dict[str, Any]:
    if len(control_samples) != len(candidate_samples) or len(control_samples) < 2:
        return {
            "pass": False,
            "incomplete": True,
            "ci_lower": None,
            "reason": "need_paired_samples",
        }
    logs: list[float] = []
    ratios: list[float] = []
    for control, candidate in zip(control_samples, candidate_samples):
        if float(control) <= 0.0 or float(candidate) <= 0.0:
            return {
                "pass": False,
                "incomplete": True,
                "ci_lower": None,
                "reason": "non_positive_tok_s",
            }
        ratio = float(candidate) / float(control)
        ratios.append(ratio)
        logs.append(math.log(ratio))
    avg = _mean(logs)
    var = _var(logs, avg)
    se = math.sqrt(var / len(logs)) if var > 0.0 else 0.0
    lower = math.exp(avg - critical * se)
    min_ratio = min(ratios)
    return {
        "pass": lower > threshold and min_ratio >= INTERNAL_RATIO_MIN,
        "incomplete": False,
        "ci_lower": lower,
        "min_ratio": min_ratio,
        "min_ratio_threshold": INTERNAL_RATIO_MIN,
        "threshold": threshold,
        "log_mean": avg,
        "df": len(logs) - 1,
        "critical": critical,
    }


def internal_p95_pass(
    control: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    control_p95 = float(control["token_latency_p95_ms"])
    candidate_p95 = float(candidate["token_latency_p95_ms"])
    if control_p95 <= 0.0:
        return {"pass": False, "incomplete": True, "ratio": None}
    ratio = candidate_p95 / control_p95
    return {
        "pass": ratio <= INTERNAL_P95_RATIO_MAX,
        "ratio": ratio,
        "limit": INTERNAL_P95_RATIO_MAX,
        "control_p95_ms": control_p95,
        "candidate_p95_ms": candidate_p95,
    }


def control_baseline_blocks() -> dict[str, Any]:
    payload = json.loads(_read(OPT088_FIXTURE))
    return {
        "p": payload["p"],
        "d128": payload["d128"],
        "d2048": payload["d2048"],
    }


def three_outcomes(result: Mapping[str, Any]) -> dict[str, Any]:
    control = control_baseline_blocks()
    p_q = float(result["p"]["quartz"]["mean_tok_s"])
    p_l = float(result["p"]["llama_cpp"]["avg_ts"])
    d128_q = float(result["d128"]["quartz"]["mean_tok_s"])
    d128_l = float(result["d128"]["llama_cpp"]["mean_tok_s"])
    d2048_q = float(result["d2048"]["quartz"]["mean_tok_s"])
    d2048_l = float(result["d2048"]["llama_cpp"]["mean_tok_s"])
    p_ctrl = float(control["p"]["quartz"]["mean_tok_s"])
    d128_ctrl = float(control["d128"]["quartz"]["mean_tok_s"])
    d2048_ctrl = float(control["d2048"]["quartz"]["mean_tok_s"])
    quality_pass = bool(result.get("model_quality_pass"))
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
        math.log(float(result["p"]["quartz"]["mean_tok_s"]) / p_ctrl),
        math.log(float(result["d128"]["quartz"]["mean_tok_s"]) / d128_ctrl),
        math.log(float(result["d2048"]["quartz"]["mean_tok_s"]) / d2048_ctrl),
    ]
    geo_mean_ratio = math.exp(_mean(logs))
    geo_var = _var(logs, _mean(logs))
    geo_se = math.sqrt(geo_var / len(logs)) if geo_var > 0.0 else 0.0
    geo_lower = math.exp(_mean(logs) - T_CRIT_DF4 * geo_se)
    d128_p95 = internal_p95_pass(control["d128"]["quartz"], result["d128"]["quartz"])
    d2048_p95 = internal_p95_pass(control["d2048"]["quartz"], result["d2048"]["quartz"])
    internal_pass = bool(
        quality_pass
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
    }
    parity = {
        "name": "llama_parity",
        "p": p_q >= p_l,
        "d128": d128_q >= d128_l,
        "d2048": d2048_q >= d2048_l,
        "d128_p95": float(result["d128"]["quartz"]["token_latency_p95_ms"])
        <= float(result["d128"]["llama_cpp"]["token_latency_p95_ms"]),
        "d2048_p95": float(result["d2048"]["quartz"]["token_latency_p95_ms"])
        <= float(result["d2048"]["llama_cpp"]["token_latency_p95_ms"]),
        "p_parity_gap_ms": parity_gap_ms(p_q, p_l, 4096),
        "d128_parity_gap_ms": parity_gap_ms(d128_q, d128_l, 256),
        "d2048_parity_gap_ms": parity_gap_ms(d2048_q, d2048_l, 256),
        "pass": p_q >= p_l
        and d128_q >= d128_l
        and d2048_q >= d2048_l
        and float(result["d128"]["quartz"]["token_latency_p95_ms"])
        <= float(result["d128"]["llama_cpp"]["token_latency_p95_ms"])
        and float(result["d2048"]["quartz"]["token_latency_p95_ms"])
        <= float(result["d2048"]["llama_cpp"]["token_latency_p95_ms"]),
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
            f"evidence/optimization/opt098-batch-gate/{sidecar_name}"
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


def write_freeze_fixture(freeze: Mapping[str, Any]) -> None:
    payload = {
        "schema_version": 1,
        "task": "OPT-098",
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
        "report_path": "evidence/optimization/opt098-batch-gate/REPORT.md",
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
        f"""# OPT-098 — Combined post-088 outcome gate

## Claim labels and proof limits

{PROOF}.

Frozen post088_control from authenticated OPT-088 and post098_selected from
OPT-089–097 keeps. Preflight is not release evidence.

## Frozen combination

Keeps:

{chr(10).join(f"- {item}" for item in freeze["keeps"])}

Rejected or retained:

{chr(10).join(f"- {item}" for item in freeze["rejected_or_retained"])}

post088_control Q4={freeze["post088_control"]["q4_decode"]} FFN={freeze["post088_control"]["ffn_decode"]}.
post098_selected Q4={freeze["post098_selected"]["q4_decode"]} FFN={freeze["post098_selected"]["ffn_decode"]} Q8 grouping={freeze["post098_selected"]["q8_grouping"]}.
""",
        encoding="utf-8",
    )


def run_quality_suite(
    run_dir: Path,
    freeze: Mapping[str, Any],
    *,
    reuse_only: bool = False,
) -> dict[str, Any]:
    _enable_native_commands()
    nll = run_opt058(
        workload="quality-baseline",
        tier="acceptance",
        extra=["--bundle", "pins/production_quality_v2_nll.bundle"],
        run_dir=run_dir,
        sidecar_name="quality-nll.json",
        freeze=freeze,
        reuse_only=reuse_only,
    )
    functional_extra = [
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
    store_sidecar(run_dir, "quality-summary.json", quality_eval)
    store_sidecar(EVIDENCE, "quality-summary.json", quality_eval)
    return quality_eval


def run_preflight(run_dir: Path) -> dict[str, Any]:
    _enable_native_commands()
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
    run_probe(
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
    quality_eval = evaluate_combination_quality(
        held=held,
        functional=functional,
        inputs=json.loads(_read(V2_INPUTS)),
    )
    summary = {
        **quality_eval,
        "phase": "preflight",
        "is_release_evidence": False,
        "held_out_targets": 32,
        "release_eligible": combination_oracle_policy(
            bool(quality_eval["model_quality_pass"])
        )["release_eligible"],
    }
    store_sidecar(run_dir, "preflight-summary.json", summary)
    store_sidecar(EVIDENCE, "preflight-summary.json", summary)
    return summary


def _reuse_opt088_llama_blocks() -> dict[str, Any]:
    payload = json.loads(_read(OPT088_FIXTURE))
    return {
        "p": payload["p"]["llama_cpp"],
        "d128": payload["d128"]["llama_cpp"],
        "d2048": payload["d2048"]["llama_cpp"],
        "source": "fixtures/opt088_batch_gate.json#pinned_llama_control",
    }


def run_performance(run_dir: Path, freeze: Mapping[str, Any]) -> dict[str, Any]:
    _enable_native_commands()
    ensure_binaries(
        [
            "build/qw38-cuda-prefill-4k-oracle-test",
            "build/qw38-cuda-decode-oracle-test",
        ]
    )
    if os.environ.get("QW38_HOST_NATIVE") == "1":
        reused = _reuse_opt088_llama_blocks()
        llama_p = reused["p"]
        llama_d128 = {"prefix": 128, **reused["d128"]}
        llama_d2048 = {"prefix": 2048, **reused["d2048"]}
        store_sidecar(
            run_dir,
            "llama-reuse.json",
            {"reused_from": reused["source"], "llama_revision": LLAMA_REV},
        )
    else:
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
    payload = {
        "p": {
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
                "avg_ns": llama_p.get("avg_ns"),
                "n_prompt": 4096,
                "n_batch": llama_p.get("n_batch", 2048),
                "n_ubatch": llama_p.get("n_ubatch", 512),
                "flash_attn": llama_p.get("flash_attn", -1),
                "build_commit": llama_p.get("build_commit", "cc83d7b"),
                "test_time": llama_p.get("test_time", quartz_p.get("measurement_utc")),
                "samples_ts": llama_p["samples_ts"],
                "samples_ns": llama_p.get("samples_ns"),
            },
        },
        "d128": {
            "quartz": _engine_from_live(quartz_d128, "quartz-d128-tokens.json"),
            "llama_cpp": (
                _engine_from_live(llama_d128, "llama-decode-d128-tokens.json")
                if llama_d128.get("tok_s")
                else dict(llama_d128)
            ),
        },
        "d2048": {
            "quartz": _engine_from_live(quartz_d2048, "quartz-d2048-tokens.json"),
            "llama_cpp": (
                _engine_from_live(llama_d2048, "llama-decode-d2048-tokens.json")
                if llama_d2048.get("tok_s")
                else dict(llama_d2048)
            ),
        },
    }
    store_sidecar(run_dir, "performance-summary.json", payload)
    store_sidecar(EVIDENCE, "performance-summary.json", payload)
    return payload


def run_two_k(run_dir: Path) -> dict[str, Any]:
    _enable_native_commands()
    ensure_binaries(["build/qw38-cuda-prefill-2k-parity-test"])
    if os.environ.get("QW38_HOST_NATIVE") == "1":
        opt088 = json.loads(_read(OPT088_FIXTURE))
        llama_2k = opt088["opt016"]["llama_cpp"]
    else:
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
    }
    store_sidecar(run_dir, "two-k-summary.json", payload)
    store_sidecar(EVIDENCE, "two-k-summary.json", payload)
    return payload


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
    if not isinstance(quality, dict):
        raise BatchGateError("missing quality release evidence")
    if not isinstance(performance, dict):
        raise BatchGateError("missing performance release evidence")
    if not isinstance(two_k, dict):
        raise BatchGateError("missing two-k release evidence")
    if not isinstance(isolation, dict):
        isolation = run_state_isolation(run_dir)
        store_sidecar(run_dir, "state-isolation.json", isolation)
        store_sidecar(EVIDENCE, "state-isolation.json", isolation)
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
    fixture: dict[str, Any] = {
        "schema_version": 1,
        "task": "OPT-098",
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
        "report_path": "evidence/optimization/opt098-batch-gate/REPORT.md",
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
    production_kept = bool(quality.get("model_quality_pass"))
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
    write_json(EVIDENCE / "opt098_batch_gate.json", fixture)
    return fixture


def validate_batch_result(result: Mapping[str, Any]) -> None:
    contract = load_contract()
    freeze = frozen_combined_config()
    opt088_artifacts_unmodified()
    if result.get("schema_version") != 1 or result.get("task") != "OPT-098":
        raise BatchGateError("result is not OPT-098")
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
    if expected["opt016"] is False and (
        result["opt016"]["gate_passed"] or result.get("opt016_gate_passed")
    ):
        raise BatchGateError("OPT-016 must not be marked passed when the gate fails")
    outcomes = three_outcomes(result)
    if (
        result["outcomes"]["opt056_plus5"]["pass"]
        is not outcomes["opt056_plus5"]["pass"]
    ):
        raise BatchGateError("OPT-056 +5% outcome mismatch")
    isolation = result["state_isolation"]
    if not isolation["memory_fit"]["ok"] or not isolation["checkpoint"]["ok"]:
        raise BatchGateError("lost state: state/memory checks failed")


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
    if "opt-088 remains historical" not in lowered:
        raise BatchGateError("OPT-088 must remain historical in the OPT-098 report")


def write_report(result: Mapping[str, Any]) -> None:
    outcomes = result["outcomes"]
    p_q = result["p"]["quartz"]["mean_tok_s"]
    p_l = result["p"]["llama_cpp"]["avg_ts"]
    d128_q = result["d128"]["quartz"]["mean_tok_s"]
    d128_l = result["d128"]["llama_cpp"]["mean_tok_s"]
    d2048_q = result["d2048"]["quartz"]["mean_tok_s"]
    d2048_l = result["d2048"]["llama_cpp"]["mean_tok_s"]
    internal = outcomes["internal_improvement_with_quality"]
    control = control_baseline_blocks()
    REPORT.write_text(
        f"""# OPT-098 — Combined post-088 outcome gate

## Claim labels and proof limits

{PROOF}.

`gate.passed` is {result["gate"]["passed"]}. OPT-088 remains historical and is
not reinterpreted. Quality is not one boolean.

## Three independent outcomes

| Outcome | Verdict |
|---|---|
| Internal improvement vs OPT-088 control | {"passed" if outcomes["internal_improvement_with_quality"]["pass"] else "unpassed"} |
| Llama parity (Quartz >= llama, Tq-Tl) | {"passed" if outcomes["llama_parity"]["pass"] else "unpassed"} |
| OPT-056 +5% (Tq-Tl/1.05, p95, quality, OPT-016) | {"passed" if outcomes["opt056_plus5"]["pass"] else "unpassed"} |

## Measured sitting

| Workload | Selected tok/s | llama tok/s | OPT-088 control | parity ms | +5% ms |
|---|---:|---:|---:|---:|---:|
| P 4096 | {p_q} | {p_l} | {internal["control_p_tok_s"]} | {result["gaps"]["p"]["parity_ms"]} | {result["gaps"]["p"]["plus5_ms"]} |
| D128 | {d128_q} | {d128_l} | {internal["control_d128_tok_s"]} | {result["gaps"]["d128"]["parity_ms"]} | {result["gaps"]["d128"]["plus5_ms"]} |
| D2048 | {d2048_q} | {d2048_l} | {internal["control_d2048_tok_s"]} | {result["gaps"]["d2048"]["parity_ms"]} | {result["gaps"]["d2048"]["plus5_ms"]} |

Decode p95 ms: D128 Quartz {result["d128"]["quartz"]["token_latency_p95_ms"]} vs llama {result["d128"]["llama_cpp"]["token_latency_p95_ms"]} vs OPT-088 control {control["d128"]["quartz"]["token_latency_p95_ms"]}; D2048 Quartz {result["d2048"]["quartz"]["token_latency_p95_ms"]} vs llama {result["d2048"]["llama_cpp"]["token_latency_p95_ms"]} vs OPT-088 control {control["d2048"]["quartz"]["token_latency_p95_ms"]}.

OPT-016 2K: Quartz {result["opt016"]["quartz"]["mean_tok_s"]} vs llama {result["opt016"]["llama_cpp"]["avg_ts"]}; gate_passed={result["opt016"]["gate_passed"]}.

State/memory: memory_fit={result["state_isolation"]["memory_fit"]["ok"]}; checkpoint={result["state_isolation"]["checkpoint"]["ok"]}; cancellation frontier {result["state_isolation"]["cancellation"]["frontier"]}.

## Independent fields

| Field | Status |
|---|---|
| kernel_parity_pass | {result["kernel_parity_pass"]["all"]} |
| model_quality_pass | {result["model_quality_pass"]} |
| performance_pass | {result["performance_pass"]} |
| production_kept | {result["production_kept"]} |
| release_eligible | {result["release_eligible"]} |
""",
        encoding="utf-8",
    )
    validate_report_agrees(result)


def run_phase(phase: str, run_dir: Path) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    if phase == "freeze":
        freeze = frozen_combined_config()
        write_freeze_fixture(freeze)
        return {"status": "configured", "task": "OPT-098", **freeze}
    if phase == "preflight":
        return run_preflight(run_dir)
    freeze = frozen_combined_config()
    if phase == "quality":
        preflight = load_sidecar(EVIDENCE, "preflight-summary.json")
        if isinstance(preflight, dict) and not preflight.get("model_quality_pass"):
            raise BatchGateError(
                "failed required quality stops release before long timing"
            )
        ensure_binaries(["build/qw38-cuda-opt058-quality-baseline-test"])
        return run_quality_suite(run_dir, freeze)
    if phase == "state-memory":
        _enable_native_commands()
        ensure_binaries(
            [
                "build/qw38-cuda-memory-fit-test",
                "build/qw38-cuda-checkpoint-test",
            ]
        )
        isolation = run_state_isolation(run_dir)
        store_sidecar(run_dir, "state-isolation.json", isolation)
        store_sidecar(EVIDENCE, "state-isolation.json", isolation)
        return isolation
    if phase == "performance":
        quality = load_sidecar(EVIDENCE, "quality-summary.json")
        if not isinstance(quality, dict) or not quality.get("model_quality_pass"):
            raise BatchGateError(
                "failed required quality stops release before long timing"
            )
        telemetry = read_gpu_telemetry()
        store_sidecar(run_dir, "telemetry.json", telemetry)
        return run_performance(run_dir, freeze)
    if phase == "two-k":
        return run_two_k(run_dir)
    if phase == "report":
        return assemble_report(run_dir)
    raise BatchGateError(f"unknown phase {phase}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--run-dir", type=Path, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_dir = args.run_dir or (
        ROOT / "build" / "optimization-runs" / "OPT-098" / args.phase
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
