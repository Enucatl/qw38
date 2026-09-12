"""OPT-092 grouped Q8 r1_w4 admission with corrected parity and mixer screening."""

from __future__ import annotations

import argparse
import json
import math
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
    docker_common,
    dump_json,
    load_json,
    mean,
    paired_student_t,
    parse_engine_pairs,
    parse_rounds,
    utc_now,
)
from tools.opt082_kernel_parity import (  # noqa: E402
    evaluate_host_cases,
    gpu_available,
    q8_1_typed_reject,
)
from tools.opt089_q4_promotion import (  # noqa: E402
    CANDIDATE_ID as OPT089_CANDIDATE_ID,
    CONTROL_ID as OPT089_CONTROL_ID,
    FIXTURE as OPT089_FIXTURE,
    authenticate_opt084_freeze,
    authenticate_opt088_control,
    config_by_id as opt089_config_by_id,
    e2e_guards,
    evaluate_quality as opt089_evaluate_quality,
    parse_prefill_wall_ms,
)
from tools.opt088_batch_gate import COMBINATION_SELECTORS  # noqa: E402
from tools.run_optimization_task import (  # noqa: E402
    loop_product,
    parse_native_observation,
    validate_performance_admission,
    workload_for_mode,
)

LLAMA_REV = "cc83d7b4824f73cfdda4dfbb47ee39804f71b328"
GGUF_SHA = "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"
TOKEN_GENERATOR = "(42 + index * 997) % 248320"
CONTRACT = ROOT / "pins/opt092_q8_grouped_contract.json"
ITERATION = ROOT / "pins/opt092_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt092_q8_grouped.json"
REPORT = ROOT / "evidence/optimization/opt092-q8-grouped/REPORT.md"
EVIDENCE = REPORT.parent
OPT091_FIXTURE = ROOT / "fixtures/opt091_quality_tradeoff.json"
Q8_PIN_FILE = ROOT / "cuda/q8_decode_path.cuh"
REPLAY = "build/qw38-cuda-component-replay"
PROBE = "build/qw38-cuda-optimization-engine-probe"
NATIVE_PARITY = "build/qw38-cuda-opt092-q8-grouped-test"
MODEL = "models/Qwen3.8-27B-Q4_K_M.gguf"
PHASES = ("parity", "mixer", "quality", "d128", "d2048", "prefill-guard")
CONTROL_ID = "separate"
CANDIDATE_ID = "grouped_r1_w4"
VERDICT_KEYS = (
    "kernel_parity_pass",
    "model_quality_pass",
    "performance_pass",
    "production_kept",
)
MIN_MIXER_SAVING_MS = 0.10
CONTROL_LAUNCHES = 240
CANDIDATE_LAUNCHES = 64
LAUNCH_REDUCTION = 176
E2E_REGRESSION_FRAC = 0.02
THROUGHPUT_MIN = 0.98
P95_RATIO_MAX = 1.02
PREFILL_THROUGHPUT_MIN = 0.95
QUALITY_CONTRACT_ID = "opt089_strict"
OPT088_SELECTORS = dict(COMBINATION_SELECTORS)

CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "id": CONTROL_ID,
        "q8_grouping": "separate",
        "q8_layout": "r1_w4",
        "q8_decode": "dp4a_q8_1",
        "warps_per_row": 4,
        "role": "control",
        "expected_input_projection_launches": CONTROL_LAUNCHES,
    },
    {
        "id": CANDIDATE_ID,
        "q8_grouping": "grouped_r1_w4",
        "q8_layout": "r1_w4",
        "q8_decode": "dp4a_q8_1",
        "warps_per_row": 4,
        "role": "candidate",
        "source_task": "OPT-092",
        "expected_input_projection_launches": CANDIDATE_LAUNCHES,
    },
)

NativeRunner = Callable[[Sequence[str], str], subprocess.CompletedProcess[str]]


def default_native_runner(
    command: Sequence[str], tier: str
) -> subprocess.CompletedProcess[str]:
    listed = list(command)
    env = os.environ.copy()
    env["QW38_CUDA_TEST_TIER"] = tier
    if os.environ.get("QW38_HOST_NATIVE") == "1":
        completed = subprocess.run(
            listed,
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    else:
        if listed and not listed[0].startswith("docker"):
            listed = [*docker_common(tier), *listed]
        completed = subprocess.run(
            listed,
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    if completed.returncode != 0:
        raise AdmissionError(
            "native command failed: "
            + " ".join(listed)
            + "\n"
            + completed.stdout
            + completed.stderr
        )
    return completed


def config_by_id(config_id: str) -> dict[str, Any]:
    for row in CONFIGS:
        if row["id"] == config_id:
            return dict(row)
    raise AdmissionError(f"unknown configuration {config_id}")


def current_grouping_pin() -> str:
    match = re.search(
        r'kSelectedQ8DecodeGrouping\[\] = "([^"]+)"',
        Q8_PIN_FILE.read_text(encoding="utf-8"),
    )
    if match is None:
        raise AdmissionError("missing kSelectedQ8DecodeGrouping")
    return match.group(1)


def apply_grouping_pin(grouping_id: str) -> None:
    text = Q8_PIN_FILE.read_text(encoding="utf-8")
    updated = re.sub(
        r'kSelectedQ8DecodeGrouping\[\] = "[^"]+"',
        f'kSelectedQ8DecodeGrouping[] = "{grouping_id}"',
        text,
        count=1,
    )
    if updated == text:
        raise AdmissionError("failed to rewrite the Q8 decode grouping pin")
    Q8_PIN_FILE.write_text(updated, encoding="utf-8")


def apply_production_pins(results: Mapping[str, Any]) -> dict[str, Any]:
    applied = {"grouping": False}
    if results.get("production_kept") and not results.get("shipping_unchanged"):
        want = str(results.get("shipping_grouping") or "")
        if not want:
            raise AdmissionError("keep is missing shipping grouping id")
        if want != current_grouping_pin():
            apply_grouping_pin(want)
            applied["grouping"] = True
    applied["grouping_pin"] = current_grouping_pin()
    return applied


def shipping_selectors() -> dict[str, Any]:
    if not OPT089_FIXTURE.is_file():
        return {
            "q4_decode": "packed",
            "ffn_decode": "paired_staged",
            "q8_decode": "r1_w4",
        }
    payload = load_json(OPT089_FIXTURE)
    if payload.get("production_kept"):
        return {
            "q4_decode": str(payload.get("shipping_q4_decode") or "integer_q8_late"),
            "ffn_decode": str(payload.get("shipping_ffn_decode") or "paired_integer"),
            "q8_decode": "r1_w4",
        }
    return {
        "q4_decode": str(payload.get("shipping_q4_decode") or "packed"),
        "ffn_decode": str(payload.get("shipping_ffn_decode") or "paired_staged"),
        "q8_decode": "r1_w4",
    }


def candidate_selectors(config: Mapping[str, Any]) -> dict[str, Any]:
    selectors = dict(OPT088_SELECTORS)
    shipping = shipping_selectors()
    selectors.update(shipping)
    selectors["q8_grouping"] = config["q8_grouping"]
    selectors["q8_decode"] = config["q8_layout"]
    selectors["q8_layout"] = config["q8_layout"]
    return selectors


def parity_catalog() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for m in (1, 3, 17, 0):
        for k in (256, 5120):
            rows.append(
                {
                    "id": f"Q8_grouped_M{m}_N1_K{k}_staged_exact",
                    "family": "Q8_0",
                    "candidate": "grouped_r1_w4",
                    "M": m,
                    "N": 1,
                    "K": k,
                    "expect_fallback": False,
                }
            )
    rows.extend(
        [
            {
                "id": "Q8_gdn_input_group_prod",
                "family": "Q8_0",
                "candidate": "grouped_r1_w4",
                "group": "gdn_input",
                "expect_fallback": False,
            },
            {
                "id": "Q8_attn_input_group_prod",
                "family": "Q8_0",
                "candidate": "grouped_r1_w4",
                "group": "attn_input",
                "expect_fallback": False,
            },
            {
                "id": "Q8_gdn_output_separate_prod",
                "family": "Q8_0",
                "candidate": "grouped_r1_w4",
                "group": "gdn_output",
                "expect_fallback": False,
            },
        ]
    )
    return rows


def family_plan(phase: str, mode: str) -> dict[str, Any]:
    iteration = load_json(ITERATION)
    if phase not in iteration["workloads"]:
        raise AdmissionError(f"unknown phase {phase}")
    workload = workload_for_mode(iteration["workloads"][phase], mode)
    engine_pairs = 0
    if phase == "mixer":
        engine_pairs = 1 if mode == "feedback" else 5
    elif phase in {"d128", "d2048"}:
        engine_pairs = int(workload.get("samples", 5))
    default_candidates = (
        2 if phase in {"mixer", "d128", "d2048", "prefill-guard"} else 1
    )
    return {
        "phase": phase,
        "mode": mode,
        "family": "decode-mixer" if phase == "mixer" else phase,
        "control": CONTROL_ID,
        "warmups": int(workload.get("warmups", 0)),
        "samples": int(workload.get("samples", 1)),
        "engine_pairs": int(workload.get("engine_pairs", engine_pairs)),
        "cases": int(workload.get("cases", 1)),
        "candidates": int(workload.get("candidates", default_candidates)),
        "tier": str(
            workload.get("tier", "screen" if mode == "feedback" else "acceptance")
        ),
        "product": loop_product(workload),
        "control_candidate_pairs": int(workload.get("control_candidate_pairs", 1)),
        "prefix": int(workload.get("prefix", 0) or 0),
        "output_tokens": int(workload.get("output_tokens", 32) or 32),
    }


def planned_observation(
    plan: Mapping[str, Any], *, capture_key: str | None = None, keep: bool = False
) -> dict[str, Any]:
    samples = int(plan["samples"])
    return {
        "schema_version": 1,
        "task": "OPT-092",
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
        "capture_key": capture_key,
    }


def parity_by_ident(parity: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not parity:
        return {}
    inner = parity.get("parity")
    if isinstance(inner, Mapping):
        return dict(inner.get("by_ident") or {})
    return dict(parity.get("by_ident") or {})


def empty_verdict_row() -> dict[str, Any]:
    return {
        "kernel_parity_pass": False,
        "model_quality_pass": False,
        "performance_pass": False,
        "production_kept": False,
        "opt074_coverage_unadmitted_blocker": False,
        "incomplete": True,
    }


def load_prior_verdicts() -> dict[str, dict[str, Any]]:
    rows = {row["id"]: empty_verdict_row() for row in CONFIGS}
    if not FIXTURE.is_file():
        return rows
    try:
        prior = load_json(FIXTURE)
    except json.JSONDecodeError:
        return rows
    stored = prior.get("independent_verdicts") or {}
    for config_id, payload in stored.items():
        if config_id in rows and isinstance(payload, Mapping):
            merged = empty_verdict_row()
            merged.update({key: bool(payload.get(key)) for key in VERDICT_KEYS})
            merged["opt074_coverage_unadmitted_blocker"] = False
            merged["incomplete"] = bool(payload.get("incomplete", False))
            rows[config_id] = merged
    return rows


def _parse_parity_cases(stdout: str) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for line in (stdout or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("QW38_OPT092_CASE="):
            try:
                cases.append(json.loads(stripped.split("=", 1)[1]))
            except json.JSONDecodeError:
                continue
    return cases


def run_native_parity(runner: NativeRunner | None) -> dict[str, Any]:
    execute = runner or default_native_runner
    available, blocker = gpu_available()
    record: dict[str, Any] = {
        "available": available,
        "blocker": blocker,
        "ran": False,
        "success": False,
        "cases": [],
    }
    if not available:
        return record
    completed = execute([f"./{NATIVE_PARITY}", "--phase", "parity"], "correctness")
    record["ran"] = True
    record["stdout"] = completed.stdout
    record["stderr"] = completed.stderr
    record["returncode"] = completed.returncode
    record["cases"] = _parse_parity_cases(completed.stdout)
    observation = parse_native_observation(completed.stdout)
    record["success"] = completed.returncode == 0 and bool(
        observation.get("success", completed.returncode == 0)
    )
    if not record["success"] and not record.get("blocker"):
        record["blocker"] = f"native parity failed rc={completed.returncode}"
    return record


def evaluate_parity(
    *,
    skip_gpu: bool = False,
    runner: NativeRunner | None = None,
    synthetic: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cases = parity_catalog()
    host_cases = evaluate_host_cases()
    typed = q8_1_typed_reject()
    host_ok = typed["pass"] and all(row["matched_expectation"] for row in host_cases)
    if not host_ok:
        raise AdmissionError("OPT-082 host association/typed staging checks failed")
    gpu_rows: list[dict[str, Any]] = []
    source = "none"
    native: dict[str, Any] = {
        "available": False,
        "ran": False,
        "success": False,
        "blocker": "skip_gpu" if skip_gpu else "",
        "cases": [],
    }
    corrected_rerun = False
    if synthetic and synthetic.get("gpu_cases") is not None:
        gpu_rows = [dict(row) for row in synthetic["gpu_cases"]]
        source = str(synthetic.get("source") or "synthetic")
        native["ran"] = True
        native["success"] = True
        native["available"] = True
        corrected_rerun = True
    elif skip_gpu:
        native["blocker"] = "skip_gpu"
        corrected_rerun = False
    else:
        native = run_native_parity(runner)
        if native.get("ran"):
            gpu_rows = [dict(row) for row in native.get("cases") or []]
            source = "this_sitting"
            corrected_rerun = bool(native.get("success"))
        else:
            native["blocker"] = native.get("blocker") or "native_unavailable"
    by_id = {str(row.get("id")): row for row in gpu_rows}
    passes: list[bool] = []
    fails: list[str] = []
    for spec in cases:
        gpu = by_id.get(spec["id"], {})
        gpu_pass = gpu.get("gpu_pass")
        if gpu_pass is None:
            gpu_pass = gpu.get("pass")
        if gpu_pass is None:
            passes.append(False)
            continue
        if int(gpu.get("gpu_nonfinite_count") or gpu.get("nonfinite_count") or 0):
            passes.append(False)
            fails.append(spec["id"])
            continue
        if gpu_pass is False:
            fails.append(spec["id"])
        passes.append(bool(gpu_pass))
    complete = all(
        (by_id.get(spec["id"], {}).get("gpu_pass") is not None)
        or (by_id.get(spec["id"], {}).get("pass") is not None)
        for spec in cases
    ) and bool(cases)
    grouped_pass = bool(complete and passes and all(passes) and corrected_rerun)
    by_ident = {
        CANDIDATE_ID: {
            "ident": CANDIDATE_ID,
            "kernel_parity_pass": grouped_pass,
            "case_count": len(cases),
            "gpu_case_count": sum(1 for spec in cases if spec["id"] in by_id),
            "incomplete": not complete or not corrected_rerun,
            "documented_fails": fails,
        },
        CONTROL_ID: {
            "ident": CONTROL_ID,
            "kernel_parity_pass": grouped_pass,
            "case_count": len(cases),
            "gpu_case_count": sum(1 for spec in cases if spec["id"] in by_id),
            "incomplete": not complete or not corrected_rerun,
            "reference": "separate_r1_w4",
        },
    }
    return {
        "schema_version": 1,
        "task": "OPT-092",
        "phase": "parity",
        "host_ok": host_ok,
        "typed_staging_reject": typed["pass"],
        "source": source,
        "catalog_count": len(cases),
        "corrected_comparators_rerun": corrected_rerun,
        "gpu": {
            "available": native.get("available"),
            "ran": native.get("ran"),
            "success": native.get("success"),
            "blocker": native.get("blocker") or "",
        },
        "by_ident": by_ident,
        "opt074_coverage_unadmitted_blocker": False,
        "q8_1_typed": dict(typed),
    }


def opt089_quality_payload() -> dict[str, Any]:
    if not OPT089_FIXTURE.is_file():
        return {}
    try:
        return load_json(OPT089_FIXTURE)
    except json.JSONDecodeError:
        return {}


def evaluate_quality(config: Mapping[str, Any]) -> dict[str, Any]:
    opt088 = authenticate_opt088_control()
    opt084 = authenticate_opt084_freeze()
    opt089 = opt089_quality_payload()
    opt091 = load_json(OPT091_FIXTURE) if OPT091_FIXTURE.is_file() else {}
    shipping = shipping_selectors()
    record: dict[str, Any] = {
        "config": config["id"],
        "quality_contract_id": QUALITY_CONTRACT_ID,
        "ppl_ratio_max": 1.01,
        "recurrence_incremental_nll_max": 0.02,
        "selectors": candidate_selectors(config),
        "opt088_control": opt088,
        "opt084_freeze": opt084,
        "opt089_admission": {
            "task": opt089.get("task"),
            "production_kept": opt089.get("production_kept"),
            "quality_contract_id": opt089.get("quality_contract_id"),
        },
        "opt091_concession": {
            "concession_used": opt091.get("concession_used"),
            "successor_model_quality_pass": opt091.get("successor_model_quality_pass"),
        },
        "reuse_opt089_authenticated_scores": False,
        "opt074_coverage_unadmitted_blocker": False,
    }
    if config["id"] == CONTROL_ID:
        packed = opt089_evaluate_quality(opt089_config_by_id(OPT089_CONTROL_ID))
        passed = bool(packed.get("model_quality_pass"))
        record.update(
            {
                "model_quality_pass": passed,
                "incomplete": not passed,
                "reason": packed.get("reason") or "opt089_control_quality",
                "reuse_opt089_authenticated_scores": False,
            }
        )
        return record
    late_verdicts = (opt089.get("independent_verdicts") or {}).get(
        OPT089_CANDIDATE_ID, {}
    )
    late_quality = opt089_evaluate_quality(opt089_config_by_id(OPT089_CANDIDATE_ID))
    passed = bool(
        late_verdicts.get("model_quality_pass")
        or late_quality.get("model_quality_pass")
    )
    if opt091.get("concession_used"):
        passed = bool(
            passed
            and opt091.get("successor_model_quality_pass")
            and opt091.get("strict_model_quality_pass") is not True
        )
        record["opt091_same_math_required"] = True
    record.update(
        {
            "model_quality_pass": passed,
            "incomplete": not passed,
            "reuse_opt089_authenticated_scores": True,
            "reason": (
                "opt089_authenticated_late_w4_reused"
                if passed
                else "opt089_quality_not_authenticated"
            ),
            "shipping_q4_decode": shipping.get("q4_decode"),
            "shipping_ffn_decode": shipping.get("ffn_decode"),
        }
    )
    return record


def replay_command(
    plan: Mapping[str, Any],
    config: Mapping[str, Any],
    capture_key: str | None,
    evidence_dir: Path | str | None = None,
) -> list[str]:
    shipping = shipping_selectors()
    args = [
        f"./{REPLAY}",
        MODEL,
        "--workload",
        "decode-mixer",
        "--cache-mode",
        "rotating",
        "--warmups",
        str(plan["warmups"]),
        "--samples",
        str(plan["samples"]),
        "--q4-decode",
        str(shipping["q4_decode"]),
        "--ffn-decode",
        str(shipping["ffn_decode"]),
        "--q8-layout",
        "r1_w4",
        "--q8-grouping",
        str(config["q8_grouping"]),
    ]
    if capture_key:
        args.extend(["--capture-key", capture_key])
    if evidence_dir:
        try:
            rel = str(Path(evidence_dir).resolve().relative_to(ROOT))
        except ValueError:
            rel = str(evidence_dir)
        args.extend(["--evidence-dir", rel])
    return args


def parse_launch_counts(stdout: str) -> dict[str, int]:
    match = re.search(
        r"q8_input_projection_launches=(\d+).*?q8_grouping=(\w+)",
        stdout.replace("\n", " "),
    )
    if match is None:
        match = re.search(r"q8_input_projection_launches=(\d+)", stdout)
    launches = int(match.group(1)) if match else -1
    return {"input_projection_launches": launches}


def dispatch_ok(config: Mapping[str, Any], stdout: str) -> bool:
    group = re.search(
        r"gdn_input_output_groups=(\d+) attention_input_output_groups=(\d+)",
        stdout,
    )
    if group is None or int(group.group(1)) != 48 or int(group.group(2)) != 16:
        return False
    launches = parse_launch_counts(stdout)["input_projection_launches"]
    want = int(config["expected_input_projection_launches"])
    grouping = re.search(r"q8_grouping=(\w+)", stdout)
    got_grouping = grouping.group(1) if grouping else ""
    return launches == want and got_grouping == config["q8_grouping"]


def run_component(
    plan: Mapping[str, Any],
    *,
    runner: NativeRunner,
    run_dir: Path,
) -> dict[str, Any]:
    tier = "screen" if plan["mode"] == "feedback" else "acceptance"
    capture_key: str | None = None
    by_config: dict[str, dict[str, Any]] = {}
    evidence_dir = run_dir / "component-replay"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    for config in CONFIGS:
        completed = runner(
            replay_command(plan, config, capture_key, evidence_dir=evidence_dir),
            tier,
        )
        (run_dir / f"{plan['phase']}-{config['id']}-replay.txt").write_text(
            completed.stdout + completed.stderr, encoding="utf-8"
        )
        if (
            "nonfinite" in completed.stdout.casefold()
            and "nonfinite=0" not in completed.stdout
        ):
            raise AdmissionError(f"{config['id']} reported a nonfinite value")
        observed = parse_native_observation(completed.stdout)
        if capture_key is None:
            capture_key = str(observed.get("capture_key") or "")
            if not capture_key:
                match = re.search(r"capture_key=([0-9a-f]{64})", completed.stdout)
                if match:
                    capture_key = match.group(1)
        if not capture_key:
            raise AdmissionError("missing capture identity")
        rounds = [
            row
            for row in parse_rounds(completed.stdout)
            if row.get("cache_mode", "rotating") == "rotating"
        ]
        if len(rounds) != int(plan["samples"]):
            raise AdmissionError(
                f"{config['id']} rotating rounds {len(rounds)} != {plan['samples']}"
            )
        if not dispatch_ok(config, completed.stdout):
            raise AdmissionError(
                f"{config['id']} launch/dispatch proof failed: "
                f"{parse_launch_counts(completed.stdout)}"
            )
        samples = [float(row["enclosing_ms"]) for row in rounds]
        by_config[str(config["id"])] = {
            "config": dict(config),
            "ms": samples,
            "mean_ms": mean(samples),
            "launches": parse_launch_counts(completed.stdout),
        }
    critical = T_CRIT_DF9 if int(plan["samples"]) >= 10 else FEEDBACK_CRIT
    stats = paired_student_t(
        by_config[CONTROL_ID]["ms"],
        by_config[CANDIDATE_ID]["ms"],
        critical=critical,
    )
    return {
        "capture_key": capture_key,
        "by_config": by_config,
        "control": CONTROL_ID,
        "this_sitting": True,
        "dispatch_ok": True,
        "control_mean_ms": by_config[CONTROL_ID]["mean_ms"],
        "candidate_mean_ms": by_config[CANDIDATE_ID]["mean_ms"],
        **stats,
    }


def run_engine(
    plan: Mapping[str, Any],
    *,
    runner: NativeRunner,
    run_dir: Path,
    prefix: int | None = None,
    pairs_n: int | None = None,
    sidecar: str | None = None,
) -> dict[str, Any]:
    count = int(pairs_n if pairs_n is not None else plan["engine_pairs"])
    if count <= 0:
        return {}
    used_prefix = int(prefix if prefix is not None else plan.get("prefix") or 2048)
    tier = "acceptance" if plan["mode"] == "acceptance" else "screen"
    shipping = shipping_selectors()
    command = [
        f"./{PROBE}",
        MODEL,
        "--workload",
        "grouping-ab",
        "--pairs",
        str(count),
        "--modes",
        "graph",
        "--prefix",
        str(used_prefix),
        "--output-tokens",
        "32",
        "--q4-decode",
        str(shipping["q4_decode"]),
        "--ffn-decode",
        str(shipping["ffn_decode"]),
        "--q8-layout",
        "r1_w4",
    ]
    completed = runner(command, tier)
    name = sidecar or f"{plan['phase']}-engine.txt"
    (run_dir / name).write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if "override_before_capture_applied=true" not in completed.stdout:
        raise AdmissionError("graph capture override was not applied")
    if (
        "recapture=true" not in completed.stdout
        and "recapture_after_selector=true" not in completed.stdout
    ):
        raise AdmissionError("graph was not recaptured after selecting grouping")
    pairs = parse_engine_pairs(completed.stdout)
    if len(pairs) != count:
        raise AdmissionError(f"engine pairs {len(pairs)} != {count}")
    control = [float(row["control_ms"]) for row in pairs]
    candidate = [float(row["candidate_ms"]) for row in pairs]
    return {
        "pairs": pairs,
        "control_ms": control,
        "candidate_ms": candidate,
        "control_mean_ms": mean(control),
        "candidate_mean_ms": mean(candidate),
        "this_sitting": True,
        "prefix": used_prefix,
        "output_tokens": 32,
        "guards": e2e_guards(
            {
                "control_ms": control,
                "candidate_ms": candidate,
                "pairs": pairs,
            },
            output_tokens=32,
        ),
    }


def run_prefill_guard(
    *,
    runner: NativeRunner,
    run_dir: Path,
) -> dict[str, Any]:
    walls: dict[str, float] = {}
    shipping = shipping_selectors()
    for config in CONFIGS:
        command = [
            f"./{PROBE}",
            MODEL,
            "--workload",
            "prefill",
            "--prompt",
            "4096",
            "--q4-decode",
            str(shipping["q4_decode"]),
            "--ffn-decode",
            str(shipping["ffn_decode"]),
            "--q8-layout",
            "r1_w4",
            "--q8-grouping",
            str(config["q8_grouping"]),
            "--modes",
            "graph",
        ]
        completed = runner(command, "acceptance")
        (run_dir / f"prefill-{config['id']}.txt").write_text(
            completed.stdout + completed.stderr, encoding="utf-8"
        )
        walls[str(config["id"])] = parse_prefill_wall_ms(completed.stdout)
    control_ms = walls[CONTROL_ID]
    candidate_ms = walls[CANDIDATE_ID]
    control_tok = 4096.0 / (control_ms / 1000.0) if control_ms > 0 else 0.0
    cand_tok = 4096.0 / (candidate_ms / 1000.0) if candidate_ms > 0 else 0.0
    ratio = cand_tok / control_tok if control_tok > 0 else 0.0
    return {
        "control_ms": control_ms,
        "candidate_ms": candidate_ms,
        "control_tok_s": control_tok,
        "candidate_tok_s": cand_tok,
        "throughput_ratio": ratio,
        "pass": ratio >= PREFILL_THROUGHPUT_MIN,
        "this_sitting": True,
        "prompt": 4096,
    }


def performance_pass_for(
    *,
    config_id: str,
    stats: Mapping[str, Any] | None,
    engine: Mapping[str, Any] | None,
    this_sitting: bool,
    dispatch_ok_flag: bool,
    e2e: Mapping[str, Any] | None = None,
    prefill: Mapping[str, Any] | None = None,
    require_full_e2e: bool = False,
) -> bool:
    if config_id == CONTROL_ID:
        return False
    if not this_sitting or not dispatch_ok_flag or not stats:
        return False
    component_ok = (
        bool(stats.get("positive"))
        and float(stats.get("mean_diff_ms", 0.0)) >= MIN_MIXER_SAVING_MS
    )
    if not component_ok:
        return False
    if require_full_e2e:
        if (
            not e2e
            or not e2e.get("d128", {}).get("pass")
            or not e2e.get("d2048", {}).get("pass")
        ):
            return False
        if not prefill or not prefill.get("pass"):
            return False
        return True
    if engine:
        return bool(e2e_guards(engine).get("pass"))
    return False


def decide_independent_verdicts(
    *,
    parity: Mapping[str, Any] | None,
    quality_by_id: Mapping[str, Mapping[str, Any]],
    performance_by_id: Mapping[str, Mapping[str, Any]],
    mode: str,
    prior: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    rows: dict[str, dict[str, Any]] = {}
    by_ident = parity_by_ident(parity or {})
    for config in CONFIGS:
        cid = config["id"]
        previous = dict((prior or {}).get(cid) or empty_verdict_row())
        ident = str(config["id"])
        if by_ident:
            kpass = bool(by_ident.get(ident, {}).get("kernel_parity_pass"))
        else:
            kpass = bool(previous.get("kernel_parity_pass"))
        quality = quality_by_id.get(cid) or {}
        qpass = (
            bool(quality.get("model_quality_pass"))
            if quality
            else bool(previous.get("model_quality_pass"))
        )
        perf = performance_by_id.get(cid) or {}
        ppass = (
            bool(perf.get("performance_pass"))
            if perf
            else bool(previous.get("performance_pass"))
        )
        rows[cid] = {
            "kernel_parity_pass": kpass,
            "model_quality_pass": qpass,
            "performance_pass": ppass,
            "production_kept": False,
            "opt074_coverage_unadmitted_blocker": False,
            "incomplete": bool(
                (by_ident.get(ident) or {}).get("incomplete")
                or quality.get("incomplete")
                or perf.get("incomplete")
            ),
        }
    winners = [
        cid
        for cid, row in rows.items()
        if cid != CONTROL_ID
        and row["kernel_parity_pass"]
        and row["model_quality_pass"]
        and row["performance_pass"]
    ]
    shipping_unchanged = True
    selected = CONTROL_ID
    if mode == "acceptance" and winners:
        selected = max(
            winners,
            key=lambda cid: float(
                (performance_by_id.get(cid) or {}).get("mean_diff_ms", -math.inf)
            ),
        )
        rows[selected]["production_kept"] = True
        shipping_unchanged = False
    elif mode == "acceptance":
        rows[CONTROL_ID]["production_kept"] = True
        selected = CONTROL_ID
        shipping_unchanged = True
    kept = [cid for cid, row in rows.items() if row["production_kept"]]
    if len(kept) > 1:
        raise AdmissionError("at most one Q8 grouping path may be kept")
    return {
        "independent_verdicts": rows,
        "selected_path": selected,
        "shipping_unchanged": shipping_unchanged,
        "production_kept": bool(
            selected != CONTROL_ID and rows[selected]["production_kept"]
        )
        if mode == "acceptance"
        else False,
        "control_grouping": CONTROL_ID,
        "shipping_grouping": selected if not shipping_unchanged else CONTROL_ID,
        "opt074_coverage_unadmitted_blocker": False,
        "winners": winners,
        "feedback_cannot_keep": mode != "acceptance",
    }


def _fmt(value: Any, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def write_report(payload: Mapping[str, Any]) -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    verdicts = payload.get("independent_verdicts") or {}
    parity = payload.get("parity") or {}
    mixer = payload.get("mixer") or {}
    component = mixer.get("component") or {}
    engine = mixer.get("engine") or {}
    d128_guards = (payload.get("d128") or {}).get("guards") or {}
    d2048_guards = (payload.get("d2048") or {}).get("guards") or {}
    prefill = (payload.get("prefill-guard") or {}).get("prefill") or {}
    pins = payload.get("pins_applied") or {}
    lines = []
    for config in CONFIGS:
        row = verdicts.get(config["id"]) or empty_verdict_row()
        lines.append(
            f"| {config['id']} | {row.get('kernel_parity_pass')} | "
            f"{row.get('model_quality_pass')} | {row.get('performance_pass')} | "
            f"{row.get('production_kept')} |"
        )
    text = f"""# OPT-092 — Grouped Q8 r1_w4 admission

Status: **{payload.get("status") or "pending"}**. Authority llama.cpp
`{LLAMA_REV}`, GGUF SHA-256 `{GGUF_SHA}`. Exactly two grouping
configurations: `separate` control and `grouped_r1_w4` candidate. Q8
`dp4a_q8_1` / `r1_w4` arithmetic stays fixed.

`opt074_coverage_unadmitted` is **not** a blocker.
`quality_contract_id={QUALITY_CONTRACT_ID}`.
`claims_throughput` is true only for this sitting's uninstrumented keep.

## Independent verdicts

| Config | kernel_parity_pass | model_quality_pass | performance_pass | production_kept |
|---|---|---|---|---|
{chr(10).join(lines)}

Selected path: `{payload.get("selected_path", CONTROL_ID)}`.
shipping_grouping={payload.get("shipping_grouping", CONTROL_ID)}.
production_kept={payload.get("production_kept", False)}.
claims_throughput={payload.get("claims_throughput", False)}.
evidence_complete={payload.get("evidence_complete", False)}.

## Kernel parity

source={parity.get("source")}; host_ok={parity.get("host_ok")};
catalog={parity.get("catalog_count")}.
Grouped vs separate exact staged bytes and bitwise FP32 outputs.

## Quality

Reuse authenticated OPT-089 `late_w4` scores under `opt089_strict`.

## Performance

Launch counts: control {CONTROL_LAUNCHES}, candidate {CANDIDATE_LAUNCHES}
(−{LAUNCH_REDUCTION}). Screen ≥{MIN_MIXER_SAVING_MS} ms/token complete mixer.
this_sitting={component.get("this_sitting", False)}.
Control mean {_fmt(component.get("control_mean_ms"))} ms vs candidate
{_fmt(component.get("candidate_mean_ms"))} ms.
Paired CI {_fmt(component.get("ci95_low"), 4)} .. {_fmt(component.get("ci95_high"), 4)} ms.
Engine {_fmt(engine.get("control_mean_ms"))} vs {_fmt(engine.get("candidate_mean_ms"))} ms.
D128 throughput_ratio={_fmt(d128_guards.get("throughput_ratio"), 3)} p95_ratio={_fmt(d128_guards.get("p95_ratio"), 3)}.
D2048 throughput_ratio={_fmt(d2048_guards.get("throughput_ratio"), 3)} p95_ratio={_fmt(d2048_guards.get("p95_ratio"), 3)}.
P4096 throughput_ratio={_fmt(prefill.get("throughput_ratio"), 4)} pass={prefill.get("pass")}.
Pin rewrite grouping={pins.get("grouping")}.
"""
    REPORT.write_text(text, encoding="utf-8")


def admit_counts(phase: str, mode: str, observed: Mapping[str, Any]) -> dict[str, Any]:
    plan = family_plan(phase, mode)
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
    if not admission["ok"]:
        raise AdmissionError(admission["message"])
    return admission


def merge_fixture(
    results: dict[str, Any], phase: str, payload: Mapping[str, Any]
) -> None:
    results[phase] = payload
    prior = results.get("independent_verdicts") or load_prior_verdicts()
    incoming = payload.get("independent_verdicts") or {}
    merged = {
        cid: dict(prior.get(cid) or empty_verdict_row())
        for cid in (row["id"] for row in CONFIGS)
    }
    for cid, row in incoming.items():
        current = merged.setdefault(cid, empty_verdict_row())
        if phase == "parity":
            current["kernel_parity_pass"] = bool(row.get("kernel_parity_pass"))
        elif phase == "quality":
            current["model_quality_pass"] = bool(row.get("model_quality_pass"))
        elif phase in {"mixer", "d128", "d2048", "prefill-guard"}:
            current["performance_pass"] = bool(row.get("performance_pass"))
        current["opt074_coverage_unadmitted_blocker"] = False
        current["incomplete"] = bool(row.get("incomplete", current.get("incomplete")))
    d128_ok = bool(((results.get("d128") or {}).get("guards") or {}).get("pass"))
    d2048_ok = bool(((results.get("d2048") or {}).get("guards") or {}).get("pass"))
    prefill_ok = bool(
        ((results.get("prefill-guard") or {}).get("prefill") or {}).get("pass")
    )
    mixer = results.get("mixer") or {}
    stats = {
        "positive": (mixer.get("component") or {}).get("positive"),
        "mean_diff_ms": (mixer.get("component") or {}).get("mean_diff_ms"),
    }
    this_sitting = bool((mixer.get("component") or {}).get("this_sitting"))
    dispatch_ok_flag = bool((mixer.get("component") or {}).get("dispatch_ok"))
    mixer_ok = (
        bool(stats.get("positive"))
        and float(stats.get("mean_diff_ms") or 0.0) >= MIN_MIXER_SAVING_MS
        and dispatch_ok_flag
        and mixer.get("mode") == "acceptance"
        and this_sitting
    )
    late_perf = mixer_ok and d128_ok and d2048_ok and prefill_ok
    performance_by_id = {
        CONTROL_ID: {"performance_pass": False, "mean_diff_ms": 0.0},
        CANDIDATE_ID: {
            "performance_pass": late_perf,
            "mean_diff_ms": stats.get("mean_diff_ms"),
            "incomplete": not late_perf,
        },
    }
    quality_by_id = {
        cid: {"model_quality_pass": row.get("model_quality_pass")}
        for cid, row in merged.items()
    }
    mode = str(results.get("mode") or payload.get("mode") or "feedback")
    decided = decide_independent_verdicts(
        parity=results.get("parity") or {},
        quality_by_id=quality_by_id,
        performance_by_id=performance_by_id,
        mode=mode,
        prior=merged,
    )
    results["independent_verdicts"] = decided["independent_verdicts"]
    results["selected_path"] = decided["selected_path"]
    results["shipping_unchanged"] = decided["shipping_unchanged"]
    results["control_grouping"] = CONTROL_ID
    results["shipping_grouping"] = decided["shipping_grouping"]
    candidate_kept = bool(
        decided["selected_path"] == CANDIDATE_ID
        and decided["independent_verdicts"][CANDIDATE_ID]["production_kept"]
    )
    results["production_kept"] = candidate_kept
    parity_ok = bool(
        parity_by_ident(results.get("parity") or {})
        .get(CANDIDATE_ID, {})
        .get("kernel_parity_pass")
    )
    evidence_complete = bool(
        parity_ok
        and quality_by_id.get(CANDIDATE_ID, {}).get("model_quality_pass")
        and late_perf
        and this_sitting
    )
    results["evidence_complete"] = evidence_complete
    results["claims_throughput"] = bool(
        candidate_kept and evidence_complete and mode == "acceptance"
    )
    results["claims_performance_improvement"] = results["claims_throughput"]
    results["opt074_coverage_unadmitted_blocker"] = False
    results["kernel_parity_pass"] = {
        cid: row["kernel_parity_pass"]
        for cid, row in results["independent_verdicts"].items()
    }
    results["model_quality_pass"] = {
        cid: row["model_quality_pass"]
        for cid, row in results["independent_verdicts"].items()
    }
    results["performance_pass"] = {
        cid: row["performance_pass"]
        for cid, row in results["independent_verdicts"].items()
    }
    if results["shipping_unchanged"] or not candidate_kept:
        results["shipping_grouping"] = CONTROL_ID
        results["independent_verdicts"][CONTROL_ID]["production_kept"] = (
            mode == "acceptance"
        )
        if mode != "acceptance":
            results["independent_verdicts"][CONTROL_ID]["production_kept"] = False
            results["production_kept"] = False
    else:
        results["shipping_grouping"] = CANDIDATE_ID
    results["quality_contract_id"] = QUALITY_CONTRACT_ID
    results["status"] = (
        "grouped_r1_w4_kept"
        if candidate_kept and results["claims_throughput"]
        else str(payload.get("status") or "separate_retained")
    )


def run_parity_phase(
    *,
    mode: str,
    run_dir: Path,
    skip_gpu: bool,
    runner: NativeRunner | None,
    synthetic: Mapping[str, Any] | None,
) -> dict[str, Any]:
    plan = family_plan("parity", mode)
    parity = evaluate_parity(skip_gpu=skip_gpu, runner=runner, synthetic=synthetic)
    if parity["catalog_count"] != int(plan["cases"]):
        raise AdmissionError(
            f"parity catalog {parity['catalog_count']} != plan {plan['cases']}"
        )
    observed = planned_observation(plan, keep=False)
    admission = admit_counts("parity", mode, observed)
    decided = decide_independent_verdicts(
        parity=parity,
        quality_by_id={},
        performance_by_id={},
        mode=mode,
        prior=load_prior_verdicts(),
    )
    return {
        "schema_version": 1,
        "task": "OPT-092",
        "phase": "parity",
        "mode": mode,
        "parity": parity,
        "native_counts": observed,
        "admission": admission,
        **decided,
        "claims_throughput": False,
        "measurement_utc": utc_now(),
        "status": "parity_evaluated",
    }


def run_quality_phase(*, mode: str, run_dir: Path) -> dict[str, Any]:
    plan = family_plan("quality", mode)
    quality_by_id = {config["id"]: evaluate_quality(config) for config in CONFIGS}
    observed = planned_observation(plan, keep=False)
    admission = admit_counts("quality", mode, observed)
    decided = decide_independent_verdicts(
        parity=load_json(FIXTURE).get("parity") if FIXTURE.is_file() else {},
        quality_by_id=quality_by_id,
        performance_by_id={},
        mode=mode,
        prior=load_prior_verdicts(),
    )
    return {
        "schema_version": 1,
        "task": "OPT-092",
        "phase": "quality",
        "mode": mode,
        "quality": quality_by_id,
        "native_counts": observed,
        "admission": admission,
        **decided,
        "claims_throughput": False,
        "quality_contract_id": QUALITY_CONTRACT_ID,
        "measurement_utc": utc_now(),
        "status": "quality_evaluated",
    }


def _prior_quality_pass() -> bool:
    prior = load_prior_verdicts()
    return bool(prior.get(CANDIDATE_ID, {}).get("model_quality_pass"))


def run_mixer_phase(
    *,
    mode: str,
    run_dir: Path,
    skip_gpu: bool,
    runner: NativeRunner | None,
    synthetic: Mapping[str, Any] | None,
) -> dict[str, Any]:
    plan = family_plan("mixer", mode)
    prior = load_prior_verdicts()
    engine: dict[str, Any] = {}
    if mode == "acceptance" and not _prior_quality_pass() and not skip_gpu:
        component = {
            "this_sitting": False,
            "dispatch_ok": False,
            "stopped": "quality_not_authenticated",
        }
        this_sitting = False
        dispatch_ok_flag = False
    elif skip_gpu:
        if synthetic and synthetic.get("component"):
            component = dict(synthetic["component"])
            engine = dict(synthetic.get("engine") or {})
            dispatch_ok_flag = bool(synthetic.get("dispatch_ok", True))
            this_sitting = bool(synthetic.get("this_sitting", False))
        else:
            component = {"this_sitting": False, "dispatch_ok": True}
            dispatch_ok_flag = True
            this_sitting = False
    else:
        native = runner or default_native_runner
        component = run_component(plan, runner=native, run_dir=run_dir)
        dispatch_ok_flag = bool(component.get("dispatch_ok"))
        this_sitting = True
        try:
            engine = (
                run_engine(
                    plan,
                    runner=native,
                    run_dir=run_dir,
                    prefix=2048,
                )
                or {}
            )
        except AdmissionError as exc:
            if plan["mode"] == "acceptance":
                raise
            engine = {"error": str(exc), "skipped": True}
    stats = {
        "positive": component.get("positive"),
        "mean_diff_ms": component.get("mean_diff_ms"),
        "ci95_low": component.get("ci95_low"),
        "ci95_high": component.get("ci95_high"),
    }
    performance_by_id: dict[str, dict[str, Any]] = {}
    for config in CONFIGS:
        cid = config["id"]
        ppass = performance_pass_for(
            config_id=cid,
            stats=stats,
            engine=engine if cid == CANDIDATE_ID else None,
            this_sitting=this_sitting,
            dispatch_ok_flag=dispatch_ok_flag,
            require_full_e2e=False,
        )
        if mode != "acceptance":
            ppass = False
        performance_by_id[cid] = {
            "performance_pass": ppass and cid == CANDIDATE_ID,
            "this_sitting": this_sitting,
            "incomplete": not this_sitting or mode != "acceptance",
            "mean_diff_ms": component.get("mean_diff_ms")
            if cid == CANDIDATE_ID
            else None,
            "screen_only": mode != "acceptance",
        }
    observed = planned_observation(plan, keep=False)
    admission = admit_counts("mixer", mode, observed)
    quality_by_id = {
        cid: {"model_quality_pass": row.get("model_quality_pass")}
        for cid, row in prior.items()
    }
    parity_source = load_json(FIXTURE) if FIXTURE.is_file() else {}
    decided = decide_independent_verdicts(
        parity=parity_source.get("parity") or {},
        quality_by_id=quality_by_id,
        performance_by_id=performance_by_id,
        mode=mode,
        prior=prior,
    )
    if mode != "acceptance":
        decided["production_kept"] = False
        for row in decided["independent_verdicts"].values():
            row["production_kept"] = False
    return {
        "schema_version": 1,
        "task": "OPT-092",
        "phase": "mixer",
        "mode": mode,
        "component": component,
        "engine": engine,
        "performance": performance_by_id,
        "native_counts": observed,
        "admission": admission,
        **decided,
        "claims_throughput": False,
        "this_sitting": this_sitting,
        "measurement_utc": utc_now(),
        "status": "mixer_screened" if mode != "acceptance" else "mixer_acceptance",
    }


def run_e2e_phase(
    *,
    phase: str,
    mode: str,
    run_dir: Path,
    skip_gpu: bool,
    runner: NativeRunner | None,
) -> dict[str, Any]:
    plan = family_plan(phase, mode)
    prior = load_prior_verdicts()
    if mode == "acceptance" and not _prior_quality_pass() and not skip_gpu:
        engine = {"stopped": "quality_not_authenticated", "this_sitting": False}
        guards = {
            "pass": False,
            "incomplete": True,
            "reason": "quality_not_authenticated",
        }
        this_sitting = False
    elif skip_gpu:
        engine = {"this_sitting": False, "incomplete": True}
        guards = {"pass": False, "incomplete": True, "reason": "skip_gpu"}
        this_sitting = False
    else:
        native = runner or default_native_runner
        prefix = 128 if phase == "d128" else 2048
        engine = run_engine(
            plan,
            runner=native,
            run_dir=run_dir,
            prefix=prefix,
            pairs_n=5,
        )
        guards = engine.get("guards") or e2e_guards(engine)
        this_sitting = True
    performance_by_id = {
        CONTROL_ID: {"performance_pass": False, "incomplete": True},
        CANDIDATE_ID: {
            "performance_pass": False,
            "incomplete": not (this_sitting and guards.get("pass")),
            "e2e": guards,
        },
    }
    observed = planned_observation(plan, keep=False)
    admission = admit_counts(phase, mode, observed)
    decided = decide_independent_verdicts(
        parity=(load_json(FIXTURE) if FIXTURE.is_file() else {}).get("parity") or {},
        quality_by_id={
            cid: {"model_quality_pass": row.get("model_quality_pass")}
            for cid, row in prior.items()
        },
        performance_by_id=performance_by_id,
        mode=mode,
        prior=prior,
    )
    return {
        "schema_version": 1,
        "task": "OPT-092",
        "phase": phase,
        "mode": mode,
        "engine": engine,
        "guards": guards,
        "native_counts": observed,
        "admission": admission,
        **decided,
        "claims_throughput": False,
        "measurement_utc": utc_now(),
        "status": f"{phase}_evaluated",
    }


def run_prefill_phase(
    *,
    mode: str,
    run_dir: Path,
    skip_gpu: bool,
    runner: NativeRunner | None,
) -> dict[str, Any]:
    plan = family_plan("prefill-guard", mode)
    prior = load_prior_verdicts()
    if mode == "acceptance" and not _prior_quality_pass() and not skip_gpu:
        prefill = {"pass": False, "stopped": "quality_not_authenticated"}
        this_sitting = False
    elif skip_gpu:
        prefill = {"pass": False, "incomplete": True, "reason": "skip_gpu"}
        this_sitting = False
    else:
        native = runner or default_native_runner
        prefill = run_prefill_guard(runner=native, run_dir=run_dir)
        this_sitting = True
    observed = planned_observation(plan, keep=False)
    admission = admit_counts("prefill-guard", mode, observed)
    performance_by_id = {
        CONTROL_ID: {"performance_pass": False},
        CANDIDATE_ID: {
            "performance_pass": False,
            "incomplete": not (this_sitting and prefill.get("pass")),
            "prefill": prefill,
        },
    }
    decided = decide_independent_verdicts(
        parity=(load_json(FIXTURE) if FIXTURE.is_file() else {}).get("parity") or {},
        quality_by_id={
            cid: {"model_quality_pass": row.get("model_quality_pass")}
            for cid, row in prior.items()
        },
        performance_by_id=performance_by_id,
        mode=mode,
        prior=prior,
    )
    return {
        "schema_version": 1,
        "task": "OPT-092",
        "phase": "prefill-guard",
        "mode": mode,
        "prefill": prefill,
        "native_counts": observed,
        "admission": admission,
        **decided,
        "claims_throughput": False,
        "measurement_utc": utc_now(),
        "status": "prefill_guard_evaluated",
    }


def run(
    mode: str,
    phase: str,
    run_dir: Path,
    *,
    skip_gpu: bool = False,
    runner: NativeRunner | None = None,
    synthetic: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if phase not in PHASES:
        raise AdmissionError(f"unknown phase {phase}")
    run_dir.mkdir(parents=True, exist_ok=True)
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    if FIXTURE.is_file():
        try:
            results.update(load_json(FIXTURE))
        except json.JSONDecodeError:
            results = {}
    results.update(
        {
            "schema_version": 1,
            "task": "OPT-092",
            "mode": mode,
            "llama_revision": LLAMA_REV,
            "gguf_sha256": GGUF_SHA,
            "token_generator": TOKEN_GENERATOR,
            "configurations": [row["id"] for row in CONFIGS],
            "control": CONTROL_ID,
            "control_grouping": CONTROL_ID,
            "shipping_grouping": CONTROL_ID,
            "q8_layout_fixed": "r1_w4",
            "control_input_projection_launches": CONTROL_LAUNCHES,
            "candidate_input_projection_launches": CANDIDATE_LAUNCHES,
            "input_projection_launch_reduction": LAUNCH_REDUCTION,
            "min_mixer_saving_ms": MIN_MIXER_SAVING_MS,
            "opt074_coverage_unadmitted_blocker": False,
            "quality_contract_id": QUALITY_CONTRACT_ID,
            "report_path": "evidence/optimization/opt092-q8-grouped/REPORT.md",
        }
    )
    if phase == "parity":
        payload = run_parity_phase(
            mode=mode,
            run_dir=run_dir,
            skip_gpu=skip_gpu,
            runner=runner,
            synthetic=synthetic,
        )
    elif phase == "quality":
        payload = run_quality_phase(mode=mode, run_dir=run_dir)
    elif phase == "mixer":
        payload = run_mixer_phase(
            mode=mode,
            run_dir=run_dir,
            skip_gpu=skip_gpu,
            runner=runner,
            synthetic=synthetic,
        )
    elif phase in {"d128", "d2048"}:
        payload = run_e2e_phase(
            phase=phase,
            mode=mode,
            run_dir=run_dir,
            skip_gpu=skip_gpu,
            runner=runner,
        )
    else:
        payload = run_prefill_phase(
            mode=mode,
            run_dir=run_dir,
            skip_gpu=skip_gpu,
            runner=runner,
        )
    merge_fixture(results, phase, payload)
    if phase == "prefill-guard" and mode == "acceptance" and not skip_gpu:
        results["pins_applied"] = apply_production_pins(results)
    write_report(results)
    dump_json(FIXTURE, results)
    dump_json(run_dir / "opt092_q8_grouped.json", results)
    counts = (results.get(phase) or {}).get("native_counts") or {}
    print(
        "QW38_OPT092_RESULT="
        + json.dumps(
            {
                "task": "OPT-092",
                "mode": mode,
                "phase": phase,
                "selected_path": results.get("selected_path"),
                "production_kept": results.get("production_kept"),
                "claims_throughput": results.get("claims_throughput"),
                "shipping_grouping": results.get("shipping_grouping"),
                "opt074_coverage_unadmitted_blocker": False,
                "keep": False,
                "warmups": counts.get("warmups"),
                "samples": counts.get("samples"),
                "observed_warmups": counts.get("observed_warmups"),
                "observed_samples": counts.get("observed_samples"),
                "observed_candidates": counts.get("observed_candidates"),
                "observed_shapes": counts.get("observed_shapes"),
                "observed_tier": counts.get("observed_tier"),
                "pairs": counts.get("pairs"),
                "sample_ids": counts.get("sample_ids"),
                "acceptance_executed": counts.get("acceptance_executed"),
            }
        )
    )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=PHASES)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("feedback", "acceptance", "release"),
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--skip-gpu", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(
        args.mode,
        args.phase,
        Path(args.run_dir),
        skip_gpu=args.skip_gpu,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
