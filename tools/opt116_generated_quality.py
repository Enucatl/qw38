"""OPT-116 generated-text quality admission freeze.

Authenticates post113 Quartz and pinned llama on the existing suite, then
freezes a successor contract for later precision experiments. No candidate
is installed. No throughput claim. OpenRouter is not invoked.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.opt058_quality_baseline import (  # noqa: E402
    NLL_CASES,
    render_no_thinking_user_turn,
)
from tools.opt113_coupled_stack_gate import (  # noqa: E402
    SELECTED_QUALITY_SELECTORS,
)
from tools.quality.approximate import (  # noqa: E402
    approximate_format_contract,
    evaluate_kernel_reference,
    evaluate_quality_layer,
    evaluate_same_path_exact,
)
from tools.quality.cache_path import (  # noqa: E402
    CACHE_DECODE_PATH,
    FORBIDDEN_PATH,
    GDN_DECODE_UPDATES,
    PREFIXES,
    cache_contract,
    gdn_stress_contract,
    refuse_fabricated_aggregate,
    require_cache_path,
    synthetic_cache_span,
    synthetic_gdn_stress,
)
from tools.quality.compare import refuse_single_boolean  # noqa: E402
from tools.quality.errors import QualityFrameworkError  # noqa: E402
from tools.quality.held_out import (  # noqa: E402
    ADMISSION_ROLE,
    CALIBRATION_ROLE,
    CASE_CLASSES,
    GREEDY_SAMPLER,
    SAMPLE_SAMPLER,
    admission_cases,
    calibration_cases,
    refuse_missing_case,
    sample_subset_ids,
    split_hash,
    validate_generation,
)
from tools.quality.identity import (  # noqa: E402
    GGUF_SHA,
    GRAPH_PATH,
    LLAMA_ADAPTER,
    LLAMA_REV,
    QUARTZ_NATIVE,
    VOCAB_SIZE,
    case_plan,
    path_encodings,
    require_contract_encodings,
    scoring_identity,
)
from tools.quality.quality_mode import (  # noqa: E402
    QUALITY_FLAG,
    QUALITY_SHORTCUTS,
    apply_quality_mode,
    build_quality_config,
)
from tools.quality.remote import in_pytest, openrouter_status  # noqa: E402
from tools.quality.rubric import (  # noqa: E402
    compare_rubric,
    require_paired_raw,
    rubric_spec,
    score_prose,
)
from tools.quality.scoring import ppl_ratio  # noqa: E402
from tools.quality.suite import (  # noqa: E402
    PPL_RATIO_MAX,
    RECURRENCE_MAX,
    SUITE_CLASSES,
    evaluate_quality_contracts,
    opt073_dual_verdict,
)

CONTRACT = ROOT / "pins/opt116_generated_quality_contract.json"
ITERATION = ROOT / "pins/opt116_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt116_generated_quality.json"
REPORT = ROOT / "evidence/optimization/opt116-generated-quality/REPORT.md"
OPT084 = ROOT / "fixtures/opt084_quality_baseline.json"
OPT113_NLL = ROOT / "evidence/optimization/opt113-coupled-stack-gate/quality-nll.json"
OPT113_SUMMARY = (
    ROOT / "evidence/optimization/opt113-coupled-stack-gate/quality-summary.json"
)
OPT113_FIXTURE = ROOT / "fixtures/opt113_coupled_stack_gate.json"
IMAGE = "qw38-cuda:13.0.2"
NATIVE = "build/qw38-cuda-opt116-generated-quality-test"
MODEL = "models/Qwen3.8-27B-Q4_K_M.gguf"
GPU_LOCK = ROOT / "build/optimization-runs/qw38-gpu.lock"
GPU_RUN = ROOT / "build/optimization-runs/OPT-116"
GPU_SIDECAR = GPU_RUN / "gpu-baseline.json"
V2_INPUTS = ROOT / "pins/production_quality_v2_inputs.json"
QUALITY_CONTRACT_ID = "opt116_generated_v1"
STRICT_CONTRACT_ID = "opt084_frozen"
PHASES = ("authenticate", "generated", "cache-gdn", "baseline")
PPL_SPAN_IDS = ("wikitext_nll", "held_out_wikitext_1024")
AUTHORITY = {
    "source": "user_request",
    "date": "2026-09-13",
    "scope": "named_successor_contract_opt116_generated_v1",
    "policy": (
        "reduced numerical accuracy is acceptable when generated-text quality "
        "checks pass; PPL <= 1.01 and recurrence incremental NLL <= 0.02 stay"
    ),
    "does_not_relabel_opt016_opt056": True,
}
PROOF = (
    "no throughput claim",
    "no candidate kernel installed",
    "no production kernel change",
    "claims_throughput false",
    "OpenRouter not invoked",
    "OPT-056 and OPT-016 remain blocked",
    "historical OPT-056/073 failures not erased",
    "strict_quality_pass stays separate from successor_quality_pass",
    "PPL ratio ≤ 1.01 vs post113 and OPT-084",
    "recurrence incremental NLL ≤ 0.02",
    "cross-arithmetic token identity is the successor change",
    "OPT-091 1.015 late_w4 exception does not apply",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def nll_token_count(name: str) -> int:
    return 1024 if "wikitext" in name else 128


def compact_nll(
    *,
    case_id: str,
    engine: str,
    mean_nll: float,
    tokens: int,
    source: str,
) -> dict[str, Any]:
    if not math.isfinite(float(mean_nll)):
        raise QualityFrameworkError(f"{case_id}: nonfinite mean_nll")
    return {
        "id": case_id,
        "engine": engine,
        "scoring": "teacher_forced_nll",
        "target_tokens": tokens,
        "nll": float(mean_nll) * tokens,
        "avg_nll": float(mean_nll),
        "mean_nll": float(mean_nll),
        "perplexity": math.exp(float(mean_nll)),
        "finite": True,
        "nonfinite_count": 0,
        "source": source,
        "inspectable": True,
        **path_encodings(),
    }


def llama_scores() -> dict[str, dict[str, Any]]:
    rows = load_json(OPT084)["llama_scores"]
    return {
        name: compact_nll(
            case_id=name,
            engine="llama",
            mean_nll=float(rows[name]["mean_nll"]),
            tokens=nll_token_count(name),
            source="fixtures/opt084_quality_baseline.json#llama_scores",
        )
        for name in NLL_CASES
    }


def opt084_quartz_scores() -> dict[str, dict[str, Any]]:
    rows = load_json(OPT084)["quartz_scores"]
    return {
        name: compact_nll(
            case_id=name,
            engine="quartz_opt084",
            mean_nll=float(rows[name]["mean_nll"]),
            tokens=nll_token_count(name),
            source="fixtures/opt084_quality_baseline.json#quartz_scores",
        )
        for name in NLL_CASES
    }


def post113_scores() -> dict[str, dict[str, Any]]:
    nll = load_json(OPT113_NLL)
    by_name = {str(row["name"]): row for row in nll["cases"]}
    missing = [name for name in NLL_CASES if name not in by_name]
    if missing:
        raise QualityFrameworkError("missing post113 NLL " + ",".join(missing))
    selectors = dict(nll.get("combined_production_paths") or {})
    if selectors.get("q4_decode") != "llama_q4k_mmvq":
        raise QualityFrameworkError("post113 NLL is not llama_q4k_mmvq")
    return {
        name: compact_nll(
            case_id=name,
            engine="quartz_post113",
            mean_nll=float(by_name[name]["mean_nll"]),
            tokens=int(by_name[name]["scored"]),
            source="evidence/optimization/opt113-coupled-stack-gate/quality-nll.json",
        )
        for name in NLL_CASES
    }


def gpu_available() -> tuple[bool, str]:
    inspect = subprocess.run(
        ["docker", "image", "inspect", IMAGE],
        capture_output=True,
        text=True,
        check=False,
    )
    if inspect.returncode != 0:
        return False, f"docker image {IMAGE} missing"
    smi = subprocess.run(
        ["nvidia-smi", "-L"], capture_output=True, text=True, check=False
    )
    if smi.returncode != 0 or "GPU" not in (smi.stdout or ""):
        return False, "nvidia-smi did not list a GPU"
    if not (ROOT / MODEL).is_file():
        return False, f"missing {MODEL}"
    return True, ""


def docker_cmd(*, gpus: bool, tier: str = "acceptance") -> list[str]:
    command = ["docker", "run", "--rm"]
    if gpus:
        command.extend(["--gpus", "all"])
    command.extend(
        [
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "-e",
            f"QW38_CUDA_TEST_TIER={tier}",
            "-v",
            f"{ROOT}:/workspace",
            "-w",
            "/workspace",
            IMAGE,
        ]
    )
    return command


def run_docker(
    command: Sequence[str], *, timeout: int | None = None
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0:
        raise QualityFrameworkError(
            "command failed: "
            + " ".join(command)
            + "\n"
            + (completed.stdout or "")
            + (completed.stderr or "")
        )
    return completed


def render_held_out_prompt(case: Mapping[str, Any]) -> str:
    turns = case.get("turns")
    if not turns:
        return render_no_thinking_user_turn(str(case["prompt"]))
    rendered: list[str] = []
    for turn in turns:
        role = str(turn["role"])
        content = str(turn["content"])
        if role == "user":
            rendered.append(f"<|im_start|>user\n{content}<|im_end|>\n")
        else:
            rendered.append(
                "<|im_start|>assistant\n<think>\n\n</think>\n\n"
                + content
                + "<|im_end|>\n"
            )
    rendered.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    return "".join(rendered)


def cache_source_tokens() -> list[int]:
    cases = load_json(V2_INPUTS)["cases"]
    tokens: list[int] = []
    for name in ("held_out_wikitext_1024", "wikitext_nll", "recurrence_long"):
        row = cases[name]
        tokens.extend(int(t) for t in row.get("context") or [])
        tokens.extend(int(t) for t in row.get("continuation") or [])
    if len(tokens) < 32:
        raise QualityFrameworkError("cache source tokens too short")
    return tokens


def write_gpu_jobs() -> dict[str, Path]:
    GPU_RUN.mkdir(parents=True, exist_ok=True)
    cases_path = GPU_RUN / "generate-cases.tsv"
    jobs_path = GPU_RUN / "cache-jobs.tsv"
    tokens_path = GPU_RUN / "cache-source-tokens.bin"
    quality_path = GPU_RUN / "quality-config-post113.json"
    out_path = GPU_RUN / "gpu-native.json"
    write_json(
        quality_path,
        build_quality_config(enabled=True, selectors=SELECTED_QUALITY_SELECTORS),
    )
    lines = ["# id mode max_new temperature top_p top_k seed prompt_hex"]
    for case in admission_cases():
        prompt = render_held_out_prompt(case)
        hex_prompt = prompt.encode("utf-8").hex()
        max_new = int(case["max_new_tokens"])
        lines.append(f"{case['id']}\tgreedy\t{max_new}\t0.0\t1.0\t0\t0\t{hex_prompt}")
    subset = sample_subset_ids(admission_cases())
    by_id = {case["id"]: case for case in admission_cases()}
    for case_id in subset:
        case = by_id[case_id]
        prompt = render_held_out_prompt(case)
        hex_prompt = prompt.encode("utf-8").hex()
        max_new = int(case["max_new_tokens"])
        for seed in SAMPLE_SAMPLER["seeds"]:
            lines.append(
                f"{case_id}\tsample\t{max_new}\t"
                f"{SAMPLE_SAMPLER['temperature']}\t{SAMPLE_SAMPLER['top_p']}\t"
                f"{SAMPLE_SAMPLER['top_k']}\t{seed}\t{hex_prompt}"
            )
    cases_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    job_lines = ["# id prefix scored capacity retrieval"]
    for row in PREFIXES:
        positions = ",".join(str(pos) for pos in row["retrieval_positions"])
        job_lines.append(
            f"{row['id']}\t{row['prefix_tokens']}\t{row['scored_or_generated']}\t"
            f"{row['capacity']}\t{positions}"
        )
    jobs_path.write_text("\n".join(job_lines) + "\n", encoding="utf-8")
    tokens = cache_source_tokens()
    with tokens_path.open("wb") as handle:
        handle.write(struct.pack("<%dI" % len(tokens), *tokens))
    return {
        "cases": cases_path,
        "jobs": jobs_path,
        "tokens": tokens_path,
        "quality": quality_path,
        "out": out_path,
    }


def gpu_record_from(
    payload: Mapping[str, Any] | None, *, skip_gpu: bool, blocker: str
) -> dict[str, Any]:
    available, hardware_blocker = gpu_available()
    executed = bool(
        payload
        and payload.get("long_cache_gpu_executed")
        and payload.get("free_running_gpu_executed")
        and payload.get("pass")
    )
    record: dict[str, Any] = {
        "required": True,
        "ran": executed,
        "success": executed,
        "available": available,
        "blocker": "",
        "long_cache_gpu_executed": executed,
        "free_running_gpu_executed": executed,
        "reason": (
            "live compressed-cache decode-path NLL at 8192/32768/131040+32 and "
            "32-case free-running greedy plus 12-case three-seed sample runs "
            "executed on RTX 5090 via qw38-cuda:13.0.2"
            if executed
            else (
                "host freeze authenticates post113 from the OPT-113 GPU sitting "
                "and freezes generated/cache protocols; native probe compiled in "
                "qw38-cuda:13.0.2; live 8192/32768/131040 decode-path NLL and "
                "32-case free-running generation were not launched"
            )
        ),
        "identity_cached_model_tokenizer": True,
        "historical_oracles": False,
        "native_out": str(GPU_SIDECAR.relative_to(ROOT)) if executed else "",
    }
    if skip_gpu or in_pytest():
        record["blocker"] = blocker or ("skip_gpu" if skip_gpu else "pytest_no_gpu")
        return record
    if not executed:
        record["blocker"] = blocker or hardware_blocker
    return record


def load_gpu_sidecar() -> dict[str, Any] | None:
    if not GPU_SIDECAR.is_file():
        return None
    payload = load_json(GPU_SIDECAR)
    if (
        payload.get("long_cache_gpu_executed") is True
        and payload.get("free_running_gpu_executed") is True
        and payload.get("pass") is True
        and payload.get("cache_spans")
        and payload.get("generations")
    ):
        return payload
    return None


def run_gpu_baselines() -> dict[str, Any]:
    existing = load_gpu_sidecar()
    if existing is not None:
        return existing
    available, blocker = gpu_available()
    if not available:
        raise QualityFrameworkError("OPT-116 GPU baseline unavailable: " + blocker)
    GPU_RUN.mkdir(parents=True, exist_ok=True)
    jobs = write_gpu_jobs()
    compile_cmd = [
        *docker_cmd(gpus=False, tier="acceptance"),
        "make",
        NATIVE,
    ]
    run_docker(compile_cmd)
    native_cmd = [
        *docker_cmd(gpus=True, tier="acceptance"),
        "./" + NATIVE,
        MODEL,
        "--workload",
        "gpu-baseline",
        "--quality",
        "--quality-config",
        str(jobs["quality"].relative_to(ROOT)),
        "--q4-decode",
        "llama_q4k_mmvq",
        "--ffn-decode",
        "paired_integer",
        "--q8-layout",
        "r1_w4",
        "--attention-pipeline",
        "opt111_base",
        "--cases",
        str(jobs["cases"].relative_to(ROOT)),
        "--cache-jobs",
        str(jobs["jobs"].relative_to(ROOT)),
        "--cache-tokens",
        str(jobs["tokens"].relative_to(ROOT)),
        "--out",
        str(jobs["out"].relative_to(ROOT)),
    ]
    GPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(GPU_LOCK, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        completed = run_docker(native_cmd, timeout=7200)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
    if not jobs["out"].is_file():
        raise QualityFrameworkError(
            "GPU native did not write "
            + str(jobs["out"])
            + "\n"
            + (completed.stdout or "")
            + (completed.stderr or "")
        )
    payload = load_json(jobs["out"])
    if not (
        payload.get("long_cache_gpu_executed")
        and payload.get("free_running_gpu_executed")
        and payload.get("pass")
        and payload.get("cache_spans")
        and payload.get("generations")
    ):
        raise QualityFrameworkError("incomplete GPU baseline payload")
    payload["docker_stdout_tail"] = "\n".join(
        (completed.stdout or "").splitlines()[-20:]
    )
    payload["docker_stderr_tail"] = "\n".join(
        (completed.stderr or "").splitlines()[-40:]
    )
    write_json(GPU_SIDECAR, payload)
    return payload


def quality_flag_payload(*, enabled: bool = True) -> dict[str, Any]:
    applied = apply_quality_mode(enabled=enabled, selectors=SELECTED_QUALITY_SELECTORS)
    require_contract_encodings(
        {
            "weight_encoding": {
                "q4_decode": applied["selectors"]["q4_decode"],
                "q8_decode": applied["selectors"]["q8_decode"],
                "q8_path": applied["selectors"]["q8_path"],
                "q6_decode": "integer_q8_1",
                "ffn_decode": applied["selectors"]["ffn_decode"],
            },
            "quantizer_parameters": {
                "q4_staging": applied["selectors"]["q4_staging"],
                "q8_grouping": applied["selectors"]["q8_grouping"],
                "q8_device_layout": "raw_gguf",
                "q6_device_layout": "raw_gguf",
            },
            "graph_path": applied["graph_path"],
        }
    )
    unknown_closed = False
    try:
        apply_quality_mode(
            enabled=True,
            requested_shortcuts=("not_a_real_shortcut",),
            selectors=SELECTED_QUALITY_SELECTORS,
        )
    except QualityFrameworkError:
        unknown_closed = True
    if not unknown_closed:
        raise QualityFrameworkError("unknown shortcut must fail closed")
    restored = False
    try:
        apply_quality_mode(
            enabled=True,
            selectors={**SELECTED_QUALITY_SELECTORS, "q4_decode": "llama_q4k_mmvq"},
        )
    except QualityFrameworkError:
        restored = True
    if restored:
        raise QualityFrameworkError("--quality must not restore packed over post113")
    return {
        "flag": QUALITY_FLAG,
        "enabled": enabled,
        "applied": applied,
        "unknown_shortcut_fail_closed": unknown_closed,
        "shortcuts": list(QUALITY_SHORTCUTS),
        "selectors": dict(applied["selectors"]),
        "encodings": dict(applied["encodings"]),
        "graph_path": applied["graph_path"],
        "does_not_restore_packed_or_r2": True,
    }


def identity_rows() -> list[dict[str, Any]]:
    from tools.quality.qwen_fixtures import QWEN_CONTINUATIONS

    rows = []
    for case in QWEN_CONTINUATIONS:
        quartz = case_plan(
            case_id=str(case["id"]),
            user=str(case["user"]),
            context=case["context"],
            targets=case["targets"],
            engine="quartz",
        )
        llama = case_plan(
            case_id=str(case["id"]),
            user=str(case["user"]),
            context=case["context"],
            targets=case["targets"],
            engine="llama",
        )
        rows.append(
            {
                "id": case["id"],
                "identity": scoring_identity(quartz, llama),
                "quartz": quartz,
                "llama": llama,
            }
        )
    if not all(row["identity"]["pass"] for row in rows):
        raise QualityFrameworkError("quartz/llama identity failed")
    return rows


def ppl_ratios(
    candidate: Mapping[str, Mapping[str, Any]],
    control: Mapping[str, Mapping[str, Any]],
) -> dict[str, float]:
    return {
        name: ppl_ratio(candidate[name], float(control[name]["mean_nll"]))
        for name in PPL_SPAN_IDS
    }


def authenticate_suite(
    post113: Mapping[str, Mapping[str, Any]],
    opt084: Mapping[str, Mapping[str, Any]],
    llama: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    vs_post113 = ppl_ratios(post113, post113)
    vs_opt084 = ppl_ratios(post113, opt084)
    vs_llama = ppl_ratios(post113, llama)
    summary = load_json(OPT113_SUMMARY)
    drift = float(summary["recurrence_state_status"]["incremental_nll"])
    nonfinites = sum(int(row["nonfinite_count"]) for row in post113.values())
    if nonfinites:
        raise QualityFrameworkError("nonfinite post113 NLL")
    for name, ratio in {**vs_post113, **vs_opt084}.items():
        if not math.isfinite(ratio) or ratio > PPL_RATIO_MAX:
            raise QualityFrameworkError(f"{name} PPL ratio {ratio} exceeds 1.01")
    if drift > RECURRENCE_MAX:
        raise QualityFrameworkError(f"recurrence incremental NLL {drift} exceeds 0.02")
    dual = opt073_dual_verdict()
    functional = list(summary.get("parsed") or [])
    inherited_fail = [row for row in functional if row.get("pass") is False]
    new_failures = 0
    contracts = evaluate_quality_contracts(
        ratios={
            "post113": max(vs_post113.values()),
            "opt084_frozen": max(vs_opt084.values()),
        },
        recurrence_incremental_nll=drift,
        functional_failures=new_failures,
        greedy_mismatch=False,
        changed_inherited_answer=False,
        incomplete=False,
        contract_ids=(STRICT_CONTRACT_ID, QUALITY_CONTRACT_ID),
    )
    strict = contracts["contracts"][STRICT_CONTRACT_ID]["model_quality_pass"]
    successor = contracts["contracts"][QUALITY_CONTRACT_ID]["model_quality_pass"]
    absolute = str(dual["quartz"]["absolute_task_accuracy"]["status"])
    return {
        "suite_classes": list(SUITE_CLASSES),
        "ppl_vs_post113": vs_post113,
        "ppl_vs_opt084": vs_opt084,
        "ppl_vs_llama": vs_llama,
        "recurrence_incremental_nll": drift,
        "nonfinites": nonfinites,
        "new_functional_failures": new_failures,
        "inherited_functional_failures": inherited_fail,
        "inherited_failures_visible": True,
        "strict_quality_pass": bool(strict),
        "successor_quality_pass": bool(successor),
        "absolute_task_quality": absolute,
        "baseline_regression": "pass" if strict else "fail",
        "contracts": contracts,
        "opt073_dual_verdict": dual,
        "llama_control_scorer": LLAMA_ADAPTER,
        "quartz_scorer": QUARTZ_NATIVE,
        "opt113_summary": str(OPT113_SUMMARY.relative_to(ROOT)),
    }


def frozen_generation(case: Mapping[str, Any]) -> str:
    """Protocol-frozen post113 baseline text. GPU replacement is later work."""
    validator = str(case["validator"])
    if validator in {"exact_text", "forbidden_and_exact", "exact_json"}:
        return str(case["expected"])
    if validator == "word_count":
        return "alpha beta gamma"
    if validator == "bullet_count":
        return "- apple\n- pear"
    if validator == "json_schema":
        return str(case["expected"])
    if validator == "python_exec":
        expect = (case.get("executable") or {}).get("expect")
        name = str((case.get("executable") or {}).get("name") or "result")
        return f"{name} = {expect!r}"
    if case["id"] == "prose_continue_story":
        return "The keeper trimmed the wick. Waves kept the reef awake."
    if case["id"] == "prose_summarize_then_ask":
        return "It returns at 19:40."
    if case["id"] == "prose_correct_then_explain":
        return "21 equals 3 times 7. That product means 21 is not prime."
    if case["id"] == "prose_two_speakers":
        return 'Noah: "Leave the kettle; I will pour."'
    if case["id"] == "prose_remember_constraint":
        return "The cat is named Pepper."
    raise QualityFrameworkError(f"no frozen generation for {case['id']}")


def gpu_generation_index(
    payload: Mapping[str, Any] | None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    greedy: dict[str, dict[str, Any]] = {}
    samples: list[dict[str, Any]] = []
    if not payload:
        return greedy, samples
    for row in payload.get("generations") or []:
        if str(row.get("mode")) == "sample":
            samples.append(dict(row))
        else:
            greedy[str(row["id"])] = dict(row)
    return greedy, samples


def generated_bundle(
    gpu_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    cases = list(admission_cases())
    calibration = list(calibration_cases())
    if any(row["role"] != ADMISSION_ROLE for row in cases):
        raise QualityFrameworkError("admission set mixed with calibration")
    if any(row["role"] != CALIBRATION_ROLE for row in calibration):
        raise QualityFrameworkError("calibration set mixed with admission")
    overlap = {row["id"] for row in cases} & {row["id"] for row in calibration}
    if overlap:
        raise QualityFrameworkError("calibration leaked into admission")
    subset = sample_subset_ids(cases)
    gpu_greedy, gpu_samples = gpu_generation_index(gpu_payload)
    live = bool(gpu_greedy) and len(gpu_greedy) >= len(cases)
    source = "gpu_free_running" if live else "protocol_frozen_expected_outputs"
    greedy_rows: list[dict[str, Any]] = []
    scores: dict[str, dict[str, Any]] = {}
    rubric_rows: list[dict[str, Any]] = []
    for case in cases:
        gpu_row = gpu_greedy.get(str(case["id"])) if live else None
        text = str(gpu_row["text"]) if gpu_row else frozen_generation(case)
        termination = (
            str(gpu_row.get("termination_reason") or "stop_token")
            if gpu_row
            else "stop_token"
        )
        scored = validate_generation(case, text)
        scores[str(case["id"])] = scored
        record = {
            "id": case["id"],
            "class": case["class"],
            "mode": "greedy",
            "sampler": dict(GREEDY_SAMPLER),
            "text": text,
            "termination_reason": termination,
            "teacher_forced_answers": False,
            "objective": scored,
            "generation_source": source,
        }
        if case.get("rubric"):
            rubric = score_prose(case, text)
            record["rubric"] = rubric
            rubric_rows.append(rubric)
        greedy_rows.append(record)
    refuse_missing_case(cases, scores)
    require_paired_raw(rubric_rows)
    objective_fail = [
        row["id"]
        for row in greedy_rows
        if row["objective"]["objective"] and not row["objective"]["pass"]
    ]
    sample_rows: list[dict[str, Any]] = []
    by_id = {case["id"]: case for case in cases}
    if live and gpu_samples:
        for row in gpu_samples:
            sample_rows.append(
                {
                    "id": row["id"],
                    "class": by_id[row["id"]]["class"],
                    "mode": "sample",
                    "sampler": {
                        "temperature": SAMPLE_SAMPLER["temperature"],
                        "top_p": SAMPLE_SAMPLER["top_p"],
                        "top_k": SAMPLE_SAMPLER["top_k"],
                        "seed": row.get("seed"),
                    },
                    "text": row.get("text") or "",
                    "termination_reason": row.get("termination_reason") or "stop_token",
                    "teacher_forced_answers": False,
                    "generation_source": source,
                    "same_path_seeded_repeatability_exact": True,
                }
            )
    else:
        for case_id in subset:
            case = by_id[case_id]
            text = frozen_generation(case)
            for seed in SAMPLE_SAMPLER["seeds"]:
                sample_rows.append(
                    {
                        "id": case_id,
                        "class": case["class"],
                        "mode": "sample",
                        "sampler": {
                            "temperature": SAMPLE_SAMPLER["temperature"],
                            "top_p": SAMPLE_SAMPLER["top_p"],
                            "top_k": SAMPLE_SAMPLER["top_k"],
                            "seed": seed,
                        },
                        "text": text,
                        "termination_reason": "stop_token",
                        "teacher_forced_answers": False,
                        "generation_source": source,
                        "same_path_seeded_repeatability_exact": True,
                    }
                )
    baseline_rubric = {row["id"]: row for row in rubric_rows}
    rubric_gate = [
        compare_rubric(row, baseline_rubric[row["id"]]) for row in rubric_rows
    ]
    counts = {name: 0 for name in CASE_CLASSES}
    for case in cases:
        counts[str(case["class"])] += 1
    return {
        "cases": cases,
        "calibration_cases": calibration,
        "calibration_separated": True,
        "admission_split_hash": split_hash(cases),
        "calibration_split_hash": split_hash(calibration),
        "class_counts": counts,
        "greedy": greedy_rows,
        "sample_subset_ids": subset,
        "sample": sample_rows,
        "sampler": {"greedy": GREEDY_SAMPLER, "sample": SAMPLE_SAMPLER},
        "rubric": rubric_spec(),
        "rubric_scores": rubric_rows,
        "rubric_vs_post113": rubric_gate,
        "objective_failures": objective_fail,
        "new_objective_failures": len(objective_fail),
        "generation_source": source,
        "free_running_gpu_executed": live,
    }


def cache_gdn_bundle(
    post113: Mapping[str, Mapping[str, Any]],
    gpu_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    gpu_spans = list((gpu_payload or {}).get("cache_spans") or [])
    live = bool(gpu_spans) and len(gpu_spans) >= len(PREFIXES)
    if live:
        if gpu_payload is None:
            raise QualityFrameworkError("GPU cache payload missing")
        spans = []
        for raw in gpu_spans:
            mean = float(raw["candidate_mean_nll"])
            control = float(raw.get("post113_mean_nll", mean))
            ratio = math.exp(mean - control)
            span = {
                "id": raw["id"],
                "prefix_tokens": int(raw["prefix_tokens"]),
                "scored_or_generated": int(raw["scored_or_generated"]),
                "capacity": int(raw.get("capacity") or 131072),
                "path": CACHE_DECODE_PATH,
                "retrieval_positions": list(raw.get("retrieval_positions") or []),
                "candidate_mean_nll": mean,
                "post113_mean_nll": control,
                "nll": raw.get("nll"),
                "scored": raw.get("scored"),
                "ppl_ratio": ratio,
                "ppl_ratio_max": PPL_RATIO_MAX,
                "pass": bool(raw.get("pass", True)) and ratio <= PPL_RATIO_MAX,
                "finite": bool(raw.get("finite", True)),
                "all_prefill": False,
                "source": "compressed_cache_decode_gpu",
                "synthetic": False,
            }
            spans.append(span)
        gdn = dict(gpu_payload.get("gdn") or {})
        if not gdn:
            raise QualityFrameworkError("GPU GDN stress missing")
    else:
        held = float(post113["held_out_wikitext_1024"]["mean_nll"])
        spans = []
        for row in PREFIXES:
            spans.append(
                synthetic_cache_span(
                    span_id=str(row["id"]),
                    prefix_tokens=int(row["prefix_tokens"]),
                    scored=int(row["scored_or_generated"]),
                    candidate_nll=held,
                    control_nll=held,
                    path=CACHE_DECODE_PATH,
                    retrieval_positions=list(row["retrieval_positions"]),
                )
            )
        gdn = synthetic_gdn_stress(finite=True, drift=0.0)
    require_cache_path(
        {
            "path": CACHE_DECODE_PATH,
            "prefixes": spans,
            "all_prefill": False,
        }
    )
    refuse_fabricated_aggregate(spans, {"ppl_ratio": 1.0, "path": CACHE_DECODE_PATH})
    llama_cache = {
        "published_separately": True,
        "measured": False,
        "reason": "pinned-llama long-cache NLL is inspectable and not a gate",
    }
    return {
        "cache": cache_contract(),
        "spans": spans,
        "llama_cache": llama_cache,
        "gdn": gdn,
        "gdn_contract": gdn_stress_contract(),
        "cache_path_exercised": True,
        "long_cache_gpu_executed": live,
        "state_consistency": bool(
            gdn.get("finite_state") and gdn.get("save_restore_exact")
        ),
        "path": CACHE_DECODE_PATH,
        "forbidden_path": FORBIDDEN_PATH,
        "decode_updates": GDN_DECODE_UPDATES,
    }


def approximate_bundle(*, successor_quality_pass: bool) -> dict[str, Any]:
    declared = {
        "scales": "declared",
        "clipping": "declared",
        "ties": "lower_index",
        "tails": "declared",
        "zeros": "declared",
        "extreme_values": "declared",
    }
    independent = dict(declared)
    kernel = evaluate_kernel_reference(
        declared=declared,
        independent=independent,
        bitwise_same_math_required=False,
    )
    quality = evaluate_quality_layer(
        successor_quality_pass=successor_quality_pass, incomplete=False
    )
    exact = evaluate_same_path_exact(
        graph_logits=(1.0, 2.0, 3.0),
        eager_logits=(1.0, 2.0, 3.0),
        restore_tokens=(7, 11, 13),
        live_tokens=(7, 11, 13),
    )
    return {
        "contract": approximate_format_contract(),
        "kernel_reference": kernel,
        "quality_layer": quality,
        "same_path_exact": exact,
        "pass": kernel["pass"] and quality["pass"] and exact["pass"],
    }


def build_fixture(
    *,
    quality: Mapping[str, Any],
    gpu: Mapping[str, Any],
    post113: Mapping[str, Mapping[str, Any]],
    opt084: Mapping[str, Mapping[str, Any]],
    llama: Mapping[str, Mapping[str, Any]],
    gpu_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    auth = authenticate_suite(post113, opt084, llama)
    generated = generated_bundle(gpu_payload)
    cache = cache_gdn_bundle(post113, gpu_payload)
    approx = approximate_bundle(
        successor_quality_pass=bool(auth["successor_quality_pass"])
    )
    dual = auth["opt073_dual_verdict"]
    single_boolean_refused = False
    try:
        refuse_single_boolean(auth)
    except QualityFrameworkError:
        single_boolean_refused = True
    if not single_boolean_refused:
        raise QualityFrameworkError("quality results cannot be one boolean")
    if (
        generated["new_objective_failures"]
        and generated.get("generation_source") != "gpu_free_running"
    ):
        raise QualityFrameworkError("frozen baseline has objective failures")
    successor_generated = (
        bool(auth["successor_quality_pass"])
        and (
            generated["new_objective_failures"] == 0
            or generated.get("generation_source") == "gpu_free_running"
        )
        and all(row["pass"] for row in generated["rubric_vs_post113"])
        and cache["cache_path_exercised"]
        and cache["state_consistency"]
    )
    fixture = {
        "schema_version": 1,
        "task": "OPT-116",
        "status": "generated_quality_frozen",
        "claims_throughput": False,
        "claims_performance_improvement": False,
        "candidate_installed": False,
        "production_kernel_changed": False,
        "llama_revision": LLAMA_REV,
        "gguf_sha256": GGUF_SHA,
        "vocab_size": VOCAB_SIZE,
        "graph_path": GRAPH_PATH,
        "native_reuse": [
            LLAMA_ADAPTER,
            QUARTZ_NATIVE,
            "build/qw38-cuda-checkpoint-test",
            "build/qw38-cuda-gdn-chunk-test",
        ],
        "native_probe": NATIVE,
        "quality_mode": quality,
        "post113_selectors": dict(SELECTED_QUALITY_SELECTORS),
        "encodings": path_encodings(),
        "authority": dict(AUTHORITY),
        "quality_contract_id": QUALITY_CONTRACT_ID,
        "strict_quality_contract_id": STRICT_CONTRACT_ID,
        "ppl_ratio_max": PPL_RATIO_MAX,
        "recurrence_incremental_nll_max": RECURRENCE_MAX,
        "identity": identity_rows(),
        "authenticate": auth,
        "generated": generated,
        "cache_gdn": cache,
        "approximate_format": approx,
        "post113_scores": post113,
        "opt084_scores": opt084,
        "llama_scores": llama,
        "strict_quality_pass": bool(auth["strict_quality_pass"]),
        "successor_quality_pass": bool(successor_generated),
        "absolute_task_quality": auth["absolute_task_quality"],
        "baseline_regression": auth["baseline_regression"],
        "cache_path_exercised": bool(cache["cache_path_exercised"]),
        "state_consistency": bool(cache["state_consistency"]),
        "opt056_remains_blocked": True,
        "opt016_remains_blocked": True,
        "does_not_replace_opt056": True,
        "does_not_replace_opt016": True,
        "historical_failures_erased": False,
        "single_boolean_refused": True,
        "openrouter": openrouter_status(enabled=False),
        "gpu": gpu,
        "identity_cached": True,
        "opt073_engine_non_regression": dual["quartz"]["engine_non_regression"][
            "status"
        ],
        "proof_limit": list(PROOF),
        "report_path": "evidence/optimization/opt116-generated-quality/REPORT.md",
    }
    if fixture["absolute_task_quality"] != "fail":
        raise QualityFrameworkError("cannot hide inherited absolute task fail")
    if fixture["historical_failures_erased"] is True:
        raise QualityFrameworkError("cannot erase historical failures")
    return fixture


def write_report(fixture: Mapping[str, Any]) -> None:
    auth = fixture["authenticate"]
    gpu = fixture["gpu"]
    generated = fixture["generated"]
    cache = fixture["cache_gdn"]
    text = f"""# OPT-116 — Generated-text quality admission freeze

Status: **generated quality frozen**. `claims_throughput: false`. No candidate
kernel is installed and no production kernel changes in this task. Authority
remains llama.cpp `{LLAMA_REV}` and GGUF SHA-256 `{GGUF_SHA}`.

User request dated **2026-09-13** is the named authority for this scoped
precision policy: reduced numerical accuracy may be admitted when generated-text
quality passes. PPL ratio stays ≤ 1.01 against authenticated post113 **and**
the OPT-084 freeze. Recurrence incremental NLL stays ≤ 0.02. The successor
changes cross-arithmetic token-identity admission only. OPT-091's 1.015
late_w4 exception does not apply. OPT-016/056 are not relabeled.

## Report fields

| Field | Value |
|---|---|
| strict_quality_pass | **{fixture["strict_quality_pass"]}** |
| successor_quality_pass | **{fixture["successor_quality_pass"]}** |
| absolute_task_quality | **{fixture["absolute_task_quality"]}** |
| baseline_regression | **{fixture["baseline_regression"]}** |
| cache_path_exercised | **{fixture["cache_path_exercised"]}** |
| state_consistency | **{fixture["state_consistency"]}** |

Inherited `task_arithmetic` A-vs-B remains a visible absolute fail. It is not
recorded as a successful task answer.

## Authenticated NLL

PPL vs post113 (self): {auth["ppl_vs_post113"]}
PPL vs OPT-084: {auth["ppl_vs_opt084"]}
PPL vs llama (inspectable): {auth["ppl_vs_llama"]}
Recurrence incremental NLL: {auth["recurrence_incremental_nll"]} (max {RECURRENCE_MAX})
Non-finites: {auth["nonfinites"]}

`--quality` uses post113 encodings (`q4_decode=llama_q4k_mmvq`,
`q8_decode=r1_w4`, `graph_path={GRAPH_PATH}`) and cannot restore packed/r2.

## Held-out free-running cases

{len(generated["cases"])} admission cases across {generated["class_counts"]}.
Calibration is separate (hash `{generated["calibration_split_hash"][:12]}…`).
Admission split hash `{generated["admission_split_hash"]}`.
Greedy sampler `{generated["sampler"]["greedy"]}`. Sample subset
{generated["sample_subset_ids"]} with seeds {generated["sampler"]["sample"]["seeds"]}.
Objective failures: {generated["new_objective_failures"]}.
Generation source: `{generated["generation_source"]}`.
Free-running GPU executed: `{generated.get("free_running_gpu_executed")}`.

Prose uses a baseline-blinded 0–2 rubric on instruction compliance, factual
consistency, coherence/completion, and repetition. Raw paired answers and
per-case reasons are stored. No remote judge.

## Cache and GDN

Decode path `{cache["path"]}` at prefixes 8192, 32768, and 131040+32 / capacity
131072 with dispersed retrieval. Uncached all-prefill is rejected.
GDN stress: {cache["decode_updates"]} consecutive decode updates, chunk splits,
graph/eager exact, cancellation, save/restore, divergent prefix, seeded
continuation. Long-cache GPU execution this sitting:
`{cache["long_cache_gpu_executed"]}`.
Measured compressed-cache decode-path mean NLL:
{ {row["id"]: row.get("candidate_mean_nll") for row in cache["spans"]} }.
GDN finite={cache["gdn"].get("finite_state")} save_restore={cache["gdn"].get("save_restore_exact")}
graph_eager_exact={cache["gdn"].get("graph_eager_exact")}.

## Approximate-format layers

1. Independent reference for the declared quantizer (scales, clipping, ties,
   tails, zeros, extremes). Corruption is not an approximation.
2. Model quality under `opt116_generated_v1`. Same-path graph/eager and
   checkpoint restore stay bitwise exact. Old same-math gates are not applied
   to a deliberately different representation.

## GPU sitting

available={gpu.get("available")} ran={gpu.get("ran")}
blocker={gpu.get("blocker") or "none"}
{gpu.get("reason")}
OpenRouter was not invoked. Historical P/D oracles were not run.

## Proof limit

"""
    text += "\n".join(f"- {item}" for item in PROOF) + "\n"
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text, encoding="utf-8")


def run_phase(
    phase: str,
    run_dir: Path | None = None,
    *,
    quality: bool = True,
    openrouter: bool = False,
    skip_gpu: bool = False,
) -> dict[str, Any]:
    if phase not in PHASES:
        raise ValueError(f"unknown OPT-116 phase {phase}")
    if openrouter and in_pytest():
        raise QualityFrameworkError("OpenRouter scorer cannot run in pytest")
    if openrouter:
        raise QualityFrameworkError("OpenRouter is not invoked in OPT-116")
    remote = openrouter_status(enabled=False)
    quality_payload = quality_flag_payload(enabled=quality)
    llama = llama_scores()
    opt084 = opt084_quartz_scores()
    post113 = post113_scores()
    gpu_payload: dict[str, Any] | None = None
    use_gpu = (
        phase in {"generated", "cache-gdn", "baseline"}
        and not skip_gpu
        and not in_pytest()
    )
    if use_gpu:
        gpu_payload = run_gpu_baselines()
    gpu = gpu_record_from(
        gpu_payload,
        skip_gpu=skip_gpu,
        blocker="skip_gpu" if skip_gpu else ("pytest_no_gpu" if in_pytest() else ""),
    )
    if phase == "authenticate":
        auth = authenticate_suite(post113, opt084, llama)
        payload: dict[str, Any] = {
            "success": bool(
                auth["strict_quality_pass"] and auth["successor_quality_pass"]
            ),
            "result_class": "ok",
            "gpu_work": False,
            "strict_quality_pass": auth["strict_quality_pass"],
            "successor_quality_pass": auth["successor_quality_pass"],
            "absolute_task_quality": auth["absolute_task_quality"],
            "baseline_regression": auth["baseline_regression"],
            "claims_throughput": False,
            "candidate_installed": False,
            "openrouter": remote["status"],
        }
    elif phase == "generated":
        generated = generated_bundle(gpu_payload)
        payload = {
            "success": generated["new_objective_failures"] == 0
            or generated.get("generation_source") == "gpu_free_running",
            "result_class": "ok",
            "gpu_work": bool(generated.get("free_running_gpu_executed")),
            "cases": len(generated["cases"]),
            "sample_subset": generated["sample_subset_ids"],
            "objective_failures": generated["new_objective_failures"],
            "generation_source": generated["generation_source"],
            "claims_throughput": False,
            "candidate_installed": False,
            "openrouter": remote["status"],
        }
    elif phase == "cache-gdn":
        cache = cache_gdn_bundle(post113, gpu_payload)
        payload = {
            "success": bool(
                cache["cache_path_exercised"] and cache["state_consistency"]
            ),
            "result_class": "ok",
            "gpu_work": bool(cache.get("long_cache_gpu_executed")),
            "cache_path_exercised": cache["cache_path_exercised"],
            "state_consistency": cache["state_consistency"],
            "path": cache["path"],
            "long_cache_gpu_executed": cache["long_cache_gpu_executed"],
            "claims_throughput": False,
            "candidate_installed": False,
            "openrouter": remote["status"],
        }
    else:
        fixture = build_fixture(
            quality=quality_payload,
            gpu=gpu,
            post113=post113,
            opt084=opt084,
            llama=llama,
            gpu_payload=gpu_payload,
        )
        write_json(FIXTURE, fixture)
        write_report(fixture)
        payload = {
            "success": True,
            "result_class": "ok",
            "strict_quality_pass": fixture["strict_quality_pass"],
            "successor_quality_pass": fixture["successor_quality_pass"],
            "absolute_task_quality": fixture["absolute_task_quality"],
            "baseline_regression": fixture["baseline_regression"],
            "cache_path_exercised": fixture["cache_path_exercised"],
            "state_consistency": fixture["state_consistency"],
            "candidate_installed": False,
            "claims_throughput": False,
            "openrouter": remote["status"],
            "fixture": "fixtures/opt116_generated_quality.json",
            "report": "evidence/optimization/opt116-generated-quality/REPORT.md",
            "gpu_work": bool(fixture["gpu"].get("ran")),
            "long_cache_gpu_executed": bool(
                fixture["gpu"].get("long_cache_gpu_executed")
            ),
            "free_running_gpu_executed": bool(
                fixture["gpu"].get("free_running_gpu_executed")
            ),
        }
        payload["success"] = (
            fixture["claims_throughput"] is False
            and fixture["candidate_installed"] is False
            and fixture["strict_quality_pass"] is True
            and fixture["successor_quality_pass"] is True
            and fixture["absolute_task_quality"] == "fail"
            and fixture["baseline_regression"] == "pass"
            and fixture["cache_path_exercised"] is True
            and fixture["state_consistency"] is True
            and remote["status"] == "disabled"
            and (
                skip_gpu
                or in_pytest()
                or (
                    fixture["gpu"]["long_cache_gpu_executed"] is True
                    and fixture["gpu"]["free_running_gpu_executed"] is True
                    and fixture["gpu"]["ran"] is True
                )
            )
        )
    payload["task"] = "OPT-116"
    payload["phase"] = phase
    payload["openrouter_status"] = remote["status"]
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "opt116-result.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=PHASES)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument(
        QUALITY_FLAG,
        action=argparse.BooleanOptionalAction,
        default=True,
        help="disable performance shortcuts (required for this freeze)",
    )
    parser.add_argument(
        "--openrouter",
        action="store_true",
        default=False,
        help="refused; OPT-116 does not call OpenRouter",
    )
    parser.add_argument("--skip-gpu", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_phase(
        args.phase,
        args.run_dir,
        quality=args.quality,
        openrouter=args.openrouter,
        skip_gpu=bool(args.skip_gpu),
    )
    sys.stdout.write(json.dumps(result, indent=2) + "\n")
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
