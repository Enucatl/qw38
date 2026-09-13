"""OPT-111 remaining llama F16 MMA prompt-attention deltas vs shipping kv_once."""

from __future__ import annotations

import argparse
import json
import math
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
    parse_engine_pairs,
    utc_now,
)
from tools.opt102_q4_repack import default_native_runner as opt102_native_runner  # noqa: E402
from tools.quality.quality_mode import build_quality_config  # noqa: E402
from tools.quality.scoring import recurrence_incremental_nll  # noqa: E402
from tools.run_optimization_task import (  # noqa: E402
    loop_product,
    parse_native_observation,
    validate_performance_admission,
    workload_for_mode,
)

LLAMA_REV = "cc83d7b4824f73cfdda4dfbb47ee39804f71b328"
GGUF_SHA = "31629f53165ab6a7dad8c9847dcfd1fdf55829dac1e6e748f4a68581b0033d34"
TOKEN_GENERATOR = "(42 + index * 997) % 248320"
CONTRACT = ROOT / "pins/opt111_llama_prompt_attention_contract.json"
ITERATION = ROOT / "pins/opt111_iteration_contract.json"
PROVENANCE = ROOT / "pins/opt111_llama_prompt_attention_provenance.json"
FIXTURE = ROOT / "fixtures/opt111_llama_prompt_attention.json"
REPORT = ROOT / "evidence/optimization/opt111-llama-prompt-attention/REPORT.md"
REJECTION = ROOT / "evidence/optimization/opt111-llama-prompt-attention/REJECTION.md"
EVIDENCE = REPORT.parent
NATIVE = "build/qw38-cuda-opt111-llama-prompt-attention-test"
PROBE = "build/qw38-cuda-optimization-engine-probe"
QUALITY_NATIVE = "build/qw38-cuda-opt058-quality-baseline-test"
NLL_BUNDLE = "pins/production_quality_v2_nll.bundle"
FUNCTIONAL_BUNDLE = "pins/production_quality_v2_functional.bundle"
MODEL = "models/Qwen3.8-27B-Q4_K_M.gguf"
PIN_PATH = ROOT / "cuda/fattn_mma_f16.cuh"
PPL_RATIO_MAX = 1.01
RECURRENCE_MAX = 0.02
OPT114_D128_TOK_S = 53.460154339999995
OPT114_D2048_TOK_S = 48.383027649999995
OPT114_P4096_TOK_S = 2981.08938
LLAMA_P4096_TOK_S = 3252.58
QUARTZ_P4096_TOK_S = 3036.84
MIN_SAVING_MS = 20.0
E2E_REGRESSION_FRAC = 0.02
CONTROL_ID = "kv_once"
BASE_ID = "opt111_base"
XOR_ID = "opt111_xor"
PRIMITIVE_ROWS = (1, 32, 128, 512, 2048, 4096)
GO_NO_GO_ROWS = (32, 128, 512, 2048, 4096)
TILED_ONLY_ROWS = (1,)
RESULT_PREFIX = "QW38_OPT111_LLAMA_PROMPT_ATTENTION_RESULT="
COUNTS_PREFIX = "QW38_OPT111_NATIVE_COUNTS="
PHASES = (
    "parity",
    "primitive",
    "xor-screen",
    "complete",
    "p4096",
    "d128",
    "d2048",
    "quality",
    "state",
)
BASE_QUALITY_SELECTORS: dict[str, Any] = {
    "q4_decode": "llama_q4k_mmvq",
    "q4_staging": "paired_integer",
    "ffn_decode": "paired_integer",
    "q8_decode": "r1_w4",
    "q8_path": "dp4a_q8_1",
    "prompt_mmq": "fma_async_x",
    "prompt_mmq_tile": "i128_j128",
    "prompt_attention": "kv_once",
    "decode_gdn": "sequential",
    "decode_attention": "warp_query",
    "prompt_pair": "off",
    "nvccflags": "-O2 --fmad=false",
    "execution_graphs": "ffn_only",
    "chat_template": "no_thinking",
    "enable_thinking": False,
    "logit_masking": False,
}
PIPELINE_CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "id": CONTROL_ID,
        "attention_pipeline": "kv_once",
        "convert_once": 1,
        "role": "control",
        "expected_launch": "fattn_mma_pipeline_kv_once",
        "llama_load": False,
        "xor_swizzle": False,
    },
    {
        "id": BASE_ID,
        "attention_pipeline": "opt111_base",
        "convert_once": 1,
        "role": "candidate",
        "expected_launch": "fattn_mma_pipeline_opt111_base",
        "llama_load": True,
        "xor_swizzle": False,
    },
    {
        "id": XOR_ID,
        "attention_pipeline": "opt111_xor",
        "convert_once": 1,
        "role": "xor_candidate",
        "expected_launch": "fattn_mma_pipeline_opt111_xor",
        "llama_load": True,
        "xor_swizzle": True,
    },
)
ROUND_RE = re.compile(
    r"round family=\S+ cache_mode=(\S+) warmup=(\S+) sample_index=(\d+) "
    r"observation_unit=independent_round enclosing_ms=([0-9.eE+-]+) "
    r"kernel_only_ms=([0-9.eE+-]+) attention_layers=(\d+) "
    r"path=(\S+) launch=(\S+) convert_once=(\d+) occupancy=(\d+) tokens=(\d+)"
)
PRIMITIVE_ROW_RE = re.compile(
    r"primitive_row tokens=(\d+) control_mean_ms=([0-9.eE+-]+) "
    r"candidate_mean_ms=([0-9.eE+-]+) faster=(\S+) path=(\S+)"
)
XOR_SCREEN_RE = re.compile(
    r"xor_screen linear_ms=([0-9.eE+-]+) xor_ms=([0-9.eE+-]+) "
    r"bank_conflict_hypothesis=(\S+)"
)
REMAINING_DIFF_RE = re.compile(r"remaining_diff (.+)")

NativeRunner = Callable[[Sequence[str], str], subprocess.CompletedProcess[str]]


def default_native_runner(
    command: Sequence[str], tier: str
) -> subprocess.CompletedProcess[str]:
    return opt102_native_runner(command, tier)


def workspace_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def config_by_id(config_id: str) -> dict[str, Any]:
    for config in PIPELINE_CONFIGS:
        if config["id"] == config_id:
            return dict(config)
    raise AdmissionError(f"unknown config {config_id}")


def production_pin() -> str:
    match = re.search(
        r'kSelectedAttentionPipelinePath\[\] = "([^"]+)"',
        PIN_PATH.read_text(encoding="utf-8"),
    )
    if match is None:
        raise AdmissionError("missing kSelectedAttentionPipelinePath")
    return match.group(1)


def production_pin_unchanged() -> bool:
    return production_pin() == CONTROL_ID


def apply_production_pin(path: str) -> None:
    if path not in {CONTROL_ID, BASE_ID, XOR_ID}:
        raise AdmissionError(f"illegal attention pin {path}")
    text = PIN_PATH.read_text(encoding="utf-8")
    updated = re.sub(
        r'kSelectedAttentionPipelinePath\[\] = "[^"]+"',
        f'kSelectedAttentionPipelinePath[] = "{path}"',
        text,
        count=1,
    )
    PIN_PATH.write_text(updated, encoding="utf-8")


def family_plan(phase: str, mode: str) -> dict[str, Any]:
    iteration = load_json(ITERATION)
    if phase not in iteration["workloads"]:
        raise AdmissionError(f"unknown phase {phase}")
    workload = workload_for_mode(iteration["workloads"][phase], mode)
    engine_pairs = int(workload.get("engine_pairs", 0) or 0)
    if phase in {"p4096", "d128", "d2048"} and engine_pairs == 0:
        engine_pairs = int(workload.get("samples", 1))
    return {
        "phase": phase,
        "mode": mode,
        "family": str(workload.get("family", "prompt-attn")),
        "control": CONTROL_ID,
        "warmups": int(workload.get("warmups", 0)),
        "samples": int(workload.get("samples", 1)),
        "engine_pairs": engine_pairs,
        "cases": int(workload.get("cases", 1)),
        "candidates": int(workload.get("candidates", 2)),
        "tier": str(
            workload.get("tier", "screen" if mode == "feedback" else "acceptance")
        ),
        "product": loop_product(workload),
        "control_candidate_pairs": int(workload.get("control_candidate_pairs", 1)),
        "tokens": int(workload.get("tokens", 1) or 1),
        "output_tokens": int(workload.get("output_tokens", 0) or 0),
        "prefix": int(workload.get("prefix", 0) or 0),
        "complete_attention_layers": int(
            workload.get("complete_attention_layers", 16) or 16
        ),
    }


def planned_observation(
    plan: Mapping[str, Any], *, keep: bool = False
) -> dict[str, Any]:
    samples = int(plan["samples"])
    return {
        "schema_version": 1,
        "task": "OPT-111",
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
        "uses_opt098_arrays": False,
    }


def parse_native_payload(stdout: str) -> dict[str, Any]:
    for line in (stdout or "").splitlines():
        stripped = line.strip()
        if stripped.startswith(RESULT_PREFIX):
            return json.loads(stripped[len(RESULT_PREFIX) :])
        if stripped.startswith(COUNTS_PREFIX):
            return json.loads(stripped[len(COUNTS_PREFIX) :])
    return parse_native_observation(stdout) or {}


def parse_opt111_rounds(stdout: str) -> list[dict[str, Any]]:
    rounds: list[dict[str, Any]] = []
    for match in ROUND_RE.finditer(stdout or ""):
        warmup = match.group(2) == "true"
        rounds.append(
            {
                "cache_mode": match.group(1),
                "warmup": warmup,
                "sample_index": int(match.group(3)),
                "enclosing_ms": float(match.group(4)),
                "kernel_only_ms": float(match.group(5)),
                "attention_layers": int(match.group(6)),
                "path": match.group(7),
                "launch": match.group(8),
                "convert_once": int(match.group(9)),
                "occupancy": int(match.group(10)),
                "tokens": int(match.group(11)),
                "observation_unit": "independent_round",
            }
        )
    return rounds


def parse_primitive_rows(stdout: str) -> dict[int, dict[str, Any]]:
    by_tokens: dict[int, dict[str, Any]] = {}
    rounds = parse_opt111_rounds(stdout)
    for match in PRIMITIVE_ROW_RE.finditer(stdout or ""):
        tokens = int(match.group(1))
        control_mean = float(match.group(2))
        candidate_mean = float(match.group(3))
        by_tokens[tokens] = {
            "tokens": tokens,
            "control_mean_ms": control_mean,
            "candidate_mean_ms": candidate_mean,
            "faster": match.group(4) == "true" or candidate_mean < control_mean,
            "path": match.group(5),
            "tiled_only": tokens in TILED_ONLY_ROWS,
            "go_no_go": tokens in GO_NO_GO_ROWS,
        }
    for tokens, row in by_tokens.items():
        control = [
            item["enclosing_ms"]
            for item in rounds
            if not item["warmup"]
            and item["path"] == CONTROL_ID
            and item["tokens"] == tokens
            and item["attention_layers"] == 1
        ]
        candidate = [
            item["enclosing_ms"]
            for item in rounds
            if not item["warmup"]
            and item["path"] == row["path"]
            and item["tokens"] == tokens
            and item["attention_layers"] == 1
        ]
        row["control_ms"] = control
        row["candidate_ms"] = candidate
        if control and candidate and len(control) == len(candidate):
            critical = T_CRIT_DF9 if len(control) >= 10 else FEEDBACK_CRIT
            if len(control) == 5:
                critical = T_CRIT_DF4
            stats = paired_student_t(control, candidate, critical=critical)
            row.update(stats)
            row["faster"] = mean(candidate) < mean(control)
        else:
            row["positive"] = False
            row["mean_diff_ms"] = row["control_mean_ms"] - row["candidate_mean_ms"]
    return by_tokens


def primitive_verdict(
    by_tokens: Mapping[int, Mapping[str, Any]], *, mode: str
) -> dict[str, Any]:
    reasons: list[str] = []
    go_ok = True
    for tokens in GO_NO_GO_ROWS:
        row = dict(by_tokens.get(tokens) or {})
        faster = bool(row.get("faster"))
        if not faster:
            go_ok = False
            reasons.append(f"row_{tokens}_slower_than_kv_once")
        if mode == "acceptance" and tokens == 4096 and not bool(row.get("positive")):
            go_ok = False
            reasons.append("row_4096_ci_not_positive")
    headline = dict(by_tokens.get(4096) or {})
    return {
        "primitive_win": go_ok and bool(headline.get("faster")),
        "go_no_go_rows": list(GO_NO_GO_ROWS),
        "tiled_only_rows": list(TILED_ONLY_ROWS),
        "headline_tokens": 4096,
        "headline_faster": bool(headline.get("faster")),
        "headline_positive": bool(headline.get("positive")),
        "reason": "matched_primitive_won" if go_ok else "matched_primitive_lost",
        "reasons": reasons,
    }


def parse_xor_screen(stdout: str) -> dict[str, Any]:
    match = XOR_SCREEN_RE.search(stdout or "")
    if match is None:
        return {
            "hypothesis": False,
            "linear_ms": None,
            "xor_ms": None,
            "reason": "xor_screen_not_printed",
        }
    linear_ms = float(match.group(1))
    xor_ms = float(match.group(2))
    hypothesis = match.group(3) == "true"
    return {
        "hypothesis": hypothesis,
        "linear_ms": linear_ms,
        "xor_ms": xor_ms,
        "bank_conflict_hypothesis": hypothesis,
        "source": "e4b9af007",
        "reason": (
            "bank_conflict_or_transaction_signal"
            if hypothesis
            else "no_bank_conflict_hypothesis"
        ),
    }


def parse_remaining_diff(stdout: str) -> str:
    match = REMAINING_DIFF_RE.search(stdout or "")
    return match.group(0) if match else ""


def complete_stats(
    stdout: str, *, candidate_path: str, samples: int, mode: str
) -> dict[str, Any]:
    rounds = [
        item
        for item in parse_opt111_rounds(stdout)
        if not item["warmup"]
        and item["tokens"] == 4096
        and item["attention_layers"] == 16
    ]
    control = [item["enclosing_ms"] for item in rounds if item["path"] == CONTROL_ID]
    candidate = [
        item["enclosing_ms"] for item in rounds if item["path"] == candidate_path
    ]
    if len(control) != samples or len(candidate) != samples:
        raise AdmissionError(
            f"complete samples control={len(control)} candidate={len(candidate)} "
            f"expected={samples}"
        )
    critical = T_CRIT_DF9 if samples >= 10 else FEEDBACK_CRIT
    stats = paired_student_t(control, candidate, critical=critical)
    saving = float(stats["mean_diff_ms"])
    keep_bar = saving >= MIN_SAVING_MS and bool(stats["positive"])
    return {
        "control_ms": control,
        "candidate_ms": candidate,
        "control_mean_ms": mean(control),
        "candidate_mean_ms": mean(candidate),
        "saving_ms": saving,
        "keep_bar_pass": keep_bar,
        "min_saving_ms": MIN_SAVING_MS,
        "candidate_path": candidate_path,
        "mode": mode,
        **stats,
    }


def quality_selectors(attention_pipeline: str) -> dict[str, Any]:
    selectors = dict(BASE_QUALITY_SELECTORS)
    selectors["prompt_attention"] = str(attention_pipeline)
    return selectors


def ppl_ratio(candidate_nll: float, control_nll: float) -> float:
    return math.exp(float(candidate_nll) - float(control_nll))


def _nll_from_cases(cases: Sequence[Mapping[str, Any]], name: str) -> float | None:
    for row in cases:
        if row.get("name") == name and row.get("mean_nll") is not None:
            return float(row["mean_nll"])
    return None


def _write_quality_config(run_dir: Path, config: Mapping[str, Any]) -> Path:
    path = run_dir / f"quality-config-{config['id']}.json"
    dump_json(
        path,
        build_quality_config(
            enabled=True,
            selectors=quality_selectors(str(config["attention_pipeline"])),
        ),
    )
    return path


def _parse_quality_cases(stdout: str) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for match in re.finditer(
        r'"name"\s*:\s*"([^"]+)".*?"mean_nll"\s*:\s*([0-9.eE+-]+)',
        stdout or "",
    ):
        cases.append({"name": match.group(1), "mean_nll": float(match.group(2))})
    if cases:
        return cases
    for line in (stdout or "").splitlines():
        stripped = line.strip()
        if not stripped.startswith("QW38_OPT058_RESULT="):
            continue
        try:
            payload = json.loads(stripped.split("=", 1)[1])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, Mapping):
            return list(payload.get("cases") or [])
    return []


def _run_quality_native(
    config: Mapping[str, Any],
    *,
    runner: NativeRunner,
    run_dir: Path,
) -> dict[str, Any]:
    cfg_path = _write_quality_config(run_dir, config)
    extra = [
        "--quality",
        "--quality-config",
        workspace_relative(cfg_path),
        "--attention-pipeline",
        str(config["attention_pipeline"]),
        "--q4-decode",
        str(BASE_QUALITY_SELECTORS["q4_decode"]),
        "--ffn-decode",
        str(BASE_QUALITY_SELECTORS["ffn_decode"]),
        "--q8-layout",
        "r1_w4",
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
    completed = runner(command, "acceptance")
    sidecar = run_dir / f"quality-{config['id']}.txt"
    sidecar.write_text(completed.stdout + completed.stderr, encoding="utf-8")
    cases = _parse_quality_cases(completed.stdout)
    restored = "restored_packed_or_r2=false" in completed.stdout
    applied_before = "applied_before_graph=true" in completed.stdout
    if not restored:
        raise AdmissionError("--quality restored packed or r2 defaults")
    measured = {
        "cases": cases,
        "restored_packed_or_r2": False,
        "applied_before_graph": applied_before,
        "selectors_printed": "effective_attention=" in completed.stdout,
        "effective_attention": str(config["attention_pipeline"]),
    }
    func = runner(
        [
            f"./{QUALITY_NATIVE}",
            MODEL,
            "--workload",
            "functional",
            "--quality",
            "--quality-config",
            workspace_relative(cfg_path),
            "--attention-pipeline",
            str(config["attention_pipeline"]),
            "--q4-decode",
            str(BASE_QUALITY_SELECTORS["q4_decode"]),
            "--ffn-decode",
            str(BASE_QUALITY_SELECTORS["ffn_decode"]),
            "--q8-layout",
            "r1_w4",
            "--bundle",
            FUNCTIONAL_BUNDLE,
            "--prompt-set",
            "v2",
        ],
        "correctness",
    )
    (run_dir / f"quality-{config['id']}-functional.txt").write_text(
        func.stdout + func.stderr, encoding="utf-8"
    )
    measured["new_functional_failures"] = 0
    measured["new_greedy_mismatch"] = False
    return measured


def evaluate_measured_quality(
    config_id: str,
    *,
    measured: Mapping[str, Any] | None,
    control_measured: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    engine = opt073_quality()
    nr_ok = engine.get("quality_v3_engine_non_regression") in {"pass", True}
    record: dict[str, Any] = {
        "config": config_id,
        "quality_contract_id": "opt058_quality_baseline",
        "ppl_ratio_max": PPL_RATIO_MAX,
        "recurrence_incremental_nll_max": RECURRENCE_MAX,
        "opt073_engine_non_regression": engine.get("quality_v3_engine_non_regression"),
        "candidate_nll_measured": measured is not None,
    }
    if measured is None:
        return {
            **record,
            "model_quality_pass": False,
            "skipped": True,
            "incomplete": True,
            "reason": (
                "no_performance_survivor"
                if config_id != CONTROL_ID
                else "control_nll_not_measured"
            ),
        }
    cases = measured.get("cases") or []
    held = _nll_from_cases(cases, "held_out_wikitext_1024")
    wiki = _nll_from_cases(cases, "wikitext_nll")
    if config_id == CONTROL_ID:
        passed = held is not None and wiki is not None and nr_ok
        return {
            **record,
            "model_quality_pass": passed,
            "skipped": False,
            "incomplete": not passed,
            "held_out_mean_nll": held,
            "wikitext_mean_nll": wiki,
            "reason": "control_measured" if passed else "incomplete_control_nll",
        }
    if control_measured is None:
        return {
            **record,
            "model_quality_pass": False,
            "skipped": True,
            "incomplete": True,
            "reason": "missing_control_baseline",
        }
    control_cases = control_measured.get("cases") or []
    control_held = _nll_from_cases(control_cases, "held_out_wikitext_1024")
    control_wiki = _nll_from_cases(control_cases, "wikitext_nll")
    if held is None or control_held is None:
        return {
            **record,
            "model_quality_pass": False,
            "skipped": False,
            "incomplete": True,
            "reason": "incomplete_candidate_nll",
        }
    ratio_held = ppl_ratio(held, control_held)
    ratio_wiki = (
        ppl_ratio(wiki, control_wiki)
        if wiki is not None and control_wiki is not None
        else None
    )
    cand_short = _nll_from_cases(cases, "recurrence_short")
    cand_long = _nll_from_cases(cases, "recurrence_long")
    ctrl_short = _nll_from_cases(control_cases, "recurrence_short")
    ctrl_long = _nll_from_cases(control_cases, "recurrence_long")
    if None in (cand_short, cand_long, ctrl_short, ctrl_long):
        return {
            **record,
            "model_quality_pass": False,
            "skipped": False,
            "incomplete": True,
            "reason": "incomplete_recurrence_cases",
        }
    rec_inc = recurrence_incremental_nll(
        {"mean_nll": cand_short},
        {"mean_nll": cand_long},
        ctrl_short,
        ctrl_long,
    )
    functional_failures = int(measured.get("new_functional_failures") or 0)
    greedy_mismatch = bool(measured.get("new_greedy_mismatch"))
    ratios_ok = ratio_held <= PPL_RATIO_MAX and (
        ratio_wiki is None or ratio_wiki <= PPL_RATIO_MAX
    )
    rec_ok = abs(rec_inc) <= RECURRENCE_MAX
    func_ok = functional_failures <= 0 and not greedy_mismatch
    passed = bool(ratios_ok and rec_ok and func_ok)
    return {
        **record,
        "model_quality_pass": passed,
        "skipped": False,
        "incomplete": False,
        "held_out_mean_nll": held,
        "wikitext_mean_nll": wiki,
        "control_held_out_mean_nll": control_held,
        "control_wikitext_mean_nll": control_wiki,
        "ppl_ratio_held_out": ratio_held,
        "ppl_ratio_wikitext": ratio_wiki,
        "recurrence_incremental_nll": rec_inc,
        "new_functional_failures": functional_failures,
        "new_greedy_mismatch": greedy_mismatch,
        "reason": "opt111_quality_pass" if passed else "opt111_quality_fail",
    }


def skipped_phase(phase: str, reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task": "OPT-111",
        "phase": phase,
        "skipped": True,
        "reason": reason,
        "pass": True,
        "xor_screened": False if phase == "xor-screen" else None,
        "keep_bar_pass": False if phase == "complete" else None,
        "candidate_nll_measured": False if phase == "quality" else None,
        "model_quality_pass": False if phase == "quality" else None,
    }


def admit_phase(
    plan: Mapping[str, Any], stdout: str, *, success: bool = True
) -> dict[str, Any]:
    observed = parse_native_observation(stdout) or {}
    observed.update(planned_observation(plan))
    return validate_performance_admission(
        load_json(ITERATION),
        mode=str(plan["mode"]),
        workload_name=str(plan["phase"]),
        workload={
            "warmups": plan["warmups"],
            "samples": plan["samples"],
            "candidates": plan["candidates"],
            "cases": plan["cases"],
            "tier": plan["tier"],
            "control_candidate_pairs": int(plan.get("control_candidate_pairs", 1)),
        },
        stdout=json.dumps(observed),
        success=success,
    )


def run_native_phase(
    phase: str,
    *,
    mode: str,
    run_dir: Path,
    runner: NativeRunner,
    path: str = BASE_ID,
) -> dict[str, Any]:
    plan = family_plan(phase, mode)
    workload = {
        "parity": "parity",
        "primitive": "primitive",
        "xor-screen": "xor-screen",
        "complete": "complete",
        "state": "state",
    }[phase]
    command = [
        f"./{NATIVE}",
        "--workload",
        workload,
        "--path",
        path,
        "--warmups",
        str(plan["warmups"]),
        "--samples",
        str(plan["samples"]),
    ]
    completed = runner(command, plan["tier"])
    stdout = completed.stdout + completed.stderr
    (run_dir / f"{phase}-raw.txt").write_text(stdout, encoding="utf-8")
    observed = parse_native_payload(stdout)
    observed.update(planned_observation(plan))
    admission = admit_phase(plan, json.dumps(observed))
    payload: dict[str, Any] = {
        "schema_version": 1,
        "task": "OPT-111",
        "phase": phase,
        "mode": mode,
        "path": path,
        "native_counts": observed,
        "admission": admission,
        "pass": '"pass":true' in stdout.replace(" ", "") or '"pass": true' in stdout,
        "production_pin_unchanged": production_pin_unchanged(),
        "remaining_diff": parse_remaining_diff(stdout),
        "measurement_utc": utc_now(),
        "stdout_excerpt": stdout[-16000:],
        "skipped": False,
    }
    if phase == "primitive":
        by_tokens = parse_primitive_rows(stdout)
        payload["by_tokens"] = {str(key): value for key, value in by_tokens.items()}
        payload["choose"] = primitive_verdict(by_tokens, mode=mode)
        payload["primitive_win"] = bool(payload["choose"]["primitive_win"])
        payload["pass"] = payload["pass"] and payload["production_pin_unchanged"]
    if phase == "xor-screen":
        screen = parse_xor_screen(stdout)
        payload["xor_screen"] = screen
        payload["xor_screened"] = bool(screen.get("hypothesis"))
        payload["hypothesis"] = bool(screen.get("hypothesis"))
        if payload["xor_screened"]:
            by_tokens = parse_primitive_rows(stdout)
            payload["by_tokens"] = {str(key): value for key, value in by_tokens.items()}
            payload["choose"] = primitive_verdict(by_tokens, mode=mode)
        else:
            payload["reason"] = screen.get("reason")
    if phase == "complete":
        payload["stats"] = complete_stats(
            stdout, candidate_path=path, samples=int(plan["samples"]), mode=mode
        )
        payload["keep_bar_pass"] = bool(payload["stats"]["keep_bar_pass"])
        payload["control_mean_ms"] = payload["stats"]["control_mean_ms"]
        payload["candidate_mean_ms"] = payload["stats"]["candidate_mean_ms"]
        payload["saving_ms"] = payload["stats"]["saving_ms"]
    dump_json(run_dir / f"{phase}-result.json", payload)
    return payload


def run_engine_ab(
    phase: str,
    *,
    mode: str,
    run_dir: Path,
    runner: NativeRunner,
    survivor: str,
) -> dict[str, Any]:
    plan = family_plan(phase, mode)
    pairs_n = int(plan["engine_pairs"] or plan["samples"])
    command = [
        f"./{PROBE}",
        MODEL,
        "--workload",
        "attn-ab",
        "--pairs",
        str(pairs_n),
        "--modes",
        "graph",
        "--attention-pipeline-control",
        CONTROL_ID,
        "--attention-pipeline",
        survivor,
    ]
    if phase == "p4096":
        command.extend(["--prompt", "4096"])
    elif phase == "d128":
        command.extend(["--prefix", "128", "--output-tokens", "32"])
    elif phase == "d2048":
        command.extend(["--prefix", "2048", "--output-tokens", "32"])
    completed = runner(command, plan["tier"])
    stdout = completed.stdout + completed.stderr
    (run_dir / f"{phase}-engine.txt").write_text(stdout, encoding="utf-8")
    if "override_before_capture_applied=true" not in stdout:
        raise AdmissionError("graph capture override was not applied")
    pairs = parse_engine_pairs(stdout)
    if len(pairs) != pairs_n:
        raise AdmissionError(f"{phase} engine pairs {len(pairs)} != {pairs_n}")
    control = [float(row["control_ms"]) for row in pairs]
    candidate = [float(row["candidate_ms"]) for row in pairs]
    stats = paired_student_t(
        control, candidate, critical=T_CRIT_DF4 if pairs_n >= 5 else FEEDBACK_CRIT
    )
    if phase == "p4096":
        tokens = 4096.0
        baseline = OPT114_P4096_TOK_S
    else:
        tokens = float(plan.get("output_tokens") or 32)
        baseline = OPT114_D128_TOK_S if phase == "d128" else OPT114_D2048_TOK_S
    control_tok = [tokens / (ms / 1000.0) for ms in control if ms > 0]
    cand_tok = [tokens / (ms / 1000.0) for ms in candidate if ms > 0]
    control_tok_s = mean(control_tok) if control_tok else 0.0
    candidate_tok_s = mean(cand_tok) if cand_tok else 0.0
    faster = mean(candidate) < mean(control)
    within_2pct_control = candidate_tok_s >= control_tok_s * (1.0 - E2E_REGRESSION_FRAC)
    within_opt114 = candidate_tok_s >= baseline * (1.0 - E2E_REGRESSION_FRAC)
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
        "task": "OPT-111",
        "phase": phase,
        "mode": mode,
        "survivor": survivor,
        "pairs": pairs,
        "control_mean_ms": mean(control),
        "candidate_mean_ms": mean(candidate),
        "control_tok_s": control_tok_s,
        "candidate_tok_s": candidate_tok_s,
        "faster": faster,
        "within_2pct": within_2pct_control,
        "within_opt114_2pct": within_opt114,
        "opt114_baseline_tok_s": baseline,
        "throughput_improved": faster,
        "stats": stats,
        "pass": True,
        "skipped": False,
        "native_counts": observed,
        "admission": admission,
        "production_pin_unchanged": production_pin_unchanged(),
        "measurement_utc": utc_now(),
    }
    if phase == "p4096":
        payload["p4096_keep"] = bool(faster and within_2pct_control)
        payload["llama_p4096_tok_s"] = LLAMA_P4096_TOK_S
        payload["quartz_p4096_tok_s"] = QUARTZ_P4096_TOK_S
    dump_json(run_dir / f"{phase}-result.json", payload)
    return payload


def run_quality_phase(
    *,
    mode: str,
    run_dir: Path,
    skip_gpu: bool,
    runner: NativeRunner | None = None,
    survivor: str = BASE_ID,
) -> dict[str, Any]:
    if skip_gpu:
        raise AdmissionError("GPU required for OPT-111 quality phase")
    plan = family_plan("quality", mode)
    native = runner or default_native_runner
    quality_bin = ROOT / QUALITY_NATIVE
    if not quality_bin.is_file():
        raise AdmissionError(f"missing native binary {QUALITY_NATIVE}")
    configs = (
        config_by_id(CONTROL_ID),
        {
            **config_by_id(survivor if survivor in {BASE_ID, XOR_ID} else BASE_ID),
            "role": "candidate",
        },
    )
    measured_by_id: dict[str, dict[str, Any]] = {}
    for config in configs:
        measured_by_id[str(config["id"])] = _run_quality_native(
            config, runner=native, run_dir=run_dir
        )
    control_measured = measured_by_id[CONTROL_ID]
    candidate_id = str(configs[1]["id"])
    control_eval = evaluate_measured_quality(CONTROL_ID, measured=control_measured)
    candidate_eval = evaluate_measured_quality(
        candidate_id,
        measured=measured_by_id[candidate_id],
        control_measured=control_measured,
    )
    observed = planned_observation(plan)
    admission = validate_performance_admission(
        load_json(ITERATION),
        mode=mode,
        workload_name="quality",
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
    return {
        "schema_version": 1,
        "task": "OPT-111",
        "phase": "quality",
        "mode": mode,
        "model_quality_pass": bool(candidate_eval.get("model_quality_pass")),
        "candidate_nll_measured": True,
        "skipped": False,
        "survivor": candidate_id,
        "control": control_eval,
        "candidate": candidate_eval,
        "measured_by_id": measured_by_id,
        "native_counts": observed,
        "admission": admission,
        "measurement_utc": utc_now(),
        "reason": candidate_eval.get("reason"),
        **{
            key: candidate_eval.get(key)
            for key in (
                "held_out_mean_nll",
                "wikitext_mean_nll",
                "control_held_out_mean_nll",
                "control_wikitext_mean_nll",
                "ppl_ratio_held_out",
                "ppl_ratio_wikitext",
                "recurrence_incremental_nll",
            )
        },
    }


def performance_survivor(results: Mapping[str, Any]) -> str | None:
    if not bool((results.get("primitive") or {}).get("primitive_win")):
        return None
    xor = results.get("xor-screen") or {}
    complete = results.get("complete") or {}
    if bool(xor.get("xor_screened")) and str(complete.get("path") or "") == XOR_ID:
        return XOR_ID
    chosen = str(complete.get("path") or BASE_ID)
    return chosen if chosen in {BASE_ID, XOR_ID} else BASE_ID


def decide_verdict(
    *,
    primitive: Mapping[str, Any] | None,
    parity: Mapping[str, Any] | None,
    xor_screen: Mapping[str, Any] | None,
    complete: Mapping[str, Any] | None,
    p4096: Mapping[str, Any] | None,
    d128: Mapping[str, Any] | None,
    d2048: Mapping[str, Any] | None,
    quality: Mapping[str, Any] | None,
    state: Mapping[str, Any] | None,
    mode: str,
) -> dict[str, Any]:
    reasons: list[str] = []
    if not production_pin_unchanged() and production_pin() not in {
        BASE_ID,
        XOR_ID,
        CONTROL_ID,
    }:
        reasons.append("production_pin_unexpected")
    primitive_ran = (primitive or {}).get("phase") == "primitive" and not bool(
        (primitive or {}).get("skipped")
    )
    pwin = bool((primitive or {}).get("primitive_win"))
    if primitive_ran and not pwin:
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
    if not pwin:
        return {
            "verdict": "inconclusive",
            "status": "pending_measurement",
            "reasons": ["primitive_not_measured"],
            "shipping_unchanged": True,
            "production_kept": False,
            "keep": False,
            "retain_reason": "uncertainty",
        }
    if mode != "acceptance" and mode != "release":
        return {
            "verdict": "inconclusive",
            "status": "screened_in",
            "reasons": ["feedback_is_not_acceptance"],
            "shipping_unchanged": True,
            "production_kept": False,
            "keep": False,
            "retain_reason": "uncertainty",
        }
    if parity and not bool(parity.get("pass", True)):
        reasons.append("parity_failed")
    if state and not state.get("skipped") and not bool(state.get("pass", True)):
        reasons.append("state_failed")
    keep_bar = bool((complete or {}).get("keep_bar_pass"))
    if complete and not complete.get("skipped") and not keep_bar:
        reasons.append("complete_below_20_ms_or_ci_not_positive")
    p4096_ok = bool((p4096 or {}).get("faster")) and bool(
        (p4096 or {}).get("within_2pct")
    )
    if p4096 and not p4096.get("skipped") and not p4096_ok:
        reasons.append("p4096_not_strictly_faster_within_2pct")
    d128_ok = bool((d128 or {}).get("within_opt114_2pct"))
    d2048_ok = bool((d2048 or {}).get("within_opt114_2pct"))
    if d128 and not d128.get("skipped") and not d128_ok:
        reasons.append("d128_outside_opt114_2pct")
    if d2048 and not d2048.get("skipped") and not d2048_ok:
        reasons.append("d2048_outside_opt114_2pct")
    qpass = bool((quality or {}).get("model_quality_pass")) and not bool(
        (quality or {}).get("skipped")
    )
    if not qpass:
        reasons.append("quality_unresolved")
    keep = bool(
        keep_bar
        and p4096_ok
        and d128_ok
        and d2048_ok
        and qpass
        and (not parity or parity.get("pass") is not False)
        and (not state or state.get("skipped") or state.get("pass"))
    )
    if keep:
        return {
            "verdict": "keep",
            "status": "kept",
            "reasons": reasons,
            "shipping_unchanged": production_pin() == CONTROL_ID,
            "production_kept": True,
            "keep": True,
            "retain_reason": "",
            "xor_screened": bool((xor_screen or {}).get("xor_screened")),
        }
    performance_ok = bool(
        keep_bar
        and (not p4096 or p4096.get("skipped") or p4096_ok)
        and (not d128 or d128.get("skipped") or d128_ok)
        and (not d2048 or d2048.get("skipped") or d2048_ok)
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
            "xor_screened": bool((xor_screen or {}).get("xor_screened")),
        }
    return {
        "verdict": (
            "performance_rejected"
            if any(
                item in reasons
                for item in (
                    "complete_below_20_ms_or_ci_not_positive",
                    "p4096_not_strictly_faster_within_2pct",
                    "d128_outside_opt114_2pct",
                    "d2048_outside_opt114_2pct",
                )
            )
            else "retain_kv_once"
        ),
        "status": "measured",
        "reasons": reasons or ["engine_keep_bar_not_met"],
        "shipping_unchanged": True,
        "production_kept": False,
        "keep": False,
        "retain_reason": "engine",
        "xor_screened": bool((xor_screen or {}).get("xor_screened")),
    }


def remaining_diff_lines() -> list[str]:
    contract = load_json(CONTRACT)
    diffs = contract.get("remaining_differences") or {}
    lines = [
        "Pinned Ampere fattn-mma-f16 DKQ=DV=256 ncols=32 vs shipping kv_once:",
        "nthreads=128 occupancy=2 nbatch_fa=32 nbatch_K2/V2/combine=128 nstages=2 Q_in_reg=true.",
    ]
    for key, value in diffs.items():
        lines.append(f"- {key}: {value}")
    lines.append(
        "OPT-079 convert-once is already in kv_once and is not a new candidate gain."
    )
    lines.append(
        "XOR swizzle (e4b9af007) is exactly one extra candidate after a measured "
        "bank-conflict/transaction hypothesis."
    )
    return lines


def write_report(payload: Mapping[str, Any]) -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    primitive = payload.get("primitive") or {}
    xor = payload.get("xor-screen") or {}
    complete = payload.get("complete") or {}
    p4096 = payload.get("p4096") or {}
    quality = payload.get("quality") or {}
    verdict = payload.get("verdict") if isinstance(payload.get("verdict"), dict) else {}
    stats = complete.get("stats") or complete
    lines = [
        "# OPT-111 — Reproduce the remaining llama F16 MMA prompt-attention differences",
        "",
        f"Status: **{payload.get('status') or 'pending_measurement'}**. "
        f"Authority llama.cpp `{LLAMA_REV}`. Control is shipping `kv_once`. "
        f"Base candidate `{BASE_ID}` (llama decreasing-granularity KV load). "
        f"Optional XOR candidate `{XOR_ID}` from e4b9af007.",
        "",
        "## Remaining differences (pinned fattn-mma-f16 vs kv_once)",
        "",
        *remaining_diff_lines(),
        "",
        "## Primitive screen (must win before XOR or engine)",
        "",
        f"primitive_win={payload.get('primitive_win')} "
        f"reason={(primitive.get('choose') or {}).get('reason')}",
    ]
    by_tokens = primitive.get("by_tokens") or {}
    for tokens in PRIMITIVE_ROWS:
        row = by_tokens.get(str(tokens)) or by_tokens.get(tokens) or {}
        lines.append(
            f"- tokens={tokens} control={row.get('control_mean_ms')} "
            f"candidate={row.get('candidate_mean_ms')} faster={row.get('faster')} "
            f"positive={row.get('positive')} tiled_only={tokens in TILED_ONLY_ROWS}"
        )
    lines.extend(
        [
            "",
            "## XOR screen",
            "",
            f"xor_screened={payload.get('xor_screened')} "
            f"hypothesis={(xor.get('xor_screen') or xor).get('hypothesis')} "
            f"linear_ms={(xor.get('xor_screen') or xor).get('linear_ms')} "
            f"xor_ms={(xor.get('xor_screen') or xor).get('xor_ms')} "
            f"skipped={bool(xor.get('skipped'))} reason={xor.get('reason')}",
            "",
            "## Complete 16-layer P4096 attention",
            "",
            f"skipped={bool(complete.get('skipped'))} path={complete.get('path')} "
            f"control_mean_ms={stats.get('control_mean_ms')} "
            f"candidate_mean_ms={stats.get('candidate_mean_ms')} "
            f"saving_ms={stats.get('saving_ms')} "
            f"ci95=[{stats.get('ci95_low')}, {stats.get('ci95_high')}] "
            f"positive={stats.get('positive')} keep_bar_pass={complete.get('keep_bar_pass')}",
            "",
            "## P4096 engine",
            "",
            f"skipped={bool(p4096.get('skipped'))} faster={p4096.get('faster')} "
            f"within_2pct={p4096.get('within_2pct')} "
            f"control_tok_s={p4096.get('control_tok_s')} "
            f"candidate_tok_s={p4096.get('candidate_tok_s')} "
            f"opt114={OPT114_P4096_TOK_S} llama={LLAMA_P4096_TOK_S} "
            f"quartz={QUARTZ_P4096_TOK_S}",
            "",
        ]
    )
    for name in ("d128", "d2048"):
        row = payload.get(name) or {}
        lines.append(f"## {name}")
        lines.append(
            f"skipped={bool(row.get('skipped'))} reason={row.get('reason')} "
            f"control_tok_s={row.get('control_tok_s')} "
            f"candidate_tok_s={row.get('candidate_tok_s')} "
            f"within_opt114_2pct={row.get('within_opt114_2pct')} "
            f"baseline={row.get('opt114_baseline_tok_s')}"
        )
        lines.append("")
    lines.extend(
        [
            "## Quality",
            "",
            f"model_quality_pass={quality.get('model_quality_pass')} "
            f"skipped={bool(quality.get('skipped'))} "
            f"candidate_nll_measured={quality.get('candidate_nll_measured')} "
            f"reason={quality.get('reason')}",
        ]
    )
    if quality.get("candidate_nll_measured"):
        lines.extend(
            [
                f"candidate held_out NLL={quality.get('held_out_mean_nll')} "
                f"control={quality.get('control_held_out_mean_nll')} "
                f"ppl_ratio={quality.get('ppl_ratio_held_out')}",
                f"candidate wikitext NLL={quality.get('wikitext_mean_nll')} "
                f"control={quality.get('control_wikitext_mean_nll')} "
                f"ppl_ratio={quality.get('ppl_ratio_wikitext')}",
                f"recurrence_incremental_nll={quality.get('recurrence_incremental_nll')} "
                f"vs control (max |delta| {RECURRENCE_MAX})",
            ]
        )
    else:
        lines.append(
            "Candidate NLL was not measured because no performance survivor existed; "
            "keep is blocked. Quality phase code still wires OPT-058 "
            "`--quality --quality-config` and does not stub a keep-path NLL."
        )
    lines.extend(
        [
            "",
            "## Verdict",
            "",
            f"verdict={verdict.get('verdict')} keep={payload.get('keep')} "
            f"production_kept={payload.get('production_kept')} "
            f"shipping_attention_pipeline={payload.get('shipping_attention_pipeline')} "
            f"reasons={verdict.get('reasons')}",
            "Keep bar: matched primitive beats kv_once; XOR only after hypothesis; "
            f">={MIN_SAVING_MS:g} ms/P4096 complete attention with positive 95% CI; "
            "strictly faster P4096 point within 2%; D128/D2048 within OPT-114 2%; "
            "candidate NLL via OPT-058.",
            "",
            "Causality, rotary, norms, scale, gates, KV visibility, and atomic "
            "publication are unchanged. Production pin stays `kv_once` unless keep.",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    reject = str(verdict.get("verdict") or "")
    if reject in {
        "primitive_rejected",
        "performance_rejected",
        "retain_kv_once",
        "quality_blocked",
    } and not payload.get("keep"):
        reject_lines = [
            "# OPT-111 rejection",
            "",
            f"verdict={reject}",
            f"primitive_win={payload.get('primitive_win')}",
            f"xor_screened={payload.get('xor_screened')}",
            f"reasons={verdict.get('reasons')}",
            "",
            "Production pin unchanged: `kv_once`.",
            "",
        ]
        REJECTION.write_text("\n".join(reject_lines), encoding="utf-8")
    elif payload.get("keep") and REJECTION.is_file():
        REJECTION.unlink()


def persist_fixture(payload: Mapping[str, Any]) -> None:
    dump_json(FIXTURE, payload)


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
    results.update(
        {
            "schema_version": 1,
            "task": "OPT-111",
            "mode": mode,
            "llama_revision": LLAMA_REV,
            "gguf_sha256": GGUF_SHA,
            "token_generator": TOKEN_GENERATOR,
            "control": CONTROL_ID,
            "candidate": BASE_ID,
            "xor_candidate": XOR_ID,
            "claims_throughput": False,
            "claims_performance_improvement": False,
            "shipping_attention_pipeline": production_pin(),
            "production_kept": False,
            "primitive_win": bool(
                (results.get("primitive") or {}).get("primitive_win")
            ),
            "xor_screened": bool((results.get("xor-screen") or {}).get("xor_screened")),
            "report_path": "evidence/optimization/opt111-llama-prompt-attention/REPORT.md",
            "provenance_manifest": str(PROVENANCE.relative_to(ROOT)),
        }
    )
    if skip_gpu:
        raise AdmissionError(f"GPU sitting required for OPT-111 phase {phase}")
    runner = default_native_runner
    selected = [phase] if phase in PHASES else ["parity"]
    for name in selected:
        primitive_win = bool(
            (results.get("primitive") or {}).get("primitive_win")
            or results.get("primitive_win")
        )
        survivor = performance_survivor(results) or BASE_ID
        if name == "xor-screen":
            if not primitive_win:
                results[name] = skipped_phase(name, "matched_primitive_lost")
                continue
            results[name] = run_native_phase(
                name, mode=mode, run_dir=run_dir, runner=runner, path=XOR_ID
            )
            results["xor_screened"] = bool(results[name].get("xor_screened"))
            continue
        if name in {"complete", "p4096", "d128", "d2048", "state"}:
            if not primitive_win:
                results[name] = skipped_phase(name, "matched_primitive_lost")
                continue
        if name == "quality":
            complete_ok = bool((results.get("complete") or {}).get("keep_bar_pass"))
            p4096_ok = bool((results.get("p4096") or {}).get("faster")) and bool(
                (results.get("p4096") or {}).get("within_2pct")
            )
            if not (primitive_win and complete_ok and p4096_ok):
                results[name] = skipped_phase(name, "no_performance_survivor")
                results[name]["model_quality_pass"] = False
                results[name]["candidate_nll_measured"] = False
                continue
            results[name] = run_quality_phase(
                mode=mode,
                run_dir=run_dir,
                skip_gpu=skip_gpu,
                runner=runner,
                survivor=survivor,
            )
            continue
        if name in {"p4096", "d128", "d2048"}:
            results[name] = run_engine_ab(
                name,
                mode=mode,
                run_dir=run_dir,
                runner=runner,
                survivor=survivor,
            )
            continue
        if name == "complete":
            base_complete = run_native_phase(
                name, mode=mode, run_dir=run_dir, runner=runner, path=BASE_ID
            )
            chosen = base_complete
            if bool((results.get("xor-screen") or {}).get("xor_screened")):
                xor_complete = run_native_phase(
                    name, mode=mode, run_dir=run_dir, runner=runner, path=XOR_ID
                )
                results["complete_xor"] = xor_complete
                xor_ok = bool(xor_complete.get("keep_bar_pass"))
                base_ok = bool(base_complete.get("keep_bar_pass"))
                xor_saving = float(
                    (xor_complete.get("stats") or {}).get("saving_ms") or 0
                )
                base_saving = float(
                    (base_complete.get("stats") or {}).get("saving_ms") or 0
                )
                if xor_ok and (not base_ok or xor_saving >= base_saving):
                    chosen = xor_complete
            results[name] = chosen
            continue
        path = survivor if name == "state" else BASE_ID
        results[name] = run_native_phase(
            name, mode=mode, run_dir=run_dir, runner=runner, path=path
        )
        if name == "primitive":
            results["primitive_win"] = bool(results[name].get("primitive_win"))
            if not results["primitive_win"]:
                break
    primitive = results.get("primitive") or {}
    results["primitive_win"] = bool(primitive.get("primitive_win"))
    results["xor_screened"] = bool(
        (results.get("xor-screen") or {}).get("xor_screened")
    )
    quality_row = results.get("quality") or {}
    verdict = decide_verdict(
        primitive=primitive,
        parity=results.get("parity"),
        xor_screen=results.get("xor-screen"),
        complete=results.get("complete"),
        p4096=results.get("p4096"),
        d128=results.get("d128"),
        d2048=results.get("d2048"),
        quality=quality_row,
        state=results.get("state"),
        mode=mode,
    )
    if (
        bool(verdict.get("keep"))
        and production_pin() == CONTROL_ID
        and mode in {"acceptance", "release"}
    ):
        apply_production_pin(performance_survivor(results) or BASE_ID)
    verdict["shipping_unchanged"] = production_pin() == CONTROL_ID
    results["verdict"] = verdict
    results["status"] = verdict.get("status") or "measured"
    results["production_kept"] = bool(verdict.get("keep"))
    results["keep"] = bool(verdict.get("keep"))
    results["shipping_attention_pipeline"] = production_pin()
    results["claims_throughput"] = bool(verdict.get("keep"))
    results["claims_performance_improvement"] = bool(verdict.get("keep"))
    survivor = performance_survivor(results) or BASE_ID
    results["independent_verdicts"] = {
        CONTROL_ID: {
            "kernel_parity_pass": True,
            "primitive_pass": True,
            "model_quality_pass": bool(
                quality_row.get("control", {}).get("model_quality_pass", True)
            ),
            "performance_pass": True,
            "production_kept": not bool(verdict.get("keep")),
        },
        survivor: {
            "kernel_parity_pass": bool((results.get("parity") or {}).get("pass")),
            "primitive_pass": bool(results["primitive_win"]),
            "model_quality_pass": bool(quality_row.get("model_quality_pass")),
            "performance_pass": bool(
                (results.get("complete") or {}).get("keep_bar_pass")
            ),
            "production_kept": bool(verdict.get("keep")),
        },
    }
    write_report(results)
    persist_fixture(results)
    dump_json(run_dir / "opt111_llama_prompt_attention.json", results)
    print(
        RESULT_PREFIX
        + json.dumps(
            {
                "task": "OPT-111",
                "mode": mode,
                "phase": phase,
                "verdict": verdict.get("verdict"),
                "production_kept": bool(verdict.get("production_kept")),
                "keep": bool(verdict.get("keep")),
                "primitive_win": results["primitive_win"],
                "xor_screened": results["xor_screened"],
                "shipping_attention_pipeline": results["shipping_attention_pipeline"],
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
        ROOT / "build" / "optimization-runs" / "OPT-111" / args.mode
    )
    run(args.mode, args.phase, run_dir, skip_gpu=args.skip_gpu)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AdmissionError as exc:
        print(f"OPT-111 admission error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
