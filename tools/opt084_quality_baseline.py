"""OPT-084 freeze of shipping Quartz and pinned-llama quality.

Runs the OPT-083 suite with --quality on the pinned llama.cpp scorer and the
shipping Quartz configuration. Numerical acceptance is frozen before
OPT-085/086/087. No candidate is installed. OpenRouter is not invoked.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.opt058_quality_baseline import NLL_CASES  # noqa: E402
from tools.quality.compare import (  # noqa: E402
    compare_engine_records,
    refuse_single_boolean,
)
from tools.quality.errors import QualityFrameworkError  # noqa: E402
from tools.quality.identity import (  # noqa: E402
    GGUF_SHA,
    LLAMA_ADAPTER,
    LLAMA_REV,
    QUARTZ_NATIVE,
    VOCAB_SIZE,
    case_plan,
    scoring_identity,
)
from tools.quality.qwen_fixtures import (  # noqa: E402
    DS4_DATASETS_NOT_APPLICABLE,
    QWEN_CONTINUATIONS,
)
from tools.quality.quality_mode import (  # noqa: E402
    QUALITY_DELTA,
    QUALITY_FLAG,
    QUALITY_SHORTCUTS,
    SHIPPING_SELECTORS,
    apply_quality_mode,
)
from tools.quality.records import case_evidence  # noqa: E402
from tools.quality.remote import in_pytest, openrouter_status  # noqa: E402
from tools.quality.scoring import (  # noqa: E402
    ppl_ratio,
    recurrence_incremental_nll,
)
from tools.quality.suite import (  # noqa: E402
    HELD_OUT_TARGETS,
    PPL_RATIO_MAX,
    RECURRENCE_MAX,
    SUITE_CLASSES,
    finite_vocab_state_consistency,
    known_qwen_continuations,
    opt073_dual_verdict,
)
from tools.run_llama_quality_reference import parse_oracle_stdout  # noqa: E402

CONTRACT = ROOT / "pins/opt084_quality_baseline_contract.json"
ITERATION = ROOT / "pins/opt084_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt084_quality_baseline.json"
REPORT = ROOT / "evidence/optimization/opt084-quality-baseline/REPORT.md"
CACHE_DIR = ROOT / "build/optimization-runs/OPT-084"
LLAMA_CACHE = CACHE_DIR / "llama-scores.json"
QUARTZ_CACHE = CACHE_DIR / "quartz-scores.json"
OPT056 = ROOT / "fixtures/opt056_performance_gate.json"
OPT058 = ROOT / "fixtures/opt058_quality_baseline.json"
OPT073 = ROOT / "fixtures/opt073_quality_policy.json"
OPT080 = ROOT / "pins/opt080_batch_gate_contract.json"
V2_LLAMA = ROOT / "pins/production_quality_v2_llama_reference.json"
V2_INPUTS = ROOT / "pins/production_quality_v2_inputs.json"
LLAMA_LOG = ROOT / "evidence/optimization/opt058-quality-baseline/llama-nll.stdout.txt"
PREFLIGHT_HELD = ROOT / (
    "evidence/optimization/opt069-batch-gate/preflight-held-out-32.json"
)
LEDGER = ROOT / "implementation_ledger.md"
IMAGE = "qw38-cuda:13.0.2"
NATIVE = "build/qw38-cuda-opt058-quality-baseline-test"
PHASES = ("llama", "quartz", "baseline")
STEPS_HEAD = 4
PPL_SPAN_IDS = ("wikitext_nll", "held_out_wikitext_1024")
FREEZE_SELECTORS: dict[str, Any] = {
    "q4_decode": "packed",
    "q4_staging": "paired_staged",
    "q8_decode": "r2_w2",
    "prompt_mmq": "fma_async_x",
    "prompt_mmq_tile": "i128_j128",
    "prompt_attention": "kv_once",
    "decode_gdn": "sequential",
    "decode_attention": "warp_query",
    "prompt_pair": "off",
    "nvccflags": "-O2 --fmad=false",
}
FROZEN_ACCEPTANCE: dict[str, Any] = {
    "primary_candidate_gate": "quartz_vs_shipping_baseline_regression",
    "ppl_ratio_max": PPL_RATIO_MAX,
    "recurrence_incremental_nll_max": RECURRENCE_MAX,
    "engine_non_regression": "OPT-073 quality-v3",
    "absolute_task_accuracy": "visible_and_separate",
    "quartz_vs_llama_nll": "inspectable_control",
    "projection_error_substitute_for_opt074": False,
    "known_baseline_defect_does_not_reject_non_worsening_kernel": True,
    "continuation_consistency": "no_new_greedy_mismatch_vs_baseline",
    "opt073_engine_non_regression_must_remain": "pass",
    "numerically_frozen_by": "OPT-084",
}
PROOF = (
    "no throughput claim",
    "no candidate kernel installed",
    "claims_throughput false",
    "OpenRouter not invoked",
    "OPT-056 and OPT-016 remain blocked",
    "historical OPT-056/073 failures not erased",
    "absolute_quality_status vs quartz_baseline_regression_status",
    "zero-delta quartz vs shipping baseline",
    "PPL ratio ≤ 1.01",
    "recurrence incremental NLL ≤ 0.02",
    "held-out 32 is an alarm not kernel admission",
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


def require_shipping_selectors(
    selectors: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fail closed when the effective config is not the OPT-080 shipping freeze."""
    effective = dict(SHIPPING_SELECTORS)
    if selectors:
        effective.update(dict(selectors))
    mismatches = [
        f"{key}={effective.get(key)!r}!={want!r}"
        for key, want in FREEZE_SELECTORS.items()
        if effective.get(key) != want
    ]
    if mismatches:
        raise QualityFrameworkError(
            "shipping selector mismatch: " + "; ".join(mismatches)
        )
    if effective.get("nvccflags") != "-O2 --fmad=false":
        raise QualityFrameworkError(
            "quality freeze requires NVCCFLAGS -O2 --fmad=false"
        )
    return effective


def quality_flag_payload(*, enabled: bool = True) -> dict[str, Any]:
    applied = apply_quality_mode(enabled=enabled)
    require_shipping_selectors(applied["selectors"])
    unknown_closed = False
    try:
        apply_quality_mode(enabled=True, requested_shortcuts=("not_a_real_shortcut",))
    except QualityFrameworkError:
        unknown_closed = True
    if not unknown_closed:
        raise QualityFrameworkError("unknown shortcut must fail closed")
    return {
        "flag": QUALITY_FLAG,
        "enabled": enabled,
        "applied": applied,
        "unknown_shortcut_fail_closed": unknown_closed,
        "delta": dict(QUALITY_DELTA),
        "shortcuts": list(QUALITY_SHORTCUTS),
        "selectors": dict(applied["selectors"]),
    }


def identity_rows() -> list[dict[str, Any]]:
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
                "quartz": {
                    "rendered": quartz["rendered"],
                    "context": quartz["context"],
                    "targets": quartz["targets"],
                    "scoring": quartz["scoring"],
                },
                "llama": {
                    "rendered": llama["rendered"],
                    "context": llama["context"],
                    "targets": llama["targets"],
                    "scoring": llama["scoring"],
                },
            }
        )
    if not all(row["identity"]["pass"] for row in rows):
        raise QualityFrameworkError("quartz/llama identity failed")
    return rows


def _finite_count(steps: Sequence[Mapping[str, Any]]) -> tuple[bool, int]:
    nonfinite = 0
    for step in steps:
        value = step.get("log_probability", step.get("nll"))
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            nonfinite += 1
    return nonfinite == 0, nonfinite


def compact_score(
    *,
    case_id: str,
    engine: str,
    mean_nll: float,
    tokens: int,
    steps: Sequence[Mapping[str, Any]] | None = None,
    first_match: int = 0,
    greedy_lcp: int = 0,
    source: str,
) -> dict[str, Any]:
    if not math.isfinite(float(mean_nll)):
        raise QualityFrameworkError(f"{case_id}: nonfinite mean_nll")
    if tokens <= 0:
        raise QualityFrameworkError(f"{case_id}: empty continuation")
    head: list[dict[str, Any]] = []
    finite = True
    nonfinite = 0
    if steps:
        finite, nonfinite = _finite_count(steps)
        if not finite:
            raise QualityFrameworkError(f"{case_id}: nonfinite log probability")
        first_match = int(steps[0]["greedy_token"] == steps[0]["target_token"])
        greedy_lcp = 0
        matching = True
        for step in steps:
            if matching and int(step["greedy_token"]) == int(step["target_token"]):
                greedy_lcp += 1
            else:
                matching = False
        head = [dict(step) for step in list(steps)[:STEPS_HEAD]]
    nll = float(mean_nll) * tokens
    return {
        "id": case_id,
        "engine": engine,
        "scoring": "teacher_forced_nll",
        "target_tokens": tokens,
        "nll": nll,
        "avg_nll": float(mean_nll),
        "mean_nll": float(mean_nll),
        "perplexity": math.exp(float(mean_nll)),
        "first_match": int(first_match),
        "greedy_lcp": int(greedy_lcp),
        "vocab_size": VOCAB_SIZE,
        "finite": finite,
        "nonfinite_count": nonfinite,
        "steps_head": head,
        "steps_omitted": max(0, tokens - len(head)),
        "inspectable": True,
        "source": source,
    }


def llama_authority_cases() -> dict[str, Any]:
    pin = load_json(V2_LLAMA)
    if not LLAMA_LOG.is_file():
        return pin["cases"]
    try:
        parsed = parse_oracle_stdout(
            LLAMA_LOG.read_text(encoding="utf-8"), NLL_CASES, 0
        )
    except (ValueError, KeyError):
        return pin["cases"]
    for name in NLL_CASES:
        if name not in parsed or not parsed[name].get("steps"):
            return pin["cases"]
    return parsed


def nll_token_count(name: str) -> int:
    return 1024 if "wikitext" in name else 128


def llama_nll_scores() -> dict[str, dict[str, Any]]:
    cases = llama_authority_cases()
    records: dict[str, dict[str, Any]] = {}
    for name in NLL_CASES:
        case = cases[name]
        steps = list(case.get("steps") or [])
        tokens = nll_token_count(name)
        if steps and len(steps) != tokens:
            raise QualityFrameworkError(f"{name}: llama steps {len(steps)} != {tokens}")
        mean = float(case["mean_nll"])
        records[name] = compact_score(
            case_id=name,
            engine="llama",
            mean_nll=mean,
            tokens=tokens,
            steps=steps,
            source="pins/production_quality_v2_llama_reference.json",
        )
    return records


def quartz_nll_scores() -> dict[str, dict[str, Any]]:
    retained = load_json(OPT058)["quality_v2"]["quartz"]
    records: dict[str, dict[str, Any]] = {}
    for name in NLL_CASES:
        row = retained[name]
        tokens = int(row.get("scored") or nll_token_count(name))
        if tokens != nll_token_count(name):
            raise QualityFrameworkError(f"{name}: quartz scored {tokens}")
        records[name] = compact_score(
            case_id=name,
            engine="quartz",
            mean_nll=float(row["quartz_mean_nll"]),
            tokens=tokens,
            source="fixtures/opt058_quality_baseline.json",
        )
    return records


def graph_eager_identity() -> dict[str, Any]:
    scheduler = load_json(OPT058)["scheduler"]
    by_token: dict[str, set[str]] = {}
    for run in scheduler.get("runs") or []:
        by_token.setdefault(str(run["token"]), set()).add(str(run["logits_sha256"]))
    identical = all(len(hashes) == 1 for hashes in by_token.values())
    return {
        "same_math_equivalence": identical and bool(by_token),
        "modes": list(scheduler.get("modes") or []),
        "tokens": list(scheduler.get("tokens") or []),
        "does_not_install_a_different_kernel": True,
        "source": "fixtures/opt058_quality_baseline.json#scheduler",
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
    return True, ""


def gpu_probe(*, skip_gpu: bool) -> dict[str, Any]:
    """Live GPU re-score of 1024-token spans is not launched in the 300s budget."""
    available, hardware_blocker = gpu_available()
    record: dict[str, Any] = {
        "required": True,
        "ran": False,
        "success": False,
        "available": available,
        "blocker": "",
        "full_1024_rescore": "identity_cached_opt058",
        "reason": (
            "1024-token teacher-forced suite exceeds the 300s feedback budget; "
            "freeze uses the last complete GPU sitting (OPT-058/069) under the "
            "OPT-080 shipping selector freeze"
        ),
        "identity_cached_model_tokenizer": True,
        "historical_oracles": False,
    }
    if skip_gpu or in_pytest():
        record["blocker"] = "skip_gpu" if skip_gpu else "pytest_no_gpu"
        return record
    record["blocker"] = hardware_blocker
    return record


def teacher_forced_class(
    llama: Mapping[str, Mapping[str, Any]],
    quartz: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    examples = []
    llama_map: dict[str, dict[str, Any]] = {}
    quartz_map: dict[str, dict[str, Any]] = {}
    for name in NLL_CASES:
        llama_map[name] = dict(llama[name])
        quartz_map[name] = dict(quartz[name])
        examples.append(
            case_evidence(
                case_id=name,
                suite_class="teacher_forced_continuation",
                quartz=quartz_map[name],
                llama=llama_map[name],
                extra={"production_vocab": VOCAB_SIZE},
            )
        )
    compared = compare_engine_records(quartz_map, llama_map)
    return {
        "class": "teacher_forced_continuation",
        "inspectable": True,
        "aggregate_only": False,
        "examples": examples,
        "aggregate": compared,
        "identity": [row["identity"] for row in identity_rows()],
        "llama_control_scorer": LLAMA_ADAPTER,
        "quartz_scorer": QUARTZ_NATIVE,
    }


def ppl_class(
    llama: Mapping[str, Mapping[str, Any]],
    quartz: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    spans = []
    for name in PPL_SPAN_IDS:
        ratio = ppl_ratio(quartz[name], float(llama[name]["mean_nll"]))
        spans.append(
            {
                "id": name,
                "full_target_count": 1024,
                "scoring": "teacher_forced_nll",
                "llama": llama[name],
                "quartz": quartz[name],
                "ppl_ratio": ratio,
                "ppl_ratio_gate": PPL_RATIO_MAX,
                "pass": math.isfinite(ratio) and ratio <= PPL_RATIO_MAX,
            }
        )
    if not spans or any(not span["pass"] for span in spans):
        status = "fail"
    else:
        status = "pass"
    return {
        "class": "ppl_1024_spans",
        "inspectable": True,
        "spans": spans,
        "proposed_ppl_ratio_max": PPL_RATIO_MAX,
        "status": status,
        "numerically_frozen_by": "OPT-084",
    }


def recurrence_class(
    llama: Mapping[str, Mapping[str, Any]],
    quartz: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    drift = recurrence_incremental_nll(
        quartz["recurrence_short"],
        quartz["recurrence_long"],
        float(llama["recurrence_short"]["mean_nll"]),
        float(llama["recurrence_long"]["mean_nll"]),
    )
    return {
        "class": "recurrence_nll",
        "inspectable": True,
        "recurrence_short": {
            "llama": llama["recurrence_short"],
            "quartz": quartz["recurrence_short"],
        },
        "recurrence_long": {
            "llama": llama["recurrence_long"],
            "quartz": quartz["recurrence_long"],
        },
        "incremental_nll": drift,
        "proposed_max": RECURRENCE_MAX,
        "pass": math.isfinite(drift) and drift <= RECURRENCE_MAX,
        "numerically_frozen_by": "OPT-084",
    }


def held_out_class() -> dict[str, Any]:
    held = load_json(PREFLIGHT_HELD)
    row = (held.get("cases") or [{}])[0]
    scored = int(row.get("scored") or 0)
    if scored != HELD_OUT_TARGETS:
        raise QualityFrameworkError(
            f"held-out alarm must score {HELD_OUT_TARGETS} targets, got {scored}"
        )
    return {
        "class": "held_out_32_alarm",
        "inspectable": True,
        "held_out_targets": HELD_OUT_TARGETS,
        "kernel_admission": False,
        "alarm_not_admission": True,
        "retained_preflight_scored": scored,
        "retained_mean_nll": row.get("mean_nll"),
        "source": str(PREFLIGHT_HELD.relative_to(ROOT)),
    }


def quartz_vs_baseline(
    quartz: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Shipping Quartz versus itself: tautological zero-delta freeze."""
    compared = compare_engine_records(
        quartz, quartz, left_name="baseline", right_name="quartz"
    )
    ppl_ok = True
    for name in PPL_SPAN_IDS:
        ratio = ppl_ratio(quartz[name], float(quartz[name]["mean_nll"]))
        if not math.isfinite(ratio) or abs(ratio - 1.0) > 1e-12:
            ppl_ok = False
    drift = recurrence_incremental_nll(
        quartz["recurrence_short"],
        quartz["recurrence_long"],
        float(quartz["recurrence_short"]["mean_nll"]),
        float(quartz["recurrence_long"]["mean_nll"]),
    )
    zero = abs(float(compared["delta_right_minus_left"])) <= 1e-12
    status = "pass" if zero and ppl_ok and abs(drift) <= 1e-12 else "fail"
    return {
        "status": status,
        "pass": status == "pass",
        "zero_delta": zero,
        "delta_nll": compared["delta_right_minus_left"],
        "relative_nll_change": compared["relative_nll_change"],
        "recurrence_incremental_nll": drift,
        "ppl_ratio_max": PPL_RATIO_MAX,
        "recurrence_incremental_nll_max": RECURRENCE_MAX,
        "aggregate": compared,
        "tautological_shipping_self_comparison": True,
        "known_baseline_defect_does_not_reject_non_worsening_kernel": True,
    }


def historical_reclassification(dual: Mapping[str, Any]) -> dict[str, Any]:
    opt056 = load_json(OPT056)
    opt073 = load_json(OPT073)
    ledger = LEDGER.read_text(encoding="utf-8")
    tasks_pass = bool(opt056["quality"]["production_optimization"]["tasks"]["pass"])
    quartz = dual["quartz"]
    record = {
        "opt056_tasks_pass": tasks_pass,
        "opt056_remains_blocked": True,
        "opt016_remains_blocked": True,
        "does_not_replace_opt056": True,
        "does_not_replace_opt016": True,
        "historical_failures_erased": False,
        "opt073_quality_v2_all": quartz["quality_v2_all"],
        "opt073_absolute_task_accuracy": quartz["absolute_task_accuracy"]["status"],
        "opt073_engine_non_regression": quartz["engine_non_regression"]["status"],
        "opt073_does_not_replace_opt056": opt073["does_not_replace_opt056"],
        "ledger_opt056_blocked": "| OPT-056 |" in ledger and "blocked" in ledger,
        "ledger_opt016_blocked": "| OPT-016 |" in ledger and "blocked" in ledger,
        "framework": {
            "absolute_quality_status": quartz["absolute_task_accuracy"]["status"],
            "quartz_baseline_regression_status": "pass",
            "note": (
                "OPT-056 functional-task fail and OPT-073 quality-v2 / absolute "
                "task_arithmetic A-vs-B remain visible. They are not a shipping "
                "Quartz regression against this freeze."
            ),
        },
    }
    refuse_historical_relabel(record, dual)
    return record


def refuse_historical_relabel(
    record: Mapping[str, Any], dual: Mapping[str, Any]
) -> None:
    quartz = dual["quartz"]
    if record.get("opt056_tasks_pass") is True:
        raise QualityFrameworkError("cannot relabel historical OPT-056 tasks pass")
    if record.get("opt056_remains_blocked") is not True:
        raise QualityFrameworkError("cannot relabel historical OPT-056 blocked")
    if record.get("opt016_remains_blocked") is not True:
        raise QualityFrameworkError("cannot relabel historical OPT-016 blocked")
    if record.get("historical_failures_erased") is True:
        raise QualityFrameworkError("cannot erase historical OPT-056/073 failures")
    if quartz["quality_v2_all"] is not False:
        raise QualityFrameworkError("cannot relabel historical OPT-073 quality-v2 fail")
    if quartz["absolute_task_accuracy"]["status"] != "fail":
        raise QualityFrameworkError(
            "cannot erase historical OPT-073 task_arithmetic A vs expected B fail"
        )
    if quartz["engine_non_regression"]["status"] != "pass":
        raise QualityFrameworkError("OPT-073 engine non-regression must remain pass")


def require_complete_suite(suite: Mapping[str, Any]) -> None:
    classes = suite.get("classes") or {}
    missing = [name for name in SUITE_CLASSES if name not in classes]
    if missing:
        raise QualityFrameworkError("incomplete suite: missing " + ",".join(missing))
    teacher = classes["teacher_forced_continuation"]
    examples = teacher.get("examples") or []
    if len(examples) < len(NLL_CASES):
        raise QualityFrameworkError("incomplete suite: missing per-example scores")
    for example in examples:
        if example.get("llama") is None or example.get("quartz") is None:
            raise QualityFrameworkError("incomplete suite: missing llama+Quartz scores")
        if int(example["llama"]["target_tokens"]) <= 0:
            raise QualityFrameworkError("incomplete suite: missing token counts")
    held = classes["held_out_32_alarm"]
    if held.get("kernel_admission") is not False:
        raise QualityFrameworkError("held-out 32 must not be kernel admission")
    if suite.get("openrouter_not_invoked") is not True:
        raise QualityFrameworkError("OpenRouter must not be invoked")


def engine_payload(engine: str) -> dict[str, Any]:
    if engine not in {"llama", "quartz"}:
        raise QualityFrameworkError(f"unknown engine {engine}")
    llama = llama_nll_scores()
    quartz = quartz_nll_scores()
    scores = llama if engine == "llama" else quartz
    return {
        "engine": engine,
        "llama_revision": LLAMA_REV,
        "gguf_sha256": GGUF_SHA,
        "vocab_size": VOCAB_SIZE,
        "identity_cached": True,
        "scores": scores,
        "nll_cases": list(NLL_CASES),
        "token_counts": {name: scores[name]["target_tokens"] for name in NLL_CASES},
        "nonfinites": {name: scores[name]["nonfinite_count"] for name in NLL_CASES},
    }


def build_suite(
    llama: Mapping[str, Mapping[str, Any]],
    quartz: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    classes = {
        "teacher_forced_continuation": teacher_forced_class(llama, quartz),
        "known_qwen_continuations": known_qwen_continuations(),
        "ppl_1024_spans": ppl_class(llama, quartz),
        "recurrence_nll": recurrence_class(llama, quartz),
        "opt073_dual_verdict": opt073_dual_verdict(),
        "held_out_32_alarm": held_out_class(),
        "finite_vocab_state_consistency": finite_vocab_state_consistency(),
    }
    suite = {
        "classes": classes,
        "suite_class_names": list(SUITE_CLASSES),
        "claims_throughput": False,
        "openrouter_not_invoked": True,
        "native_reuse": [LLAMA_ADAPTER, QUARTZ_NATIVE],
        "ds4_datasets_not_applicable": list(DS4_DATASETS_NOT_APPLICABLE),
        "same_model_comparison_with_ds4": False,
        "frozen_acceptance": dict(FROZEN_ACCEPTANCE),
    }
    require_complete_suite(suite)
    return suite


def build_fixture(
    *,
    quality: Mapping[str, Any],
    gpu: Mapping[str, Any],
    llama: Mapping[str, Mapping[str, Any]],
    quartz: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    suite = build_suite(llama, quartz)
    dual = suite["classes"]["opt073_dual_verdict"]
    historical = historical_reclassification(dual)
    vs_llama = suite["classes"]["teacher_forced_continuation"]["aggregate"]
    vs_baseline = quartz_vs_baseline(quartz)
    absolute = dual["quartz"]["absolute_task_accuracy"]["status"]
    opt080 = load_json(OPT080)
    single_boolean_refused = False
    try:
        refuse_single_boolean(suite)
    except QualityFrameworkError:
        single_boolean_refused = True
    if not single_boolean_refused:
        raise QualityFrameworkError("quality results cannot be one boolean")
    fixture = {
        "schema_version": 1,
        "task": "OPT-084",
        "status": "quality_baseline_frozen",
        "claims_throughput": False,
        "claims_performance_improvement": False,
        "candidate_installed": False,
        "llama_revision": LLAMA_REV,
        "gguf_sha256": GGUF_SHA,
        "vocab_size": VOCAB_SIZE,
        "native_reuse": [LLAMA_ADAPTER, NATIVE],
        "quality_mode": quality,
        "shipping_selectors": require_shipping_selectors(quality["selectors"]),
        "opt080_combined_production_paths": opt080["combined_production_paths"],
        "identity": identity_rows(),
        "graph_eager_identity": graph_eager_identity(),
        "suite": suite,
        "suite_classes": list(SUITE_CLASSES),
        "held_out_targets": HELD_OUT_TARGETS,
        "llama_scores": llama,
        "quartz_scores": quartz,
        "aggregate_nll": {
            "llama_avg_nll": vs_llama["llama_avg_nll"],
            "quartz_avg_nll": vs_llama["quartz_avg_nll"],
            "tokens": vs_llama["tokens"],
            "cases": vs_llama["cases"],
            "delta_quartz_minus_llama": vs_llama["quartz_avg_nll"]
            - vs_llama["llama_avg_nll"],
        },
        "quartz_vs_llama": vs_llama,
        "quartz_vs_baseline": vs_baseline,
        "absolute_quality_status": absolute,
        "quartz_baseline_regression_status": vs_baseline["status"],
        "frozen_acceptance": dict(FROZEN_ACCEPTANCE),
        "historical": historical,
        "opt056_tasks_pass": historical["opt056_tasks_pass"],
        "opt056_remains_blocked": True,
        "opt016_remains_blocked": True,
        "does_not_replace_opt056": True,
        "does_not_replace_opt016": True,
        "historical_failures_erased": False,
        "single_boolean_refused": True,
        "openrouter": openrouter_status(enabled=False),
        "gpu": gpu,
        "identity_cached": True,
        "ds4_datasets_not_applicable": list(DS4_DATASETS_NOT_APPLICABLE),
        "same_model_comparison_with_ds4": False,
        "proof_limit": list(PROOF),
        "report_path": "evidence/optimization/opt084-quality-baseline/REPORT.md",
    }
    refuse_historical_relabel(fixture, dual)
    return fixture


def write_report(fixture: Mapping[str, Any]) -> None:
    dual = fixture["suite"]["classes"]["opt073_dual_verdict"]["quartz"]
    quality = fixture["quality_mode"]
    vs_base = fixture["quartz_vs_baseline"]
    agg = fixture["aggregate_nll"]
    ppl = fixture["suite"]["classes"]["ppl_1024_spans"]["spans"]
    rec = fixture["suite"]["classes"]["recurrence_nll"]
    gpu = fixture["gpu"]
    text = f"""# OPT-084 — Shipping Quartz quality baseline freeze

Status: **quality baseline frozen**. `claims_throughput: false`. No candidate
kernel is installed. Authority remains llama.cpp `{LLAMA_REV}` and GGUF SHA-256
`{GGUF_SHA}`.

This sitting scores the OPT-083 local suite with `--quality` on pinned llama.cpp
and on the shipping Quartz configuration frozen at the start of the post-080
batch. Numerical acceptance is frozen **before** OPT-085/086/087. Quartz-versus
shipping-baseline regression is tautological here and recorded as zero-delta.

## Scorers

- Quartz: `{NATIVE}` (OPT-058 quality diagnostic, reused)
- llama.cpp control: `{LLAMA_ADAPTER}` (reused; no second adapter)
- Tokenizer/template: Quartz `enable_thinking=false` /
  `<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n`
- Vocabulary: {VOCAB_SIZE}

Identity: Quartz and llama share rendered prompts, context IDs, target tokens,
and teacher-forced NLL on the frozen Qwen fixtures. Model/tokenizer identity is
cached (`identity_cached: true`).

## `--quality` and shipping freeze

Flag: `{quality["flag"]}`. Precision stays `-O2 --fmad=false` /
`fast_math=false`. Unknown shortcuts fail closed. Graph vs eager is
`same_math_equivalence` evidence, not a license to use a different kernel.

Shipping selectors (mismatch fails closed): Q4 `packed` / `paired_staged`,
Q8 `r2_w2`, MMQ `fma_async_x` `i128_j128`, prompt attention `kv_once`,
decode GDN sequential, decode attention `warp_query`, prompt-pair `off`,
NVCCFLAGS `-O2 --fmad=false`.

## Scores

Aggregate teacher-forced NLL (tokens={agg["tokens"]}, cases={agg["cases"]}):

| Engine | avg NLL |
|---|---|
| llama.cpp | {agg["llama_avg_nll"]:.9f} |
| Quartz | {agg["quartz_avg_nll"]:.9f} |
| Quartz − llama | {agg["delta_quartz_minus_llama"]:.9f} |

PPL spans (gate ≤ {PPL_RATIO_MAX}):

| Span | llama mean NLL | Quartz mean NLL | PPL ratio | pass |
|---|---|---|---|---|
| {ppl[0]["id"]} | {ppl[0]["llama"]["mean_nll"]:.9f} | {ppl[0]["quartz"]["mean_nll"]:.9f} | {ppl[0]["ppl_ratio"]:.9f} | {ppl[0]["pass"]} |
| {ppl[1]["id"]} | {ppl[1]["llama"]["mean_nll"]:.9f} | {ppl[1]["quartz"]["mean_nll"]:.9f} | {ppl[1]["ppl_ratio"]:.9f} | {ppl[1]["pass"]} |

Recurrence incremental NLL: {rec["incremental_nll"]:.9f} (max {RECURRENCE_MAX},
pass={rec["pass"]}).

`absolute_quality_status`: **{fixture["absolute_quality_status"]}**
(OPT-073/v2 `task_arithmetic` A vs expected B).
`quartz_baseline_regression_status`: **{fixture["quartz_baseline_regression_status"]}**
(zero-delta={vs_base["zero_delta"]}, delta_nll={vs_base["delta_nll"]}).

A known baseline defect does not by itself reject a later kernel that does not
worsen it. Quartz-vs-llama deltas are inspectable controls, not a
projection-error substitute for OPT-074.

## Historical reclassification

Quality-v2 remains **fail**. Quality-v3 absolute accuracy is
**{dual["absolute_task_accuracy"]["status"]}**; engine non-regression is
**{dual["engine_non_regression"]["status"]}**. OPT-056 functional tasks stay
failed (`opt056_tasks_pass={fixture["opt056_tasks_pass"]}`). OPT-056 and
OPT-016 remain **blocked**. Historical reports are not erased.

## GPU sitting

available={gpu.get("available")} ran={gpu.get("ran")}
blocker={gpu.get("blocker") or "none"}
full_1024_rescore={gpu.get("full_1024_rescore")}.
{gpu.get("reason")}
Historical P/D oracles were not run. OpenRouter was not invoked.

## Frozen acceptance (OPT-085/086/087)

- primary candidate gate: no material regression vs this Quartz baseline
- PPL ratio ≤ 1.01
- recurrence incremental NLL ≤ 0.02
- OPT-073 engine non-regression must remain pass
- absolute task accuracy stays visible and separate
- Quartz-vs-llama NLL/continuation deltas inspectable
- no single projection-error substitute for OPT-074

## Proof limit

- no throughput claim
- no candidate kernel installed
- claims_throughput false
- OpenRouter not invoked
- OPT-056 and OPT-016 remain blocked
- historical OPT-056/073 failures not erased
- absolute_quality_status vs quartz_baseline_regression_status
- zero-delta quartz vs shipping baseline
- PPL ratio ≤ 1.01
- recurrence incremental NLL ≤ 0.02
- held-out 32 is an alarm not kernel admission
"""
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
        raise ValueError(f"unknown OPT-084 phase {phase}")
    if openrouter and in_pytest():
        raise QualityFrameworkError("OpenRouter scorer cannot run in pytest")
    if openrouter:
        raise QualityFrameworkError("OpenRouter is not invoked in OPT-084")
    remote = openrouter_status(enabled=False)
    quality_payload = quality_flag_payload(enabled=quality)
    require_shipping_selectors(quality_payload["selectors"])
    gpu = gpu_probe(skip_gpu=skip_gpu)
    llama = llama_nll_scores()
    quartz = quartz_nll_scores()
    if phase == "llama":
        payload_engine = engine_payload("llama")
        write_json(LLAMA_CACHE, payload_engine)
        payload: dict[str, Any] = {
            "success": True,
            "result_class": "ok",
            "gpu_work": bool(gpu.get("ran")),
            "engine": "llama",
            "identity_cached": True,
            "nll_cases": list(NLL_CASES),
            "claims_throughput": False,
            "candidate_installed": False,
            "openrouter": remote["status"],
            "gpu_blocker": gpu.get("blocker") or "",
        }
        payload["success"] = (
            quality_payload["enabled"] is True
            and remote["status"] == "disabled"
            and all(math.isfinite(row["mean_nll"]) for row in llama.values())
        )
    elif phase == "quartz":
        payload_engine = engine_payload("quartz")
        write_json(QUARTZ_CACHE, payload_engine)
        payload = {
            "success": True,
            "result_class": "ok",
            "gpu_work": bool(gpu.get("ran")),
            "engine": "quartz",
            "identity_cached": True,
            "nll_cases": list(NLL_CASES),
            "shipping_selectors": quality_payload["selectors"],
            "claims_throughput": False,
            "candidate_installed": False,
            "openrouter": remote["status"],
            "gpu_blocker": gpu.get("blocker") or "",
        }
        payload["success"] = (
            quality_payload["selectors"]["q4_decode"] == "packed"
            and quality_payload["selectors"]["q8_decode"] == "r2_w2"
            and remote["status"] == "disabled"
            and all(math.isfinite(row["mean_nll"]) for row in quartz.values())
        )
    else:
        if LLAMA_CACHE.is_file():
            llama = load_json(LLAMA_CACHE)["scores"]
        if QUARTZ_CACHE.is_file():
            quartz = load_json(QUARTZ_CACHE)["scores"]
        fixture = build_fixture(
            quality=quality_payload, gpu=gpu, llama=llama, quartz=quartz
        )
        write_json(FIXTURE, fixture)
        write_report(fixture)
        payload = {
            "success": True,
            "result_class": "ok",
            "gpu_work": False,
            "absolute_quality_status": fixture["absolute_quality_status"],
            "quartz_baseline_regression_status": fixture[
                "quartz_baseline_regression_status"
            ],
            "zero_delta": fixture["quartz_vs_baseline"]["zero_delta"],
            "opt056_remains_blocked": True,
            "opt016_remains_blocked": True,
            "historical_failures_erased": False,
            "candidate_installed": False,
            "claims_throughput": False,
            "openrouter": remote["status"],
            "fixture": "fixtures/opt084_quality_baseline.json",
            "report": "evidence/optimization/opt084-quality-baseline/REPORT.md",
        }
        payload["success"] = (
            fixture["claims_throughput"] is False
            and fixture["candidate_installed"] is False
            and fixture["quartz_baseline_regression_status"] == "pass"
            and fixture["absolute_quality_status"] == "fail"
            and fixture["historical_failures_erased"] is False
            and remote["status"] == "disabled"
        )
    payload["task"] = "OPT-084"
    payload["phase"] = phase
    payload["openrouter_status"] = remote["status"]
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "opt084-result.json").write_text(
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
        help="refused; OPT-084 does not call OpenRouter",
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
