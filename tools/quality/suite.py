"""OPT-083 suite classes. Inspectable, not aggregate-only. Tiny synthetic proof."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from tools.opt073_quality_policy import (
    quality_v3_verdicts,
    retained_generated,
    retained_nll_bundle,
)
from tools.quality.compare import compare_engine_records, evaluate_ppl_contract
from tools.quality.errors import QualityFrameworkError
from tools.quality.identity import (
    GGUF_SHA,
    LLAMA_ADAPTER,
    LLAMA_REV,
    QUARTZ_NATIVE,
    VOCAB_SIZE,
    case_plan,
    score_both_engines,
    scoring_identity,
)
from tools.quality.qwen_fixtures import (
    DS4_DATASETS_NOT_APPLICABLE,
    QWEN_CONTINUATIONS,
    qwen_case_records,
)
from tools.quality.quality_mode import SHIPPING_SELECTORS
from tools.quality.records import case_evidence
from tools.quality.scoring import recurrence_incremental_nll, teacher_forced_nll

ROOT = Path(__file__).resolve().parents[2]
V2_INPUTS = ROOT / "pins/production_quality_v2_inputs.json"
OPT058_FIXTURE = ROOT / "fixtures/opt058_quality_baseline.json"
OPT056 = ROOT / "fixtures/opt056_performance_gate.json"
OPT073_FIXTURE = ROOT / "fixtures/opt073_quality_policy.json"
PREFLIGHT_HELD = ROOT / (
    "evidence/optimization/opt069-batch-gate/preflight-held-out-32.json"
)
SYNTHETIC_VOCAB = 8
PPL_PREFIX = 4
HELD_OUT_TARGETS = 32
PPL_RATIO_MAX = 1.01
RECURRENCE_MAX = 0.02

QUALITY_CONTRACT_SPECS: dict[str, dict[str, Any]] = {
    "opt084_frozen": {
        "quality_contract_id": "opt084_frozen",
        "ppl_ratio_max": 1.01,
        "aggregate_ppl_ratio_max": 1.01,
        "recurrence_incremental_nll_max": 0.02,
        "require_full_candidate_Q": True,
        "allow_incomplete_quality": False,
        "new_functional_failures_max": 0,
        "allow_new_greedy_mismatch": False,
        "allow_changed_inherited_answer": False,
    },
    "opt089_strict": {
        "quality_contract_id": "opt089_strict",
        "ppl_ratio_max": 1.01,
        "aggregate_ppl_ratio_max": 1.01,
        "recurrence_incremental_nll_max": 0.02,
        "require_full_candidate_Q": True,
        "allow_incomplete_quality": False,
        "new_functional_failures_max": 0,
        "allow_new_greedy_mismatch": False,
        "allow_changed_inherited_answer": False,
    },
    "opt091_late_w4_v1": {
        "quality_contract_id": "opt091_late_w4_v1",
        "default_ppl_ratio_max": 1.01,
        "candidate_ppl_ratio_max": 1.015,
        "aggregate_ppl_ratio_max": 1.015,
        "recurrence_incremental_nll_max": 0.02,
        "require_full_candidate_Q": True,
        "allow_incomplete_quality": False,
        "new_functional_failures_max": 0,
        "allow_new_greedy_mismatch": False,
        "allow_changed_inherited_answer": False,
    },
}


def quality_contract_spec(contract_id: str) -> dict[str, Any]:
    spec = QUALITY_CONTRACT_SPECS.get(contract_id)
    if spec is None:
        raise QualityFrameworkError(f"unknown quality contract {contract_id}")
    return dict(spec)


def contract_ppl_ratio_max(contract_id: str | None) -> float:
    if not contract_id:
        return PPL_RATIO_MAX
    spec = quality_contract_spec(contract_id)
    return float(
        spec.get("ppl_ratio_max")
        or spec.get("candidate_ppl_ratio_max")
        or spec.get("default_ppl_ratio_max")
        or PPL_RATIO_MAX
    )


def evaluate_quality_contracts(
    *,
    ratios: Mapping[str, float],
    recurrence_incremental_nll: float,
    functional_failures: int = 0,
    greedy_mismatch: bool = False,
    changed_inherited_answer: bool = False,
    incomplete: bool = False,
    contract_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Evaluate one or more explicit quality contracts without changing defaults."""
    selected = list(contract_ids or QUALITY_CONTRACT_SPECS)
    verdicts: dict[str, dict[str, Any]] = {}
    for contract_id in selected:
        spec = quality_contract_spec(contract_id)
        ppl = evaluate_ppl_contract(ratios, spec)
        rec_ok = float(recurrence_incremental_nll) <= float(
            spec["recurrence_incremental_nll_max"]
        )
        func_ok = int(functional_failures) <= int(spec["new_functional_failures_max"])
        greedy_ok = not greedy_mismatch or bool(spec["allow_new_greedy_mismatch"])
        inherited_ok = (
            not changed_inherited_answer or bool(spec["allow_changed_inherited_answer"])
        )
        complete_ok = not incomplete or bool(spec["allow_incomplete_quality"])
        passed = bool(
            complete_ok
            and ppl["pass"]
            and rec_ok
            and func_ok
            and greedy_ok
            and inherited_ok
        )
        verdicts[contract_id] = {
            "model_quality_pass": passed,
            "ppl": ppl,
            "recurrence_incremental_nll": float(recurrence_incremental_nll),
            "recurrence_pass": rec_ok,
            "functional_failures": int(functional_failures),
            "functional_pass": func_ok,
            "greedy_mismatch": bool(greedy_mismatch),
            "greedy_pass": greedy_ok,
            "changed_inherited_answer": bool(changed_inherited_answer),
            "inherited_pass": inherited_ok,
            "incomplete": bool(incomplete),
            "complete_pass": complete_ok,
        }
    return {"contracts": verdicts, "contract_ids": selected}

SUITE_CLASSES: tuple[str, ...] = (
    "teacher_forced_continuation",
    "known_qwen_continuations",
    "ppl_1024_spans",
    "recurrence_nll",
    "opt073_dual_verdict",
    "held_out_32_alarm",
    "finite_vocab_state_consistency",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def peaked_logits(
    target: int, vocab: int = SYNTHETIC_VOCAB, peak: float = 5.0
) -> list[float]:
    if not 0 <= target < vocab:
        raise QualityFrameworkError(f"target {target} outside synthetic vocab {vocab}")
    logits = [-1.25] * vocab
    logits[target] = peak
    if vocab > 1:
        runner = 0 if target != 0 else 1
        logits[runner] = 1.0
    return logits


def synthetic_steps(
    targets: Sequence[int], *, jitter: float = 0.0
) -> list[list[float]]:
    rows: list[list[float]] = []
    for index, target in enumerate(targets):
        logits = peaked_logits(int(target) % SYNTHETIC_VOCAB)
        if jitter:
            logits = [value + jitter * (index + 1) * 0.01 for value in logits]
        rows.append(logits)
    return rows


def teacher_forced_continuation() -> dict[str, Any]:
    examples: list[dict[str, Any]] = []
    quartz_map: dict[str, dict[str, Any]] = {}
    llama_map: dict[str, dict[str, Any]] = {}
    identity_rows: list[dict[str, Any]] = []
    for case in QWEN_CONTINUATIONS:
        local_targets = [int(token) % SYNTHETIC_VOCAB for token in case["targets"]]
        quartz_logits = synthetic_steps(local_targets)
        llama_logits = synthetic_steps(local_targets)
        scored = score_both_engines(
            case_id=str(case["id"]),
            targets=local_targets,
            quartz_logits=quartz_logits,
            llama_logits=llama_logits,
            vocab_size=SYNTHETIC_VOCAB,
        )
        quartz_plan = case_plan(
            case_id=str(case["id"]),
            user=str(case["user"]),
            context=case["context"],
            targets=case["targets"],
            engine="quartz",
        )
        llama_plan = case_plan(
            case_id=str(case["id"]),
            user=str(case["user"]),
            context=case["context"],
            targets=case["targets"],
            engine="llama",
        )
        identity_rows.append(scoring_identity(quartz_plan, llama_plan))
        quartz_map[str(case["id"])] = scored["quartz"]
        llama_map[str(case["id"])] = scored["llama"]
        examples.append(
            case_evidence(
                case_id=str(case["id"]),
                suite_class="teacher_forced_continuation",
                quartz=scored["quartz"],
                llama=scored["llama"],
                extra={
                    "continuation_text": case["continuation_text"],
                    "framework_vocab": SYNTHETIC_VOCAB,
                    "production_vocab": VOCAB_SIZE,
                },
            )
        )
    compared = compare_engine_records(quartz_map, llama_map)
    return {
        "class": "teacher_forced_continuation",
        "inspectable": True,
        "aggregate_only": False,
        "examples": examples,
        "aggregate": compared,
        "identity": identity_rows,
        "llama_control_scorer": LLAMA_ADAPTER,
        "quartz_scorer": QUARTZ_NATIVE,
    }


def known_qwen_continuations() -> dict[str, Any]:
    return {
        "class": "known_qwen_continuations",
        "inspectable": True,
        "remote": False,
        "ds4_datasets_not_applicable": list(DS4_DATASETS_NOT_APPLICABLE),
        "same_model_comparison_with_ds4": False,
        "cases": qwen_case_records(),
    }


def ppl_1024_spans(contract_id: str | None = None) -> dict[str, Any]:
    inputs = load_json(V2_INPUTS)
    retained = load_json(OPT058_FIXTURE)["quality_v2"]
    prefixes: list[dict[str, Any]] = []
    for name in ("wikitext_nll", "held_out_wikitext_1024"):
        case = inputs["cases"][name]
        targets = list(case["continuation"][:PPL_PREFIX])
        local = [token % SYNTHETIC_VOCAB for token in targets]
        scored = score_both_engines(
            case_id=name,
            targets=local,
            quartz_logits=synthetic_steps(local),
            llama_logits=synthetic_steps(local),
            vocab_size=SYNTHETIC_VOCAB,
        )
        quartz = retained["quartz"][name]
        prefixes.append(
            {
                "id": name,
                "full_target_count": 1024,
                "framework_prefix": PPL_PREFIX,
                "opt084_scores_full_span": True,
                "context_token": list(case["context"])[:1],
                "prefix_targets": targets,
                "scoring": "teacher_forced_nll",
                "retained_quartz_mean_nll": quartz["quartz_mean_nll"],
                "retained_llama_mean_nll": quartz["llama_mean_nll"],
                "retained_ppl_ratio": quartz["ppl_ratio"],
                "ppl_ratio_gate_proposed": contract_ppl_ratio_max(contract_id),
                "framework_prefix_scores": scored,
            }
        )
    proposed = contract_ppl_ratio_max(contract_id)
    return {
        "class": "ppl_1024_spans",
        "inspectable": True,
        "spans": prefixes,
        "historical_opt058_pass": {
            "wikitext_nll": retained["wikitext_nll"]["pass"],
            "held_out_wikitext_1024": retained["held_out_wikitext_1024"]["pass"],
        },
        "proposed_ppl_ratio_max": proposed,
        "quality_contract_id": contract_id,
        "numerically_frozen_by": "OPT-084",
    }


def recurrence_nll(contract_id: str | None = None) -> dict[str, Any]:
    retained = load_json(OPT058_FIXTURE)["quality_v2"]
    short = retained["quartz"]["recurrence_short"]
    long = retained["quartz"]["recurrence_long"]
    drift = recurrence_incremental_nll(
        {"mean_nll": short["quartz_mean_nll"]},
        {"mean_nll": long["quartz_mean_nll"]},
        short["llama_mean_nll"],
        long["llama_mean_nll"],
    )
    return {
        "class": "recurrence_nll",
        "inspectable": True,
        "recurrence_short": short,
        "recurrence_long": long,
        "incremental_nll": drift,
        "retained_incremental_nll": retained["recurrence"]["incremental_nll"],
        "proposed_max": (
            quality_contract_spec(contract_id)["recurrence_incremental_nll_max"]
            if contract_id
            else RECURRENCE_MAX
        ),
        "pass_vs_proposed": drift
        <= (
            quality_contract_spec(contract_id)["recurrence_incremental_nll_max"]
            if contract_id
            else RECURRENCE_MAX
        ),
        "quality_contract_id": contract_id,
        "numerically_frozen_by": "OPT-084",
    }


def opt073_dual_verdict() -> dict[str, Any]:
    nll, authority = retained_nll_bundle()
    quartz, llama = retained_generated()
    inputs = load_json(V2_INPUTS)
    v3_q = quality_v3_verdicts(
        nll, authority, inputs, generated=quartz, engine_logits=quartz
    )
    v3_l = quality_v3_verdicts(
        nll, authority, inputs, generated=llama, engine_logits=llama
    )
    opt056 = load_json(OPT056)
    opt073 = load_json(OPT073_FIXTURE)
    return {
        "class": "opt073_dual_verdict",
        "inspectable": True,
        "quartz": {
            "status": v3_q["status"],
            "absolute_task_accuracy": v3_q["absolute_task_accuracy"],
            "engine_non_regression": v3_q["engine_non_regression"],
            "quality_v2_all": v3_q["quality_v2"]["all"],
        },
        "llama": {
            "status": v3_l["status"],
            "absolute_task_accuracy": v3_l["absolute_task_accuracy"],
            "engine_non_regression": v3_l["engine_non_regression"],
        },
        "historical_failures_retained": True,
        "opt056_tasks_pass": opt056["quality"]["production_optimization"]["tasks"][
            "pass"
        ],
        "opt073_does_not_replace_opt056": opt073["does_not_replace_opt056"],
        "opt073_does_not_replace_opt016": opt073["does_not_replace_opt016"],
        "opt056_remains_blocked": True,
        "opt016_remains_blocked": True,
    }


def held_out_32_alarm() -> dict[str, Any]:
    held = load_json(PREFLIGHT_HELD)
    targets = list(range(HELD_OUT_TARGETS))
    local = [token % SYNTHETIC_VOCAB for token in targets]
    scored = score_both_engines(
        case_id="held_out_32",
        targets=local,
        quartz_logits=synthetic_steps(local),
        llama_logits=synthetic_steps(local),
        vocab_size=SYNTHETIC_VOCAB,
    )
    scored_count = int((held.get("cases") or [{}])[0].get("scored") or 0)
    return {
        "class": "held_out_32_alarm",
        "inspectable": True,
        "held_out_targets": HELD_OUT_TARGETS,
        "kernel_admission": False,
        "alarm_not_admission": True,
        "retained_preflight_scored": scored_count,
        "retained_mean_nll": (held.get("cases") or [{}])[0].get("mean_nll"),
        "framework_scores": scored,
        "opt084_scores_full_alarm": True,
    }


def finite_vocab_state_consistency() -> dict[str, Any]:
    targets = [3, 1, 0, 2]
    full = synthetic_steps(targets)
    quartz = teacher_forced_nll(
        full, targets, case_id="finite_ok", engine="quartz", vocab_size=SYNTHETIC_VOCAB
    )
    truncated_ok = False
    try:
        teacher_forced_nll(
            [row[:4] for row in full],
            targets,
            case_id="truncated",
            engine="quartz",
            vocab_size=SYNTHETIC_VOCAB,
        )
    except QualityFrameworkError:
        truncated_ok = True
    state = []
    context = [7]
    for index, target in enumerate(targets):
        state.append(
            {
                "position": index,
                "context": list(context),
                "target": target,
                "next_context": [*context, target],
            }
        )
        context = [*context, target]
    return {
        "class": "finite_vocab_state_consistency",
        "inspectable": True,
        "production_vocab_size": VOCAB_SIZE,
        "framework_vocab_size": SYNTHETIC_VOCAB,
        "finite": quartz["finite"],
        "incomplete_vocab_rejected": truncated_ok,
        "state_continuation": state,
        "selectors": dict(SHIPPING_SELECTORS),
        "gguf_sha256": GGUF_SHA,
        "llama_revision": LLAMA_REV,
    }


def run_suite() -> dict[str, Any]:
    classes = {
        "teacher_forced_continuation": teacher_forced_continuation(),
        "known_qwen_continuations": known_qwen_continuations(),
        "ppl_1024_spans": ppl_1024_spans(),
        "recurrence_nll": recurrence_nll(),
        "opt073_dual_verdict": opt073_dual_verdict(),
        "held_out_32_alarm": held_out_32_alarm(),
        "finite_vocab_state_consistency": finite_vocab_state_consistency(),
    }
    missing = [name for name in SUITE_CLASSES if name not in classes]
    if missing:
        raise QualityFrameworkError("missing suite class " + ",".join(missing))
    return {
        "classes": classes,
        "suite_class_names": list(SUITE_CLASSES),
        "claims_throughput": False,
        "opt084_baseline_not_run": True,
        "openrouter_not_invoked": True,
        "native_reuse": [LLAMA_ADAPTER, QUARTZ_NATIVE],
        "ds4_datasets_not_applicable": list(DS4_DATASETS_NOT_APPLICABLE),
        "same_model_comparison_with_ds4": False,
        "proposed_acceptance": {
            "quartz_vs_shipping_baseline_regression": "primary_candidate_gate",
            "ppl_ratio_max": PPL_RATIO_MAX,
            "recurrence_incremental_nll_max": RECURRENCE_MAX,
            "engine_non_regression": "OPT-073 quality-v3",
            "absolute_task_accuracy": "visible_and_separate",
            "quartz_vs_llama_nll": "inspectable_control",
            "projection_error_substitute_for_opt074": False,
            "numerically_frozen_by": "OPT-084",
        },
    }
