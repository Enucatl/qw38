"""OPT-083 ds4-style quality framework for pinned Qwen3.8.

Host-only framework proof with tiny synthetic continuations. Reuses the
pinned llama quality adapter and the OPT-058 quality diagnostic. Does not
run the OPT-084 shipping baseline or OpenRouter.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.quality.compare import refuse_single_boolean  # noqa: E402
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
from tools.quality.remote import in_pytest, openrouter_status  # noqa: E402
from tools.quality.suite import (  # noqa: E402
    HELD_OUT_TARGETS,
    SUITE_CLASSES,
    run_suite,
)
from tools.run_llama_quality_reference import parse_oracle_stdout  # noqa: E402

CONTRACT = ROOT / "pins/opt083_quality_framework_contract.json"
ITERATION = ROOT / "pins/opt083_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt083_quality_framework.json"
REPORT = ROOT / "evidence/optimization/opt083-quality-framework/REPORT.md"
OPT056 = ROOT / "fixtures/opt056_performance_gate.json"
NATIVE = "build/qw38-cuda-opt058-quality-baseline-test"
PHASES = ("identity", "quality-flag", "framework")
PROOF = (
    "no throughput claim",
    "no production kernel change",
    "claims_throughput false",
    "ds4 methodology port not same-model comparison",
    "DeepSeek/GLM fixtures not applicable",
    "OpenRouter default off",
    "missing credentials are not a failure",
    "no pytest network",
    "OPT-056 and OPT-016 remain blocked",
    "OPT-084 freezes numerical gates",
    "held-out 32 is an alarm not kernel admission",
    "quality results are not one boolean",
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


def llama_control_parse_identity() -> dict[str, Any]:
    """Reuse tools/run_llama_quality_reference.parse_oracle_stdout."""
    lines = []
    for position, target in enumerate((3, 1)):
        lines.append(
            "\t".join(
                [
                    "step",
                    "qwen_tf_identity_0",
                    str(position),
                    str(target),
                    f"{-0.2 - 0.01 * position:.9f}",
                    str(target),
                    "4.0",
                    "0",
                    "1.0",
                    "3.0",
                ]
            )
        )
    stdout = "\n".join(lines) + "\n"
    parsed = parse_oracle_stdout(stdout, ("qwen_tf_identity_0",), 0)
    case = parsed["qwen_tf_identity_0"]
    return {
        "adapter": LLAMA_ADAPTER,
        "case": "qwen_tf_identity_0",
        "steps": len(case["steps"]),
        "mean_nll": case["mean_nll"],
        "target_tokens": [row["target_token"] for row in case["steps"]],
        "reused_parser": True,
    }


def identity_payload() -> dict[str, Any]:
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
    control = llama_control_parse_identity()
    return {
        "cases": rows,
        "llama_control": control,
        "native_reuse": [LLAMA_ADAPTER, QUARTZ_NATIVE],
        "vocab_size": VOCAB_SIZE,
        "gguf_sha256": GGUF_SHA,
        "llama_revision": LLAMA_REV,
        "scoring": "teacher_forced_nll",
        "chat_template": "no_thinking",
        "ds4_datasets_not_applicable": list(DS4_DATASETS_NOT_APPLICABLE),
    }


def quality_flag_payload(*, enabled: bool = True) -> dict[str, Any]:
    applied = apply_quality_mode(enabled=enabled)
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
        "selectors": dict(SHIPPING_SELECTORS),
    }


def build_fixture() -> dict[str, Any]:
    identity = identity_payload()
    quality = quality_flag_payload(enabled=True)
    suite = run_suite()
    remote = openrouter_status(enabled=False)
    opt056 = load_json(OPT056)
    single_boolean_refused = False
    try:
        refuse_single_boolean(suite)
    except QualityFrameworkError:
        single_boolean_refused = True
    return {
        "schema_version": 1,
        "task": "OPT-083",
        "status": "quality_framework",
        "claims_throughput": False,
        "claims_performance_improvement": False,
        "llama_revision": LLAMA_REV,
        "gguf_sha256": GGUF_SHA,
        "vocab_size": VOCAB_SIZE,
        "native_reuse": [LLAMA_ADAPTER, NATIVE],
        "quality_mode": quality,
        "identity": identity,
        "suite": suite,
        "openrouter": remote,
        "suite_classes": list(SUITE_CLASSES),
        "held_out_targets": HELD_OUT_TARGETS,
        "ds4_datasets_not_applicable": list(DS4_DATASETS_NOT_APPLICABLE),
        "same_model_comparison_with_ds4": False,
        "opt084_baseline_not_run": True,
        "single_boolean_refused": single_boolean_refused,
        "opt056_tasks_pass": opt056["quality"]["production_optimization"]["tasks"][
            "pass"
        ],
        "opt056_remains_blocked": True,
        "opt016_remains_blocked": True,
        "does_not_replace_opt056": True,
        "does_not_replace_opt016": True,
        "proposed_acceptance": suite["proposed_acceptance"],
        "proof_limit": list(PROOF),
        "report_path": "evidence/optimization/opt083-quality-framework/REPORT.md",
    }


def write_report(fixture: Mapping[str, Any]) -> None:
    dual = fixture["suite"]["classes"]["opt073_dual_verdict"]["quartz"]
    quality = fixture["quality_mode"]
    text = f"""# OPT-083 — ds4-style quality framework for Qwen3.8

Status: **quality framework frozen**. `claims_throughput: false`. No production
kernel change and no speedup are claimed. Authority remains llama.cpp
`{LLAMA_REV}` and GGUF SHA-256 `{GGUF_SHA}`.

This is a methodology port of `../ds4/gguf-tools/quality-testing/`
(teacher-forced continuation NLL, llama control scorer, inspectable per-case
TSV/JSON, comparator with per-example deltas). It is **not** a same-model
comparison with ds4. DeepSeek V4 Flash/PRO and GLM 5.2 official continuation
directories are **not applicable**.

## Scorers

- Quartz: `{NATIVE}` (OPT-058 quality diagnostic, reused)
- llama.cpp control: `{LLAMA_ADAPTER}` (reused; no second adapter)
- Tokenizer/template: Quartz `enable_thinking=false` /
  `<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n`
- Vocabulary: {VOCAB_SIZE}

Identity phase proves Quartz and llama plans share rendered prompts, context
IDs, target tokens, and the teacher-forced NLL definition on frozen Qwen
fixtures owned by this task.

## `--quality` mode

Flag: `{quality["flag"]}`. Selector:
`production_arithmetic_under_test`. Precision stays
`-O2 --fmad=false` / `fast_math=false`. Quality mode disables diagnostic
fallbacks and every catalogued speed-oriented numeric bypass. Unknown
shortcuts fail closed. Graph vs eager, if both run, is
`same_math_equivalence` evidence, not a license to use a different kernel.

Disabled shortcuts: {", ".join(quality["shortcuts"])}.

Shipping selectors recorded (not changed): Q4 `packed` / `paired_staged`,
Q8 `r2_w2`, MMQ `fma_async_x` `i128_j128`, prompt attention `kv_once`,
decode GDN sequential, decode attention `warp_query`, prompt-pair `off`.

## Suite classes

All inspectable; none is a single boolean.

| Class | Role |
|---|---|
| teacher_forced_continuation | NLL per example and aggregate vs llama |
| known_qwen_continuations | deterministic Qwen fixtures; not remote |
| ppl_1024_spans | existing 1024-target WikiText spans; prefix proof here |
| recurrence_nll | short/long incremental NLL |
| opt073_dual_verdict | absolute vs engine non-regression |
| held_out_32_alarm | 32 teacher-forced targets; alarm, not kernel admission |
| finite_vocab_state_consistency | full vocab + continuation state |

OPT-084 scores the full shipping baseline. This increment uses tiny synthetic
continuations for framework proof.

## Retained historical quality

Quality-v2 remains **fail** (both engines emit A on 17+25). Quality-v3
absolute accuracy is **{dual["absolute_task_accuracy"]["status"]}**; engine
non-regression is **{dual["engine_non_regression"]["status"]}**. OPT-056
functional tasks stay failed (`opt056_tasks_pass={fixture["opt056_tasks_pass"]}`).
OPT-056 and OPT-016 remain blocked and are not relabeled.

## OpenRouter

Interface for `{fixture["openrouter"]["model"]}` exists and is **default off**.
Outputs would be labeled `external_scorer` and non-authoritative. Ordinary
tests must not use the network. Missing `OPENROUTER_API_KEY` is success.
This batch does not call the API.

## Proposed acceptance (numerical freeze is OPT-084)

- primary candidate gate: Quartz vs shipping-baseline regression
- PPL ratio ≤ 1.01 on both 1024-target spans
- recurrence incremental NLL ≤ 0.02
- engine non-regression vs OPT-073 v3
- absolute task accuracy visible and separate
- Quartz-vs-llama NLL/continuation deltas inspectable
- no single projection-error substitute for OPT-074

## Proof limit

- no throughput claim
- no production kernel change
- claims_throughput false
- ds4 methodology port not same-model comparison
- DeepSeek/GLM fixtures not applicable
- OpenRouter default off
- missing credentials are not a failure
- no pytest network
- OPT-056 and OPT-016 remain blocked
- OPT-084 freezes numerical gates
- held-out 32 is an alarm not kernel admission
- quality results are not one boolean
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text, encoding="utf-8")


def run_phase(
    phase: str,
    run_dir: Path | None = None,
    *,
    quality: bool = True,
    openrouter: bool = False,
) -> dict[str, Any]:
    if phase not in PHASES:
        raise ValueError(f"unknown OPT-083 phase {phase}")
    if openrouter and in_pytest():
        raise QualityFrameworkError("OpenRouter scorer cannot run in pytest")
    remote = openrouter_status(enabled=openrouter)
    if phase == "identity":
        identity = identity_payload()
        payload: dict[str, Any] = {
            "success": True,
            "result_class": "ok",
            "gpu_work": False,
            "identity_cases": len(identity["cases"]),
            "native_reuse": identity["native_reuse"],
            "llama_control_reused": identity["llama_control"]["reused_parser"],
            "vocab_size": VOCAB_SIZE,
            "claims_throughput": False,
        }
        payload["success"] = (
            all(row["identity"]["pass"] for row in identity["cases"])
            and identity["llama_control"]["reused_parser"]
        )
    elif phase == "quality-flag":
        flag = quality_flag_payload(enabled=quality)
        payload = {
            "success": True,
            "result_class": "ok",
            "gpu_work": False,
            "flag": flag["flag"],
            "quality": flag["enabled"],
            "unknown_shortcut_fail_closed": flag["unknown_shortcut_fail_closed"],
            "disabled_shortcut_count": len(flag["shortcuts"]),
            "argv": flag["applied"]["argv"],
            "claims_throughput": False,
        }
        payload["success"] = (
            flag["unknown_shortcut_fail_closed"]
            and flag["enabled"] is True
            and flag["applied"]["argv"] == [QUALITY_FLAG]
        )
    else:
        fixture = build_fixture()
        write_json(FIXTURE, fixture)
        write_report(fixture)
        payload = {
            "success": True,
            "result_class": "ok",
            "gpu_work": False,
            "suite_classes": list(SUITE_CLASSES),
            "held_out_targets": HELD_OUT_TARGETS,
            "openrouter": remote["status"],
            "opt084_baseline_not_run": True,
            "single_boolean_refused": fixture["single_boolean_refused"],
            "opt056_remains_blocked": True,
            "opt016_remains_blocked": True,
            "claims_throughput": False,
            "fixture": "fixtures/opt083_quality_framework.json",
            "report": "evidence/optimization/opt083-quality-framework/REPORT.md",
        }
        payload["success"] = (
            fixture["single_boolean_refused"] is True
            and fixture["opt056_remains_blocked"] is True
            and fixture["claims_throughput"] is False
            and remote["status"] == "disabled"
        )
    payload["task"] = "OPT-083"
    payload["phase"] = phase
    payload["openrouter_status"] = remote["status"]
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "opt083-result.json").write_text(
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
        help="disable performance shortcuts (documented default for this suite)",
    )
    parser.add_argument(
        "--openrouter",
        action="store_true",
        default=False,
        help="optional external scorer; default off; refused in pytest",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_phase(
        args.phase,
        args.run_dir,
        quality=args.quality,
        openrouter=args.openrouter,
    )
    sys.stdout.write(json.dumps(result, indent=2) + "\n")
    return 0 if result.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
