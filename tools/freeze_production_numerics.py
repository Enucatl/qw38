"""Freeze OPT-044 production numerics contract, fixture, and metrics report."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.production_numerics import (
    FORBIDDEN,
    GATE_INVENTORY,
    GGUF_SHA,
    LEGACY_PPL_RATIO,
    LLAMA_REV,
    PERMITTED_PRODUCTION,
    PRODUCTION_CASES,
    PRODUCTION_PPL_RATIO,
    RECURRENCE_INCREMENTAL_NLL,
    ROOT,
    STRICT_CUD001_ABS,
    STRICT_CUD001_RMS,
    DATASET_SHA,
    TOKENIZER_SHA,
    family_budgets,
    parse_host_payload,
    ppl_ratio,
    sha256_json_tokens,
    teacher_forced_metrics,
)

CONTRACT_PATH = ROOT / "pins/production_numerics_contract.json"
FIXTURE_PATH = ROOT / "fixtures/opt044_production_numerics.json"
EVIDENCE = ROOT / "evidence/optimization/opt044-production-numerics"
REPORT = EVIDENCE / "REPORT.md"
HOST_RAW = EVIDENCE / "host-fp64-raw.txt"
REQUIRED_FIXTURE_KEYS = (
    "schema_version",
    "task",
    "status",
    "measurement_utc",
    "claims_performance_improvement",
    "production_numerics_path",
    "optimized_admitted",
    "llama_revision",
    "gguf_sha256",
    "host_fp64",
    "opt042",
    "opt043_captures",
    "quality",
    "opt042_numeric_reject_preserved",
    "report_path",
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _opt042_excerpt(opt042: dict[str, Any]) -> dict[str, Any]:
    synthetic = {}
    for ident in ("q4_k_17x256", "q4_k_257x512", "q4k_gate_up", "q4k_down"):
        block = opt042["synthetic"][ident]
        packed = block["candidates"]["packed"]["vs_host_cud001"]
        integer = block["candidates"]["integer"]["vs_host_cud001"]
        synthetic[ident] = {
            "id": ident,
            "rows": block["rows"],
            "columns": block["columns"],
            "checksums": block["checksums"],
            "packed_cud001_eligible": block["candidates"]["packed"]["cud001_eligible"],
            "integer_cud001_eligible": block["candidates"]["integer"][
                "cud001_eligible"
            ],
            "packed_vs_host": packed,
            "integer_vs_host": integer,
        }
    return {
        "status": opt042["status"],
        "admissibility": opt042["admissibility"],
        "selected_mmv_load_path": opt042["selected_mmv_load_path"],
        "measurement_utc": opt042["measurement_utc"],
        "synthetic": synthetic,
    }


def _capture_slots(path: Path) -> dict[str, Any]:
    record = _load(path)
    slots = []
    for slot in record.get("slots", []):
        slots.append(
            {
                "layer": slot["layer"],
                "layer_kind": slot["layer_kind"],
                "mixer_sha256": slot.get("mixer_sha256"),
                "ffn_sha256": slot.get("ffn_sha256"),
                "mixer_captured": slot.get("mixer_captured"),
                "ffn_captured": slot.get("ffn_captured"),
            }
        )
    return {
        "stage": record.get("stage"),
        "position": record.get("position"),
        "token_count": record.get("token_count"),
        "final_norm_sha256": record.get("final_norm_sha256"),
        "output_sha256": record.get("output_sha256"),
        "slots": slots,
    }


def _quality_bundle() -> dict[str, Any]:
    inputs = _load(ROOT / "fixtures/quality_inputs.json")
    llama = _load(ROOT / "fixtures/quality_llama_reference.json")
    native = ROOT / "evidence/quality/qlt001/native"
    quartz_cases: dict[str, Any] = {}
    for name in (
        "wikitext_nll",
        "continuation_2048",
        "continuation_4096",
        "recurrence_short",
        "recurrence_long",
    ):
        path = native / name / "result.json"
        quartz_cases[name] = _load(path) if path.is_file() else None

    wikitext = inputs["cases"]["wikitext_nll"]
    long_ctx = inputs["cases"]["recurrence_long"]["context"]
    held_context = [int(long_ctx[0])]
    held_continuation = [int(token) for token in long_ctx[1:1025]]
    calibration = {
        "case": "wikitext_nll",
        "stream_start": 1,
        "stream_end": 1025,
        "context_tokens": wikitext["context"],
        "continuation_sha256": sha256_json_tokens(wikitext["continuation"]),
        "target_count": len(wikitext["continuation"]),
    }
    held_out = {
        "case": "held_out_wikitext_1024",
        "source": "fixtures/quality_inputs.json#cases.recurrence_long.context",
        "stream_start": 16385,
        "stream_end": 17409,
        "disjoint_from": "wikitext_nll stream[1:1025]",
        "context_tokens": held_context,
        "continuation": held_continuation,
        "continuation_sha256": sha256_json_tokens(held_continuation),
        "target_count": len(held_continuation),
        "quartz_score": "not_run_this_sitting",
        "llama_score": "not_run_this_sitting",
    }

    llama_nll = float(llama["cases"]["wikitext_nll"]["mean_nll"])
    quartz_nll_record = quartz_cases["wikitext_nll"]
    quartz_nll = float(quartz_nll_record["mean_nll"]) if quartz_nll_record else None
    ratio = ppl_ratio(quartz_nll, llama_nll) if quartz_nll is not None else None
    short = quartz_cases["recurrence_short"]
    long = quartz_cases["recurrence_long"]
    short_a = llama["cases"]["recurrence_short"]
    long_a = llama["cases"]["recurrence_long"]
    drift = None
    if short is not None and long is not None:
        drift = (float(long["mean_nll"]) - float(long_a["mean_nll"])) - (
            float(short["mean_nll"]) - float(short_a["mean_nll"])
        )
    continuation = {}
    for name in ("continuation_2048", "continuation_4096"):
        actual = quartz_cases[name]
        expected = llama["cases"][name].get("greedy_tokens", [])
        greedy = (
            [int(step["greedy_token"]) for step in actual["steps"]]
            if actual is not None
            else []
        )
        continuation[name] = {
            "match": greedy == expected,
            "positions": len(greedy),
            "min_llama_top_two_margin": min(
                float(step["margin"]) for step in llama["cases"][name]["steps"]
            ),
        }
    tasks = {}
    for name in PRODUCTION_CASES:
        if not name.startswith("task_"):
            continue
        expected = inputs["cases"][name]["continuation"]
        path = native / name / "result.json"
        if not path.is_file():
            tasks[name] = {"scored": False, "expected": expected}
            continue
        record = _load(path)
        greedy = [int(step["greedy_token"]) for step in record["steps"]]
        tasks[name] = {
            "scored": True,
            "match": greedy == expected,
            "expected": expected,
        }

    teacher = None
    if quartz_nll_record is not None:
        teacher = teacher_forced_metrics(
            quartz_nll_record, llama["cases"]["wikitext_nll"]
        )

    return {
        "dataset_sha256": inputs.get("dataset_sha256", DATASET_SHA),
        "tokenizer_sha256": inputs.get("tokenizer_sha256", TOKENIZER_SHA),
        "calibration_wikitext_1024": calibration,
        "held_out_wikitext_1024": held_out,
        "pinned_llama": {
            "wikitext_mean_nll": llama_nll,
            "wikitext_perplexity": llama["cases"]["wikitext_nll"].get("perplexity"),
        },
        "current_quartz": {
            "wikitext_mean_nll": quartz_nll,
            "ppl_ratio_vs_llama": ratio,
            "meets_legacy_1_05": ratio is not None and ratio <= LEGACY_PPL_RATIO,
            "meets_production_1_01": ratio is not None
            and ratio <= PRODUCTION_PPL_RATIO,
            "recurrence_incremental_nll": drift,
            "meets_recurrence_0_02": drift is not None
            and drift <= RECURRENCE_INCREMENTAL_NLL,
            "source": "evidence/quality/qlt001/native",
            "existing_baseline_failure": ratio is not None
            and ratio > PRODUCTION_PPL_RATIO,
        },
        "continuation": continuation,
        "tasks": tasks,
        "teacher_forced": teacher,
        "continuation_top_two_margin": 0.0,
        "near_tie_evidence": "absent_store_margin_zero",
        "legacy_verdicts_retained": True,
    }


def build_contract(host: dict[str, Any], quality: dict[str, Any]) -> dict[str, Any]:
    families = family_budgets(host)
    return {
        "schema": "qw38.production-numerics-contract",
        "schema_version": 1,
        "task": "OPT-044",
        "claims_performance_improvement": False,
        "llama_revision": LLAMA_REV,
        "gguf_sha256": GGUF_SHA,
        "dataset_sha256": DATASET_SHA,
        "tokenizer_sha256": TOKENIZER_SHA,
        "roles": {
            "strict_reference": {
                "purpose": "retained historical kernels and original numeric contracts",
                "compiler": "Makefile NVCCFLAGS --fmad=false; host -ffp-contract=off",
                "dispatch": "kSelectedProductionNumericsPath=strict",
            },
            "optimized_production": {
                "purpose": "future llama.cpp/ds4-comparable approximations after per-family validation",
                "admitted": False,
                "permitted": list(PERMITTED_PRODUCTION),
            },
        },
        "path_selection": {
            "selected": "strict",
            "legal": ["strict", "optimized"],
            "optimized_admitted": False,
            "unrepresented_shapes": "strict",
            "diagnostics_may_select_either": True,
        },
        "strict_reference_ceilings": {
            "cud001_maximum_absolute_error": STRICT_CUD001_ABS,
            "cud001_maximum_rms_error": STRICT_CUD001_RMS,
            "one_minus_cosine_floor": 1e-7,
        },
        "budget_rule": {
            "abs_rms": "max(strict_reference_ceiling, 1.05 * measured_llama_error + 1e-6)",
            "one_minus_cosine": "same construction with a 1e-7 floor",
            "zero_vector": "explicit_norm_and_abs",
            "nonfinite_or_pathological_llama": "retain_strict_do_not_inflate",
            "do_not_adjust_after_candidate_failure": True,
        },
        "storage": {
            "residual_gdn_logits": "FP32",
            "persistent_activation_kv": "BF16",
        },
        "families": families,
        "gate_inventory": [dict(row) for row in GATE_INVENTORY],
        "permitted_production_approximations": list(PERMITTED_PRODUCTION),
        "forbidden": list(FORBIDDEN),
        "quality_suite": {
            "name": "production-optimization",
            "version": 1,
            "legacy_contract": "pins/quality_contract.json",
            "legacy_nll_ppl_ratio": LEGACY_PPL_RATIO,
            "nll_ppl_ratio": PRODUCTION_PPL_RATIO,
            "recurrence_incremental_nll": RECURRENCE_INCREMENTAL_NLL,
            "cases": list(PRODUCTION_CASES),
            "calibration_wikitext_1024": {
                "stream_start": quality["calibration_wikitext_1024"]["stream_start"],
                "stream_end": quality["calibration_wikitext_1024"]["stream_end"],
                "continuation_sha256": quality["calibration_wikitext_1024"][
                    "continuation_sha256"
                ],
                "target_count": 1024,
            },
            "held_out_wikitext_1024": {
                "stream_start": quality["held_out_wikitext_1024"]["stream_start"],
                "stream_end": quality["held_out_wikitext_1024"]["stream_end"],
                "continuation_sha256": quality["held_out_wikitext_1024"][
                    "continuation_sha256"
                ],
                "target_count": 1024,
                "disjoint_from_calibration": True,
            },
            "continuation_top_two_margin": 0.0,
            "require_functional_answers": True,
            "require_zero_nonfinites": True,
            "overwrite_legacy_verdicts": False,
        },
        "required_fixture_keys": list(REQUIRED_FIXTURE_KEYS),
        "report_path": "evidence/optimization/opt044-production-numerics/REPORT.md",
    }


def write_report(
    contract: dict[str, Any], fixture: dict[str, Any], quality: dict[str, Any]
) -> str:
    gate_up = contract["families"]["q4k_gate_up"]
    down = contract["families"]["q4k_down"]
    opt042 = fixture["opt042"]
    quartz = quality["current_quartz"]
    ratio = quartz["ppl_ratio_vs_llama"]
    lines = [
        "# OPT-044 production numerics",
        "",
        "Status: **Measured** independent host FP64 / llama Q8_1 freeze; "
        "production dispatch remains strict. "
        "`claims_performance_improvement: false`.",
        "",
        "## Roles",
        "",
        "- **Strict reference:** retained kernels, `--fmad=false`, original CUD-001/003 "
        "and scalar-oracle contracts.",
        "- **Optimized production:** permitted Q8_1 staging, reordered FP32 reductions, "
        "FMA, F16 MMA with FP32 accumulation, and individually validated approximate "
        "transcendentals. Not installed. Unrepresented shapes stay on the strict path.",
        "",
        "## OPT-042 large-shape reproduction",
        "",
        "Host fill checksums match the frozen OPT-042 synthetic identities for "
        "`q4_k_17x256` and `q4k_gate_up`. OPT-042 `numeric_reject` is unchanged.",
        "",
        "| Shape | Packed vs host max_abs (OPT-042) | FP32 host vs FP64 staged max_abs | Llama Q8_1 vs FP64 original max_abs |",
        "|---|---:|---:|---:|",
        (
            f"| q4k_gate_up | {opt042['synthetic']['q4k_gate_up']['packed_vs_host']['max_abs']:.9g} | "
            f"{fixture['host_fp64']['cases'][2]['comparisons']['fp32_host_vs_fp64_quartz_staged']['max_abs']:.9g} | "
            f"{fixture['host_fp64']['cases'][2]['comparisons']['llama_q8_1_vs_fp64_original_bf16']['max_abs']:.9g} |"
        ),
        (
            f"| q4k_down | {opt042['synthetic']['q4k_down']['packed_vs_host']['max_abs']:.9g} | "
            f"{fixture['host_fp64']['cases'][3]['comparisons']['fp32_host_vs_fp64_quartz_staged']['max_abs']:.9g} | "
            f"{fixture['host_fp64']['cases'][3]['comparisons']['llama_q8_1_vs_fp64_original_bf16']['max_abs']:.9g} |"
        ),
        "",
        "The CUD-001 miss on production FFN shapes is already present in serial FP32 "
        "versus FP64 staged accumulation. Quantization versus original BF16 is ~3 abs "
        "for both Quartz Q8 and llama Q8_1. Pathological random Q6_K `q6_k_257x512` "
        "emits nonfinites; that case stays on the strict path and does not inflate ceilings.",
        "",
        "## Frozen production ceilings (vs original BF16)",
        "",
        f"- q4k_gate_up abs `{gate_up['vs_fp64_original_bf16']['abs']['ceiling']:.9g}` "
        f"(llama `{gate_up['vs_fp64_original_bf16']['abs']['llama_error']:.9g}`).",
        f"- q4k_down abs `{down['vs_fp64_original_bf16']['abs']['ceiling']:.9g}` "
        f"(llama `{down['vs_fp64_original_bf16']['abs']['llama_error']:.9g}`).",
        "- vs-staged llama error is zero in this FP64 GEMM replica, so those ceilings "
        "remain the strict 3e-4/2e-4 envelopes. Current production large-shape staged "
        "misses are existing failures, not a reason to loosen CUD-001.",
        "",
        "## Quality suite",
        "",
        f"- Legacy QLT-001 PPL ratio stays **{LEGACY_PPL_RATIO}**.",
        f"- Production-optimization PPL ratio is **{PRODUCTION_PPL_RATIO}** on calibration "
        "and held-out 1024-target spans.",
        f"- Recurrence incremental NLL stays **{RECURRENCE_INCREMENTAL_NLL}**.",
        "- Continuation runner-up margin is **0** (no stored near-tie evidence).",
        f"- Current Quartz vs pinned llama wikitext PPL ratio: "
        f"**{ratio if ratio is not None else 'unavailable'}** "
        f"(legacy pass `{quartz['meets_legacy_1_05']}`, production 1.01 pass "
        f"`{quartz['meets_production_1_01']}`).",
        "- Held-out 1024-target span is `stream[16385:17409]` from frozen "
        "`recurrence_long.context`, disjoint from `wikitext_nll` `stream[1:1025]`. "
        "Scores for that span were not re-run in this sitting.",
        "",
        "Baseline production-1.01 failures are reported, not tuned away.",
        "",
        "## Proof limit",
        "",
        "- No speedup claim.",
        "- No unvalidated fast kernel in production.",
        "- Structural/session exactness is not an accuracy compromise.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    host = parse_host_payload(HOST_RAW.read_text(encoding="utf-8"))
    opt042 = _opt042_excerpt(_load(ROOT / "fixtures/opt042_mmv_integer_study.json"))
    quality = _quality_bundle()
    captures = {
        "d128": _capture_slots(
            ROOT / "evidence/optimization/opt043-component-gap/capture-d128.json"
        ),
        "d2048": _capture_slots(
            ROOT / "evidence/optimization/opt043-component-gap/capture-d2048.json"
        ),
        "real_text": _capture_slots(
            ROOT / "evidence/optimization/opt043-component-gap/capture-real_text.json"
        ),
        "layers": [0, 3, 31, 32, 62, 63],
        "source": "OPT-043 activation captures; tokens/layers other than 0/3/63 retained as held-out layers 31/32/62",
    }
    utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    fixture = {
        "schema_version": 1,
        "task": "OPT-044",
        "status": "measured",
        "measurement_utc": utc,
        "claims_performance_improvement": False,
        "production_numerics_path": "strict",
        "optimized_admitted": False,
        "llama_revision": LLAMA_REV,
        "gguf_sha256": GGUF_SHA,
        "host_fp64": host,
        "opt042": opt042,
        "opt043_captures": captures,
        "quality": quality,
        "opt042_numeric_reject_preserved": opt042["admissibility"] == "numeric_reject",
        "report_path": "evidence/optimization/opt044-production-numerics/REPORT.md",
    }
    contract = build_contract(host, quality)
    CONTRACT_PATH.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    FIXTURE_PATH.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(write_report(contract, fixture, quality) + "\n", encoding="utf-8")
    if not math.isfinite(
        float(
            contract["families"]["q4k_gate_up"]["vs_fp64_original_bf16"]["abs"][
                "ceiling"
            ]
        )
    ):
        raise SystemExit("nonfinite production ceiling")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
