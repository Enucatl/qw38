"""Freeze OPT-059 v2 numerics contract, admission manifest, and fixture."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.production_numerics import (
    DATASET_SHA,
    FORBIDDEN,
    GATE_INVENTORY,
    GGUF_SHA,
    LEGACY_PPL_RATIO,
    LLAMA_REV,
    PERMITTED_PRODUCTION,
    PRODUCTION_PPL_RATIO,
    RECURRENCE_INCREMENTAL_NLL,
    ROOT,
    STRICT_CUD001_ABS,
    TOKENIZER_SHA,
)
from tools.production_numerics_v2 import (
    STAGING_LLAMA_SUM_X,
    STAGING_Q8_FP32,
    STAGING_QUARTZ_SUM_Q,
    V2_SCHEMA,
    any_optimized_selected,
    engine_summary_path,
    family_v2_budgets,
    freeze_case_ids,
    opt046_independent_verdict,
    q4k_fp32_reductions,
    staged_rounding_guard,
    staging_semantics,
    teacher_forced_margin,
    validate_v2_contract,
)

CONTRACT_PATH = ROOT / "pins/production_numerics_v2_contract.json"
MANIFEST_PATH = ROOT / "pins/opt059_admission_manifest.json"
ITERATION_PATH = ROOT / "pins/opt059_iteration_contract.json"
FIXTURE_PATH = ROOT / "fixtures/opt059_gpu_numerics.json"
EVIDENCE = ROOT / "evidence/optimization/opt059-gpu-numerics"
REPORT = EVIDENCE / "REPORT.md"


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _admission_entries() -> list[dict[str, Any]]:
    return [
        {
            "id": "q4_k_k5120_packed_q8_fp32",
            "family": "q4_k",
            "columns": 5120,
            "rows_class": "gate_up_17408",
            "staging": STAGING_Q8_FP32,
            "variant": "packed_fp32",
            "production_dispatch": "strict",
            "currently_selected": True,
            "v2_admitted": False,
            "testing_admitted": True,
            "evidence": "cuda/q4k_decode_path.cuh kSelectedQ4DecodePath=packed",
            "reason": "current Q4 production is packed FP32; v2 GPU coverage pending or unadmitted for optimized integer",
        },
        {
            "id": "q4_k_k17408_packed_q8_fp32",
            "family": "q4_k",
            "columns": 17408,
            "rows_class": "down_5120",
            "staging": STAGING_Q8_FP32,
            "variant": "packed_fp32",
            "production_dispatch": "strict",
            "currently_selected": True,
            "v2_admitted": False,
            "testing_admitted": True,
            "evidence": "cuda/q4k_decode_path.cuh kSelectedQ4DecodePath=packed",
        },
        {
            "id": "q4_k_integer_q8",
            "family": "q4_k",
            "columns": 0,
            "rows_class": "*",
            "staging": STAGING_Q8_FP32,
            "variant": "integer_dp4a_q8",
            "production_dispatch": "strict",
            "currently_selected": False,
            "v2_admitted": False,
            "testing_admitted": True,
            "evidence": "fixtures/opt046_q4_decode.json",
            "reason": "OPT-046 candidate is testable, not installed",
        },
        {
            "id": "q4_k_integer_quartz_q8_1_sum_q",
            "family": "q4_k",
            "columns": 0,
            "rows_class": "*",
            "staging": STAGING_QUARTZ_SUM_Q,
            "variant": "integer_dp4a_q8_1",
            "production_dispatch": "strict",
            "currently_selected": False,
            "v2_admitted": False,
            "testing_admitted": True,
            "evidence": "cuda/q4k_decode_dots.cuh quantize_bf16_q8_1 stores half(sum(q))",
            "reason": "existing Q8_1 buffer is Quartz sum(q), not llama sum(x)",
        },
        {
            "id": "q4_k_integer_llama_q8_1_sum_x",
            "family": "q4_k",
            "columns": 0,
            "rows_class": "*",
            "staging": STAGING_LLAMA_SUM_X,
            "variant": "integer_dp4a_q8_1_sum_x",
            "production_dispatch": "strict",
            "currently_selected": False,
            "v2_admitted": False,
            "testing_admitted": True,
            "evidence": "launch_quantize_bf16_q8_1_sum_x",
            "reason": "true llama staging is a distinct variant; consumer change is a later candidate",
        },
        {
            "id": "q8_0_k5120_dp4a_q8_1",
            "family": "q8_0",
            "columns": 5120,
            "rows_class": "mixer_qkv",
            "staging": STAGING_QUARTZ_SUM_Q,
            "variant": "dp4a_q8_1",
            "production_dispatch": "optimized",
            "currently_selected": True,
            "v2_admitted": False,
            "testing_admitted": True,
            "evidence": "cuda/q8_decode_path.cuh kSelectedQ8DecodePath=dp4a_q8_1",
            "reason": "already selected by OPT-047; v2 GPU admission is recorded separately and does not silently revert",
        },
        {
            "id": "q8_0_k6144_dp4a_q8_1",
            "family": "q8_0",
            "columns": 6144,
            "rows_class": "mixer_output",
            "staging": STAGING_QUARTZ_SUM_Q,
            "variant": "dp4a_q8_1",
            "production_dispatch": "optimized",
            "currently_selected": True,
            "v2_admitted": False,
            "testing_admitted": True,
            "evidence": "cuda/q8_decode_path.cuh kSelectedQ8DecodePath=dp4a_q8_1",
        },
        {
            "id": "q8_0_direct_bf16",
            "family": "q8_0",
            "columns": 0,
            "rows_class": "*",
            "staging": "bf16_direct",
            "variant": "direct_bf16",
            "production_dispatch": "strict",
            "currently_selected": False,
            "v2_admitted": False,
            "testing_admitted": True,
            "evidence": "retained q8_mmv_bf16 strict reference",
        },
        {
            "id": "q6_k_k5120_integer_q8_1",
            "family": "q6_k",
            "columns": 5120,
            "rows_class": "vocab_248320",
            "staging": STAGING_QUARTZ_SUM_Q,
            "variant": "integer_q8_1",
            "production_dispatch": "optimized",
            "currently_selected": True,
            "v2_admitted": False,
            "testing_admitted": True,
            "evidence": "cuda/q6k_decode_path.cuh kSelectedQ6DecodePath=integer_q8_1",
            "reason": "already selected by OPT-048 for vocabulary rows; small probes stay packed",
        },
        {
            "id": "q6_k_packed_fp32",
            "family": "q6_k",
            "columns": 256,
            "rows_class": "small_probe",
            "staging": STAGING_Q8_FP32,
            "variant": "packed_fp32",
            "production_dispatch": "strict",
            "currently_selected": True,
            "v2_admitted": False,
            "testing_admitted": True,
            "evidence": "q6_decode_uses_integer_for_rows requires 248320 rows",
        },
    ]


def _families_from_v1() -> dict[str, Any]:
    v1 = _load(ROOT / "pins/production_numerics_contract.json")
    families: dict[str, Any] = {}
    for ident, row in v1["families"].items():
        llama = row["vs_fp64_original_bf16"]["abs"]["llama_error"]
        rms = row["vs_fp64_original_bf16"]["rms"]["llama_error"]
        cosine = row["vs_fp64_original_bf16"]["one_minus_cosine"]["llama_error"]
        nonfinite = int(row.get("nonfinite", 0))
        gpu = {
            "max_abs": None if nonfinite else None,
            "rms": None if nonfinite else None,
            "one_minus_cosine": None if nonfinite else None,
        }
        # GPU llama error is unknown until the authority export fills it.
        # Host-replica llama error is retained only as a v1 diagnostic.
        family = family_v2_budgets(
            gpu,
            family=row["family"],
            ident=ident,
            activation_source=row["activation_source"],
            nonfinite=nonfinite,
        )
        family["rows"] = row["rows"]
        family["columns"] = row["columns"]
        family["v1_1_05_diagnostic"] = {
            "abs": row["vs_fp64_original_bf16"]["abs"],
            "rms": row["vs_fp64_original_bf16"]["rms"],
            "one_minus_cosine": row["vs_fp64_original_bf16"]["one_minus_cosine"],
            "host_llama_error_not_gpu": {
                "max_abs": llama,
                "rms": rms,
                "one_minus_cosine": cosine,
            },
        }
        family["checksums"] = row["checksums"]
        family["production_dispatch"] = "strict"
        family["coverage"] = "unadmitted"
        families[ident] = family
    return families


def _small_probe_families() -> dict[str, Any]:
    # Q8 17x32 is new; Q4/Q6 17x256 reuse v1 IDs. Unadmitted until GPU export.
    return {
        "q8_0_17x32": family_v2_budgets(
            None,
            family="q8_0",
            ident="q8_0_17x32",
            activation_source="unit_normal",
            nonfinite=0,
        )
        | {
            "rows": 17,
            "columns": 32,
            "production_dispatch": "strict",
            "coverage": "unadmitted",
            "finite_fp16_scales_required": True,
        }
    }


def build_manifest() -> dict[str, Any]:
    entries = _admission_entries()
    manifest = {
        "schema": "qw38.opt059-admission-manifest",
        "schema_version": 1,
        "task": "OPT-059",
        "llama_revision": LLAMA_REV,
        "gguf_sha256": GGUF_SHA,
        "unrepresented_shapes": "strict",
        "engine_summary_path": engine_summary_path({"entries": entries}),
        "any_optimized_selected": any_optimized_selected({"entries": entries}),
        "opt046_installed": False,
        "entries": entries,
        "notes": [
            "Mapping is explicit per family/shape/staging/variant.",
            "currently_selected is the live pin; v2_admitted is GPU-calibrated.",
            "An admission record allows testing; production keep is a later task.",
            "Do not label the whole engine strict while Q8/Q6 optimized pins are selected.",
        ],
    }
    return manifest


def build_iteration() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "task": "OPT-059",
        "yardstick": "gpu_numerics_v2_admission",
        "claims_throughput": False,
        "image": "qw38-cuda:13.0.2",
        "model": "models/Qwen3.8-27B-Q4_K_M.gguf",
        "gpu_lock": "build/optimization-runs/qw38-gpu.lock",
        "aggregate_deadline_s": 300,
        "target": "build/qw38-cuda-opt059-numerics-test",
        "diagnostics_make_target": "cuda-opt059-diagnostics",
        "modes": {
            "feedback": {
                "target": "build/qw38-cuda-opt059-numerics-test",
                "tier_sequence": ["smoke", "q4"],
                "make_targets": ["build/qw38-cuda-opt059-numerics-test"],
                "build_once": True,
                "aggregate_deadline_s": 300,
                "workloads": ["smoke", "q4"],
            },
            "acceptance": {
                "target": "build/qw38-cuda-opt059-numerics-test",
                "tier_sequence": ["smoke", "correctness", "calibrate"],
                "make_targets": ["build/qw38-cuda-opt059-numerics-test"],
                "warm_repetitions": 0,
                "aggregate_deadline_s": 7200,
                "workloads": ["smoke", "correctness", "calibrate"],
            },
            "release": {
                "target": "build/qw38-cuda-opt059-numerics-test",
                "tier": "acceptance",
                "make_targets": ["build/qw38-cuda-opt059-numerics-test"],
                "historical_oracles": False,
                "workloads": [],
            },
        },
        "workloads": {
            "smoke": {
                "target": "build/qw38-cuda-opt059-numerics-test",
                "tier": "smoke",
                "kind": "small_random_two_refs",
                "cases": 3,
                "candidates": 1,
                "warmups": 0,
                "samples": 1,
                "tokens": 1,
                "execution_modes": 2,
                "control_candidate_pairs": 1,
                "load_model": 0,
                "reference": 1,
                "args": ["--workload", "smoke"],
            },
            "q4": {
                "target": "build/qw38-cuda-opt059-numerics-test",
                "tier": "correctness",
                "kind": "q4_family_feedback",
                "cases": 4,
                "candidates": 1,
                "warmups": 0,
                "samples": 1,
                "tokens": 1,
                "execution_modes": 1,
                "control_candidate_pairs": 1,
                "load_model": 0,
                "reference": 1,
                "fp64_dots": 64,
                "args": ["--workload", "q4"],
            },
            "correctness": {
                "target": "build/qw38-cuda-opt059-numerics-test",
                "tier": "correctness",
                "kind": "small_patterns_and_sampled_k",
                "cases": 12,
                "candidates": 1,
                "warmups": 0,
                "samples": 1,
                "tokens": 1,
                "execution_modes": 1,
                "control_candidate_pairs": 1,
                "load_model": 0,
                "reference": 1,
                "fp64_dots": 64,
                "args": ["--workload", "correctness"],
            },
            "calibrate": {
                "target": "build/qw38-cuda-opt059-numerics-test",
                "tier": "acceptance",
                "kind": "reference_only_gpu_calibration",
                "cases": 1,
                "candidates": 1,
                "warmups": 0,
                "samples": 1,
                "tokens": 1,
                "execution_modes": 1,
                "control_candidate_pairs": 1,
                "load_model": 1,
                "reference": 1,
                "args": [
                    "--workload",
                    "calibrate",
                    "{model}",
                    "--llama-export",
                    ".cache/authorities/llama-build/bin/qw38-llama-projection-export",
                ],
            },
        },
        "proof_limit": [
            "no throughput claim",
            "no P/D performance oracles",
            "no OPT-046 install",
            "v1 evidence preserved",
            "numeric pins frozen before candidates",
        ],
        "report_path": "evidence/optimization/opt059-gpu-numerics/REPORT.md",
    }


def build_contract(manifest: dict[str, Any]) -> dict[str, Any]:
    cases = freeze_case_ids()
    families = _families_from_v1()
    families.update(_small_probe_families())
    inventory = list(GATE_INVENTORY) + [
        {
            "gate": "OPT-059-v2",
            "class": "production-primitive-quality",
            "notes": (
                "v2 ceilings from actual llama GPU error with 1.25 headroom; "
                "explicit per-family admission; Q8_1 staging labeled honestly."
            ),
        }
    ]
    opt046 = _load(ROOT / "fixtures/opt046_q4_decode.json")
    # Missing llama GPU error keeps the candidate unadmitted under v2. Do not
    # substitute the OPT-044 host replica as GPU authority.
    verdict = opt046_independent_verdict(
        opt046, STRICT_CUD001_ABS, llama_gpu_error=None
    )
    return {
        "schema": V2_SCHEMA,
        "schema_version": 2,
        "task": "OPT-059",
        "claims_performance_improvement": False,
        "llama_revision": LLAMA_REV,
        "gguf_sha256": GGUF_SHA,
        "dataset_sha256": DATASET_SHA,
        "tokenizer_sha256": TOKENIZER_SHA,
        "v1_contract": "pins/production_numerics_contract.json",
        "admission_manifest": "pins/opt059_admission_manifest.json",
        "budget_rule": {
            "abs_rms": "max(strict_reference_ceiling, 1.25 * llama_gpu_error + 1e-6)",
            "one_minus_cosine": "max(1e-7, 1.25*(1-cos_llama)+1e-7)",
            "zero_vector": "explicit_norm_and_abs",
            "nonfinite_or_pathological_llama": "retain_strict_do_not_inflate",
            "v1_1_05_headroom": "diagnostic_only",
            "do_not_adjust_after_candidate_failure": True,
            "missing_coverage": "unadmitted",
        },
        "engine_summary_path": manifest["engine_summary_path"],
        "unrepresented_shapes": "strict",
        "any_optimized_selected": manifest["any_optimized_selected"],
        "opt046_installed": False,
        "opt046_independent_verdict": verdict,
        "case_ids": cases,
        "families": families,
        "staging": staging_semantics(),
        "rounding_guard": staged_rounding_guard(
            fp32_scale_reductions=q4k_fp32_reductions(5120),
            sum_abs_products=1.0,
        ),
        "teacher_forced": teacher_forced_margin(None),
        "quality": {
            "nll_ppl_ratio": PRODUCTION_PPL_RATIO,
            "legacy_nll_ppl_ratio": LEGACY_PPL_RATIO,
            "recurrence_incremental_nll": RECURRENCE_INCREMENTAL_NLL,
            "opt058_functional_required": True,
            "no_free_running_token_identity": True,
        },
        "gate_inventory": inventory,
        "permitted_production_approximations": list(PERMITTED_PRODUCTION),
        "forbidden": list(FORBIDDEN),
        "report_path": "evidence/optimization/opt059-gpu-numerics/REPORT.md",
    }


def build_fixture(contract: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    opt042 = _load(ROOT / "fixtures/opt042_mmv_integer_study.json")
    opt043 = _load(ROOT / "fixtures/opt043_component_gap.json")
    opt044 = _load(ROOT / "fixtures/opt044_production_numerics.json")
    return {
        "schema_version": 1,
        "task": "OPT-059",
        "status": "calibrated_policy",
        "measurement_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "claims_performance_improvement": False,
        "opt046_installed": False,
        "opt042_numeric_reject_preserved": opt042["admissibility"] == "numeric_reject",
        "llama_revision": LLAMA_REV,
        "gguf_sha256": GGUF_SHA,
        "engine_summary_path": manifest["engine_summary_path"],
        "any_optimized_selected": manifest["any_optimized_selected"],
        "v1_path_selection_preserved": opt044["production_numerics_path"] == "strict",
        "opt043_capture_layers": opt043.get("quartz_decode", {}).get("capture_layers")
        or [0, 3, 31, 32, 62, 63],
        "llama_gpu_sidecars": {
            "exporter": "tools/llama_authority/projection_export.cpp",
            "status": "probe_exported_production_k_unadmitted",
            "not_cpu_replica": True,
            "cuda_f32_to_q8_1_cpy": "unsupported_this_revision",
            "probe": _load(EVIDENCE / "llama-gpu-export" / "probe.json")
            if (EVIDENCE / "llama-gpu-export" / "probe.json").is_file()
            else None,
        },
        "case_id_hash": contract["case_ids"]["id_manifest_sha256"],
        "opt046_independent_verdict": contract["opt046_independent_verdict"],
        "report_path": str(REPORT.relative_to(ROOT)),
    }


def main() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest()
    contract = build_contract(manifest)
    fixture = build_fixture(contract, manifest)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    CONTRACT_PATH.write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    ITERATION_PATH.write_text(
        json.dumps(build_iteration(), indent=2) + "\n", encoding="utf-8"
    )
    FIXTURE_PATH.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
    validate_v2_contract(contract, fixture)
    print(f"wrote {CONTRACT_PATH}")
    print(f"wrote {MANIFEST_PATH}")
    print(f"wrote {ITERATION_PATH}")
    print(f"wrote {FIXTURE_PATH}")


if __name__ == "__main__":
    main()
