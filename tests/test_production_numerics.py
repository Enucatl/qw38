from __future__ import annotations

import copy
import json
import math
import subprocess
from typing import Any

import pytest

from tools.production_numerics import (
    COSINE_FLOOR,
    GATE_INVENTORY,
    GGUF_SHA,
    LEGACY_PPL_RATIO,
    LLAMA_REV,
    PRODUCTION_PPL_RATIO,
    RECURRENCE_INCREMENTAL_NLL,
    ROOT,
    STRICT_CUD001_ABS,
    STRICT_CUD001_RMS,
    family_budgets,
    parse_host_payload,
    production_ceiling,
    validate_contract,
)
from tools.qw38_quality import PRODUCTION_OPTIMIZATION_CASES

CONTRACT = ROOT / "pins/production_numerics_contract.json"
FIXTURE = ROOT / "fixtures/opt044_production_numerics.json"
QUALITY = ROOT / "pins/quality_contract.json"
OPT042 = ROOT / "fixtures/opt042_mmv_integer_study.json"
REPORT = ROOT / "evidence/optimization/opt044-production-numerics/REPORT.md"
HOST_RAW = ROOT / "evidence/optimization/opt044-production-numerics/host-fp64-raw.txt"
DIAGNOSTIC = ROOT / "build/qw38-opt044-production-numerics"
MMV = ROOT / "cuda/quant_mmv.cu"
HEADER = ROOT / "cuda/production_numerics.h"


def _contract() -> dict[str, Any]:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def _fixture() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_contract_fixture_and_report_are_connected() -> None:
    contract = _contract()
    fixture = _fixture()
    assert CONTRACT.is_file() and FIXTURE.is_file() and REPORT.is_file()
    assert HOST_RAW.is_file()
    assert set(fixture) == set(contract["required_fixture_keys"])
    assert contract["schema"] == "qw38.production-numerics-contract"
    assert contract["task"] == fixture["task"] == "OPT-044"
    assert contract["claims_performance_improvement"] is False
    assert fixture["claims_performance_improvement"] is False
    assert contract["path_selection"]["selected"] == "strict"
    assert contract["path_selection"]["optimized_admitted"] is False
    assert fixture["production_numerics_path"] == "strict"
    assert fixture["optimized_admitted"] is False
    assert contract["llama_revision"] == LLAMA_REV
    assert contract["gguf_sha256"] == GGUF_SHA
    validate_contract(contract, fixture)
    text = REPORT.read_text(encoding="utf-8")
    assert (
        "claims_performance_improvement" in text.casefold()
        or "no speedup" in text.casefold()
    )
    assert "strict" in text.casefold()
    assert "1.01" in text


def test_legacy_quality_contract_is_not_overwritten() -> None:
    legacy = json.loads(QUALITY.read_text(encoding="utf-8"))
    contract = _contract()
    assert legacy["thresholds"]["nll_ppl_ratio"] == LEGACY_PPL_RATIO
    assert (
        legacy["thresholds"]["recurrence_incremental_nll"] == RECURRENCE_INCREMENTAL_NLL
    )
    assert contract["quality_suite"]["legacy_nll_ppl_ratio"] == LEGACY_PPL_RATIO
    assert contract["quality_suite"]["nll_ppl_ratio"] == PRODUCTION_PPL_RATIO
    assert contract["quality_suite"]["overwrite_legacy_verdicts"] is False
    assert "near_tie" not in json.dumps(legacy)
    assert set(PRODUCTION_OPTIMIZATION_CASES).issubset(set(legacy["tasks"]))


def test_held_out_span_is_disjoint_and_hashed() -> None:
    contract = _contract()
    fixture = _fixture()
    cal = contract["quality_suite"]["calibration_wikitext_1024"]
    held = contract["quality_suite"]["held_out_wikitext_1024"]
    assert cal["target_count"] == held["target_count"] == 1024
    assert held["disjoint_from_calibration"] is True
    assert held["continuation_sha256"] != cal["continuation_sha256"]
    assert int(held["stream_end"]) <= int(cal["stream_start"]) or int(
        held["stream_start"]
    ) >= int(cal["stream_end"])
    live = fixture["quality"]["held_out_wikitext_1024"]
    assert live["continuation_sha256"] == held["continuation_sha256"]
    assert len(live["continuation"]) == 1024
    opt043 = fixture["opt043_captures"]
    assert opt043["layers"] == [0, 3, 31, 32, 62, 63]
    assert {slot["layer"] for slot in opt043["real_text"]["slots"]} >= {
        0,
        3,
        31,
        32,
        62,
        63,
    }


def test_opt042_numeric_reject_and_checksums_are_preserved() -> None:
    fixture = _fixture()
    opt042 = json.loads(OPT042.read_text(encoding="utf-8"))
    assert fixture["opt042_numeric_reject_preserved"] is True
    assert opt042["admissibility"] == "numeric_reject"
    assert opt042["selected_mmv_load_path"] == "packed"
    host = {case["id"]: case for case in fixture["host_fp64"]["cases"]}
    for ident in ("q4_k_17x256", "q4k_gate_up"):
        assert (
            host[ident]["checksums"]["weights"]
            == opt042["synthetic"][ident]["checksums"]["weights"]
        )
        assert (
            host[ident]["checksums"]["activation"]
            == opt042["synthetic"][ident]["checksums"]["activation"]
        )
    packed = opt042["synthetic"]["q4k_gate_up"]["candidates"]["packed"]
    assert packed["cud001_eligible"] is False
    assert float(packed["vs_host_cud001"]["max_abs"]) > STRICT_CUD001_ABS


def test_zero_vector_uses_norm_abs_and_nonfinites_stay_strict() -> None:
    contract = _contract()
    zero = contract["families"]["q4_k_17x256_zero"]
    policy = zero["vs_fp64_original_bf16"]["zero_vector"]
    assert policy["rule"] == "explicit_norm_and_abs"
    assert policy["cosine_not_used"] is True
    assert policy["abs_ok"] is True
    bad = contract["families"]["q6_k_257x512"]
    assert bad["pathological_llama"] is True
    assert bad["vs_fp64_original_bf16"]["abs"]["ceiling"] == STRICT_CUD001_ABS
    assert bad["vs_fp64_original_bf16"]["abs"]["path"] == "strict"
    assert float(bad["vs_fp64_original_bf16"]["abs"]["llama_error"]) > 1.0


def test_budget_formula_and_malformed_mutations_are_rejected() -> None:
    contract = _contract()
    fixture = _fixture()
    llama = contract["families"]["q4k_gate_up"]["vs_fp64_original_bf16"]["abs"][
        "llama_error"
    ]
    expected = production_ceiling(STRICT_CUD001_ABS, llama)
    assert contract["families"]["q4k_gate_up"]["vs_fp64_original_bf16"]["abs"][
        "ceiling"
    ] == pytest.approx(expected["ceiling"], rel=0.0, abs=1e-18)
    cosine = contract["families"]["q4_k_17x256"]["vs_fp64_original_bf16"][
        "one_minus_cosine"
    ]
    assert cosine["ceiling"] >= COSINE_FLOOR
    mutated = copy.deepcopy(contract)
    mutated["families"]["q4k_gate_up"]["vs_fp64_original_bf16"]["abs"]["ceiling"] *= 2
    with pytest.raises(ValueError, match="budget mutated"):
        validate_contract(mutated, fixture)
    nonfinite = copy.deepcopy(contract)
    nonfinite["families"]["q6_k_257x512"]["vs_fp64_original_bf16"]["abs"]["path"] = (
        "optimized_eligible"
    )
    nonfinite["families"]["q6_k_257x512"]["vs_fp64_original_bf16"]["abs"]["ceiling"] = (
        1.0e9
    )
    with pytest.raises(ValueError):
        validate_contract(nonfinite, fixture)
    fast = copy.deepcopy(contract)
    fast["path_selection"]["selected"] = "optimized"
    with pytest.raises(ValueError, match="unvalidated"):
        validate_contract(fast, fixture)
    speedup = copy.deepcopy(contract)
    speedup["claims_performance_improvement"] = True
    with pytest.raises(ValueError, match="speedup"):
        validate_contract(speedup, fixture)


def test_gate_inventory_and_strict_references_remain() -> None:
    contract = _contract()
    classes = {row["gate"]: row["class"] for row in contract["gate_inventory"]}
    assert classes["CUD-001"] == "retained-reference-arithmetic"
    assert classes["OPT-003-graph"] == "structural-exactness"
    assert classes["SES-001-prefix"] == "structural-exactness"
    assert classes["QLT-001"] == "end-to-end-quality"
    assert classes["OPT-044-primitive"] == "production-primitive-quality"
    assert {row["gate"] for row in GATE_INVENTORY} == set(classes)
    quant = json.loads((ROOT / "pins/cuda_quant_contract.json").read_text())
    assert quant["admission"]["maximum_absolute_error"] == STRICT_CUD001_ABS
    assert quant["admission"]["maximum_rms_error"] == STRICT_CUD001_RMS
    quality = json.loads(QUALITY.read_text())
    assert quality["thresholds"]["nll_ppl_ratio"] == LEGACY_PPL_RATIO
    mmv = MMV.read_text()
    assert 'kSelectedProductionNumericsPath[] = "strict"' in mmv
    assert "kProductionAdmission[]" in mmv
    assert (
        "production_numerics_optimized_admitted() noexcept { return false; }" not in mmv
    )
    assert "unrepresented_production_numerics_path" in mmv
    header = HEADER.read_text()
    assert "kLegalProductionNumericsPathOptimized" in header
    handbook = (ROOT / "docs/04-numerics.md").read_text().casefold()
    assert "strict reference versus optimized production" in handbook
    assert "1.01" in handbook
    makefile = (ROOT / "Makefile").read_text()
    assert "--fmad=false" in makefile
    assert "-ffp-contract=off" in makefile
    assert "use_fast_math" not in makefile


def test_quality_baseline_is_computed_independently() -> None:
    fixture = _fixture()
    quality = fixture["quality"]
    quartz = float(quality["current_quartz"]["wikitext_mean_nll"])
    llama = float(quality["pinned_llama"]["wikitext_mean_nll"])
    ratio = math.exp(quartz - llama)
    assert quality["current_quartz"]["ppl_ratio_vs_llama"] == pytest.approx(
        ratio, rel=1e-12, abs=0.0
    )
    assert quality["current_quartz"]["meets_production_1_01"] is (
        ratio <= PRODUCTION_PPL_RATIO
    )
    assert quality["continuation_top_two_margin"] == 0.0
    drift = quality["current_quartz"]["recurrence_incremental_nll"]
    assert drift <= RECURRENCE_INCREMENTAL_NLL
    teacher = quality["teacher_forced"]
    assert teacher["positions"] == 1024
    assert 0.0 <= teacher["top1_agreement"] <= 1.0


def test_host_diagnostic_reproduces_frozen_checksums() -> None:
    if not DIAGNOSTIC.is_file():
        build = subprocess.run(
            ["make", "build/qw38-opt044-production-numerics"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert build.returncode == 0, build.stdout + build.stderr
    run = subprocess.run(
        [str(DIAGNOSTIC)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert run.returncode == 0, run.stderr
    live = parse_host_payload(run.stdout)
    frozen = _fixture()["host_fp64"]
    live_cases = {case["id"]: case for case in live["cases"]}
    frozen_cases = {case["id"]: case for case in frozen["cases"]}
    assert set(live_cases) == set(frozen_cases)
    for ident, case in frozen_cases.items():
        assert live_cases[ident]["checksums"] == case["checksums"]
        for name, metrics in case["comparisons"].items():
            got = live_cases[ident]["comparisons"][name]
            assert got["nonfinite"] == metrics["nonfinite"]
            assert got["max_abs"] == pytest.approx(
                metrics["max_abs"], rel=1e-9, abs=1e-12
            )
    recomputed = family_budgets(live)
    contract = _contract()
    for ident, family in recomputed.items():
        left = contract["families"][ident]["vs_fp64_original_bf16"]["abs"]["ceiling"]
        right = family["vs_fp64_original_bf16"]["abs"]["ceiling"]
        assert left == pytest.approx(right, rel=0.0, abs=1e-18)
