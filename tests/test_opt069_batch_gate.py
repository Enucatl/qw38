"""Host tests for the OPT-069 combined batch validation gate."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import pytest

from tools.opt069_batch_gate import (
    EXPECTED_PATHS,
    MARGIN,
    plus5_gap_ms,
    compute_gate,
    frozen_combined_config,
    load_contract,
    parity_gap_ms,
    source_paths,
    three_outcomes,
    validate_batch_result,
)
from tools.opt073_quality_policy import (
    QualityPolicyError,
    evaluate_preflight_quality,
    oracle_policy,
    validate_parsed_functional_answers,
)
from tools.run_optimization_task import (
    describe_plan,
    load_contract as load_iteration,
    loop_product,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "pins/opt069_batch_gate_contract.json"
ITERATION = ROOT / "pins/opt069_iteration_contract.json"
REPORT = ROOT / "evidence/optimization/opt069-batch-gate/REPORT.md"
OPT056 = ROOT / "fixtures/opt056_performance_gate.json"
MAKEFILE = ROOT / "Makefile"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _synthetic_measured() -> dict[str, Any]:
    freeze = frozen_combined_config()
    opt056 = _json(OPT056)
    p = copy.deepcopy(opt056["p"])
    p["quartz"]["attribution"] = None
    p["quartz"]["instrumented"] = False
    d128 = copy.deepcopy(opt056["d128"])
    d2048 = copy.deepcopy(opt056["d2048"])
    d128["quartz"]["attribution"] = None
    d2048["quartz"]["attribution"] = None
    p_q = float(p["quartz"]["mean_tok_s"])
    p_l = float(p["llama_cpp"]["avg_ts"])
    d128_q = float(d128["quartz"]["mean_tok_s"])
    d128_l = float(d128["llama_cpp"]["mean_tok_s"])
    d2048_q = float(d2048["quartz"]["mean_tok_s"])
    d2048_l = float(d2048["llama_cpp"]["mean_tok_s"])
    proof = " ".join(load_contract()["proof_limit"])
    result: dict[str, Any] = {
        "schema_version": 1,
        "task": "OPT-069",
        "status": "measured",
        "measurement_utc": "2026-09-11T12:00:00Z",
        "device": "NVIDIA GeForce RTX 5090",
        "compute_capability": "12.0",
        "power_limit_w": 400.0,
        "telemetry": {
            "device": "NVIDIA GeForce RTX 5090",
            "device_substring": "RTX 5090",
            "power_limit_w": 400.0,
        },
        "llama_revision": opt056["llama_revision"],
        "gguf_sha256": opt056["gguf_sha256"],
        "source_revision": "synthetic",
        "source_state": "dirty",
        "hardware_executed": True,
        "keep_sitting_skipped": False,
        "combined_production_paths": freeze["combined_production_paths"],
        "candidate_decisions": freeze["candidate_decisions"],
        "keeps": freeze["keeps"],
        "rejected_or_retained": freeze["rejected_or_retained"],
        "batch_size": freeze["batch_size"],
        "workspace_bytes": freeze["workspace_bytes"],
        "p": p,
        "d128": d128,
        "d2048": d2048,
        "opt016": copy.deepcopy(opt056["opt016"]),
        "quality": {
            "quality_v2": {
                "suite": "quality-v2",
                "status": "fail",
                "all": False,
                "wikitext_nll": {"pass": True, "mean_nll": 1.5252005926497396},
                "held_out_wikitext_1024": {
                    "pass": True,
                    "mean_nll": 1.7878782710057632,
                },
                "recurrence": {"pass": True, "incremental_nll": -0.00831433},
                "tasks": {"pass": False, "count": 8},
                "llama_reference": {
                    "wikitext_nll": {"mean_nll": 1.52508},
                    "held_out_wikitext_1024": {"mean_nll": 1.78826},
                },
            },
            "legacy": {
                "opt056_tasks_pass": False,
                "historical_only": True,
            },
            "qlt001_not_claimed": True,
        },
        "gaps": {
            "p": {
                "parity_ms": parity_gap_ms(p_q, p_l, 4096),
                "plus5_ms": plus5_gap_ms(p_q, p_l, 4096),
            },
            "d128": {
                "parity_ms": parity_gap_ms(d128_q, d128_l, 256),
                "plus5_ms": plus5_gap_ms(d128_q, d128_l, 256),
            },
            "d2048": {
                "parity_ms": parity_gap_ms(d2048_q, d2048_l, 256),
                "plus5_ms": plus5_gap_ms(d2048_q, d2048_l, 256),
            },
        },
        "state_isolation": copy.deepcopy(opt056["state_isolation"]),
        "owns_opt016_parity_gate": False,
        "opt056_gate_passed": False,
        "opt016_gate_passed": False,
        "preflight_is_release_evidence": False,
        "nsight_systems": "not_used",
        "nsight_compute": "not_used",
        "proof_limit": proof,
        "report_path": "evidence/optimization/opt069-batch-gate/REPORT.md",
    }
    result["opt016"]["gate_passed"] = False
    result["gate"] = compute_gate(result)
    result["outcomes"] = three_outcomes(result)
    return result


def test_contracts_iteration_plan_and_freeze() -> None:
    contract = load_contract()
    iteration = load_iteration("OPT-069")
    freeze = frozen_combined_config()
    paths = source_paths()
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert REPORT.is_file()
    assert contract["task"] == "OPT-069"
    assert contract["relaxes_opt056"] is False
    assert contract["relaxes_opt016"] is False
    assert contract["preflight_is_release_evidence"] is False
    assert contract["keeps"] == ["OPT-064 r2_w2", "OPT-066 fma_async_x"]
    assert iteration["target"] == "build/qw38-cuda-optimization-engine-probe"
    assert iteration["diagnostics_make_target"] == "cuda-opt069-diagnostics"
    assert iteration["modes"]["feedback"]["tier_sequence"] == ["preflight"]
    assert loop_product(iteration["workloads"]["preflight"]) == 4
    assert loop_product(iteration["workloads"]["d2048"]) == 33 * 256
    makefile = MAKEFILE.read_text(encoding="utf-8")
    assert "cuda-opt069-diagnostics" in makefile
    assert "qw38-cuda-prefill-2k-parity-test" in makefile
    assert paths["q8_layout"] == "r2_w2"
    assert paths["mmq_async_x"] is True
    assert paths["q4_decode"] == "packed"
    assert paths["ffn_prompt_pair"] == "off"
    assert paths["nvccflags"] == "-O2 --fmad=false"
    for key, value in EXPECTED_PATHS.items():
        assert freeze["combined_production_paths"][key] == value
    assert freeze["candidate_decisions"]["OPT-064"]["installed"] is True
    assert freeze["candidate_decisions"]["OPT-066"]["installed"] is True
    assert freeze["candidate_decisions"]["OPT-067"]["installed"] is False
    assert freeze["candidate_decisions"]["OPT-068"]["installed"] is False
    plan = describe_plan("OPT-069", "release", iteration, None)
    assert "engines=" in plan
    assert "d128:" in plan
    assert "warmups=3" in plan
    assert "replicates=30" in plan
    assert "output_tokens=256" in plan
    assert "historical_oracles=" in plan
    preflight = describe_plan("OPT-069", "feedback", iteration, "preflight")
    assert "phase=preflight" in preflight
    assert "historical_oracles=none" in preflight
    assert "preflight:" in preflight


def test_parity_and_plus5_millisecond_arithmetic() -> None:
    quartz = 2808.49609
    llama = 3263.516321
    parity = parity_gap_ms(quartz, llama, 4096)
    plus5 = plus5_gap_ms(quartz, llama, 4096)
    tq = 4096 * 1000.0 / quartz
    tl = 4096 * 1000.0 / llama
    assert parity == pytest.approx(tq - tl)
    assert plus5 == pytest.approx(tq - tl / MARGIN)
    assert parity != pytest.approx(plus5)
    fake = _synthetic_measured()
    fake["gaps"]["p"]["parity_ms"] = fake["gaps"]["p"]["plus5_ms"]
    with pytest.raises(AssertionError, match="parity gap labeled|wrong parity"):
        validate_batch_result(fake)
    fake = _synthetic_measured()
    fake["gaps"]["p"]["plus5_ms"] = fake["gaps"]["p"]["parity_ms"]
    with pytest.raises(AssertionError, match=r"\+5%|parity gap labeled"):
        validate_batch_result(fake)


def test_rejects_missing_or_nan_quality_references() -> None:
    missing = _synthetic_measured()
    del missing["quality"]["quality_v2"]["held_out_wikitext_1024"]["mean_nll"]
    with pytest.raises(AssertionError, match="missing quality"):
        validate_batch_result(missing)
    nan = _synthetic_measured()
    nan["quality"]["quality_v2"]["llama_reference"]["held_out_wikitext_1024"][
        "mean_nll"
    ] = math.nan
    with pytest.raises(AssertionError, match="NaN|nonfinite"):
        validate_batch_result(nan)
    omitted = _synthetic_measured()
    omitted["quality"]["quality_v2"]["llama_reference"]["held_out_wikitext_1024"][
        "mean_nll"
    ] = "missing"
    with pytest.raises(AssertionError, match="missing quality"):
        validate_batch_result(omitted)


def test_rejects_mismatched_power_selector_and_tokens() -> None:
    power = _synthetic_measured()
    power["power_limit_w"] = 0
    with pytest.raises(AssertionError, match="power"):
        validate_batch_result(power)
    selector = _synthetic_measured()
    selector["combined_production_paths"]["q8_layout"] = "r1_w4"
    with pytest.raises(AssertionError, match="selector|freeze|combined"):
        validate_batch_result(selector)
    tokens = _synthetic_measured()
    tokens["p"]["quartz"]["prompt_tokens"] = 2048
    with pytest.raises(AssertionError, match="token"):
        validate_batch_result(tokens)
    prefix = _synthetic_measured()
    prefix["d2048"]["quartz"]["prefix"] = 1024
    with pytest.raises(AssertionError, match="prefix|token"):
        validate_batch_result(prefix)


def test_rejects_short_samples_and_hidden_instrumentation() -> None:
    short_p = _synthetic_measured()
    short_p["p"]["quartz"]["tok_s"] = short_p["p"]["quartz"]["tok_s"][:1]
    short_p["p"]["quartz"]["replicates"] = 1
    with pytest.raises(AssertionError, match="short samples"):
        validate_batch_result(short_p)
    short_d = _synthetic_measured()
    short_d["d2048"]["quartz"]["tok_s"] = short_d["d2048"]["quartz"]["tok_s"][:5]
    short_d["d2048"]["quartz"]["runs"] = 5
    with pytest.raises(AssertionError, match="short samples"):
        validate_batch_result(short_d)
    instrumented = _synthetic_measured()
    instrumented["p"]["quartz"]["attribution"] = {"ffn_mmq": 1.0}
    instrumented["p"]["quartz"]["instrumented"] = True
    with pytest.raises(AssertionError, match="instrumented"):
        validate_batch_result(instrumented)


def test_rejects_rejected_candidate_leakage() -> None:
    leaked = _synthetic_measured()
    leaked["candidate_decisions"]["OPT-067"]["installed"] = True
    with pytest.raises(AssertionError, match="leakage"):
        validate_batch_result(leaked)
    q4 = _synthetic_measured()
    q4["combined_production_paths"]["q4_decode"] = "integer_q8_1"
    with pytest.raises(AssertionError, match="leak|freeze|selector|combined"):
        validate_batch_result(q4)
    flags = _synthetic_measured()
    flags["combined_production_paths"]["nvccflags"] = "-O3 --fmad=true"
    with pytest.raises(AssertionError, match="leak|freeze|selector|combined|O3"):
        validate_batch_result(flags)


def test_does_not_mark_opt056_or_opt016_passed_on_failure() -> None:
    fake = _synthetic_measured()
    assert fake["gate"]["passed"] is False
    assert fake["outcomes"]["opt056_plus5"]["pass"] is False
    assert fake["outcomes"]["llama_parity"]["pass"] is False
    fake["opt056_gate_passed"] = True
    with pytest.raises(AssertionError, match="OPT-056"):
        validate_batch_result(fake)
    fake = _synthetic_measured()
    fake["opt016"]["gate_passed"] = True
    fake["opt016_gate_passed"] = True
    with pytest.raises(AssertionError, match="OPT-016"):
        validate_batch_result(fake)
    honest = _synthetic_measured()
    validate_batch_result(honest)
    assert honest["opt056_gate_passed"] is False
    assert honest["opt016"]["gate_passed"] is False


def test_preflight_requires_parsed_answers_not_token_count() -> None:
    inputs = json.loads(
        (ROOT / "pins/production_quality_v2_inputs.json").read_text(encoding="utf-8")
    )
    with pytest.raises(QualityPolicyError, match="missing"):
        validate_parsed_functional_answers(
            [{"case": "task_arithmetic", "tokens": [32]}],
            inputs,
        )
    functional = json.loads(
        (
            ROOT / "evidence/optimization/opt069-batch-gate/preflight-functional.json"
        ).read_text(encoding="utf-8")
    )
    held = json.loads(
        (
            ROOT / "evidence/optimization/opt069-batch-gate/preflight-held-out-32.json"
        ).read_text(encoding="utf-8")
    )
    evaluated = evaluate_preflight_quality(functional, held, inputs)
    assert evaluated["selected_quality_verdicts"]["parsed_functional_answers"] is True
    assert evaluated["opt056_quality_requirement_met"] is False
    assert evaluated["status"] == "quality_blocked"
    assert evaluated["is_release_evidence"] is False
    arithmetic = next(
        row for row in evaluated["parsed"] if row["case"] == "task_arithmetic"
    )
    assert arithmetic["actual"] == "A"
    assert arithmetic["expected"] == "B"


def test_release_stops_before_oracles_unless_diagnostic() -> None:
    stop = oracle_policy(False, diagnostic_performance=False)
    assert stop["run_oracles"] is False
    assert stop["release_eligible"] is False
    assert stop["keep_claims_allowed"] is False
    diagnostic = oracle_policy(False, diagnostic_performance=True)
    assert diagnostic["run_oracles"] is True
    assert diagnostic["release_eligible"] is False
    assert diagnostic["keep_claims_allowed"] is False
    assert diagnostic["retain_quality_failure"] is True
    passing = oracle_policy(True, diagnostic_performance=False)
    assert passing["run_oracles"] is True
    assert passing["release_eligible"] is True
    skipped = oracle_policy(None, diagnostic_performance=False)
    assert skipped["run_oracles"] is False
    assert skipped["stop_reason"] == "skipped quality phase"
