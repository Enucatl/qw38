"""Host tests for OPT-116 generated-text quality freeze. GPU-free."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tools.opt116_generated_quality import (
    CONTRACT,
    ITERATION,
    PROOF,
    QUALITY_CONTRACT_ID,
    STRICT_CONTRACT_ID,
    load_json,
    run_phase,
)
from tools.quality.approximate import (
    evaluate_kernel_reference,
    evaluate_same_path_exact,
)
from tools.quality.cache_path import (
    CACHE_DECODE_PATH,
    FORBIDDEN_PATH,
    refuse_fabricated_aggregate,
    require_cache_path,
)
from tools.quality.errors import QualityFrameworkError
from tools.quality.held_out import (
    CASE_CLASSES,
    MIN_CASES,
    MIN_PER_CLASS,
    SAMPLE_SUBSET,
    admission_cases,
    calibration_cases,
    sample_subset_ids,
    validate_generation,
)
from tools.quality.identity import (
    GRAPH_PATH,
    case_plan,
    encoding_identity,
    require_contract_encodings,
    scoring_identity,
)
from tools.quality.quality_mode import QUALITY_FLAG, QUALITY_SHORTCUTS
from tools.quality.remote import API_KEY_ENV
from tools.quality.suite import (
    QUALITY_CONTRACT_SPECS,
    SUITE_CLASSES,
    evaluate_quality_contracts,
    quality_contract_spec,
)
from tools.run_optimization_task import describe_plan, load_contract, loop_product

ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = ROOT / "Makefile"
LEDGER = ROOT / "implementation_ledger.md"


def test_contracts_iteration_and_native_hooks() -> None:
    contract = load_json(CONTRACT)
    iteration = load_contract("OPT-116")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    assert contract["task"] == "OPT-116"
    assert contract["claims_throughput"] is False
    assert contract["candidate_installed"] is False
    assert contract["production_kernel_changed"] is False
    assert contract["quality_contract_id"] == QUALITY_CONTRACT_ID
    assert contract["strict_quality_contract_id"] == STRICT_CONTRACT_ID
    assert contract["frozen_acceptance"]["ppl_ratio_max"] == 1.01
    assert contract["frozen_acceptance"]["recurrence_incremental_nll_max"] == 0.02
    assert contract["frozen_acceptance"]["opt091_late_w4_exception_does_not_apply"] is (
        True
    )
    assert contract["authority"]["date"] == "2026-09-13"
    assert contract["authority"]["source"] == "user_request"
    assert contract["quality_mode"]["flag"] == QUALITY_FLAG
    assert contract["quality_mode"]["disables"] == list(QUALITY_SHORTCUTS)
    assert contract["post113_selectors"]["q4_decode"] == "llama_q4k_mmvq"
    assert contract["encodings"]["graph_path"] == GRAPH_PATH
    assert contract["cache_reading"]["path"] == CACHE_DECODE_PATH
    assert contract["cache_reading"]["forbidden_path"] == FORBIDDEN_PATH
    assert contract["cache_reading"]["prefixes"] == [8192, 32768, 131040]
    assert contract["cache_reading"]["capacity"] == 131072
    assert contract["gdn_stress"]["decode_updates"] == 2048
    assert contract["held_out_free_running"]["targets"] == 32
    assert contract["opt056_remains_blocked"] is True
    assert contract["opt016_remains_blocked"] is True
    assert contract["suite_classes"] == list(SUITE_CLASSES)
    assert iteration["claims_throughput"] is False
    assert iteration["candidate_installed"] is False
    assert iteration["skip_compile"] is True
    assert iteration["case_ids"] == [
        "authenticate",
        "generated",
        "cache-gdn",
        "baseline",
    ]
    assert iteration["modes"]["release"]["historical_oracles"] is False
    assert "cuda-opt116-diagnostics" in makefile
    assert "qw38-cuda-opt116-generated-quality-test" in makefile
    assert "qw38-cuda-opt058-quality-baseline-test" in makefile
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    spec = quality_contract_spec(QUALITY_CONTRACT_ID)
    assert spec["allow_new_greedy_mismatch"] is True
    assert spec["ppl_ratio_max"] == 1.01
    assert QUALITY_CONTRACT_ID in QUALITY_CONTRACT_SPECS


def test_successor_allows_token_identity_not_ppl_bounds() -> None:
    ratios = {"post113": 1.005, "opt084_frozen": 1.005}
    greedy = evaluate_quality_contracts(
        ratios=ratios,
        recurrence_incremental_nll=0.0,
        greedy_mismatch=True,
        contract_ids=[STRICT_CONTRACT_ID, QUALITY_CONTRACT_ID],
    )
    assert greedy["contracts"][STRICT_CONTRACT_ID]["model_quality_pass"] is False
    assert greedy["contracts"][QUALITY_CONTRACT_ID]["model_quality_pass"] is True
    wide = evaluate_quality_contracts(
        ratios={"post113": 1.012, "opt084_frozen": 1.012},
        recurrence_incremental_nll=0.0,
        contract_ids=[STRICT_CONTRACT_ID, QUALITY_CONTRACT_ID],
    )
    assert wide["contracts"][STRICT_CONTRACT_ID]["model_quality_pass"] is False
    assert wide["contracts"][QUALITY_CONTRACT_ID]["model_quality_pass"] is False
    rec = evaluate_quality_contracts(
        ratios=ratios,
        recurrence_incremental_nll=0.021,
        contract_ids=[QUALITY_CONTRACT_ID],
    )
    assert rec["contracts"][QUALITY_CONTRACT_ID]["model_quality_pass"] is False


def test_held_out_coverage_and_calibration_separation() -> None:
    cases = admission_cases()
    calibration = calibration_cases()
    assert len(cases) >= MIN_CASES
    counts = {name: 0 for name in CASE_CLASSES}
    for case in cases:
        counts[str(case["class"])] += 1
        assert case["role"] == "admission"
        assert case["teacher_forced_answers"] is False
        assert case["consumes_own_sampled_tokens"] is True
    assert all(count >= MIN_PER_CLASS for count in counts.values())
    subset = sample_subset_ids(cases)
    assert len(subset) >= SAMPLE_SUBSET
    cal_ids = {row["id"] for row in calibration}
    adm_ids = {row["id"] for row in cases}
    assert not (cal_ids & adm_ids)
    assert all(row["role"] == "calibration" for row in calibration)


def test_objective_validators_and_code_exec() -> None:
    cases = {case["id"]: case for case in admission_cases()}
    add = validate_generation(cases["arith_add"], "65")
    assert add["pass"] is True
    bad = validate_generation(cases["arith_add"], "66")
    assert bad["pass"] is False
    weather = validate_generation(
        cases["tool_weather_city"],
        '{"name": "get_weather", "arguments": {"city": "Oslo"}}',
    )
    assert weather["pass"] is True
    code = validate_generation(cases["code_len"], "result = 3")
    assert code["pass"] is True
    bullets = validate_generation(cases["instr_two_bullets"], "- apple\n- pear")
    assert bullets["pass"] is True


def test_encoding_identity_rejects_wrong_path() -> None:
    from tools.quality.qwen_fixtures import QWEN_CONTINUATIONS

    case = QWEN_CONTINUATIONS[0]
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
    identity = scoring_identity(quartz, llama)
    assert identity["identical_encodings"] is True
    assert identity["identical_graph_path"] is True
    packed = dict(quartz)
    packed["weight_encoding"] = {
        **quartz["weight_encoding"],
        "q4_decode": "packed",
    }
    with pytest.raises(QualityFrameworkError, match="encoding/graph-path"):
        encoding_identity(packed, llama)
    with pytest.raises(QualityFrameworkError, match="packed Q4"):
        require_contract_encodings(packed)


def test_negative_wrong_path_missing_case_fabricated_aggregate() -> None:
    with pytest.raises(QualityFrameworkError, match="all-prefill"):
        require_cache_path({"path": FORBIDDEN_PATH, "all_prefill": True})
    with pytest.raises(QualityFrameworkError, match="missing long-cache"):
        require_cache_path({"path": CACHE_DECODE_PATH, "prefixes": []})
    spans = [
        {"id": "cache_8192", "ppl_ratio": 1.0},
        {"id": "cache_32768", "ppl_ratio": 1.0},
        {"id": "cache_131040_plus_32", "ppl_ratio": 1.0},
    ]
    with pytest.raises(QualityFrameworkError, match="fabricated aggregate"):
        refuse_fabricated_aggregate(spans, {"ppl_ratio": 1.5})
    with pytest.raises(QualityFrameworkError, match="bitwise"):
        evaluate_kernel_reference(
            declared={"scales": "a"},
            independent={
                "scales": "a",
                "clipping": "a",
                "ties": "a",
                "tails": "a",
                "zeros": "a",
                "extreme_values": "a",
            },
            bitwise_same_math_required=True,
        )
    with pytest.raises(QualityFrameworkError, match="corruption"):
        evaluate_kernel_reference(
            declared={
                "scales": "a",
                "clipping": "a",
                "ties": "a",
                "tails": "a",
                "zeros": "a",
                "extreme_values": "a",
                "corruption": True,
            },
            independent={
                "scales": "a",
                "clipping": "a",
                "ties": "a",
                "tails": "a",
                "zeros": "a",
                "extreme_values": "a",
            },
        )
    with pytest.raises(QualityFrameworkError, match="graph/eager"):
        evaluate_same_path_exact(
            graph_logits=(1.0, 2.0),
            eager_logits=(1.0, 2.1),
            restore_tokens=(1,),
            live_tokens=(1,),
        )


def test_remote_scorer_invocation_refused_in_pytest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.opt116_generated_quality as mod

    monkeypatch.setattr(mod, "FIXTURE", tmp_path / "opt116_generated_quality.json")
    monkeypatch.setattr(mod, "REPORT", tmp_path / "REPORT.md")
    with pytest.raises(QualityFrameworkError, match="cannot run in pytest"):
        run_phase("baseline", openrouter=True)
    os.environ.pop(API_KEY_ENV, None)
    auth = run_phase("authenticate", skip_gpu=True)
    assert auth["openrouter"] == "disabled"


def test_iteration_loop_products_and_dry_run_phases() -> None:
    iteration = load_contract("OPT-116")
    assert loop_product(iteration["workloads"]["authenticate"]) == 7168
    assert loop_product(iteration["workloads"]["generated"]) == 2048
    assert loop_product(iteration["workloads"]["cache-gdn"]) == 128
    assert loop_product(iteration["workloads"]["baseline"]) == 6
    plan = describe_plan("OPT-116", "feedback", iteration, None)
    assert "case_ids=authenticate,generated,cache-gdn,baseline" in plan
    assert "historical_oracles=none" in plan
    auth = describe_plan("OPT-116", "feedback", iteration, "authenticate")
    assert "phase=authenticate" in auth
    assert "product=7168" in auth
    generated = describe_plan("OPT-116", "feedback", iteration, "generated")
    assert "held_out_targets=32" in generated
    assert "product=2048" in generated
    cache = describe_plan("OPT-116", "acceptance", iteration, "cache-gdn")
    assert "phase=cache-gdn" in cache
    assert "product=128" in cache
    baseline = describe_plan("OPT-116", "acceptance", iteration, "baseline")
    assert "phase=baseline" in baseline
    assert "product=6" in baseline


def test_host_phases_write_fixture_and_report_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.opt116_generated_quality as mod

    fixture_path = tmp_path / "opt116_generated_quality.json"
    report_path = tmp_path / "REPORT.md"
    monkeypatch.setattr(mod, "FIXTURE", fixture_path)
    monkeypatch.setattr(mod, "REPORT", report_path)
    auth = run_phase("authenticate", skip_gpu=True)
    assert auth["success"] is True
    assert auth["strict_quality_pass"] is True
    assert auth["successor_quality_pass"] is True
    assert auth["absolute_task_quality"] == "fail"
    assert auth["baseline_regression"] == "pass"
    generated = run_phase("generated", skip_gpu=True)
    assert generated["success"] is True
    assert generated["cases"] == 32
    assert generated["objective_failures"] == 0
    cache = run_phase("cache-gdn", skip_gpu=True)
    assert cache["success"] is True
    assert cache["cache_path_exercised"] is True
    assert cache["state_consistency"] is True
    assert cache["path"] == CACHE_DECODE_PATH
    baseline = run_phase("baseline", skip_gpu=True)
    assert baseline["success"] is True
    assert baseline["strict_quality_pass"] is True
    assert baseline["successor_quality_pass"] is True
    assert baseline["absolute_task_quality"] == "fail"
    assert baseline["baseline_regression"] == "pass"
    assert baseline["cache_path_exercised"] is True
    assert baseline["state_consistency"] is True
    assert baseline["candidate_installed"] is False
    assert fixture_path.is_file()
    assert report_path.is_file()
    fixture = load_json(fixture_path)
    assert fixture["claims_throughput"] is False
    assert fixture["candidate_installed"] is False
    assert fixture["production_kernel_changed"] is False
    assert fixture["strict_quality_pass"] is True
    assert fixture["successor_quality_pass"] is True
    assert fixture["absolute_task_quality"] == "fail"
    assert fixture["baseline_regression"] == "pass"
    assert fixture["cache_path_exercised"] is True
    assert fixture["state_consistency"] is True
    assert fixture["opt056_remains_blocked"] is True
    assert fixture["opt016_remains_blocked"] is True
    assert fixture["historical_failures_erased"] is False
    assert fixture["single_boolean_refused"] is True
    assert fixture["openrouter"]["status"] == "disabled"
    assert fixture["authority"]["date"] == "2026-09-13"
    assert len(fixture["generated"]["cases"]) >= 32
    assert fixture["generated"]["calibration_separated"] is True
    assert fixture["generated"]["new_objective_failures"] == 0
    assert fixture["generated"]["generation_source"] == (
        "protocol_frozen_expected_outputs"
    )
    assert fixture["gpu"]["long_cache_gpu_executed"] is False
    assert fixture["gpu"]["free_running_gpu_executed"] is False
    assert fixture["gpu"]["ran"] is False
    assert len(fixture["generated"]["sample_subset_ids"]) >= 12
    assert len(fixture["generated"]["sample"]) == 12 * 3
    prose = [row for row in fixture["generated"]["greedy"] if row.get("rubric")]
    assert prose
    for row in prose:
        assert "raw_text" in row["rubric"]
        assert "reasons" in row["rubric"]
    dual = fixture["authenticate"]["opt073_dual_verdict"]["quartz"]
    assert dual["absolute_task_accuracy"]["status"] == "fail"
    assert dual["engine_non_regression"]["status"] == "pass"
    text = report_path.read_text(encoding="utf-8")
    for item in PROOF:
        assert item in text
    assert "strict_quality_pass" in text
    assert "2026-09-13" in text
    ledger = LEDGER.read_text(encoding="utf-8")
    assert "| OPT-056 |" in ledger and "blocked" in ledger
    assert os.environ.get("PYTEST_CURRENT_TEST")
    with pytest.raises(QualityFrameworkError, match="cannot run in pytest"):
        run_phase("baseline", openrouter=True)
