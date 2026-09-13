"""Host tests for OPT-112 decode norm → typed Q8 staging screen."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.opt075_q4_production_admission import AdmissionError, load_json
from tools.opt112_norm_q8_staging import (
    CANDIDATE_IDS,
    CONFIGS,
    CONSUMERS,
    CONTRACT,
    CONTROL_ID,
    DECODE_TOKENS,
    EXCLUDED_FAMILIES,
    FIXTURE,
    ITERATION,
    MIN_SAVING_MS,
    OPT090_FIXTURE,
    OPT099_FIXTURE,
    PRODUCER_FAMILIES,
    REMOVABLE_FAMILIES,
    REPORT,
    TRIGGER_MS,
    VERDICT_DEFER,
    decide_deferred_verdicts,
    evaluate_screen,
    family_plan,
    mean_family_ms,
    per_token,
    require_eligible,
    run,
)
from tools.run_optimization_task import (
    describe_plan,
    load_contract,
    loop_product,
    parse_native_observation,
    validate_future_keep_policy,
    workload_for_mode,
)

ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = ROOT / "Makefile"
SCHEDULER = ROOT / "cuda/full_scheduler.cu"
Q4_PATH = ROOT / "cuda/q4k_decode_path.cuh"
FFN_PATH = ROOT / "cuda/ffn_decode_path.cuh"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_makefile_and_host_hooks() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-112")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert FIXTURE.is_file()
    assert REPORT.is_file()
    assert contract["task"] == "OPT-112"
    assert contract["control"] == CONTROL_ID
    assert contract["candidates"] == list(CANDIDATE_IDS)
    assert contract["screen_trigger_ms_per_token"] == TRIGGER_MS
    assert contract["min_saving_ms_per_token"] == MIN_SAVING_MS
    assert contract["fusion_implemented"] is False
    assert contract["require_candidate_nll"] is False
    assert contract["opt074_family_admission_required"] is False
    assert iteration["task"] == "OPT-112"
    assert iteration["diagnostics_make_target"] == "cuda-opt112-diagnostics"
    assert iteration["host_only"] is True
    assert iteration["skip_compile"] is True
    assert iteration["performance_admission"]["instrumentation_only"] is True
    assert "cuda-opt112-diagnostics" in makefile
    diagnostics = makefile.split("cuda-opt112-diagnostics:", 1)[1].split("\n", 1)[0]
    assert "qw38-cuda-opt058-quality-baseline-test" not in diagnostics
    assert not (ROOT / "cuda/opt112_norm_q8_staging_test.cu").is_file()
    q4 = Q4_PATH.read_text(encoding="utf-8")
    ffn = FFN_PATH.read_text(encoding="utf-8")
    scheduler = SCHEDULER.read_text(encoding="utf-8")
    assert 'kSelectedQ4DecodePath[] = "llama_q4k_mmvq"' in q4
    assert 'kSelectedFfnDecodePath[] = "paired_integer"' in ffn
    assert "norm_to_typed_q8_rounded" not in scheduler
    assert "residual_norm_to_typed_q8_rounded" not in scheduler


def test_iteration_loop_products_and_plan() -> None:
    iteration = load_contract("OPT-112")
    screen = iteration["workloads"]["screen"]
    same_math = iteration["workloads"]["same-math"]
    decode = iteration["workloads"]["decode"]
    body128 = iteration["workloads"]["body128"]
    d128 = iteration["workloads"]["d128"]
    quality = iteration["workloads"]["quality"]
    assert loop_product(workload_for_mode(screen, "feedback")) == 1
    assert loop_product(workload_for_mode(same_math, "feedback")) == 3
    assert loop_product(workload_for_mode(decode, "feedback")) == 384
    assert loop_product(workload_for_mode(body128, "acceptance")) == 39
    assert loop_product(workload_for_mode(d128, "acceptance")) == 39
    assert loop_product(workload_for_mode(quality, "release")) == 1
    described = describe_plan("OPT-112", "feedback", iteration, "screen")
    assert "phase=screen" in described
    assert "historical_oracles=none" in described
    plan = family_plan("screen", "acceptance")
    assert plan["gpu_work"] is False
    assert plan["samples"] == 1
    proof = " ".join(iteration["proof_limit"]).casefold()
    assert "upper-bound clue" in proof
    assert "0.50" in proof
    validate_future_keep_policy("OPT-112", iteration)


def test_two_fusion_candidates_only() -> None:
    assert [row["id"] for row in CONFIGS] == [CONTROL_ID, *CANDIDATE_IDS]
    assert CONFIGS[0]["implemented"] is True
    assert CONFIGS[1]["implemented"] is False
    assert CONFIGS[2]["implemented"] is False
    assert len(CANDIDATE_IDS) == 2


def test_consumer_inventory_retains_logits_bf16() -> None:
    by_id = {row["id"]: row for row in CONSUMERS}
    assert by_id["logits_output_norm"]["retain_existing_path"] is True
    assert by_id["logits_output_norm"]["bf16_externally_observed"] is True
    assert by_id["ffn_gate_up"]["encoding"] == "Q8_1Block"
    assert by_id["ffn_gate_up"]["alternate_encoding"] == "Q8Block"
    assert by_id["mixer_gdn_qkv_gate_alpha_beta"]["encoding"] == "Q8_1Block"
    assert by_id["mixer_attn_query_gate_key_value"]["layer_count"] == 16


def test_opt099_ffn_norm_clue_is_not_the_removable_budget() -> None:
    assert OPT099_FIXTURE.is_file()
    opt099 = load_json(OPT099_FIXTURE)
    families = mean_family_ms(opt099["d2048"])
    clue = per_token(float(families["ffn_norm"]), DECODE_TOKENS)
    result = evaluate_screen(opt099, load_json(OPT090_FIXTURE))
    assert clue >= TRIGGER_MS
    assert result["max_upper_bound_clue_ms_per_token"] == pytest.approx(clue)
    assert result["d2048"]["upper_bound_clue_ms_per_token"] == pytest.approx(clue)
    assert result["min_potentially_removable_ms_per_token"] < TRIGGER_MS
    assert result["eligible"] is False
    assert result["verdict"] == VERDICT_DEFER
    assert result["reason"] == "upper_bound_clue_is_not_removable_budget"
    assert result["fusion_implemented"] is False
    assert "ffn_norm" in PRODUCER_FAMILIES
    assert "activation_quant" in REMOVABLE_FAMILIES
    assert "logits_norm" in EXCLUDED_FAMILIES


def test_screen_defers_when_isolated_staging_below_trigger() -> None:
    synthetic = {
        "d128": {
            "output_tokens": 32,
            "samples": [
                {
                    "ident": "post098_selected",
                    "families": {
                        "ffn_norm": 26.29,
                        "input_norm": 0.52,
                        "activation_quant": 0.0,
                        "residual_mixer": 8.0,
                        "logits_norm": 9.3,
                    },
                }
            ],
        },
        "d2048": {
            "output_tokens": 32,
            "samples": [
                {
                    "ident": "post098_selected",
                    "families": {
                        "ffn_norm": 26.29,
                        "input_norm": 0.52,
                        "activation_quant": 0.0,
                        "residual_mixer": 8.08,
                        "logits_norm": 9.3,
                    },
                }
            ],
        },
    }
    result = evaluate_screen(synthetic)
    assert result["eligible"] is False
    assert result["verdict"] == VERDICT_DEFER
    assert result["d128"]["potentially_removable_ms_per_token"] == pytest.approx(0.25)
    assert result["d2048"]["potentially_removable_ms_per_token"] == pytest.approx(
        0.2525
    )


def test_screen_proceeds_only_when_both_prefixes_meet_trigger() -> None:
    synthetic = {
        "d128": {
            "output_tokens": 32,
            "samples": [
                {
                    "ident": "post098_selected",
                    "families": {"activation_quant": 20.0, "residual_mixer": 0.0},
                }
            ],
        },
        "d2048": {
            "output_tokens": 32,
            "samples": [
                {
                    "ident": "post098_selected",
                    "families": {"activation_quant": 16.0, "residual_mixer": 0.0},
                }
            ],
        },
    }
    result = evaluate_screen(synthetic)
    assert result["eligible"] is True
    assert result["verdict"] == "proceed"
    assert result["d128"]["potentially_removable_ms_per_token"] == pytest.approx(0.625)
    assert result["d2048"]["potentially_removable_ms_per_token"] == pytest.approx(0.5)

    mixed = {
        "d128": synthetic["d128"],
        "d2048": {
            "output_tokens": 32,
            "samples": [
                {
                    "ident": "post098_selected",
                    "families": {"activation_quant": 8.0, "residual_mixer": 0.0},
                }
            ],
        },
    }
    blocked = evaluate_screen(mixed)
    assert blocked["eligible"] is False
    assert blocked["verdict"] == VERDICT_DEFER


def test_deferred_verdicts_keep_current_path() -> None:
    decided = decide_deferred_verdicts(VERDICT_DEFER)
    assert decided["status"] == VERDICT_DEFER
    assert decided["production_kept"] is True
    assert decided["selected_path"] == CONTROL_ID
    assert decided["fusion_implemented"] is False
    assert decided["claims_throughput"] is False
    assert decided["independent_verdicts"][CONTROL_ID]["production_kept"] is True
    for candidate in CANDIDATE_IDS:
        assert decided["independent_verdicts"][candidate]["production_kept"] is False


def test_stub_fusion_phases_require_eligibility() -> None:
    with pytest.raises(AdmissionError, match="deferred_below_trigger"):
        require_eligible({"screen": {"eligible": False, "verdict": VERDICT_DEFER}})
    require_eligible({"screen": {"eligible": True}})


def test_parse_native_observation_opt112_prefix() -> None:
    stdout = (
        'QW38_OPT112_NATIVE_COUNTS={"schema_version":1,"task":"OPT-112",'
        '"warmups":0,"samples":1,"observed_warmups":0,"observed_samples":1,'
        '"observed_candidates":1,"observed_shapes":1,"observed_tier":"correctness",'
        '"pairs":1,"sample_ids":[0],"acceptance_executed":false,"keep":false}\n'
    )
    observed = parse_native_observation(stdout)
    assert observed["task"] == "OPT-112"
    assert observed["samples"] == 1
    assert observed["keep"] is False


def test_fixture_records_deferred_screen() -> None:
    fixture = _json(FIXTURE)
    assert fixture["task"] == "OPT-112"
    assert fixture["status"] == VERDICT_DEFER
    assert fixture["fusion_implemented"] is False
    assert fixture["selected_path"] == CONTROL_ID
    assert fixture["claims_throughput"] is False
    assert fixture["screen"]["eligible"] is False
    assert fixture["screen"]["verdict"] == VERDICT_DEFER
    assert (
        float(fixture["screen"]["min_potentially_removable_ms_per_token"]) < TRIGGER_MS
    )
    report = REPORT.read_text(encoding="utf-8")
    assert VERDICT_DEFER in report
    assert "upper-bound" in report.casefold() or "upper-bound" in report


def test_run_screen_acceptance(tmp_path: Path) -> None:
    results = run("acceptance", "screen", tmp_path)
    assert results["status"] == VERDICT_DEFER
    assert results["fusion_implemented"] is False
    assert (tmp_path / "opt112_norm_q8_staging.json").is_file()
    with pytest.raises(AdmissionError, match="deferred_below_trigger"):
        run("acceptance", "same-math", tmp_path)
    with pytest.raises(AdmissionError, match="deferred_below_trigger"):
        run("acceptance", "quality", tmp_path)
