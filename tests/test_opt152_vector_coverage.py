"""Host tests for OPT-152 vector coverage dispatch screen wiring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.opt152_vector_coverage import (
    ALL_SHORT,
    CONTRACT,
    CROSSOVER,
    FIXTURE,
    GAP_ONLY,
    ITERATION,
    MMA_THRESHOLD,
    PARENT_STACK,
    PHASES,
    QUALITY_NATIVE,
    REPORT,
    REQUIRED_FIXTURE_KEYS,
    SCREEN_PAIRS,
    SCREEN_SHAPES,
    SCREEN_WARMUPS,
    SHIPPING_DECODE,
    SHIPPING_PROMPT,
    VERIFIED_MAX,
    PAIR_COUNT,
    WARMUPS,
    apply_choice_rule,
    evaluate_keep,
    family_plan,
    load_contract,
    production_coverage_pin,
    production_pins_parent,
    skip_payload,
    validate_fixture,
)
from tools.performance_keep_policy import freeze_hash, validate_opt_in_contract
from tools.run_optimization_task import (
    describe_plan,
    load_contract as load_iteration,
    loop_product,
    validate_future_keep_policy,
    workload_for_mode,
)

ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = ROOT / "Makefile"
DECODE_PIN = ROOT / "cuda/attention_decode_path.cuh"
PROMPT_PIN = ROOT / "cuda/fattn_mma_f16.cuh"
NATIVE = ROOT / "cuda/opt152_vector_coverage_test.cu"
RUNNER = ROOT / "tools/opt152_vector_coverage.py"
TASK_RUNNER = ROOT / "tools/run_optimization_task.py"
PROVENANCE = ROOT / "pins/opt152_vector_coverage_provenance.json"
ENGINE = ROOT / "cuda/optimization_engine_probe.cu"
QUALITY = ROOT / "cuda/opt058_quality_baseline_test.cu"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_makefile_and_registration() -> None:
    contract = load_contract()
    iteration = load_iteration("OPT-152")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    decode_pin = DECODE_PIN.read_text(encoding="utf-8")
    prompt_pin = PROMPT_PIN.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    task_runner = TASK_RUNNER.read_text(encoding="utf-8")
    native = NATIVE.read_text(encoding="utf-8")
    engine = ENGINE.read_text(encoding="utf-8")
    quality = QUALITY.read_text(encoding="utf-8")
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert PROVENANCE.is_file()
    assert NATIVE.is_file()
    assert contract["task"] == "OPT-152"
    assert contract["parent"] == PARENT_STACK
    assert contract["parent_execution_graphs"] == "decode_segments8"
    assert contract["parent_prompt_attention"] == SHIPPING_PROMPT
    assert contract["parent_decode_attention"] == SHIPPING_DECODE
    assert contract["keep_policy_id"] == "target_guard_v2"
    assert contract["require_candidate_nll"] is True
    assert contract["mma_threshold"] == MMA_THRESHOLD
    assert contract["crossover_threshold"] == CROSSOVER
    assert contract["parent_verified_max"] == VERIFIED_MAX
    assert contract["do_not_increase_verified_max"] is True
    assert contract["aa_warmups"] == WARMUPS
    assert contract["screen_pairs"] == SCREEN_PAIRS
    assert contract["screen_warmups"] == SCREEN_WARMUPS
    assert contract["aa_pairs"] == PAIR_COUNT
    assert GAP_ONLY in contract["candidates"]
    assert ALL_SHORT in contract["candidates"]
    assert contract["screen_shapes"] == list(SCREEN_SHAPES)
    assert iteration["diagnostics_make_target"] == "cuda-opt152-diagnostics"
    assert iteration["aggregate_deadline_s"] == 7200
    assert "quality" in iteration["modes"]["acceptance"]["workloads"]
    assert iteration["workloads"]["quality"]["target"].endswith(
        "opt058-quality-baseline-test"
    )
    assert "--phase" in runner
    assert "quality" in PHASES
    assert "cuda-opt152-diagnostics" in makefile
    assert "qw38-cuda-opt152-vector-coverage-test" in makefile
    assert "qw38-cuda-opt058-quality-baseline-test" in makefile
    assert "QW38_OPT152_VECTOR_COVERAGE_RESULT=" in task_runner
    assert "QW38_OPT152_NATIVE_COUNTS=" in task_runner
    assert "--quality" in runner
    assert "--quality-config" in runner
    assert "--q4-decode" in runner
    assert QUALITY_NATIVE in runner
    assert "candidate_nll_not_measured" in runner
    assert "missing_required_quality_not_no_opportunity" in runner
    assert "no_opportunity" not in " ".join(contract["verdicts"])
    assert f'kSelectedAttentionPipelinePath[] = "{SHIPPING_PROMPT}"' in prompt_pin
    assert "kOpt137MmaThreshold = 8192" in decode_pin
    assert "kSelectedOpt137DenseMma = true" in decode_pin
    assert "kSelectedVec128NParts = 16" in decode_pin
    assert "kSelectedDecodeAttentionVerifiedMax = 4096" in decode_pin
    if FIXTURE.is_file() and _json(FIXTURE).get("production_kept"):
        assert production_coverage_pin() in {"gap_only", "all_short"}
    else:
        assert 'kSelectedDecodeAttentionFlashVecCoverage[] = "parent"' in decode_pin
    assert "kLegalDecodeAttentionFlashVecGapOnly" in decode_pin
    assert "kLegalDecodeAttentionFlashVecAllShort" in decode_pin
    assert "opt152_flash_vec_covers" in decode_pin
    assert GAP_ONLY in native
    assert ALL_SHORT in native
    assert "live_device_position=true" in native
    assert "host_selector_only=false" in native
    assert "DecodeLaunchState" in native
    assert GAP_ONLY in engine
    assert ALL_SHORT in engine
    assert "options.prefix == 512" in engine
    assert "options.prefix == 6144" in engine
    assert GAP_ONLY in quality
    assert ALL_SHORT in quality
    provenance = _json(PROVENANCE)
    assert "dispatch-only" in provenance["specialization"]
    validate_future_keep_policy("OPT-152", iteration)
    validate_opt_in_contract(contract)


def test_choice_rule_prefers_all_short_only_when_both_targets_and_d128_hold() -> None:
    parsed = {
        "shapes": {
            GAP_ONLY: {
                128: {"control_mean_ms": 1.0, "candidate_mean_ms": 1.0},
                512: {"control_mean_ms": 1.0, "candidate_mean_ms": 1.0},
                6144: {"control_mean_ms": 2.0, "candidate_mean_ms": 1.5},
            },
            ALL_SHORT: {
                128: {"control_mean_ms": 1.0, "candidate_mean_ms": 1.01},
                512: {"control_mean_ms": 1.2, "candidate_mean_ms": 1.0},
                6144: {"control_mean_ms": 2.0, "candidate_mean_ms": 1.4},
            },
        }
    }
    choice = apply_choice_rule(parsed)
    assert choice["survivor"] == ALL_SHORT
    parsed["shapes"][ALL_SHORT][128]["candidate_mean_ms"] = 1.05
    choice = apply_choice_rule(parsed)
    assert choice["survivor"] == GAP_ONLY
    parsed["shapes"][GAP_ONLY][6144]["candidate_mean_ms"] = 2.1
    parsed["shapes"][ALL_SHORT][512]["candidate_mean_ms"] = 1.3
    choice = apply_choice_rule(parsed)
    assert choice["survivor"] is None


def test_iteration_loop_products() -> None:
    iteration = load_iteration("OPT-152")
    correctness = iteration["workloads"]["correctness"]
    screen = iteration["workloads"]["screen"]
    quality = iteration["workloads"]["quality"]
    performance = iteration["workloads"]["performance"]
    assert loop_product(workload_for_mode(correctness, "feedback")) == 84
    assert loop_product(workload_for_mode(screen, "feedback")) == 24
    assert loop_product(workload_for_mode(quality, "acceptance")) == 12288
    assert loop_product(workload_for_mode(performance, "acceptance")) == 182
    described = describe_plan("OPT-152", "feedback", iteration, "correctness")
    assert "phase=correctness" in described
    assert "loop_product=84" in family_plan("feedback", "correctness")
    proof = " ".join(iteration["proof_limit"]).casefold()
    assert "7200 is a ceiling" in proof
    assert "opt-058" in proof
    assert "no_opportunity" in proof
    assert "verified_max" in proof


def test_quality_phase_is_wired() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    iteration = load_iteration("OPT-152")
    assert "run_quality" in runner
    assert "build/qw38-cuda-opt058-quality-baseline-test" in runner
    assert (
        iteration["workloads"]["quality"]["kind"] == "opt058_quality_flag_candidate_nll"
    )
    assert "quality-region" in runner
    assert 'selectors["decode_attention"] = config_id' in runner


def test_dispatch_guards_and_no_verified_max_raise() -> None:
    decode_pin = DECODE_PIN.read_text(encoding="utf-8")
    native = NATIVE.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    assert "kSelectedDecodeAttentionVerifiedMax = 4096" in decode_pin
    assert "kOpt152FlashVecHiExclusive = 8192" in decode_pin
    assert (
        "Do not raise verified_max" in decode_pin
        or "do not raise verified_max" in decode_pin
    )
    assert "kSelectedDecodeAttentionVerifiedMax == 4096" in native
    assert "apply_vec128_n_parts(8)" not in runner
    assert "kSelectedDecodeAttentionVerifiedMax = 8192" not in decode_pin
    freeze_hash(load_contract())
    for key in REQUIRED_FIXTURE_KEYS:
        assert key in load_contract()["required_fixture_keys"]
    assert REPORT.parent.as_posix().endswith("opt152-vector-coverage")


def test_production_pin_matches_parent_until_keep() -> None:
    decode_pin = DECODE_PIN.read_text(encoding="utf-8")
    prompt_pin = PROMPT_PIN.read_text(encoding="utf-8")
    if FIXTURE.is_file() and _json(FIXTURE).get("production_kept"):
        assert production_coverage_pin() in {"gap_only", "all_short"}
    else:
        assert production_pins_parent() is True
        assert production_coverage_pin() == "parent"
        assert 'kSelectedDecodeAttentionFlashVecCoverage[] = "parent"' in decode_pin
    assert f'kSelectedAttentionPipelinePath[] = "{SHIPPING_PROMPT}"' in prompt_pin
    assert "kSelectedDecodeAttentionFlashVec = true" in decode_pin
    assert "kSelectedOpt137DenseMma = true" in decode_pin


def test_screened_in_missing_quality_is_incomplete_not_no_opportunity(
    tmp_path: Path,
) -> None:
    (tmp_path / "correctness.json").write_text(
        json.dumps({"ok": True, "blocked": False, "task": "OPT-152"}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "screen.json").write_text(
        json.dumps(
            {
                "ok": True,
                "screened_in": True,
                "screened_out": False,
                "survivor": GAP_ONLY,
                "choice": {"reason": "gap_only_d6144_improved"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    result = evaluate_keep(tmp_path)
    assert result["verdict"] == "incomplete"
    assert result["shipping_delta"] == 0
    assert result["production_kept"] is False
    assert "no_opportunity" not in result["verdict"]
    assert "quality" in " ".join(result["reasons"])


def test_quality_skip_on_screened_out_does_not_stub_nll(tmp_path: Path) -> None:
    payload = skip_payload(
        tmp_path, "acceptance", "quality", reason="screened_out_retain_parent"
    )
    assert payload["ok"] is True
    assert payload["skipped"] is True
    assert payload["opt058_invoked"] is False
    assert payload["candidate_nll_measured"] is False
    assert payload["candidate_nll_not_measured"] is False
    assert payload["nll_required"] is False


def test_validate_fixture_when_present() -> None:
    if not FIXTURE.is_file():
        return
    payload = _json(FIXTURE)
    for key in REQUIRED_FIXTURE_KEYS:
        assert key in payload
    assert payload["task"] == "OPT-152"
    checked = validate_fixture(payload)
    assert checked["ok"] is True
