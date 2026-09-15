"""Host tests for OPT-147 prefill 8x8 hypothesis screen wiring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.opt147_prefill_8x8 import (
    CANDIDATE,
    CANDIDATE_LAUNCH,
    CONTRACT,
    CROSSOVER,
    FIXTURE,
    ITERATION,
    MMA_THRESHOLD,
    PARENT_STACK,
    PHASES,
    PREFILL_KERNEL,
    QUALITY_NATIVE,
    REPORT,
    REQUIRED_FIXTURE_KEYS,
    SCREEN_PAIRS,
    SCREEN_WARMUPS,
    SHIPPING_PROMPT,
    PAIR_COUNT,
    WARMUPS,
    evaluate_keep,
    family_plan,
    load_contract,
    production_pin,
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
PIPELINE = ROOT / "cuda/fattn_mma_f16_pipeline.cuh"
NATIVE = ROOT / "cuda/opt147_prefill_8x8_test.cu"
HEADER = ROOT / "cuda/opt147_prefill_8x8.cuh"
RUNNER = ROOT / "tools/opt147_prefill_8x8.py"
TASK_RUNNER = ROOT / "tools/run_optimization_task.py"
PROVENANCE = ROOT / "pins/opt147_prefill_8x8_provenance.json"
ENGINE = ROOT / "cuda/optimization_engine_probe.cu"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_makefile_and_registration() -> None:
    contract = load_contract()
    iteration = load_iteration("OPT-147")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    decode_pin = DECODE_PIN.read_text(encoding="utf-8")
    prompt_pin = PROMPT_PIN.read_text(encoding="utf-8")
    pipeline = PIPELINE.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    task_runner = TASK_RUNNER.read_text(encoding="utf-8")
    native = NATIVE.read_text(encoding="utf-8")
    header = HEADER.read_text(encoding="utf-8")
    engine = ENGINE.read_text(encoding="utf-8")
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert PROVENANCE.is_file()
    assert HEADER.is_file()
    assert contract["task"] == "OPT-147"
    assert contract["parent"] == PARENT_STACK
    assert contract["parent_execution_graphs"] == "decode_segments8"
    assert contract["parent_prompt_attention"] == "opt111_base"
    assert contract["candidate"] == CANDIDATE
    assert contract["candidate_id"] == "prefill_attention_8x8_v1"
    assert contract["keep_policy_id"] == "target_guard_v2"
    assert contract["require_candidate_nll"] is True
    assert contract["mma_threshold"] == MMA_THRESHOLD
    assert contract["crossover_threshold"] == CROSSOVER
    assert contract["aa_warmups"] == WARMUPS
    assert contract["screen_pairs"] == SCREEN_PAIRS
    assert contract["screen_warmups"] == SCREEN_WARMUPS
    assert contract["aa_pairs"] == PAIR_COUNT
    assert contract["ncols1"] == 8
    assert contract["ncols2"] == 8
    assert contract["gqa_ratio"] == 6
    assert contract["target_workloads"] == ["p4096"]
    assert "d128" in contract["guard_workloads"]
    assert "d2048" in contract["guard_workloads"]
    assert "d8192" in contract["guard_workloads"]
    assert "d32768" in contract["guard_workloads"]
    assert iteration["diagnostics_make_target"] == "cuda-opt147-diagnostics"
    assert iteration["aggregate_deadline_s"] == 7200
    assert "quality" in iteration["modes"]["acceptance"]["workloads"]
    assert iteration["workloads"]["quality"]["target"].endswith(
        "opt058-quality-baseline-test"
    )
    assert "--phase" in runner
    assert '"quality"' in runner or "quality" in PHASES
    assert "cuda-opt147-diagnostics" in makefile
    assert "qw38-cuda-opt147-prefill-8x8-test" in makefile
    assert "qw38-cuda-opt058-quality-baseline-test" in makefile
    assert "QW38_OPT147_PREFILL_8X8_RESULT=" in task_runner
    assert "QW38_OPT147_NATIVE_COUNTS=" in task_runner
    assert "--quality" in runner
    assert "--quality-config" in runner
    assert "--q4-decode" in runner
    assert QUALITY_NATIVE in runner
    assert "candidate_nll_not_measured" in runner
    assert "missing_required_quality_not_no_opportunity" in runner
    assert "no_opportunity" not in " ".join(contract["verdicts"])
    if FIXTURE.is_file() and _json(FIXTURE).get("production_kept"):
        assert f'kSelectedAttentionPipelinePath[] = "{CANDIDATE}"' in prompt_pin
    else:
        assert 'kSelectedAttentionPipelinePath[] = "opt111_base"' in prompt_pin
    assert "prefill_attention_8x8_v1" in pipeline
    assert "fattn_mma_pipeline_prefill_attention_8x8_v1" in pipeline
    assert "live_head" in pipeline
    assert "member >= kGqaRatio" in pipeline
    assert "kLegalAttentionPipelinePrefill8x8" in pipeline
    assert CANDIDATE in header
    assert CANDIDATE_LAUNCH in header
    assert (
        "ncols1=8" in native or "ncols1=8" in header.lower() or "kNcols1 = 8" in header
    )
    assert "--pairs must be 1..10" in engine
    assert "prefill_attention_8x8_v1" in engine
    assert "options.pairs <= 10" in engine
    assert "options.prefix == 32768" in engine
    assert "options.output_tokens == 256" in engine
    assert "kOpt137MmaThreshold = 8192" in decode_pin
    assert PHASES[0] == "correctness"
    assert "quality" in PHASES
    provenance = _json(PROVENANCE)
    assert provenance["candidate"] == CANDIDATE
    validate_future_keep_policy("OPT-147", iteration)
    validate_opt_in_contract(contract)


def test_target_guard_v2_roles() -> None:
    contract = load_contract()
    roles = {row["workload"]: "target" for row in contract["targets"]}
    roles.update({row["workload"]: "guard" for row in contract["guards"]})
    assert roles["p4096"] == "target"
    assert roles["d128"] == "guard"
    assert roles["d2048"] == "guard"
    assert roles["d8192"] == "guard"
    assert roles["d32768"] == "guard"
    assert "p4096" not in {row["workload"] for row in contract["guards"]}
    freeze_hash(contract)
    for key in REQUIRED_FIXTURE_KEYS:
        assert key in contract["required_fixture_keys"]
    assert REPORT.parent.as_posix().endswith("opt147-prefill-8x8")


def test_iteration_loop_products() -> None:
    iteration = load_iteration("OPT-147")
    correctness = iteration["workloads"]["correctness"]
    screen = iteration["workloads"]["screen"]
    quality = iteration["workloads"]["quality"]
    performance = iteration["workloads"]["performance"]
    assert loop_product(workload_for_mode(correctness, "feedback")) == 16
    assert loop_product(workload_for_mode(screen, "feedback")) == 32768
    assert loop_product(workload_for_mode(quality, "acceptance")) == 12288
    assert loop_product(workload_for_mode(performance, "acceptance")) == 130
    described = describe_plan("OPT-147", "feedback", iteration, "correctness")
    assert "phase=correctness" in described
    assert "loop_product=16" in family_plan("feedback", "correctness")
    proof = " ".join(iteration["proof_limit"]).casefold()
    assert "7200 is a ceiling" in proof
    assert "opt-058" in proof
    assert "unknown" in proof
    assert "no_opportunity" in proof


def test_quality_phase_is_wired() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    iteration = load_iteration("OPT-147")
    assert "--phase" in runner
    assert "run_quality" in runner
    assert "build/qw38-cuda-opt058-quality-baseline-test" in runner
    assert (
        iteration["workloads"]["quality"]["kind"] == "opt058_quality_flag_candidate_nll"
    )
    assert "--phase" in " ".join(iteration["workloads"]["quality"]["args"])
    assert "quality" in iteration["workloads"]["quality"]["args"]


def test_production_pin_matches_keep_verdict() -> None:
    prompt_pin = PROMPT_PIN.read_text(encoding="utf-8")
    decode_pin = DECODE_PIN.read_text(encoding="utf-8")
    if FIXTURE.is_file() and _json(FIXTURE).get("production_kept"):
        assert production_pin() == CANDIDATE
        assert f'kSelectedAttentionPipelinePath[] = "{CANDIDATE}"' in prompt_pin
    else:
        assert production_pins_parent() is True
        assert 'kSelectedAttentionPipelinePath[] = "opt111_base"' in prompt_pin
        assert (
            'kSelectedAttentionPipelinePath[] = "prefill_attention_8x8_v1"'
            not in prompt_pin
        )
    assert "kSelectedOpt137DenseMma = true" in decode_pin
    assert "kOpt137MmaThreshold = 8192" in decode_pin


def test_gqa_pad_and_opt111_arithmetic_retained() -> None:
    pipeline = PIPELINE.read_text(encoding="utf-8")
    header = HEADER.read_text(encoding="utf-8")
    native = NATIVE.read_text(encoding="utf-8")
    assert "Ncols2 == 8" in pipeline or "Ncols2 != 8" in pipeline
    assert "live_head" in pipeline
    assert "member >= kGqaRatio" in pipeline
    assert "kConvertKvOnce = true" in header
    assert "kLlamaLoad = true" in header
    assert "kXorSwizzle = false" in header
    assert "gqa_pad_heads=2" in native
    assert PREFILL_KERNEL in header
    assert CANDIDATE_LAUNCH in pipeline
    launch = "launch_fattn_mma_pipeline_typed<8, false, true, 32, 2, 8, 1, 2, true"
    assert launch in pipeline


def test_screened_in_missing_quality_is_incomplete_not_no_opportunity(
    tmp_path: Path,
) -> None:
    dump = tmp_path / "correctness.json"
    dump.write_text(
        json.dumps({"ok": True, "blocked": False, "task": "OPT-147"}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "screen.json").write_text(
        json.dumps(
            {
                "ok": True,
                "screened_in": True,
                "screened_out": False,
                "parsed": {"saving_ms": 12.0},
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
    reasons = " ".join(result["reasons"])
    assert "quality" in reasons
    assert result["shipping_prompt_attention"] == SHIPPING_PROMPT


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
    assert payload["skip_reason"] == "screened_out_retain_parent"


def test_validate_fixture_when_present() -> None:
    if not FIXTURE.is_file():
        return
    payload = _json(FIXTURE)
    for key in REQUIRED_FIXTURE_KEYS:
        assert key in payload
    assert payload["task"] == "OPT-147"
    assert payload["parent"] == PARENT_STACK
    assert payload["candidate"] == CANDIDATE
    quality = payload.get("quality") or {}
    if payload.get("verdict") in {"screened_out", "reject", "blocked"}:
        assert payload["production_kept"] is False
        assert payload["shipping_prompt_attention"] == SHIPPING_PROMPT
        assert payload["shipping_delta"] == 0
        assert quality.get("candidate_nll_not_measured") is not True
    if payload.get("production_kept"):
        assert payload["verdict"] == "keep"
        assert quality.get("opt058_invoked") is True
        assert quality.get("candidate_nll_measured") is True
        assert quality.get("candidate_nll_not_measured") is not True
    checked = validate_fixture(payload)
    assert checked["ok"] is True
    assert REPORT.is_file() or payload.get("mode") != "acceptance"
