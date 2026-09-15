"""Host tests for OPT-149 decode mixer Q8_1 fusion hypothesis screen wiring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.opt149_norm_q8 import (
    CANDIDATE,
    CANDIDATE_LAUNCH,
    CONTRACT,
    CONTROL,
    FIXTURE,
    ITERATION,
    MMA_THRESHOLD,
    PARENT_STACK,
    PHASES,
    QUALITY_NATIVE,
    REPORT,
    REQUIRED_FIXTURE_KEYS,
    SCREEN_PAIRS,
    SCREEN_WARMUPS,
    SHIPPING_DECODE,
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
FUSION_PIN = ROOT / "cuda/opt149_norm_q8.cuh"
Q4_PIN = ROOT / "cuda/q4k_decode_path.cuh"
NATIVE = ROOT / "cuda/opt149_norm_q8_test.cu"
HEADER = ROOT / "cuda/opt149_norm_q8.cuh"
RUNNER = ROOT / "tools/opt149_norm_q8.py"
TASK_RUNNER = ROOT / "tools/run_optimization_task.py"
PROVENANCE = ROOT / "pins/opt149_norm_q8_provenance.json"
ENGINE = ROOT / "cuda/optimization_engine_probe.cu"
QUALITY = ROOT / "cuda/opt058_quality_baseline_test.cu"
SCHEDULER = ROOT / "cuda/full_scheduler.cu"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_makefile_and_registration() -> None:
    contract = load_contract()
    iteration = load_iteration("OPT-149")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    decode_pin = DECODE_PIN.read_text(encoding="utf-8")
    prompt_pin = PROMPT_PIN.read_text(encoding="utf-8")
    fusion = FUSION_PIN.read_text(encoding="utf-8")
    q4 = Q4_PIN.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    task_runner = TASK_RUNNER.read_text(encoding="utf-8")
    native = NATIVE.read_text(encoding="utf-8")
    header = HEADER.read_text(encoding="utf-8")
    engine = ENGINE.read_text(encoding="utf-8")
    quality = QUALITY.read_text(encoding="utf-8")
    scheduler = SCHEDULER.read_text(encoding="utf-8")
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert PROVENANCE.is_file()
    assert HEADER.is_file()
    assert contract["task"] == "OPT-149"
    assert contract["parent"] == PARENT_STACK
    assert contract["parent_execution_graphs"] == "decode_segments8"
    assert contract["parent_prompt_attention"] == SHIPPING_PROMPT
    assert contract["parent_decode_attention"] == SHIPPING_DECODE
    assert contract["parent_decode_norm_q8"] == CONTROL
    assert contract["candidate"] == CANDIDATE
    assert contract["candidate_id"] == CANDIDATE
    assert contract["control"] == CONTROL
    assert contract["keep_policy_id"] == "target_guard_v2"
    assert contract["require_candidate_nll"] is True
    assert contract["mma_threshold"] == MMA_THRESHOLD
    assert contract["aa_warmups"] == WARMUPS
    assert contract["screen_pairs"] == SCREEN_PAIRS
    assert contract["screen_warmups"] == SCREEN_WARMUPS
    assert contract["aa_pairs"] == PAIR_COUNT
    assert contract["target_workloads"] == ["d2048"]
    assert "d128" in contract["guard_workloads"]
    assert "d8192" in contract["guard_workloads"]
    assert "d32768" in contract["guard_workloads"]
    assert "p4096" in contract["guard_workloads"]
    assert iteration["diagnostics_make_target"] == "cuda-opt149-diagnostics"
    assert iteration["aggregate_deadline_s"] == 7200
    assert "quality" in iteration["modes"]["acceptance"]["workloads"]
    assert iteration["workloads"]["quality"]["target"].endswith(
        "opt058-quality-baseline-test"
    )
    assert "--phase" in runner
    assert "quality" in PHASES
    assert "cuda-opt149-diagnostics" in makefile
    assert "qw38-cuda-opt149-norm-q8-test" in makefile
    assert "qw38-cuda-opt058-quality-baseline-test" in makefile
    assert "opt149_norm_q8.cuh" in makefile
    assert "QW38_OPT149_NORM_Q8_RESULT=" in task_runner
    assert "QW38_OPT149_NATIVE_COUNTS=" in task_runner
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
    assert "kSelectedDecodeAttentionFlashVec = true" in decode_pin
    assert "kSelectedDecodeNormQ81Fusion = false" in fusion
    assert 'kSelectedQ4DecodePath[] = "llama_q4k_mmvq"' in q4
    assert CANDIDATE in header
    assert CANDIDATE_LAUNCH in header
    assert "quartz_q8_1_sum_q" in header
    assert "OPT-110 is not revived" in header
    assert "Byte-size alone is not a win" in header
    assert "--pairs must be 1..10" in engine
    assert CANDIDATE in engine
    assert "--decode-norm-q8" in engine
    assert "prefix 2048 + 32" in engine
    assert "decode_norm_q8" in quality
    assert "launch_decode_mixer_input_norm" in scheduler
    assert "launch_decode_mixer_residual_norm" in scheduler
    assert PHASES[0] == "correctness"
    assert "quality" in PHASES
    provenance = _json(PROVENANCE)
    assert provenance["candidate"] == CANDIDATE
    assert "OPT-110" in json.dumps(provenance)
    validate_future_keep_policy("OPT-149", iteration)
    validate_opt_in_contract(contract)
    assert "opt110_revived=false" in native
    assert "byte_size_only_claim=false" in native
    assert "family=decode-mixer" in native
    assert "kMixerLayers = 64" in native or "kMixerLayers" in native
    assert "Do not revive OPT-110" in header or "OPT-110 is not revived" in header


def test_target_guard_v2_roles() -> None:
    contract = load_contract()
    roles = {row["workload"]: "target" for row in contract["targets"]}
    roles.update({row["workload"]: "guard" for row in contract["guards"]})
    assert roles["d2048"] == "target"
    assert roles["d128"] == "guard"
    assert roles["d8192"] == "guard"
    assert roles["d32768"] == "guard"
    assert roles["p4096"] == "guard"
    assert "d2048" not in {row["workload"] for row in contract["guards"]}
    freeze_hash(contract)
    for key in REQUIRED_FIXTURE_KEYS:
        assert key in contract["required_fixture_keys"]
    assert REPORT.parent.as_posix().endswith("opt149-norm-q8")


def test_iteration_loop_products() -> None:
    iteration = load_iteration("OPT-149")
    correctness = iteration["workloads"]["correctness"]
    screen = iteration["workloads"]["screen"]
    quality = iteration["workloads"]["quality"]
    performance = iteration["workloads"]["performance"]
    assert loop_product(workload_for_mode(correctness, "feedback")) == 16
    assert loop_product(workload_for_mode(screen, "feedback")) == 16384
    assert loop_product(workload_for_mode(quality, "acceptance")) == 12288
    assert loop_product(workload_for_mode(performance, "acceptance")) == 130
    described = describe_plan("OPT-149", "feedback", iteration, "correctness")
    assert "phase=correctness" in described
    assert "loop_product=16" in family_plan("feedback", "correctness")
    proof = " ".join(iteration["proof_limit"]).casefold()
    assert "7200 is a ceiling" in proof
    assert "opt-058" in proof
    assert "unknown" in proof
    assert "no_opportunity" in proof
    assert "opt-110" in proof
    assert "byte size" in proof


def test_quality_phase_is_wired() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    iteration = load_iteration("OPT-149")
    assert "--phase" in runner
    assert "run_quality" in runner
    assert "build/qw38-cuda-opt058-quality-baseline-test" in runner
    assert (
        iteration["workloads"]["quality"]["kind"] == "opt058_quality_flag_candidate_nll"
    )
    assert "--phase" in " ".join(iteration["workloads"]["quality"]["args"])
    assert "quality" in iteration["workloads"]["quality"]["args"]
    assert 'selectors["decode_norm_q8"] = config_id' in runner
    assert 'selectors["decode_attention"] = SHIPPING_DECODE' in runner
    assert "--attention-pipeline" in runner
    assert "prefill_attention_8x8_v1" in runner
    assert "llama_q4k_mmvq" in runner
    assert "decode_attention_flash_vec_v1" in runner


def test_no_opt110_revival_and_parent_pins() -> None:
    fusion = FUSION_PIN.read_text(encoding="utf-8")
    native = NATIVE.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    q4 = Q4_PIN.read_text(encoding="utf-8")
    decode_pin = DECODE_PIN.read_text(encoding="utf-8")
    assert "kSelectedDecodeNormQ81Fusion = false" in fusion
    assert 'kSelectedQ4DecodePath[] = "llama_q4k_mmvq"' in q4
    assert "kSelectedDecodeAttentionFlashVec = true" in decode_pin
    assert "opt110_revived" in native
    assert "opt110_revived" in runner
    assert "byte_size_only_claim" in native
    assert "byte_size_only_claim" in runner
    assert "apply_opt110" not in runner
    assert "Do not revive OPT-110" in fusion or "OPT-110 is not revived" in fusion


def test_production_pin_matches_keep_verdict() -> None:
    fusion = FUSION_PIN.read_text(encoding="utf-8")
    prompt_pin = PROMPT_PIN.read_text(encoding="utf-8")
    decode_pin = DECODE_PIN.read_text(encoding="utf-8")
    if FIXTURE.is_file() and _json(FIXTURE).get("production_kept"):
        assert production_pin() == CANDIDATE
        assert "kSelectedDecodeNormQ81Fusion = true" in fusion
    else:
        assert production_pins_parent() is True
        assert production_pin() == CONTROL
        assert "kSelectedDecodeNormQ81Fusion = false" in fusion
        assert "kSelectedDecodeNormQ81Fusion = true" not in fusion
    assert f'kSelectedAttentionPipelinePath[] = "{SHIPPING_PROMPT}"' in prompt_pin
    assert "kSelectedDecodeAttentionFlashVec = true" in decode_pin
    assert "kSelectedOpt137DenseMma = true" in decode_pin
    assert "kOpt137MmaThreshold = 8192" in decode_pin


def test_screened_in_missing_quality_is_incomplete_not_no_opportunity(
    tmp_path: Path,
) -> None:
    dump = tmp_path / "correctness.json"
    dump.write_text(
        json.dumps({"ok": True, "blocked": False, "task": "OPT-149"}) + "\n",
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
    assert result["shipping_decode_attention"] == SHIPPING_DECODE
    assert result["shipping_decode_norm_q8"] == CONTROL
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
    assert payload["task"] == "OPT-149"
    assert payload["parent"] == PARENT_STACK
    assert payload["candidate"] == CANDIDATE
    quality = payload.get("quality") or {}
    if payload.get("verdict") in {"screened_out", "reject", "blocked"}:
        assert payload["production_kept"] is False
        assert payload["shipping_decode_attention"] == SHIPPING_DECODE
        assert payload["shipping_decode_norm_q8"] == CONTROL
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
