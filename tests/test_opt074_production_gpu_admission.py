"""Host tests for OPT-074 production-shape llama GPU numerical admission."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.opt074_production_gpu_admission import (
    CONSUMERS,
    CONTRACT,
    EXPORT_BIN,
    FIXTURE,
    ITERATION,
    PRODUCER,
    REPORT,
    STAGING_LLAMA_SUM_X,
    STAGING_QUARTZ_SUM_Q,
    SUM_VARIANTS,
    AdmissionError,
    cases_for_phase,
    family_id,
    freeze_family_ceilings,
    frozen_cases,
    held_out_validates,
    historical_opt059_preserved,
    layer_kind,
    load_json,
    reference_cache_key,
    require_authority,
    require_dispatch,
    require_finite,
    require_inventory_match,
    require_rows,
    run_phase,
    sample_rows,
    tensor_name_for_case,
)
from tools.production_numerics import GGUF_SHA, LLAMA_REV
from tools.production_numerics_v2 import freeze_case_ids
from tools.run_optimization_task import describe_plan, load_contract, loop_product

ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = ROOT / "Makefile"
EXPORT = ROOT / "tools/llama_authority/projection_export.cpp"
CAPTURE = ROOT / "cuda/opt043_activation_capture_test.cu"
SCHEDULER = ROOT / "cuda/full_scheduler.cu"
OPT059_FIXTURE = ROOT / "fixtures/opt059_gpu_numerics.json"
OPT059_MANIFEST = ROOT / "pins/opt059_admission_manifest.json"


def test_contracts_iteration_and_native_hooks() -> None:
    contract = load_json(CONTRACT)
    iteration = load_contract("OPT-074")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    export = EXPORT.read_text(encoding="utf-8")
    capture = CAPTURE.read_text(encoding="utf-8")
    scheduler = SCHEDULER.read_text(encoding="utf-8")
    assert contract["task"] == "OPT-074"
    assert contract["claims_throughput"] is False
    assert contract["claims_performance_improvement"] is False
    assert contract["llama_revision"] == LLAMA_REV
    assert contract["gguf_sha256"] == GGUF_SHA
    assert contract["dispatch"]["row_limit_evidence"] is False
    assert contract["dispatch"]["n"] == 1
    assert contract["dispatch"]["full_m"] is True
    assert contract["opt059_freeze"]["held_out_cannot_enlarge"] is True
    assert contract["q8_1"]["adapter"] == "private_adapter_quantize_q8_1_sum_x"
    assert iteration["task"] == "OPT-074"
    assert iteration["claims_throughput"] is False
    assert iteration["host_only"] is False
    assert iteration["skip_gpu_setup"] is False
    assert iteration["diagnostics_make_target"] == "cuda-opt074-diagnostics"
    assert iteration["target"] == "build/qw38-cuda-opt059-numerics-test"
    assert iteration["modes"]["release"]["historical_oracles"] is False
    assert iteration["case_ids"] == ["q4-gate-up", "q4-down", "q4", "q8-q6"]
    assert iteration["workloads"]["q4-gate-up"]["runner"] == "host"
    assert iteration["workloads"]["q4"]["load_model"] == 1
    assert "cuda-opt074-diagnostics" in makefile
    assert "qw38-cuda-opt059-numerics-test" in makefile
    assert "qw38-cuda-opt043-activation-capture-test" in makefile
    assert "--evidence" in export
    assert "cudaDeviceSynchronize" in export
    assert "--input-bf16" in export
    assert "private_adapter_quantize_q8_1_sum_x" in export
    assert "evidence forbids --row-limit" in export
    assert "ggml_cpy" in export
    assert "--dump-dir" in capture
    assert "gdn_or_attn_mix_6144" in capture
    assert "maybe_capture_mix" in scheduler
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert FIXTURE.is_file()
    assert REPORT.is_file()
    assert str(EXPORT_BIN).endswith("qw38-llama-projection-export")


def test_opt059_case_ids_and_historical_missing_evidence() -> None:
    frozen = freeze_case_ids()
    contract = load_json(CONTRACT)
    assert contract["case_ids"]["calibration"] == [
        row["id"] for row in frozen["calibration"]
    ]
    assert contract["case_ids"]["held_out"] == [row["id"] for row in frozen["held_out"]]
    assert (
        contract["opt059_freeze"]["id_manifest_sha256"] == frozen["id_manifest_sha256"]
    )
    assert frozen["calibration_layers"] == [0, 3, 31, 32]
    assert frozen["held_out_layers"] == [62, 63]
    assert frozen["calibration_token_positions"] == [128, 512]
    assert frozen["held_out_token_positions"] == [2048, 4095]
    assert historical_opt059_preserved() is True
    opt059 = json.loads(OPT059_FIXTURE.read_text(encoding="utf-8"))
    manifest = json.loads(OPT059_MANIFEST.read_text(encoding="utf-8"))
    assert opt059["llama_gpu_sidecars"]["status"] == (
        "probe_exported_production_k_unadmitted"
    )
    assert any(int(entry.get("columns", -1)) == 0 for entry in manifest["entries"])
    assert any(entry.get("v2_admitted") is False for entry in manifest["entries"])


def test_attention_q8_mixer_is_explicit_absent_not_substituted() -> None:
    frozen = frozen_cases()
    attention = [
        row
        for row in frozen["calibration"] + frozen["held_out"]
        if row["role"] in {"mixer_qkv", "mixer_output"}
        and layer_kind(row["layer"]) == "attention"
    ]
    assert attention
    for row in attention:
        assert tensor_name_for_case(row) is None
        assert row["layer"] % 4 == 3
    gdn = next(
        row
        for row in frozen["calibration"]
        if row["role"] == "mixer_qkv" and row["layer"] == 0
    )
    assert tensor_name_for_case(gdn) == "blk.0.attn_qkv.weight"
    vocab = next(row for row in frozen["calibration"] if row["role"] == "vocab")
    assert tensor_name_for_case(vocab) == "output.weight"
    assert family_id(gdn) == "q8_mixer_k5120"


def test_feedback_selects_one_layer_family_case() -> None:
    gate = cases_for_phase("q4-gate-up", feedback=True)
    down = cases_for_phase("q4-down", feedback=True)
    assert len(gate) == 1
    assert gate[0]["id"] == "q4_gate_up_k5120_L0_t128"
    assert len(down) == 1
    assert down[0]["id"] == "q4_down_k17408_L0_t128"
    q4 = cases_for_phase("q4", feedback=False)
    assert len(q4) == 24
    assert all(row["role"] in {"ffn_gate_up", "ffn_down"} for row in q4)


def test_host_phase_validates_without_gpu() -> None:
    result = run_phase("q4-gate-up", skip_gpu=True)
    assert result["success"] is True
    assert result["status"] == "host_validated"
    assert result["claims_throughput"] is False
    assert result["row_limit_used_for_evidence"] is False
    assert result["cases"][0]["tensor"] == "blk.0.ffn_gate.weight"
    assert result["cases"][0]["sampled_rows"] == sample_rows(17408)


def test_mutate_model_tensor_shape_and_revision() -> None:
    case = cases_for_phase("q4-gate-up", feedback=True)[0]
    tensor = {
        "name": "blk.0.ffn_gate.weight",
        "shape": [5120, 17408],
        "dtype": "Q4_K",
    }
    require_inventory_match(case, tensor)
    with pytest.raises(AdmissionError, match="shape mismatch"):
        require_inventory_match(case, {**tensor, "shape": [5120, 16]})
    with pytest.raises(AdmissionError, match="shape mismatch"):
        require_inventory_match(case, {**tensor, "shape": [256, 17408]})
    with pytest.raises(AdmissionError, match="dtype mismatch"):
        require_inventory_match(case, {**tensor, "dtype": "Q8_0"})
    good = {
        "llama_revision": LLAMA_REV,
        "gguf_sha256": GGUF_SHA,
        "producer": PRODUCER,
        "dispatch_family": "mmvq",
        "n": 1,
        "full_m": 17408,
        "ne1": 17408,
        "ne0": 5120,
        "row_limit": 0,
        "not_cpu_replica": True,
    }
    require_authority(good)
    require_dispatch(good, rows=17408, columns=5120)
    with pytest.raises(AdmissionError, match="authority revision"):
        require_authority({**good, "llama_revision": "deadbeef"})
    with pytest.raises(AdmissionError, match="GGUF identity"):
        require_authority({**good, "gguf_sha256": "0" * 64})
    with pytest.raises(AdmissionError, match="producer tag"):
        require_authority({**good, "producer": "other"})
    with pytest.raises(AdmissionError, match="row-limit"):
        require_dispatch({**good, "row_limit": 16}, rows=17408, columns=5120)
    with pytest.raises(AdmissionError, match="N must be 1"):
        require_dispatch({**good, "n": 8}, rows=17408, columns=5120)
    with pytest.raises(AdmissionError, match="dispatch family"):
        require_dispatch({**good, "dispatch_family": "mmq"}, rows=17408, columns=5120)
    with pytest.raises(AdmissionError, match="full M"):
        require_dispatch({**good, "full_m": 16, "ne1": 16}, rows=17408, columns=5120)
    with pytest.raises(AdmissionError, match="wrong K"):
        require_dispatch({**good, "ne0": 256}, rows=17408, columns=5120)


def test_missing_rows_nonfinite_and_cache_key() -> None:
    rows = sample_rows(248320)
    assert rows == list(range(8)) + list(range(248320 - 8, 248320))
    require_rows(rows, rows)
    with pytest.raises(AdmissionError, match="missing rows"):
        require_rows(rows[:-1], rows)
    require_finite([0.0, 1.0], label="ok")
    with pytest.raises(AdmissionError, match="nonfinite llama"):
        require_finite([0.0, float("nan")], label="llama")
    key = reference_cache_key(
        tensor="blk.0.ffn_gate.weight",
        columns=5120,
        rows=17408,
        input_sha="abc",
        fused=False,
    )
    other = reference_cache_key(
        tensor="blk.0.ffn_gate.weight",
        columns=5120,
        rows=16,
        input_sha="abc",
        fused=False,
    )
    mutated_producer = reference_cache_key(
        tensor="blk.0.ffn_gate.weight",
        columns=5120,
        rows=17408,
        input_sha="abc",
        fused=False,
        producer="mutated",
    )
    assert key != other
    assert key != mutated_producer
    assert len(key) == 64


def test_sum_variant_consumers_are_distinct() -> None:
    assert SUM_VARIANTS["sum_q"] == STAGING_QUARTZ_SUM_Q
    assert SUM_VARIANTS["sum_x"] == STAGING_LLAMA_SUM_X
    assert CONSUMERS[STAGING_QUARTZ_SUM_Q] != CONSUMERS[STAGING_LLAMA_SUM_X]
    assert CONSUMERS[STAGING_QUARTZ_SUM_Q] == "quartz_q8_1_integer_dot"
    assert CONSUMERS[STAGING_LLAMA_SUM_X] == "llama_gpu_q8_1_mmvq"
    assert "packed_fp32_q4_mmv" in CONSUMERS.values()


def test_held_out_cannot_enlarge_calibration_ceiling() -> None:
    calib = [{"max_abs": 1e-4, "rms": 1e-5, "one_minus_cosine": 1e-8, "nonfinite": 0}]
    ceilings = freeze_family_ceilings(calib)
    assert ceilings["frozen_before_candidates"] is True
    assert ceilings["held_out_cannot_enlarge"] is True
    assert held_out_validates(
        ceilings,
        {"max_abs": 1e-4, "rms": 1e-5, "one_minus_cosine": 1e-8, "nonfinite": 0},
    )
    assert not held_out_validates(
        ceilings, {"max_abs": 1.0, "rms": 1.0, "one_minus_cosine": 1.0, "nonfinite": 0}
    )
    assert ceilings["abs"]["ceiling"] >= calib[0]["max_abs"]


def test_iteration_loop_products_and_dry_run_phases() -> None:
    iteration = load_contract("OPT-074")
    assert loop_product(iteration["workloads"]["q4-gate-up"]) == 1
    assert loop_product(iteration["workloads"]["q4-down"]) == 1
    assert loop_product(iteration["workloads"]["q4"]) == 24
    assert loop_product(iteration["workloads"]["q8-q6"]) == 36
    plan = describe_plan("OPT-074", "feedback", iteration, None)
    assert "case_ids=q4-gate-up,q4-down,q4,q8-q6" in plan
    assert "historical_oracles=none" in plan
    assert "host_only=false" in plan
    gate = describe_plan("OPT-074", "feedback", iteration, "q4-gate-up")
    assert "phase=q4-gate-up" in gate
    assert "product=1" in gate
    down = describe_plan("OPT-074", "feedback", iteration, "q4-down")
    assert "phase=q4-down" in down
    q4 = describe_plan("OPT-074", "acceptance", iteration, "q4")
    assert "phase=q4" in q4
    assert "product=24" in q4
    mixer = describe_plan("OPT-074", "acceptance", iteration, "q8-q6")
    assert "phase=q8-q6" in mixer
    assert "product=36" in mixer
    release = describe_plan("OPT-074", "release", iteration, None)
    assert "historical_oracles=none" in release or "historical_oracles=" in release


def test_exporter_evidence_path_never_uses_row_limit() -> None:
    text = EXPORT.read_text(encoding="utf-8")
    assert "if (evidence && row_limit > 0)" in text
    assert "full M N=1 is required" in text
    assert "QW38_OPT074_LLAMA_GPU_EXPORT=" in text
    makefile = MAKEFILE.read_text(encoding="utf-8")
    assert (
        "cuda-opt074-diagnostics: $(BUILD_DIR)/qw38-cuda-opt059-numerics-test "
        "$(BUILD_DIR)/qw38-cuda-opt043-activation-capture-test"
        in makefile
        or "cuda-opt074-diagnostics:" in makefile
    )
