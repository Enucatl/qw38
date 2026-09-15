"""Host tests for OPT-150 attention dataflow diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tools.opt150_attention_dataflow import (
    ATTENTION_LAYERS,
    CAPACITY,
    CONTRACT,
    CROSSOVER,
    DISPATCH_POSITIONS,
    EVIDENCE,
    FIXTURE,
    HISTORICAL_CORRECTIONS,
    ITERATION,
    LAYERS,
    LLAMA_REV,
    MATCHED_EVALS,
    MMA_THRESHOLD,
    NATIVE,
    NUMERICAL_POSITIONS,
    PARENT,
    PHASES,
    REPORT,
    REQUIRED_FIXTURE_KEYS,
    SCREEN_PAIRS,
    SCREEN_WARMUPS,
    SELECTOR,
    TIMING_SHAPES,
    VERIFIED_MAX,
    build_dispatch_table,
    family_plan,
    inspect_dataflow,
    llama_padded_nkv,
    llama_selected_kernel,
    persist_time_reconstruction,
    quartz_n_parts,
    quartz_path,
    quartz_topology,
    reconstruct_family_time,
    validate_fixture,
)
from tools.run_optimization_task import (
    describe_plan,
    load_contract,
    loop_product,
    validate_future_keep_policy,
    workload_for_mode,
)

ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = ROOT / "Makefile"
NATIVE_SRC = ROOT / "cuda/opt150_attention_dataflow_test.cu"
PIN = ROOT / "cuda/attention_decode_path.cuh"
MMA_SRC = ROOT / "cuda/opt137_dense_mma_decode.cuh"
RUNNER = ROOT / "tools/run_optimization_task.py"
TOOL = ROOT / "tools/opt150_attention_dataflow.py"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_makefile_and_registration() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-150")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    native = NATIVE_SRC.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    tool = TOOL.read_text(encoding="utf-8")
    pin = PIN.read_text(encoding="utf-8")
    mma = MMA_SRC.read_text(encoding="utf-8")
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert NATIVE_SRC.is_file()
    assert contract["task"] == "OPT-150"
    assert contract["claims_throughput"] is False
    assert contract["claims_performance_improvement"] is False
    assert contract["production_selector_change"] is False
    assert contract["require_candidate_nll"] is False
    assert contract["llama_revision"] == LLAMA_REV
    assert contract["selected_execution_graph_path"] == SELECTOR
    assert contract["parent"] == PARENT
    assert contract["crossover_threshold"] == CROSSOVER
    assert contract["verified_max"] == VERIFIED_MAX
    assert contract["mma_threshold"] == MMA_THRESHOLD
    assert contract["allocated_capacity"] == CAPACITY
    assert contract["dispatch_positions"] == list(DISPATCH_POSITIONS)
    assert contract["timing_shapes"] == list(TIMING_SHAPES)
    assert contract["attention_layers"] == list(LAYERS)
    assert contract["matched_decode_evals"] == MATCHED_EVALS
    assert iteration["diagnostics_make_target"] == "cuda-opt150-diagnostics"
    assert iteration["gpu_lock"] == "build/optimization-runs/qw38-gpu.lock"
    assert iteration["claims_throughput"] is False
    assert iteration["performance_admission"]["instrumentation_only"] is True
    assert "cuda-opt150-diagnostics" in makefile
    assert "qw38-cuda-opt150-attention-dataflow-test" in makefile
    assert Path(NATIVE).name in makefile
    assert "qw38-cuda-opt136-matched-decode-profile-test" in makefile
    assert "identity|correctness|replay" in native
    assert "QW38_OPT150_ATTENTION_DATAFLOW_RESULT=" in native
    assert "QW38_OPT150_NATIVE_COUNTS=" in native
    assert "QW38_OPT150_ATTENTION_DATAFLOW_RESULT=" in runner
    assert "0.0F * mma_scores[0]" in mma
    assert "kSelectedDecodeAttentionFlashVec = true" in pin
    assert "kSelectedOpt137DenseMma = true" in pin
    assert "kOpt137MmaThreshold = 8192" in pin
    assert PHASES == ("smoke", "correctness", "replay", "matched", "report")
    assert "claims_throughput=false" in tool
    assert ATTENTION_LAYERS == 16
    validate_future_keep_policy("OPT-150", iteration)
    for key in REQUIRED_FIXTURE_KEYS:
        assert key in contract["required_fixture_keys"]


def test_iteration_loop_products() -> None:
    iteration = load_contract("OPT-150")
    smoke = iteration["workloads"]["smoke"]
    correctness = iteration["workloads"]["correctness"]
    replay = iteration["workloads"]["replay"]
    matched = iteration["workloads"]["matched"]
    report = iteration["workloads"]["report"]
    assert loop_product(workload_for_mode(smoke, "feedback")) == 13
    assert loop_product(workload_for_mode(correctness, "feedback")) == 22
    assert loop_product(workload_for_mode(replay, "feedback")) == 24
    assert loop_product(workload_for_mode(matched, "feedback")) == 768
    assert loop_product(workload_for_mode(report, "acceptance")) == 1
    described = describe_plan("OPT-150", "feedback", iteration, "smoke")
    assert "phase=smoke" in described
    assert "loop_product=13" in family_plan("feedback", "smoke")
    proof = " ".join(iteration["proof_limit"]).casefold()
    assert "claims_throughput=false" in proof
    assert replay["warmups"] == SCREEN_WARMUPS
    assert replay["samples"] == SCREEN_PAIRS
    assert matched["tokens"] == MATCHED_EVALS
    assert NUMERICAL_POSITIONS == (128, 1024, 8192)


def test_dispatch_identities_and_llama_padding() -> None:
    assert quartz_path(0) == "warp_query"
    assert quartz_path(128) == "warp_query"
    assert quartz_path(1023) == "warp_query"
    assert quartz_path(1024) == "decode_attention_flash_vec_v1"
    assert quartz_path(4096) == "decode_attention_flash_vec_v1"
    assert quartz_path(4097) == "warp_query"
    assert quartz_path(6144) == "warp_query"
    assert quartz_path(8191) == "warp_query"
    assert quartz_path(8192) == "dense_bf16_tile_f16_mma_decode_v1"
    assert quartz_path(32768) == "dense_bf16_tile_f16_mma_decode_v1"
    assert quartz_topology(1023) == 0
    assert quartz_topology(1024) == 1
    assert quartz_topology(4097) == 0
    assert quartz_topology(8192) == 2
    assert quartz_topology(32768) == 3
    assert quartz_n_parts(128) == 16
    assert quartz_n_parts(2048) == 16
    assert llama_padded_nkv(1) == 256
    assert llama_padded_nkv(129) == 256
    assert llama_padded_nkv(1024) == 1024
    assert llama_padded_nkv(1025) == 1280
    assert llama_padded_nkv(7936) == 7936
    assert llama_padded_nkv(7937) == 8192
    assert llama_padded_nkv(8193) == 8448
    assert llama_padded_nkv(32769) == 33024
    assert llama_selected_kernel(7936) == "flash_attn_ext_vec<256,1>"
    assert llama_selected_kernel(8192) == "fattn-mma-f16_ncols1=1_ncols2=8"
    rows = {int(row["position"]): row for row in build_dispatch_table()}
    assert set(rows) == set(DISPATCH_POSITIONS)
    assert rows[1024]["graph_boundary_crossing"] == "topology_0_to_1"
    assert rows[4097]["graph_boundary_crossing"] == "topology_1_to_0"
    assert rows[8192]["graph_boundary_crossing"] == "topology_0_to_2"
    assert rows[32768]["graph_boundary_crossing"] == "topology_2_to_3"
    assert rows[7935]["llama_vec_is_selected"] is True
    assert rows[7936]["llama_vec_is_selected"] is False
    assert rows[7936]["llama_selected_kernel"] == ("fattn-mma-f16_ncols1=1_ncols2=8")
    assert rows[8191]["quartz_path"] == "warp_query"
    assert rows[8191]["llama_selected_kernel"] == ("fattn-mma-f16_ncols1=1_ncols2=8")
    dataflow = inspect_dataflow()
    assert dataflow["quartz_mma"]["zero_weight_mma_in_source"] is True
    assert dataflow["quartz_mma"]["pv_mma"] is False
    assert dataflow["zero_weight_mma_is_source_fact_not_cost"] is True


def test_historical_corrections_and_reconstruction() -> None:
    blob = json.dumps(HISTORICAL_CORRECTIONS)
    assert "0.050" in blob
    assert "0.243909" in blob
    assert "adapter" in blob.casefold()
    assert "whole-engine" in blob.casefold() or "decode-only" in blob.casefold()
    evidence_path = EVIDENCE / "time-reconstruction.json"
    before = evidence_path.read_bytes() if evidence_path.is_file() else None
    reconstructed = reconstruct_family_time(
        {
            "shapes": [
                {
                    "shape": "D2048",
                    "position": 2048,
                    "quartz_round_ms": [1.0, 1.2, 0.8],
                    "llama_round_ms": [0.5, 0.5, 0.5],
                    "adapter_round_ms": [0.1, 0.1, 0.1],
                    "llama_selected_kernel": "flash_attn_ext_vec<256,1>",
                }
            ]
        }
    )
    chosen = reconstructed["chosen"]
    assert chosen["quartz_mean_ms"] == 1.0
    assert chosen["llama_native_kernel_mean_ms"] == pytest.approx(0.4)
    assert chosen["adapter_charged_separately"] is True
    assert chosen["ratio"]["denominator"] == (
        "llama_complete_family_ms_including_adapter"
    )
    after = evidence_path.read_bytes() if evidence_path.is_file() else None
    assert after == before


def test_measured_d2048_reconstruction_and_evidence() -> None:
    fixture = _json(FIXTURE)
    measured = reconstruct_family_time(fixture["replay"])
    chosen = measured["chosen"]
    assert chosen["shape"] == "D2048"
    assert chosen["quartz_mean_ms"] == pytest.approx(0.6303786633333334)
    assert chosen["llama_enclosing_mean_ms"] == pytest.approx(0.0365333334)
    assert chosen["adapter_mean_ms"] == pytest.approx(0.0135253333)
    assert chosen["llama_native_kernel_mean_ms"] == pytest.approx(0.0230080001)
    assert chosen["ratio"]["quartz_over_llama_family"] == pytest.approx(17.254890388221)
    evidence = _json(EVIDENCE / "time-reconstruction.json")
    assert evidence["shape"] == "D2048"
    assert evidence["quartz_mean_ms"] == pytest.approx(0.6303786633333334)
    assert evidence["quartz_mean_ms"] != 1.0
    assert evidence["llama_enclosing_mean_ms"] != 0.5
    report = REPORT.read_text(encoding="utf-8")
    assert "## Independent time reconstruction" in report
    assert str(evidence["quartz_mean_ms"]) in report or (
        json.dumps(evidence["quartz_mean_ms"]) in report
    )


def test_persist_rejects_unit_test_placeholders(tmp_path: Path) -> None:
    persisted = persist_time_reconstruction(
        {
            "ok": True,
            "chosen": {
                "shape": "D2048",
                "quartz_mean_ms": 1.0,
                "llama_enclosing_mean_ms": 0.5,
            },
            "all": [],
        },
        run_dir=tmp_path,
    )
    chosen = persisted["chosen"]
    assert chosen["quartz_mean_ms"] != 1.0
    assert chosen["quartz_mean_ms"] == pytest.approx(0.6303786633333334)
    written = _json(tmp_path / "time-reconstruction.json")
    assert written["quartz_mean_ms"] == pytest.approx(0.6303786633333334)
    evidence = _json(EVIDENCE / "time-reconstruction.json")
    assert evidence["quartz_mean_ms"] == pytest.approx(0.6303786633333334)


def test_fixture_and_no_production_keep() -> None:
    fixture = _json(FIXTURE)
    validate_fixture(fixture)
    for key in REQUIRED_FIXTURE_KEYS:
        assert key in fixture
    assert fixture["claims_throughput"] is False
    assert fixture["production_kept"] is True
    assert fixture["selected_execution_graph_path"] == SELECTOR
    assert REPORT.parent.as_posix().endswith("opt150-attention-dataflow")
    pin = PIN.read_text(encoding="utf-8")
    assert "kSelectedDecodeAttentionFlashVec = true" in pin
    assert "kSelectedOpt137DenseMma = true" in pin
    assert "Do not revive OPT-130" in pin or "no OPT-130" in pin.lower()
