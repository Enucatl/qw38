"""Host tests for OPT-108 source-faithful NVIDIA llama vector stack."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.opt075_q4_production_admission import load_json
from tools.opt108_llama_vector_stack import (
    CANDIDATE_ID,
    CONTRACT,
    CONTROL_ID,
    FIXTURE,
    ITERATION,
    MIN_SAVING_MS,
    PRIMITIVE_POSITIONS,
    PROVENANCE,
    REPORT,
    family_plan,
    primitive_verdict,
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
ATTN_PATH = ROOT / "cuda/attention_decode_path.cuh"
ADAPTER = ROOT / "cuda/opt108_llama_vector_adapter.cuh"
NATIVE = ROOT / "cuda/opt108_llama_vector_stack_test.cu"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_makefile_and_native_hooks() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-108")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    path = ATTN_PATH.read_text(encoding="utf-8")
    adapter = ADAPTER.read_text(encoding="utf-8")
    native = NATIVE.read_text(encoding="utf-8")
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert PROVENANCE.is_file()
    assert FIXTURE.is_file()
    assert REPORT.is_file()
    assert NATIVE.is_file()
    assert contract["task"] == "OPT-108"
    assert contract["candidate"] == CANDIDATE_ID
    assert contract["control"] == CONTROL_ID
    assert contract["threads"] == 128
    assert contract["cpy_bytes"] == 16
    assert contract["nvidia_float2_v_accum"] is True
    assert contract["amd_half2_v_accum"] is False
    assert contract["f16_cache_migration"] is False
    assert contract["prepared_q_once"] is True
    assert contract["stop_if_primitive_loses"] is True
    assert contract["min_saving_ms_per_token"] == MIN_SAVING_MS
    assert contract["primitive_positions"] == [128, 512, 2048, 4096]
    assert iteration["task"] == "OPT-108"
    assert iteration["diagnostics_make_target"] == "cuda-opt108-diagnostics"
    assert "cuda-opt108-diagnostics" in makefile
    assert "qw38-cuda-opt108-llama-vector-stack-test" in makefile
    assert 'kSelectedDecodeAttentionVec128Path[] = "warp_query"' in path
    assert "kSelectedDecodeAttentionCrossoverThreshold = 1024" in path
    assert "float2" in adapter
    assert "kAmdHalf2VAccum = false" in adapter
    assert "kCpyBytes = 16" in adapter
    assert "kThreads = 128" in adapter
    assert "launch_prepare_decode_query" in adapter
    assert "parallel_blocks_for_kv" in adapter
    assert "flash_attn_ext_vec_nvidia_bf16" in adapter
    assert "AMD/HIP" in adapter or "HIP/AMD" in adapter
    assert "kNvidiaFloat2VAccum = true" in adapter
    assert "llama_vec_nvidia" in native
    assert "128, 512, 2048, 4096" in native or "kPrimitivePositions" in native
    provenance = _json(PROVENANCE)
    assert provenance["llama_revision"] == contract["llama_revision"]
    assert provenance["math_flags"]["nvidia_float2_v_accum"] is True
    assert provenance["math_flags"]["v_dot2_f32_f16_available"] is False
    fixture = load_json(FIXTURE)
    for key in contract["required_fixture_keys"]:
        assert key in fixture


def test_iteration_loop_products() -> None:
    iteration = load_contract("OPT-108")
    eligibility = iteration["workloads"]["eligibility"]
    parity = iteration["workloads"]["parity"]
    primitive = iteration["workloads"]["primitive"]
    d128 = iteration["workloads"]["d128"]
    d2048 = iteration["workloads"]["d2048"]
    assert loop_product(workload_for_mode(eligibility, "feedback")) == 1
    assert loop_product(workload_for_mode(parity, "feedback")) == 2
    assert loop_product(workload_for_mode(primitive, "feedback")) == 32
    assert loop_product(workload_for_mode(primitive, "acceptance")) == 104
    assert loop_product(workload_for_mode(d128, "acceptance")) == 26
    assert loop_product(workload_for_mode(d2048, "acceptance")) == 832
    described = describe_plan("OPT-108", "feedback", iteration, "eligibility")
    assert "phase=eligibility" in described
    proof = " ".join(iteration["proof_limit"]).casefold()
    assert "128 threads" in proof
    assert "16-byte" in proof
    assert "float2" in proof
    assert "half2" in proof
    assert "stop if the source-faithful primitive loses" in proof
    assert "0.50" in proof


def test_primitive_stop_rule() -> None:
    lose = primitive_verdict(
        {
            128: {"mean_diff_ms": 0.01, "faster": True, "ci95_high": 0.02},
            512: {"mean_diff_ms": 0.02, "faster": True},
            2048: {"mean_diff_ms": -0.40, "faster": False, "positive": False},
            4096: {"mean_diff_ms": 0.10, "faster": True},
        }
    )
    assert lose["primitive_win"] is False
    ci_includes_zero = primitive_verdict(
        {
            128: {"mean_diff_ms": -0.001, "faster": False, "ci95_high": 0.0004},
            512: {"mean_diff_ms": 0.006, "faster": True, "positive": True},
            2048: {
                "mean_diff_ms": 0.002,
                "faster": True,
                "positive": False,
                "ci95_low": -0.0006,
            },
            4096: {"mean_diff_ms": 0.011, "faster": True, "positive": True},
        }
    )
    assert ci_includes_zero["primitive_win"] is False
    win = primitive_verdict(
        {
            128: {"mean_diff_ms": 0.01, "faster": True, "ci95_high": 0.04},
            512: {"mean_diff_ms": 0.02, "faster": True},
            2048: {"mean_diff_ms": 0.20, "faster": True, "positive": True},
            4096: {"mean_diff_ms": 0.10, "faster": True},
        }
    )
    assert win["primitive_win"] is True
    assert PRIMITIVE_POSITIONS == (128, 512, 2048, 4096)
    plan = family_plan("primitive", "feedback")
    assert plan["cases"] == 4
    assert plan["candidates"] == 2


def test_tiny_and_production_positions() -> None:
    native = NATIVE.read_text(encoding="utf-8")
    for position in (0, 1, 31, 32, 127, 128, 2047, 2048, 4096):
        assert str(position) in native


def test_future_keep_policy() -> None:
    validate_future_keep_policy("OPT-108", load_contract("OPT-108"))
    contract = _json(CONTRACT)
    assert contract["opt107_hybrid_control"] is True
