"""Host tests for OPT-110 source-faithful llama Q4_K MMVQ adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.opt075_q4_production_admission import load_json
from tools.opt110_llama_q4_adapter import (
    CANDIDATE_ID,
    CONTRACT,
    CONTROL_ID,
    FIXTURE,
    ITERATION,
    MIN_SAVING_MS,
    PRIMITIVE_ROLES,
    PROVENANCE,
    REPORT,
    decide_verdict,
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
Q4_PATH = ROOT / "cuda/q4k_decode_path.cuh"
ADAPTER = ROOT / "cuda/opt110_llama_q4_adapter.cuh"
ADAPTER_CU = ROOT / "cuda/opt110_llama_q4_adapter.cu"
NATIVE = ROOT / "cuda/opt110_llama_q4_adapter_test.cu"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_makefile_and_native_hooks() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-110")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    path = Q4_PATH.read_text(encoding="utf-8")
    adapter = ADAPTER.read_text(encoding="utf-8")
    adapter_cu = ADAPTER_CU.read_text(encoding="utf-8")
    native = NATIVE.read_text(encoding="utf-8")
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert PROVENANCE.is_file()
    assert FIXTURE.is_file()
    assert REPORT.is_file()
    assert NATIVE.is_file()
    assert contract["task"] == "OPT-110"
    assert contract["candidate"] == CANDIDATE_ID
    assert contract["control"] == CONTROL_ID
    assert contract["control_staging"] == "Q8Block"
    assert contract["candidate_staging"] == "block_q8_1"
    assert contract["encodings_interchangeable"] is False
    assert contract["nwarps"] == 4
    assert contract["vdr"] == 2
    assert contract["stop_if_primitive_loses"] is True
    assert contract["opt093_factored_not_candidate"] is True
    assert contract["opt102_aligned_not_candidate"] is True
    assert contract["no_warp_sweep"] is True
    assert contract["min_saving_ms_per_token"] == MIN_SAVING_MS
    assert iteration["task"] == "OPT-110"
    assert iteration["diagnostics_make_target"] == "cuda-opt110-diagnostics"
    assert "cuda-opt110-diagnostics" in makefile
    assert "qw38-cuda-opt110-llama-q4-adapter-test" in makefile
    assert "--use_fast_math" in makefile
    assert 'kSelectedQ4DecodePath[] = "integer_q8_late"' in path
    assert "llama_q4k_mmvq" in path
    assert "kLegalQ4DecodePathLlamaMmvq" in path
    assert "BlockQ81" in adapter
    assert "kNwarps = 4" in adapter
    assert "kVdr = 2" in adapter
    assert "ProductionShape<kDownRows, kDownCols>" in adapter
    assert "vec_dot_q4_K_q8_1" in adapter_cu
    assert "mul_mat_vec_q4_k" in adapter_cu
    assert "quantize_bf16_block_q8_1" in adapter_cu
    assert "llama_q4k_mmvq" in native
    assert "integer_q8_late" in native
    assert "OPT110_LLAMA_OBJECT" in makefile
    assert "5120" in native and "17408" in native
    provenance = _json(PROVENANCE)
    assert provenance["llama_revision"] == contract["llama_revision"]
    assert provenance["license"] == "MIT"
    fixture = load_json(FIXTURE)
    for key in contract["required_fixture_keys"]:
        assert key in fixture


def test_iteration_loop_products() -> None:
    iteration = load_contract("OPT-110")
    parity = iteration["workloads"]["parity"]
    primitive = iteration["workloads"]["primitive"]
    assert loop_product(workload_for_mode(parity, "feedback")) == 2
    assert loop_product(workload_for_mode(primitive, "feedback")) == 16
    assert loop_product(workload_for_mode(primitive, "acceptance")) == 52
    described = describe_plan("OPT-110", "feedback", iteration, "parity")
    assert "phase=parity" in described
    proof = " ".join(iteration["proof_limit"]).casefold()
    assert "block_q8_1" in proof
    assert "integer_q8_late" in proof
    assert "stop if the source-faithful complete primitive loses" in proof
    assert "0.50" in proof
    assert "5120x17408" in proof or "5120×17408" in proof


def test_primitive_stop_rule() -> None:
    lose = primitive_verdict(
        {
            "down": {"mean_diff_ms": -0.40, "faster": False, "positive": False},
            "gate_up": {"mean_diff_ms": 0.10, "faster": True, "positive": True},
        }
    )
    assert lose["primitive_win"] is False
    ci_includes_zero = primitive_verdict(
        {
            "down": {
                "mean_diff_ms": 0.002,
                "faster": True,
                "positive": False,
                "ci95_low": -0.0006,
                "control_mean_ms": 1.0,
                "candidate_mean_ms": 0.998,
            },
            "gate_up": {
                "mean_diff_ms": 0.20,
                "faster": True,
                "positive": True,
                "control_mean_ms": 2.0,
                "candidate_mean_ms": 1.8,
            },
        }
    )
    assert ci_includes_zero["primitive_win"] is False
    win = primitive_verdict(
        {
            "down": {
                "mean_diff_ms": 0.20,
                "faster": True,
                "positive": True,
                "control_mean_ms": 1.2,
                "candidate_mean_ms": 1.0,
            },
            "gate_up": {
                "mean_diff_ms": 0.30,
                "faster": True,
                "positive": True,
                "control_mean_ms": 2.0,
                "candidate_mean_ms": 1.7,
            },
        }
    )
    assert win["primitive_win"] is True
    assert PRIMITIVE_ROLES == ("down", "gate_up")
    plan = family_plan("primitive", "feedback")
    assert plan["cases"] == 2
    assert plan["candidates"] == 2


def test_production_shapes_and_type_split() -> None:
    native = NATIVE.read_text(encoding="utf-8")
    adapter = ADAPTER.read_text(encoding="utf-8")
    assert "Q8Block" in native
    assert "BlockQ81" in native
    assert "reinterpret_forbidden" in native or "type_isolation" in native
    assert "kDownRows = 5120" in adapter
    assert "kGateCols = 5120" in adapter
    assert "sizeof(BlockQ81) == 36" in adapter
    assert "sizeof(BlockQ4K) == 144" in adapter


def test_future_keep_policy() -> None:
    validate_future_keep_policy("OPT-110", load_contract("OPT-110"))
    contract = _json(CONTRACT)
    assert contract["stop_if_primitive_loses"] is True
    performance_ok = {
        "ffn_keep": True,
        "skipped": False,
    }
    d_ok = {"throughput_improved": True, "skipped": False}
    prefill_ok = {"pass": True, "skipped": False}
    blocked = decide_verdict(
        primitive={"primitive_win": True},
        parity={"pass": True},
        quality={"model_quality_pass": False, "skipped": True},
        q4=performance_ok,
        d128=d_ok,
        d2048=d_ok,
        prefill=prefill_ok,
        mode="acceptance",
    )
    assert blocked["verdict"] == "quality_blocked"
    assert blocked["keep"] is False
    assert blocked["production_kept"] is False
