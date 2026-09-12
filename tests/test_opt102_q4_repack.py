"""Host tests for OPT-102 Q4 layout/unpack admission. GPU-free."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tools.opt102_q4_repack import (
    ALIGNED_ID,
    ALIGNMENT_BYTES,
    BRANCHLESS_ID,
    CONFIGS,
    CONTRACT,
    CONTROL_ID,
    FIXTURE,
    ITERATION,
    LAYOUT_VERSION,
    MIN_FFN_SAVING_MS,
    REPORT,
    VERDICT_KEYS,
    decide_independent_verdicts,
    dispatch_ok,
    family_plan,
    load_json,
    parity_catalog,
    replay_command,
)
from tools.run_optimization_task import (
    KernelParityPolicyError,
    describe_plan,
    load_contract,
    loop_product,
    parse_native_observation,
    validate_future_keep_policy,
    workload_for_mode,
)

ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = ROOT / "Makefile"
CUDA_PARITY = ROOT / "cuda/opt102_q4_repack_test.cu"
Q4_PATH = ROOT / "cuda/q4k_decode_path.cuh"
LAYOUT = ROOT / "cuda/q4k_aligned_layout.cuh"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_makefile_and_native_hooks() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-102")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    q4 = Q4_PATH.read_text(encoding="utf-8")
    layout = LAYOUT.read_text(encoding="utf-8")
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert contract["task"] == "OPT-102"
    assert contract["claims_throughput"] is True
    assert contract["control"] == CONTROL_ID
    assert contract["min_ffn_saving_ms"] == MIN_FFN_SAVING_MS
    assert contract["quality_contract_id"] == "opt089_strict"
    assert contract["opt093_factored_not_candidate"] is True
    assert contract["replacement_mandatory_on_keep"] is True
    assert contract["layout_version"] == LAYOUT_VERSION
    assert contract["alignment_bytes"] == ALIGNMENT_BYTES
    assert contract["independent_verdicts"] == list(VERDICT_KEYS)
    assert [row["id"] for row in contract["configurations"]] == [
        CONTROL_ID,
        BRANCHLESS_ID,
        ALIGNED_ID,
    ]
    assert iteration["task"] == "OPT-102"
    assert iteration["diagnostics_make_target"] == "cuda-opt102-diagnostics"
    assert iteration["target"] == "tools/opt102_q4_repack.py"
    assert iteration["case_ids"] == [
        "parity",
        "q4",
        "quality",
        "d128",
        "d2048",
        "prefill-guard",
    ]
    assert "cuda-opt102-diagnostics" in makefile
    assert "qw38-cuda-opt102-q4-repack-test" in makefile
    assert "integer_q8_branchless" in q4
    assert "integer_q8_aligned" in q4
    shipping = str(
        (load_json(FIXTURE) if FIXTURE.is_file() else {}).get("shipping_q4_decode")
        or "integer_q8_late"
    )
    layout_pin = str(
        (load_json(FIXTURE) if FIXTURE.is_file() else {}).get(
            "shipping_q4_device_layout"
        )
        or "raw_gguf"
    )
    assert f'kSelectedQ4DecodePath[] = "{shipping}"' in q4
    assert f'kSelectedQ4DeviceLayout[] = "{layout_pin}"' in q4
    assert "kQ4KAlignedLayoutVersion = 1" in layout
    assert "kQ4KAlignedAlignBytes = 64" in layout
    assert "q4k_scale_min_branchless" in layout
    assert "QW38_OPT102_RESULT=" in CUDA_PARITY.read_text(encoding="utf-8")
    assert "QW38_OPT102_CASE=" in CUDA_PARITY.read_text(encoding="utf-8")
    assert REPORT.parent.as_posix().endswith("opt102-q4-layout-unpack")


def test_iteration_loop_products() -> None:
    iteration = load_contract("OPT-102")
    parity = iteration["workloads"]["parity"]
    q4 = iteration["workloads"]["q4"]
    quality = iteration["workloads"]["quality"]
    d128 = iteration["workloads"]["d128"]
    d2048 = iteration["workloads"]["d2048"]
    prefill = iteration["workloads"]["prefill-guard"]
    assert loop_product(workload_for_mode(parity, "feedback")) == 16
    assert loop_product(workload_for_mode(q4, "feedback")) == 12
    assert loop_product(workload_for_mode(q4, "acceptance")) == 39
    assert loop_product(workload_for_mode(quality, "release")) == 12
    assert loop_product(workload_for_mode(d128, "acceptance")) == 10
    assert loop_product(workload_for_mode(d2048, "acceptance")) == 10
    assert loop_product(workload_for_mode(prefill, "acceptance")) == 2
    screen = family_plan("q4", "feedback")
    assert screen["candidates"] == 3
    assert screen["warmups"] == 1
    assert screen["samples"] == 3
    accept = family_plan("q4", "acceptance")
    assert accept["warmups"] == 3
    assert accept["samples"] == 10
    described = describe_plan("OPT-102", "feedback", iteration, "parity")
    assert "phase=parity" in described
    proof = " ".join(iteration["proof_limit"]).casefold()
    assert "opt-093 factored association is not a candidate" in proof
    assert "replacement mandatory" in proof


def test_parity_catalog_shapes() -> None:
    cases = parity_catalog()
    assert len(cases) == 16
    ids = {row["id"] for row in cases}
    assert "Q4_layout_M1_N1_K256_inverse" in ids
    assert "Q4_layout_M17_N1_K17408_sampled" in ids
    assert "Q4_layout_M17_N1_K256_misaligned" in ids
    assert "Q4_layout_llama_mmvq_diagnostic" in ids
    assert "Q4_layout_occupancy_resources" in ids


def test_three_q4_configs_fixed() -> None:
    assert [row["id"] for row in CONFIGS] == [CONTROL_ID, BRANCHLESS_ID, ALIGNED_ID]
    assert CONFIGS[0]["q4_decode"] == "integer_q8_late"
    assert CONFIGS[0]["q4_device_layout"] == "raw_gguf"
    assert CONFIGS[1]["q4_decode"] == "integer_q8_branchless"
    assert CONFIGS[2]["q4_decode"] == "integer_q8_aligned"
    assert CONFIGS[2]["q4_device_layout"] == "aligned_meta"
    assert all(row["ffn_decode"] == "paired_integer" for row in CONFIGS)
    assert all(int(row["warps_per_row"]) == 4 for row in CONFIGS)
    factored = (ROOT / "cuda/q4k_decode_path.cuh").read_text(encoding="utf-8")
    assert "integer_q8_factored" in factored
    assert "Rejected" in factored
    assert "OPT-102 candidate" in factored


def test_future_keep_policy_rejects_opt074_unadmitted_blocker() -> None:
    load_contract("OPT-102")
    validate_future_keep_policy("OPT-102", load_json(ITERATION))
    with pytest.raises(KernelParityPolicyError, match="opt074_coverage_unadmitted"):
        validate_future_keep_policy(
            "OPT-102",
            {
                "task": "OPT-102",
                "promotion_blockers": ["opt074_coverage_unadmitted"],
            },
        )


def test_independent_verdicts_keep_aligned_when_all_pass() -> None:
    decided = decide_independent_verdicts(
        parity={
            "by_ident": {
                CONTROL_ID: {"kernel_parity_pass": True},
                BRANCHLESS_ID: {"kernel_parity_pass": True},
                ALIGNED_ID: {"kernel_parity_pass": True},
            }
        },
        quality_by_id={
            CONTROL_ID: {"model_quality_pass": True},
            BRANCHLESS_ID: {"model_quality_pass": True},
            ALIGNED_ID: {"model_quality_pass": True},
        },
        performance_by_id={
            CONTROL_ID: {"performance_pass": False, "mean_diff_ms": 0.0},
            BRANCHLESS_ID: {"performance_pass": True, "mean_diff_ms": 0.12},
            ALIGNED_ID: {"performance_pass": True, "mean_diff_ms": 0.21},
        },
        mode="acceptance",
    )
    assert decided["selected_path"] == ALIGNED_ID
    assert decided["production_kept"] is True
    assert decided["shipping_q4_decode"] == "integer_q8_aligned"
    assert decided["shipping_q4_device_layout"] == "aligned_meta"


def test_parse_native_observation_opt102_prefix() -> None:
    observed = parse_native_observation(
        'QW38_OPT102_RESULT={"task":"OPT-102","phase":"parity","success":true}\n'
        'QW38_OPT102_NATIVE_COUNTS={"observed_shapes":16}\n'
    )
    assert observed["task"] == "OPT-102"
    assert observed["phase"] == "parity"
    assert observed["observed_shapes"] == 16


def test_replay_command_includes_layout_selectors() -> None:
    plan = family_plan("q4", "feedback")
    cmd = replay_command(plan, CONFIGS[2], None)
    joined = " ".join(cmd)
    assert "--q4-decode integer_q8_aligned" in joined
    assert "--q4-device-layout aligned_meta" in joined
    assert "--ffn-decode paired_integer" in joined
    assert "--q4-warps 4" in joined


def test_dispatch_ok_matches_expected_variants() -> None:
    expected = CONFIGS[1]
    observed = {
        "gate_variant": expected["expected_gate_variant"],
        "up_variant": expected["expected_up_variant"],
        "down_variant": expected["expected_down_variant"],
        "gate_up_stage_count": 1,
        "down_stage_count": 1,
        "warps_per_row": 4,
        "q4_device_layout": expected["q4_device_layout"],
    }
    assert dispatch_ok(observed, expected) is True
