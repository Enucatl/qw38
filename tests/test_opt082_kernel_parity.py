"""Host tests for OPT-082 CUDA kernel-parity suite. GPU-free."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tools.kernel_parity import (
    ASSOCIATION,
    HEADER,
    SAME_MATH,
    STAGING_SUM_Q,
    STAGING_SUM_X,
    KernelParityError,
    abs_tolerance,
    check_close,
)
from tools.opt082_kernel_parity import (
    ITERATION,
    K_CORE,
    K_PROD,
    MAKEFILE,
    MMQ_N,
    MMV_M,
    NATIVE_REL,
    Q4_MMQ,
    Q4_MMV,
    Q8_LAYOUTS,
    catalog,
    evaluate_host_cases,
    load_json,
    q8_1_typed_reject,
    required_coverage,
    run_phase,
    source_uses_canonical_checker,
)
from tools.run_optimization_task import (
    describe_plan,
    load_contract,
    loop_product,
    parse_native_observation,
)

ROOT = Path(__file__).resolve().parents[1]
CUDA = ROOT / "cuda/opt082_kernel_parity_test.cu"
QUANT = ROOT / "cuda/quant_mmv_test.cu"
PROMPT = ROOT / "cuda/prompt_mmq_test.cu"
TILE = ROOT / "cuda/ffn_tile_ab_test.cu"
WRAPPERS = (QUANT, PROMPT, TILE)


def test_contracts_makefile_and_native_hooks() -> None:
    iteration = load_contract("OPT-082")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    assert ITERATION.is_file()
    assert HEADER.is_file()
    assert CUDA.is_file()
    assert iteration["task"] == "OPT-082"
    assert iteration["claims_throughput"] is False
    assert iteration["diagnostics_make_target"] == "cuda-opt082-diagnostics"
    assert iteration["target"] == NATIVE_REL
    assert iteration["case_ids"] == ["q4", "q8-q6", "parity"]
    assert iteration["modes"]["release"]["historical_oracles"] is False
    assert "cuda-opt082-diagnostics" in makefile
    assert "qw38-cuda-opt082-kernel-parity-test" in makefile
    assert loop_product(iteration["workloads"]["q4"]) == len(catalog("q4"))
    assert loop_product(iteration["workloads"]["q8-q6"]) == len(catalog("q8-q6"))
    assert loop_product(iteration["workloads"]["parity"]) == len(catalog("parity"))


def test_catalog_covers_required_product() -> None:
    coverage = required_coverage()
    cases = catalog("parity")
    ids = {row["id"] for row in cases}
    assert coverage["q4_mmv"] == list(Q4_MMV)
    assert coverage["q4_mmq"] == list(Q4_MMQ)
    assert coverage["q8"] == list(Q8_LAYOUTS)
    assert coverage["has_fused"] is True
    assert coverage["has_down"] is True
    assert coverage["has_same_math"] is True
    assert coverage["has_staging"] is True
    for candidate in (*Q4_MMV, *Q4_MMQ, *Q8_LAYOUTS, "integer_q8_1"):
        assert any(row["candidate"] == candidate for row in cases), candidate
    for m in MMV_M:
        assert any(row["M"] == m and row["op"] == "mmv" for row in cases)
    for n in MMQ_N:
        assert any(row["N"] == n and row["op"] == "mmq" for row in cases)
    for k in K_CORE:
        assert any(row["K"] == k for row in cases)
    for k in K_PROD:
        assert any(row["K"] == k and row["candidate"] == "packed" for row in cases)
    assert any(
        row["pattern"] == "unaligned" and row["expect_fallback"] for row in cases
    )
    assert any(row["pattern"] == "aligned" and row["M"] == 128 for row in cases)
    assert "Q4_K_mmq_fma_async_x_M128_N128_K256_aligned_assoc" in ids
    assert "Q4_K_fused_vs_independent_M3_N1_K256_random_same" in ids


def test_association_scales_follow_k_and_family() -> None:
    q4 = check_close(
        [11.0],
        [10.0],
        family="Q4_K",
        columns=256,
        class_name=ASSOCIATION,
    )
    q8 = check_close(
        [11.0],
        [10.0],
        family="Q8_0",
        columns=256,
        class_name=ASSOCIATION,
    )
    q6 = check_close(
        [11.0],
        [10.0],
        family="Q6_K",
        columns=256,
        class_name=ASSOCIATION,
    )
    assert abs_tolerance(0.20, 256) == pytest.approx(0.20 * math.sqrt(256))
    assert q4["abs_tol"] == q6["abs_tol"] == pytest.approx(3.2)
    assert q8["abs_tol"] == pytest.approx(0.8)
    assert q4["pass"] is True
    assert q6["pass"] is True
    assert q8["pass"] is False
    mutated_k = check_close(
        [11.0],
        [10.0],
        family="Q4_K",
        columns=1024,
        class_name=ASSOCIATION,
    )
    assert mutated_k["abs_tol"] == pytest.approx(6.4)
    mutated_m = catalog("q4")
    assert {row["M"] for row in mutated_m if row["op"] == "mmv"} >= {1, 3, 17}


def test_same_math_and_nonfinite_and_staging_tags() -> None:
    identical = check_close(
        [1.0, -2.5, 0.0],
        [1.0, -2.5, 0.0],
        family="Q4_K",
        columns=256,
        class_name=SAME_MATH,
    )
    drifted = check_close(
        [1.0, -2.5, 0.25],
        [1.0, -2.5, 0.0],
        family="Q4_K",
        columns=256,
        class_name=SAME_MATH,
    )
    nan = check_close(
        [1.0, math.nan],
        [1.0, 1.0],
        family="Q8_0",
        columns=64,
        class_name=ASSOCIATION,
    )
    inf = check_close(
        [math.inf],
        [1.0],
        family="Q4_K",
        columns=256,
        class_name=ASSOCIATION,
    )
    assert identical["pass"] is True
    assert drifted["pass"] is False
    assert nan["pass"] is False and nan["reason"] == "nonfinite"
    assert inf["pass"] is False
    with pytest.raises(KernelParityError, match="sum_q"):
        check_close(
            [1.0],
            [1.0],
            family="Q8_0",
            columns=32,
            class_name=ASSOCIATION,
            candidate_field=STAGING_SUM_Q,
            reference_field=STAGING_SUM_X,
        )
    typed = q8_1_typed_reject()
    assert typed["pass"] is True


def test_wrappers_call_canonical_checker() -> None:
    sources = source_uses_canonical_checker()
    assert sources["pass"] is True
    assert sources["missing_include"] == []
    assert sources["duplicate_abs_scale"] == []
    cuda = CUDA.read_text(encoding="utf-8")
    for token in (
        "kernel_parity.cuh",
        "Q4DecodePathScope",
        "Q8DecodeLayoutScope",
        "Q6DecodePathScope",
        "MmqAsyncXOverrideScope",
        "integer_q8_late",
        "fma_async_x",
        "r2_w2",
        "same_math_equivalence",
        "launch_quantize_bf16_q8_1",
        "launch_q8_coop_mmv",
        "packed_down",
    ):
        assert token in cuda, token
    for path in WRAPPERS:
        text = path.read_text(encoding="utf-8")
        assert "kernel_parity.cuh" in text
        assert "ds4_q4k_association_ok" in text
        assert "kAbsScale" not in text


def test_host_phase_writes_fixture_without_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tools.opt082_kernel_parity as mod

    fixture = tmp_path / "opt082_kernel_parity.json"
    report = tmp_path / "REPORT.md"
    monkeypatch.setattr(mod, "FIXTURE", fixture)
    monkeypatch.setattr(mod, "REPORT", report)
    result = run_phase("q4", skip_gpu=True)
    assert result["success"] is True
    assert result["claims_throughput"] is False
    assert result["host_ok"] is True
    assert result["gpu_ran"] is False
    assert fixture.is_file()
    assert report.is_file()
    payload = load_json(fixture)
    assert payload["claims_throughput"] is False
    assert payload["catalog_count"] == len(catalog("q4"))
    assert all(row["matched_expectation"] for row in evaluate_host_cases())
    text = report.read_text(encoding="utf-8")
    assert "0.20" in text and "0.05" in text
    assert "claims_throughput: false" in text
    assert "no full production M" in text


def test_iteration_plans_and_native_observation_prefix() -> None:
    iteration = load_contract("OPT-082")
    q4 = describe_plan("OPT-082", "feedback", iteration, "q4")
    q8 = describe_plan("OPT-082", "feedback", iteration, "q8-q6")
    parity = describe_plan("OPT-082", "acceptance", iteration, "parity")
    assert "phase=q4" in q4
    assert f"product={len(catalog('q4'))}" in q4
    assert "phase=q8-q6" in q8
    assert "phase=parity" in parity
    assert "historical_oracles=none" in q4
    observed = parse_native_observation(
        'QW38_OPT082_KERNEL_PARITY_RESULT={"success":true,"case_count":3,'
        '"passed":3,"failed":0,"phase":"q4"}\n'
    )
    assert observed["success"] is True
    assert observed["case_count"] == 3


def test_dtype_and_payload_mutations_fail_closed() -> None:
    mismatch = check_close(
        [1.0, 2.0, 3.0],
        [1.0, 2.0],
        family="Q4_K",
        columns=256,
        class_name=ASSOCIATION,
    )
    assert mismatch["pass"] is False
    assert mismatch["reason"] == "shape_mismatch"
    q2 = check_close(
        [1.0],
        [1.0],
        family="Q2_K",
        columns=256,
        class_name=ASSOCIATION,
    )
    assert q2["applicable"] is False
    assert q2["reason"] == "not_applicable"
    json.dumps(catalog("parity"))
