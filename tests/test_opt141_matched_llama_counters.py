"""Host tests for OPT-141 matched llama NCU counters."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.opt139_counter_identity import (
    LAUNCH_PREFILL_ATTN,
    LAUNCH_WARP_QUERY,
    kernel_stem,
    production_prefill_attn_identity,
)
from tools.opt140_prefill_attention_replay import boundary_manifest
from tools.opt141_matched_llama_counters import (
    CONTRACT,
    FIXTURE,
    ITERATION,
    PHASES,
    REPORT,
    REQUIRED_FIXTURE_KEYS,
    SELECTIONS,
    UNRESOLVED,
    compare_quartz_llama,
    family_plan,
    fused_boundary_ok,
    llama_identity_match,
    ncu_profiled_nothing,
    ncu_regex_for,
    select_matched_kernel,
    selection_key,
    source_grounded_identity,
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
RUNNER = ROOT / "tools/run_optimization_task.py"
FATTN = ROOT / ".cache/authorities/llama.cpp/ggml/src/ggml-cuda/fattn.cu"
LLAMA_SRC = ROOT / "tools/llama_authority/opt136_decode_profile.cpp"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_makefile_and_registration() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-141")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    fattn = FATTN.read_text(encoding="utf-8")
    llama = LLAMA_SRC.read_text(encoding="utf-8")
    tool = (ROOT / "tools/opt141_matched_llama_counters.py").read_text(encoding="utf-8")
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert contract["task"] == "OPT-141"
    assert contract["claims_throughput"] is False
    assert contract["claims_performance_improvement"] is False
    assert contract["production_selector_change"] is False
    assert contract["require_candidate_nll"] is False
    assert contract["selected_execution_graph_path"] == "decode_segments8"
    assert contract["selections"] == list(SELECTIONS)
    assert contract["decode_attn_d128_kernel"] == "flash_attn_ext_vec"
    assert contract["decode_attn_d2048_kernel"] == "flash_attn_ext_vec"
    assert contract["prefill_attn_kernel"] == "flash_attn_ext_f16"
    assert contract["prefill_attn_replay_family"] == "prompt-attention"
    assert iteration["diagnostics_make_target"] == "cuda-opt141-diagnostics"
    assert iteration["performance_admission"]["instrumentation_only"] is True
    assert iteration["aggregate_deadline_s"] == 7200
    assert "cuda-opt141-diagnostics" in makefile
    assert "QW38_OPT141_MATCHED_LLAMA_COUNTERS_RESULT=" in runner
    assert "ggml_cuda_get_best_fattn_kernel" in fattn
    assert "prefill-unprofiled" in llama
    assert UNRESOLVED in tool
    assert "ncu_duration_is_not_a_benchmark" in tool
    assert PHASES == ("preflight", "identity", "counters", "report")
    validate_future_keep_policy("OPT-141", iteration)
    for key in REQUIRED_FIXTURE_KEYS:
        assert key in contract["required_fixture_keys"]


def test_iteration_loop_products() -> None:
    iteration = load_contract("OPT-141")
    preflight = iteration["workloads"]["preflight"]
    identity = iteration["workloads"]["identity"]
    counters = iteration["workloads"]["counters"]
    report = iteration["workloads"]["report"]
    assert loop_product(workload_for_mode(preflight, "feedback")) == 1
    assert loop_product(workload_for_mode(identity, "acceptance")) == 6
    assert loop_product(workload_for_mode(counters, "acceptance")) == 5
    assert loop_product(workload_for_mode(report, "acceptance")) == 1
    described = describe_plan("OPT-141", "acceptance", iteration, "counters")
    assert "phase=counters" in described
    assert "loop_product=1" in family_plan("acceptance", "preflight")


def test_source_maps_d128_d2048_and_prefill_separately() -> None:
    d128 = source_grounded_identity("decode", "attn_core", 128)
    d2048 = source_grounded_identity("decode", "attn_core", 2048)
    prefill = source_grounded_identity("prefill", "attn_core", 4096)
    residual = source_grounded_identity("decode", "residual_norm_quant", 128)
    mmq = source_grounded_identity("prefill", "prompt_mmq", 4096)
    assert d128["expected_kernel"] == "flash_attn_ext_vec"
    assert d2048["expected_kernel"] == "flash_attn_ext_vec"
    assert d128["shape"]["n_kv"] == 128
    assert d2048["shape"]["n_kv"] == 2048
    assert prefill["expected_kernel"] == "flash_attn_ext_f16"
    assert prefill["shape"]["ncols1"] == 8
    assert prefill["shape"]["ncols2"] == 8
    assert residual["expected_kernel"] == "quantize_q8_1"
    assert mmq["expected_kernel"] == "mul_mat_q"
    assert ncu_regex_for("quantize_q8_1") == r"^quantize_q8_1$"
    assert ncu_regex_for("mul_mat_q") == r"^mul_mat_q$"
    assert ncu_profiled_nothing("==WARNING== No kernels were profiled.")
    assert not ncu_profiled_nothing('==PROF== Profiling "mul_mat_q" - 0')
    assert selection_key("decode", "attn_core") == "decode.attn_core"


def test_missing_identity_is_blocked_not_skipped() -> None:
    matched = select_matched_kernel(
        {"stems": [], "largest_leaf": None},
        phase="decode",
        family="attn_core",
        expected="flash_attn_ext_vec",
    )
    identity = llama_identity_match(
        phase="decode",
        family="attn_core",
        workload="D128",
        kernel=None,
        expected_kernel="flash_attn_ext_vec",
        replay_family="decode-attention",
    )
    assert matched["ok"] is True or matched["matched_kernel"] == "flash_attn_ext_vec"
    assert identity["ok"] is False
    assert UNRESOLVED in identity["mismatches"]
    assert identity["reason"] != "skipped"


def test_ambiguous_attn_stems_are_rejected() -> None:
    summary = {
        "stems": [
            {"stem": "flash_attn_ext_vec", "count": 10, "total_ns": 1000},
            {"stem": "flash_attn_ext_f16", "count": 10, "total_ns": 1100},
        ],
        "largest_leaf": {"stem": "flash_attn_ext_f16", "duration_ns": 200},
    }
    matched = select_matched_kernel(
        summary,
        phase="decode",
        family="attn_core",
        expected="flash_attn_ext_vec",
    )
    assert matched["ok"] is False
    assert matched["ambiguous"] is True
    assert "ambiguous_attn_compute_stems" in matched["mismatches"]


def test_combine_results_is_fused_boundary_not_matched_leaf() -> None:
    summary = {
        "stems": [
            {"stem": "flash_attn_ext_vec", "count": 192, "total_ns": 1_436_000},
            {"stem": "flash_attn_combine_results", "count": 96, "total_ns": 738_000},
        ],
        "largest_leaf": {
            "stem": "flash_attn_combine_results",
            "duration_ns": 9664,
        },
    }
    matched = select_matched_kernel(
        summary,
        phase="decode",
        family="attn_core",
        expected="flash_attn_ext_vec",
    )
    assert matched["matched_kernel"] == "flash_attn_ext_vec"
    assert matched["largest_leaf"] == "flash_attn_combine_results"
    rejected = llama_identity_match(
        phase="decode",
        family="attn_core",
        workload="D128",
        kernel="flash_attn_combine_results",
        expected_kernel="flash_attn_ext_vec",
        replay_family="decode-attention",
    )
    assert rejected["ok"] is False
    assert "fused_boundary_mismatch" in rejected["mismatches"]


def test_vec_is_not_p4096_prefill_evidence() -> None:
    quartz = production_prefill_attn_identity()
    assert quartz["expected_kernel"] == LAUNCH_PREFILL_ATTN
    assert quartz["replay_family"] == "prompt-attention"
    rejected = llama_identity_match(
        phase="prefill",
        family="attn_core",
        workload="P4096",
        kernel="flash_attn_ext_vec",
        expected_kernel="flash_attn_ext_f16",
        replay_family="prompt-attention",
    )
    assert rejected["ok"] is False
    assert "vec_is_not_p4096_prefill_evidence" in rejected["mismatches"]
    decode_replay = llama_identity_match(
        phase="prefill",
        family="attn_core",
        workload="P4096",
        kernel="flash_attn_ext_f16",
        expected_kernel="flash_attn_ext_f16",
        replay_family="decode-attention",
    )
    assert decode_replay["ok"] is False
    manifest = boundary_manifest(
        replay_family="prompt-attention",
        dispatch={"path": "opt111_base", "launch": LAUNCH_PREFILL_ATTN},
        identity={"ok": True, "mismatches": []},
    )
    assert manifest["decode_attention_substitution"] is False


def test_fused_boundary_mismatch_rejects_leaf_comparison() -> None:
    fusion = fused_boundary_ok(
        family="attn_core",
        quartz_kernel=LAUNCH_PREFILL_ATTN,
        llama_kernel="flash_attn_ext_f16",
        llama_enclosing=["flash_attn_stream_k_fixup_general"],
    )
    assert fusion["ok"] is False
    assert fusion["reason"] == "fused_boundary_mismatch"
    residual = fused_boundary_ok(
        family="residual_norm_quant",
        quartz_kernel="rms_norm_fp32_to_bf16_parallel",
        llama_kernel="quantize_q8_1",
        llama_enclosing=["rms_norm_f32", "quantize_q8_1"],
    )
    assert residual["ok"] is False
    compared = compare_quartz_llama(
        {
            "dram_read_bytes": 10.0,
            "units": {"dram_read_bytes": "byte"},
            "error": None,
        },
        {
            "dram_read_bytes": 12.0,
            "units": {"dram_read_bytes": "byte"},
            "error": None,
        },
        fusion=fusion,
    )
    assert compared["comparable"] is False
    assert compared["ncu_duration_benchmark"] is False
    assert compared["claims_throughput"] is False


def test_compare_requires_matching_units_and_ignores_ncu_duration() -> None:
    fusion = {"ok": True, "reason": None, "enclosing": [], "incomparable_counters": []}
    compared = compare_quartz_llama(
        {
            "dram_read_bytes": 100.0,
            "dram_throughput": 2.0,
            "units": {"dram_read_bytes": "byte", "dram_throughput": "%"},
            "gpu__time_duration": 0.4,
        },
        {
            "dram_read_bytes": 80.0,
            "dram_throughput": 3.0,
            "units": {"dram_read_bytes": "Kbyte", "dram_throughput": "%"},
            "gpu__time_duration": 0.2,
        },
        fusion=fusion,
    )
    slots = {item["slot"] for item in compared["compared_slots"]}
    skipped = {item["slot"]: item["reason"] for item in compared["skipped_slots"]}
    assert "dram_read_bytes" not in slots
    assert skipped["dram_read_bytes"] == "unit_mismatch"
    assert "dram_throughput" in slots
    assert skipped["gpu__time_duration"] == "ncu_duration_is_not_a_benchmark"
    mixed = compare_quartz_llama(
        {"dram_read_bytes": 1.0, "leaf_isolated": False, "units": {}},
        {"dram_read_bytes": 1.0, "units": {"dram_read_bytes": "byte"}},
        fusion=fusion,
    )
    assert mixed["comparable"] is False
    assert mixed["reason"] == "quartz_counters_unusable"


def test_kernel_stems_and_fixture_schema() -> None:
    mangled = (
        "void flash_attn_ext_f16<(int)256, (int)256, (int)8, (int)8, "
        "(bool)0, (bool)0>(const char *)"
    )
    assert kernel_stem(mangled) == "flash_attn_ext_f16"
    assert kernel_stem("quantize_mmq_q8_1") == "quantize_mmq_q8_1"
    assert kernel_stem("quantize_q8_1(const float *)") == "quantize_q8_1"
    assert (
        kernel_stem("void mul_mat_q<(ggml_type)12, (int)128, (bool)0>") == "mul_mat_q"
    )
    payload = {
        "schema_version": 1,
        "task": "OPT-141",
        "status": "structure",
        "mode": "acceptance",
        "claims_throughput": False,
        "claims_performance_improvement": False,
        "production_kept": True,
        "selected_execution_graph_path": "decode_segments8",
        "prefixes": [128, 2048],
        "preflight": {},
        "identity": {},
        "counters": {},
        "comparisons": {},
        "answers": {},
        "report_path": str(REPORT.relative_to(ROOT))
        if REPORT.is_relative_to(ROOT)
        else "evidence/optimization/opt141-matched-llama-counters/REPORT.md",
    }
    validate_fixture(payload)
    assert LAUNCH_WARP_QUERY != "flash_attn_ext_vec"
    if FIXTURE.is_file():
        fixture = _json(FIXTURE)
        assert fixture["claims_throughput"] is False
