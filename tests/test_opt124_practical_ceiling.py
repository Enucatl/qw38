"""Host tests for OPT-124 practical ceiling analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.opt115_pipeline_traffic import bytes_to_peak_ms, compulsory_decode_bytes
from tools.opt124_practical_ceiling import (
    COMBINED_DECODE_D2H,
    CONCLUSION_LABELS,
    CONTRACT,
    FIXTURE,
    ITERATION,
    MATERIALITY,
    NATIVE,
    PEAK_FP32_TFLOPS,
    PHASES,
    REPORT,
    REQUIRED_FIXTURE_KEYS,
    WORKLOADS,
    authenticate_pinned_llama,
    combined_decode_bytes,
    compute_bound,
    identities_match,
    lower_precision_alternatives,
    matched_comparison,
    mechanism_disposition,
    overlap_aware_bound,
    recompute_bounds,
    stop_reopen_criteria,
    supported_conclusions,
    tok_s_to_ms,
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
NATIVE_SRC = ROOT / "cuda/opt124_practical_ceiling_test.cu"
RUNNER = ROOT / "tools/run_optimization_task.py"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_makefile_and_native_hooks() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-124")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    native = NATIVE_SRC.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert NATIVE_SRC.is_file()
    assert contract["task"] == "OPT-124"
    assert contract["claims_throughput"] is False
    assert contract["claims_performance_improvement"] is False
    assert contract["production_selector_change"] is False
    assert contract["pinned_llama_authority_unchanged"] is True
    assert contract["llama_revision"] == "cc83d7b4824f73cfdda4dfbb47ee39804f71b328"
    assert contract["final_stack"] == "combined_opt118_opt119"
    assert contract["materiality_fraction"] == MATERIALITY
    assert "universal_optimality" in contract["forbidden_claims"]
    assert iteration["diagnostics_make_target"] == "cuda-opt124-diagnostics"
    assert iteration["performance_admission"]["instrumentation_only"] is True
    assert iteration["modes"]["release"]["historical_oracles"] is False
    assert "cuda-opt124-diagnostics" in makefile
    assert "qw38-cuda-opt124-practical-ceiling-test" in makefile
    assert Path(NATIVE).name in makefile
    assert "OPT110_LLAMA_OBJECT" in makefile
    assert "--workload identity|combined-timeline" in native
    assert "independently_restored=true" in native
    assert "combined_opt118_opt119" in native
    assert "QW38_OPT124_PRACTICAL_CEILING_RESULT=" in native
    assert "QW38_OPT124_NATIVE_COUNTS=" in native
    assert "QW38_OPT124_PRACTICAL_CEILING_RESULT=" in runner
    assert "QW38_OPT124_NATIVE_COUNTS=" in runner
    assert PHASES == (
        "freeze",
        "bounds",
        "llama-compare",
        "mechanisms",
        "identity-timeline",
        "conclusions",
        "report",
    )
    validate_future_keep_policy("OPT-124", iteration)


def test_iteration_loop_products_and_plan() -> None:
    iteration = load_contract("OPT-124")
    freeze = iteration["workloads"]["freeze"]
    bounds = iteration["workloads"]["bounds"]
    llama = iteration["workloads"]["llama-compare"]
    mechanisms = iteration["workloads"]["mechanisms"]
    timeline = iteration["workloads"]["identity-timeline"]
    conclusions = iteration["workloads"]["conclusions"]
    report = iteration["workloads"]["report"]
    assert loop_product(workload_for_mode(freeze, "feedback")) == 1
    assert loop_product(workload_for_mode(bounds, "feedback")) == 14
    assert loop_product(workload_for_mode(llama, "feedback")) == 3
    assert loop_product(workload_for_mode(mechanisms, "feedback")) == 6
    assert loop_product(workload_for_mode(timeline, "feedback")) == 4
    assert loop_product(workload_for_mode(conclusions, "acceptance")) == 7
    assert loop_product(workload_for_mode(report, "acceptance")) == 1
    described = describe_plan("OPT-124", "feedback", iteration, "freeze")
    assert "phase=freeze" in described
    proof = " ".join(iteration["proof_limit"]).casefold()
    assert "universal" in proof
    assert "1792" in " ".join(iteration["proof_limit"]) or "peak" in proof


def test_combined_bytes_use_lazy_index_not_full_logits() -> None:
    original = compulsory_decode_bytes(prefix=128)
    combined = combined_decode_bytes(prefix=128)
    assert combined["units"] == "bytes"
    assert combined["representation"] == "combined_opt118_opt119"
    assert combined["logits_d2h_bytes"] == COMBINED_DECODE_D2H
    assert combined["phases"]["transfers"] == COMBINED_DECODE_D2H
    assert original["phases"]["transfers"] == 1_013_760
    assert combined["phases"]["weights"] == original["phases"]["weights"]
    assert combined["phases"]["kv"] == original["phases"]["kv"]
    d2048 = combined_decode_bytes(prefix=2048)
    assert d2048["kv_read_bytes"] > combined["kv_read_bytes"]


def test_bounds_use_critical_path_not_independent_sum() -> None:
    overlap = overlap_aware_bound(
        byte_critical_ms=10.18, compute_ms=0.26, serial_idle_ms=9.97
    )
    assert overlap["independent_sum_not_a_bound"] is True
    assert overlap["busy_overlap_ms"] == 10.18
    assert overlap["overlap_aware_ms"] == 10.18 + 9.97
    compute = compute_bound(tokens=1, prefix=128)
    assert compute["excludes_unpack_scale"] is True
    assert compute["peak_fp32_tflops"] == PEAK_FP32_TFLOPS
    assert compute["optimistic_compute_ms"] < 1.0
    prefill = compute_bound(tokens=4096, prefix=None)
    assert prefill["fp32_peak_ms"] > 1000.0
    alts = lower_precision_alternatives(prefix=128)
    assert alts["admitted"] is False
    assert alts["opt121_q8_to_q4k"]["quality_matched"] is False
    bounds = recompute_bounds()
    d128 = bounds["workloads"]["d128"]
    assert d128["byte_bound"]["independent_sum_not_a_bound"] is True
    assert d128["optimistic_not_attainable"] is True
    assert d128["traffic"]["logits_d2h_bytes"] == COMBINED_DECODE_D2H
    p4096 = bounds["workloads"]["p4096"]
    assert p4096["compute_bound"]["total_flops"] > d128["compute_bound"]["total_flops"]


def test_opt123_identities_and_llama_baseline() -> None:
    match = identities_match()
    assert match["ok"] is True
    assert match["reuse_opt123_samples"] is True
    assert match["expected"]["candidate"] == "combined_opt118_opt119"
    assert match["expected"]["llama_revision"] == (
        "cc83d7b4824f73cfdda4dfbb47ee39804f71b328"
    )
    llama = authenticate_pinned_llama()
    assert llama["ok"] is True
    assert llama["authority_unchanged"] is True
    assert llama["n_gpu_layers"] == 99
    assert llama["flash_attn_auto"] is True
    assert llama["kv_policy"]["matches_quartz_dense_bf16"] is True
    assert tok_s_to_ms(50.0) == 20.0


def test_mechanism_disposition_covers_117_122_and_does_not_reopen() -> None:
    table = mechanism_disposition()
    tasks = [row["task"] for row in table["ranked"]]
    assert tasks == [
        "OPT-117",
        "OPT-118",
        "OPT-119",
        "OPT-120",
        "OPT-121",
        "OPT-122",
    ]
    by_task = {row["task"]: row for row in table["ranked"]}
    assert by_task["OPT-117"]["complete_request_outcome"] == "retain_ffn_only"
    assert by_task["OPT-118"]["complete_request_outcome"] == "keep"
    assert by_task["OPT-119"]["complete_request_outcome"] == "keep"
    assert by_task["OPT-120"]["complete_request_outcome"] == "quality_blocked"
    assert by_task["OPT-121"]["complete_request_outcome"] == "quality_blocked"
    assert by_task["OPT-122"]["complete_request_outcome"] == "no_material_opportunity"
    assert all(row["reopen"] is False for row in table["ranked"])
    assert table["rejected_paths_do_not_leak"] is True


def test_conclusions_are_labeled_and_forbid_universal_optimality() -> None:
    bounds = recompute_bounds()
    conclusions = supported_conclusions(
        bounds=bounds,
        comparison=matched_comparison(),
        mechanisms=mechanism_disposition(),
    )
    per = conclusions["per_workload"]
    for name in WORKLOADS:
        assert per[name]["label"] in CONCLUSION_LABELS
        assert per[name]["universal_optimality"] is False
    assert per["p4096"]["label"] == "measurable_headroom"
    assert per["d128"]["label"] == "measurable_headroom"
    assert per["d2048"]["label"] == "measurable_headroom"
    assert per["d131040"]["label"] == "insufficient_evidence"
    assert per["opt016_2k"]["label"] == "insufficient_evidence"
    assert conclusions["universal_optimality_claimed"] is False
    assert conclusions["supports_claiming_llama_is_practical_ceiling"] is False
    assert conclusions["supports_continuing"] is False
    criteria = stop_reopen_criteria()
    assert "llama.cpp is globally optimal" in criteria["do_not_stop_claiming"]
    assert any("recapture" in item for item in criteria["reopen_when"])


def test_validate_scaffold_fixture_when_present() -> None:
    if not FIXTURE.is_file():
        return
    payload = _json(FIXTURE)
    result = validate_fixture(payload)
    assert result["ok"] is True
    assert payload["claims_throughput"] is False
    assert payload["universal_optimality_claimed"] is False
    assert REPORT.is_file()
    report = REPORT.read_text(encoding="utf-8")
    assert "universal_optimality_claimed" in report
    assert "Supported conclusions per workload" in report
    for key in REQUIRED_FIXTURE_KEYS:
        assert key in payload


def test_bytes_to_peak_ms_matches_listed_1792() -> None:
    assert abs(bytes_to_peak_ms(1_792_000_000) - 1.0) < 1e-9
