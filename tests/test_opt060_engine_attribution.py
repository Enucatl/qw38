"""Host tests for OPT-060 matched full-engine family attribution."""

from __future__ import annotations

import json
from pathlib import Path

from tools.opt060_engine_attribution import (
    GGUF_SHA,
    LLAMA_REV,
    OVERLAY,
    PATCH,
    chargeable_records,
    compare_engines,
    family_table,
    missing_roles,
    overhead_ms,
    patch_identity,
    stream_aware_totals,
)
from tools.run_optimization_task import load_contract, loop_product

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "pins/opt060_engine_attribution_contract.json"
ITERATION = ROOT / "pins/opt060_iteration_contract.json"
FIXTURE = ROOT / "fixtures/opt060_engine_attribution.json"
REPORT = ROOT / "evidence/optimization/opt060-engine-attribution/REPORT.md"
NATIVE = ROOT / "cuda/opt060_engine_attribution_test.cu"
HEADER = ROOT / "cuda/engine_attribution.h"
SCHEDULER = ROOT / "cuda/full_scheduler.cu"
HELPER = ROOT / "tools/llama_authority/build_opt060_instrumented.sh"
ADAPTER = ROOT / "tools/llama_authority/engine_attribution.cpp"
MAKEFILE = ROOT / "Makefile"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_and_patch_identity() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-060")
    fixture = _json(FIXTURE)
    identity = patch_identity()
    assert PATCH.is_file()
    assert OVERLAY.is_file()
    assert HELPER.is_file()
    assert ADAPTER.is_file()
    assert HEADER.is_file()
    assert NATIVE.is_file()
    assert REPORT.is_file()
    assert contract["task"] == "OPT-060"
    assert contract["claims_performance_improvement"] is False
    assert contract["isolated_op_timings_insufficient"] is True
    assert contract["llama_revision"] == LLAMA_REV
    assert contract["gguf_sha256"] == GGUF_SHA
    assert iteration["task"] == "OPT-060"
    assert iteration["claims_throughput"] is False
    assert iteration["target"] == "build/qw38-cuda-opt060-engine-attribution-test"
    assert identity["revision"] == LLAMA_REV
    assert len(identity["patch_sha256"]) == 64
    assert len(identity["overlay_sha256"]) == 64
    text = PATCH.read_text(encoding="utf-8")
    scheduler = SCHEDULER.read_text(encoding="utf-8")
    assert "GpuPhaseRecorder" in scheduler
    assert "bind_families" in scheduler
    assert "ggml_cuda_try_fuse" in text
    assert "qw38_opt060::end_op" in text
    assert "qw38_opt060::begin_op" in text
    assert "mmvq" in text
    assert "mmq" in text
    assert "cc83d7b4824f73cfdda4dfbb47ee39804f71b328" in HELPER.read_text(
        encoding="utf-8"
    )
    assert "llama.cpp-opt060" in HELPER.read_text(encoding="utf-8")
    assert "production_source_untouched" in HELPER.read_text(encoding="utf-8")
    assert fixture["isolated_opt043_component_gap_not_this_sitting"] is True
    assert fixture["not_llama_bench"] is True
    for field in contract["required_record_fields"]:
        assert field in fixture["synthetic_overlap"]["records"][0]


def test_stream_aware_sum_exceeds_wall() -> None:
    records = _json(FIXTURE)["synthetic_overlap"]["records"]
    charged = chargeable_records(records)
    totals = stream_aware_totals(charged)
    assert totals["sum_exceeds_wall"] is True
    assert totals["summed_gpu_work_ms"] == 6.0
    assert totals["interval_union_ms"] == 4.0
    assert totals["overlap_ms"] == 2.0


def test_fused_member_not_double_counted() -> None:
    records = _json(FIXTURE)["synthetic_overlap"]["records"]
    table = family_table(records)
    assert table["ffn_gate_up_glu"]["ms"] == 4.0
    assert table["ffn_gate_up_glu"]["fused_member_count"] == 3
    assert "ffn_gate" not in table
    assert table["copy"]["ms"] == 2.0


def test_missing_and_unmatched_roles() -> None:
    quartz = _json(FIXTURE)["synthetic_overlap"]["records"]
    llama = _json(FIXTURE)["llama_synthetic"]
    missing = missing_roles(quartz)
    assert "embedding" in missing
    assert "logits_projection" in missing
    comparison = compare_engines(quartz, llama)
    assert "logits_projection" in comparison["unmatched"]["llama_only"]
    assert comparison["families"]
    ffn = next(
        row for row in comparison["families"] if row["role"] == "ffn_gate_up_glu"
    )
    assert ffn["quartz_minus_llama_ms"] == 1.5


def test_event_pool_overflow_and_overhead() -> None:
    fixture = _json(FIXTURE)
    assert fixture["event_pool_overflow"]["detected"] is True
    assert overhead_ms(10.0, 11.5) == 1.5
    native = NATIVE.read_text(encoding="utf-8")
    assert "EngineEventPool<1>" in native
    assert "pool_overflow" in native
    assert "exclusive_gpu" in native
    assert "release_gpu" in native
    assert "cudaDeviceReset" in native
    scheduler = SCHEDULER.read_text(encoding="utf-8")
    assert "bind_families" in scheduler
    assert "record_epoch" in scheduler
    assert "EngineAttribution* families = attribution->families;" in scheduler
    header = HEADER.read_text(encoding="utf-8")
    assert "kEngineEventPoolDefault" in header


def test_iteration_contract_phases_and_no_oracles() -> None:
    iteration = load_contract("OPT-060")
    assert iteration["modes"]["feedback"]["aggregate_deadline_s"] == 300
    assert iteration["modes"]["acceptance"]["aggregate_deadline_s"] == 7200
    assert iteration["aggregate_deadline_s"] == 300
    assert "decode" in iteration["modes"]["feedback"]["tier_sequence"]
    assert "prefill" in iteration["modes"]["feedback"]["tier_sequence"]
    assert iteration["workloads"]["decode"]["engines"] == 2
    assert iteration["workloads"]["decode"]["output_tokens"] == 16
    assert iteration["workloads"]["prefill"]["prompt"] == 4096
    assert loop_product(iteration["workloads"]["decode"]) == 64
    assert "historical_oracles" not in iteration["modes"]["feedback"] or not iteration[
        "modes"
    ]["feedback"].get("historical_oracles")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    assert "qw38-cuda-opt060-engine-attribution-test" in makefile
    assert "cuda-opt060-diagnostics" in makefile
    cmake = (ROOT / "tools/llama_authority/CMakeLists.txt").read_text(encoding="utf-8")
    assert "qw38-llama-engine-attribution" in cmake
    adapter = ADAPTER.read_text(encoding="utf-8")
    assert "prefix_batch" in adapter
    assert "not_llama_bench" in adapter
    assert iteration["setup_host_commands"][0][1].endswith(
        "build_opt060_instrumented.sh"
    )


def test_report_has_no_speedup_claim() -> None:
    report = REPORT.read_text(encoding="utf-8")
    assert "claims no performance improvement" in report.lower() or (
        "no speedup" in report.lower()
    )
    assert "eager_diagnostic" in report
    assert "overlap" in report.lower()
    assert "OPT-043" in report
    assert "llama-bench" in report.lower()
