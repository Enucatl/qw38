"""Host tests for OPT-120 packed KV keep/reject."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools.opt120_packed_kv import (
    CONTRACT,
    DECODE_TOKENS,
    FIXTURE,
    ITERATION,
    LONG_CAPACITY,
    LONG_DECODE,
    LONG_PAIRS,
    NATIVE,
    PAIR_COUNT,
    PARENT,
    REPORT,
    REQUIRED_FIXTURE_KEYS,
    WARMUPS,
    family_plan,
    pair_order,
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
NATIVE_SRC = ROOT / "cuda/opt120_packed_kv_test.cu"
HEADER = ROOT / "cuda/opt120_packed_kv.cuh"
SCHEDULER = ROOT / "cuda/full_scheduler.cu"
CHECKPOINT = ROOT / "cuda/checkpoint.cu"
ATTENTION = ROOT / "cuda/attention_decode.cu"
RUNNER = ROOT / "tools/run_optimization_task.py"
TOOL = ROOT / "tools/opt120_packed_kv.py"


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_contracts_makefile_and_native_hooks() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-120")
    makefile = MAKEFILE.read_text(encoding="utf-8")
    native = NATIVE_SRC.read_text(encoding="utf-8")
    header = HEADER.read_text(encoding="utf-8")
    scheduler = SCHEDULER.read_text(encoding="utf-8")
    checkpoint = CHECKPOINT.read_text(encoding="utf-8")
    attention = ATTENTION.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    tool = TOOL.read_text(encoding="utf-8")
    assert CONTRACT.is_file()
    assert ITERATION.is_file()
    assert NATIVE_SRC.is_file()
    assert contract["task"] == "OPT-120"
    assert contract["parent"] == PARENT
    assert contract["parent_execution_graphs"] == "ffn_only"
    assert contract["primary_target"] == "long_context_decode"
    assert contract["max_changes"] == 1
    assert contract["integer_8bit"] is True
    assert contract["aa_warmups"] == WARMUPS
    assert contract["aa_pairs"] == PAIR_COUNT
    assert contract["long_context_pairs"] == LONG_PAIRS
    assert contract["long_context_decode_tokens"] == LONG_DECODE
    assert contract["long_context_capacity"] == LONG_CAPACITY
    assert contract["decode_output_tokens"] == DECODE_TOKENS
    assert contract["throughput_ratio_lower_bound"] == 1.0
    assert contract["non_target_point_ratio_min"] == 0.98
    assert contract["decode_p95_ratio_max"] == 1.05
    assert contract["require_candidate_nll"] is True
    assert iteration["diagnostics_make_target"] == "cuda-opt120-diagnostics"
    assert iteration["aggregate_deadline_s"] == 7200
    assert "cuda-opt120-diagnostics" in makefile
    assert "qw38-cuda-opt120-packed-kv-test" in makefile
    assert Path(NATIVE).name in makefile
    assert "OPT110_LLAMA_OBJECT" in makefile
    assert "kSelectedPackedKvFormat" in header
    assert "PackedKvFormat::kDenseBf16" in header
    assert "packed_kv_committed_key" in scheduler
    assert "legacy dense checkpoint cannot be interpreted as packed KV" in checkpoint
    assert "g_opt120_packed_kv_format" in attention
    assert "--workload reference|inventory|greedy-parity|api|" in native
    assert "independently_restored=true" in native
    assert "QW38_OPT120_PACKED_KV_RESULT=" in native
    assert "QW38_OPT120_NATIVE_COUNTS=" in runner
    assert "QW38_OPT120_PACKED_KV_RESULT=" in runner
    assert "require_candidate_nll" in tool
    assert "same_math_copied_from_control" in tool
    validate_future_keep_policy("OPT-120", iteration)


def test_iteration_loop_products() -> None:
    iteration = load_contract("OPT-120")
    inventory = iteration["workloads"]["inventory"]
    greedy = iteration["workloads"]["greedy-parity"]
    api = iteration["workloads"]["api"]
    quality = iteration["workloads"]["quality"]
    d128 = iteration["workloads"]["d128"]
    d8192 = iteration["workloads"]["d8192"]
    d131040 = iteration["workloads"]["d131040"]
    assert loop_product(workload_for_mode(inventory, "feedback")) == 3
    assert loop_product(workload_for_mode(greedy, "feedback")) == 2
    assert loop_product(workload_for_mode(api, "feedback")) == 2
    assert loop_product(workload_for_mode(quality, "acceptance")) == 12288
    assert loop_product(workload_for_mode(d128, "acceptance")) == 26
    assert loop_product(workload_for_mode(d8192, "acceptance")) == 16
    assert loop_product(workload_for_mode(d131040, "acceptance")) == 6
    described = describe_plan("OPT-120", "feedback", iteration, "inventory")
    assert "phase=inventory" in described
    assert "loop_product=3" in family_plan("feedback", "inventory")
    assert pair_order(0) == "AB"
    assert pair_order(1) == "BA"
    proof = " ".join(iteration["proof_limit"]).casefold()
    assert "3 warmups plus 5 interleaved ab/ba" in proof
    assert "7200 is a ceiling" in proof
    assert "primary target is long-context decode" in proof
    assert "memory-only" in proof


def test_keep_requires_opt115_and_opt116_gates() -> None:
    contract = _json(CONTRACT)
    iteration = load_contract("OPT-120")
    assert contract["target_workloads"] == ["d8192", "d32768", "d131040"]
    assert contract["non_target_workloads"] == ["d128", "d2048", "p4096"]
    assert iteration["workloads"]["quality"]["aggregate_deadline_s"] == 7200
    assert "opt116_generated_v1" in contract["opt116_quality_contract"]
    assert iteration["workloads"]["quality"]["target"].endswith(
        "opt058-quality-baseline-test"
    )
    assert iteration["workloads"]["long-cache"]["target"].endswith(
        "opt116-generated-quality-test"
    )
    for key in REQUIRED_FIXTURE_KEYS:
        assert key in contract["required_fixture_keys"]
    assert REPORT.parent.as_posix().endswith("opt120-packed-kv")
    assert FIXTURE.name == "opt120_packed_kv.json"
    assert PARENT == "post113_selected"


def test_pins_match_keep_state() -> None:
    kept = "dense_bf16"
    if FIXTURE.is_file():
        payload = _json(FIXTURE)
        if payload.get("production_kept"):
            kept = str(payload.get("selected_packed_kv_format") or "dense_bf16")
    header = HEADER.read_text(encoding="utf-8")
    enum = {
        "dense_bf16": "PackedKvFormat::kDenseBf16",
        "q8q8": "PackedKvFormat::kQ8Q8",
        "q8q4": "PackedKvFormat::kQ8Q4",
        "q4q4": "PackedKvFormat::kQ4Q4",
    }[kept]
    assert f"kSelectedPackedKvFormat = {enum}" in header
    contract = _json(CONTRACT)
    assert contract["selected_packed_kv_format"] == kept


def test_validate_scaffold_fixture_when_present() -> None:
    if not FIXTURE.is_file():
        return
    payload = _json(FIXTURE)
    for key in REQUIRED_FIXTURE_KEYS:
        assert key in payload
    assert payload["task"] == "OPT-120"
    assert payload["parent"] == "post113_selected"
    assert payload["primary_target"] == "long_context_decode"
    if payload.get("production_kept"):
        assert payload["verdict"] == "keep"
        assert payload["selected_packed_kv_format"] != "dense_bf16"
    else:
        assert payload["verdict"] in {
            "reject",
            "zero_redundancy",
            "quality_blocked",
            "inconclusive",
        }
        assert payload["selected_packed_kv_format"] == "dense_bf16"
    quality = payload.get("quality") or {}
    if payload.get("verdict") in {"keep", "reject", "quality_blocked"}:
        assert quality.get("candidate_nll_measured") is True
        assert quality.get("opt058_invoked") is True
